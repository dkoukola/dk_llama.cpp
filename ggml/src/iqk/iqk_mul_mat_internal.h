#pragma once

#include <stdint.h>

#include "ggml.h"
#include "iqk_mul_mat.h"

enum {
    IQK_DSV4_ATTN_WO_A_NX = 1024,
    IQK_DSV4_ATTN_WO_A_NY = 8,
    IQK_DSV4_ATTN_WO_A_K = 4096,
    IQK_DSV4_ATTN_WO_A_GROUPS = 8,
    IQK_DSV4_ATTN_WO_A_WORKERS = 64,
    IQK_DSV4_ATTN_WO_A_GROUP_WORKERS = 8,
    IQK_NUMA_MIRROR_SMALL_TILE = 16,
    IQK_NUMA_MIRROR_SMALL_NY = 8,
    IQK_NUMA_MIRROR_SMALL_K = 4096,
};

static inline bool iqk_dsv4_attn_wo_a_shape_matches(
        long Nx, long Ny, long ne00,
        long ne02, long ne03, long ne12, long ne13,
        int typeA, int typeB, int nth) {
    return nth > IQK_DSV4_ATTN_WO_A_WORKERS &&
        typeA == GGML_TYPE_BF16 && typeB == GGML_TYPE_F32 &&
        Nx == IQK_DSV4_ATTN_WO_A_NX &&
        Ny == IQK_DSV4_ATTN_WO_A_NY &&
        ne00 == IQK_DSV4_ATTN_WO_A_K &&
        ne02 == IQK_DSV4_ATTN_WO_A_GROUPS && ne03 == 1 &&
        ne12 == IQK_DSV4_ATTN_WO_A_GROUPS && ne13 == 1;
}

static inline bool iqk_dsv4_attn_wo_a_numa_matches(
        bool mirror_active, int n_nodes, uint32_t mirror_flags) {
    return mirror_active && n_nodes == 2 &&
        (mirror_flags & GGML_NUMA_MIRROR_WEIGHTS) != 0;
}

static inline bool iqk_dsv4_attn_wo_a_use_partition(
        bool mirror_active, int n_nodes, uint32_t mirror_flags,
        long Nx, long Ny, long ne00,
        long ne02, long ne03, long ne12, long ne13,
        int typeA, int typeB, int nth) {
    return iqk_dsv4_attn_wo_a_shape_matches(
                Nx, Ny, ne00, ne02, ne03, ne12, ne13,
                typeA, typeB, nth) &&
        iqk_dsv4_attn_wo_a_numa_matches(
                mirror_active, n_nodes, mirror_flags);
}

// Return the logical rank of a worker selected from a larger physical team,
// or -1 for an inactive worker. Logical ranks are sampled at the centers of
// 64 equal intervals, which keeps 32 workers in each half of a two-node NUMA
// mirror team.
static inline int iqk_dsv4_attn_wo_a_active_rank(int ith, int nth) {
    const int n_active = IQK_DSV4_ATTN_WO_A_WORKERS;

    if (ith < 0 || ith >= nth || nth < n_active) {
        return -1;
    }
    for (int rank = 0; rank < n_active; ++rank) {
        const int owner = (int) (((int64_t) (2*rank + 1)*nth)/(2*n_active));
        if (owner == ith) {
            return rank;
        }
    }
    return -1;
}

// Execute the exact 64-worker DeepSeek-V4 attention partition. Keeping the
// pointer arithmetic here lets the production path and its direct test share
// the same group selection and tensor strides.
static inline bool iqk_dsv4_attn_wo_a_mul(
        const void * A, long strideA, long nb02,
        const void * B, long strideB, long nb12,
        float * C, long stride_C, long nb2,
        int ith, int nth, bool kernel_available) {
    // Every physical worker must report the same kernel availability. A
    // mixed result would make ggml's per-thread fallback decision diverge.
    if (!kernel_available) {
        return false;
    }
    const int active_rank = iqk_dsv4_attn_wo_a_active_rank(ith, nth);
    if (active_rank < 0) {
        return true;
    }
    const int group = active_rank % IQK_DSV4_ATTN_WO_A_GROUPS;
    return iqk_mul_mat(
            IQK_DSV4_ATTN_WO_A_NX,
            IQK_DSV4_ATTN_WO_A_NY,
            IQK_DSV4_ATTN_WO_A_K,
            GGML_TYPE_BF16, (const char *) A + group*nb02, strideA,
            GGML_TYPE_F32, (const char *) B + group*nb12, strideB,
            C + group*nb2, stride_C,
            active_rank/IQK_DSV4_ATTN_WO_A_GROUPS,
            IQK_DSV4_ATTN_WO_A_GROUP_WORKERS);
}

static inline int iqk_numa_mirror_small_tile_count(
        long Nx, long Ny, long ne00,
        long ne02, long ne03, long ne12, long ne13,
        int typeA, int typeB, int nth) {
    if (nth != 128 ||
            typeA != GGML_TYPE_BF16 || typeB != GGML_TYPE_F32 ||
            Ny != IQK_NUMA_MIRROR_SMALL_NY ||
            ne00 != IQK_NUMA_MIRROR_SMALL_K ||
            ne02 != 1 || ne03 != 1 || ne12 != 1 || ne13 != 1) {
        return 0;
    }
    return Nx == 1024 ?
        (int) Nx/IQK_NUMA_MIRROR_SMALL_TILE : 0;
}

static inline int iqk_numa_mirror_small_use_partition(
        bool tensor_mirrored, bool mirror_active, int n_nodes,
        uint32_t mirror_flags,
        long Nx, long Ny, long ne00,
        long ne02, long ne03, long ne12, long ne13,
        int typeA, int typeB, int nth) {
    if (!tensor_mirrored ||
            !iqk_dsv4_attn_wo_a_numa_matches(
                    mirror_active, n_nodes, mirror_flags)) {
        return 0;
    }
    return iqk_numa_mirror_small_tile_count(
            Nx, Ny, ne00, ne02, ne03, ne12, ne13,
            typeA, typeB, nth);
}

// Map an interval-center physical owner back to its logical tile rank.
// This preserves the existing 16-row tiles while selecting half of their
// owners from each contiguous block of a two-node NUMA mirror team.
static inline int iqk_numa_mirror_small_active_rank(
        int ith, int nth, int n_active) {
    if (ith < 0 || ith >= nth || n_active <= 0 || nth < n_active) {
        return -1;
    }
    const int rank = (int) (((2*(int64_t) ith + 1)*n_active)/(2*(int64_t) nth));
    const int owner =
        (int) (((2*(int64_t) rank + 1)*nth)/(2*(int64_t) n_active));
    if (owner == ith) {
        return rank;
    }
    return -1;
}

static inline void iqk_numa_mirror_small_team(
        int ith, int nth, int n_active,
        int * logical_ith, int * logical_nth) {
    if (n_active <= 0) {
        *logical_ith = ith;
        *logical_nth = nth;
        return;
    }
    const int rank =
        iqk_numa_mirror_small_active_rank(ith, nth, n_active);
    *logical_ith = rank >= 0 ? rank : n_active;
    // Reserve one valid, idle logical rank so inactive physical workers still
    // satisfy ith < nth while running the same IQK capability check.
    *logical_nth = n_active + 1;
}
