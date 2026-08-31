//
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "iqk_config.h"
#include "iqk_mul_mat.h"
#include "iqk_flash_impl.h"
#include "ggml.h"

#if defined IQK_IMPLEMENT && defined GGML_IQK_FLASH_ATTENTION

#include <algorithm>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <unordered_set>

namespace {
inline uint32_t simple_gcd(uint32_t a, uint32_t b) {
    while (a != b) {
        if (a > b) a -= b;
        else b -= a;
    }
    return a;
}
inline size_t split_k_work_size(int nek1, int nek2, int rk2, int rv2, int Dv, int nth) {
    if (nek1 < 1 || nek2 < 1 || rk2 < 1 || Dv < 1 || nth < 1) {
        return 0;
    }

    if (nek2 == 1 && nek1/32 > 1) {
        int nstep_k = nek1/32;
        if (nstep_k >= 4*nth) {
            return size_t(Dv + 16)*rk2*sizeof(float)*nth;
        }
        int gcd_k = simple_gcd(nstep_k, nth);
        int nth_k = nth/gcd_k;
        int nq_per_thread = (rk2 + nth_k - 1)/nth_k;
        if (nq_per_thread > 1) {
            return size_t(Dv + 16)*nq_per_thread*sizeof(float)*nth;
        }
    }

    if (rk2 != rv2 || int64_t(nek2)*nek1 < 32LL*nth) {
        return 0;
    }

    int gcd = simple_gcd(nek2, nth);
    int nth_k = nth/gcd;
    int nek2_k = nek2/gcd;
    int nchunk = nek2_k*nek1/32;
    int npt = (nchunk + nth_k - 1)/nth_k;
    int nk;
    if (npt*nth_k == nchunk) {
        nk = 32 * (int64_t(nek1)*nek2/(32*nth));
    } else {
        int nm = 1;
        while (nm*4 < npt) {
            nm *= 2;
        }
        nk = 32*nm;
    }
    int nkk = (nek1 + nk - 1)/nk;
    int nstep_k = nek2*nkk;
    size_t result_size = size_t(Dv + 16)*rk2*sizeof(float);
    return nstep_k*result_size;
}

inline int effective_k_rows(const ggml_tensor * dst, int nek1) {
    const auto Q = dst->src[0];
    const auto mask = dst->src[3];
    const int n_swa = dst->op_params[4];
    if (n_swa <= 0 || !mask) {
        return nek1;
    }

    constexpr int kMinBatch = 256;
    const int ntokens = std::max<int64_t>(kMinBatch, Q->ne[1]);
    const int nblock = (ntokens + n_swa + kMinBatch - 1)/kMinBatch;
    return std::min(nek1, nblock*kMinBatch);
}

inline size_t max_split_k_work_size(const ggml_tensor * dst, int nth) {
    const auto Q = dst->src[0];
    const auto K = dst->src[1];
    const auto V = dst->src[2];
    const auto mask = dst->src[3];
    const int rk2 = Q->ne[2]/K->ne[2];
    const int rv2 = Q->ne[2]/V->ne[2];
    const int nek2 = K->ne[2];
    const int nek1 = effective_k_rows(dst, K->ne[1]);

    size_t result = split_k_work_size(nek1, nek2, rk2, rv2, V->ne[0], nth);
    if (!mask || nek1 <= 256) {
        return result;
    }

    // The runtime may narrow the key range after inspecting the mask. Its thread split can
    // require more scratch than the split planned from the full tensor shape, so cover the
    // 32-row ranges that can survive the 3/4-size narrowing threshold.
    const int max_narrowed_chunks = int64_t(3)*nek1/4/32;
    // Check every small split because gcd changes make its scratch size non-monotonic.
    // The loop is bounded by four entries per worker, independent of context length.
    const int first_chunk = nek2 == 1 ? 2 : 1;
    const int last_chunk = std::min(max_narrowed_chunks, 4*nth);
    for (int chunks = first_chunk; chunks <= last_chunk; ++chunks) {
        result = std::max(result, split_k_work_size(32*chunks, nek2, rk2, rv2, V->ne[0], nth));
    }
    if (max_narrowed_chunks <= last_chunk || nek2 == 1) {
        return result;
    }

    if (rk2 == rv2 && int64_t(nek2)*max_narrowed_chunks >= nth) {
        // Beyond the exact-search range, nstep_k is a multiple of nek2 and is
        // bounded by nek2*ceil(4*nth/nek2).
        const int64_t max_steps = int64_t(nek2)*((int64_t(4)*nth + nek2 - 1)/nek2);
        const size_t result_size = size_t(V->ne[0] + 16)*rk2*sizeof(float);
        result = std::max(result, size_t(max_steps)*result_size);
    }
    return result;
}
inline void accumulate_qkv(int Dv, float& M, float& S, float Mj, float Sj, float * Racc, const float * R) {
    if (Mj == -INFINITY) return;
    if (Mj > M) {
        if (M == -INFINITY) {
            std::memcpy(Racc, R, Dv*sizeof(float));
            S = Sj;
        } else {
            float c = exp(M - Mj);
            S = c*S + Sj;
            for (int i = 0; i < Dv; ++i) Racc[i] = c*Racc[i] + R[i];
        }
        M = Mj;
    } else {
        float c = exp(Mj - M);
        S += c*Sj;
        for (int i = 0; i < Dv; ++i) Racc[i] += c*R[i];
    }
}

static inline const std::unordered_set<ggml_type> & supported_kv_types() {
#ifdef GGML_IQK_FA_ALL_QUANTS
    static std::unordered_set<ggml_type> k_supported = {
        GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q8_KV, GGML_TYPE_Q6_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL
    };
#else
    static std::unordered_set<ggml_type> k_supported = {
        GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q8_KV, GGML_TYPE_Q6_0,
    };
#endif
    return k_supported;
}

static inline bool are_kv_types_supported(ggml_type type_k, ggml_type type_v) {
    if (type_k == GGML_TYPE_BF16) {
        if (type_v != type_k) {
            return false;
        }
#ifdef __AVX512BF16__
        return true;
#else
        return false;
#endif
    }
    auto & supported = supported_kv_types();
    auto it_k = supported.find(type_k);
    auto it_v = supported.find(type_v);
    return it_k != supported.end() && it_v != supported.end();
}

bool indexed_fa_dims_supported(int Dk, int Dv) {
    return (Dk == 576 && Dv == 512) ||
        (Dk == 512 && Dv == 512) ||
        (Dk == 320 && Dv == 256) ||
        (Dk == 192 && Dv == 128) ||
        (Dk == 192 && Dv == 192) ||
        (Dk == 256 && Dv == 256) ||
        (Dk == 128 && Dv == 128) ||
        (Dk == 96  && Dv == 96)  ||
        (Dk == 64  && Dv == 64);
}

bool indexed_fa_types_supported(ggml_type type_k, ggml_type type_v, int Dk, int Dv) {
    if (type_k == GGML_TYPE_BF16 || type_v == GGML_TYPE_BF16) {
#ifdef __AVX512BF16__
        return type_k == GGML_TYPE_BF16 && type_v == GGML_TYPE_BF16;
#else
        return false;
#endif
    }

    const bool unequal_head_sizes = Dk != Dv;
    if (unequal_head_sizes && type_k != type_v) {
        return false;
    }

    const bool supported_k = type_k == GGML_TYPE_F16 ||
        type_k == GGML_TYPE_Q8_0 || type_k == GGML_TYPE_Q6_0
#ifdef GGML_IQK_FA_ALL_QUANTS
        || type_k == GGML_TYPE_Q8_KV || type_k == GGML_TYPE_Q4_0 ||
        type_k == GGML_TYPE_Q4_1 || type_k == GGML_TYPE_IQ4_NL
#endif
        ;
    const bool supported_v = type_v == GGML_TYPE_F16 ||
        type_v == GGML_TYPE_Q8_0 || type_v == GGML_TYPE_Q8_KV ||
        type_v == GGML_TYPE_Q6_0
#ifdef GGML_IQK_FA_ALL_QUANTS
        || type_v == GGML_TYPE_Q4_0 || type_v == GGML_TYPE_Q4_1 ||
        type_v == GGML_TYPE_IQ4_NL
#endif
        ;
    return supported_k && supported_v;
}

bool indexed_fa_compatible(
        const ggml_tensor * indexer,
        int type_q,
        int type_mask,
        float max_bias,
        int neq3,
        int neq2,
        int neq1,
        int nek3,
        int nek2,
        int nek1,
        int nev3,
        int nev2,
        ggml_type type_k,
        ggml_type type_v,
        int Dk,
        int Dv,
        bool have_mask) {
    return indexer != nullptr &&
        indexer->type == GGML_TYPE_I32 &&
        indexer->ne[0] > 0 && indexer->ne[0] < nek1 && indexer->ne[0] % 32 == 0 &&
        indexer->ne[1] >= neq1 && indexer->ne[2] == 1 && indexer->ne[3] == 1 &&
        indexer->nb[0] == sizeof(int32_t) && indexer->nb[1] >= (size_t) indexer->ne[0] * sizeof(int32_t) &&
        type_q == GGML_TYPE_F32 && type_mask == GGML_TYPE_F16 && max_bias <= 0.0f && have_mask &&
        neq3 == 1 && nek3 == 1 && nev3 == 1 &&
        nek2 > 0 && nek2 == nev2 && neq2 % nek2 == 0 &&
        indexed_fa_types_supported(type_k, type_v, Dk, Dv) && indexed_fa_dims_supported(Dk, Dv) &&
        (neq1 != 1 || neq2 <= 256);
}
}

size_t iqk_fa_work_buffer_size(const struct ggml_tensor * dst, int nth) {
    auto Q = dst->src[0];
    auto K = dst->src[1];
    auto V = dst->src[2];
    auto M = dst->src[3];
    auto indexer = dst->src[5];
    float max_bias = 0.0f;
    std::memcpy(&max_bias, (const char *) dst->op_params + sizeof(float), sizeof(float));
    if (indexed_fa_compatible(indexer,
            Q->type, M ? M->type : GGML_TYPE_COUNT, max_bias,
            Q->ne[3], Q->ne[2], Q->ne[1],
            K->ne[3], K->ne[2], K->ne[1],
            V->ne[3], V->ne[2], K->type, V->type, K->ne[0], V->ne[0], M != nullptr)) {
        auto row_size_k = ggml_row_size(K->type, K->ne[0]);
        auto row_size_v = ggml_row_size(V->type, V->ne[0]);
        if (Q->ne[1] == 1) {
            return (row_size_k * K->ne[2] + row_size_v * V->ne[2] +
                    sizeof(ggml_fp16_t) * K->ne[2]) * indexer->ne[0]
                + 512 * sizeof(float);
        }
        return (row_size_k + row_size_v + sizeof(ggml_fp16_t)) * indexer->ne[0] * nth;
    }
    size_t size = 0;
    if (Q->ne[1] >= 8 && K->type == GGML_TYPE_Q8_0) {
        size = ggml_row_size(GGML_TYPE_Q8_0, K->ne[0]) * K->ne[1]*K->ne[2]*K->ne[3];
    }
    if (Q->ne[1] == 1 && Q->ne[3] == 1 && Q->ne[2]/K->ne[2] > 1 && nth >= 1) {
        size += max_split_k_work_size(dst, nth);
        return size;
    }
    return size;
}

// TODO: get the ggml_type enum here without polution
//
extern "C" IQK_API bool iqk_flash_attn_noalibi(int type_q, int type_mask, float max_bias,
                            int neq3, int neq2, long nbq3, long nbq2,
                            int nek3, int nek2, long nbk3, long nbk2,
                            int nev3, int nev2, long nbv3, long nbv2,
                            int ne2,  int ne1,  long nb1,
                            int int_type_k_in,      // type of k
                            int int_type_v,         // type of v
                            int Dk,                 // K head size
                            int Dv,                 // V head size
                            int neq1,               // number of columns in q
                            int nek1,               // number of rows in k
                            int stride_q,           // distance between q columns in bytes
                            int stride_k,           // distance between k rows in bytes
                            int stride_v,           // distance between v rows in bytes
                            int stride_m,           // distance between mask rows (in bytes
                            const void  * q,        // q matrix.
                            const void  * k,        // k matrix. Assumed to be fp16, nq x nk elements
                            const void  * v,        // v matrix. Assumed to be fp16, nq x nk elements
                            const void  * mask,     // mask. If not null, assumed to be fp16. nq x nk elements
                            const void  * sinks,    // mask. If not null, assumed to be fp16. nq x nk elements
                            float         scale,    // scale applied before softmax
                            float         softcap,  // if > 0, a "soft-cap" operation is applied before softmax
                            float       * qkv,      // v*softmax(scale*(k*q))
                            [[maybe_unused]] void * work_buffer_in, [[maybe_unused]] barrier_t barrier, [[maybe_unused]] void * barrier_data,
                            int ith, int nth, int n_swa, [[maybe_unused]] ggml_tensor * indexer) {

    if (type_q != 0 || type_mask != 1 || max_bias > 0) return false;

    if (indexed_fa_compatible(indexer,
            type_q, type_mask, max_bias,
            neq3, neq2, neq1,
            nek3, nek2, nek1,
            nev3, nev2, ggml_type(int_type_k_in), ggml_type(int_type_v), Dk, Dv, mask != nullptr)) {
        // Each selected row needs one packed K row, one packed V row and one mask value.
        // Query heads are grouped by KV head. Single-token decode gives each KV head a
        // contiguous thread team and a shared packed slot, preserving NUMA locality. A
        // multi-token batch gives each worker a private slot reused across (token, KV-head)
        // tasks, so no synchronization is needed there.
        const size_t row_size_k = ggml_row_size(ggml_type(int_type_k_in), Dk);
        const size_t row_size_v = ggml_row_size(ggml_type(int_type_v), Dv);
        const size_t work_size = (row_size_k + row_size_v + sizeof(ggml_fp16_t)) * indexer->ne[0];
        const ggml_fp16_t h_inf = ggml_fp32_to_fp16(-INFINITY);
        const int index_width = indexer->ne[0];
        const int q_per_kv = neq2 / nek2;

        if (neq1 == 1) {
            const auto idx = (const int *) indexer->data;
            const auto M = (const ggml_fp16_t *) mask;
            auto work_k = (char *) work_buffer_in;
            auto work_v = work_k + row_size_k*index_width*nek2;
            auto work_m = (ggml_fp16_t *) (work_v + row_size_v*index_width*nev2);
            int last_found = -1;

            for (int j = index_width - 1; j >= 0; --j) {
                if (idx[j] >= 0) {
                    last_found = j;
                    break;
                }
            }

            int team_head = -1;
            int team_first = 0;
            int team_size = 0;
            int team_rank = 0;
            if (nth >= nek2) {
                team_head = ith*nek2/nth;
                team_first = (team_head*nth + nek2 - 1)/nek2;
                const int team_last = ((team_head + 1)*nth + nek2 - 1)/nek2;
                team_size = team_last - team_first;
                team_rank = ith - team_first;
            }

            auto pack_head = [&](int ikv, int first_j, int step_j) {
                auto head_k = work_k + row_size_k*ikv*index_width;
                auto head_v = work_v + row_size_v*ikv*index_width;
                auto head_m = work_m + ikv*index_width;
                for (int j = first_j; j < index_width; j += step_j) {
                    if (idx[j] >= 0) {
                        GGML_ASSERT(idx[j] < nek1);
                        std::memcpy(head_k + row_size_k*j,
                                (const char *) k + ikv*nbk2 + idx[j]*stride_k, row_size_k);
                        if (k != v) {
                            std::memcpy(head_v + row_size_v*j,
                                    (const char *) v + ikv*nbv2 + idx[j]*stride_v, row_size_v);
                        }
                        head_m[j] = M[idx[j]];
                    } else {
                        std::memset(head_k + row_size_k*j, 0, row_size_k);
                        if (k != v) {
                            std::memset(head_v + row_size_v*j, 0, row_size_v);
                        }
                        head_m[j] = h_inf;
                    }
                }
            };

            if (nth >= nek2) {
                pack_head(team_head, team_rank, team_size);
            } else {
                for (int ikv = ith; ikv < nek2; ikv += nth) {
                    pack_head(ikv, 0, 1);
                }
            }
            barrier(barrier_data);
            if (last_found < 0) {
                if (nth >= nek2) {
                    const int npt = (q_per_kv + team_size - 1)/team_size;
                    const int query_offset = team_rank*npt;
                    const int query_count = std::min(npt, q_per_kv - query_offset);
                    for (int h = 0; h < query_count; ++h) {
                        std::memset((char *) qkv +
                                (team_head*q_per_kv + query_offset + h)*nb1,
                                0, Dv*sizeof(float));
                    }
                } else {
                    for (int ikv = ith; ikv < nek2; ikv += nth) {
                        for (int h = 0; h < q_per_kv; ++h) {
                            std::memset((char *) qkv + (ikv*q_per_kv + h)*nb1,
                                    0, Dv*sizeof(float));
                        }
                    }
                }
                return true;
            }

            const int packed_rows = GGML_PAD(last_found + 1, 32);
            auto compute_head = [&](int ikv, int query_first, int query_count) {
                const auto this_q = (const char *) q + query_first*nbq2;
                auto this_qkv = qkv + query_first*nb1/sizeof(float);
                const auto this_k = work_k + row_size_k*ikv*index_width;
                const auto this_v = k == v ? this_k : work_v + row_size_v*ikv*index_width;
                const auto this_m = work_m + ikv*index_width;
                const auto this_sinks = sinks ? (const float *) sinks + query_first : nullptr;
                return iqk_flash_attn_impl(int_type_k_in, int_type_v,
                        Dk, Dv, query_count, packed_rows,
                        nbq2, row_size_k, row_size_v, 0, Dv,
                        (const float *) this_q, this_k, this_v, this_m, this_sinks, 1,
                        scale, softcap, this_qkv, nullptr, nullptr);
            };

            if (nth >= nek2) {
                const int npt = (q_per_kv + team_size - 1)/team_size;
                const int group_offset = team_head*q_per_kv;
                const int query_offset = team_rank*npt;
                const int query_count = std::min(npt, q_per_kv - query_offset);
                if (query_count > 0 && !compute_head(
                            team_head, group_offset + query_offset, query_count)) {
                    return false;
                }
            } else {
                for (int ikv = ith; ikv < nek2; ikv += nth) {
                    if (!compute_head(ikv, ikv*q_per_kv, q_per_kv)) {
                        return false;
                    }
                }
            }
            return true;
        }

        auto work_k = (char *) work_buffer_in + ith*work_size;
        auto work_v = work_k + row_size_k*index_width;
        auto work_m = (ggml_fp16_t *) (work_v + row_size_v*index_width);
        auto compute_task = [&](int iq, int ikv) {
            const auto idx = (const int *) ((const char *) indexer->data + iq*indexer->nb[1]);
            const auto M = (const ggml_fp16_t *) ((const char *) mask + iq*stride_m);
            int last_found = -1;
            for (int j = 0; j < index_width; ++j) {
                if (idx[j] >= 0) {
                    GGML_ASSERT(idx[j] < nek1);
                    work_m[j] = M[idx[j]];
                    last_found = j;
                } else {
                    work_m[j] = h_inf;
                }
            }
            if (last_found < 0) {
                const int query_first = ikv*q_per_kv;
                auto this_qkv = (char *) qkv + iq*ne1*nb1 + query_first*nb1;
                for (int h = 0; h < q_per_kv; ++h) {
                    std::memset(this_qkv + h*nb1, 0, Dv*sizeof(float));
                }
                return true;
            }

            const int packed_rows = GGML_PAD(last_found + 1, 32);
            for (int j = 0; j < index_width; ++j) {
                if (idx[j] >= 0) {
                    std::memcpy(work_k + row_size_k*j,
                            (const char *) k + ikv*nbk2 + idx[j]*stride_k, row_size_k);
                    if (k != v) {
                        std::memcpy(work_v + row_size_v*j,
                                (const char *) v + ikv*nbv2 + idx[j]*stride_v, row_size_v);
                    }
                } else {
                    std::memset(work_k + row_size_k*j, 0, row_size_k);
                    if (k != v) {
                        std::memset(work_v + row_size_v*j, 0, row_size_v);
                    }
                }
            }

            const int query_first = ikv*q_per_kv;
            const auto this_q = (const char *) q + iq*stride_q + query_first*nbq2;
            auto this_qkv = qkv + iq*ne1*nb1/sizeof(float) + query_first*nb1/sizeof(float);
            const auto this_sinks = sinks ? (const float *) sinks + query_first : nullptr;
            return iqk_flash_attn_impl(int_type_k_in, int_type_v,
                    Dk, Dv, q_per_kv, packed_rows,
                    nbq2, row_size_k, row_size_v, 0, Dv,
                    (const float *) this_q, work_k, k == v ? work_k : work_v,
                    work_m, this_sinks, 1,
                    scale, softcap, this_qkv, nullptr, nullptr);
        };

        if (nth >= nek2) {
            const int team_head = ith*nek2/nth;
            const int team_first = (team_head*nth + nek2 - 1)/nek2;
            const int team_last = ((team_head + 1)*nth + nek2 - 1)/nek2;
            const int team_size = team_last - team_first;
            const int team_rank = ith - team_first;
            for (int iq = team_rank; iq < neq1; iq += team_size) {
                if (!compute_task(iq, team_head)) {
                    return false;
                }
            }
        } else {
            const int n_tasks = neq1*nek2;
            for (int task = ith; task < n_tasks; task += nth) {
                if (!compute_task(task/nek2, task % nek2)) {
                    return false;
                }
            }
        }
        return true;
    }

    if (auto type_k = ggml_type(int_type_k_in), type_v = ggml_type(int_type_v); !are_kv_types_supported(type_k, type_v)) {
        if (ith == 0) {
            fprintf(stderr, "\n==================== K cache %s coupled with V cache %s is not a supported combination on the CPU backend.\n",
                    ggml_type_name(type_k), ggml_type_name(type_v));
            auto & supported = supported_kv_types();
            fprintf(stderr, "Supported types are:\n");
            for (auto type : supported) {
                fprintf(stderr, "    %s\n", ggml_type_name(type));
            }
            fprintf(stderr, "    Warning: ik_llama.cpp does not support Q5_0 or Q5_1 KV cache on the CPU.\n");
#ifdef __AVX512BF16__
            fprintf(stderr, "    %s, but only if K and V are both %s\n", ggml_type_name(GGML_TYPE_BF16), ggml_type_name(GGML_TYPE_BF16));
#endif
#ifndef GGML_IQK_FA_ALL_QUANTS
            fprintf(stderr, "    To enable q4_0, q4_1, and iq4_nl KV cache types, recompile with -DGGML_IQK_FA_ALL_QUANTS=ON\n");
#endif
        }
        barrier(barrier_data);
        GGML_ABORT("Fatal error");
    }

    if (n_swa > 0 && mask) {
        constexpr int kMinBatch = 256;
        int ntokens = std::max(kMinBatch, neq1);
        int nblock  = (ntokens + n_swa + kMinBatch - 1)/kMinBatch;
        int first   = nek1 - nblock*kMinBatch;
        if (first > 0) {
            k = (const char *)k + int64_t(first)*stride_k;
            v = (const char *)v + int64_t(first)*stride_v;
            mask = (const uint16_t *)mask + first;
            nek1 -= first;
        }
    }

    int rk2 = neq2/nek2;
    int rv2 = neq2/nev2;
    int rk3 = neq3/nek3;
    int rv3 = neq3/nev3;

    int first_k = 0, last_k = nek1;
    if (neq3 == 1 && rk2 > 1 && neq1 == 1 && nek1 > 256 && mask) {
        // This is a quick hack for SWA models.
        // Given that the mask is the same for all layers, ideally we should determine the
        // cache bounds once, and reuse for the whole graph. But even with this simple hack
        // we get non-negligible performance gains for SWA models and long context.
        auto umask = (const uint16_t *)mask;
        for (; first_k < last_k; ++first_k) {
            if (umask[first_k] == 0) break;
        }
        if (first_k == last_k) {
            fprintf(stderr, "============================== %s: found empty attention mask: nek1 = %d, first_k = %d\n", __func__, nek1, first_k);
            GGML_ABORT("Fatal error");
        }
        for (; last_k > first_k; --last_k) {
            if (umask[last_k-1] == 0) break;
        }
        int non = 32*((last_k - first_k + 31)/32);
        first_k = std::max(0, last_k - non);
        last_k = std::min(first_k + non, nek1);
        //printf("nek1 = %d, first = %d, last = %d\n", nek1, first, last);
        if (last_k - first_k <= 3*nek1/4 && (last_k - first_k)%32 == 0) {
            //printf("Reducing from %d to %d\n", nek1, last_k - first_k);
            k = (const void *)((const char *)k + first_k*stride_k);
            v = (const void *)((const char *)v + first_k*stride_v);
            mask = (const void *)((const uint16_t *)mask + first_k);
            nek1 = last_k - first_k;
        }
    }

    int int_type_k = int_type_k_in;
    auto work_buffer = work_buffer_in;
    if (neq1 >= 8) {
        uint64_t row_size = 0;
        work_buffer = iqk_repack_k(int_type_k, Dk, nek1, nek2, nek3, stride_k, nbk2, nbk3, k, work_buffer_in, ith, nth, int_type_k, row_size);
        if (int_type_k != int_type_k_in) {
            stride_k = row_size;
            nbk2 = stride_k*nek1;
            nbk3 = nbk2*nek2;
            k = work_buffer_in;
            barrier(barrier_data);
        }
    }
    //uint64_t row_size = 0;
    //auto work_buffer = iqk_repack_k(int_type_k, Dk, nek1, nek2, nek3, stride_k, nbk2, nbk3, k, work_buffer_in, ith, nth, int_type_k, row_size);
    //if (int_type_k != int_type_k_in) {
    //    stride_k = row_size;
    //    nbk2 = stride_k*nek1;
    //    nbk3 = nbk2*nek2;
    //    k = work_buffer_in;
    //    barrier(barrier_data);
    //}

    // Getting confused all the time about where to load data from and store the results to
    // (especially when combining the results from the threads).
    // So, for now, making it work just for MLA (nek2 = 1).
    // I think it would also speed up things for GQA, but I'm leaving this for another day.
    if (neq3 == 1 && rk2 > 1 && neq1 == 1 && nth >= 1 && nek1/32 > 1 && nek2 == 1) {
        int nstep_k = nek1/32;
        if (nstep_k >= 4*nth) {
            int nstep_k_per_thread = (nstep_k + nth - 1)/nth;
            int ith_mid = nth;
            int nstep_k_this_thread = nstep_k_per_thread;
            if (nstep_k_per_thread*nth > nstep_k) {
                ith_mid = nstep_k - nth*(nstep_k_per_thread - 1);
                if (ith >= ith_mid) --nstep_k_this_thread;
            }
            //if (ith == 0) fprintf(stderr, "nstep_k = %d, nstep_k_per_thread = %d, ith_mid = %d\n", nstep_k, nstep_k_per_thread, ith_mid);
            nstep_k_per_thread *= 32;
            nstep_k_this_thread *= 32;

            auto kv_offset = ith <= ith_mid ? ith*nstep_k_per_thread
                                           : ith_mid*nstep_k_per_thread + (ith - ith_mid)*nstep_k_this_thread;
            auto kth = (const char *)k + kv_offset*stride_k;
            auto vth = (const char *)v + kv_offset*stride_v;
            auto qth = (const char *)q;
            auto mth = mask ? (const char *)mask + kv_offset*sizeof(uint16_t) : nullptr; // we don't have ggml_half available here

            auto work = (char *)work_buffer;
            auto size_thread = (Dv + 16)*rk2*sizeof(float);
            auto result_buffer = work;
            auto work_this_thread = (float *)(result_buffer + ith*size_thread);
            if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                     Dk, Dv, rk2, nstep_k_this_thread, nbq2, stride_k, stride_v, 0, Dv, //Dk*sizeof(uint16_t), Dv,
                     (const float *)qth, (const void *)kth, (const void *)vth, (const void *)mth, nullptr, 0,
                     scale, softcap,
                     work_this_thread, work_this_thread + (Dv+0)*rk2, work_this_thread + (Dv+1)*rk2)) return false;

            barrier(barrier_data);

            for (int j = ith; j < rk2; j += nth) {
                auto Racc = qkv + j*nb1/sizeof(float);
                float M = -INFINITY, S = 0;
                for (int jth = 0; jth < nth; ++jth) {
                    auto R = (const float *)(result_buffer + jth*size_thread);
                    auto Mj = R + Dv*rk2;
                    auto Sj = Mj + rk2;
                    R += j*Dv;
                    accumulate_qkv(Dv, M, S, Mj[j], Sj[j], Racc, R);
                }
                float norm = S > 0 ? 1/S : 1;
                for (int i = 0; i < Dv; ++i) Racc[i] *= norm;
            }
            return true;
        }
        int gcd_k   = simple_gcd(nstep_k, nth);
        if (gcd_k >= 1) {
            int nth_k = nth/gcd_k;
            int ith_k = ith%gcd_k;
            int ith_q = ith/gcd_k;
            int nq_per_thread = (rk2 + nth_k - 1)/nth_k;
            if (nq_per_thread > 1) {
                int ith_mid = nth_k;
                int nq_this_thread = nq_per_thread;
                if (nq_per_thread*nth_k > rk2) {
                    ith_mid = rk2 - nth_k*(nq_per_thread - 1);
                    if (ith_q >= ith_mid) --nq_this_thread;
                }
                int j_mid = ith_mid*nq_per_thread;
                auto work = (char *)work_buffer;
                auto size_thread = (Dv + 16)*nq_per_thread*sizeof(float);
                auto result_buffer = work;

                auto kth = (const char *)k + ith_k*(nek1/gcd_k)*stride_k;
                auto vth = (const char *)v + ith_k*(nek1/gcd_k)*stride_v;
                auto q_offset = ith_q < ith_mid ? ith_q*nq_per_thread*nbq2 : (ith_mid*nq_per_thread + (ith_q - ith_mid)*nq_this_thread)*nbq2;
                auto qth = (const char *)q + q_offset;
                auto mth = mask ? (const char *)mask + ith_k*(nek1/gcd_k)*sizeof(uint16_t) : nullptr; // we don't have ggml_half available here

                // Each thread will produce a result of size Dv*nq_this_thread*sizeof(float)
                // In addition, we need M, S for the nq_this_thread rows the thread is processing
                // => (Dv + 2)*nq_per_thread*sizeof(float). We use (Dv + 16) instead to make sure threads are not
                // writing onto the same cache line.
                auto work_this_thread = (float *)(result_buffer + ith*size_thread);
                if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                            Dk, Dv, nq_this_thread, nek1/gcd_k, nbq2, stride_k, stride_v, 0, Dv, //Dk*sizeof(uint16_t), Dv,
                            (const float *)qth, (const void *)kth, (const void *)vth, (const void *)mth, nullptr, 0,
                            scale, softcap,
                            work_this_thread, work_this_thread + (Dv+0)*nq_this_thread, work_this_thread + (Dv+1)*nq_this_thread)) return false;

                barrier(barrier_data);

                // There are nek1/gcd_k contributions for each j that we need to sum up
                // Thread i computed k/v (i%gcd_k)*(nek1/gcd_k) for j (i/gcd_k)*(rk2/nth_k)...((i/gcd_k)+1)*(rk2/nth_k) and results at offset i*size_thread

                // TODO: simdify this
                // TODO: if nth > rk2, have threads process portions of the rows instead of entire rows as it is now
                for (int j = ith; j < rk2; j += nth) {
                    auto Racc = qkv + j*nb1/sizeof(float);
                    float M = -INFINITY, S = 0;
                    int jth_first, jj, nq_this_j;
                    if (j < j_mid) {
                        jth_first = j/nq_per_thread;
                        jj = j%nq_per_thread;
                        nq_this_j = nq_per_thread;
                    } else {
                        jth_first = ith_mid + (j - j_mid)/(nq_per_thread-1);
                        jj = (j - j_mid)%(nq_per_thread-1);
                        nq_this_j = nq_per_thread - 1;
                    }
                    jth_first *= gcd_k;
                    for (int jth = jth_first; jth < jth_first + gcd_k; ++jth) {
                        auto R = (const float *)(result_buffer + jth*size_thread);
                        auto Mj = R + Dv*nq_this_j;
                        auto Sj = Mj + nq_this_j;
                        R += jj*Dv;
                        accumulate_qkv(Dv, M, S, Mj[jj], Sj[jj], Racc, R);
                    }
                    float norm = S > 0 ? 1/S : 1;
                    for (int i = 0; i < Dv; ++i) Racc[i] *= norm;
                }
                return true;
            }
        }
    }

    if (neq3 == 1 && rk2 > 1 && rk2 == rv2 && neq1 == 1 && nth >= 1 && nek2*nek1 >= 32*nth) {
        auto result_size = (Dv + 16)*rk2*sizeof(float);
        int gcd = simple_gcd(nek2, nth);
        int nth_k  = nth/gcd;
        int nek2_k = nek2/gcd;
        int nchunk = nek2_k*nek1/32;
        int npt = (nchunk + nth_k - 1)/nth_k;
        int nk;
        if (npt*nth_k == nchunk) {
            nk = 32 * (nek2*nek1/(32*nth));
        } else {
            //int nm = std::max(1, npt/8);
            int nm = 1;
            while (true) {
                if (nm*4 >= npt) break;
                nm *= 2;
            }
            nk = 32*nm;
        }
        //int nk = 32 * (nek2*nek1/(32*nth));
        int nkk = (nek1 + nk - 1)/nk;
        int nstep_k = nek2*nkk;
        //if (ith == 0) printf("rk2 = %d, nek1 = %d, nek2 = %d, nk = %d, nkk = %d, nstep_k = %d\n", (int)rk2, (int)nek1, (int)nek2, nk, nkk, nstep_k);
        for (int istep_k = ith; istep_k < nstep_k; istep_k += nth) {
            int ik02 = istep_k/nkk;
            int ik01 = nk*(istep_k - ik02*nkk);
            int this_nk = ik01 + nk <= nek1 ? nk : nek1 - ik01;
            if (this_nk <= 0) break;
            auto this_result = (float *)((char *)work_buffer + istep_k*result_size);
            auto this_q = (const float *)((const char *)q + ik02*rk2*nbq2);
            auto this_k = (const char *)k + ik01*stride_k + ik02*nbk2;
            auto this_v = (const char *)v + ik01*stride_v + ik02*nbv2;
            auto this_m = mask ? (const char *)mask + ik01*sizeof(uint16_t) : nullptr; // we don't have ggml_half available here
            if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                     Dk, Dv, rk2, this_nk, nbq2, stride_k, stride_v, 0, Dv,
                     this_q, (const void *)this_k, (const void *)this_v, (const void *)this_m, nullptr, 0,
                     scale, softcap, this_result, this_result + (Dv+0)*rk2, this_result + (Dv+1)*rk2)) return false;
        }

        barrier(barrier_data);

        // We have nkk results for each head
        for (int iq2 = ith; iq2 < neq2; iq2 += nth) {
            // ik02*rk2 + il = iq2 (il = 0...rk2-1) => ik02 = iq2/rk2, il = iq2%rk2;
            int ik02 = iq2/rk2;
            int il = iq2 - ik02*rk2;
            auto Racc = qkv + iq2*nb1/sizeof(float);
            //std::memset(Racc, 0, Dv*sizeof(float));
            float M = -INFINITY, S = 0;
            for (int ikk = 0; ikk < nkk; ++ikk) {
                int istep_k = ik02*nkk + ikk;
                auto this_result = (float *)((char *)work_buffer + istep_k*result_size);
                const float * R  = this_result + il*Dv;
                const float * Mj = this_result + Dv*rk2;
                const float * Sj = Mj + rk2;
                accumulate_qkv(Dv, M, S, Mj[il], Sj[il], Racc, R);
            }
            if (sinks) {
                float s = ((const float *)sinks)[iq2];
                if (s > M) {
                    float m = expf(M - s);
                    for (int i = 0; i < Dv; ++i) Racc[i] *= m;
                    S = S*m + 1;
                } else {
                    S += expf(s - M);
                }
            }
            float norm = S > 0 ? 1/S : 1;
            for (int i = 0; i < Dv; ++i) Racc[i] *= norm;
        }
        return true;
    }

    // I keep changing my mind what is the best strategy to split the threads when processing
    // multiple heads. This is my current thinking, the commented out code below was the previous.
    int ntg = nth/simple_gcd(neq2*neq3, nth);
    int neq1g = (neq1 + ntg - 1)/ntg;
    //int64_t work_per_slice = D*nek1*neq1;
    //int ntg = 1;
    //
    // When neq1 is large, it is better to have more than one thread process one (iq2,iq3) matrix
    // But we also want each thread to process the same amount of rows, so neq1 must be a multiple of
    // the number of threads processing the (iq2, iq3) matrix.
    //
    //if (neq1 >= 8*nth) {
    //    if      (nth%8 == 0 && neq1%8 == 0 && work_per_slice >= (1 << 23)) ntg = 8;
    //    else if (nth%4 == 0 && neq1%4 == 0 && work_per_slice >= (1 << 21)) ntg = 4;
    //    else if (nth%2 == 0 && neq1%2 == 0 && work_per_slice >= (1 << 19)) ntg = 2;
    //}
    int counter = 0;
    for (int64_t iq3 = 0; iq3 < neq3; iq3++) {
        for (int64_t iq2 = 0; iq2 < neq2; iq2++) {
            auto sinksf = sinks ? (const float *)sinks + iq2 : nullptr;
            if (counter++ % (nth/ntg) == ith/ntg) {
                int iq1 = (ith%ntg)*neq1g;
                int this_neq1 = std::min(neq1g, neq1-iq1);
                if (this_neq1 > 0) {
                if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                        Dk, Dv, this_neq1, nek1, stride_q, stride_k, stride_v, stride_m, ne1*nb1/sizeof(float),
                        (const float *)((const char *)q + iq2*nbq2 + iq3*nbq3 + iq1*stride_q),
                        (const void  *)((const char *)k + iq2/rk2*nbk2 + iq3/rk3*nbk3),
                        (const void  *)((const char *)v + iq2/rv2*nbv2 + iq3/rv3*nbv3),
                        mask ? (const void  *)((const char *)mask + iq1*stride_m) : nullptr, sinksf, 0,
                        scale, softcap,
                        (float *)((char *)qkv + (iq3*ne2*ne1 + iq2 + iq1*ne1)*nb1), nullptr, nullptr)) return false;
                }
            }
        }
    }

    return true;
}

#else

bool iqk_flash_attn_noalibi([[maybe_unused]] int type_q, [[maybe_unused]] int type_mask, [[maybe_unused]] float max_bias,
                            [[maybe_unused]] int neq3, [[maybe_unused]] int neq2, [[maybe_unused]] long nbq3, [[maybe_unused]] long nbq2,
                            [[maybe_unused]] int nek3, [[maybe_unused]] int nek2, [[maybe_unused]] long nbk3, [[maybe_unused]] long nbk2,
                            [[maybe_unused]] int nev3, [[maybe_unused]] int nev2, [[maybe_unused]] long nbv3, [[maybe_unused]] long nbv2,
                            [[maybe_unused]] int ne2,  [[maybe_unused]] int ne1,  [[maybe_unused]] long nb1,
                            [[maybe_unused]] int type_k,             // type of k
                            [[maybe_unused]] int type_v,             // type of v
                            [[maybe_unused]] int Dk,                 // K head size
                            [[maybe_unused]] int Dv,                 // V head size
                            [[maybe_unused]] int nq,                 // number of columns in q
                            [[maybe_unused]] int nk,                 // number of rows in k
                            [[maybe_unused]] int stride_q,           // distance between q columns in bytes
                            [[maybe_unused]] int stride_k,           // distance between k rows in bytes
                            [[maybe_unused]] int stride_v,           // distance between v rows in bytes
                            [[maybe_unused]] int stride_m,           // distance between mask rows (in bytes
                            [[maybe_unused]] const void  * q,        // q matrix.
                            [[maybe_unused]] const void  * k,        // k matrix. Assumed to be fp16, nq x nk elements
                            [[maybe_unused]] const void  * v,        // v matrix. Assumed to be fp16, nq x nk elements
                            [[maybe_unused]] const void  * mask,     // mask. If not null, assumed to be fp16. nq x nk elements
                            [[maybe_unused]] float         scale,    // scale applied before softmax
                            [[maybe_unused]] float         softcap,  // if > 0, a "soft-cap" operation is applied before softmax
                            [[maybe_unused]] float       * qkv,      // v*softmax(scale*(k*q))
                            [[maybe_unused]] void * work_buffer, [[maybe_unused]] barrier_t barrier, [[maybe_unused]] void * barrier_data,
                            [[maybe_unused]] int ith, [[maybe_unused]] int nth, [[maybe_unused]] int n_swa, [[maybe_unused]] ggml_tensor * indexer) {
    return false;
}

#endif
