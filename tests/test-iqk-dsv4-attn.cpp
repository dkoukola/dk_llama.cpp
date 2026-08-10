#include "ggml.h"
#include "iqk/iqk_mul_mat.h"
#include "iqk/iqk_mul_mat_internal.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace {

constexpr int kNx = IQK_DSV4_ATTN_WO_A_NX;
constexpr int kNy = IQK_DSV4_ATTN_WO_A_NY;
constexpr int kK = IQK_DSV4_ATTN_WO_A_K;
constexpr int kGroups = IQK_DSV4_ATTN_WO_A_GROUPS;
constexpr int kLogicalThreads = IQK_DSV4_ATTN_WO_A_WORKERS;
constexpr int kGroupThreads = IQK_DSV4_ATTN_WO_A_GROUP_WORKERS;

struct test_data {
    std::vector<ggml_bf16_t> weights;
    std::vector<float> input;
};

struct gate_config {
    bool mirror_active = true;
    int n_nodes = 2;
    uint32_t mirror_flags = GGML_NUMA_MIRROR_WEIGHTS;
    long Nx = kNx;
    long Ny = kNy;
    long ne00 = kK;
    long ne02 = kGroups;
    long ne03 = 1;
    long ne12 = kGroups;
    long ne13 = 1;
    int typeA = GGML_TYPE_BF16;
    int typeB = GGML_TYPE_F32;
    int n_threads = 96;
};

enum class reference_result {
    success,
    unavailable,
    failure,
};

bool use_partition(const gate_config & config) {
    return iqk_dsv4_attn_wo_a_use_partition(
            config.mirror_active, config.n_nodes, config.mirror_flags,
            config.Nx, config.Ny, config.ne00,
            config.ne02, config.ne03, config.ne12, config.ne13,
            config.typeA, config.typeB, config.n_threads);
}

bool check_partition_gate() {
    const gate_config accepted;
    if (!use_partition(accepted)) {
        std::puts("exact DeepSeek-V4 mirror shape was not accepted");
        return false;
    }

    const auto expect_rejected = [&](const char * name, const auto & mutate) {
        gate_config config = accepted;
        mutate(config);
        if (use_partition(config)) {
            std::fprintf(stderr, "unexpectedly accepted %s\n", name);
            return false;
        }
        return true;
    };
    return
        expect_rejected("non-mirror mode",
                [](gate_config & c) { c.mirror_active = false; }) &&
        expect_rejected("single-node mode",
                [](gate_config & c) { c.n_nodes = 1; }) &&
        expect_rejected("three-node mode",
                [](gate_config & c) { c.n_nodes = 3; }) &&
        expect_rejected("non-mirrored weights",
                [](gate_config & c) { c.mirror_flags = 0; }) &&
        expect_rejected("64-thread team",
                [](gate_config & c) { c.n_threads = 64; }) &&
        expect_rejected("F32 weights",
                [](gate_config & c) { c.typeA = GGML_TYPE_F32; }) &&
        expect_rejected("BF16 input",
                [](gate_config & c) { c.typeB = GGML_TYPE_BF16; }) &&
        expect_rejected("different output rank",
                [](gate_config & c) { ++c.Nx; }) &&
        expect_rejected("different token count",
                [](gate_config & c) { ++c.Ny; }) &&
        expect_rejected("different reduction size",
                [](gate_config & c) { ++c.ne00; }) &&
        expect_rejected("different weight groups",
                [](gate_config & c) { --c.ne02; }) &&
        expect_rejected("non-unit weight batch",
                [](gate_config & c) { ++c.ne03; }) &&
        expect_rejected("different output groups",
                [](gate_config & c) { --c.ne12; }) &&
        expect_rejected("non-unit output batch",
                [](gate_config & c) { ++c.ne13; });
}

bool check_worker_mapping(int n_threads) {
    std::array<int, kLogicalThreads> owners;
    owners.fill(-1);
    std::array<std::array<int, kGroups>, 2> node_group_counts = {};
    std::vector<int> output_owners((size_t) kGroups*kNx);

    int n_selected = 0;
    for (int thread = 0; thread < n_threads; ++thread) {
        const int rank = iqk_dsv4_attn_wo_a_active_rank(thread, n_threads);
        if (rank < 0) {
            continue;
        }
        if (rank >= kLogicalThreads || owners[rank] >= 0) {
            std::fprintf(stderr,
                    "invalid or duplicate logical rank %d at %d threads\n",
                    rank, n_threads);
            return false;
        }
        owners[rank] = thread;
        ++n_selected;

        const int group = rank % kGroups;
        const int group_thread = rank/kGroups;
        const int node = (2*thread)/n_threads;
        if (node < 0 || node >= 2) {
            std::fprintf(stderr,
                    "selected worker %d maps outside two NUMA nodes\n",
                    thread);
            return false;
        }
        ++node_group_counts[node][group];

        const int first_x = group_thread*(kNx/kGroupThreads);
        const int last_x = first_x + kNx/kGroupThreads;
        for (int x = first_x; x < last_x; ++x) {
            ++output_owners[(size_t) group*kNx + x];
        }
    }

    if (n_selected != kLogicalThreads) {
        std::fprintf(stderr,
                "selected %d workers instead of %d at %d threads\n",
                n_selected, kLogicalThreads, n_threads);
        return false;
    }
    for (int rank = 0; rank < kLogicalThreads; ++rank) {
        if (owners[rank] < 0) {
            std::fprintf(stderr,
                    "logical rank %d is missing at %d threads\n",
                    rank, n_threads);
            return false;
        }
    }
    for (int node = 0; node < 2; ++node) {
        for (int group = 0; group < kGroups; ++group) {
            if (node_group_counts[node][group] != kGroupThreads/2) {
                std::fprintf(stderr,
                        "node %d contributes %d workers to group %d "
                        "instead of %d at %d threads\n",
                        node, node_group_counts[node][group], group,
                        kGroupThreads/2, n_threads);
                return false;
            }
        }
    }
    for (size_t i = 0; i < output_owners.size(); ++i) {
        if (output_owners[i] != 1) {
            std::fprintf(stderr,
                    "output partition %zu has %d owners at %d threads\n",
                    i, output_owners[i], n_threads);
            return false;
        }
    }
    return true;
}

test_data make_test_data() {
    test_data data;
    data.weights.resize((size_t) kGroups*kNx*kK);
    data.input.resize((size_t) kGroups*kNy*kK);

    for (int group = 0; group < kGroups; ++group) {
        for (int x = 0; x < kNx; ++x) {
            const size_t row = ((size_t) group*kNx + x)*kK;
            data.weights[row] = ggml_fp32_to_bf16(
                    (float) ((group + 1)*(x % 17 + 1))/32.0f);
            data.weights[row + 37] = ggml_fp32_to_bf16(
                    (float) ((x % 13) - group)/64.0f);
        }
        for (int y = 0; y < kNy; ++y) {
            // The production input is permuted from [K, group, token] to
            // [K, token, group], so tokens have a group-wide stride.
            const size_t row = ((size_t) y*kGroups + group)*kK;
            data.input[row] = (float) (y + 1)/7.0f;
            data.input[row + 37] = (float) (group - 2*y)/11.0f;
        }
    }
    return data;
}

reference_result run_reference(const test_data & data,
        std::vector<float> & output) {
    output.assign((size_t) kGroups*kNy*kNx,
            std::numeric_limits<float>::quiet_NaN());
    std::array<int, kLogicalThreads> status = {};
    std::vector<std::thread> threads;
    threads.reserve(kLogicalThreads);
    for (int thread = 0; thread < kLogicalThreads; ++thread) {
        threads.emplace_back([&, thread] {
            status[thread] = iqk_mul_mat_4d(
                    kNx, kNy, kK,
                    kGroups, 1, kGroups, 1,
                    (long) kNx*kK*sizeof(ggml_bf16_t),
                    (long) kGroups*kNx*kK*sizeof(ggml_bf16_t),
                    (long) kK*sizeof(float),
                    (long) kGroups*kNy*kK*sizeof(float),
                    (long) kNx*kNy,
                    (long) kGroups*kNx*kNy,
                    GGML_TYPE_BF16, data.weights.data(),
                    (long) kK*sizeof(ggml_bf16_t),
                    GGML_TYPE_F32, data.input.data(),
                    (long) kGroups*kK*sizeof(float),
                    output.data(), kNx,
                    thread, kLogicalThreads) ? 1 : 0;
        });
    }
    for (std::thread & thread : threads) {
        thread.join();
    }
    const int first_status = status[0];
    for (int thread = 1; thread < kLogicalThreads; ++thread) {
        if (status[thread] != first_status) {
            std::fprintf(stderr,
                    "reference workers disagreed about kernel availability: "
                    "worker 0=%d, worker %d=%d\n",
                    first_status, thread, status[thread]);
            return reference_result::failure;
        }
    }
    if (first_status == 0) {
        std::puts("IQK BF16 multiplication is unavailable; skipping values");
        return reference_result::unavailable;
    }
    return reference_result::success;
}

bool run_capped(const test_data & data, int n_threads,
        std::vector<float> & output) {
    output.assign((size_t) kGroups*kNy*kNx,
            std::numeric_limits<float>::quiet_NaN());
    std::vector<int> status(n_threads);
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int thread = 0; thread < n_threads; ++thread) {
        threads.emplace_back([&, thread] {
            status[thread] = iqk_dsv4_attn_wo_a_mul(
                    data.weights.data(),
                    (long) kK*sizeof(ggml_bf16_t),
                    (long) kNx*kK*sizeof(ggml_bf16_t),
                    data.input.data(),
                    (long) kGroups*kK*sizeof(float),
                    (long) kK*sizeof(float),
                    output.data(), kNx, (long) kNx*kNy,
                    thread, n_threads, true) ? 1 : 0;
        });
    }
    for (std::thread & thread : threads) {
        thread.join();
    }
    for (int thread = 0; thread < n_threads; ++thread) {
        if (status[thread] == 0) {
            std::fprintf(stderr,
                    "capped IQK multiplication failed for thread %d/%d\n",
                    thread, n_threads);
            return false;
        }
    }
    return true;
}

bool check_uniform_unsupported_result() {
    const test_data data = make_test_data();
    std::vector<float> output((size_t) kGroups*kNy*kNx);

    for (int n_threads : { 96, 128 }) {
        for (int thread = 0; thread < n_threads; ++thread) {
            if (iqk_dsv4_attn_wo_a_mul(
                        data.weights.data(),
                        (long) kK*sizeof(ggml_bf16_t),
                        (long) kNx*kK*sizeof(ggml_bf16_t),
                        data.input.data(),
                        (long) kGroups*kK*sizeof(float),
                        (long) kK*sizeof(float),
                        output.data(), kNx, (long) kNx*kNy,
                        thread, n_threads, false)) {
                std::fprintf(stderr,
                        "worker %d/%d accepted an unavailable kernel\n",
                        thread, n_threads);
                return false;
            }
        }
    }
    return true;
}

bool check_bit_identity() {
    const test_data data = make_test_data();
    std::vector<float> reference;
    const reference_result result = run_reference(data, reference);
    if (result == reference_result::failure) {
        return false;
    }
    if (result == reference_result::unavailable) {
        return true;
    }
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i])) {
            std::fprintf(stderr,
                    "reference output %zu is not finite\n", i);
            return false;
        }
    }

    for (int n_threads : { 64, 96, 128 }) {
        std::vector<float> actual;
        if (!run_capped(data, n_threads, actual)) {
            return false;
        }
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i])) {
                std::fprintf(stderr,
                        "capped output %zu is not finite at %d threads\n",
                        i, n_threads);
                return false;
            }
        }
        if (std::memcmp(reference.data(), actual.data(),
                    reference.size()*sizeof(float)) != 0) {
            std::fprintf(stderr,
                    "DeepSeek-V4 attention result differs bitwise at "
                    "%d threads\n",
                    n_threads);
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    if (!check_partition_gate()) {
        return 1;
    }
    for (int n_threads = 64; n_threads <= 256; ++n_threads) {
        if (!check_worker_mapping(n_threads)) {
            return 1;
        }
    }
    return check_uniform_unsupported_result() && check_bit_identity() ? 0 : 1;
}
