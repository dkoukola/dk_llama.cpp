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

constexpr int kK = 4096;
constexpr int kNy = 8;
constexpr std::array<int, 1> kNxs = { 1024 };
constexpr std::array<int, 1> kThreadCounts = { 128 };

struct gate_config {
    bool tensor_mirrored = true;
    bool mirror_active = true;
    int n_nodes = 2;
    uint32_t mirror_flags = GGML_NUMA_MIRROR_WEIGHTS;
    long Nx = 1024;
    long Ny = kNy;
    long ne00 = kK;
    long ne02 = 1;
    long ne03 = 1;
    long ne12 = 1;
    long ne13 = 1;
    int typeA = GGML_TYPE_BF16;
    int typeB = GGML_TYPE_F32;
    int n_threads = 128;
};

struct test_data {
    std::array<std::vector<ggml_bf16_t>, 2> weights;
    std::vector<float> input;
};

enum class matrix_result {
    unavailable,
    available,
    mixed,
};

int tile_count(const gate_config & config) {
    return iqk_numa_mirror_small_use_partition(
            config.tensor_mirrored,
            config.mirror_active, config.n_nodes, config.mirror_flags,
            config.Nx, config.Ny, config.ne00,
            config.ne02, config.ne03, config.ne12, config.ne13,
            config.typeA, config.typeB, config.n_threads);
}

bool check_partition_gate() {
    for (int Nx : kNxs) {
        for (int n_threads : kThreadCounts) {
            gate_config config;
            config.Nx = Nx;
            config.n_threads = n_threads;
            if (tile_count(config) != Nx/IQK_NUMA_MIRROR_SMALL_TILE) {
                std::fprintf(stderr,
                        "rejected supported %d x %d shape at %d threads\n",
                        Nx, kNy, n_threads);
                return false;
            }
        }
    }

    const gate_config accepted;
    const auto expect_rejected = [&](const char * name, const auto & mutate) {
        gate_config config = accepted;
        mutate(config);
        if (tile_count(config) != 0) {
            std::fprintf(stderr, "unexpectedly accepted %s\n", name);
            return false;
        }
        return true;
    };
    return
        expect_rejected("unmirrored tensor",
                [](gate_config & c) { c.tensor_mirrored = false; }) &&
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
        expect_rejected("96-thread team",
                [](gate_config & c) { c.n_threads = 96; }) &&
        expect_rejected("129-thread team",
                [](gate_config & c) { c.n_threads = 129; }) &&
        expect_rejected("F32 weights",
                [](gate_config & c) { c.typeA = GGML_TYPE_F32; }) &&
        expect_rejected("BF16 input",
                [](gate_config & c) { c.typeB = GGML_TYPE_BF16; }) &&
        expect_rejected("256-row output",
                [](gate_config & c) { c.Nx = 256; }) &&
        expect_rejected("512-row output",
                [](gate_config & c) { c.Nx = 512; }) &&
        expect_rejected("unsupported output width",
                [](gate_config & c) { c.Nx = 768; }) &&
        expect_rejected("one input row",
                [](gate_config & c) { c.Ny = 1; }) &&
        expect_rejected("seven input rows",
                [](gate_config & c) { c.Ny = 7; }) &&
        expect_rejected("sixteen input rows",
                [](gate_config & c) { c.Ny = 16; }) &&
        expect_rejected("different reduction size",
                [](gate_config & c) { --c.ne00; }) &&
        expect_rejected("grouped weights",
                [](gate_config & c) { ++c.ne02; }) &&
        expect_rejected("batched weights",
                [](gate_config & c) { ++c.ne03; }) &&
        expect_rejected("grouped input",
                [](gate_config & c) { ++c.ne12; }) &&
        expect_rejected("batched input",
                [](gate_config & c) { ++c.ne13; });
}

bool check_worker_mapping(int n_tiles, int n_threads) {
    std::vector<int> owners(n_tiles, -1);
    std::array<int, 2> node_counts = {};

    int n_selected = 0;
    for (int thread = 0; thread < n_threads; ++thread) {
        const int rank = iqk_numa_mirror_small_active_rank(
                thread, n_threads, n_tiles);
        int logical_ith;
        int logical_nth;
        iqk_numa_mirror_small_team(
                thread, n_threads, n_tiles, &logical_ith, &logical_nth);
        if (logical_nth != n_tiles + 1 ||
                logical_ith != (rank >= 0 ? rank : n_tiles)) {
            std::fprintf(stderr,
                    "team mapping disagrees for worker %d/%d with %d tiles\n",
                    thread, n_threads, n_tiles);
            return false;
        }
        if (rank < 0) {
            continue;
        }
        if (rank >= n_tiles || owners[rank] >= 0) {
            std::fprintf(stderr,
                    "invalid or duplicate rank %d at %d threads/%d tiles\n",
                    rank, n_threads, n_tiles);
            return false;
        }
        owners[rank] = thread;
        ++n_selected;

        const int node = (2*thread)/n_threads;
        if (node < 0 || node >= 2) {
            std::fprintf(stderr,
                    "selected worker %d maps outside two NUMA blocks\n",
                    thread);
            return false;
        }
        ++node_counts[node];
    }

    if (n_selected != n_tiles) {
        std::fprintf(stderr,
                "selected %d workers instead of %d at %d threads\n",
                n_selected, n_tiles, n_threads);
        return false;
    }
    for (int rank = 0; rank < n_tiles; ++rank) {
        if (owners[rank] < 0) {
            std::fprintf(stderr,
                    "logical tile %d is missing at %d threads\n",
                    rank, n_threads);
            return false;
        }
    }
    for (int node = 0; node < 2; ++node) {
        if (node_counts[node] != n_tiles/2) {
            std::fprintf(stderr,
                    "NUMA block %d owns %d tiles instead of %d at %d threads\n",
                    node, node_counts[node], n_tiles/2, n_threads);
            return false;
        }
    }
    return true;
}

bool check_default_team() {
    if (iqk_numa_mirror_small_active_rank(0, 64, 0) != -1) {
        std::fprintf(stderr, "zero active workers produced a rank\n");
        return false;
    }
    for (int n_threads : { 64, 96, 128 }) {
        for (int thread = 0; thread < n_threads; ++thread) {
            int logical_ith;
            int logical_nth;
            iqk_numa_mirror_small_team(
                    thread, n_threads, 0, &logical_ith, &logical_nth);
            if (logical_ith != thread || logical_nth != n_threads) {
                std::fprintf(stderr,
                        "default team changed worker %d/%d to %d/%d\n",
                        thread, n_threads, logical_ith, logical_nth);
                return false;
            }
        }
    }
    return true;
}

test_data make_test_data(int Nx) {
    test_data data;
    for (auto & weights : data.weights) {
        weights.resize((size_t) Nx*kK);
    }
    data.input.resize((size_t) kNy*kK);

    for (int x = 0; x < Nx; ++x) {
        const size_t row = (size_t) x*kK;
        for (auto & weights : data.weights) {
            weights[row] = ggml_fp32_to_bf16(
                    (float) (x % 17 + 1)/32.0f);
            weights[row + 37] = ggml_fp32_to_bf16(
                    (float) ((x % 13) - 4)/64.0f);
            weights[row + 101] = ggml_fp32_to_bf16(
                    (float) ((x % 7) + 1)/128.0f);
        }
    }
    for (int y = 0; y < kNy; ++y) {
        const size_t row = (size_t) y*kK;
        data.input[row] = (float) (y + 1)/7.0f;
        data.input[row + 37] = (float) (3 - y)/11.0f;
        data.input[row + 101] = (float) (2*y + 1)/19.0f;
    }
    return data;
}

matrix_result run_matrix(const test_data & data, int Nx, int Ny,
        int n_threads, bool remap, int typeA, int weight_copy,
        std::vector<float> & output) {
    const int n_tiles = Nx/IQK_NUMA_MIRROR_SMALL_TILE;
    output.assign((size_t) Ny*Nx,
            std::numeric_limits<float>::quiet_NaN());
    std::vector<int> status(n_threads);
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int thread = 0; thread < n_threads; ++thread) {
        threads.emplace_back([&, thread] {
            int logical_ith = thread;
            int logical_nth = n_threads;
            if (remap) {
                iqk_numa_mirror_small_team(
                        thread, n_threads, n_tiles,
                        &logical_ith, &logical_nth);
            }
            const int node = weight_copy >= 0 ?
                weight_copy : (2*thread)/n_threads;
            status[thread] = iqk_mul_mat(
                    Nx, Ny, kK,
                    typeA, data.weights[node].data(),
                    (long) kK*sizeof(ggml_bf16_t),
                    GGML_TYPE_F32, data.input.data(),
                    (long) kK*sizeof(float),
                    output.data(), Nx,
                    logical_ith, logical_nth) ? 1 : 0;
        });
    }
    for (std::thread & thread : threads) {
        thread.join();
    }
    const int first_status = status[0];
    for (int thread = 1; thread < n_threads; ++thread) {
        if (status[thread] != first_status) {
            std::fprintf(stderr,
                    "workers disagreed for %d x %d at %d threads: "
                    "worker 0=%d, worker %d=%d\n",
                    Nx, Ny, n_threads,
                    first_status, thread, status[thread]);
            return matrix_result::mixed;
        }
    }
    return first_status != 0 ?
        matrix_result::available : matrix_result::unavailable;
}

bool check_uniform_unsupported_result() {
    const int Nx = kNxs.front();
    const int Ny = kNy;
    const test_data data = make_test_data(Nx);

    for (int n_threads : kThreadCounts) {
        std::vector<float> output;
        const matrix_result result = run_matrix(
                data, Nx, Ny, n_threads, true,
                GGML_TYPE_I8, -1, output);
        if (result != matrix_result::unavailable) {
            std::fprintf(stderr,
                    "unsupported IQK type returned %s at %d threads\n",
                    result == matrix_result::mixed ? "mixed status" : "success",
                    n_threads);
            return false;
        }
    }
    return true;
}

bool check_bit_identity() {
    for (int Nx : kNxs) {
        const test_data data = make_test_data(Nx);
        std::vector<float> reference;
        const matrix_result reference_result = run_matrix(
                data, Nx, kNy, 64, false,
                GGML_TYPE_BF16, 0, reference);
        if (reference_result == matrix_result::mixed) {
            return false;
        }
        if (reference_result == matrix_result::unavailable) {
            std::fprintf(stderr,
                    "skipping unavailable IQK kernel for %d x %d\n",
                    Nx, kNy);
            continue;
        }
        for (size_t i = 0; i < reference.size(); ++i) {
            if (!std::isfinite(reference[i])) {
                std::fprintf(stderr,
                        "reference output %zu is not finite for %d x %d\n",
                        i, Nx, kNy);
                return false;
            }
        }

        for (int n_threads : kThreadCounts) {
            std::vector<float> actual;
            if (run_matrix(data, Nx, kNy, n_threads, true,
                        GGML_TYPE_BF16, -1, actual) !=
                    matrix_result::available) {
                std::fprintf(stderr,
                        "remapped IQK kernel failed for %d x %d "
                        "at %d threads\n",
                        Nx, kNy, n_threads);
                return false;
            }
            if (std::memcmp(reference.data(), actual.data(),
                        reference.size()*sizeof(float)) != 0) {
                std::fprintf(stderr,
                        "result differs bitwise for %d x %d "
                        "at %d threads\n",
                        Nx, kNy, n_threads);
                return false;
            }
        }
    }
    return true;
}

bool check_physical_mirror_selection() {
    for (int Nx : kNxs) {
        test_data data = make_test_data(Nx);
        for (int x = 0; x < Nx; ++x) {
            const size_t row = (size_t) x*kK;
            data.weights[1][row] = ggml_fp32_to_bf16(
                    (float) (x % 17 + 33)/32.0f);
        }

        std::array<std::vector<float>, 2> references;
        for (int node = 0; node < 2; ++node) {
            const matrix_result result = run_matrix(
                    data, Nx, kNy, 64, false,
                    GGML_TYPE_BF16, node, references[node]);
            if (result == matrix_result::mixed) {
                return false;
            }
            if (result == matrix_result::unavailable) {
                // Availability is already handled by the bit-identity test.
                return true;
            }
        }

        const int n_tiles = Nx/IQK_NUMA_MIRROR_SMALL_TILE;
        for (int n_threads : kThreadCounts) {
            std::vector<float> actual;
            if (run_matrix(data, Nx, kNy, n_threads, true,
                        GGML_TYPE_BF16, -1, actual) !=
                    matrix_result::available) {
                return false;
            }

            std::vector<int> tile_nodes(n_tiles, -1);
            for (int thread = 0; thread < n_threads; ++thread) {
                const int rank = iqk_numa_mirror_small_active_rank(
                        thread, n_threads, n_tiles);
                if (rank >= 0) {
                    tile_nodes[rank] = (2*thread)/n_threads;
                }
            }
            for (int x = 0; x < Nx; ++x) {
                const int node = tile_nodes[x/IQK_NUMA_MIRROR_SMALL_TILE];
                if (node < 0) {
                    return false;
                }
                for (int y = 0; y < kNy; ++y) {
                    const size_t i = (size_t) y*Nx + x;
                    if (std::memcmp(&actual[i], &references[node][i],
                                sizeof(actual[i])) != 0) {
                        std::fprintf(stderr,
                                "physical mirror selection differs at output "
                                "%zu for %d x %d at %d threads\n",
                                i, Nx, kNy, n_threads);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

} // namespace

int main() {
    if (!check_partition_gate() || !check_default_team()) {
        return 1;
    }
    for (int n_tiles : { 16, 32, 64 }) {
        for (int n_threads = 65; n_threads <= 256; ++n_threads) {
            if (!check_worker_mapping(n_tiles, n_threads)) {
                return 1;
            }
        }
    }
    return check_uniform_unsupported_result() &&
        check_bit_identity() &&
        check_physical_mirror_selection() ? 0 : 1;
}
