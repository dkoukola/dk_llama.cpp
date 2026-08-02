#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr int64_t kParallelMinElements = 8192;
constexpr int64_t kChunkElements = 256;
constexpr int64_t kReferenceRowSize = 16384;
constexpr float kEps = 1e-6f;
constexpr int kHighThreadCount = 64;

enum class rms_mode {
    out_of_place,
    in_place,
};

struct compute_result {
    std::vector<float> values;
    size_t work_size = 0;
    int n_threads = 0;
};

const char * mode_name(rms_mode mode) {
    return mode == rms_mode::in_place ? "in-place" : "out-of-place";
}

std::vector<float> make_input(int64_t row_size, int64_t n_rows) {
    std::vector<float> input(row_size*n_rows);
    for (size_t i = 0; i < input.size(); ++i) {
        const int32_t coarse = (37*(int32_t) (i % 257) + 11) % 257 - 128;
        const int32_t fine = (17*(int32_t) (i % 31) + 3) % 31;
        input[i] = coarse/32.0f + fine/1024.0f;
    }
    return input;
}

bool compute_rms(const std::vector<float> & input, int64_t row_size,
        int64_t n_rows, int n_nodes, rms_mode mode, int n_threads,
        compute_result & result) {
    const size_t tensor_bytes = input.size()*sizeof(float);
    const size_t n_tensors = (size_t) n_nodes + 1;
    const size_t mem_size = n_tensors*tensor_bytes +
        n_tensors*ggml_tensor_overhead() + ggml_graph_overhead() + 64*1024;
    const ggml_init_params params = {
        /* .mem_size   = */ mem_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create ggml context\n");
        return false;
    }

    ggml_tensor * current = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, row_size, n_rows);
    std::memcpy(current->data, input.data(), tensor_bytes);
    for (int i = 0; i < n_nodes; ++i) {
        current = mode == rms_mode::in_place ?
            ggml_rms_norm_inplace(ctx, current, kEps) :
            ggml_rms_norm(ctx, current, kEps);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, current);
    if (ggml_graph_n_nodes(graph) != n_nodes) {
        std::fprintf(stderr,
                "%s RMS graph contains %d nodes instead of %d\n",
                mode_name(mode), ggml_graph_n_nodes(graph), n_nodes);
        ggml_free(ctx);
        return false;
    }

    ggml_cplan plan = ggml_graph_plan(graph, n_threads);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.empty() ? nullptr : work.data();
    const ggml_status status = ggml_graph_compute(graph, &plan);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
                "%s RMS graph (%lld elements, %d nodes) failed with %d threads: %s\n",
                mode_name(mode), (long long) row_size, n_nodes, n_threads,
                ggml_status_to_string(status));
        ggml_free(ctx);
        return false;
    }

    result.values.resize(input.size());
    std::memcpy(result.values.data(), current->data, tensor_bytes);
    result.work_size = plan.work_size;
    result.n_threads = plan.n_threads;
    ggml_free(ctx);
    return true;
}

std::vector<float> rms_reference(std::vector<float> values,
        int64_t row_size, int64_t n_rows, int n_nodes, bool fixed_chunks) {
    for (int node = 0; node < n_nodes; ++node) {
        for (int64_t row = 0; row < n_rows; ++row) {
            const size_t row_begin = row*row_size;
            const size_t n_chunks = fixed_chunks ?
                (row_size - 1)/kChunkElements + 1 : 1;
            std::vector<double> chunk_sums(n_chunks);
            for (size_t chunk = 0; chunk < n_chunks; ++chunk) {
                const size_t begin = row_begin + chunk*kChunkElements;
                const size_t end = fixed_chunks ?
                    std::min(begin + (size_t) kChunkElements,
                            row_begin + (size_t) row_size) :
                    row_begin + row_size;
                double sum = 0.0;
                for (size_t i = begin; i < end; ++i) {
                    sum += (double) (values[i]*values[i]);
                }
                chunk_sums[chunk] = sum;
            }

            double sum = 0.0;
            for (double chunk_sum : chunk_sums) {
                sum += chunk_sum;
            }
            const float mean = sum/row_size;
            const float scale = 1.0f/std::sqrt(mean + kEps);
            for (int64_t i = 0; i < row_size; ++i) {
                values[row_begin + i] *= scale;
            }
        }
    }
    return values;
}

bool same_bits(const std::vector<float> & lhs,
        const std::vector<float> & rhs, const char * label) {
    if (lhs.size() != rhs.size() ||
            std::memcmp(lhs.data(), rhs.data(), lhs.size()*sizeof(float)) != 0) {
        std::fprintf(stderr, "%s results differ bitwise\n", label);
        return false;
    }
    return true;
}

bool check_close(const std::vector<float> & actual,
        const std::vector<float> & expected, const char * label) {
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "%s result size mismatch\n", label);
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        const float tolerance = 4*std::numeric_limits<float>::epsilon()*
            std::max(1.0f, std::fabs(expected[i]));
        if (!std::isfinite(actual[i]) ||
                std::fabs(actual[i] - expected[i]) > tolerance) {
            std::fprintf(stderr,
                    "%s mismatch at %zu: got %.9g, expected %.9g (tolerance %.9g)\n",
                    label, i, actual[i], expected[i], tolerance);
            return false;
        }
    }
    return true;
}

bool check_plan(const compute_result & result, int64_t row_size,
        int64_t n_rows, int requested_threads, const char * label) {
    const bool expects_work = row_size >= kParallelMinElements &&
        n_rows == 1 && requested_threads > 1;
    if ((result.work_size > 0) != expects_work) {
        std::fprintf(stderr, "%s unexpected work size %zu\n",
                label, result.work_size);
        return false;
    }
    if (result.n_threads != requested_threads) {
        std::fprintf(stderr, "%s planned %d threads instead of %d\n",
                label, result.n_threads, requested_threads);
        return false;
    }
    return true;
}

bool check_case(int64_t row_size, rms_mode mode, int n_nodes) {
    const std::vector<float> input = make_input(row_size, 1);
    const bool uses_parallel_path = row_size >= kParallelMinElements;
    const std::vector<float> reference = rms_reference(
            input, row_size, 1, n_nodes, uses_parallel_path);
    compute_result single;
    compute_result parallel_2;
    compute_result parallel_high;
    if (!compute_rms(input, row_size, 1, n_nodes, mode, 1, single) ||
            !compute_rms(input, row_size, 1, n_nodes, mode, 2, parallel_2) ||
            !compute_rms(input, row_size, 1, n_nodes, mode,
                    kHighThreadCount, parallel_high)) {
        return false;
    }

    char label[128];
    std::snprintf(label, sizeof(label), "%s RMS[%lld] x%d",
            mode_name(mode), (long long) row_size, n_nodes);
    if (!check_plan(single, row_size, 1, 1, label) ||
            !check_plan(parallel_2, row_size, 1, 2, label) ||
            !check_plan(parallel_high, row_size, 1,
                    kHighThreadCount, label) ||
            !check_close(single.values, reference, label) ||
            !check_close(parallel_2.values, reference, label) ||
            !check_close(parallel_high.values, reference, label)) {
        return false;
    }

    if (uses_parallel_path) {
        return same_bits(parallel_2.values, parallel_high.values, label);
    }
    return same_bits(single.values, parallel_2.values, label) &&
        same_bits(single.values, parallel_high.values, label);
}

bool check_width_matrix() {
    constexpr std::array<int64_t, 5> row_sizes = {
        8191,
        8192,
        8193,
        kReferenceRowSize,
        16385,
    };
    constexpr std::array<rms_mode, 2> modes = {
        rms_mode::out_of_place,
        rms_mode::in_place,
    };
    for (rms_mode mode : modes) {
        for (int64_t row_size : row_sizes) {
            if (!check_case(row_size, mode, 1)) {
                return false;
            }
        }
    }
    return true;
}

bool check_reference_thread_counts() {
    const std::vector<float> input = make_input(kReferenceRowSize, 1);
    compute_result parallel_32;
    compute_result parallel_64;
    if (!compute_rms(input, kReferenceRowSize, 1, 1,
                rms_mode::out_of_place, 32, parallel_32) ||
            !compute_rms(input, kReferenceRowSize, 1, 1,
                rms_mode::out_of_place, 64, parallel_64)) {
        return false;
    }
    return same_bits(parallel_32.values, parallel_64.values,
            "out-of-place RMS[16384] at 32/64 threads");
}

bool check_multi_row_fallback() {
    const std::vector<float> input = make_input(kReferenceRowSize, 2);
    compute_result single;
    compute_result parallel;
    if (!compute_rms(input, kReferenceRowSize, 2, 1,
                rms_mode::out_of_place, 1, single) ||
            !compute_rms(input, kReferenceRowSize, 2, 1,
                rms_mode::out_of_place, kHighThreadCount, parallel)) {
        return false;
    }
    return check_plan(single, kReferenceRowSize, 2, 1,
                "multi-row RMS fallback") &&
        check_plan(parallel, kReferenceRowSize, 2,
                kHighThreadCount, "multi-row RMS fallback") &&
        same_bits(single.values, parallel.values, "multi-row RMS fallback");
}

bool check_chained_nodes() {
    constexpr int n_nodes = 3;
    return check_case(8193, rms_mode::out_of_place, n_nodes) &&
        check_case(8193, rms_mode::in_place, n_nodes);
}

} // namespace

int main() {
    return check_width_matrix() && check_reference_thread_counts() &&
        check_multi_row_fallback() && check_chained_nodes() ? 0 : 1;
}
