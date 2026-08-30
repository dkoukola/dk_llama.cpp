#include "ggml.h"
#include "iqk/iqk_mul_mat.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace {

constexpr int kNx = 24;
constexpr int kNy = 8;
constexpr int kK = 16384;
constexpr int kParallelThreads = 64;
constexpr int kTaskCount = 12;
constexpr std::array<int, 6> kXOffsets = { 0, 5, 10, 15, 16, 21 };
constexpr std::array<int, 6> kXSizes = { 5, 5, 5, 1, 5, 3 };
constexpr std::array<int, kTaskCount> kTaskOwners = {
    2, 8, 13, 18, 24, 29, 34, 40, 45, 50, 56, 61,
};

bool mul_mat(
        const float * weights,
        const float * input,
        float * output,
        int thread,
        int n_threads) {
    return iqk_mul_mat(
        kNx, kNy, kK,
        GGML_TYPE_F32, weights, kK*sizeof(float),
        GGML_TYPE_F32, input, kK*sizeof(float),
        output, kNx, thread, n_threads);
}

bool run_parallel(
        const std::vector<const float *> & weights,
        const std::vector<float> & input,
        std::vector<float> & output) {
    const int n_threads = (int) weights.size();
    output.assign(kNx*kNy, std::numeric_limits<float>::quiet_NaN());
    std::vector<int> status(n_threads);
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int thread = 0; thread < n_threads; ++thread) {
        threads.emplace_back([&, thread] {
            status[thread] = mul_mat(
                weights[thread], input.data(), output.data(),
                thread, n_threads) ? 1 : 0;
        });
    }
    for (std::thread & thread : threads) {
        thread.join();
    }
    for (int thread = 0; thread < n_threads; ++thread) {
        if (status[thread] == 0) {
            std::fprintf(stderr,
                "IQK F32 multiplication failed for thread %d/%d\n",
                thread, n_threads);
            return false;
        }
    }
    return true;
}

std::vector<float> make_weights() {
    std::vector<float> weights((size_t) kNx*kK);
    for (size_t i = 0; i < weights.size(); ++i) {
        const int32_t value = (int32_t) ((i*37 + 11) % 257) - 128;
        weights[i] = value/127.0f;
    }
    return weights;
}

std::vector<float> make_input() {
    std::vector<float> input((size_t) kNy*kK);
    for (size_t i = 0; i < input.size(); ++i) {
        const int32_t value = (int32_t) ((i*17 + 3) % 251) - 125;
        input[i] = value/113.0f;
    }
    return input;
}

bool check_bit_identity(bool & available) {
    available = false;
    const std::vector<float> weights = make_weights();
    const std::vector<float> input = make_input();
    std::vector<float> reference(kNx*kNy);
    if (!mul_mat(weights.data(), input.data(), reference.data(), 0, 1)) {
        std::puts("IQK F32 multiplication is unavailable; skipping");
        return true;
    }
    available = true;

    for (int n_threads : { 11, 12, kParallelThreads }) {
        const std::vector<const float *> thread_weights(
            n_threads, weights.data());
        std::vector<float> actual;
        if (!run_parallel(thread_weights, input, actual)) {
            return false;
        }
        if (std::memcmp(reference.data(), actual.data(),
                    reference.size()*sizeof(float)) != 0) {
            std::fprintf(stderr,
                "IQK HC F32 result differs bitwise at %d threads\n",
                n_threads);
            return false;
        }
    }
    return true;
}

bool check_task_owners() {
    std::vector<float> input((size_t) kNy*kK);
    for (int y = 0; y < kNy; ++y) {
        input[(size_t) y*kK] = 1.0f;
    }

    std::vector<float> zero_weights((size_t) kNx*kK);
    std::array<std::vector<float>, kTaskCount> task_weights;
    std::vector<const float *> thread_weights(
        kParallelThreads, zero_weights.data());
    for (int task = 0; task < kTaskCount; ++task) {
        task_weights[task].resize((size_t) kNx*kK);
        for (int x = 0; x < kNx; ++x) {
            task_weights[task][(size_t) x*kK] = (float) (task + 1);
        }
        thread_weights[kTaskOwners[task]] = task_weights[task].data();
    }

    std::vector<float> output;
    if (!run_parallel(thread_weights, input, output)) {
        return false;
    }
    for (int y_tile = 0; y_tile < 2; ++y_tile) {
        for (int x_tile = 0; x_tile < (int) kXOffsets.size(); ++x_tile) {
            const int task = y_tile*(int) kXOffsets.size() + x_tile;
            for (int y = 4*y_tile; y < 4*(y_tile + 1); ++y) {
                for (int x = kXOffsets[x_tile];
                        x < kXOffsets[x_tile] + kXSizes[x_tile]; ++x) {
                    const float expected = (float) (task + 1);
                    const float actual = output[(size_t) y*kNx + x];
                    if (actual != expected) {
                        std::fprintf(stderr,
                            "IQK HC F32 task owner mismatch at (%d, %d): "
                            "got %.9g, expected %.9g\n",
                            x, y, actual, expected);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

bool check_short_inner_dimension() {
    constexpr int nx = 8;
    constexpr int ny = 1;
    constexpr int k = 4;
    std::array<float, nx*k + k> weights = {};
    std::array<float, ny*k + k> input = { 0.5f, -1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    for (int row = 0; row < nx; ++row) {
        for (int column = 0; column < k; ++column) {
            weights[(size_t) row*k + column] = (float) ((row + 1)*(column + 1))/8.0f;
        }
    }
    for (int column = 0; column < k; ++column) {
        weights[(size_t) nx*k + column] = (float) (column + 9);
    }

    std::array<float, nx*ny> output;
    output.fill(std::numeric_limits<float>::quiet_NaN());
    if (!iqk_mul_mat(
            nx, ny, k,
            GGML_TYPE_F32, weights.data(), k*sizeof(float),
            GGML_TYPE_F32, input.data(), k*sizeof(float),
            output.data(), nx, 0, 1)) {
        std::fprintf(stderr, "IQK F32 multiplication rejected a four-column input\n");
        return false;
    }

    for (int row = 0; row < nx; ++row) {
        float expected = 0.0f;
        for (int column = 0; column < k; ++column) {
            expected += weights[(size_t) row*k + column]*input[column];
        }
        if (!std::isfinite(output[row]) || std::abs(output[row] - expected) > 1e-6f) {
            std::fprintf(stderr,
                "IQK short F32 result mismatch at row %d: got %.9g, expected %.9g\n",
                row, output[row], expected);
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    bool available;
    if (!check_bit_identity(available)) {
        return 1;
    }
    if (!available) {
        return 0;
    }
    if (!check_short_inner_dimension()) {
        return 1;
    }
    return check_task_owners() ? 0 : 1;
}
