#include "ggml.h"
#include "iqk/iqk_cpu_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr int64_t kOutputSize = 19;
constexpr int64_t kExperts = 5;
constexpr int64_t kTokens = 2;
constexpr int kThreads = 8;
constexpr int kProbeThread = 3;

void init_tensor(ggml_tensor & tensor, ggml_type type,
        int64_t ne0, int64_t ne1, int64_t ne2, void * data) {
    tensor = {};
    tensor.type = type;
    tensor.ne[0] = ne0;
    tensor.ne[1] = ne1;
    tensor.ne[2] = ne2;
    tensor.ne[3] = 1;
    tensor.nb[0] = sizeof(float);
    tensor.nb[1] = tensor.nb[0]*ne0;
    tensor.nb[2] = tensor.nb[1]*ne1;
    tensor.nb[3] = tensor.nb[2]*ne2;
    tensor.data = data;
}

bool same_float(float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

std::vector<float> scalar_reference(const std::vector<float> & src0,
        const std::vector<float> & src1, const std::vector<float> & scales,
        const std::vector<int32_t> & ids, bool scaled) {
    std::vector<float> result(kOutputSize*kTokens);
    for (int64_t ir = 0; ir < kTokens; ++ir) {
        for (int64_t k = 0; k < kOutputSize; ++k) {
            float s = src1[ir*kExperts];
            if (scaled) s = scales[ids[ir*kExperts]] * s;
            float sum = src0[k + kOutputSize*kExperts*ir] * s;
            for (int64_t j = 1; j < kExperts; ++j) {
                s = src1[j + kExperts*ir];
                if (scaled) s *= scales[ids[j + kExperts*ir]];
                sum += src0[k + kOutputSize*(j + kExperts*ir)] * s;
            }
            result[k + kOutputSize*ir] = sum;
        }
    }
    return result;
}

bool check_partition(ggml_tensor & dst, std::vector<float> & output,
        const std::vector<float> & expected, float sentinel,
        int probe_thread, int n_threads, const char * label) {
    std::fill(output.begin(), output.end(), sentinel);
    iqk_mul_multi_add(&dst, probe_thread, n_threads);
    const int64_t nwork = output.size();
    const int64_t npt = (nwork - 1)/n_threads + 1;
    const int64_t first = probe_thread*npt;
    const int64_t last = std::min(nwork, first + npt);
    for (int64_t i = 0; i < nwork; ++i) {
        const float want = i >= first && i < last ? expected[i] : sentinel;
        if (!same_float(output[i], want)) {
            std::fprintf(stderr, "%s partition mismatch at output %lld (%s)\n",
                    label, (long long) i, nwork < n_threads ? "sparse" : "dense");
            return false;
        }
    }
    return true;
}

bool run_case(bool scaled) {
    std::vector<float> src0_data(kOutputSize*kExperts*kTokens);
    std::vector<float> src1_data(kExperts*kTokens);
    std::vector<float> dst_data(kOutputSize*kTokens);
    std::vector<float> scales_data = { 0.5f, -1.0f, 2.0f, 0.25f, -0.5f, 4.0f, -2.0f };
    std::vector<int32_t> ids_data(kExperts*kTokens);

    for (int64_t ir = 0; ir < kTokens; ++ir) {
        for (int64_t j = 0; j < kExperts; ++j) {
            src1_data[ir*kExperts + j] = 0.125f*(3 + 5*ir - 7*j);
            ids_data[ir*kExperts + j] = (2*j + 3*ir) % scales_data.size();
            for (int64_t k = 0; k < kOutputSize; ++k) {
                src0_data[k + kOutputSize*(j + kExperts*ir)] = 0.0625f*(11 + 3*k - 5*j + 7*ir);
            }
        }
    }

    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor dst;
    ggml_tensor scales;
    ggml_tensor ids;
    init_tensor(src0, GGML_TYPE_F32, kOutputSize, kExperts, kTokens, src0_data.data());
    init_tensor(src1, GGML_TYPE_F32, 1, kExperts, kTokens, src1_data.data());
    init_tensor(dst, GGML_TYPE_F32, kOutputSize, kTokens, 1, dst_data.data());
    init_tensor(scales, GGML_TYPE_F32, scales_data.size(), 1, 1, scales_data.data());
    init_tensor(ids, GGML_TYPE_I32, kExperts, kTokens, 1, ids_data.data());
    dst.src[0] = &src0;
    dst.src[1] = &src1;
    dst.src[2] = scaled ? &scales : nullptr;
    dst.src[3] = scaled ? &ids : nullptr;

    const char * label = scaled ? "scaled" : "plain";
    const std::vector<float> reference = scalar_reference(
            src0_data, src1_data, scales_data, ids_data, scaled);
    iqk_mul_multi_add(&dst, 0, 1);
    for (size_t i = 0; i < dst_data.size(); ++i) {
        if (!std::isfinite(dst_data[i]) || !std::isfinite(reference[i]) ||
                !same_float(dst_data[i], reference[i])) {
            std::fprintf(stderr, "%s single-thread result differs from scalar reference at output %zu\n",
                    label, i);
            return false;
        }
    }
    const std::vector<float> expected = dst_data;

    uint32_t sentinel_bits = 0x7fc01234;
    float sentinel;
    std::memcpy(&sentinel, &sentinel_bits, sizeof(sentinel));
    if (!check_partition(dst, dst_data, expected, sentinel,
                kProbeThread, kThreads, label) ||
            !check_partition(dst, dst_data, expected, sentinel,
                (int) dst_data.size() - 1, 64, label) ||
            !check_partition(dst, dst_data, expected, sentinel,
                63, 64, label)) {
        return false;
    }

    std::fill(dst_data.begin(), dst_data.end(), sentinel);
    std::vector<std::thread> workers;
    for (int ith = 0; ith < kThreads; ++ith) {
        workers.emplace_back([&dst, ith]() { iqk_mul_multi_add(&dst, ith, kThreads); });
    }
    for (auto & worker : workers) {
        worker.join();
    }
    if (std::memcmp(dst_data.data(), expected.data(), dst_data.size()*sizeof(float)) != 0) {
        std::fprintf(stderr, "%s parallel result differs from scalar reference\n", label);
        return false;
    }

    return true;
}

bool run_zero_expert_case(bool scaled) {
    std::vector<float> src0_data;
    std::vector<float> src1_data;
    std::vector<float> dst_data(kOutputSize*kTokens);
    std::vector<float> scales_data = { 1.0f };
    std::vector<int32_t> ids_data;

    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor dst;
    ggml_tensor scales;
    ggml_tensor ids;
    init_tensor(src0, GGML_TYPE_F32, kOutputSize, 0, kTokens, src0_data.data());
    init_tensor(src1, GGML_TYPE_F32, 1, 0, kTokens, src1_data.data());
    init_tensor(dst, GGML_TYPE_F32, kOutputSize, kTokens, 1, dst_data.data());
    init_tensor(scales, GGML_TYPE_F32, scales_data.size(), 1, 1, scales_data.data());
    init_tensor(ids, GGML_TYPE_I32, 0, kTokens, 1, ids_data.data());
    dst.src[0] = &src0;
    dst.src[1] = &src1;
    dst.src[2] = scaled ? &scales : nullptr;
    dst.src[3] = scaled ? &ids : nullptr;

    const char * label = scaled ? "scaled zero-expert" : "plain zero-expert";
    const std::vector<float> expected(dst_data.size(), 0.0f);
    uint32_t sentinel_bits = 0x7fc01234;
    float sentinel;
    std::memcpy(&sentinel, &sentinel_bits, sizeof(sentinel));
    if (!check_partition(dst, dst_data, expected, sentinel,
                kProbeThread, kThreads, label) ||
            !check_partition(dst, dst_data, expected, sentinel,
                (int) dst_data.size() - 1, 64, label) ||
            !check_partition(dst, dst_data, expected, sentinel,
                63, 64, label)) {
        return false;
    }

    std::fill(dst_data.begin(), dst_data.end(), sentinel);
    std::vector<std::thread> workers;
    for (int ith = 0; ith < kThreads; ++ith) {
        workers.emplace_back([&dst, ith]() { iqk_mul_multi_add(&dst, ith, kThreads); });
    }
    for (auto & worker : workers) {
        worker.join();
    }
    if (std::memcmp(dst_data.data(), expected.data(), dst_data.size()*sizeof(float)) != 0) {
        std::fprintf(stderr, "%s parallel result is not all-zero\n", label);
        return false;
    }

    return true;
}

} // namespace

int main() {
    return run_case(false) && run_case(true) &&
        run_zero_expert_case(false) && run_zero_expert_case(true) ? 0 : 1;
}
