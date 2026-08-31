#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

namespace {

constexpr int kOperationsA = 256;
constexpr int kOperationsB = 160;
constexpr int kRepetitions = 4;
constexpr int64_t kElements = 8192;

struct graph_fixture {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * result = nullptr;
    ggml_cplan plan = {};
    std::vector<uint8_t> work;
    std::vector<float> expected;

    ~graph_fixture() {
        ggml_free(ctx);
    }
};

void delayed_copy(
        ggml_tensor * dst,
        const ggml_tensor * src,
        int ith,
        int nth,
        void * userdata) {
    (void) userdata;
    if (ith == nth - 1) {
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
    const int64_t n = ggml_nelements(dst);
    const int64_t begin = n * ith / nth;
    const int64_t end = n * (ith + 1) / nth;
    const float * src_data = static_cast<const float *>(src->data);
    float * dst_data = static_cast<float *>(dst->data);
    std::copy(src_data + begin, src_data + end, dst_data + begin);
}

bool prepare_graph(
        graph_fixture & fixture,
        int n_threads,
        int n_operations,
        float initial,
        float increment) {
    const ggml_init_params params = {
        /* .mem_size   = */ 32 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    fixture.ctx = ggml_init(params);
    if (fixture.ctx == nullptr) {
        return false;
    }

    ggml_tensor * value = ggml_new_tensor_1d(fixture.ctx, GGML_TYPE_F32, kElements);
    ggml_tensor * addend = ggml_new_tensor_1d(fixture.ctx, GGML_TYPE_F32, kElements);
    float * input = static_cast<float *>(value->data);
    for (int64_t i = 0; i < kElements; ++i) {
        input[i] = initial + (float) (i % 17) / 32.0f;
    }
    ggml_set_f32(addend, increment);
    for (int i = 0; i < n_operations; ++i) {
        value = ggml_add(fixture.ctx, value, addend);
        if ((i + 1) % 32 == 0) {
            value = ggml_map_custom1(
                    fixture.ctx, value, delayed_copy, GGML_N_TASKS_MAX, nullptr);
            value = ggml_rms_norm(fixture.ctx, value, 1e-5f);
        }
    }

    const size_t graph_capacity = static_cast<size_t>(n_operations + 2 * (n_operations / 32) + 8);
    fixture.graph = ggml_new_graph_custom(fixture.ctx, graph_capacity, false);
    ggml_build_forward_expand(fixture.graph, value);
    fixture.result = value;

    ggml_cplan reference_plan = ggml_graph_plan(fixture.graph, 1);
    std::vector<uint8_t> reference_work(reference_plan.work_size);
    reference_plan.work_data = reference_work.empty() ? nullptr : reference_work.data();
    if (ggml_graph_compute(fixture.graph, &reference_plan) != GGML_STATUS_SUCCESS) {
        return false;
    }
    const float * reference = static_cast<const float *>(fixture.result->data);
    fixture.expected.assign(reference, reference + kElements);

    fixture.plan = ggml_graph_plan(fixture.graph, n_threads);
    fixture.work.resize(fixture.plan.work_size);
    fixture.plan.work_data = fixture.work.empty() ? nullptr : fixture.work.data();
    return true;
}

bool run_graph(graph_fixture & fixture, std::atomic<int> & ready, std::atomic<bool> & start) {
    ready.fetch_add(1, std::memory_order_release);
    while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    for (int i = 0; i < kRepetitions; ++i) {
        if (ggml_graph_compute(fixture.graph, &fixture.plan) != GGML_STATUS_SUCCESS) {
            return false;
        }
    }
    return true;
}

bool check_result(const graph_fixture & fixture) {
    const float * data = static_cast<const float *>(fixture.result->data);
    for (int64_t i = 0; i < kElements; ++i) {
        if (std::fabs(data[i] - fixture.expected[(size_t) i]) > 1e-5f) {
            std::fprintf(stderr, "result mismatch at element %lld: %.9g != %.9g\n",
                    (long long) i, data[i], fixture.expected[(size_t) i]);
            return false;
        }
    }
    return true;
}

bool check_concurrent_pair(int threads_a, int threads_b) {
    graph_fixture graph_a;
    graph_fixture graph_b;
    if (!prepare_graph(graph_a, threads_a, kOperationsA, 1.0f, 0.25f) ||
            !prepare_graph(graph_b, threads_b, kOperationsB, -2.0f, 0.5f)) {
        std::fputs("failed to prepare concurrent NUMA barrier graphs\n", stderr);
        return false;
    }

    std::atomic<int> ready { 0 };
    std::atomic<bool> start { false };
    bool ok_a = false;
    bool ok_b = false;
    std::thread worker_a([&]() { ok_a = run_graph(graph_a, ready, start); });
    std::thread worker_b([&]() { ok_b = run_graph(graph_b, ready, start); });
    while (ready.load(std::memory_order_acquire) != 2) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    worker_a.join();
    worker_b.join();

    return ok_a && ok_b && check_result(graph_a) && check_result(graph_b);
}

} // namespace

int main() {
    ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
    if (!ggml_numa_mirror_active()) {
        std::puts("NUMA barrier concurrency test skipped: mirror needs at least two nodes");
        return 77;
    }

    const unsigned int available = std::max(4u, std::thread::hardware_concurrency());
    const int threads_a = (int) std::max(2u, std::min(32u, available / 3));
    const int threads_b = 2;
    const int equal_threads = threads_a;
    if (!check_concurrent_pair(threads_a, threads_b) ||
            !check_concurrent_pair(equal_threads, equal_threads)) {
        return 1;
    }
    std::puts("concurrent NUMA barrier test passed");
    return 0;
}
