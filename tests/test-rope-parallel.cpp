#include "ggml.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr size_t kContextSize = 32*1024*1024;
constexpr size_t kBenchmarkContextSize = 128*1024*1024;
constexpr int kDsv4Layers = 41;

enum class rope_direction {
    forward,
    backward,
};

struct rope_case {
    const char * name;
    ggml_type type;
    int64_t ne0;
    int64_t ne1;
    int64_t ne2;
    int64_t ne3;
    int n_dims;
    int mode;
    rope_direction direction;
    bool inplace;
    bool flipped;
    bool freq_factors;
};

struct compute_result {
    std::vector<uint8_t> output;
    int n_threads;
};

bool is_mrope(const rope_case & test) {
    return (test.mode & GGML_ROPE_TYPE_MROPE) != 0;
}

void initialize_input(ggml_tensor * input) {
    const int64_t size = ggml_nelements(input);
    if (input->type == GGML_TYPE_F32) {
        float * data = static_cast<float *>(input->data);
        for (int64_t i = 0; i < size; ++i) {
            data[i] = ((37*(i % 257) + 11) % 257 - 128)/64.0f;
        }
    } else {
        ggml_fp16_t * data = static_cast<ggml_fp16_t *>(input->data);
        for (int64_t i = 0; i < size; ++i) {
            const float value = ((37*(i % 257) + 11) % 257 - 128)/64.0f;
            data[i] = ggml_fp32_to_fp16(value);
        }
    }
}

void initialize_positions(const rope_case & test, ggml_tensor * positions) {
    int32_t * data = static_cast<int32_t *>(positions->data);
    for (int64_t i = 0; i < test.ne2; ++i) {
        data[i] = 2000 + 17*i;
        if (is_mrope(test)) {
            data[i + test.ne2] = 31 + 3*i;
            data[i + 2*test.ne2] = 47 + 5*i;
            data[i + 3*test.ne2] = 59 + 7*i;
        }
    }
}

ggml_tensor * build_rope(const rope_case & test, ggml_context * ctx,
                         ggml_tensor * input, ggml_tensor * positions,
                         ggml_tensor * freq) {
    constexpr float freq_base = 100000.0f;
    constexpr float freq_scale = 0.75f;
    constexpr float ext_factor = 0.5f;
    constexpr float attn_factor = 1.1f;
    constexpr float beta_fast = 32.0f;
    constexpr float beta_slow = 1.0f;
    constexpr int n_ctx_orig = 4096;
    int sections[GGML_MROPE_SECTIONS] = {
        test.n_dims/4,
        test.n_dims/8,
        test.n_dims/8,
        0,
    };

    ggml_tensor * output;
    if (test.direction == rope_direction::backward) {
        if (test.inplace) {
            output = ggml_rope_ext_inplace(
                    ctx, input, positions, nullptr, test.n_dims, test.mode,
                    n_ctx_orig, freq_base, freq_scale, ext_factor,
                    attn_factor, beta_fast, beta_slow);
            output->op = GGML_OP_ROPE_BACK;
        } else {
            output = ggml_rope_back(ctx, input, positions, nullptr,
                                    test.n_dims, test.mode, n_ctx_orig,
                                    freq_base, freq_scale, ext_factor,
                                    attn_factor, beta_fast, beta_slow);
        }
    } else if (is_mrope(test)) {
        output = test.inplace ?
            ggml_rope_multi_inplace(ctx, input, positions, freq,
                                    test.n_dims, sections, test.mode,
                                    n_ctx_orig, freq_base, freq_scale,
                                    ext_factor, attn_factor,
                                    beta_fast, beta_slow) :
            ggml_rope_multi(ctx, input, positions, freq,
                            test.n_dims, sections, test.mode,
                            n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor,
                            beta_fast, beta_slow);
    } else {
        output = test.inplace ?
            ggml_rope_ext_inplace(ctx, input, positions, freq,
                                  test.n_dims, test.mode, n_ctx_orig,
                                  freq_base, freq_scale, ext_factor,
                                  attn_factor, beta_fast, beta_slow) :
            ggml_rope_ext(ctx, input, positions, freq,
                          test.n_dims, test.mode, n_ctx_orig,
                          freq_base, freq_scale, ext_factor,
                          attn_factor, beta_fast, beta_slow);
    }
    if (test.flipped) {
        output->op_params[15] = 1;
    }
    return output;
}

bool compute(const rope_case & test, int n_threads, compute_result & result) {
    const ggml_init_params params = {
        /* .mem_size   = */ kContextSize,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "%s: failed to create context\n", test.name);
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_4d(
            ctx, test.type, test.ne0, test.ne1, test.ne2, test.ne3);
    const int64_t n_positions = test.ne2*(is_mrope(test) ? 4 : 1);
    ggml_tensor * positions = ggml_new_tensor_1d(
            ctx, GGML_TYPE_I32, n_positions);
    ggml_tensor * freq = test.freq_factors ?
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, test.n_dims/2) : nullptr;
    initialize_input(input);
    initialize_positions(test, positions);
    if (freq != nullptr) {
        float * data = static_cast<float *>(freq->data);
        for (int i = 0; i < test.n_dims/2; ++i) {
            data[i] = 0.9f + 0.2f*i/(test.n_dims/2 - 1);
        }
    }

    ggml_tensor * output = build_rope(test, ctx, input, positions, freq);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_cplan plan = ggml_graph_plan(graph, n_threads);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.empty() ? nullptr : work.data();
    const ggml_status status = ggml_graph_compute(graph, &plan);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: compute failed at %d threads: %s\n",
                     test.name, n_threads, ggml_status_to_string(status));
        ggml_free(ctx);
        return false;
    }

    result.output.resize(ggml_nbytes(output));
    std::memcpy(result.output.data(), output->data, result.output.size());
    result.n_threads = plan.n_threads;
    ggml_free(ctx);
    return true;
}

bool check_case(const rope_case & test) {
    constexpr std::array<int, 5> thread_counts = { 1, 2, 16, 64, 128 };
    compute_result reference;
    if (!compute(test, thread_counts[0], reference)) {
        return false;
    }
    for (size_t i = 1; i < thread_counts.size(); ++i) {
        compute_result actual;
        const int n_threads = thread_counts[i];
        if (!compute(test, n_threads, actual)) {
            return false;
        }
        if (actual.n_threads != n_threads) {
            std::fprintf(stderr, "%s: planned %d threads instead of %d\n",
                         test.name, actual.n_threads, n_threads);
            return false;
        }
        if (reference.output != actual.output) {
            std::fprintf(stderr,
                         "%s: output differs between 1 and %d threads\n",
                         test.name, n_threads);
            return false;
        }
    }
    return true;
}

uint64_t hash_outputs(const std::vector<ggml_tensor *> & outputs) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (const ggml_tensor * output : outputs) {
        const uint8_t * data = static_cast<const uint8_t *>(output->data);
        for (size_t i = 0; i < ggml_nbytes(output); ++i) {
            hash ^= data[i];
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

bool run_benchmark(int n_threads, int repetitions, int64_t n_tokens) {
    const ggml_init_params params = {
        /* .mem_size   = */ kBenchmarkContextSize,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }
    ggml_tensor * positions = ggml_new_tensor_1d(
            ctx, GGML_TYPE_I32, n_tokens);
    const rope_case q_forward = {
        "dsv4-q-forward", GGML_TYPE_F32, 512, 64, n_tokens, 1, 64,
        0, rope_direction::forward, true, true, false,
    };
    const rope_case kv_forward = {
        "dsv4-kv-forward", GGML_TYPE_F32, 512, 1, n_tokens, 1, 64,
        0, rope_direction::forward, true, true, false,
    };
    const rope_case q_backward = {
        "dsv4-q-backward", GGML_TYPE_F32, 512, 64, n_tokens, 1, 64,
        0, rope_direction::backward, true, true, false,
    };
    initialize_positions(q_forward, positions);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    std::vector<ggml_tensor *> outputs;
    outputs.reserve(3*kDsv4Layers);
    for (int layer = 0; layer < kDsv4Layers; ++layer) {
        ggml_tensor * q_input = ggml_new_tensor_4d(
                ctx, q_forward.type, q_forward.ne0, q_forward.ne1,
                q_forward.ne2, q_forward.ne3);
        ggml_tensor * kv_input = ggml_new_tensor_4d(
                ctx, kv_forward.type, kv_forward.ne0, kv_forward.ne1,
                kv_forward.ne2, kv_forward.ne3);
        initialize_input(q_input);
        initialize_input(kv_input);

        ggml_tensor * q = build_rope(
                q_forward, ctx, q_input, positions, nullptr);
        ggml_build_forward_expand(graph, q);
        outputs.push_back(q);

        ggml_tensor * kv = build_rope(
                kv_forward, ctx, kv_input, positions, nullptr);
        ggml_build_forward_expand(graph, kv);
        outputs.push_back(kv);

        q = build_rope(q_backward, ctx, q, positions, nullptr);
        ggml_build_forward_expand(graph, q);
        outputs.push_back(q);
    }
    const int n_nodes = ggml_graph_n_nodes(graph);
    if (n_nodes != 3*kDsv4Layers) {
        std::fprintf(stderr, "benchmark graph has %d nodes instead of %d\n",
                     n_nodes, 3*kDsv4Layers);
        ggml_free(ctx);
        return false;
    }

    ggml_cplan plan = ggml_graph_plan(graph, n_threads);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.empty() ? nullptr : work.data();

    for (int i = 0; i < 3; ++i) {
        if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) {
            ggml_free(ctx);
            return false;
        }
    }
    std::vector<double> samples;
    samples.reserve(repetitions);
    for (int i = 0; i < repetitions; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        const ggml_status status = ggml_graph_compute(graph, &plan);
        const auto end = std::chrono::steady_clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            ggml_free(ctx);
            return false;
        }
        samples.push_back(std::chrono::duration<double, std::micro>(
                end - begin).count());
    }
    std::sort(samples.begin(), samples.end());
    const double median = samples[samples.size()/2];
    std::printf("dsv4-rope-graph tokens=%2lld threads=%3d nodes=%3d "
                "median_us=%10.3f per_node_us=%8.3f min_us=%10.3f "
                "hash=%016llx\n",
                (long long) n_tokens, n_threads, n_nodes,
                median, median/n_nodes, samples[0],
                (unsigned long long) hash_outputs(outputs));
    ggml_free(ctx);
    return true;
}

bool run_tests() {
    constexpr std::array<rope_case, 11> cases = {{
        { "f32-standard-partial", GGML_TYPE_F32, 512, 16, 8, 2, 64,
          0, rope_direction::forward, false, false, true },
        { "f32-neox-flipped-inplace", GGML_TYPE_F32, 512, 1, 14, 1, 64,
          GGML_ROPE_TYPE_NEOX, rope_direction::forward,
          true, true, false },
        { "f32-neox-flipped-back", GGML_TYPE_F32, 512, 16, 8, 1, 64,
          GGML_ROPE_TYPE_NEOX, rope_direction::backward,
          false, true, false },
        { "f32-chatglm", GGML_TYPE_F32, 160, 3, 6, 1, 64,
          4, rope_direction::forward, false, false, false },
        { "f16-standard-partial", GGML_TYPE_F16, 512, 16, 8, 1, 64,
          0, rope_direction::forward, true, false, true },
        { "f16-neox-back", GGML_TYPE_F16, 512, 3, 6, 2, 64,
          GGML_ROPE_TYPE_NEOX, rope_direction::backward,
          false, false, false },
        { "f32-mrope", GGML_TYPE_F32, 128, 7, 8, 1, 64,
          GGML_ROPE_TYPE_MROPE, rope_direction::forward,
          false, false, false },
        { "f16-imrope-inplace", GGML_TYPE_F16, 128, 7, 8, 1, 64,
          GGML_ROPE_TYPE_IMROPE, rope_direction::forward,
          true, false, false },
        { "f32-vision", GGML_TYPE_F32, 128, 7, 8, 1, 64,
          GGML_ROPE_TYPE_VISION, rope_direction::forward,
          false, false, false },
        { "f16-vision-inplace", GGML_TYPE_F16, 128, 7, 8, 1, 64,
          GGML_ROPE_TYPE_VISION, rope_direction::forward,
          true, false, false },
        { "f32-single-row", GGML_TYPE_F32, 512, 1, 1, 1, 64,
          GGML_ROPE_TYPE_NEOX, rope_direction::forward,
          false, false, false },
    }};
    for (const rope_case & test : cases) {
        if (!check_case(test)) {
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 5 && std::strcmp(argv[1], "--bench") == 0) {
        const int n_threads = std::stoi(argv[2]);
        const int repetitions = std::stoi(argv[3]);
        const int n_tokens = std::stoi(argv[4]);
        ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
        return n_threads > 0 && repetitions > 0 && n_tokens > 0 &&
            run_benchmark(n_threads, repetitions, n_tokens) ? 0 : 1;
    }
    if (argc != 1) {
        std::fprintf(stderr,
                     "usage: %s [--bench THREADS REPETITIONS TOKENS]\n",
                     argv[0]);
        return 2;
    }
    return run_tests() ? 0 : 1;
}
