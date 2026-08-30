#include "ggml.h"
#include "llama.h"

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto base  = std::filesystem::temp_directory_path() /
                       ("llama-quantize-shape-fallback-" + std::to_string(nonce));
    const auto input  = base.string() + "-input.gguf";
    const auto output = base.string() + "-output.gguf";

    gguf_context * gguf = gguf_init_empty();
    require(gguf != nullptr, "failed to create input GGUF context");

    gguf_set_val_str(gguf, "general.architecture", "llama");
    gguf_set_val_u32(gguf, "llama.block_count", 1);
    gguf_set_val_u32(gguf, "llama.context_length", 8);
    gguf_set_val_u32(gguf, "llama.embedding_length", 4);
    gguf_set_val_u32(gguf, "llama.feed_forward_length", 4);
    gguf_set_val_u32(gguf, "llama.attention.head_count", 1);
    gguf_set_val_u32(gguf, "llama.attention.head_count_kv", 1);
    gguf_set_val_f32(gguf, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(gguf, "llama.rope.dimension_count", 4);

    ggml_init_params init_params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * data = ggml_init(init_params);
    require(data != nullptr, "failed to create input tensor context");

    ggml_tensor * tensor = ggml_new_tensor_2d(data, GGML_TYPE_F32, 4, 4);
    ggml_set_name(tensor, "blk.0.ple_conv1d.weight");
    for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
        static_cast<float *>(tensor->data)[i] = static_cast<float>(i);
    }
    gguf_add_tensor(gguf, tensor);
    gguf_write_to_file(gguf, input.c_str(), false);

    llama_model_quantize_params quant_params = llama_model_quantize_default_params();
    quant_params.ftype = LLAMA_FTYPE_MOSTLY_Q4_0;
    require(
            llama_model_quantize(input.c_str(), output.c_str(), &quant_params) == 0,
            "model quantization failed");

    gguf_init_params read_params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    gguf_context * result = gguf_init_from_file(output.c_str(), read_params);
    require(result != nullptr, "failed to read quantized GGUF");
    const int tensor_index = gguf_find_tensor(result, "blk.0.ple_conv1d.weight");
    require(tensor_index >= 0, "quantized GGUF is missing the test tensor");
    require(
            gguf_get_tensor_type(result, tensor_index) == GGML_TYPE_F16,
            "unrepresentable four-column tensor did not fall back to F16");

    gguf_free(result);
    ggml_free(data);
    gguf_free(gguf);
    std::filesystem::remove(input);
    std::filesystem::remove(output);
    return 0;
}
