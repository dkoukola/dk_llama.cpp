#include "ggml.h"
#include "iqk/iqk_mul_mat.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int64_t kHeadSize = 256;
constexpr int64_t kQueryHeads = 24;
constexpr int64_t kKvHeads = 2;
constexpr int64_t kHeadsPerKv = kQueryHeads/kKvHeads;
constexpr int64_t kKvRows = 128;
constexpr int64_t kIndexWidth = 64;
constexpr size_t kContextSize = 32*1024*1024;

struct selection_data {
    std::vector<ggml_fp16_t> mask;
    std::vector<int32_t> indices;
    int64_t mask_rows = 0;
};

struct attention_result {
    std::vector<float> dense;
    std::vector<float> indexed;
};

float query_value(int64_t dimension, int64_t token, int64_t global_head) {
    const int64_t value =
        (13*dimension + 17*global_head + 29*token + 7) % 251 - 125;
    return value/256.0f;
}

float key_value(int64_t dimension, int64_t row, int64_t global_kv_head) {
    const int64_t value =
        (19*dimension + 23*row + 31*global_kv_head + 11) % 241 - 120;
    return value/256.0f;
}

float value_value(int64_t dimension, int64_t row, int64_t global_kv_head) {
    const int64_t value =
        (29*dimension + 7*row + 37*global_kv_head + 5) % 233 - 116;
    return value/128.0f;
}

selection_data make_selection(int64_t n_tokens) {
    selection_data result;
    result.mask_rows = GGML_PAD(n_tokens, GGML_KQ_MASK_PAD);
    result.mask.assign((size_t) kKvRows*result.mask_rows,
            ggml_fp32_to_fp16(-INFINITY));
    result.indices.assign((size_t) kIndexWidth*n_tokens, -1);

    for (int64_t token = 0; token < n_tokens; ++token) {
        const int64_t causal_limit = 72 + 8*token;
        int64_t used = 0;
        int32_t previous = -1;
        int n_masked_indices = 0;

        for (int32_t row = 0; row < kKvRows; ++row) {
            const bool selected = row < causal_limit &&
                ((row + 2*token) % 5 == 0 || row >= causal_limit - 12);
            const bool causally_masked =
                row == causal_limit + 3 || row == causal_limit + 11;
            if (!selected && !causally_masked) {
                continue;
            }
            if (used >= kIndexWidth) {
                std::fprintf(stderr,
                        "token %lld needs more than %lld selected indices\n",
                        (long long) token, (long long) kIndexWidth);
                std::abort();
            }
            if (row <= previous) {
                std::fprintf(stderr,
                        "token %lld indices are not in physical order\n",
                        (long long) token);
                std::abort();
            }
            result.indices[(size_t) token*kIndexWidth + used++] = row;
            previous = row;

            if (selected) {
                result.mask[(size_t) token*kKvRows + row] =
                    ggml_fp32_to_fp16(0.0f);
            } else {
                ++n_masked_indices;
            }
        }
        if (used == 0 || used >= kIndexWidth || n_masked_indices != 2) {
            std::fprintf(stderr,
                    "invalid selection for token %lld: used=%lld masked=%d\n",
                    (long long) token, (long long) used, n_masked_indices);
            std::abort();
        }
    }
    return result;
}

selection_data make_uniform_selection(int64_t n_tokens, int64_t selected_rows) {
    selection_data result;
    result.mask_rows = GGML_PAD(n_tokens, GGML_KQ_MASK_PAD);
    result.mask.assign((size_t) kKvRows*result.mask_rows,
            ggml_fp32_to_fp16(-INFINITY));
    result.indices.assign((size_t) kIndexWidth*n_tokens, -1);

    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int32_t row = 0; row < selected_rows; ++row) {
            result.mask[(size_t) token*kKvRows + row] =
                ggml_fp32_to_fp16(0.0f);
            result.indices[(size_t) token*kIndexWidth + row] = row;
        }
    }
    return result;
}

void initialize_query(
        ggml_tensor * query,
        int64_t global_head_offset) {
    float * data = static_cast<float *>(query->data);
    for (int64_t head = 0; head < query->ne[2]; ++head) {
        for (int64_t token = 0; token < query->ne[1]; ++token) {
            for (int64_t dimension = 0; dimension < query->ne[0];
                    ++dimension) {
                const size_t index =
                    ((size_t) head*query->ne[1] + token)*query->ne[0] +
                    dimension;
                data[index] = query_value(
                        dimension, token, global_head_offset + head);
            }
        }
    }
}

void initialize_cache(
        ggml_tensor * key,
        ggml_tensor * value,
        int64_t global_kv_head_offset) {
    std::vector<float> key_row((size_t) key->ne[0]);
    std::vector<float> value_row((size_t) value->ne[0]);

    auto store_row = [](ggml_tensor * tensor, int64_t head, int64_t row,
            const std::vector<float> & values) {
        char * dst = static_cast<char *>(tensor->data) +
            head*tensor->nb[2] + row*tensor->nb[1];
        if (tensor->type == GGML_TYPE_F16) {
            auto * dst_f16 = reinterpret_cast<ggml_fp16_t *>(dst);
            for (size_t dimension = 0; dimension < values.size(); ++dimension) {
                dst_f16[dimension] = ggml_fp32_to_fp16(values[dimension]);
            }
            return;
        }
        if (tensor->type == GGML_TYPE_Q8_0) {
            ggml_quantize_chunk(tensor->type, values.data(), dst,
                    0, 1, tensor->ne[0], nullptr, nullptr);
            return;
        }
        std::fprintf(stderr, "unsupported test cache type: %s\n",
                ggml_type_name(tensor->type));
        std::abort();
    };

    for (int64_t head = 0; head < key->ne[2]; ++head) {
        for (int64_t row = 0; row < key->ne[1]; ++row) {
            const int64_t global_head = global_kv_head_offset + head;
            for (int64_t dimension = 0; dimension < key->ne[0];
                    ++dimension) {
                key_row[(size_t) dimension] =
                    key_value(dimension, row, global_head);
            }
            for (int64_t dimension = 0; dimension < value->ne[0];
                    ++dimension) {
                value_row[(size_t) dimension] =
                    value_value(dimension, row, global_head);
            }
            store_row(key, head, row, key_row);
            store_row(value, head, row, value_row);
        }
    }
}

bool compute_attention(
        const char * label,
        int64_t n_tokens,
        int64_t n_query_heads,
        int64_t n_kv_heads,
        int64_t global_query_head_offset,
        int64_t global_kv_head_offset,
        int n_threads,
        const selection_data & selection,
        attention_result & result,
        ggml_type key_type = GGML_TYPE_F16,
        ggml_type value_type = GGML_TYPE_F16,
        int64_t key_size = kHeadSize,
        int64_t value_size = kHeadSize) {
    const ggml_init_params params = {
        /* .mem_size   = */ kContextSize,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "%s: failed to create ggml context\n", label);
        return false;
    }

    ggml_tensor * query = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F32, key_size, n_tokens, n_query_heads, 1);
    ggml_tensor * key = ggml_new_tensor_4d(
            ctx, key_type, key_size, kKvRows, n_kv_heads, 1);
    ggml_tensor * value = ggml_new_tensor_4d(
            ctx, value_type, value_size, kKvRows, n_kv_heads, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F16, kKvRows, selection.mask_rows);
    ggml_tensor * indices = ggml_new_tensor_2d(
            ctx, GGML_TYPE_I32, kIndexWidth, n_tokens);

    const size_t key_row_size = ggml_row_size(key_type, key_size);
    const size_t value_row_size = ggml_row_size(value_type, value_size);
    key->nb[1] = key_row_size*n_kv_heads;
    key->nb[2] = key_row_size;
    key->nb[3] = key_row_size*n_kv_heads*kKvRows;
    value->nb[1] = value_row_size*n_kv_heads;
    value->nb[2] = value_row_size;
    value->nb[3] = value_row_size*n_kv_heads*kKvRows;

    initialize_query(query, global_query_head_offset);
    initialize_cache(key, value, global_kv_head_offset);
    std::memcpy(mask->data, selection.mask.data(), ggml_nbytes(mask));
    std::memcpy(indices->data, selection.indices.data(), ggml_nbytes(indices));

    const float scale = 1.0f/std::sqrt((float) key_size);
    ggml_tensor * dense = ggml_flash_attn_ext(
            ctx, query, key, value, mask, scale, 0.0f, 0.0f);
    ggml_tensor * indexed = ggml_flash_attn_ext(
            ctx, query, key, value, mask, scale, 0.0f, 0.0f);
    indexed->src[5] = indices;

    const size_t dense_work = iqk_fa_work_buffer_size(dense, n_threads);
    const size_t indexed_work = iqk_fa_work_buffer_size(indexed, n_threads);
    const size_t expected = n_tokens == 1 ?
        (key_row_size*n_kv_heads + value_row_size*n_kv_heads +
            sizeof(ggml_fp16_t)*n_kv_heads)*
            kIndexWidth + 512*sizeof(float) :
        (key_row_size + value_row_size + sizeof(ggml_fp16_t))*
            kIndexWidth*n_threads;
    if (indexed_work != expected) {
        std::fprintf(stderr,
                "%s: indexed work size is %zu, expected %zu "
                "(dense uses %zu)\n",
                label, indexed_work, expected, dense_work);
        ggml_free(ctx);
        return false;
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, dense);
    ggml_build_forward_expand(graph, indexed);
    ggml_cplan plan = ggml_graph_plan(graph, n_threads);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.empty() ? nullptr : work.data();
    const ggml_status status = ggml_graph_compute(graph, &plan);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: graph compute failed: %s\n",
                label, ggml_status_to_string(status));
        ggml_free(ctx);
        return false;
    }

    result.dense.resize((size_t) ggml_nelements(dense));
    result.indexed.resize((size_t) ggml_nelements(indexed));
    std::memcpy(result.dense.data(), dense->data, ggml_nbytes(dense));
    std::memcpy(result.indexed.data(), indexed->data, ggml_nbytes(indexed));
    ggml_free(ctx);
    return true;
}

bool check_close(
        const std::vector<float> & actual,
        const std::vector<float> & expected,
        const char * label) {
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "%s: result sizes differ\n", label);
        return false;
    }

    float max_error = 0.0f;
    size_t max_index = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            std::fprintf(stderr,
                    "%s: non-finite result at %zu: %.9g vs %.9g\n",
                    label, i, actual[i], expected[i]);
            return false;
        }
        const float error = std::fabs(actual[i] - expected[i]);
        if (error > max_error) {
            max_error = error;
            max_index = i;
        }
        const float tolerance = 2.0e-5f + 2.0e-5f*std::fabs(expected[i]);
        if (error > tolerance) {
            std::fprintf(stderr,
                    "%s: mismatch at %zu: %.9g vs %.9g "
                    "(error %.9g, tolerance %.9g)\n",
                    label, i, actual[i], expected[i], error, tolerance);
            return false;
        }
    }
    std::printf("%s: max absolute error %.9g at %zu\n",
            label, max_error, max_index);
    return true;
}

bool check_shard_against_full(
        const std::vector<float> & shard,
        const std::vector<float> & full,
        int64_t n_tokens,
        int64_t global_head_offset,
        const char * label) {
    std::vector<float> expected(shard.size());
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t head = 0; head < kHeadsPerKv; ++head) {
            const size_t shard_offset =
                ((size_t) token*kHeadsPerKv + head)*kHeadSize;
            const size_t full_offset =
                ((size_t) token*kQueryHeads + global_head_offset + head)*
                kHeadSize;
            std::copy_n(full.begin() + full_offset, kHeadSize,
                    expected.begin() + shard_offset);
        }
    }
    return check_close(shard, expected, label);
}

bool check_case(int64_t n_tokens, int n_threads) {
    const selection_data selection = make_selection(n_tokens);
    attention_result full;
    attention_result shard_0;
    attention_result shard_1;
    const std::string suffix = "T=" + std::to_string(n_tokens) +
        ", threads=" + std::to_string(n_threads);
    const std::string full_label = "full Hq=24/Hkv=2, " + suffix;
    const std::string shard_0_label = "shard 0 Hq=12/Hkv=1, " + suffix;
    const std::string shard_1_label = "shard 1 Hq=12/Hkv=1, " + suffix;

    if (!compute_attention(full_label.c_str(), n_tokens,
                kQueryHeads, kKvHeads, 0, 0, n_threads, selection, full) ||
            !compute_attention(shard_0_label.c_str(), n_tokens,
                kHeadsPerKv, 1, 0, 0, n_threads, selection, shard_0) ||
            !compute_attention(shard_1_label.c_str(), n_tokens,
                kHeadsPerKv, 1, kHeadsPerKv, 1, n_threads,
                selection, shard_1)) {
        return false;
    }

    return check_close(full.indexed, full.dense,
                (full_label + " indexed vs dense").c_str()) &&
        check_close(shard_0.indexed, shard_0.dense,
                (shard_0_label + " indexed vs dense").c_str()) &&
        check_close(shard_1.indexed, shard_1.dense,
                (shard_1_label + " indexed vs dense").c_str()) &&
        check_shard_against_full(shard_0.dense, full.dense,
                n_tokens, 0, (shard_0_label + " vs full").c_str()) &&
        check_shard_against_full(shard_1.dense, full.dense,
                n_tokens, kHeadsPerKv,
                (shard_1_label + " vs full").c_str());
}

bool check_special_selection(
        int64_t n_tokens,
        int n_threads,
        int64_t selected_rows,
        const char * label) {
    const selection_data selection =
        make_uniform_selection(n_tokens, selected_rows);
    attention_result result;
    return compute_attention(label, n_tokens, kQueryHeads, kKvHeads,
                0, 0, n_threads, selection, result) &&
        check_close(result.indexed, result.dense, label);
}

bool check_asymmetric_quantized_case(int64_t n_tokens, int n_threads) {
    const selection_data selection = make_selection(n_tokens);
    attention_result result;
    const std::string label = "asymmetric Q8_0 Dk=320/Dv=256, T=" +
        std::to_string(n_tokens) + ", threads=" + std::to_string(n_threads);
    return compute_attention(label.c_str(), n_tokens,
                8, 1, 0, 0, n_threads, selection, result,
                GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 320, 256) &&
        check_close(result.indexed, result.dense, label.c_str());
}

bool check_unsupported_pair_uses_dense_planner() {
    const ggml_init_params params = {
        /* .mem_size   = */ kContextSize,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * query = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F32, 320, 1, kQueryHeads, 1);
    ggml_tensor * key = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F16, 320, kKvRows, kKvHeads, 1);
    ggml_tensor * value = ggml_new_tensor_4d(
            ctx, GGML_TYPE_Q8_0, 256, kKvRows, kKvHeads, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F16, kKvRows, GGML_KQ_MASK_PAD);
    ggml_tensor * indices = ggml_new_tensor_2d(
            ctx, GGML_TYPE_I32, kIndexWidth, 1);
    ggml_tensor * dense = ggml_flash_attn_ext(
            ctx, query, key, value, mask, 1.0f/std::sqrt(320.0f),
            0.0f, 0.0f);
    ggml_tensor * indexed = ggml_flash_attn_ext(
            ctx, query, key, value, mask, 1.0f/std::sqrt(320.0f),
            0.0f, 0.0f);
    indexed->src[5] = indices;

    constexpr int n_threads = 64;
    const size_t dense_work = iqk_fa_work_buffer_size(dense, n_threads);
    const size_t indexed_work = iqk_fa_work_buffer_size(indexed, n_threads);
    ggml_free(ctx);
    if (indexed_work != dense_work) {
        std::fprintf(stderr,
                "unsupported K/V pair selected indexed planner: %zu vs %zu\n",
                indexed_work, dense_work);
        return false;
    }
    return true;
}

bool check_mask_to_index_exact_capacity() {
    const ggml_init_params params = {
        /* .mem_size   = */ kContextSize,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    constexpr int64_t width = 32;
    ggml_tensor * mask = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F16, width, 1);
    std::fill_n(static_cast<ggml_fp16_t *>(mask->data), width,
            ggml_fp32_to_fp16(0.0f));
    ggml_tensor * indices = ggml_mask_to_index(ctx, mask, width);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, indices);
    ggml_cplan plan = ggml_graph_plan(graph, 1);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.empty() ? nullptr : work.data();
    const ggml_status status = ggml_graph_compute(graph, &plan);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "exact-capacity mask_to_index failed: %s\n",
                ggml_status_to_string(status));
        ggml_free(ctx);
        return false;
    }

    const int32_t * data = static_cast<const int32_t *>(indices->data);
    for (int32_t row = 0; row < width; ++row) {
        if (data[row] != row) {
            std::fprintf(stderr,
                    "exact-capacity mask_to_index row %d is %d\n",
                    row, data[row]);
            ggml_free(ctx);
            return false;
        }
    }
    ggml_free(ctx);
    return true;
}

} // namespace

int main() {
    constexpr std::array<int64_t, 2> token_counts = { 1, 4 };
    constexpr std::array<int, 4> thread_counts = { 1, 4, 31, 64 };
    for (int64_t n_tokens : token_counts) {
        for (int n_threads : thread_counts) {
            if (!check_case(n_tokens, n_threads)) {
                return 1;
            }
        }
    }
    if (!check_special_selection(1, 64, 0, "empty decode selection") ||
            !check_special_selection(4, 64, 0, "empty prompt selection") ||
            !check_special_selection(1, 64, 40, "64-row decode boundary") ||
            !check_special_selection(4, 64, 40, "64-row prompt boundary") ||
            !check_unsupported_pair_uses_dense_planner() ||
            !check_mask_to_index_exact_capacity()) {
        return 1;
    }
    for (int64_t n_tokens : token_counts) {
        for (int n_threads : { 1, 64 }) {
            if (!check_asymmetric_quantized_case(n_tokens, n_threads)) {
                return 1;
            }
        }
    }
    std::puts("IQK indexed flash-attention differential tests passed");
    return 0;
}
