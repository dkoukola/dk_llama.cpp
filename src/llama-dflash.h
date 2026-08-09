#pragma once

#include <algorithm>
#include <cstdint>

struct llama_context;

static inline uint32_t llama_dflash_runtime_token_capacity(
        uint32_t trained_block_size,
        uint32_t n_ubatch) {
    return std::max<uint32_t>(1, std::max(trained_block_size, n_ubatch));
}

static inline int64_t llama_dflash_default_cross_context(
        uint32_t n_ctx,
        uint32_t trained_block_size,
        uint32_t n_ubatch) {
    return std::max<int64_t>(
        1,
        (int64_t) n_ctx - (int64_t) llama_dflash_runtime_token_capacity(trained_block_size, n_ubatch));
}

bool llama_prepare_dflash_graph_inputs(llama_context & lctx, uint32_t n_tokens);
