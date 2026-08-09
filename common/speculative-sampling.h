#pragma once

#include "llama.h"

#include <random>
#include <vector>

namespace llama_speculative_internal {

struct sampler_token_carry {
    llama_token token = LLAMA_TOKEN_NULL;
    std::mt19937 rng;
    bool ready = false;
};

bool sampler_token_carry_apply(
        const sampler_token_carry & carry,
        llama_token & token,
        std::mt19937 & rng) noexcept;

void sampler_token_carry_capture(
        sampler_token_carry & carry,
        llama_token token,
        const std::mt19937 & rng) noexcept;

void sampler_token_carry_commit(
        sampler_token_carry & current,
        const sampler_token_carry & outgoing,
        bool keep_outgoing) noexcept;

void sampler_token_carry_discard(sampler_token_carry & carry) noexcept;

llama_token sample_candidates(
        llama_context * context,
        llama_token_data_array * candidates,
        float temperature,
        std::mt19937 & rng,
        std::vector<float> & probabilities);

} // namespace llama_speculative_internal
