#pragma once

#include "llama.h"

#include <random>
#include <vector>

namespace llama_speculative_internal {

llama_token sample_candidates(
        llama_context * context,
        llama_token_data_array * candidates,
        float temperature,
        std::mt19937 & rng,
        std::vector<float> & probabilities);

} // namespace llama_speculative_internal
