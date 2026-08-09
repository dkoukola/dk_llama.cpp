#include "speculative-sampling.h"

#include <algorithm>
#include <cmath>

namespace llama_speculative_internal {

llama_token sample_candidates(
        llama_context * context,
        llama_token_data_array * candidates,
        float temperature,
        std::mt19937 & rng,
        std::vector<float> & probabilities) {
    if (candidates == nullptr || candidates->size == 0) {
        return LLAMA_TOKEN_NULL;
    }
    // Match the common sampler's nonpositive-temperature contract. In particular,
    // llama_sample_temp() scales by division and must never receive zero here.
    if (!std::isnan(temperature) && temperature <= 0.0f) {
        return llama_sample_token_greedy(context, candidates);
    }
    if (!std::isnan(temperature)) {
        llama_sample_temp(context, candidates, temperature);
    }
    if (candidates->size == 1) {
        return candidates->data[0].id;
    }

    probabilities.resize(candidates->size);
    probabilities[0] = candidates->data[0].logit;
    float maximum = probabilities[0];
    for (size_t i = 1; i < candidates->size; ++i) {
        probabilities[i] = candidates->data[i].logit;
        maximum = std::max(maximum, probabilities[i]);
    }

    float sum = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        const float probability = std::exp(probabilities[i] - maximum);
        sum += probability;
        probabilities[i] = sum;
    }
    probabilities[candidates->size - 1] += sum;

    const auto random = rng();
    const auto point = sum * random / rng.max();
    const auto selected = std::upper_bound(probabilities.begin(), probabilities.end(), point);
    if (selected == probabilities.end()) {
        return LLAMA_TOKEN_NULL;
    }
    return candidates->data[selected - probabilities.begin()].id;
}

} // namespace llama_speculative_internal
