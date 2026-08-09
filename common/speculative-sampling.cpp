#include "speculative-sampling.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace llama_speculative_internal {

static_assert(std::is_nothrow_copy_assignable_v<std::mt19937>);
static_assert(std::is_nothrow_copy_assignable_v<sampler_token_carry>);

bool sampler_token_carry_apply(
        const sampler_token_carry & carry,
        llama_token & token,
        std::mt19937 & rng) noexcept {
    if (!carry.ready) {
        return false;
    }
    token = carry.token;
    rng = carry.rng;
    return true;
}

void sampler_token_carry_capture(
        sampler_token_carry & carry,
        llama_token token,
        const std::mt19937 & rng) noexcept {
    carry.token = token;
    carry.rng = rng;
    carry.ready = token != LLAMA_TOKEN_NULL;
}

void sampler_token_carry_discard(sampler_token_carry & carry) noexcept {
    carry.token = LLAMA_TOKEN_NULL;
    carry.ready = false;
}

void sampler_token_carry_commit(
        sampler_token_carry & current,
        const sampler_token_carry & outgoing,
        bool keep_outgoing) noexcept {
    if (keep_outgoing && outgoing.ready) {
        current = outgoing;
        return;
    }
    sampler_token_carry_discard(current);
}

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
