#include "speculative-sampling.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        std::abort(); \
    } \
} while (0)

static llama_token sample(float temperature, std::mt19937 & rng) {
    llama_token_data data[] = {
        { 11, -4.0f, 0.0f },
        { 22, -1.0f, 0.0f },
        { 33, -3.0f, 0.0f },
    };
    llama_token_data_array candidates = {
        data,
        sizeof(data) / sizeof(data[0]),
        -1,
        false,
    };
    std::vector<float> probabilities;
    const llama_token token = llama_speculative_internal::sample_candidates(
        nullptr, &candidates, temperature, rng, probabilities);
    for (const llama_token_data & candidate : data) {
        CHECK(std::isfinite(candidate.logit));
    }
    return token;
}

static void check_greedy(float temperature) {
    std::mt19937 rng(1234);
    const std::mt19937 initial = rng;
    CHECK(sample(temperature, rng) == 22);
    CHECK(rng == initial);
}

static void check_positive_temperature() {
    llama_token_data data[] = {
        { 11, -4.0f, 0.0f },
        { 22, -1.0f, 0.0f },
        { 33, -3.0f, 0.0f },
    };
    llama_token_data_array candidates = {
        data,
        sizeof(data) / sizeof(data[0]),
        -1,
        false,
    };
    std::mt19937 rng(1234);
    const std::mt19937 initial = rng;
    std::vector<float> probabilities;
    CHECK(llama_speculative_internal::sample_candidates(
              nullptr, &candidates, 2.0f, rng, probabilities) == 22);
    CHECK(data[0].logit == -2.0f);
    CHECK(data[1].logit == -0.5f);
    CHECK(data[2].logit == -1.5f);
    CHECK(rng != initial);
}

static void check_single_candidate() {
    llama_token_data data = { 44, 2.0f, 0.0f };
    llama_token_data_array candidates = { &data, 1, -1, false };
    std::mt19937 rng(1234);
    const std::mt19937 initial = rng;
    std::vector<float> probabilities;
    CHECK(llama_speculative_internal::sample_candidates(
              nullptr, &candidates, 2.0f, rng, probabilities) == 44);
    CHECK(data.logit == 1.0f);
    CHECK(rng == initial);
}

int main() {
    check_greedy(0.0f);
    check_greedy(-0.0f);
    check_greedy(-1.0f);
    check_positive_temperature();
    check_single_candidate();

    std::mt19937 stochastic_rng(1234);
    const std::mt19937 stochastic_initial = stochastic_rng;
    CHECK(sample(std::numeric_limits<float>::quiet_NaN(), stochastic_rng) != LLAMA_TOKEN_NULL);
    CHECK(stochastic_rng != stochastic_initial);

    std::mt19937 empty_rng(1234);
    const std::mt19937 empty_initial = empty_rng;
    llama_token_data_array empty = { nullptr, 0, -1, false };
    std::vector<float> probabilities;
    CHECK(llama_speculative_internal::sample_candidates(
              nullptr, &empty, 0.0f, empty_rng, probabilities) == LLAMA_TOKEN_NULL);
    CHECK(empty_rng == empty_initial);
    return 0;
}
