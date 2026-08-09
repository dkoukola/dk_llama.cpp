#include "speculative-sampling.h"

#include <cstdio>
#include <cstdlib>
#include <random>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        std::abort(); \
    } \
} while (0)

using llama_speculative_internal::sampler_token_carry;

static llama_token sample_token(std::mt19937 & rng, int & sample_calls) {
    const llama_token token = 100 + sample_calls;
    ++sample_calls;
    (void) rng();
    return token;
}

static void check_full_commit_carries_without_resampling() {
    sampler_token_carry current;
    sampler_token_carry outgoing;
    std::mt19937 canonical_rng(1234);
    std::mt19937 provisional_rng = canonical_rng;
    llama_token first = LLAMA_TOKEN_NULL;
    int sample_calls = 0;

    CHECK(!llama_speculative_internal::sampler_token_carry_apply(
        current, first, provisional_rng));
    first = sample_token(provisional_rng, sample_calls);
    CHECK(first == 100);
    const std::mt19937 committed_first_rng = provisional_rng;

    const llama_token outgoing_token = sample_token(provisional_rng, sample_calls);
    llama_speculative_internal::sampler_token_carry_capture(
        outgoing, outgoing_token, provisional_rng);
    llama_speculative_internal::sampler_token_carry_commit(current, outgoing, true);
    CHECK(current.ready);
    CHECK(sample_calls == 2);

    provisional_rng = committed_first_rng;
    first = LLAMA_TOKEN_NULL;
    CHECK(llama_speculative_internal::sampler_token_carry_apply(
        current, first, provisional_rng));
    CHECK(first == outgoing_token);
    CHECK(provisional_rng == outgoing.rng);
    CHECK(sample_calls == 2);

    sampler_token_carry next_outgoing;
    const std::mt19937 committed_second_rng = provisional_rng;
    const llama_token next_outgoing_token = sample_token(provisional_rng, sample_calls);
    llama_speculative_internal::sampler_token_carry_capture(
        next_outgoing, next_outgoing_token, provisional_rng);
    llama_speculative_internal::sampler_token_carry_commit(current, next_outgoing, true);
    CHECK(current.ready);
    CHECK(current.token == next_outgoing_token);
    CHECK(committed_second_rng == outgoing.rng);
    CHECK(sample_calls == 3);
}

static void check_abort_preserves_incoming_carry() {
    sampler_token_carry current;
    std::mt19937 carry_rng(5678);
    (void) carry_rng();
    llama_speculative_internal::sampler_token_carry_capture(current, 77, carry_rng);

    std::mt19937 provisional_rng(42);
    llama_token first = LLAMA_TOKEN_NULL;
    CHECK(llama_speculative_internal::sampler_token_carry_apply(
        current, first, provisional_rng));
    CHECK(first == 77);
    CHECK(provisional_rng == carry_rng);

    // An aborted round never commits its round-local outgoing carry.
    CHECK(current.ready);
    CHECK(current.token == 77);
    provisional_rng.seed(99);
    first = LLAMA_TOKEN_NULL;
    CHECK(llama_speculative_internal::sampler_token_carry_apply(
        current, first, provisional_rng));
    CHECK(first == 77);
    CHECK(provisional_rng == carry_rng);
}

static void check_partial_commit_and_quiesce_discard_carry() {
    sampler_token_carry current;
    sampler_token_carry outgoing;
    std::mt19937 rng(9012);
    llama_speculative_internal::sampler_token_carry_capture(current, 55, rng);
    (void) rng();
    llama_speculative_internal::sampler_token_carry_capture(outgoing, 66, rng);

    llama_speculative_internal::sampler_token_carry_commit(current, outgoing, false);
    CHECK(!current.ready);
    CHECK(current.token == LLAMA_TOKEN_NULL);

    llama_speculative_internal::sampler_token_carry_capture(current, 77, rng);
    sampler_token_carry empty;
    llama_speculative_internal::sampler_token_carry_commit(current, empty, true);
    CHECK(!current.ready);

    llama_speculative_internal::sampler_token_carry_capture(current, 88, rng);
    llama_speculative_internal::sampler_token_carry_discard(current);
    CHECK(!current.ready);
    CHECK(current.token == LLAMA_TOKEN_NULL);
}

int main() {
    check_full_commit_carries_without_resampling();
    check_abort_preserves_incoming_carry();
    check_partial_commit_and_quiesce_discard_carry();
    return 0;
}
