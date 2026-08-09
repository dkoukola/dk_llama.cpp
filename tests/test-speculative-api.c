#include "llama-speculative.h"

#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        abort(); \
    } \
} while (0)

int main(void) {
    struct llama_speculative_error *error = NULL;
    struct llama_speculative_engine_params engine = llama_speculative_engine_default_params();
    CHECK(engine.target_model == NULL);
    CHECK(engine.auxiliary_models == NULL);
    CHECK(engine.auxiliary_model_count == 0);
    CHECK(engine.checkpoint_mode == LLAMA_SPEC_CKPT_AUTO);

    struct llama_speculative_engine *created = NULL;
    CHECK(llama_speculative_engine_create(&engine, &created, &error) == LLAMA_SPECULATIVE_INVALID_ARGUMENT);
    CHECK(created == NULL);
    CHECK(error != NULL);
    CHECK(llama_speculative_error_status(error) == LLAMA_SPECULATIVE_INVALID_ARGUMENT);
    CHECK(llama_speculative_error_message(error)[0] != '\0');
    llama_speculative_error_free(error);
    error = NULL;

    struct llama_speculative_call_result call = {0};
    CHECK(llama_speculative_session_generation_synchronize(NULL, &call, &error) ==
          LLAMA_SPECULATIVE_INVALID_ARGUMENT);
    CHECK(call.status == LLAMA_SPECULATIVE_INVALID_ARGUMENT);
    CHECK(call.effect == LLAMA_SPECULATIVE_EFFECT_UNCHANGED);
    CHECK(call.boundary == LLAMA_SPECULATIVE_BOUNDARY_PRE_CALL);
    CHECK(call.consumed_count == 0);
    CHECK(call.resulting_position == 0);
    CHECK(error != NULL);
    llama_speculative_error_free(error);
    error = NULL;

    struct llama_speculative_round *round = NULL;
    struct llama_speculative_round_commit_result commit = {0};
    CHECK(llama_speculative_round_commit(&round, 0, &commit, &error) ==
          LLAMA_SPECULATIVE_INVALID_ARGUMENT);
    CHECK(commit.call.status == LLAMA_SPECULATIVE_INVALID_ARGUMENT);
    CHECK(commit.call.effect == LLAMA_SPECULATIVE_EFFECT_UNCHANGED);
    CHECK(commit.call.boundary == LLAMA_SPECULATIVE_BOUNDARY_PRE_CALL);
    CHECK(error != NULL);
    llama_speculative_error_free(error);

    struct llama_speculative_sampler_params sampler = llama_speculative_sampler_default_params();
    CHECK(sampler.seed == LLAMA_DEFAULT_SEED);
    CHECK(sampler.repeat_penalty == 1.0f);

    struct llama_batch invalid_batch = llama_batch_init(-1, 0, 1);
    CHECK(invalid_batch.token == NULL);
    CHECK(invalid_batch.embd == NULL);
    CHECK(invalid_batch.pos == NULL);
    CHECK(invalid_batch.n_seq_id == NULL);
    CHECK(invalid_batch.seq_id == NULL);
    CHECK(invalid_batch.logits == NULL);
    llama_batch_free(invalid_batch);
    return 0;
}
