// Copyright (C) 2026 The llama.cpp authors
// MIT license
// SPDX-License-Identifier: MIT

#ifndef LLAMA_SPECULATIVE_H
#define LLAMA_SPECULATIVE_H

#include "llama.h"

#include <stddef.h>
#include <stdint.h>

#if defined(LLAMA_SPECULATIVE_SHARED)
#    if defined(_WIN32) && !defined(__MINGW32__)
#        if defined(LLAMA_SPECULATIVE_BUILD)
#            define LLAMA_SPECULATIVE_API __declspec(dllexport)
#        else
#            define LLAMA_SPECULATIVE_API __declspec(dllimport)
#        endif
#    else
#        define LLAMA_SPECULATIVE_API __attribute__((visibility("default")))
#    endif
#else
#    define LLAMA_SPECULATIVE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define LLAMA_SPECULATIVE_ABI_VERSION 1u
#define LLAMA_SPECULATIVE_BUILD_ID_SIZE 65u
#define LLAMA_SPECULATIVE_STATE_LAYOUT_ID_SIZE 32u

struct llama_speculative_engine;
struct llama_speculative_session;
struct llama_speculative_sampler;
struct llama_speculative_round;
struct llama_speculative_error;

typedef int32_t llama_speculative_status;

enum {
    LLAMA_SPECULATIVE_OK = 0,
    LLAMA_SPECULATIVE_BUFFER_TOO_SMALL = 1,
    LLAMA_SPECULATIVE_STATE_INCOMPATIBLE = 2,
    LLAMA_SPECULATIVE_HISTORY_INSUFFICIENT = 3,
    LLAMA_SPECULATIVE_INVALID_ARGUMENT = -1,
    LLAMA_SPECULATIVE_ABI_MISMATCH = -2,
    LLAMA_SPECULATIVE_UNSUPPORTED = -3,
    LLAMA_SPECULATIVE_OUT_OF_MEMORY = -4,
    LLAMA_SPECULATIVE_MODEL_INCOMPATIBLE = -5,
    LLAMA_SPECULATIVE_DECODE_FAILED = -6,
    LLAMA_SPECULATIVE_STATE_CORRUPT = -7,
    LLAMA_SPECULATIVE_BUSY = -8,
    LLAMA_SPECULATIVE_POISONED = -9,
    LLAMA_SPECULATIVE_ROLLBACK_FAILED = -10,
    LLAMA_SPECULATIVE_INTERNAL_ERROR = -11,
};

typedef int32_t llama_speculative_effect;

enum {
    LLAMA_SPECULATIVE_EFFECT_UNCHANGED = 0,
    LLAMA_SPECULATIVE_EFFECT_SPEC_STATE_CLEARED_TARGET_VALID = 1,
    LLAMA_SPECULATIVE_EFFECT_SESSION_DETACHED_TARGET_VALID = 2,
    LLAMA_SPECULATIVE_EFFECT_SESSION_POISONED_TARGET_UNKNOWN = 3,
};

typedef int32_t llama_speculative_boundary;

enum {
    LLAMA_SPECULATIVE_BOUNDARY_PRE_CALL = 0,
    LLAMA_SPECULATIVE_BOUNDARY_POST_CALL_COMMITTED = 1,
    LLAMA_SPECULATIVE_BOUNDARY_UNKNOWN = 2,
};

struct llama_speculative_call_result {
    llama_speculative_status status;
    llama_speculative_effect effect;
    llama_speculative_boundary boundary;
    uint64_t consumed_count;
    llama_pos resulting_position;
};

struct llama_speculative_model_binding {
    const char * name;
    struct llama_model * model;
    uint32_t context_size;
    int32_t threads;
    int32_t batch_threads;
    uint32_t flags;
};

enum {
    LLAMA_SPECULATIVE_ENGINE_REQUIRE_CANONICAL_STATE = 1u << 0,
    LLAMA_SPECULATIVE_ENGINE_REQUIRE_FIXED_SEED_TARGET_IDENTITY = 1u << 1,
};

struct llama_speculative_engine_params {
    /* Target and auxiliary models must remain alive until engine destruction. A DFlash-family
     * auxiliary whose missing IO tensors are borrowed or cloned from the target becomes bound
     * to that target; keep the target alive until the auxiliary is freed, and reload the
     * auxiliary model instead of rebinding it to another target. */
    const struct llama_model * target_model;
    const struct llama_speculative_model_binding * auxiliary_models;
    uint32_t auxiliary_model_count;
    const char * stage_expression;
    int32_t checkpoint_mode;
    uint32_t flags;
};

enum {
    // Guarantees target-stream identity, but does not promise speculative acceleration while
    // LLAMA_SPECULATIVE_SAMPLER_FIXED_SEED is active.
    LLAMA_SPECULATIVE_CAP_FIXED_SEED_TARGET_IDENTITY = UINT64_C(1) << 0,
    LLAMA_SPECULATIVE_CAP_CANONICAL_STATE = UINT64_C(1) << 1,
    LLAMA_SPECULATIVE_CAP_PORTABLE_CANONICAL_STATE = UINT64_C(1) << 2,
    LLAMA_SPECULATIVE_CAP_TARGET_ONLY_WARMUP = UINT64_C(1) << 3,
    LLAMA_SPECULATIVE_CAP_TRIM = UINT64_C(1) << 4,
    LLAMA_SPECULATIVE_CAP_CONTEXT_SHIFT = UINT64_C(1) << 5,
    LLAMA_SPECULATIVE_CAP_TOKEN_INPUT = UINT64_C(1) << 6,
    LLAMA_SPECULATIVE_CAP_EMBEDDING_INPUT = UINT64_C(1) << 7,
    LLAMA_SPECULATIVE_CAP_DERIVED_STATE = UINT64_C(1) << 8,
};

typedef int32_t llama_speculative_history_requirement;

enum {
    LLAMA_SPECULATIVE_HISTORY_NONE = 0,
    LLAMA_SPECULATIVE_HISTORY_CONTIGUOUS_SUFFIX = 1,
    LLAMA_SPECULATIVE_HISTORY_COMPLETE_PREFIX = 2,
};

struct llama_speculative_engine_capabilities {
    uint64_t flags;
    uint32_t required_target_sequence_count;
    uint32_t minimum_target_batch_tokens;
    uint32_t minimum_target_ubatch_tokens;
    llama_speculative_history_requirement history_requirement;
    uint32_t history_lookback_tokens;
    uint32_t configured_max_draft_tokens;
    uint32_t state_section_count;
    uint64_t maximum_canonical_state_bytes;
    uint64_t maximum_derived_state_bytes;
    uint32_t maximum_quality_key_bytes;
};

enum {
    LLAMA_SPECULATIVE_SAMPLER_FIXED_SEED = 1u << 0,
    LLAMA_SPECULATIVE_SAMPLER_SUPPRESS_NON_EOG_CONTROL = 1u << 1,
};

struct llama_speculative_sampler_params {
    /* LLAMA_DEFAULT_SEED keeps its legacy randomized-sentinel meaning and is invalid with
     * LLAMA_SPECULATIVE_SAMPLER_FIXED_SEED. */
    uint32_t seed;
    int32_t top_k;
    int32_t penalty_last_n;
    float top_p;
    float min_p;
    float temperature;
    float repeat_penalty;
    float frequency_penalty;
    float presence_penalty;
    uint32_t flags;
};

struct llama_speculative_session_params {
    struct llama_context * target_context;
    llama_pos sequence_start_position;
    llama_pos next_position;
};

enum {
    LLAMA_SPECULATIVE_HISTORY_COMPLETE_FROM_SEQUENCE_START = 1u << 0,
};

struct llama_speculative_token_history_view {
    const llama_token * tokens;
    uint64_t token_count;
    llama_pos sequence_start_position;
    llama_pos history_start_position;
    llama_pos next_position;
    uint32_t flags;
};

enum {
    LLAMA_SPECULATIVE_GENERATION_ALLOW_TARGET_ONLY_HISTORY_WARMUP = 1u << 0,
};

struct llama_speculative_generation_begin_params {
    struct llama_speculative_sampler * sampler;
    struct llama_speculative_token_history_view history;
    uint32_t flags;
};

typedef int32_t llama_speculative_decode_phase;

enum {
    LLAMA_SPECULATIVE_DECODE_PROMPT = 0,
    LLAMA_SPECULATIVE_DECODE_EXTERNAL_INPUT = 1,
    LLAMA_SPECULATIVE_DECODE_GENERATED_REPLAY = 2,
};

struct llama_speculative_decode_params {
    const struct llama_batch * batch;
    llama_pos expected_position;
    llama_speculative_decode_phase phase;
    uint32_t flags;
};

typedef int32_t llama_speculative_terminal;

enum {
    LLAMA_SPECULATIVE_TERMINAL_NONE = 0,
    LLAMA_SPECULATIVE_TERMINAL_EOG = 1,
    LLAMA_SPECULATIVE_TERMINAL_GENERATION_LIMIT = 2,
    LLAMA_SPECULATIVE_TERMINAL_CONTEXT_LIMIT = 3,
    LLAMA_SPECULATIVE_TERMINAL_BAD_CONTROL_TOKEN = 4,
};

enum {
    LLAMA_SPECULATIVE_ROUND_FORCE_TARGET_ONLY = 1u << 0,
};

struct llama_speculative_round_params {
    uint64_t cooperative_token_allowance;
    uint64_t generation_token_allowance;
    uint64_t context_token_allowance;
    uint32_t flags;
};

struct llama_speculative_round_view {
    const llama_token * tokens;
    uint64_t committable_count;
    llama_speculative_terminal terminal;
    llama_pos initial_position;
    llama_pos provisional_position;
    uint32_t selected_stage_index;
    uint32_t proposed_draft_tokens;
    uint32_t accepted_draft_tokens;
    uint32_t flags;
};

enum {
    LLAMA_SPECULATIVE_ROUND_ATTEMPTED = 1u << 0,
    LLAMA_SPECULATIVE_ROUND_USED_SPECULATION = 1u << 1,
    LLAMA_SPECULATIVE_ROUND_TARGET_ONLY = 1u << 2,
    LLAMA_SPECULATIVE_ROUND_RECOVERED_FALLBACK = 1u << 3,
};

struct llama_speculative_round_commit_result {
    struct llama_speculative_call_result call;
    llama_speculative_terminal terminal;
};

typedef int32_t llama_speculative_conditioning_readiness;

enum {
    LLAMA_SPECULATIVE_CONDITIONING_WARMING = 0,
    LLAMA_SPECULATIVE_CONDITIONING_READY = 1,
};

enum {
    LLAMA_SPECULATIVE_STATE_CAN_DRAFT = 1u << 0,
    LLAMA_SPECULATIVE_STATE_PARTIAL_CONDITIONING = 1u << 1,
    LLAMA_SPECULATIVE_STATE_HAS_DERIVED = 1u << 2,
};

struct llama_speculative_state_inspect_params {
    const void * state;
    uint64_t state_size;
    uint8_t * quality_key;
    uint32_t quality_key_capacity;
    uint32_t flags;
};

struct llama_speculative_state_info {
    uint32_t container_version;
    uint32_t flags;
    uint8_t state_layout_id[LLAMA_SPECULATIVE_STATE_LAYOUT_ID_SIZE];
    llama_pos sequence_start_position;
    llama_pos next_position;
    uint32_t section_count;
    uint32_t quality_key_size;
    uint64_t canonical_state_bytes;
    uint64_t derived_state_bytes;
    llama_speculative_conditioning_readiness readiness;
};

struct llama_speculative_metrics {
    uint64_t rounds_attempted;
    uint64_t speculative_rounds;
    uint64_t target_only_rounds;
    uint64_t proposed_draft_tokens;
    uint64_t accepted_draft_tokens;
    uint64_t committed_tokens;
    uint64_t target_decode_calls;
    uint64_t target_decode_tokens;
    uint64_t state_saves;
    uint64_t state_loads;
    uint64_t state_save_bytes;
    uint64_t state_load_bytes;
    uint64_t recoverable_fallbacks;
    uint64_t poisoned_failures;
};

LLAMA_SPECULATIVE_API uint32_t llama_speculative_abi_version(void);
LLAMA_SPECULATIVE_API const char * llama_speculative_build_id(void);
LLAMA_SPECULATIVE_API const char * llama_speculative_compatible_llama_build_id(void);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_validate_runtime(
        struct llama_speculative_error ** error);

LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_error_status(
        const struct llama_speculative_error * error);
LLAMA_SPECULATIVE_API const char * llama_speculative_error_message(
        const struct llama_speculative_error * error);
LLAMA_SPECULATIVE_API void llama_speculative_error_free(
        struct llama_speculative_error * error);

LLAMA_SPECULATIVE_API struct llama_speculative_engine_params llama_speculative_engine_default_params(void);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_engine_create(
        const struct llama_speculative_engine_params * params,
        struct llama_speculative_engine ** engine,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_engine_destroy(
        struct llama_speculative_engine ** engine,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_engine_get_capabilities(
        const struct llama_speculative_engine * engine,
        struct llama_speculative_engine_capabilities * capabilities,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_engine_get_state_layout_id(
        const struct llama_speculative_engine * engine,
        uint8_t state_layout_id[LLAMA_SPECULATIVE_STATE_LAYOUT_ID_SIZE],
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_engine_copy_stage_expression(
        const struct llama_speculative_engine * engine,
        char * destination,
        uint64_t capacity,
        uint64_t * required_size,
        struct llama_speculative_error ** error);

LLAMA_SPECULATIVE_API struct llama_speculative_sampler_params llama_speculative_sampler_default_params(void);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_sampler_create(
        const struct llama_speculative_engine * engine,
        const struct llama_speculative_sampler_params * params,
        struct llama_speculative_sampler ** sampler,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_sampler_reset(
        struct llama_speculative_sampler * sampler,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API void llama_speculative_sampler_destroy(
        struct llama_speculative_sampler * sampler);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_sampler_probe(
        struct llama_speculative_sampler * sampler,
        struct llama_context * target_context,
        int32_t output_index,
        llama_token * token,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_sampler_probe_commit(
        struct llama_speculative_sampler * sampler,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_sampler_probe_abort(
        struct llama_speculative_sampler * sampler,
        struct llama_speculative_error ** error);

LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_session_create(
        struct llama_speculative_engine * engine,
        const struct llama_speculative_session_params * params,
        struct llama_speculative_session ** session,
        struct llama_speculative_call_result * result,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_session_quiesce_and_detach(
        struct llama_speculative_session * session,
        struct llama_speculative_call_result * result,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API void llama_speculative_session_destroy(
        struct llama_speculative_session * session);

LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_session_generation_begin(
        struct llama_speculative_session * session,
        const struct llama_speculative_generation_begin_params * params,
        struct llama_speculative_call_result * result,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_session_generation_synchronize(
        struct llama_speculative_session * session,
        struct llama_speculative_call_result * result,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_session_generation_end(
        struct llama_speculative_session * session,
        struct llama_speculative_call_result * result,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_session_decode(
        struct llama_speculative_session * session,
        const struct llama_speculative_decode_params * params,
        struct llama_speculative_call_result * result,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_session_clear_conditioning(
        struct llama_speculative_session * session,
        llama_pos expected_position,
        struct llama_speculative_call_result * result,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_session_trim(
        struct llama_speculative_session * session,
        llama_pos expected_position,
        llama_pos new_next_position,
        struct llama_speculative_call_result * result,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_session_context_shift(
        struct llama_speculative_session * session,
        llama_pos expected_start_position,
        llama_pos expected_next_position,
        llama_pos keep_position,
        llama_pos discard_count,
        struct llama_speculative_call_result * result,
        struct llama_speculative_error ** error);

LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_round_begin(
        struct llama_speculative_session * session,
        const struct llama_speculative_round_params * params,
        struct llama_speculative_round ** round,
        struct llama_speculative_call_result * result,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_round_get_view(
        const struct llama_speculative_round * round,
        struct llama_speculative_round_view * view,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_round_commit(
        struct llama_speculative_round ** round,
        uint64_t prefix_count,
        struct llama_speculative_round_commit_result * result,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_round_abort(
        struct llama_speculative_round ** round,
        struct llama_speculative_call_result * result,
        struct llama_speculative_error ** error);

LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_session_state_size(
        const struct llama_speculative_session * session,
        uint64_t * required_size,
        struct llama_speculative_state_info * info,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_session_state_save(
        const struct llama_speculative_session * session,
        void * destination,
        uint64_t capacity,
        uint64_t * written_size,
        struct llama_speculative_state_info * info,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_engine_state_inspect(
        const struct llama_speculative_engine * engine,
        const struct llama_speculative_state_inspect_params * params,
        struct llama_speculative_state_info * info,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_session_state_load(
        struct llama_speculative_session * session,
        const void * state,
        uint64_t state_size,
        struct llama_speculative_call_result * result,
        struct llama_speculative_error ** error);

LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_session_get_metrics(
        const struct llama_speculative_session * session,
        struct llama_speculative_metrics * metrics,
        struct llama_speculative_error ** error);
LLAMA_SPECULATIVE_API llama_speculative_status llama_speculative_session_reset_metrics(
        struct llama_speculative_session * session,
        struct llama_speculative_error ** error);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_SPECULATIVE_H
