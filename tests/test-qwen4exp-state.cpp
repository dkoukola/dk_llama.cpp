#include "llama.h"

#include "llama-context.h"
#include "llama-model.h"
#include "llama-qwen4exp.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        std::abort(); \
    } \
} while (0)

static constexpr uint32_t TEST_CONTEXT_SIZE = 128;
static constexpr uint32_t TEST_BATCH_SIZE   = 32;
static constexpr uint32_t TEST_N_SEQ        = 4;
static constexpr size_t   TEST_PROMPT_SIZE  = 16;
static constexpr size_t   TEST_NEXT_SIZE    = 6;
// n_max=4 drafts produce five target verification rows.
static constexpr int      TEST_SPEC_SIZE    = 5;

using model_ptr = std::unique_ptr<llama_model, decltype(&llama_free_model)>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
using token_map = std::map<llama_pos, llama_token>;
using qsa_cell_meta = std::array<llama_pos, 5>;

struct temporary_file {
    std::filesystem::path path;

    ~temporary_file() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};

static model_ptr load_model(const char * path) {
    llama_model_params params = llama_model_default_params();
    const char * n_gpu_layers = std::getenv("LLAMACPP_TEST_QWEN4EXP_N_GPU_LAYERS");
    params.n_gpu_layers = n_gpu_layers != nullptr ? std::atoi(n_gpu_layers) : 0;
    params.max_ctx_size = TEST_CONTEXT_SIZE;
    params.n_seq_max    = TEST_N_SEQ;
    params.n_ubatch     = TEST_BATCH_SIZE;
    params.flash_attn   = true;
    return model_ptr(llama_model_load_from_file(path, params), llama_free_model);
}

static llama_context_params make_context_params(uint32_t n_seq_max) {
    llama_context_params params = llama_context_default_params();
    const uint32_t hardware_threads = std::thread::hardware_concurrency();
    const uint32_t test_threads = std::min<uint32_t>(hardware_threads > 0 ? hardware_threads : 8, 64);
    params.n_ctx           = TEST_CONTEXT_SIZE;
    params.n_batch         = TEST_BATCH_SIZE;
    params.n_ubatch        = TEST_BATCH_SIZE;
    params.n_seq_max       = n_seq_max;
    params.n_threads       = test_threads;
    params.n_threads_batch = test_threads;
    params.flash_attn      = true;
    return params;
}

static context_ptr make_context(llama_model * model) {
    llama_context_params params = make_context_params(TEST_N_SEQ);
    return context_ptr(llama_init_from_model(model, params), llama_free);
}

static context_ptr make_mtp_context(llama_model * model, bool embeddings) {
    llama_context_params params = make_context_params(1);
    params.embeddings  = embeddings;
    params.mtp         = true;
    params.mtp_op_type = MTP_OP_NONE;
    return context_ptr(llama_init_from_model(model, params), llama_free);
}

static int32_t try_decode_tokens(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        llama_pos pos,
        llama_seq_id seq_id,
        bool logits_all) {
    CHECK(!tokens.empty());
    CHECK(tokens.size() <= TEST_BATCH_SIZE);

    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    CHECK(batch.token != nullptr);
    batch.n_tokens = (int32_t) tokens.size();
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i]       = tokens[i];
        batch.pos[i]         = pos + (llama_pos) i;
        batch.n_seq_id[i]    = 1;
        batch.seq_id[i][0]   = seq_id;
        batch.logits[i]      = logits_all || i + 1 == tokens.size();
    }

    const int32_t status = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return status;
}

static void decode_tokens(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        llama_pos pos,
        llama_seq_id seq_id,
        bool logits_all) {
    CHECK(try_decode_tokens(ctx, tokens, pos, seq_id, logits_all) == 0);
}

static void decode_image_grid(
        llama_context * ctx,
        llama_pos temporal_pos,
        int32_t nx,
        int32_t ny,
        llama_seq_id seq_id) {
    const int32_t n_tokens = nx*ny;
    const int32_t n_embd = llama_model_n_embd(llama_get_model(ctx));
    std::vector<float> embeddings((size_t) n_tokens*n_embd);
    std::vector<llama_pos> positions((size_t) 4*n_tokens);
    std::vector<int32_t> n_seq_id(n_tokens, 1);
    std::vector<llama_seq_id> seq_ids(n_tokens, seq_id);
    std::vector<llama_seq_id *> seq_id_ptrs(n_tokens);
    std::vector<int8_t> logits(n_tokens, 0);

    for (int32_t i = 0; i < n_tokens; ++i) {
        for (int32_t d = 0; d < n_embd; ++d) {
            embeddings[(size_t) i*n_embd + d] = (float) ((i + 1)*(d + 1))/100.0f;
        }
        const int32_t y = i/nx;
        const int32_t x = i%nx;
        positions[                       i] = temporal_pos;
        positions[    (size_t) n_tokens + i] = temporal_pos + y;
        positions[2 * (size_t) n_tokens + i] = temporal_pos + x;
        positions[3 * (size_t) n_tokens + i] = 0;
        seq_id_ptrs[i] = &seq_ids[i];
    }
    logits.back() = 1;

    llama_batch batch = {
        /* .n_tokens   = */ n_tokens,
        /* .token      = */ nullptr,
        /* .embd       = */ embeddings.data(),
        /* .pos        = */ positions.data(),
        /* .n_seq_id   = */ n_seq_id.data(),
        /* .seq_id     = */ seq_id_ptrs.data(),
        /* .logits     = */ logits.data(),
        /* .all_pos_0  = */ 0,
        /* .all_pos_1  = */ 0,
        /* .all_seq_id = */ 0,
    };
    CHECK(llama_decode(ctx, batch) == 0);
}

static std::vector<qsa_cell_meta> qsa_meta_for_seq(
        const llama_context * ctx,
        llama_seq_id seq_id) {
    std::vector<qsa_cell_meta> result;
    for (const llama_kv_cell & cell : ctx->kv_self.cells) {
        if (cell.has_seq_id(seq_id)) {
            result.push_back({ cell.qsa_pos, cell.pos, cell.pos_h, cell.pos_w, cell.pos_e });
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

static std::vector<float> copy_logits(llama_context * ctx, int32_t output, int32_t n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, output);
    CHECK(logits != nullptr);
    return std::vector<float>(logits, logits + n_vocab);
}

static void check_logits_equal(const std::vector<float> & expected, const std::vector<float> & actual) {
    CHECK(expected.size() == actual.size());
    if (std::memcmp(expected.data(), actual.data(), expected.size()*sizeof(float)) == 0) {
        return;
    }

    size_t first = 0;
    while (first < expected.size() &&
            std::memcmp(&expected[first], &actual[first], sizeof(float)) == 0) {
        ++first;
    }
    CHECK(first < expected.size());
    std::fprintf(stderr, "logits differ at index %zu: expected %.9g, got %.9g\n",
            first, (double) expected[first], (double) actual[first]);
    std::abort();
}

static void check_logits_close(
        const std::vector<float> & expected,
        const std::vector<float> & actual,
        float tolerance) {
    CHECK(expected.size() == actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(expected[i] - actual[i]) > tolerance) {
            std::fprintf(stderr, "logits differ at index %zu: expected %.9g, got %.9g\n",
                    i, (double) expected[i], (double) actual[i]);
            std::abort();
        }
    }
}

static std::vector<float> copy_embeddings(
        llama_context * ctx, int32_t n_outputs, uint32_t width) {
    CHECK(n_outputs >= 0);
    std::vector<float> result;
    result.reserve((size_t) n_outputs * width);
    for (int32_t i = 0; i < n_outputs; ++i) {
        const int32_t index = i - n_outputs;
        const float * data = llama_get_embeddings_ith(ctx, index);
        CHECK(data != nullptr);
        result.insert(result.end(), data, data + width);
    }
    return result;
}

static std::vector<uint8_t> save_state(llama_context * ctx) {
    const size_t size = llama_state_get_size(ctx);
    CHECK(size > 0);
    std::vector<uint8_t> result(size);
    CHECK(llama_state_get_data(ctx, result.data(), result.size()) == result.size());
    return result;
}

static std::vector<uint8_t> save_seq_state(llama_context * ctx, llama_seq_id seq_id) {
    const size_t size = llama_state_seq_get_size(ctx, seq_id, 0);
    CHECK(size > 0);
    std::vector<uint8_t> result(size);
    CHECK(llama_state_seq_get_data(ctx, result.data(), result.size(), seq_id, 0) == result.size());
    return result;
}

static token_map tokens_for_seq(const llama_context * ctx, llama_seq_id seq_id) {
    token_map result;
    for (const llama_kv_cell & cell : ctx->kv_self.cells) {
        if (!cell.has_seq_id(seq_id)) {
            continue;
        }
        CHECK(cell.pos >= 0);
        CHECK(cell.token != LLAMA_TOKEN_NULL);
        CHECK(result.emplace(cell.pos, cell.token).second);
    }
    return result;
}

static token_map expected_tokens(const std::vector<llama_token> & tokens, llama_pos pos = 0) {
    token_map result;
    for (llama_token token : tokens) {
        CHECK(result.emplace(pos++, token).second);
    }
    return result;
}

static void check_empty_cells_have_no_tokens(const llama_context * ctx) {
    for (const llama_kv_cell & cell : ctx->kv_self.cells) {
        if (cell.is_empty()) {
            CHECK(cell.pos == -1);
            CHECK(cell.token == LLAMA_TOKEN_NULL);
        }
    }
}

static void check_partial_defrag_remap(void) {
    llama_kv_cache cache;
    cache.cells.resize(4);
    cache.cells[3].pos = 3;
    cache.cells[3].seq_id.insert(0);

    const std::vector<uint32_t> ids { 0, 4, 1, 4 };
    CHECK(cache.remap_cell_id_after_defrag(ids, 0) == 0);
    CHECK(cache.remap_cell_id_after_defrag(ids, 1) == -1);
    CHECK(cache.remap_cell_id_after_defrag(ids, 2) == 1);
    CHECK(cache.remap_cell_id_after_defrag(ids, 3) == 3);
    CHECK(cache.remap_cell_id_after_defrag(ids, 4) == -1);
}

static void check_continuation(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_pos pos,
        const std::vector<llama_token> & continuation,
        const std::vector<std::vector<float>> & expected_logits,
        int32_t n_vocab,
        size_t begin,
        size_t end) {
    CHECK(end <= continuation.size());
    CHECK(end <= expected_logits.size());
    for (size_t i = begin; i < end; ++i) {
        decode_tokens(ctx, { continuation[i] }, pos + (llama_pos) i, seq_id, false);
        check_logits_equal(expected_logits[i], copy_logits(ctx, 0, n_vocab));
    }
}

static void check_mtp_whole_state_round_trip(
        llama_model * model,
        const std::vector<llama_token> & prompt,
        llama_token continuation,
        int32_t n_vocab,
        bool embeddings) {
    const uint32_t width = llama_model_mtp_feature_width(model);
    CHECK(width > 0);

    context_ptr original = make_mtp_context(model, embeddings);
    CHECK(original != nullptr);
    decode_tokens(original.get(), prompt, 0, 0, false);
    CHECK(original->n_outputs == 1);
    CHECK(original->n_outputs_embd == (int32_t) prompt.size());

    const int32_t prompt_output = (int32_t) prompt.size() - 1;
    const std::vector<float> prompt_logits = copy_logits(original.get(), prompt_output, n_vocab);
    const std::vector<float> prompt_embeddings =
            copy_embeddings(original.get(), original->n_outputs_embd, width);
    const std::vector<uint8_t> whole_state = save_state(original.get());

    decode_tokens(original.get(), { continuation }, (llama_pos) prompt.size(), 0, false);
    const std::vector<float> continuation_logits = copy_logits(original.get(), 0, n_vocab);
    const std::vector<float> continuation_embeddings =
            copy_embeddings(original.get(), original->n_outputs_embd, width);

    context_ptr restored = make_mtp_context(model, embeddings);
    CHECK(restored != nullptr);
    CHECK(restored->output_size < prompt.size());
    CHECK(llama_state_set_data(
            restored.get(), whole_state.data(), whole_state.size()) == whole_state.size());
    CHECK(restored->n_outputs == 1);
    CHECK(restored->n_outputs_embd == (int32_t) prompt.size());
    check_logits_equal(prompt_logits, copy_logits(restored.get(), prompt_output, n_vocab));
    CHECK(copy_embeddings(restored.get(), restored->n_outputs_embd, width) == prompt_embeddings);
    CHECK(save_state(restored.get()) == whole_state);

    decode_tokens(restored.get(), { continuation }, (llama_pos) prompt.size(), 0, false);
    check_logits_equal(continuation_logits, copy_logits(restored.get(), 0, n_vocab));
    CHECK(copy_embeddings(restored.get(), restored->n_outputs_embd, width) == continuation_embeddings);
    CHECK(tokens_for_seq(restored.get(), 0) == tokens_for_seq(original.get(), 0));
}

static void check_lifecycle(llama_context * ctx, const std::vector<llama_token> & prompt) {
    const llama_pos split = (llama_pos) prompt.size()/2;
    const token_map original = expected_tokens(prompt);
    CHECK(tokens_for_seq(ctx, 0) == original);

    llama_kv_cache_seq_cp(ctx, 0, 1, -1, -1);
    CHECK(tokens_for_seq(ctx, 0) == original);
    CHECK(tokens_for_seq(ctx, 1) == original);

    CHECK(llama_kv_cache_seq_rm(ctx, 0, split, -1));
    token_map prefix;
    token_map tail;
    for (const auto & item : original) {
        (item.first < split ? prefix : tail).insert(item);
    }
    CHECK(tokens_for_seq(ctx, 0) == prefix);
    CHECK(tokens_for_seq(ctx, 1) == original);

    static constexpr llama_pos shift = 7;
    llama_kv_cache_seq_add(ctx, 1, split, -1, shift);
    token_map shifted = prefix;
    for (const auto & item : tail) {
        CHECK(shifted.emplace(item.first + shift, item.second).second);
    }
    CHECK(tokens_for_seq(ctx, 1) == shifted);

    CHECK(llama_kv_cache_seq_rm(ctx, 1, 0, split));
    CHECK(tokens_for_seq(ctx, 0) == prefix);
    token_map shifted_tail;
    for (const auto & item : tail) {
        CHECK(shifted_tail.emplace(item.first + shift, item.second).second);
    }
    CHECK(tokens_for_seq(ctx, 1) == shifted_tail);

    llama_kv_cache_seq_keep(ctx, 1);
    CHECK(tokens_for_seq(ctx, 0).empty());
    CHECK(tokens_for_seq(ctx, 1) == shifted_tail);
    check_empty_cells_have_no_tokens(ctx);

    CHECK(llama_kv_cache_seq_rm(ctx, 1, -1, -1));
    CHECK(tokens_for_seq(ctx, 1).empty());
    check_empty_cells_have_no_tokens(ctx);
}

static bool qsa_graph_has_full_pool_window(const llama_context * ctx) {
    CHECK(!ctx->inp_qsa.empty());
    for (const llama_context::qsa_input & input : ctx->inp_qsa) {
        const int32_t n_blocks = (input.n_kv + input.ratio - 1)/input.ratio;
        if (input.win_blocks->ne[0] != n_blocks) {
            return false;
        }
    }
    return true;
}

static void check_qsa_window_blocks_unique(const llama_context * ctx) {
    CHECK(!ctx->inp_qsa.empty());
    for (const llama_context::qsa_input & input : ctx->inp_qsa) {
        const int32_t * blocks = (const int32_t *) input.win_blocks->data;
        std::vector<int32_t> seen;
        for (int64_t i = 0; i < input.win_blocks->ne[0]; ++i) {
            CHECK(std::find(seen.begin(), seen.end(), blocks[i]) == seen.end());
            seen.push_back(blocks[i]);
        }
    }
}

static void check_qsa_suffix_removal(
        llama_model * model,
        const std::vector<llama_token> & prompt,
        const std::vector<llama_token> & continuation,
        int32_t n_vocab) {
    auto make_seeded = [&]() {
        context_ptr ctx = make_context(model);
        CHECK(ctx != nullptr);
        decode_tokens(ctx.get(), prompt, 0, 0, false);
        CHECK(ctx->qsa_pooled_seq == 0);
        CHECK(!ctx->qsa_pooled_stale);
        CHECK(qsa_graph_has_full_pool_window(ctx.get()));
        check_qsa_window_blocks_unique(ctx.get());
        return ctx;
    };

    context_ptr no_op = make_seeded();
    CHECK(llama_kv_cache_seq_rm(no_op.get(), 0, (llama_pos) prompt.size(), -1));
    CHECK(!no_op->qsa_pooled_stale);
    no_op->qsa_pooled_stale = true;
    CHECK(llama_kv_cache_seq_rm(no_op.get(), 0, (llama_pos) prompt.size(), -1));
    CHECK(no_op->qsa_pooled_stale);
    no_op.reset();

    const llama_pos aligned_suffix = (llama_pos) prompt.size() - 2;
    context_ptr aligned = make_seeded();
    CHECK(llama_kv_cache_seq_rm(aligned.get(), 0, aligned_suffix, -1));
    CHECK(!aligned->qsa_pooled_stale);
    aligned.reset();

    const llama_pos partial_suffix = (llama_pos) prompt.size() - 3;
    context_ptr suffix = make_seeded();
    context_ptr full_repool = make_seeded();
    CHECK(llama_kv_cache_seq_rm(suffix.get(), 0, partial_suffix, -1));
    CHECK(llama_kv_cache_seq_rm(full_repool.get(), 0, partial_suffix, -1));
    CHECK(!suffix->qsa_pooled_stale);
    full_repool->qsa_pooled_stale = true;
    for (size_t i = 0; i < 2; ++i) {
        decode_tokens(suffix.get(), { continuation[i] }, partial_suffix + (llama_pos) i, 0, false);
        decode_tokens(full_repool.get(), { continuation[i] }, partial_suffix + (llama_pos) i, 0, false);
        check_logits_equal(
                copy_logits(full_repool.get(), 0, n_vocab),
                copy_logits(suffix.get(), 0, n_vocab));
        CHECK(qsa_meta_for_seq(suffix.get(), 0) == qsa_meta_for_seq(full_repool.get(), 0));
        if (i == 0) {
            CHECK(!qsa_graph_has_full_pool_window(suffix.get()));
            CHECK(qsa_graph_has_full_pool_window(full_repool.get()));
            check_qsa_window_blocks_unique(suffix.get());
            check_qsa_window_blocks_unique(full_repool.get());
        }
    }
    CHECK(!qsa_graph_has_full_pool_window(suffix.get()));
    suffix.reset();
    full_repool.reset();

    context_ptr interior = make_seeded();
    CHECK(llama_kv_cache_seq_rm(interior.get(), 0, 2, 3));
    CHECK(interior->qsa_pooled_stale);
    interior.reset();

    context_ptr prefix = make_seeded();
    CHECK(llama_kv_cache_seq_rm(prefix.get(), 0, 0, 2));
    CHECK(prefix->qsa_pooled_stale);
    prefix.reset();

    context_ptr inactive = make_seeded();
    llama_kv_cache_seq_cp(inactive.get(), 0, 1, -1, -1);
    inactive->qsa_pooled_stale = false;
    CHECK(llama_kv_cache_seq_rm(inactive.get(), 1, aligned_suffix, -1));
    CHECK(!inactive->qsa_pooled_stale);
    decode_tokens(inactive.get(), { continuation[0] }, aligned_suffix, 1, false);
    CHECK(inactive->qsa_pooled_seq == 1);
    CHECK(!inactive->qsa_pooled_stale);
    CHECK(qsa_graph_has_full_pool_window(inactive.get()));
    inactive.reset();

    context_ptr shared_suffix = make_seeded();
    context_ptr shared_full_repool = make_seeded();
    llama_kv_cache_seq_cp(shared_suffix.get(), 0, 1, -1, -1);
    llama_kv_cache_seq_cp(shared_full_repool.get(), 0, 1, -1, -1);
    shared_suffix->qsa_pooled_stale = false;
    shared_full_repool->qsa_pooled_stale = false;
    CHECK(llama_kv_cache_seq_rm(shared_suffix.get(), 0, aligned_suffix, -1));
    CHECK(llama_kv_cache_seq_rm(shared_full_repool.get(), 0, aligned_suffix, -1));
    CHECK(!shared_suffix->qsa_pooled_stale);
    shared_full_repool->qsa_pooled_stale = true;
    CHECK(tokens_for_seq(shared_suffix.get(), 1) == expected_tokens(prompt));
    decode_tokens(shared_suffix.get(), { continuation[0] }, aligned_suffix, 0, false);
    decode_tokens(shared_full_repool.get(), { continuation[0] }, aligned_suffix, 0, false);
    check_logits_equal(
            copy_logits(shared_full_repool.get(), 0, n_vocab),
            copy_logits(shared_suffix.get(), 0, n_vocab));
    CHECK(qsa_meta_for_seq(shared_suffix.get(), 0) ==
            qsa_meta_for_seq(shared_full_repool.get(), 0));
    shared_suffix.reset();
    shared_full_repool.reset();

    context_ptr shared_interior = make_seeded();
    llama_kv_cache_seq_cp(shared_interior.get(), 0, 1, -1, -1);
    shared_interior->qsa_pooled_stale = false;
    CHECK(llama_kv_cache_seq_rm(shared_interior.get(), 0, 2, 3));
    CHECK(shared_interior->qsa_pooled_stale);
}

static void check_speculative_direct_restore(
        llama_model * model,
        const std::vector<uint8_t> & seq_state,
        const std::vector<llama_token> & prompt,
        const std::vector<llama_token> & continuation,
        const std::vector<std::vector<float>> & expected_logits,
        int32_t n_vocab,
        int accepted,
        llama_seq_id seq_id) {
    CHECK(accepted >= 0 && accepted <= TEST_SPEC_SIZE);
    context_ptr ctx = make_context(model);
    CHECK(ctx != nullptr);
    CHECK(llama_state_seq_set_data(
            ctx.get(), seq_state.data(), seq_state.size(), seq_id, 0) == seq_state.size());
    CHECK(tokens_for_seq(ctx.get(), seq_id) == expected_tokens(prompt));

    const int mode = llama_spec_ckpt_init(ctx.get(), LLAMA_SPEC_CKPT_PER_STEP, TEST_SPEC_SIZE);
    CHECK(mode == LLAMA_SPEC_CKPT_PER_STEP);
    CHECK(llama_spec_ckpt_save(ctx.get(), seq_id));
    const std::vector<uint8_t> saved_state = save_seq_state(ctx.get(), seq_id);
    CHECK(llama_spec_ckpt_restore_ex(
            ctx.get(), seq_id, (llama_pos) prompt.size(), -1) ==
            LLAMA_SPEC_CKPT_RESTORE_FAILED);
    CHECK(save_seq_state(ctx.get(), seq_id) == saved_state);

    const std::vector<llama_token> speculation(
            continuation.begin(), continuation.begin() + TEST_SPEC_SIZE);
    decode_tokens(ctx.get(), speculation, (llama_pos) prompt.size(), seq_id, true);

    // Boundary failures must leave the completed capture usable by a later valid
    // direct restore.
    const std::vector<uint8_t> captured_state = save_seq_state(ctx.get(), seq_id);
    CHECK(!ctx->qsa_pooled_stale);
    CHECK(try_decode_tokens(
            ctx.get(), { continuation[TEST_SPEC_SIZE] },
            (llama_pos) prompt.size() + TEST_SPEC_SIZE, seq_id, false) != 0);
    CHECK(save_seq_state(ctx.get(), seq_id) == captured_state);
    CHECK(llama_spec_ckpt_restore_ex(
            ctx.get(), seq_id, (llama_pos) prompt.size() + 1, accepted - 1) ==
            LLAMA_SPEC_CKPT_RESTORE_FAILED);
    CHECK(save_seq_state(ctx.get(), seq_id) == captured_state);
    CHECK(llama_spec_ckpt_restore_ex(
            ctx.get(), seq_id, (llama_pos) prompt.size(), TEST_SPEC_SIZE) ==
            LLAMA_SPEC_CKPT_RESTORE_FAILED);
    CHECK(save_seq_state(ctx.get(), seq_id) == captured_state);

    const llama_spec_ckpt_restore_result restored = llama_spec_ckpt_restore_ex(
            ctx.get(), seq_id, (llama_pos) prompt.size(), accepted - 1);
    CHECK(restored == LLAMA_SPEC_CKPT_RESTORE_DIRECT);
    CHECK(!ctx->qsa_pooled_stale);

    std::vector<llama_token> committed = prompt;
    committed.insert(committed.end(), speculation.begin(), speculation.begin() + accepted);
    CHECK(tokens_for_seq(ctx.get(), seq_id) == expected_tokens(committed));

    // Decode only new tokens. If the accepted prefix had to be replayed, the
    // recurrent/PLE state here would not match the sequential reference. The
    // abort case crosses the ratio-2 boundary so its lazily stale pooled block
    // must be rewritten before becoming visible.
    llama_spec_ckpt_discard(ctx.get());
    const size_t end = accepted == 0 ? 2 : (size_t) accepted + 1;
    check_continuation(
            ctx.get(), seq_id, (llama_pos) prompt.size(), continuation, expected_logits, n_vocab,
            (size_t) accepted, end);
    committed.insert(committed.end(), continuation.begin() + accepted, continuation.begin() + end);
    CHECK(tokens_for_seq(ctx.get(), seq_id) == expected_tokens(committed));
}

static void check_single_token_checkpoint(
        llama_model * model,
        const std::vector<uint8_t> & seq_state,
        const std::vector<llama_token> & prompt,
        const std::vector<llama_token> & continuation,
        const std::vector<std::vector<float>> & expected_logits,
        int32_t n_vocab) {
    context_ptr ctx = make_context(model);
    CHECK(ctx != nullptr);
    CHECK(llama_state_seq_set_data(
            ctx.get(), seq_state.data(), seq_state.size(), 0, 0) == seq_state.size());
    CHECK(llama_spec_ckpt_init(ctx.get(), LLAMA_SPEC_CKPT_PER_STEP, 1) == LLAMA_SPEC_CKPT_PER_STEP);
    CHECK(llama_spec_ckpt_save(ctx.get(), 0));
    decode_tokens(ctx.get(), { continuation[0] }, (llama_pos) prompt.size(), 0, false);
    CHECK(llama_spec_ckpt_restore_ex(
            ctx.get(), 0, (llama_pos) prompt.size(), 0) == LLAMA_SPEC_CKPT_RESTORE_DIRECT);
    std::vector<llama_token> committed = prompt;
    committed.push_back(continuation[0]);
    CHECK(tokens_for_seq(ctx.get(), 0) == expected_tokens(committed));
    llama_spec_ckpt_discard(ctx.get());
    check_continuation(
            ctx.get(), 0, (llama_pos) prompt.size(), continuation, expected_logits, n_vocab, 1, 2);
}

static void check_begin_only_base_restore(
        llama_model * model,
        const std::vector<uint8_t> & seq_state,
        const std::vector<llama_token> & prompt,
        const std::vector<llama_token> & continuation) {
    context_ptr ctx = make_context(model);
    CHECK(ctx != nullptr);
    CHECK(llama_state_seq_set_data(
            ctx.get(), seq_state.data(), seq_state.size(), 0, 0) == seq_state.size());
    CHECK(llama_spec_ckpt_init(ctx.get(), LLAMA_SPEC_CKPT_PER_STEP, 2) == LLAMA_SPEC_CKPT_PER_STEP);
    CHECK(llama_spec_ckpt_save(ctx.get(), 0));
    const std::vector<uint8_t> base_state = save_seq_state(ctx.get(), 0);

    llama_batch batch = llama_batch_init(2, 0, 1);
    batch.n_tokens = 2;
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = continuation[(size_t) i];
        batch.pos[i] = (llama_pos) prompt.size() + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
    }
    CHECK(llama_qwen4exp_spec_ckpt_begin_capture(ctx.get(), batch));
    llama_batch_free(batch);

    // A decode can fail after begin_capture has established the batch boundary
    // but before finish_capture records step rows. Zero-prefix rollback must
    // still restore the saved base and trim any partial KV metadata.
    CHECK(llama_spec_ckpt_restore_ex(
            ctx.get(), 0, (llama_pos) prompt.size(), -1) == LLAMA_SPEC_CKPT_RESTORE_DIRECT);
    CHECK(save_seq_state(ctx.get(), 0) == base_state);
    llama_spec_ckpt_discard(ctx.get());
}

static void check_multi_ubatch_checkpoint_rejected(llama_model * model) {
    llama_context_params params = make_context_params(1);
    params.n_ubatch = 2;
    context_ptr ctx(llama_init_from_model(model, params), llama_free);
    CHECK(ctx != nullptr);
    CHECK(llama_spec_ckpt_init(
            ctx.get(), LLAMA_SPEC_CKPT_PER_STEP, TEST_SPEC_SIZE) == LLAMA_SPEC_CKPT_NONE);
}

int main(int argc, char ** argv) {
    check_partial_defrag_remap();

    const char * model_path = argc > 1 ? argv[1] : std::getenv("LLAMACPP_TEST_QWEN4EXP_MODELFILE");
    if (model_path == nullptr || model_path[0] == '\0') {
        std::fprintf(stderr,
                "test-qwen4exp-state: skipped (set LLAMACPP_TEST_QWEN4EXP_MODELFILE or pass a model path)\n");
        return 0;
    }

    llama_backend_init();

    model_ptr model = load_model(model_path);
    CHECK(model != nullptr);
    CHECK(std::strcmp(llama_model_arch_string(model.get()), "qwen4exp") == 0);
    CHECK(!llama_model_supports_ctx_shift(model.get()));
    CHECK(model->hparams.is_recurrent(0));
    CHECK(model->hparams.is_ple(0));
    CHECK(!model->hparams.is_recurrent(1));
    CHECK(model->hparams.is_ple(1));
    CHECK(model->hparams.n_embd_ple_conv(0) > 0);
    CHECK(model->hparams.n_embd_ple_conv(1) > 0);

    const int32_t n_vocab = llama_n_vocab(model.get());
    CHECK(n_vocab > 3);
    std::vector<llama_token> all_tokens(TEST_PROMPT_SIZE + TEST_NEXT_SIZE);
    for (size_t i = 0; i < all_tokens.size(); ++i) {
        all_tokens[i] = 3 + (llama_token) (i % (size_t) (n_vocab - 3));
    }
    const std::vector<llama_token> prompt(
            all_tokens.begin(), all_tokens.begin() + TEST_PROMPT_SIZE);
    const std::vector<llama_token> continuation(
            all_tokens.begin() + TEST_PROMPT_SIZE,
            all_tokens.begin() + TEST_PROMPT_SIZE + TEST_NEXT_SIZE);

    // MTP requires hidden outputs regardless of the ordinary embeddings flag.
    check_mtp_whole_state_round_trip(model.get(), prompt, continuation[0], n_vocab, false);
    check_mtp_whole_state_round_trip(model.get(), prompt, continuation[0], n_vocab, true);

    context_ptr original = make_context(model.get());
    CHECK(original != nullptr);
    CHECK(llama_supports_full_state_io(original.get()));
    CHECK(!llama_supports_ctx_shift(original.get()));
    CHECK(original->kv_self.hybrid);
    CHECK(!original->kv_self.recurrent);

    decode_tokens(original.get(), prompt, 0, 0, false);
    CHECK(tokens_for_seq(original.get(), 0) == expected_tokens(prompt));

    const std::vector<uint8_t> whole_state = save_state(original.get());
    const std::vector<uint8_t> seq_state = save_seq_state(original.get(), 0);

    const auto unique = std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
    temporary_file session_file {
        std::filesystem::temp_directory_path() / ("llama-qwen4exp-session-" + unique + ".bin")
    };
    temporary_file sequence_file {
        std::filesystem::temp_directory_path() / ("llama-qwen4exp-sequence-" + unique + ".bin")
    };
    CHECK(llama_state_save_file(
            original.get(), session_file.path.string().c_str(), prompt.data(), prompt.size()));
    CHECK(llama_state_seq_save_file(
            original.get(), sequence_file.path.string().c_str(), 0, prompt.data(), prompt.size()) > 0);

    std::vector<std::vector<float>> expected_logits;
    expected_logits.reserve(continuation.size());
    for (size_t i = 0; i < continuation.size(); ++i) {
        decode_tokens(original.get(), { continuation[i] },
                (llama_pos) prompt.size() + (llama_pos) i, 0, false);
        expected_logits.push_back(copy_logits(original.get(), 0, n_vocab));
    }
    original.reset();

    context_ptr whole_restored = make_context(model.get());
    CHECK(whole_restored != nullptr);
    CHECK(llama_state_set_data(
            whole_restored.get(), whole_state.data(), whole_state.size()) == whole_state.size());
    CHECK(whole_restored->qsa_pooled_stale);
    CHECK(tokens_for_seq(whole_restored.get(), 0) == expected_tokens(prompt));
    CHECK(save_state(whole_restored.get()) == whole_state);
    check_continuation(
            whole_restored.get(), 0, (llama_pos) prompt.size(), continuation, expected_logits, n_vocab, 0, 3);
    whole_restored.reset();

    context_ptr seq_restored = make_context(model.get());
    CHECK(seq_restored != nullptr);
    CHECK(llama_state_seq_set_data(
            seq_restored.get(), seq_state.data(), seq_state.size(), 1, 0) == seq_state.size());
    CHECK(seq_restored->qsa_pooled_stale);
    CHECK(tokens_for_seq(seq_restored.get(), 1) == expected_tokens(prompt));
    CHECK(save_seq_state(seq_restored.get(), 1) == seq_state);
    check_continuation(
            seq_restored.get(), 1, (llama_pos) prompt.size(), continuation, expected_logits, n_vocab, 0, 3);
    seq_restored.reset();

    context_ptr session_restored = make_context(model.get());
    CHECK(session_restored != nullptr);
    std::vector<llama_token> loaded_prompt(prompt.size());
    size_t loaded_count = 0;
    CHECK(llama_state_load_file(
            session_restored.get(), session_file.path.string().c_str(),
            loaded_prompt.data(), loaded_prompt.size(), &loaded_count));
    CHECK(loaded_count == prompt.size());
    CHECK(loaded_prompt == prompt);
    CHECK(tokens_for_seq(session_restored.get(), 0) == expected_tokens(prompt));
    CHECK(save_state(session_restored.get()) == whole_state);
    session_restored.reset();

    context_ptr sequence_restored = make_context(model.get());
    CHECK(sequence_restored != nullptr);
    std::fill(loaded_prompt.begin(), loaded_prompt.end(), LLAMA_TOKEN_NULL);
    loaded_count = 0;
    CHECK(llama_state_seq_load_file(
            sequence_restored.get(), sequence_file.path.string().c_str(), 1,
            loaded_prompt.data(), loaded_prompt.size(), &loaded_count) > 0);
    CHECK(loaded_count == prompt.size());
    CHECK(loaded_prompt == prompt);
    CHECK(tokens_for_seq(sequence_restored.get(), 1) == expected_tokens(prompt));
    CHECK(save_seq_state(sequence_restored.get(), 1) == seq_state);
    sequence_restored.reset();

    check_speculative_direct_restore(
            model.get(), seq_state, prompt, continuation, expected_logits, n_vocab, 0, 0);
    check_speculative_direct_restore(
            model.get(), seq_state, prompt, continuation, expected_logits, n_vocab, 1, 0);
    check_speculative_direct_restore(
            model.get(), seq_state, prompt, continuation, expected_logits, n_vocab, 2, 1);
    check_speculative_direct_restore(
            model.get(), seq_state, prompt, continuation, expected_logits, n_vocab, TEST_SPEC_SIZE, 0);
    check_single_token_checkpoint(
            model.get(), seq_state, prompt, continuation, expected_logits, n_vocab);
    check_begin_only_base_restore(model.get(), seq_state, prompt, continuation);
    check_multi_ubatch_checkpoint_rejected(model.get());
    check_qsa_suffix_removal(model.get(), prompt, continuation, n_vocab);

    context_ptr lifecycle = make_context(model.get());
    CHECK(lifecycle != nullptr);
    CHECK(llama_state_seq_set_data(
            lifecycle.get(), seq_state.data(), seq_state.size(), 0, 0) == seq_state.size());

    llama_kv_cache_seq_cp(lifecycle.get(), 0, 1, -1, -1);
    CHECK(llama_kv_cache_seq_rm(lifecycle.get(), 0, -1, -1));
    llama_kv_cache_seq_keep(lifecycle.get(), 1);
    CHECK(tokens_for_seq(lifecycle.get(), 0).empty());
    CHECK(tokens_for_seq(lifecycle.get(), 1) == expected_tokens(prompt));
    check_empty_cells_have_no_tokens(lifecycle.get());
    check_continuation(
            lifecycle.get(), 1, (llama_pos) prompt.size(), continuation, expected_logits, n_vocab, 0, 3);

    llama_kv_cache_clear(lifecycle.get());
    CHECK(llama_state_seq_set_data(
            lifecycle.get(), seq_state.data(), seq_state.size(), 0, 0) == seq_state.size());
    check_lifecycle(lifecycle.get(), prompt);
    lifecycle.reset();

    context_ptr image_original = make_context(model.get());
    CHECK(image_original != nullptr);
    decode_tokens(image_original.get(), { prompt[0], prompt[1] }, 0, 0, false);
    decode_image_grid(image_original.get(), 2, 3, 2, 0);

    std::vector<qsa_cell_meta> expected_image_meta {
        { 0, 0, 0, 0, 0 },
        { 1, 1, 1, 1, 0 },
    };
    for (int32_t y = 0; y < 2; ++y) {
        for (int32_t x = 0; x < 3; ++x) {
            expected_image_meta.push_back({
                    (llama_pos) expected_image_meta.size(), 2, 2 + y, 2 + x, 0 });
        }
    }
    CHECK(qsa_meta_for_seq(image_original.get(), 0) == expected_image_meta);

    context_ptr image_suffix = make_context(model.get());
    context_ptr image_full_repool = make_context(model.get());
    CHECK(image_suffix != nullptr && image_full_repool != nullptr);
    for (llama_context * ctx : { image_suffix.get(), image_full_repool.get() }) {
        decode_tokens(ctx, { prompt[0], prompt[1] }, 0, 0, false);
        decode_image_grid(ctx, 2, 3, 2, 0);
        CHECK(!ctx->qsa_pooled_stale);
        CHECK(llama_kv_cache_seq_rm(ctx, 0, 2, -1));
    }
    CHECK(!image_suffix->qsa_pooled_stale);
    image_full_repool->qsa_pooled_stale = true;
    decode_tokens(image_suffix.get(), { continuation[0] }, 5, 0, false);
    decode_tokens(image_full_repool.get(), { continuation[0] }, 5, 0, false);
    check_logits_equal(
            copy_logits(image_full_repool.get(), 0, n_vocab),
            copy_logits(image_suffix.get(), 0, n_vocab));
    CHECK(qsa_meta_for_seq(image_suffix.get(), 0) ==
            qsa_meta_for_seq(image_full_repool.get(), 0));
    image_suffix.reset();
    image_full_repool.reset();

    const std::vector<uint8_t> image_state = save_state(image_original.get());
    const std::vector<uint8_t> image_seq_state = save_seq_state(image_original.get(), 0);
    decode_tokens(image_original.get(), { continuation[0] }, 5, 0, false);
    const std::vector<float> expected_image_logits = copy_logits(image_original.get(), 0, n_vocab);
    const std::vector<qsa_cell_meta> expected_image_after_first =
            qsa_meta_for_seq(image_original.get(), 0);
    decode_tokens(image_original.get(), { continuation[1] }, 6, 0, false);
    const std::vector<float> expected_image_next_logits =
            copy_logits(image_original.get(), 0, n_vocab);
    image_original.reset();

    context_ptr image_whole = make_context(model.get());
    CHECK(image_whole != nullptr);
    CHECK(llama_state_set_data(
            image_whole.get(), image_state.data(), image_state.size()) == image_state.size());
    CHECK(qsa_meta_for_seq(image_whole.get(), 0) == expected_image_meta);
    decode_tokens(image_whole.get(), { continuation[0] }, 5, 0, false);
    check_logits_equal(expected_image_logits, copy_logits(image_whole.get(), 0, n_vocab));
    image_whole.reset();

    context_ptr image_sequence = make_context(model.get());
    CHECK(image_sequence != nullptr);
    CHECK(llama_state_seq_set_data(
            image_sequence.get(), image_seq_state.data(), image_seq_state.size(), 1, 0) ==
            image_seq_state.size());
    CHECK(qsa_meta_for_seq(image_sequence.get(), 1) == expected_image_meta);
    decode_tokens(image_sequence.get(), { continuation[0] }, 5, 1, false);
    check_logits_equal(expected_image_logits, copy_logits(image_sequence.get(), 0, n_vocab));
    image_sequence.reset();

    // Scalar M-RoPE time does not encode the logical next token after an image
    // grid. The checkpoint start must come from this verified batch (position 5),
    // not max(cell.pos) + 1 (which would be 3 here).
    context_ptr image_checkpoint = make_context(model.get());
    CHECK(image_checkpoint != nullptr);
    CHECK(llama_state_set_data(
            image_checkpoint.get(), image_state.data(), image_state.size()) == image_state.size());
    CHECK(llama_spec_ckpt_init(
            image_checkpoint.get(), LLAMA_SPEC_CKPT_PER_STEP, 2) == LLAMA_SPEC_CKPT_PER_STEP);
    CHECK(llama_spec_ckpt_save(image_checkpoint.get(), 0));
    decode_tokens(image_checkpoint.get(), { continuation[0], continuation[1] }, 5, 0, true);
    CHECK(llama_spec_ckpt_restore_ex(
            image_checkpoint.get(), 0, 5, 0) == LLAMA_SPEC_CKPT_RESTORE_DIRECT);
    CHECK(!image_checkpoint->qsa_pooled_stale);
    CHECK(qsa_meta_for_seq(image_checkpoint.get(), 0) == expected_image_after_first);
    llama_spec_ckpt_discard(image_checkpoint.get());
    decode_tokens(image_checkpoint.get(), { continuation[1] }, 6, 0, false);
    check_logits_equal(expected_image_next_logits,
            copy_logits(image_checkpoint.get(), 0, n_vocab));
    image_checkpoint.reset();

    context_ptr image_reference = make_context(model.get());
    context_ptr image_defrag = make_context(model.get());
    CHECK(image_reference != nullptr);
    CHECK(image_defrag != nullptr);
    CHECK(llama_state_set_data(
            image_reference.get(), image_state.data(), image_state.size()) == image_state.size());
    CHECK(llama_state_set_data(
            image_defrag.get(), image_state.data(), image_state.size()) == image_state.size());
    CHECK(llama_kv_cache_seq_rm(image_reference.get(), 0, 0, 2));
    CHECK(llama_kv_cache_seq_rm(image_defrag.get(), 0, 0, 2));
    llama_kv_cache_defrag(image_defrag.get());
    decode_tokens(image_reference.get(), { continuation[0] }, 5, 0, false);
    decode_tokens(image_defrag.get(), { continuation[0] }, 5, 0, false);
    CHECK(qsa_meta_for_seq(image_defrag.get(), 0) == qsa_meta_for_seq(image_reference.get(), 0));
    // Defrag changes the physical key order. The attention reduction can then
    // differ by one rounding bit even though the logical cache is identical.
    check_logits_close(
            copy_logits(image_reference.get(), 0, n_vocab),
            copy_logits(image_defrag.get(), 0, n_vocab), 1e-6f);
    image_defrag.reset();
    image_reference.reset();

    model.reset();
    llama_backend_free();

    std::fprintf(stderr, "test-qwen4exp-state: success\n");
    return 0;
}
