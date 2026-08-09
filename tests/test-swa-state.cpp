#include "llama.h"

#include "llama-context.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        std::abort(); \
    } \
} while (0)

static constexpr uint32_t TEST_CONTEXT_SIZE = 2048;
static constexpr uint32_t TEST_BATCH_SIZE   = 64;
static constexpr uint32_t TEST_SAVE_POS     = 1536;
static constexpr uint32_t TEST_FINAL_POS    = 1984;
static constexpr uint32_t TEST_SWAC_MAGIC   = 0x53574143u;

using model_ptr = std::unique_ptr<llama_model, decltype(&llama_free_model)>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;

static model_ptr load_model(const char * path, bool swa_compress, uint32_t n_ubatch) {
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.max_ctx_size = TEST_CONTEXT_SIZE;
    params.n_seq_max    = 1;
    params.n_ubatch     = n_ubatch;
    params.flash_attn   = true;
    params.swa_compress = swa_compress;
    return model_ptr(llama_model_load_from_file(path, params), llama_free_model);
}

static context_ptr make_context(llama_model * model, bool swa_compress, uint32_t n_ubatch) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx           = TEST_CONTEXT_SIZE;
    params.n_batch         = n_ubatch;
    params.n_ubatch        = n_ubatch;
    params.n_seq_max       = 1;
    params.n_threads       = 8;
    params.n_threads_batch = 8;
    params.flash_attn      = true;
    params.swa_compress    = swa_compress;
    return context_ptr(llama_init_from_model(model, params), llama_free);
}

static void decode_tokens(
        llama_context * ctx,
           llama_token token,
              uint32_t pos,
              uint32_t count,
                  bool sparse_outputs) {
    CHECK(count > 0 && count <= TEST_BATCH_SIZE);
    llama_batch batch = llama_batch_init((int32_t) count, 0, 1);
    batch.n_tokens = (int32_t) count;
    for (uint32_t i = 0; i < count; ++i) {
        batch.token[i]       = token;
        batch.pos[i]         = (llama_pos) (pos + i);
        batch.n_seq_id[i]    = 1;
        batch.seq_id[i][0]   = 0;
        batch.logits[i]      = sparse_outputs ? (i == 0 || i == 7 || i + 1 == count) : (i + 1 == count);
    }
    const int32_t status = llama_decode(ctx, batch);
    llama_batch_free(batch);
    CHECK(status == 0);
}

static void decode_until(
        llama_context * ctx,
           llama_token token,
             uint32_t & pos,
              uint32_t end,
                  bool sparse_final_outputs = false) {
    while (pos < end) {
        const uint32_t count = std::min(TEST_BATCH_SIZE, end - pos);
        decode_tokens(ctx, token, pos, count, sparse_final_outputs && pos + count == end);
        pos += count;
    }
}

static std::vector<float> copy_logits(llama_context * ctx, int32_t output, int32_t n_vocab) {
    float * logits = llama_get_logits_ith(ctx, output);
    CHECK(logits != nullptr);
    return std::vector<float>(logits, logits + n_vocab);
}

static void check_logits_equal(
        llama_context * a,
        llama_context * b,
               int32_t output,
               int32_t n_vocab) {
    const std::vector<float> logits_a = copy_logits(a, output, n_vocab);
    const std::vector<float> logits_b = copy_logits(b, output, n_vocab);
    CHECK(logits_a.size() == logits_b.size());
    CHECK(std::memcmp(logits_a.data(), logits_b.data(), logits_a.size()*sizeof(float)) == 0);
}

static void check_logits_equal(const std::vector<float> & a, const std::vector<float> & b) {
    CHECK(a.size() == b.size());
    CHECK(std::memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0);
}

static std::vector<uint8_t> save_state(llama_context * ctx) {
    const size_t size = llama_state_get_size(ctx);
    CHECK(size > 0);
    std::vector<uint8_t> state(size);
    CHECK(llama_state_get_data(ctx, state.data(), state.size()) == state.size());
    return state;
}

static llama_token sample_token(llama_context * ctx, int32_t output, int32_t n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, output);
    CHECK(logits != nullptr);
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token = 0; token < n_vocab; ++token) {
        candidates.push_back({ token, logits[token], 0.0f });
    }
    llama_token_data_array candidate_array = { candidates.data(), candidates.size(), false };
    return llama_sample_token(ctx, &candidate_array);
}

static size_t find_swac_offset(const std::vector<uint8_t> & state, const llama_context * ctx) {
    size_t offset = SIZE_MAX;
    const llama_kv_cache & cache = ctx->kv_self;
    const size_t header_size = 5*sizeof(uint32_t) + sizeof(llama_pos);
    for (size_t i = 0; i + header_size <= state.size(); ++i) {
        uint32_t value;
        std::memcpy(&value, state.data() + i, sizeof(value));
        uint32_t size_swa;
        uint32_t live_swa;
        llama_pos pos_base_swa;
        uint32_t head_swa;
        uint32_t cell_count;
        std::memcpy(&size_swa,     state.data() + i + 1*sizeof(uint32_t), sizeof(size_swa));
        std::memcpy(&live_swa,     state.data() + i + 2*sizeof(uint32_t), sizeof(live_swa));
        std::memcpy(&pos_base_swa, state.data() + i + 3*sizeof(uint32_t), sizeof(pos_base_swa));
        std::memcpy(&head_swa,     state.data() + i + 3*sizeof(uint32_t) + sizeof(llama_pos), sizeof(head_swa));
        std::memcpy(&cell_count,   state.data() + i + 4*sizeof(uint32_t) + sizeof(llama_pos), sizeof(cell_count));
        if (value == TEST_SWAC_MAGIC &&
            size_swa == cache.size_swa &&
            head_swa >= cache.sink_rows &&
            head_swa - cache.sink_rows == live_swa &&
            pos_base_swa >= 0 &&
            (uint64_t) pos_base_swa + live_swa == cell_count) {
            CHECK(offset == SIZE_MAX);
            offset = i;
        }
    }
    CHECK(offset != SIZE_MAX);
    return offset;
}

static void check_compact_geometry(const llama_context * ctx, uint32_t expected_end) {
    const llama_kv_cache & cache = ctx->kv_self;
    CHECK(cache.any_compacted());
    CHECK(cache.head_swa >= cache.sink_rows);
    CHECK(cache.pos_base_swa >= 0);
    CHECK((uint64_t) cache.pos_base_swa + cache.live_swa() == expected_end);

    std::vector<bool> seen(expected_end, false);
    uint32_t count = 0;
    for (const auto & cell : cache.cells) {
        if (cell.is_empty()) {
            continue;
        }
        CHECK(cell.pos >= 0 && (uint64_t) cell.pos < expected_end);
        CHECK(!seen[cell.pos]);
        seen[cell.pos] = true;
        ++count;
    }
    CHECK(count == expected_end);
}

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : std::getenv("LLAMACPP_TEST_DEEPSEEK4_MODELFILE");
    if (model_path == nullptr || model_path[0] == '\0') {
        std::fprintf(stderr, "test-swa-state: skipped (set LLAMACPP_TEST_DEEPSEEK4_MODELFILE)\n");
        return 0;
    }
    const bool numa_mirror = std::getenv("LLAMACPP_TEST_NUMA_MIRROR") != nullptr;

    llama_backend_init();
    if (numa_mirror) {
        llama_numa_init(GGML_NUMA_STRATEGY_MIRROR);
    }

    model_ptr model = load_model(model_path, true, TEST_BATCH_SIZE);
    CHECK(model != nullptr);
    CHECK(llama_model_is_deepseek4(model.get()));

    context_ptr original = make_context(model.get(), true, TEST_BATCH_SIZE);
    context_ptr restored = make_context(model.get(), true, TEST_BATCH_SIZE);
    CHECK(original != nullptr && restored != nullptr);
    CHECK(llama_supports_full_state_io(original.get()));
    CHECK(!numa_mirror || !original->kv_self.numa_mirror_bufs.empty());

    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    llama_token repeated = llama_vocab_bos(vocab);
    if (repeated == LLAMA_TOKEN_NULL || llama_vocab_is_eog(vocab, repeated)) {
        repeated = 0;
    }

    uint32_t original_pos = 0;
    decode_until(original.get(), repeated, original_pos, TEST_SAVE_POS, true);
    check_compact_geometry(original.get(), TEST_SAVE_POS);
    CHECK(original->kv_self.pos_base_swa > 0);

    const int32_t n_vocab = llama_n_vocab(model.get());
    const std::vector<float> output_0  = copy_logits(original.get(), 0, n_vocab);
    const std::vector<float> output_7  = copy_logits(original.get(), 7, n_vocab);
    const std::vector<float> output_63 = copy_logits(original.get(), 63, n_vocab);

    const std::vector<uint8_t> state = save_state(original.get());
    CHECK(llama_state_set_data(restored.get(), state.data(), state.size()) == state.size());
    check_compact_geometry(restored.get(), TEST_SAVE_POS);
    CHECK(save_state(restored.get()) == state);
    check_logits_equal(copy_logits(restored.get(), 0, n_vocab), output_0);
    check_logits_equal(copy_logits(restored.get(), 7, n_vocab), output_7);
    check_logits_equal(copy_logits(restored.get(), 63, n_vocab), output_63);

    const llama_token next_original = sample_token(original.get(), 63, n_vocab);
    const llama_token next_restored = sample_token(restored.get(), 63, n_vocab);
    CHECK(next_original == next_restored);
    decode_tokens(original.get(), next_original, original_pos, 1, false);
    uint32_t restored_pos = original_pos;
    decode_tokens(restored.get(), next_restored, restored_pos, 1, false);
    ++original_pos;
    ++restored_pos;
    check_logits_equal(original.get(), restored.get(), 0, n_vocab);

    decode_until(original.get(), next_original, original_pos, TEST_FINAL_POS);
    decode_until(restored.get(), next_restored, restored_pos, TEST_FINAL_POS);
    CHECK(original_pos == TEST_FINAL_POS && restored_pos == TEST_FINAL_POS);
    check_compact_geometry(original.get(), TEST_FINAL_POS);
    check_compact_geometry(restored.get(), TEST_FINAL_POS);
    check_logits_equal(original.get(), restored.get(), -1, n_vocab);
    CHECK(save_state(original.get()) == save_state(restored.get()));

    std::vector<llama_token> tokens(TEST_FINAL_POS, next_original);
    std::fill(tokens.begin(), tokens.begin() + TEST_SAVE_POS, repeated);
    tokens[TEST_SAVE_POS] = next_original;
    const std::filesystem::path session_path = std::filesystem::temp_directory_path() /
            ("llama-swa-state-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
    const std::string session_path_string = session_path.string();
    CHECK(llama_state_save_file(original.get(), session_path_string.c_str(), tokens.data(), tokens.size()));

    context_ptr file_restored = make_context(model.get(), true, TEST_BATCH_SIZE);
    CHECK(file_restored != nullptr);
    std::vector<llama_token> loaded_tokens(tokens.size());
    size_t loaded_count = 0;
    CHECK(llama_state_load_file(
                file_restored.get(),
                session_path_string.c_str(),
                loaded_tokens.data(),
                loaded_tokens.size(),
                &loaded_count));
    CHECK(loaded_count == tokens.size());
    CHECK(loaded_tokens == tokens);
    CHECK(save_state(file_restored.get()) == save_state(original.get()));
    CHECK(std::filesystem::remove(session_path));

    std::vector<uint8_t> corrupt = state;
    const size_t swac = find_swac_offset(corrupt, original.get());
    const llama_pos invalid_base = -1;
    std::memcpy(corrupt.data() + swac + 3*sizeof(uint32_t), &invalid_base, sizeof(invalid_base));
    context_ptr malformed_target = make_context(model.get(), true, TEST_BATCH_SIZE);
    CHECK(malformed_target != nullptr);
    CHECK(llama_state_set_data(malformed_target.get(), corrupt.data(), corrupt.size()) == 0);

    corrupt = state;
    const uint32_t short_live = original->kv_self.window_swa - 1;
    const llama_pos short_base = TEST_SAVE_POS - short_live;
    const uint32_t short_head = original->kv_self.sink_rows + short_live;
    std::memcpy(corrupt.data() + swac + 2*sizeof(uint32_t), &short_live, sizeof(short_live));
    std::memcpy(corrupt.data() + swac + 3*sizeof(uint32_t), &short_base, sizeof(short_base));
    std::memcpy(
            corrupt.data() + swac + 3*sizeof(uint32_t) + sizeof(llama_pos),
            &short_head,
            sizeof(short_head));
    CHECK(llama_state_set_data(malformed_target.get(), corrupt.data(), corrupt.size()) == 0);

    CHECK(llama_state_set_data(malformed_target.get(), state.data(), state.size()) == state.size());
    CHECK(llama_state_set_data(malformed_target.get(), state.data(), state.size() - 1) == 0);

    model_ptr wide_model = load_model(model_path, true, 512);
    CHECK(wide_model != nullptr);
    context_ptr wide_context = make_context(wide_model.get(), true, 512);
    CHECK(wide_context != nullptr);
    CHECK(wide_context->kv_self.size_swa != original->kv_self.size_swa);
    CHECK(llama_state_set_data(wide_context.get(), state.data(), state.size()) == 0);

    model_ptr dense_model = load_model(model_path, false, TEST_BATCH_SIZE);
    CHECK(dense_model != nullptr);
    context_ptr dense_context = make_context(dense_model.get(), false, TEST_BATCH_SIZE);
    CHECK(dense_context != nullptr);
    CHECK(llama_state_set_data(dense_context.get(), state.data(), state.size()) == 0);
    uint32_t dense_pos = 0;
    decode_until(dense_context.get(), repeated, dense_pos, TEST_BATCH_SIZE);
    const std::vector<uint8_t> dense_state = save_state(dense_context.get());
    context_ptr compact_mismatch = make_context(model.get(), true, TEST_BATCH_SIZE);
    CHECK(compact_mismatch != nullptr);
    CHECK(llama_state_set_data(compact_mismatch.get(), dense_state.data(), dense_state.size()) == 0);

    CHECK(llama_kv_cache_seq_rm(file_restored.get(), 0, 1800, -1));
    check_compact_geometry(file_restored.get(), 1800);
    const std::vector<uint8_t> rewind_state = save_state(file_restored.get());
    context_ptr rewind_restored = make_context(model.get(), true, TEST_BATCH_SIZE);
    CHECK(rewind_restored != nullptr);
    CHECK(llama_state_set_data(rewind_restored.get(), rewind_state.data(), rewind_state.size()) == rewind_state.size());
    check_compact_geometry(rewind_restored.get(), 1800);
    CHECK(save_state(rewind_restored.get()) == rewind_state);

    for (auto & cell : rewind_restored->kv_self.cells) {
        if (!cell.is_empty() && cell.pos == 1799) {
            cell.pos = 0;
            break;
        }
    }
    CHECK(llama_state_get_size(rewind_restored.get()) == 0);
    CHECK(llama_state_set_data(rewind_restored.get(), rewind_state.data(), rewind_state.size()) == rewind_state.size());
    check_compact_geometry(rewind_restored.get(), 1800);

    compact_mismatch.reset();
    dense_context.reset();
    dense_model.reset();
    wide_context.reset();
    wide_model.reset();
    rewind_restored.reset();
    malformed_target.reset();
    file_restored.reset();
    restored.reset();
    original.reset();
    model.reset();
    llama_backend_free();

    std::fprintf(stderr, "test-swa-state: success\n");
    return 0;
}
