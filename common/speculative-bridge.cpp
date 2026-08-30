#define LLAMA_API_INTERNAL
#include "llama-speculative.h"

#include "common.h"
#include "llama-spec-features-dflash.h"
#include "speculative.h"
#include "speculative-sampling.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t STATE_CONTAINER_VERSION = 1;
constexpr uint32_t STATE_SECTION_DSPARK = 1;
constexpr uint32_t STATE_SECTION_MTP = 2;
constexpr uint32_t STATE_SECTION_REQUIRED = 1u << 0;
constexpr size_t STATE_HEADER_SIZE = 72;
constexpr size_t STATE_SECTION_ENTRY_SIZE = 32;
constexpr size_t STATE_DSPARK_PREFIX_SIZE = 48;
constexpr size_t STATE_MTP_PREFIX_SIZE = 72;
constexpr size_t STATE_QUALITY_KEY_SIZE = 17;
constexpr std::array<uint8_t, 8> STATE_MAGIC = { 'L', 'L', 'S', 'P', 'S', 'T', 1, 0 };

struct sha256_context {
    uint32_t state[8];
    uint64_t bit_count;
    uint8_t buffer[64];
    size_t buffer_size;
};

constexpr uint32_t sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

constexpr uint32_t rotate_right(uint32_t value, uint32_t count) {
    return (value >> count) | (value << (32 - count));
}

void sha256_transform(sha256_context & context, const uint8_t block[64]) {
    uint32_t words[64];
    for (size_t i = 0; i < 16; ++i) {
        words[i] = (uint32_t) block[i*4] << 24 |
                   (uint32_t) block[i*4 + 1] << 16 |
                   (uint32_t) block[i*4 + 2] << 8 |
                   (uint32_t) block[i*4 + 3];
    }
    for (size_t i = 16; i < 64; ++i) {
        const uint32_t s0 = rotate_right(words[i - 15], 7) ^ rotate_right(words[i - 15], 18) ^ (words[i - 15] >> 3);
        const uint32_t s1 = rotate_right(words[i - 2], 17) ^ rotate_right(words[i - 2], 19) ^ (words[i - 2] >> 10);
        words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    uint32_t a = context.state[0];
    uint32_t b = context.state[1];
    uint32_t c = context.state[2];
    uint32_t d = context.state[3];
    uint32_t e = context.state[4];
    uint32_t f = context.state[5];
    uint32_t g = context.state[6];
    uint32_t h = context.state[7];
    for (size_t i = 0; i < 64; ++i) {
        const uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const uint32_t choose = (e & f) ^ (~e & g);
        const uint32_t temp1 = h + sum1 + choose + sha256_k[i] + words[i];
        const uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    context.state[0] += a;
    context.state[1] += b;
    context.state[2] += c;
    context.state[3] += d;
    context.state[4] += e;
    context.state[5] += f;
    context.state[6] += g;
    context.state[7] += h;
}

void sha256_init(sha256_context & context) {
    context = {};
    context.state[0] = 0x6a09e667;
    context.state[1] = 0xbb67ae85;
    context.state[2] = 0x3c6ef372;
    context.state[3] = 0xa54ff53a;
    context.state[4] = 0x510e527f;
    context.state[5] = 0x9b05688c;
    context.state[6] = 0x1f83d9ab;
    context.state[7] = 0x5be0cd19;
}

void sha256_update(sha256_context & context, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    if (size > (std::numeric_limits<uint64_t>::max() - context.bit_count) / 8) {
        throw std::overflow_error("SHA-256 input is too large");
    }
    context.bit_count += (uint64_t) size * 8;
    while (size > 0) {
        const size_t count = std::min(size, sizeof(context.buffer) - context.buffer_size);
        std::memcpy(context.buffer + context.buffer_size, bytes, count);
        context.buffer_size += count;
        bytes += count;
        size -= count;
        if (context.buffer_size == sizeof(context.buffer)) {
            sha256_transform(context, context.buffer);
            context.buffer_size = 0;
        }
    }
}

std::array<uint8_t, 32> sha256_finish(sha256_context & context) {
    const uint64_t bit_count = context.bit_count;
    const uint8_t one = 0x80;
    sha256_update(context, &one, 1);
    const uint8_t zero = 0;
    while (context.buffer_size != 56) {
        sha256_update(context, &zero, 1);
    }
    uint8_t length[8];
    for (size_t i = 0; i < 8; ++i) {
        length[7 - i] = (uint8_t) (bit_count >> (i * 8));
    }
    sha256_update(context, length, sizeof(length));

    std::array<uint8_t, 32> digest;
    for (size_t i = 0; i < 8; ++i) {
        digest[i*4] = (uint8_t) (context.state[i] >> 24);
        digest[i*4 + 1] = (uint8_t) (context.state[i] >> 16);
        digest[i*4 + 2] = (uint8_t) (context.state[i] >> 8);
        digest[i*4 + 3] = (uint8_t) context.state[i];
    }
    return digest;
}

std::array<uint8_t, 32> sha256(const std::string & text) {
    sha256_context context;
    sha256_init(context);
    sha256_update(context, text.data(), text.size());
    return sha256_finish(context);
}

std::array<uint8_t, 32> sha256_mtp_payload(const uint8_t * payload, size_t size) {
    sha256_context context;
    sha256_init(context);
    sha256_update(context, payload, 40);
    sha256_update(context, payload + STATE_MTP_PREFIX_SIZE, size - STATE_MTP_PREFIX_SIZE);
    return sha256_finish(context);
}

enum class bridge_stage_type {
    DSPARK,
    MTP,
};

struct parsed_stage {
    bridge_stage_type type = bridge_stage_type::DSPARK;
    int32_t n_max = 5;
    float p_min = 0.0f;
    int32_t cross_ctx = 512;
    int32_t mtp_heads = 1;
    std::string normalized;
};

struct sampler_state {
    std::vector<llama_token> accepted;
    std::mt19937 rng;
};

} // namespace

struct llama_speculative_error {
    llama_speculative_status status = LLAMA_SPECULATIVE_INTERNAL_ERROR;
    std::string message;
};

struct llama_speculative_engine {
    const llama_model * target_model = nullptr;
    llama_model * draft_model = nullptr;
    uint32_t draft_context_size = 0;
    int32_t draft_threads = -1;
    int32_t draft_batch_threads = -1;
    int32_t checkpoint_mode = LLAMA_SPEC_CKPT_AUTO;
    parsed_stage stage;
    std::vector<int32_t> target_layer_ids;
    int32_t feature_width = 0;
    int32_t block_size = 0;   // trained model metadata; part of the canonical state layout
    int32_t decode_tokens = 0; // prepared runtime query width; may exceed block_size for DSpark
    std::array<uint8_t, LLAMA_SPECULATIVE_STATE_LAYOUT_ID_SIZE> state_layout_id = {};
    llama_speculative_engine_capabilities capabilities = {};
    std::atomic<uint64_t> child_count{0};
};

struct llama_speculative_sampler {
    llama_speculative_engine * engine = nullptr;
    llama_speculative_sampler_params params = {};
    sampler_state canonical;
    std::vector<llama_token_data> candidates;
    std::vector<float> control_bias;
    std::vector<float> probabilities;
    bool probe_open = false;
    llama_token probe_token = LLAMA_TOKEN_NULL;
    std::mt19937 probe_rng;
    bool bound = false;
};

struct llama_speculative_session {
    llama_speculative_engine * engine = nullptr;
    llama_context * target_context = nullptr;
    common_speculative * speculative = nullptr;
    llama_speculative_sampler * sampler = nullptr;
    llama_speculative_round * active_round = nullptr;
    llama_pos sequence_start_position = 0;
    llama_pos next_position = 0;
    std::thread::id owner_thread;
    llama_speculative_metrics metrics = {};
    bool generation_open = false;
    bool round_open = false;
    bool attached = false;
    bool poisoned = false;
    bool mtp_prefix_complete = false;
    std::vector<float> authoritative_logits;
    bool has_authoritative_logits = false;
    llama_speculative_internal::sampler_token_carry private_carry;

    ~llama_speculative_session() {
        if (speculative != nullptr) {
            common_speculative_free(speculative);
        }
    }
};

struct llama_speculative_round {
    llama_speculative_session * session = nullptr;
    llama_speculative_round_params params = {};
    llama_speculative_round_view view = {};
    llama_batch verify_batch = {};
    llama_pos * verify_owned_positions = nullptr;
    std::vector<llama_pos> verify_scalar_positions;
    std::vector<llama_pos> verify_positions;
    sampler_state provisional;
    std::vector<llama_token> committable_tokens;
    std::vector<llama_token> commit_ids;
    std::vector<std::mt19937> prefix_rng;
    std::vector<float> prefix_logits;
    std::vector<float> hidden_rows;
    std::vector<int32_t> output_indices;
    std::vector<float> previous_logits;
    llama_speculative_internal::sampler_token_carry outgoing_carry;
    bool checkpoint_saved = false;
    bool companion_draft_tail_open = false;
    bool companion_boundary_row_retained = false;

    ~llama_speculative_round() {
        if (companion_draft_tail_open && session != nullptr && session->speculative != nullptr) {
            try {
                if (!common_speculative_mtp_discard_draft_tail(
                        session->speculative, 0, view.initial_position)) {
                    common_speculative_mtp_state_reset(session->speculative, 0);
                    session->mtp_prefix_complete = false;
                }
            } catch (...) {
                session->mtp_prefix_complete = false;
            }
        }
        if (verify_batch.token != nullptr) {
            verify_batch.pos = verify_owned_positions;
            llama_batch_free(verify_batch);
        }
    }
};

namespace {

std::mutex ownership_mutex;
std::unordered_set<llama_model *> reserved_draft_models;
std::unordered_set<llama_context *> attached_target_contexts;

void release_draft_model(llama_model * model) noexcept {
    try {
        std::lock_guard<std::mutex> lock(ownership_mutex);
        reserved_draft_models.erase(model);
    } catch (...) {
        // Cleanup APIs must never propagate a C++ exception across the C ABI.
    }
}

void release_target_context(llama_context * context) noexcept {
    try {
        std::lock_guard<std::mutex> lock(ownership_mutex);
        attached_target_contexts.erase(context);
    } catch (...) {
        // Cleanup APIs must never propagate a C++ exception across the C ABI.
    }
}

void clear_error(llama_speculative_error ** error) {
    if (error != nullptr) {
        *error = nullptr;
    }
}

void set_error(
        llama_speculative_error ** error,
        llama_speculative_status status,
        std::string_view message) noexcept {
    if (error == nullptr) {
        return;
    }
    try {
        *error = new llama_speculative_error{status, std::string(message)};
    } catch (...) {
        *error = nullptr;
    }
}

llama_speculative_status fail(
        llama_speculative_error ** error,
        llama_speculative_status status,
        std::string_view message) noexcept {
    set_error(error, status, message);
    return status;
}

void init_call_result(
        llama_speculative_call_result * result,
        llama_pos position,
        llama_speculative_status status = LLAMA_SPECULATIVE_INVALID_ARGUMENT) {
    if (result != nullptr) {
        *result = {
            status,
            LLAMA_SPECULATIVE_EFFECT_UNCHANGED,
            LLAMA_SPECULATIVE_BOUNDARY_PRE_CALL,
            0,
            position,
        };
    }
}

llama_speculative_status finish_call(
        llama_speculative_call_result * result,
        llama_speculative_status status,
        llama_speculative_effect effect,
        llama_speculative_boundary boundary,
        uint64_t consumed_count,
        llama_pos resulting_position) {
    if (result != nullptr) {
        *result = { status, effect, boundary, consumed_count, resulting_position };
    }
    return status;
}

llama_speculative_status fail_call(
        llama_speculative_call_result * result,
        llama_speculative_error ** error,
        llama_speculative_status status,
        std::string_view message,
        llama_pos position,
        llama_speculative_effect effect = LLAMA_SPECULATIVE_EFFECT_UNCHANGED,
        llama_speculative_boundary boundary = LLAMA_SPECULATIVE_BOUNDARY_PRE_CALL,
        uint64_t consumed_count = 0) noexcept {
    finish_call(result, status, effect, boundary, consumed_count, position);
    return fail(error, status, message);
}

bool position_add_fits(llama_pos position, uint64_t count) {
    return position >= 0 && count <= (uint64_t) std::numeric_limits<llama_pos>::max() - (uint64_t) position;
}

bool checked_position_add(llama_pos position, uint64_t count, llama_pos & result) {
    if (!position_add_fits(position, count)) {
        return false;
    }
    result = position + (llama_pos) count;
    return true;
}

bool parse_i32(const std::string & text, int32_t minimum, int32_t maximum, int32_t & value) {
    if (text.empty()) {
        return false;
    }
    int32_t parsed = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (result.ec != std::errc() || result.ptr != text.data() + text.size() || parsed < minimum || parsed > maximum) {
        return false;
    }
    value = parsed;
    return true;
}

bool parse_float(const std::string & text, float minimum, float maximum, float & value) {
    if (text.empty()) {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const float parsed = std::strtof(text.c_str(), &end);
    if (errno != 0 || end != text.c_str() + text.size() || !std::isfinite(parsed) || parsed < minimum || parsed > maximum) {
        return false;
    }
    value = parsed;
    return true;
}

bool parse_stage_expression(const char * expression, parsed_stage & stage, std::string & error) {
    if (expression == nullptr || *expression == '\0') {
        error = "stage_expression is empty";
        return false;
    }

    stage = {};
    const std::string input(expression);
    const size_t colon = input.find(':');
    const std::string name = input.substr(0, colon);
    const bool is_dspark = name == "draft-dspark" || name == "draft_dspark" || name == "dspark";
    const bool is_mtp = name == "mtp" || name == "draft-mtp" || name == "draft_mtp";
    if (!is_dspark && !is_mtp) {
        error = "v1 supports exactly one draft-dspark or mtp stage";
        return false;
    }
    stage.type = is_mtp ? bridge_stage_type::MTP : bridge_stage_type::DSPARK;
    if (input.find('+') != std::string::npos || input.find(';') != std::string::npos) {
        error = "v1 does not support speculative stage chains";
        return false;
    }

    bool saw_n_max = false;
    bool saw_p_min = false;
    bool saw_cross_ctx = false;
    bool saw_mtp_heads = false;
    if (colon != std::string::npos) {
        size_t begin = colon + 1;
        while (begin <= input.size()) {
            const size_t comma = input.find(',', begin);
            const std::string item = input.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
            const size_t equal = item.find('=');
            if (item.empty() || equal == std::string::npos || item.find('=', equal + 1) != std::string::npos) {
                error = "invalid speculative stage key/value list";
                return false;
            }
            const std::string key = item.substr(0, equal);
            const std::string value = item.substr(equal + 1);
            if (key == "n_max") {
                if (saw_n_max || !parse_i32(value, 1, 1024, stage.n_max)) {
                    error = "invalid or duplicate n_max";
                    return false;
                }
                saw_n_max = true;
            } else if (key == "p_min") {
                if (saw_p_min || !parse_float(value, 0.0f, 1.0f, stage.p_min)) {
                    error = "invalid or duplicate p_min";
                    return false;
                }
                saw_p_min = true;
            } else if (key == "cross_ctx") {
                if (!is_dspark || saw_cross_ctx ||
                        !parse_i32(value, 1, std::numeric_limits<int32_t>::max(), stage.cross_ctx)) {
                    error = "invalid or duplicate cross_ctx";
                    return false;
                }
                saw_cross_ctx = true;
            } else if (key == "heads" || key == "mtp_heads") {
                if (!is_mtp || saw_mtp_heads || !parse_i32(value, 0, 1024, stage.mtp_heads)) {
                    error = "invalid or duplicate MTP heads";
                    return false;
                }
                saw_mtp_heads = true;
            } else {
                error = "unknown speculative stage key '" + key + "'";
                return false;
            }
            if (comma == std::string::npos) {
                break;
            }
            begin = comma + 1;
        }
    }

    std::ostringstream normalized;
    if (is_mtp) {
        normalized << "mtp:n_max=" << stage.n_max
                   << ",p_min=" << stage.p_min
                   << ",heads=" << stage.mtp_heads;
    } else {
        normalized << "draft-dspark:n_max=" << stage.n_max
                   << ",p_min=" << stage.p_min
                   << ",cross_ctx=" << stage.cross_ctx;
    }
    stage.normalized = normalized.str();
    return true;
}

bool dflash_vocab_compatible(const llama_model * target, const llama_model * draft, std::string & error) {
    const llama_vocab * target_vocab = llama_model_get_vocab(target);
    const llama_vocab * draft_vocab = llama_model_get_vocab(draft);
    if (target_vocab == nullptr || draft_vocab == nullptr || llama_vocab_type(target_vocab) != llama_vocab_type(draft_vocab)) {
        error = "target and draft vocabulary types differ";
        return false;
    }
    if (llama_vocab_get_add_bos(target_vocab) != llama_vocab_get_add_bos(draft_vocab) ||
            llama_vocab_get_add_eos(target_vocab) != llama_vocab_get_add_eos(draft_vocab) ||
            llama_vocab_bos(target_vocab) != llama_vocab_bos(draft_vocab) ||
            llama_vocab_eos(target_vocab) != llama_vocab_eos(draft_vocab)) {
        error = "target and draft special-token contracts differ";
        return false;
    }
    const int32_t target_count = llama_vocab_n_tokens(target_vocab);
    const int32_t draft_count = llama_vocab_n_tokens(draft_vocab);
    if (target_count <= 0 || draft_count <= 0 || std::abs(target_count - draft_count) > 128) {
        error = "target and draft vocabulary sizes are incompatible";
        return false;
    }
    for (int32_t token = 5; token < std::min(target_count, draft_count); ++token) {
        if (std::strcmp(llama_vocab_get_text(target_vocab, token), llama_vocab_get_text(draft_vocab, token)) != 0) {
            error = "target and draft token text differs at token " + std::to_string(token);
            return false;
        }
    }
    return true;
}

bool mtp_vocab_compatible(const llama_model * target, const llama_model * companion, std::string & error) {
    const llama_vocab * target_vocab = llama_model_get_vocab(target);
    const llama_vocab * companion_vocab = llama_model_get_vocab(companion);
    if (target_vocab == nullptr || companion_vocab == nullptr ||
            llama_vocab_type(target_vocab) != llama_vocab_type(companion_vocab) ||
            llama_vocab_get_add_bos(target_vocab) != llama_vocab_get_add_bos(companion_vocab) ||
            llama_vocab_get_add_eos(target_vocab) != llama_vocab_get_add_eos(companion_vocab) ||
            llama_vocab_bos(target_vocab) != llama_vocab_bos(companion_vocab) ||
            llama_vocab_eos(target_vocab) != llama_vocab_eos(companion_vocab)) {
        error = "target and MTP companion vocabulary contracts differ";
        return false;
    }

    const int32_t target_count = llama_vocab_n_tokens(target_vocab);
    if (target_count <= 0 || target_count != llama_vocab_n_tokens(companion_vocab)) {
        error = "target and MTP companion vocabulary sizes differ";
        return false;
    }
    for (llama_token token = 0; token < target_count; ++token) {
        const char * target_text = llama_vocab_get_text(target_vocab, token);
        const char * companion_text = llama_vocab_get_text(companion_vocab, token);
        const float target_score = llama_token_get_score(target, token);
        const float companion_score = llama_token_get_score(companion, token);
        if (target_text == nullptr || companion_text == nullptr ||
                std::strcmp(target_text, companion_text) != 0 ||
                llama_token_get_attr(target, token) != llama_token_get_attr(companion, token) ||
                std::memcmp(&target_score, &companion_score, sizeof(target_score)) != 0) {
            error = "target and MTP companion token mapping differs at token " + std::to_string(token);
            return false;
        }
    }
    return true;
}

bool checked_add(uint64_t left, uint64_t right, uint64_t & result) {
    if (right > std::numeric_limits<uint64_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

bool checked_multiply(uint64_t left, uint64_t right, uint64_t & result) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

uint16_t read_u16(const uint8_t * source) {
    return (uint16_t) source[0] | (uint16_t) source[1] << 8;
}

uint32_t read_u32(const uint8_t * source) {
    return (uint32_t) source[0] |
           (uint32_t) source[1] << 8 |
           (uint32_t) source[2] << 16 |
           (uint32_t) source[3] << 24;
}

uint64_t read_u64(const uint8_t * source) {
    uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i) {
        value |= (uint64_t) source[i] << (i * 8);
    }
    return value;
}

int32_t read_i32(const uint8_t * source) {
    const uint32_t bits = read_u32(source);
    int32_t value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

int64_t read_i64(const uint8_t * source) {
    const uint64_t bits = read_u64(source);
    int64_t value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float read_f32(const uint8_t * source) {
    const uint32_t bits = read_u32(source);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void write_u16(uint8_t * destination, uint16_t value) {
    destination[0] = (uint8_t) value;
    destination[1] = (uint8_t) (value >> 8);
}

void write_u32(uint8_t * destination, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) {
        destination[i] = (uint8_t) (value >> (i * 8));
    }
}

void write_u64(uint8_t * destination, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
        destination[i] = (uint8_t) (value >> (i * 8));
    }
}

void write_i32(uint8_t * destination, int32_t value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32(destination, bits);
}

void write_i64(uint8_t * destination, int64_t value) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u64(destination, bits);
}

void write_f32(uint8_t * destination, float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32(destination, bits);
}

void write_be64(uint8_t * destination, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
        destination[i] = (uint8_t) (value >> ((7 - i) * 8));
    }
}

struct parsed_dspark_state {
    uint32_t row_count = 0;
    uint32_t feature_width = 0;
    uint32_t cross_ctx = 0;
    uint32_t target_layer_count = 0;
    llama_pos sequence_start_position = 0;
    llama_pos coverage_start_position = 0;
    llama_pos window_start_position = 0;
    llama_pos next_position = 0;
    const uint8_t * layer_ids = nullptr;
    const uint8_t * positions = nullptr;
    const uint8_t * features = nullptr;
    uint64_t feature_count = 0;
    uint64_t total_size = 0;
    bool ready = false;
};

struct parsed_mtp_state {
    uint32_t feature_width = 0;
    uint32_t warmed_heads = 0;
    uint32_t hidden_count = 0;
    llama_pos sequence_start_position = 0;
    llama_pos next_position = 0;
    const uint8_t * hidden = nullptr;
    const uint8_t * context_state = nullptr;
    uint64_t context_state_size = 0;
    uint64_t total_size = 0;
    bool prefix_complete = false;
    bool ready = false;
};

bool position_from_wire(int64_t wire, llama_pos & position) {
    if (wire < (int64_t) std::numeric_limits<llama_pos>::min() ||
            wire > (int64_t) std::numeric_limits<llama_pos>::max()) {
        return false;
    }
    position = (llama_pos) wire;
    return true;
}

void make_quality_key(const parsed_dspark_state & state, uint8_t key[STATE_QUALITY_KEY_SIZE]) {
    key[0] = state.ready ? 1 : 0;
    write_be64(key + 1, UINT64_MAX - (uint64_t) state.coverage_start_position);
    write_be64(key + 9, state.row_count);
}

void make_quality_key(const parsed_mtp_state & state, uint8_t key[STATE_QUALITY_KEY_SIZE]) {
    key[0] = state.ready ? 1 : 0;
    write_be64(key + 1, state.prefix_complete ? UINT64_MAX : 0);
    write_be64(key + 9, state.context_state_size);
}

void fill_state_info(
        const llama_speculative_engine & engine,
        const parsed_dspark_state & state,
        llama_speculative_state_info * info) {
    if (info == nullptr) {
        return;
    }
    *info = {};
    info->container_version = STATE_CONTAINER_VERSION;
    info->flags = state.ready ? LLAMA_SPECULATIVE_STATE_CAN_DRAFT : 0;
    if (!state.ready && state.row_count != 0) {
        info->flags |= LLAMA_SPECULATIVE_STATE_PARTIAL_CONDITIONING;
    }
    std::memcpy(info->state_layout_id, engine.state_layout_id.data(), engine.state_layout_id.size());
    info->sequence_start_position = state.sequence_start_position;
    info->next_position = state.next_position;
    info->section_count = 1;
    info->quality_key_size = STATE_QUALITY_KEY_SIZE;
    info->canonical_state_bytes = state.total_size;
    info->derived_state_bytes = 0;
    info->readiness = state.ready
        ? LLAMA_SPECULATIVE_CONDITIONING_READY
        : LLAMA_SPECULATIVE_CONDITIONING_WARMING;
}

void fill_state_info(
        const llama_speculative_engine & engine,
        const parsed_mtp_state & state,
        llama_speculative_state_info * info) {
    if (info == nullptr) {
        return;
    }
    *info = {};
    info->container_version = STATE_CONTAINER_VERSION;
    info->flags = state.ready ? LLAMA_SPECULATIVE_STATE_CAN_DRAFT : 0;
    if (!state.prefix_complete && state.next_position > state.sequence_start_position) {
        info->flags |= LLAMA_SPECULATIVE_STATE_PARTIAL_CONDITIONING;
    }
    std::memcpy(info->state_layout_id, engine.state_layout_id.data(), engine.state_layout_id.size());
    info->sequence_start_position = state.sequence_start_position;
    info->next_position = state.next_position;
    info->section_count = 1;
    info->quality_key_size = STATE_QUALITY_KEY_SIZE;
    info->canonical_state_bytes = state.total_size;
    info->derived_state_bytes = 0;
    info->readiness = state.ready
        ? LLAMA_SPECULATIVE_CONDITIONING_READY
        : LLAMA_SPECULATIVE_CONDITIONING_WARMING;
}

llama_speculative_status parse_state(
        const llama_speculative_engine & engine,
        const void * state,
        uint64_t state_size,
        parsed_dspark_state & parsed,
        std::string & message) {
    parsed = {};
    if (state == nullptr || state_size < STATE_HEADER_SIZE + STATE_SECTION_ENTRY_SIZE ||
            state_size > engine.capabilities.maximum_canonical_state_bytes || state_size > UINT32_MAX) {
        message = "speculative state length is outside the v1 bounds";
        return LLAMA_SPECULATIVE_STATE_CORRUPT;
    }
    const auto * bytes = static_cast<const uint8_t *>(state);
    if (!std::equal(STATE_MAGIC.begin(), STATE_MAGIC.end(), bytes)) {
        message = "speculative state magic or container version differs";
        return LLAMA_SPECULATIVE_STATE_INCOMPATIBLE;
    }
    if (read_u32(bytes + 8) != STATE_HEADER_SIZE || read_u32(bytes + 12) != state_size) {
        message = "speculative state header or total length is invalid";
        return LLAMA_SPECULATIVE_STATE_CORRUPT;
    }
    if (std::memcmp(bytes + 16, engine.state_layout_id.data(), engine.state_layout_id.size()) != 0) {
        message = "speculative state layout differs from the engine";
        return LLAMA_SPECULATIVE_STATE_INCOMPATIBLE;
    }
    if (read_u32(bytes + 64) != 0 || read_u32(bytes + 68) != 1) {
        message = "unsupported speculative container flags or section count";
        return LLAMA_SPECULATIVE_STATE_INCOMPATIBLE;
    }

    llama_pos header_sequence_start;
    llama_pos header_next;
    if (!position_from_wire(read_i64(bytes + 48), header_sequence_start) ||
            !position_from_wire(read_i64(bytes + 56), header_next) ||
            header_sequence_start < 0 || header_next < header_sequence_start) {
        message = "speculative state header positions are invalid";
        return LLAMA_SPECULATIVE_STATE_CORRUPT;
    }

    const uint8_t * section = bytes + STATE_HEADER_SIZE;
    const uint64_t section_offset = read_u64(section + 16);
    const uint64_t section_size = read_u64(section + 24);
    uint64_t section_end = 0;
    if (read_u32(section) != 0 || read_u32(section + 4) != STATE_SECTION_DSPARK ||
            read_u16(section + 8) != 1 || read_u16(section + 10) != 0 ||
            read_u32(section + 12) != STATE_SECTION_REQUIRED) {
        message = "required DSpark state section is incompatible";
        return LLAMA_SPECULATIVE_STATE_INCOMPATIBLE;
    }
    if (section_offset != STATE_HEADER_SIZE + STATE_SECTION_ENTRY_SIZE ||
            !checked_add(section_offset, section_size, section_end) || section_end != state_size ||
            section_size < STATE_DSPARK_PREFIX_SIZE) {
        message = "DSpark section boundaries are invalid";
        return LLAMA_SPECULATIVE_STATE_CORRUPT;
    }

    const uint8_t * payload = bytes + section_offset;
    parsed.row_count = read_u32(payload);
    parsed.feature_width = read_u32(payload + 4);
    parsed.cross_ctx = read_u32(payload + 8);
    parsed.target_layer_count = read_u32(payload + 12);
    if (parsed.row_count > parsed.cross_ctx || parsed.cross_ctx != (uint32_t) engine.stage.cross_ctx ||
            parsed.feature_width != (uint32_t) engine.feature_width ||
            parsed.target_layer_count != engine.target_layer_ids.size()) {
        message = "DSpark section dimensions differ from the engine";
        return LLAMA_SPECULATIVE_STATE_INCOMPATIBLE;
    }
    if (!position_from_wire(read_i64(payload + 16), parsed.sequence_start_position) ||
            !position_from_wire(read_i64(payload + 24), parsed.coverage_start_position) ||
            !position_from_wire(read_i64(payload + 32), parsed.window_start_position) ||
            !position_from_wire(read_i64(payload + 40), parsed.next_position) ||
            parsed.sequence_start_position < 0 ||
            parsed.sequence_start_position > parsed.coverage_start_position ||
            parsed.coverage_start_position > parsed.window_start_position ||
            parsed.window_start_position > parsed.next_position ||
            parsed.sequence_start_position != header_sequence_start || parsed.next_position != header_next) {
        message = "DSpark section positions are invalid";
        return LLAMA_SPECULATIVE_STATE_CORRUPT;
    }

    uint64_t layer_bytes = 0;
    uint64_t position_bytes = 0;
    uint64_t feature_bytes = 0;
    uint64_t expected_size = STATE_DSPARK_PREFIX_SIZE;
    if (!checked_multiply(parsed.target_layer_count, sizeof(int32_t), layer_bytes) ||
            !checked_multiply(parsed.row_count, sizeof(int64_t), position_bytes) ||
            !checked_multiply(parsed.row_count, parsed.feature_width, parsed.feature_count) ||
            !checked_multiply(parsed.feature_count, sizeof(float), feature_bytes) ||
            !checked_add(expected_size, layer_bytes, expected_size) ||
            !checked_add(expected_size, position_bytes, expected_size) ||
            !checked_add(expected_size, feature_bytes, expected_size) || expected_size != section_size) {
        message = "DSpark section length is invalid";
        return LLAMA_SPECULATIVE_STATE_CORRUPT;
    }
    parsed.layer_ids = payload + STATE_DSPARK_PREFIX_SIZE;
    parsed.positions = parsed.layer_ids + layer_bytes;
    parsed.features = parsed.positions + position_bytes;
    for (uint32_t i = 0; i < parsed.target_layer_count; ++i) {
        if (read_i32(parsed.layer_ids + (uint64_t) i * sizeof(int32_t)) != engine.target_layer_ids[i]) {
            message = "DSpark target layer IDs differ from the engine";
            return LLAMA_SPECULATIVE_STATE_INCOMPATIBLE;
        }
    }

    const int64_t horizon_wire = std::max<int64_t>(
        parsed.sequence_start_position,
        (int64_t) parsed.next_position - (int64_t) parsed.cross_ctx);
    const llama_pos horizon_start = (llama_pos) horizon_wire;
    if (parsed.window_start_position != std::max(parsed.coverage_start_position, horizon_start) ||
            (int64_t) parsed.next_position - parsed.window_start_position != parsed.row_count) {
        message = "DSpark coverage horizon is inconsistent";
        return LLAMA_SPECULATIVE_STATE_CORRUPT;
    }
    for (uint32_t i = 0; i < parsed.row_count; ++i) {
        llama_pos position;
        if (!position_from_wire(read_i64(parsed.positions + (uint64_t) i * sizeof(int64_t)), position) ||
                position != parsed.window_start_position + (llama_pos) i) {
            message = "DSpark feature positions are not an exact contiguous suffix";
            return LLAMA_SPECULATIVE_STATE_CORRUPT;
        }
    }
    for (uint64_t i = 0; i < parsed.feature_count; ++i) {
        if (!std::isfinite(read_f32(parsed.features + i * sizeof(float)))) {
            message = "DSpark state contains a non-finite feature value";
            return LLAMA_SPECULATIVE_STATE_CORRUPT;
        }
    }
    parsed.total_size = state_size;
    parsed.ready = parsed.coverage_start_position <= horizon_start;
    return LLAMA_SPECULATIVE_OK;
}

llama_speculative_status parse_mtp_state(
        const llama_speculative_engine & engine,
        const void * state,
        uint64_t state_size,
        parsed_mtp_state & parsed,
        std::string & message) {
    parsed = {};
    if (state == nullptr || state_size < STATE_HEADER_SIZE + STATE_SECTION_ENTRY_SIZE + STATE_MTP_PREFIX_SIZE ||
            state_size > engine.capabilities.maximum_canonical_state_bytes || state_size > UINT32_MAX) {
        message = "MTP speculative state length is outside the v1 bounds";
        return LLAMA_SPECULATIVE_STATE_CORRUPT;
    }
    const auto * bytes = static_cast<const uint8_t *>(state);
    if (!std::equal(STATE_MAGIC.begin(), STATE_MAGIC.end(), bytes)) {
        message = "speculative state magic or container version differs";
        return LLAMA_SPECULATIVE_STATE_INCOMPATIBLE;
    }
    if (read_u32(bytes + 8) != STATE_HEADER_SIZE || read_u32(bytes + 12) != state_size) {
        message = "speculative state header or total length is invalid";
        return LLAMA_SPECULATIVE_STATE_CORRUPT;
    }
    if (std::memcmp(bytes + 16, engine.state_layout_id.data(), engine.state_layout_id.size()) != 0) {
        message = "speculative state layout differs from the engine";
        return LLAMA_SPECULATIVE_STATE_INCOMPATIBLE;
    }
    if (read_u32(bytes + 64) != 0 || read_u32(bytes + 68) != 1) {
        message = "unsupported speculative container flags or section count";
        return LLAMA_SPECULATIVE_STATE_INCOMPATIBLE;
    }

    llama_pos header_sequence_start;
    llama_pos header_next;
    if (!position_from_wire(read_i64(bytes + 48), header_sequence_start) ||
            !position_from_wire(read_i64(bytes + 56), header_next) ||
            header_sequence_start < 0 || header_next < header_sequence_start) {
        message = "speculative state header positions are invalid";
        return LLAMA_SPECULATIVE_STATE_CORRUPT;
    }

    const uint8_t * section = bytes + STATE_HEADER_SIZE;
    const uint64_t section_offset = read_u64(section + 16);
    const uint64_t section_size = read_u64(section + 24);
    uint64_t section_end = 0;
    if (read_u32(section) != 0 || read_u32(section + 4) != STATE_SECTION_MTP ||
            read_u16(section + 8) != 1 || read_u16(section + 10) != 0 ||
            read_u32(section + 12) != STATE_SECTION_REQUIRED) {
        message = "required MTP state section is incompatible";
        return LLAMA_SPECULATIVE_STATE_INCOMPATIBLE;
    }
    if (section_offset != STATE_HEADER_SIZE + STATE_SECTION_ENTRY_SIZE ||
            !checked_add(section_offset, section_size, section_end) || section_end != state_size ||
            section_size < STATE_MTP_PREFIX_SIZE) {
        message = "MTP section boundaries are invalid";
        return LLAMA_SPECULATIVE_STATE_CORRUPT;
    }

    const uint8_t * payload = bytes + section_offset;
    parsed.feature_width = read_u32(payload);
    parsed.warmed_heads = read_u32(payload + 4);
    parsed.hidden_count = read_u32(payload + 8);
    const uint32_t flags = read_u32(payload + 12);
    parsed.prefix_complete = (flags & 1u) != 0;
    parsed.context_state_size = read_u64(payload + 32);
    if (parsed.feature_width != (uint32_t) engine.feature_width || parsed.warmed_heads > 1 ||
            (parsed.hidden_count != 0 && parsed.hidden_count != parsed.feature_width) ||
            (flags & ~1u) != 0 || parsed.context_state_size == 0) {
        message = "MTP section dimensions or flags differ from the engine";
        return LLAMA_SPECULATIVE_STATE_INCOMPATIBLE;
    }
    if (!position_from_wire(read_i64(payload + 16), parsed.sequence_start_position) ||
            !position_from_wire(read_i64(payload + 24), parsed.next_position) ||
            parsed.sequence_start_position < 0 || parsed.next_position < parsed.sequence_start_position ||
            parsed.sequence_start_position != header_sequence_start || parsed.next_position != header_next ||
            (parsed.prefix_complete && parsed.next_position > parsed.sequence_start_position &&
             (parsed.hidden_count != parsed.feature_width || parsed.warmed_heads != 1))) {
        message = "MTP section positions or prefix coverage are invalid";
        return LLAMA_SPECULATIVE_STATE_CORRUPT;
    }

    uint64_t hidden_bytes = 0;
    uint64_t expected_size = STATE_MTP_PREFIX_SIZE;
    if (!checked_multiply(parsed.hidden_count, sizeof(float), hidden_bytes) ||
            !checked_add(expected_size, hidden_bytes, expected_size) ||
            !checked_add(expected_size, parsed.context_state_size, expected_size) || expected_size != section_size) {
        message = "MTP section length is invalid";
        return LLAMA_SPECULATIVE_STATE_CORRUPT;
    }
    parsed.hidden = payload + STATE_MTP_PREFIX_SIZE;
    parsed.context_state = parsed.hidden + hidden_bytes;
    const auto digest = sha256_mtp_payload(payload, (size_t) section_size);
    if (!std::equal(digest.begin(), digest.end(), payload + 40)) {
        message = "MTP state checksum differs";
        return LLAMA_SPECULATIVE_STATE_CORRUPT;
    }
    for (uint32_t i = 0; i < parsed.hidden_count; ++i) {
        if (!std::isfinite(read_f32(parsed.hidden + (uint64_t) i * sizeof(float)))) {
            message = "MTP state contains a non-finite hidden value";
            return LLAMA_SPECULATIVE_STATE_CORRUPT;
        }
    }
    parsed.total_size = state_size;
    parsed.ready = parsed.prefix_complete && parsed.hidden_count == parsed.feature_width && parsed.warmed_heads == 1;
    return LLAMA_SPECULATIVE_OK;
}

bool state_from_session(
        llama_speculative_session & session,
        common_speculative_dflash_state_view & view,
        parsed_dspark_state & state,
        uint64_t & required_size) {
    if (!common_speculative_dflash_state_get_view(session.speculative, view) ||
            view.data.row_count < 0 || view.data.feature_width < 0 || view.data.cross_ctx < 0 ||
            view.data.target_layer_count < 0) {
        return false;
    }
    uint64_t layer_bytes = 0;
    uint64_t position_bytes = 0;
    uint64_t feature_bytes = 0;
    uint64_t payload_size = STATE_DSPARK_PREFIX_SIZE;
    if (!checked_multiply((uint64_t) view.data.target_layer_count, sizeof(int32_t), layer_bytes) ||
            !checked_multiply((uint64_t) view.data.row_count, sizeof(int64_t), position_bytes) ||
            !checked_multiply((uint64_t) view.data.feature_count, sizeof(float), feature_bytes) ||
            !checked_add(payload_size, layer_bytes, payload_size) ||
            !checked_add(payload_size, position_bytes, payload_size) ||
            !checked_add(payload_size, feature_bytes, payload_size) ||
            !checked_add(STATE_HEADER_SIZE + STATE_SECTION_ENTRY_SIZE, payload_size, required_size) ||
            required_size > UINT32_MAX) {
        return false;
    }
    state = {};
    state.row_count = (uint32_t) view.data.row_count;
    state.feature_width = (uint32_t) view.data.feature_width;
    state.cross_ctx = (uint32_t) view.data.cross_ctx;
    state.target_layer_count = (uint32_t) view.data.target_layer_count;
    state.sequence_start_position = view.data.sequence_start_position;
    state.coverage_start_position = view.data.coverage_start_position;
    state.window_start_position = view.data.window_start_position;
    state.next_position = view.data.next_position;
    state.feature_count = view.data.feature_count;
    state.total_size = required_size;
    state.ready = view.ready;
    return state.next_position == session.next_position &&
            state.sequence_start_position == session.sequence_start_position;
}

bool mtp_state_from_session(
        llama_speculative_session & session,
        common_speculative_mtp_state_view & view,
        std::vector<uint8_t> & context_state,
        parsed_mtp_state & state,
        uint64_t & required_size) {
    if (!common_speculative_mtp_state_get_view(session.speculative, 0, view) ||
            view.target_hidden_count > UINT32_MAX || view.warmed_heads < 0 || view.warmed_heads > 1) {
        LLAMA_LOG_ERROR("%s: invalid live MTP hidden-state view (count=%zu, warmed=%d)\n",
                __func__, view.target_hidden_count, view.warmed_heads);
        return false;
    }
    if (view.target_hidden_count != 0 &&
            (view.target_hidden_count != (size_t) session.engine->feature_width || view.target_hidden == nullptr)) {
        LLAMA_LOG_ERROR("%s: MTP hidden width differs from engine (count=%zu, expected=%d, data=%p)\n",
                __func__, view.target_hidden_count, session.engine->feature_width,
                (const void *) view.target_hidden);
        return false;
    }
    for (size_t i = 0; i < view.target_hidden_count; ++i) {
        if (!std::isfinite(view.target_hidden[i])) {
            LLAMA_LOG_ERROR("%s: live MTP hidden row contains a non-finite value at %zu\n", __func__, i);
            return false;
        }
    }

    llama_context * companion = common_speculative_get_companion_ctx(session.speculative);
    if (companion == nullptr) {
        return false;
    }
    const size_t context_size = llama_state_seq_get_size(companion, 0, 0);
    if (context_size == 0 || context_size == SIZE_MAX) {
        LLAMA_LOG_ERROR("%s: could not size live MTP companion state\n", __func__);
        return false;
    }
    context_state.resize(context_size);
    if (llama_state_seq_get_data(companion, context_state.data(), context_state.size(), 0, 0) != context_state.size()) {
        LLAMA_LOG_ERROR("%s: live MTP companion state changed while being serialized\n", __func__);
        return false;
    }

    uint64_t hidden_bytes = 0;
    uint64_t payload_size = STATE_MTP_PREFIX_SIZE;
    if (!checked_multiply(view.target_hidden_count, sizeof(float), hidden_bytes) ||
            !checked_add(payload_size, hidden_bytes, payload_size) ||
            !checked_add(payload_size, context_state.size(), payload_size) ||
            !checked_add(STATE_HEADER_SIZE + STATE_SECTION_ENTRY_SIZE, payload_size, required_size) ||
            required_size > UINT32_MAX) {
        LLAMA_LOG_ERROR("%s: live MTP canonical-state length overflows\n", __func__);
        return false;
    }

    state = {};
    state.feature_width = (uint32_t) session.engine->feature_width;
    state.warmed_heads = (uint32_t) view.warmed_heads;
    state.hidden_count = (uint32_t) view.target_hidden_count;
    state.sequence_start_position = session.sequence_start_position;
    state.next_position = session.next_position;
    state.context_state_size = context_state.size();
    state.total_size = required_size;
    state.prefix_complete = session.mtp_prefix_complete;
    state.ready = state.prefix_complete && state.hidden_count == state.feature_width && state.warmed_heads == 1;
    const bool valid = (!state.prefix_complete || state.next_position == state.sequence_start_position || state.ready) &&
        state.next_position >= state.sequence_start_position;
    if (!valid) {
        LLAMA_LOG_ERROR("%s: inconsistent live MTP readiness (prefix=%d, ready=%d, hidden=%u, warmed=%u)\n",
                __func__, state.prefix_complete, state.ready, state.hidden_count, state.warmed_heads);
    }
    return valid;
}

bool fill_dspark_engine_metadata(llama_speculative_engine & engine, std::string & error) {
    if (!dflash_vocab_compatible(engine.target_model, engine.draft_model, error)) {
        return false;
    }
    if (!llama_model_is_deepseek4(engine.target_model)) {
        error = "draft-dspark requires a DeepSeek-V4 target model with checkpoint support";
        return false;
    }
    if (!llama_model_dflash_has_markov_head(engine.draft_model)) {
        error = "draft model does not contain a DSpark Markov head";
        return false;
    }

    engine.block_size = llama_model_dflash_block_size(engine.draft_model);
    const int32_t layer_count = llama_model_dflash_n_target_layers(engine.draft_model);
    engine.feature_width = llama_model_dflash_n_target_features(engine.draft_model);
    const int32_t target_width = llama_model_n_embd(engine.target_model);
    if (engine.block_size <= 0 || layer_count <= 0 || engine.feature_width <= 0 || target_width <= 0 ||
            (int64_t) target_width * layer_count != engine.feature_width) {
        error = "invalid DSpark block, layer, or feature dimensions";
        return false;
    }
    const auto width = common_speculative_plan_dflash_width(
        COMMON_SPECULATIVE_TYPE_DSPARK,
        engine.block_size,
        engine.stage.n_max);
    engine.decode_tokens = width.decode_tokens;
    if (width.effective_n_max != engine.stage.n_max || engine.decode_tokens <= 0) {
        error = "invalid DSpark runtime draft width";
        return false;
    }
    engine.target_layer_ids.resize((size_t) layer_count);
    if (llama_model_dflash_target_layer_ids(engine.draft_model, engine.target_layer_ids.data(), layer_count) != layer_count) {
        error = "could not read DSpark target layer IDs";
        return false;
    }
    std::vector<int32_t> sorted_layers = engine.target_layer_ids;
    std::sort(sorted_layers.begin(), sorted_layers.end());
    if (std::adjacent_find(sorted_layers.begin(), sorted_layers.end()) != sorted_layers.end()) {
        error = "DSpark target layer IDs contain duplicates";
        return false;
    }
    for (int32_t layer : engine.target_layer_ids) {
        if (layer < 0 || layer >= llama_n_layer(engine.target_model)) {
            error = "DSpark target layer ID is outside the target model";
            return false;
        }
    }
    const int32_t target_mask = llama_model_dflash_target_mask_token_id(engine.target_model);
    if (target_mask != LLAMA_TOKEN_NULL && target_mask != llama_model_dflash_mask_token_id(engine.draft_model)) {
        error = "target and DSpark mask token IDs differ";
        return false;
    }

    uint32_t required_context = 0;
    if (!common_speculative_dflash_context_size(engine.stage.cross_ctx, width, required_context)) {
        error = "derived draft context size overflows int32_t";
        return false;
    }
    if (engine.draft_context_size != 0 && engine.draft_context_size < required_context) {
        error = "draft context request is smaller than cross_ctx + runtime draft width";
        return false;
    }
    engine.draft_context_size = std::max(engine.draft_context_size, required_context);

    std::ostringstream layout;
    layout << "llama-speculative-state-layout" << '\0'
           << "container=1" << '\0'
           << "stage=0:draft-dspark" << '\0'
           << "binding=draft" << '\0'
           << "canonical-schema=1.0" << '\0'
           << "cross_ctx=" << engine.stage.cross_ctx << '\0'
           << "feature_width=" << engine.feature_width << '\0'
           << "block_size=" << engine.block_size << '\0'
           << "target_vocab=" << llama_vocab_n_tokens(llama_model_get_vocab(engine.target_model)) << '\0'
           << "draft_vocab=" << llama_vocab_n_tokens(llama_model_get_vocab(engine.draft_model)) << '\0'
           << "target_layers=";
    for (int32_t layer : engine.target_layer_ids) {
        layout << layer << ',';
    }
    engine.state_layout_id = sha256(layout.str());

    uint64_t feature_bytes = 0;
    uint64_t position_bytes = 0;
    uint64_t section_bytes = 0;
    if (!checked_multiply((uint64_t) engine.stage.cross_ctx, (uint64_t) engine.feature_width, feature_bytes) ||
            !checked_multiply(feature_bytes, sizeof(float), feature_bytes) ||
            !checked_multiply((uint64_t) engine.stage.cross_ctx, sizeof(int64_t), position_bytes) ||
            !checked_add(48, (uint64_t) engine.target_layer_ids.size() * sizeof(int32_t), section_bytes) ||
            !checked_add(section_bytes, position_bytes, section_bytes) ||
            !checked_add(section_bytes, feature_bytes, section_bytes)) {
        error = "maximum DSpark state size overflows uint64_t";
        return false;
    }
    uint64_t maximum_state = 0;
    if (!checked_add(STATE_HEADER_SIZE + STATE_SECTION_ENTRY_SIZE, section_bytes, maximum_state) ||
            maximum_state > UINT32_MAX) {
        error = "maximum speculative container size exceeds the v1 32-bit length field";
        return false;
    }

    engine.capabilities.flags = LLAMA_SPECULATIVE_CAP_FIXED_SEED_TARGET_IDENTITY |
            LLAMA_SPECULATIVE_CAP_CANONICAL_STATE |
            LLAMA_SPECULATIVE_CAP_PORTABLE_CANONICAL_STATE |
            LLAMA_SPECULATIVE_CAP_TARGET_ONLY_WARMUP |
            LLAMA_SPECULATIVE_CAP_TOKEN_INPUT;
    engine.capabilities.required_target_sequence_count = 1;
    engine.capabilities.minimum_target_batch_tokens = (uint32_t) engine.stage.n_max + 1;
    engine.capabilities.minimum_target_ubatch_tokens = (uint32_t) engine.stage.n_max + 1;
    engine.capabilities.history_requirement = LLAMA_SPECULATIVE_HISTORY_NONE;
    engine.capabilities.history_lookback_tokens = (uint32_t) engine.stage.cross_ctx;
    engine.capabilities.configured_max_draft_tokens = (uint32_t) engine.stage.n_max;
    engine.capabilities.state_section_count = 1;
    engine.capabilities.maximum_canonical_state_bytes = maximum_state;
    engine.capabilities.maximum_derived_state_bytes = 0;
    engine.capabilities.maximum_quality_key_bytes = 17;
    return true;
}

bool fill_mtp_engine_metadata(llama_speculative_engine & engine, std::string & error) {
    const char * target_arch = llama_model_arch_string(engine.target_model);
    const char * companion_arch = llama_model_arch_string(engine.draft_model);
    if (target_arch == nullptr || companion_arch == nullptr ||
            std::strcmp(target_arch, "qwen4exp") != 0 || std::strcmp(companion_arch, "qwen4exp") != 0) {
        error = "mtp bridge stage requires Qwen3.8 qwen4exp target and companion models";
        return false;
    }
    const llama_mtp_package target_package = llama_model_mtp_package(engine.target_model);
    if (target_package != LLAMA_MTP_PACKAGE_TARGET_ONLY && target_package != LLAMA_MTP_PACKAGE_EMBEDDED) {
        error = "MTP target is a companion or invalid package";
        return false;
    }
    if (llama_model_mtp_package(engine.draft_model) != LLAMA_MTP_PACKAGE_COMPANION) {
        error = "MTP draft binding must be a predictor-only companion package";
        return false;
    }
    if (llama_model_n_nextn_layer(engine.draft_model) != 1 || engine.stage.mtp_heads > 1) {
        error = "Qwen3.8 MTP requires exactly one predictor layer and heads=0 or heads=1";
        return false;
    }
    if (!mtp_vocab_compatible(engine.target_model, engine.draft_model, error)) {
        return false;
    }

    const int32_t target_width = llama_model_n_embd(engine.target_model);
    const int32_t companion_width = llama_model_n_embd(engine.draft_model);
    const uint32_t target_feature_width = llama_model_mtp_feature_width(engine.target_model);
    const uint32_t companion_feature_width = llama_model_mtp_feature_width(engine.draft_model);
    if (target_width <= 0 || target_width != companion_width || target_feature_width == 0 ||
            target_feature_width > INT32_MAX || target_feature_width != companion_feature_width) {
        error = "Qwen3.8 target and companion embedding or MTP feature widths differ";
        return false;
    }
    engine.feature_width = (int32_t) target_feature_width;
    engine.decode_tokens = engine.stage.n_max + 1;

    std::ostringstream layout;
    layout << "llama-speculative-state-layout" << '\0'
           << "container=1" << '\0'
           << "stage=0:mtp" << '\0'
           << "binding=draft" << '\0'
           << "canonical-schema=mtp-1.0" << '\0'
           << "arch=qwen4exp" << '\0'
           << "n_embd=" << target_width << '\0'
           << "feature_width=" << engine.feature_width << '\0'
           << "heads=1" << '\0'
           << "draft_context=" << engine.draft_context_size << '\0'
           << "vocab=" << llama_vocab_n_tokens(llama_model_get_vocab(engine.target_model));
    engine.state_layout_id = sha256(layout.str());

    engine.capabilities.flags = LLAMA_SPECULATIVE_CAP_FIXED_SEED_TARGET_IDENTITY |
            LLAMA_SPECULATIVE_CAP_CANONICAL_STATE |
            LLAMA_SPECULATIVE_CAP_PORTABLE_CANONICAL_STATE |
            LLAMA_SPECULATIVE_CAP_TOKEN_INPUT |
            LLAMA_SPECULATIVE_CAP_REQUIRES_TARGET_MTP_OUTPUT;
    engine.capabilities.required_target_sequence_count = 1;
    engine.capabilities.minimum_target_batch_tokens = (uint32_t) engine.stage.n_max + 1;
    engine.capabilities.minimum_target_ubatch_tokens = (uint32_t) engine.stage.n_max + 1;
    engine.capabilities.history_requirement = LLAMA_SPECULATIVE_HISTORY_COMPLETE_PREFIX;
    engine.capabilities.history_lookback_tokens = 0;
    engine.capabilities.configured_max_draft_tokens = (uint32_t) engine.stage.n_max;
    engine.capabilities.state_section_count = 1;
    engine.capabilities.maximum_canonical_state_bytes = UINT32_MAX;
    engine.capabilities.maximum_derived_state_bytes = 0;
    engine.capabilities.maximum_quality_key_bytes = STATE_QUALITY_KEY_SIZE;
    return true;
}

bool fill_engine_metadata(llama_speculative_engine & engine, std::string & error) {
    return engine.stage.type == bridge_stage_type::MTP
        ? fill_mtp_engine_metadata(engine, error)
        : fill_dspark_engine_metadata(engine, error);
}

common_params_speculative make_common_params(
        const llama_speculative_engine & engine,
        const llama_context * target_context) {
    common_params_speculative params;
    const bool mtp = engine.stage.type == bridge_stage_type::MTP;
    params.type = mtp ? COMMON_SPECULATIVE_TYPE_MTP : COMMON_SPECULATIVE_TYPE_DSPARK;
    params.spec_ckpt_mode = engine.checkpoint_mode;
    params.n_threads = engine.draft_threads;
    params.n_threads_batch = engine.draft_batch_threads;
    params.n_max = engine.stage.n_max;
    params.n_min = 0;
    params.p_min = engine.stage.p_min;
    params.mtp_heads = engine.stage.mtp_heads;
    params.dflash_cross_ctx = engine.stage.cross_ctx;
    params.model_dft = engine.draft_model;
    params.cparams_dft = llama_context_default_params();
    if (mtp) {
        params.cparams_dft.n_ctx = engine.draft_context_size != 0
            ? engine.draft_context_size
            : llama_n_ctx(target_context);
        params.cparams_dft.n_batch = llama_n_batch(target_context);
        params.cparams_dft.n_ubatch = llama_n_ubatch(target_context);
        params.cparams_dft.n_seq_max = 1;
        params.cparams_dft.embeddings = true;
        params.cparams_dft.mtp = true;
        params.cparams_dft.mtp_op_type = MTP_OP_WARMUP;
    } else {
        params.cparams_dft.n_ctx = engine.draft_context_size;
        params.cparams_dft.n_batch = (uint32_t) engine.decode_tokens;
        params.cparams_dft.n_ubatch = (uint32_t) engine.decode_tokens;
    }
    if (engine.draft_threads > 0) {
        params.cparams_dft.n_threads = (uint32_t) engine.draft_threads;
    }
    if (engine.draft_batch_threads > 0) {
        params.cparams_dft.n_threads_batch = (uint32_t) engine.draft_batch_threads;
    }
    common_speculative_stage_params stage;
    stage.type = mtp ? COMMON_SPECULATIVE_TYPE_MTP : COMMON_SPECULATIVE_TYPE_DSPARK;
    stage.n_max = engine.stage.n_max;
    stage.n_min = 0;
    stage.p_min = engine.stage.p_min;
    stage.mtp_heads = engine.stage.mtp_heads;
    stage.dflash_cross_ctx = engine.stage.cross_ctx;
    params.stages.push_back(stage);
    return params;
}

uint32_t effective_seed(const llama_speculative_sampler & sampler) {
    if ((sampler.params.flags & LLAMA_SPECULATIVE_SAMPLER_FIXED_SEED) != 0 &&
            sampler.params.seed != LLAMA_DEFAULT_SEED) {
        return sampler.params.seed;
    }
    return (uint32_t) std::time(nullptr);
}

void initialize_sampler_rng(llama_speculative_sampler & sampler) {
    sampler.canonical.rng.seed(effective_seed(sampler));
}

llama_token sample_token(
        llama_speculative_sampler & sampler,
        sampler_state & state,
        llama_context * context,
        int32_t output_index,
        const float * logits_override = nullptr) {
    const int32_t vocabulary_size = (int32_t) sampler.candidates.size();
    const float * logits = logits_override != nullptr ? logits_override : llama_get_logits_ith(context, output_index);
    if (logits == nullptr) {
        return LLAMA_TOKEN_NULL;
    }
    for (int32_t token = 0; token < vocabulary_size; ++token) {
        sampler.candidates[(size_t) token] = {
            token,
            logits[token] + sampler.control_bias[(size_t) token],
            0.0f,
        };
    }
    llama_token_data_array candidates = {
        sampler.candidates.data(),
        sampler.candidates.size(),
        -1,
        false,
    };
    if (!state.accepted.empty() &&
            (sampler.params.repeat_penalty != 1.0f || sampler.params.frequency_penalty != 0.0f ||
             sampler.params.presence_penalty != 0.0f)) {
        size_t count = state.accepted.size();
        if (sampler.params.penalty_last_n >= 0) {
            count = std::min(count, (size_t) sampler.params.penalty_last_n);
        }
        llama_sample_repetition_penalties(
            context,
            &candidates,
            state.accepted.data() + state.accepted.size() - count,
            count,
            sampler.params.repeat_penalty,
            sampler.params.frequency_penalty,
            sampler.params.presence_penalty);
    }
    if (sampler.params.top_k >= 0) {
        llama_sample_top_k(context, &candidates, sampler.params.top_k, 1);
    }
    if (!std::isnan(sampler.params.top_p)) {
        llama_sample_top_p(context, &candidates, sampler.params.top_p, 1);
    }
    if (!std::isnan(sampler.params.min_p)) {
        llama_sample_min_p(context, &candidates, sampler.params.min_p, 1);
    }
    return llama_speculative_internal::sample_candidates(
        context, &candidates, sampler.params.temperature, state.rng, sampler.probabilities);
}

bool session_on_owner_thread(const llama_speculative_session * session) {
    return session != nullptr && session->owner_thread == std::this_thread::get_id();
}

struct materialized_primary_batch {
    llama_batch batch = {};
    std::vector<llama_pos> positions;
    std::vector<int32_t> sequence_counts;
    std::vector<llama_seq_id> sequence_ids;
    std::vector<llama_seq_id *> sequence_id_pointers;
    std::vector<int8_t> logits;
};

bool materialize_primary_batch(
        const llama_batch & batch,
        llama_pos expected_position,
        uint32_t maximum_tokens,
        llama_pos & resulting_position,
        materialized_primary_batch & materialized,
        std::string & error) {
    if (batch.n_tokens <= 0 || (uint32_t) batch.n_tokens > maximum_tokens || batch.token == nullptr ||
            batch.embd != nullptr ||
            !checked_position_add(expected_position, (uint64_t) batch.n_tokens, resulting_position)) {
        error = "target decode requires a bounded nonempty token batch and a valid resulting position";
        return false;
    }

    const size_t token_count = (size_t) batch.n_tokens;
    materialized.positions.resize(token_count * 4);
    materialized.sequence_counts.assign(token_count, 1);
    materialized.sequence_ids.assign(token_count, 0);
    materialized.sequence_id_pointers.resize(token_count);
    materialized.logits.resize(token_count);
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        const int64_t position = batch.pos != nullptr
            ? batch.pos[i]
            : (int64_t) batch.all_pos_0 + (int64_t) i * batch.all_pos_1;
        const bool compact_sequence = batch.n_seq_id == nullptr && batch.seq_id == nullptr;
        const bool explicit_sequence = batch.n_seq_id != nullptr && batch.seq_id != nullptr &&
            batch.n_seq_id[i] == 1 && batch.seq_id[i] != nullptr && batch.seq_id[i][0] == 0;
        if (position != (int64_t) expected_position + i ||
                (!compact_sequence && !explicit_sequence) ||
                (compact_sequence && batch.all_seq_id != 0)) {
            error = "target decode batch is not contiguous on primary sequence 0";
            return false;
        }
        materialized.positions[(size_t) i] = (llama_pos) position;
        materialized.positions[token_count + (size_t) i] = (llama_pos) position;
        materialized.positions[2 * token_count + (size_t) i] = (llama_pos) position;
        materialized.positions[3 * token_count + (size_t) i] = 0;
        materialized.sequence_id_pointers[(size_t) i] = &materialized.sequence_ids[(size_t) i];
        materialized.logits[(size_t) i] = batch.logits != nullptr ? batch.logits[i] : 0;
    }
    if (batch.logits == nullptr) {
        materialized.logits.back() = 1;
    } else if (batch.logits[batch.n_tokens - 1] == 0) {
        error = "target decode batch must request final boundary logits";
        return false;
    }

    materialized.batch = batch;
    materialized.batch.pos = materialized.positions.data();
    materialized.batch.n_seq_id = materialized.sequence_counts.data();
    materialized.batch.seq_id = materialized.sequence_id_pointers.data();
    materialized.batch.logits = materialized.logits.data();
    return true;
}

llama_speculative_status check_session_call(
        llama_speculative_session * session,
        llama_speculative_call_result * result,
        llama_speculative_error ** error,
        bool allow_detached = false) {
    const llama_pos position = session != nullptr ? session->next_position : 0;
    init_call_result(result, position);
    clear_error(error);
    if (session == nullptr || result == nullptr) {
        return fail_call(result, error, LLAMA_SPECULATIVE_INVALID_ARGUMENT,
                "session and result are required", position);
    }
    if (!session_on_owner_thread(session)) {
        return fail_call(result, error, LLAMA_SPECULATIVE_BUSY,
                "speculative session is thread-affine", position);
    }
    if (session->poisoned) {
        finish_call(result, LLAMA_SPECULATIVE_POISONED,
                LLAMA_SPECULATIVE_EFFECT_SESSION_POISONED_TARGET_UNKNOWN,
                LLAMA_SPECULATIVE_BOUNDARY_UNKNOWN, 0, session->next_position);
        return fail(error, LLAMA_SPECULATIVE_POISONED, "speculative session is poisoned");
    }
    if (!allow_detached && !session->attached) {
        return fail_call(result, error, LLAMA_SPECULATIVE_INVALID_ARGUMENT,
                "speculative session is detached", position);
    }
    return LLAMA_SPECULATIVE_OK;
}

bool reset_session_conditioning(
        llama_speculative_session & session,
        llama_pos boundary) {
    if (session.engine->stage.type == bridge_stage_type::MTP) {
        const bool reset = common_speculative_mtp_state_reset(session.speculative, 0);
        if (reset) {
            session.mtp_prefix_complete = boundary == session.sequence_start_position;
        }
        return reset;
    }
    return common_speculative_dflash_state_reset(
        session.speculative,
        session.sequence_start_position,
        boundary,
        boundary);
}

void poison_session(
        llama_speculative_session & session,
        llama_speculative_call_result * result,
        llama_speculative_status status) {
    session.poisoned = true;
    session.metrics.poisoned_failures++;
    finish_call(result, status,
            LLAMA_SPECULATIVE_EFFECT_SESSION_POISONED_TARGET_UNKNOWN,
            LLAMA_SPECULATIVE_BOUNDARY_UNKNOWN, 0, session.next_position);
}

llama_speculative_terminal candidate_terminal(
        const llama_speculative_round & round,
        llama_token token,
        uint64_t committed_prefix) {
    const llama_model * model = round.session->engine->target_model;
    if (llama_token_is_eog(model, token)) {
        return LLAMA_SPECULATIVE_TERMINAL_EOG;
    }
    if (committed_prefix >= round.params.generation_token_allowance) {
        return LLAMA_SPECULATIVE_TERMINAL_GENERATION_LIMIT;
    }
    if (llama_token_is_control(model, token)) {
        return LLAMA_SPECULATIVE_TERMINAL_BAD_CONTROL_TOKEN;
    }
    if (committed_prefix >= round.params.context_token_allowance) {
        return LLAMA_SPECULATIVE_TERMINAL_CONTEXT_LIMIT;
    }
    return LLAMA_SPECULATIVE_TERMINAL_NONE;
}

bool round_copy_output_state(llama_speculative_round & round) {
    llama_speculative_session & session = *round.session;
    const size_t vocabulary_size = session.authoritative_logits.size();
    const size_t output_count = round.output_indices.size();
    for (size_t i = 0; i < output_count; ++i) {
        const float * logits = llama_get_logits_ith(session.target_context, (int32_t) i);
        if (logits == nullptr) {
            return false;
        }
        std::memcpy(round.prefix_logits.data() + i * vocabulary_size, logits,
                vocabulary_size * sizeof(*logits));
    }
    return common_speculative_copy_output_hidden_rows(
        session.speculative,
        session.target_context,
        round.output_indices,
        round.hidden_rows);
}

void round_set_verify_token_count(llama_speculative_round & round, int32_t token_count) {
    GGML_ASSERT(token_count >= 0 && (size_t) token_count <= round.verify_scalar_positions.size());
    const size_t count = (size_t) token_count;
    for (size_t i = 0; i < count; ++i) {
        const llama_pos position = round.verify_scalar_positions[i];
        round.verify_positions[i] = position;
        round.verify_positions[count + i] = position;
        round.verify_positions[2 * count + i] = position;
        round.verify_positions[3 * count + i] = 0;
    }
    round.verify_batch.n_tokens = token_count;
}

bool round_restore_target(llama_speculative_round & round, uint64_t prefix_count) noexcept {
    try {
        llama_speculative_session & session = *round.session;
        if (!round.checkpoint_saved) {
            return prefix_count == 0;
        }
        if (prefix_count > round.committable_tokens.size() ||
                prefix_count > (uint64_t) std::numeric_limits<int>::max()) {
            return false;
        }

        const int accepted_step = (int) prefix_count - 1;
        const llama_spec_ckpt_restore_result restore = llama_spec_ckpt_restore_ex(
            session.target_context, 0, round.view.initial_position, accepted_step);
        if (restore == LLAMA_SPEC_CKPT_RESTORE_FAILED) {
            llama_spec_ckpt_discard(session.target_context);
            round.checkpoint_saved = false;
            return false;
        }

        if (restore == LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED && prefix_count != 0) {
            round_set_verify_token_count(round, (int32_t) prefix_count);
            if (llama_decode(session.target_context, round.verify_batch) != 0) {
                llama_spec_ckpt_discard(session.target_context);
                round.checkpoint_saved = false;
                return false;
            }
            session.metrics.target_decode_calls++;
            session.metrics.target_decode_tokens += prefix_count;
            if (session.engine->stage.type == bridge_stage_type::MTP) {
                const std::vector<int32_t> committed_outputs(
                    round.output_indices.begin(), round.output_indices.begin() + (size_t) prefix_count);
                if (!common_speculative_copy_output_hidden_rows(
                        session.speculative,
                        session.target_context,
                        committed_outputs,
                        round.hidden_rows)) {
                    llama_spec_ckpt_discard(session.target_context);
                    round.checkpoint_saved = false;
                    return false;
                }
            }
        }

        llama_spec_ckpt_discard(session.target_context);
        round.checkpoint_saved = false;
        const float * logits = round.previous_logits.data();
        if (prefix_count != 0) {
            const size_t vocabulary_size = session.authoritative_logits.size();
            logits = round.prefix_logits.data() + (prefix_count - 1) * vocabulary_size;
        }
        if (!llama_spec_restore_logits(session.target_context, logits, session.authoritative_logits.size())) {
            return false;
        }
        std::memcpy(session.authoritative_logits.data(), logits,
                session.authoritative_logits.size() * sizeof(*logits));
        session.has_authoritative_logits = false;
        return true;
    } catch (...) {
        try {
            llama_spec_ckpt_discard(round.session->target_context);
        } catch (...) {
        }
        round.checkpoint_saved = false;
        return false;
    }
}

enum class companion_tail_result {
    UNCHANGED,
    CLEARED,
    FAILED,
};

companion_tail_result round_discard_companion_tail(
        llama_speculative_round & round,
        bool keep_boundary_row = false) noexcept {
    if (!round.companion_draft_tail_open) {
        return companion_tail_result::UNCHANGED;
    }
    round.companion_draft_tail_open = false;
    try {
        const llama_pos discard_position = round.view.initial_position + (keep_boundary_row ? 1 : 0);
        if (common_speculative_mtp_discard_draft_tail(
                round.session->speculative, 0, discard_position)) {
            return companion_tail_result::UNCHANGED;
        }
        if (common_speculative_mtp_state_reset(round.session->speculative, 0)) {
            round.session->mtp_prefix_complete = false;
            return companion_tail_result::CLEARED;
        }
    } catch (...) {
    }
    return companion_tail_result::FAILED;
}

void finish_round(llama_speculative_round ** round) {
    if (round == nullptr || *round == nullptr) {
        return;
    }
    (*round)->session->round_open = false;
    if ((*round)->session->active_round == *round) {
        (*round)->session->active_round = nullptr;
    }
    delete *round;
    *round = nullptr;
}

} // namespace

extern "C" {

llama_speculative_status llama_speculative_error_status(const llama_speculative_error * error) {
    return error != nullptr ? error->status : LLAMA_SPECULATIVE_OK;
}

const char * llama_speculative_error_message(const llama_speculative_error * error) {
    return error != nullptr ? error->message.c_str() : "";
}

void llama_speculative_error_free(llama_speculative_error * error) {
    delete error;
}

llama_speculative_engine_params llama_speculative_engine_default_params(void) {
    llama_speculative_engine_params params = {};
    params.checkpoint_mode = LLAMA_SPEC_CKPT_AUTO;
    params.flags = LLAMA_SPECULATIVE_ENGINE_REQUIRE_CANONICAL_STATE |
            LLAMA_SPECULATIVE_ENGINE_REQUIRE_FIXED_SEED_TARGET_IDENTITY;
    return params;
}

llama_speculative_status llama_speculative_engine_create(
        const llama_speculative_engine_params * params,
        llama_speculative_engine ** out_engine,
        llama_speculative_error ** error) {
    clear_error(error);
    if (out_engine != nullptr) {
        *out_engine = nullptr;
    }
    if (params == nullptr || out_engine == nullptr || params->target_model == nullptr) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "engine params, target model, and output are required");
    }
    if ((params->flags & ~(LLAMA_SPECULATIVE_ENGINE_REQUIRE_CANONICAL_STATE |
            LLAMA_SPECULATIVE_ENGINE_REQUIRE_FIXED_SEED_TARGET_IDENTITY)) != 0) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "unknown engine requirement flag");
    }
    if (params->auxiliary_model_count != 1 || params->auxiliary_models == nullptr) {
        return fail(error, LLAMA_SPECULATIVE_MODEL_INCOMPATIBLE, "speculative bridge v1 requires exactly one auxiliary model");
    }
    const llama_speculative_model_binding & binding = params->auxiliary_models[0];
    if (binding.name == nullptr || std::strcmp(binding.name, "draft") != 0 || binding.model == nullptr || binding.flags != 0) {
        return fail(error, LLAMA_SPECULATIVE_MODEL_INCOMPATIBLE, "speculative bridge v1 requires one unflagged binding named draft");
    }
    if (binding.threads < -1 || binding.batch_threads < -1) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "draft thread counts must be -1, zero, or positive");
    }
    if (params->checkpoint_mode != LLAMA_SPEC_CKPT_AUTO && params->checkpoint_mode != LLAMA_SPEC_CKPT_PER_STEP &&
            params->checkpoint_mode != LLAMA_SPEC_CKPT_GPU_FALLBACK && params->checkpoint_mode != LLAMA_SPEC_CKPT_CPU) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "checkpoint mode is invalid");
    }
    bool draft_reserved = false;
    try {
        auto engine = std::make_unique<llama_speculative_engine>();
        engine->target_model = params->target_model;
        engine->draft_model = binding.model;
        engine->draft_context_size = binding.context_size;
        engine->draft_threads = binding.threads;
        engine->draft_batch_threads = binding.batch_threads;
        engine->checkpoint_mode = params->checkpoint_mode;
        std::string message;
        if (!parse_stage_expression(params->stage_expression, engine->stage, message)) {
            return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, message);
        }
        if (!fill_engine_metadata(*engine, message)) {
            return fail(error, LLAMA_SPECULATIVE_MODEL_INCOMPATIBLE, message);
        }
        if (((params->flags & LLAMA_SPECULATIVE_ENGINE_REQUIRE_CANONICAL_STATE) != 0 &&
             (engine->capabilities.flags & LLAMA_SPECULATIVE_CAP_CANONICAL_STATE) == 0) ||
                ((params->flags & LLAMA_SPECULATIVE_ENGINE_REQUIRE_FIXED_SEED_TARGET_IDENTITY) != 0 &&
                 (engine->capabilities.flags & LLAMA_SPECULATIVE_CAP_FIXED_SEED_TARGET_IDENTITY) == 0)) {
            return fail(error, LLAMA_SPECULATIVE_UNSUPPORTED, "speculative backend does not satisfy engine requirements");
        }

        {
            std::lock_guard<std::mutex> lock(ownership_mutex);
            if (!reserved_draft_models.insert(engine->draft_model).second) {
                return fail(error, LLAMA_SPECULATIVE_BUSY, "draft model is already reserved by another speculative engine");
            }
            draft_reserved = true;
        }
        if (engine->stage.type == bridge_stage_type::DSPARK) {
            if (!llama_model_share_dflash_io_tensors(engine->draft_model, engine->target_model)) {
                release_draft_model(engine->draft_model);
                draft_reserved = false;
                return fail(error, LLAMA_SPECULATIVE_MODEL_INCOMPATIBLE, "could not establish DSpark target IO sharing");
            }
            if (llama_model_dflash_io_mode(engine->draft_model, engine->target_model) == LLAMA_DFLASH_IO_MODE_INVALID ||
                    llama_model_dflash_io_mode(engine->draft_model, engine->target_model) == LLAMA_DFLASH_IO_MODE_MIXED) {
                release_draft_model(engine->draft_model);
                draft_reserved = false;
                return fail(error, LLAMA_SPECULATIVE_MODEL_INCOMPATIBLE, "DSpark target IO sharing is incomplete");
            }
        }

        *out_engine = engine.release();
        draft_reserved = false;
        return LLAMA_SPECULATIVE_OK;
    } catch (const std::bad_alloc &) {
        if (draft_reserved) {
            release_draft_model(binding.model);
        }
        return fail(error, LLAMA_SPECULATIVE_OUT_OF_MEMORY, "could not allocate speculative engine");
    } catch (const std::exception & exception) {
        if (draft_reserved) {
            release_draft_model(binding.model);
        }
        return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, exception.what());
    } catch (...) {
        if (draft_reserved) {
            release_draft_model(binding.model);
        }
        return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, "unknown engine creation failure");
    }
}

llama_speculative_status llama_speculative_engine_destroy(
        llama_speculative_engine ** engine,
        llama_speculative_error ** error) {
    clear_error(error);
    if (engine == nullptr) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "engine output pointer is required");
    }
    if (*engine == nullptr) {
        return LLAMA_SPECULATIVE_OK;
    }
    if ((*engine)->child_count.load() != 0) {
        return fail(error, LLAMA_SPECULATIVE_BUSY, "speculative engine still has live sessions or samplers");
    }
    release_draft_model((*engine)->draft_model);
    delete *engine;
    *engine = nullptr;
    return LLAMA_SPECULATIVE_OK;
}

llama_speculative_status llama_speculative_engine_get_capabilities(
        const llama_speculative_engine * engine,
        llama_speculative_engine_capabilities * capabilities,
        llama_speculative_error ** error) {
    clear_error(error);
    if (engine == nullptr || capabilities == nullptr) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "engine and capabilities output are required");
    }
    *capabilities = engine->capabilities;
    return LLAMA_SPECULATIVE_OK;
}

llama_speculative_status llama_speculative_engine_get_state_layout_id(
        const llama_speculative_engine * engine,
        uint8_t state_layout_id[LLAMA_SPECULATIVE_STATE_LAYOUT_ID_SIZE],
        llama_speculative_error ** error) {
    clear_error(error);
    if (engine == nullptr || state_layout_id == nullptr) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "engine and state-layout output are required");
    }
    std::memcpy(state_layout_id, engine->state_layout_id.data(), engine->state_layout_id.size());
    return LLAMA_SPECULATIVE_OK;
}

llama_speculative_status llama_speculative_engine_copy_stage_expression(
        const llama_speculative_engine * engine,
        char * destination,
        uint64_t capacity,
        uint64_t * required_size,
        llama_speculative_error ** error) {
    clear_error(error);
    if (required_size != nullptr) {
        *required_size = 0;
    }
    if (engine == nullptr || required_size == nullptr) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "engine and required-size output are required");
    }
    *required_size = (uint64_t) engine->stage.normalized.size() + 1;
    if (destination == nullptr || capacity < *required_size) {
        return LLAMA_SPECULATIVE_BUFFER_TOO_SMALL;
    }
    std::memcpy(destination, engine->stage.normalized.c_str(), (size_t) *required_size);
    return LLAMA_SPECULATIVE_OK;
}

llama_speculative_sampler_params llama_speculative_sampler_default_params(void) {
    llama_speculative_sampler_params params = {};
    params.seed = LLAMA_DEFAULT_SEED;
    params.top_k = 40;
    params.penalty_last_n = 64;
    params.top_p = 0.95f;
    params.min_p = 0.05f;
    params.temperature = 0.8f;
    params.repeat_penalty = 1.0f;
    params.flags = LLAMA_SPECULATIVE_SAMPLER_SUPPRESS_NON_EOG_CONTROL;
    return params;
}

llama_speculative_status llama_speculative_sampler_create(
        const llama_speculative_engine * const_engine,
        const llama_speculative_sampler_params * params,
        llama_speculative_sampler ** out_sampler,
        llama_speculative_error ** error) {
    clear_error(error);
    if (out_sampler != nullptr) {
        *out_sampler = nullptr;
    }
    if (const_engine == nullptr || params == nullptr || out_sampler == nullptr) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "engine, sampler params, and output are required");
    }
    if ((params->flags & ~(LLAMA_SPECULATIVE_SAMPLER_FIXED_SEED |
            LLAMA_SPECULATIVE_SAMPLER_SUPPRESS_NON_EOG_CONTROL)) != 0) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "unknown sampler flag");
    }
    if ((params->flags & LLAMA_SPECULATIVE_SAMPLER_FIXED_SEED) != 0 &&
            params->seed == LLAMA_DEFAULT_SEED) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "fixed-seed sampling requires an explicit seed");
    }
    if (params->top_k < -1 || params->penalty_last_n < -1 ||
            (!std::isnan(params->top_p) && !std::isfinite(params->top_p)) ||
            (!std::isnan(params->min_p) && !std::isfinite(params->min_p)) ||
            (!std::isnan(params->temperature) && !std::isfinite(params->temperature)) ||
            !std::isfinite(params->repeat_penalty) || params->repeat_penalty <= 0.0f ||
            !std::isfinite(params->frequency_penalty) || !std::isfinite(params->presence_penalty)) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "sampler parameters are invalid");
    }
    try {
        auto * engine = const_cast<llama_speculative_engine *>(const_engine);
        auto sampler = std::make_unique<llama_speculative_sampler>();
        sampler->engine = engine;
        sampler->params = *params;
        const int32_t vocabulary_size = llama_vocab_n_tokens(llama_model_get_vocab(engine->target_model));
        if (vocabulary_size <= 0) {
            return fail(error, LLAMA_SPECULATIVE_MODEL_INCOMPATIBLE, "target vocabulary is empty");
        }
        sampler->candidates.resize((size_t) vocabulary_size);
        sampler->control_bias.assign((size_t) vocabulary_size, 0.0f);
        sampler->probabilities.resize((size_t) vocabulary_size);
        if ((params->flags & LLAMA_SPECULATIVE_SAMPLER_SUPPRESS_NON_EOG_CONTROL) != 0) {
            for (llama_token token = 0; token < vocabulary_size; ++token) {
                if (llama_token_is_control(engine->target_model, token) && !llama_token_is_eog(engine->target_model, token)) {
                    sampler->control_bias[(size_t) token] = -INFINITY;
                }
            }
        }
        initialize_sampler_rng(*sampler);
        engine->child_count.fetch_add(1);
        *out_sampler = sampler.release();
        return LLAMA_SPECULATIVE_OK;
    } catch (const std::bad_alloc &) {
        return fail(error, LLAMA_SPECULATIVE_OUT_OF_MEMORY, "could not allocate speculative sampler");
    } catch (const std::exception & exception) {
        return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, exception.what());
    } catch (...) {
        return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, "unknown sampler creation failure");
    }
}

llama_speculative_status llama_speculative_sampler_reset(
        llama_speculative_sampler * sampler,
        llama_speculative_error ** error) {
    clear_error(error);
    if (sampler == nullptr) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "sampler is required");
    }
    if (sampler->bound || sampler->probe_open) {
        return fail(error, LLAMA_SPECULATIVE_BUSY, "sampler has an open session or probe transaction");
    }
    sampler->canonical.accepted.clear();
    initialize_sampler_rng(*sampler);
    return LLAMA_SPECULATIVE_OK;
}

void llama_speculative_sampler_destroy(llama_speculative_sampler * sampler) {
    if (sampler == nullptr) {
        return;
    }
    llama_speculative_engine * engine = sampler->engine;
    delete sampler;
    engine->child_count.fetch_sub(1);
}

llama_speculative_status llama_speculative_sampler_probe(
        llama_speculative_sampler * sampler,
        llama_context * target_context,
        int32_t output_index,
        llama_token * token,
        llama_speculative_error ** error) {
    clear_error(error);
    if (token != nullptr) {
        *token = LLAMA_TOKEN_NULL;
    }
    if (sampler == nullptr || target_context == nullptr || token == nullptr) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "sampler, target context, and token output are required");
    }
    if (sampler->bound || sampler->probe_open) {
        return fail(error, LLAMA_SPECULATIVE_BUSY, "sampler already has an open transaction");
    }
    if (llama_get_model(target_context) != sampler->engine->target_model) {
        return fail(error, LLAMA_SPECULATIVE_MODEL_INCOMPATIBLE, "target context belongs to another model");
    }
    try {
        sampler->canonical.accepted.reserve(sampler->canonical.accepted.size() + 1);
        sampler_state provisional;
        provisional.rng = sampler->canonical.rng;
        provisional.accepted = sampler->canonical.accepted;
        sampler->probe_token = sample_token(*sampler, provisional, target_context, output_index);
        if (sampler->probe_token == LLAMA_TOKEN_NULL) {
            return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, "target logits are unavailable");
        }
        provisional.accepted.push_back(sampler->probe_token);
        sampler->probe_rng = provisional.rng;
        sampler->probe_open = true;
        *token = sampler->probe_token;
        return LLAMA_SPECULATIVE_OK;
    } catch (const std::bad_alloc &) {
        return fail(error, LLAMA_SPECULATIVE_OUT_OF_MEMORY, "could not prepare sampler probe");
    } catch (const std::exception & exception) {
        return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, exception.what());
    } catch (...) {
        return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, "unknown sampler probe failure");
    }
}

llama_speculative_status llama_speculative_sampler_probe_commit(
        llama_speculative_sampler * sampler,
        llama_speculative_error ** error) {
    clear_error(error);
    if (sampler == nullptr || !sampler->probe_open) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "sampler has no open probe");
    }
    sampler->canonical.accepted.push_back(sampler->probe_token);
    sampler->canonical.rng = sampler->probe_rng;
    sampler->probe_open = false;
    sampler->probe_token = LLAMA_TOKEN_NULL;
    return LLAMA_SPECULATIVE_OK;
}

llama_speculative_status llama_speculative_sampler_probe_abort(
        llama_speculative_sampler * sampler,
        llama_speculative_error ** error) {
    clear_error(error);
    if (sampler == nullptr || !sampler->probe_open) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "sampler has no open probe");
    }
    sampler->probe_open = false;
    sampler->probe_token = LLAMA_TOKEN_NULL;
    return LLAMA_SPECULATIVE_OK;
}

llama_speculative_status llama_speculative_session_create(
        llama_speculative_engine * engine,
        const llama_speculative_session_params * params,
        llama_speculative_session ** out_session,
        llama_speculative_call_result * result,
        llama_speculative_error ** error) {
    clear_error(error);
    if (out_session != nullptr) {
        *out_session = nullptr;
    }
    const llama_pos initial_position = params != nullptr ? params->next_position : 0;
    init_call_result(result, initial_position);
    if (engine == nullptr || params == nullptr || out_session == nullptr || result == nullptr || params->target_context == nullptr) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "engine, session params, target context, result, and output are required");
    }
    if (params->sequence_start_position < 0 || params->next_position < params->sequence_start_position) {
        return fail_call(result, error, LLAMA_SPECULATIVE_INVALID_ARGUMENT,
                "invalid target sequence boundary", initial_position);
    }
    if (llama_get_model(params->target_context) != engine->target_model) {
        return fail_call(result, error, LLAMA_SPECULATIVE_MODEL_INCOMPATIBLE,
                "target context belongs to another model", initial_position);
    }
    if (engine->stage.type == bridge_stage_type::MTP &&
            (!llama_mtp_state_enabled(params->target_context) ||
             llama_mtp_state_n_embd(params->target_context) != (uint32_t) engine->feature_width)) {
        return fail_call(result, error, LLAMA_SPECULATIVE_MODEL_INCOMPATIBLE,
                "MTP target context was not created with llama_context_params.mtp=true", initial_position);
    }
    if (llama_n_seq_max(params->target_context) < engine->capabilities.required_target_sequence_count ||
            llama_n_batch(params->target_context) < engine->capabilities.minimum_target_batch_tokens ||
            llama_n_ubatch(params->target_context) < engine->capabilities.minimum_target_ubatch_tokens) {
        return fail_call(result, error, LLAMA_SPECULATIVE_UNSUPPORTED,
                "target context does not satisfy engine requirements", initial_position);
    }

    bool target_reserved = false;
    try {
        {
            std::lock_guard<std::mutex> lock(ownership_mutex);
            target_reserved = attached_target_contexts.insert(params->target_context).second;
        }
        if (!target_reserved) {
            return fail_call(result, error, LLAMA_SPECULATIVE_BUSY,
                    "target context already has a speculative session", initial_position);
        }
        auto session = std::make_unique<llama_speculative_session>();
        session->engine = engine;
        session->target_context = params->target_context;
        session->sequence_start_position = params->sequence_start_position;
        session->next_position = params->next_position;
        session->owner_thread = std::this_thread::get_id();
        session->authoritative_logits.resize((size_t) llama_vocab_n_tokens(llama_model_get_vocab(engine->target_model)));

        common_params_speculative common_params = make_common_params(*engine, params->target_context);
        session->speculative = engine->stage.type == bridge_stage_type::MTP
            ? common_speculative_init(common_params, params->target_context)
            : common_speculative_init_prepared(common_params, params->target_context);
        if (session->speculative == nullptr) {
            llama_spec_ckpt_release(params->target_context);
            release_target_context(params->target_context);
            return fail_call(result, error, LLAMA_SPECULATIVE_MODEL_INCOMPATIBLE,
                    "could not initialize speculative session state", initial_position);
        }
        if (engine->stage.type == bridge_stage_type::MTP) {
            const llama_context * companion_context =
                common_speculative_get_companion_ctx(session->speculative);
            if (companion_context == nullptr ||
                    llama_n_ctx(companion_context) < llama_n_ctx(params->target_context)) {
                common_speculative_free(session->speculative);
                session->speculative = nullptr;
                llama_spec_ckpt_release(params->target_context);
                release_target_context(params->target_context);
                return fail_call(result, error, LLAMA_SPECULATIVE_UNSUPPORTED,
                        "MTP companion context is smaller than the target context", initial_position);
            }
        }
        const bool state_reset = engine->stage.type == bridge_stage_type::MTP
            ? common_speculative_mtp_state_reset(session->speculative, 0)
            : common_speculative_dflash_state_reset(
                session->speculative,
                params->sequence_start_position,
                params->next_position,
                params->next_position);
        if (!state_reset) {
            common_speculative_free(session->speculative);
            session->speculative = nullptr;
            llama_spec_ckpt_release(params->target_context);
            release_target_context(params->target_context);
            return fail_call(result, error, LLAMA_SPECULATIVE_INTERNAL_ERROR,
                    "could not establish empty speculative conditioning state", initial_position);
        }
        session->mtp_prefix_complete = engine->stage.type == bridge_stage_type::MTP &&
            params->next_position == params->sequence_start_position;
        session->attached = true;
        engine->child_count.fetch_add(1);
        *out_session = session.release();
        return finish_call(result, LLAMA_SPECULATIVE_OK, LLAMA_SPECULATIVE_EFFECT_UNCHANGED,
                LLAMA_SPECULATIVE_BOUNDARY_POST_CALL_COMMITTED, 0, params->next_position);
    } catch (const std::bad_alloc &) {
        if (target_reserved) {
            llama_spec_ckpt_release(params->target_context);
            release_target_context(params->target_context);
        }
        return fail_call(result, error, LLAMA_SPECULATIVE_OUT_OF_MEMORY,
                "could not allocate speculative session", initial_position);
    } catch (const std::exception & exception) {
        if (target_reserved) {
            llama_spec_ckpt_release(params->target_context);
            release_target_context(params->target_context);
        }
        return fail_call(result, error, LLAMA_SPECULATIVE_INTERNAL_ERROR, exception.what(), initial_position);
    } catch (...) {
        if (target_reserved) {
            llama_spec_ckpt_release(params->target_context);
            release_target_context(params->target_context);
        }
        return fail_call(result, error, LLAMA_SPECULATIVE_INTERNAL_ERROR,
                "unknown session creation failure", initial_position);
    }
}

llama_speculative_status llama_speculative_session_quiesce_and_detach(
        llama_speculative_session * session,
        llama_speculative_call_result * result,
        llama_speculative_error ** error) {
    const llama_speculative_status checked = check_session_call(session, result, error);
    if (checked != LLAMA_SPECULATIVE_OK) {
        return checked;
    }
    if (session->round_open) {
        return fail_call(result, error, LLAMA_SPECULATIVE_BUSY,
                "cannot detach with an open round", session->next_position);
    }
    if (session->generation_open) {
        const llama_speculative_status status = llama_speculative_session_generation_end(session, result, error);
        if (status != LLAMA_SPECULATIVE_OK) {
            return status;
        }
    }
    if (session->has_authoritative_logits) {
        // The core output buffer setter is added with the bridge checkpoint primitives.
        bool restored = false;
        try {
            restored = llama_spec_restore_logits(session->target_context, session->authoritative_logits.data(),
                session->authoritative_logits.size());
        } catch (...) {
            restored = false;
        }
        if (!restored) {
            poison_session(*session, result, LLAMA_SPECULATIVE_POISONED);
            return fail(error, LLAMA_SPECULATIVE_POISONED, "could not materialize authoritative target logits");
        }
        session->has_authoritative_logits = false;
    }
    common_speculative_free(session->speculative);
    session->speculative = nullptr;
    llama_spec_ckpt_release(session->target_context);
    session->attached = false;
    release_target_context(session->target_context);
    return finish_call(result, LLAMA_SPECULATIVE_OK,
            LLAMA_SPECULATIVE_EFFECT_SESSION_DETACHED_TARGET_VALID,
            LLAMA_SPECULATIVE_BOUNDARY_POST_CALL_COMMITTED, 0, session->next_position);
}

void llama_speculative_session_destroy(llama_speculative_session * session) {
    if (session == nullptr) {
        return;
    }
    if (session->active_round != nullptr) {
        llama_speculative_round * round = session->active_round;
        if (round->checkpoint_saved) {
            try {
                llama_spec_ckpt_discard(session->target_context);
            } catch (...) {
            }
            round->checkpoint_saved = false;
        }
        finish_round(&round);
    }
    if (session->sampler != nullptr) {
        session->sampler->bound = false;
        session->sampler = nullptr;
    }
    if (session->speculative != nullptr) {
        common_speculative_free(session->speculative);
        session->speculative = nullptr;
    }
    if (session->attached) {
        release_target_context(session->target_context);
    }
    llama_speculative_engine * engine = session->engine;
    delete session;
    engine->child_count.fetch_sub(1);
}

llama_speculative_status llama_speculative_session_generation_begin(
        llama_speculative_session * session,
        const llama_speculative_generation_begin_params * params,
        llama_speculative_call_result * result,
        llama_speculative_error ** error) {
    const llama_speculative_status checked = check_session_call(session, result, error);
    if (checked != LLAMA_SPECULATIVE_OK) {
        return checked;
    }
    if (params == nullptr || params->sampler == nullptr) {
        return fail_call(result, error, LLAMA_SPECULATIVE_INVALID_ARGUMENT,
                "generation begin params and sampler are required", session->next_position);
    }
    if (session->generation_open || session->round_open || session->private_carry.ready ||
            params->sampler->bound || params->sampler->probe_open || params->sampler->engine != session->engine) {
        return fail_call(result, error, LLAMA_SPECULATIVE_BUSY,
                "session or sampler already has an open transaction", session->next_position);
    }
    const llama_speculative_token_history_view & history = params->history;
    if ((params->flags & ~LLAMA_SPECULATIVE_GENERATION_ALLOW_TARGET_ONLY_HISTORY_WARMUP) != 0 ||
            (history.flags & ~LLAMA_SPECULATIVE_HISTORY_COMPLETE_FROM_SEQUENCE_START) != 0 ||
            history.sequence_start_position != session->sequence_start_position || history.next_position != session->next_position ||
            history.history_start_position < history.sequence_start_position || history.history_start_position > history.next_position ||
            history.token_count != (uint64_t) (history.next_position - history.history_start_position) ||
            (history.token_count != 0 && history.tokens == nullptr) ||
            ((history.flags & LLAMA_SPECULATIVE_HISTORY_COMPLETE_FROM_SEQUENCE_START) != 0 &&
             history.history_start_position != history.sequence_start_position)) {
        return fail_call(result, error, LLAMA_SPECULATIVE_INVALID_ARGUMENT,
                "generation token-history view does not match the target boundary", session->next_position);
    }
    if (session->engine->stage.type == bridge_stage_type::DSPARK) {
        try {
            const llama_tokens prompt;
            common_speculative_begin(session->speculative, prompt);
        } catch (const std::bad_alloc &) {
            return fail_call(result, error, LLAMA_SPECULATIVE_OUT_OF_MEMORY,
                    "could not begin speculative generation", session->next_position);
        } catch (const std::exception & exception) {
            return fail_call(result, error, LLAMA_SPECULATIVE_INTERNAL_ERROR,
                    exception.what(), session->next_position);
        } catch (...) {
            return fail_call(result, error, LLAMA_SPECULATIVE_INTERNAL_ERROR,
                    "unknown generation-begin failure", session->next_position);
        }
    }
    session->sampler = params->sampler;
    session->sampler->bound = true;
    session->generation_open = true;
    const llama_speculative_status status = session->engine->stage.type == bridge_stage_type::MTP &&
            (!session->mtp_prefix_complete ||
             !common_speculative_has_sequence_hidden(session->speculative, 0))
        ? LLAMA_SPECULATIVE_HISTORY_INSUFFICIENT
        : LLAMA_SPECULATIVE_OK;
    return finish_call(result, status, LLAMA_SPECULATIVE_EFFECT_UNCHANGED,
            LLAMA_SPECULATIVE_BOUNDARY_POST_CALL_COMMITTED, 0, session->next_position);
}

llama_speculative_status llama_speculative_session_generation_synchronize(
        llama_speculative_session * session,
        llama_speculative_call_result * result,
        llama_speculative_error ** error) {
    const llama_speculative_status checked = check_session_call(session, result, error);
    if (checked != LLAMA_SPECULATIVE_OK) {
        return checked;
    }
    if (!session->generation_open || session->round_open) {
        return fail_call(result, error, LLAMA_SPECULATIVE_BUSY,
                "generation is not open or has an open round", session->next_position);
    }
    llama_speculative_internal::sampler_token_carry_discard(session->private_carry);
    return finish_call(result, LLAMA_SPECULATIVE_OK, LLAMA_SPECULATIVE_EFFECT_UNCHANGED,
            LLAMA_SPECULATIVE_BOUNDARY_POST_CALL_COMMITTED, 0, session->next_position);
}

llama_speculative_status llama_speculative_session_generation_end(
        llama_speculative_session * session,
        llama_speculative_call_result * result,
        llama_speculative_error ** error) {
    const llama_speculative_status checked = check_session_call(session, result, error);
    if (checked != LLAMA_SPECULATIVE_OK) {
        return checked;
    }
    if (!session->generation_open || session->round_open || session->sampler == nullptr) {
        return fail_call(result, error, LLAMA_SPECULATIVE_BUSY,
                "generation is not open or has an open round", session->next_position);
    }
    llama_speculative_internal::sampler_token_carry_discard(session->private_carry);
    session->sampler->bound = false;
    session->sampler = nullptr;
    session->generation_open = false;
    return finish_call(result, LLAMA_SPECULATIVE_OK, LLAMA_SPECULATIVE_EFFECT_UNCHANGED,
            LLAMA_SPECULATIVE_BOUNDARY_POST_CALL_COMMITTED, 0, session->next_position);
}

llama_speculative_status llama_speculative_session_decode(
        llama_speculative_session * session,
        const llama_speculative_decode_params * params,
        llama_speculative_call_result * result,
        llama_speculative_error ** error) {
    const llama_speculative_status checked = check_session_call(session, result, error);
    if (checked != LLAMA_SPECULATIVE_OK) {
        return checked;
    }
    if (params == nullptr || params->batch == nullptr || params->expected_position != session->next_position ||
            params->flags != 0 || params->phase < LLAMA_SPECULATIVE_DECODE_PROMPT ||
            params->phase > LLAMA_SPECULATIVE_DECODE_GENERATED_REPLAY || session->generation_open || session->round_open) {
        return fail_call(result, error, LLAMA_SPECULATIVE_INVALID_ARGUMENT,
                "decode params, expected position, or session state is invalid", session->next_position);
    }
    llama_pos resulting_position = session->next_position;
    materialized_primary_batch materialized;
    try {
        std::string message;
        if (!materialize_primary_batch(
                *params->batch,
                params->expected_position,
                llama_n_batch(session->target_context),
                resulting_position,
                materialized,
                message)) {
            return fail_call(result, error, LLAMA_SPECULATIVE_INVALID_ARGUMENT,
                    message, session->next_position);
        }
    } catch (const std::bad_alloc &) {
        return fail_call(result, error, LLAMA_SPECULATIVE_OUT_OF_MEMORY,
                "could not validate target decode batch", session->next_position);
    } catch (...) {
        return fail_call(result, error, LLAMA_SPECULATIVE_INTERNAL_ERROR,
                "unknown target batch validation failure", session->next_position);
    }
    int decode_result = -1;
    try {
        decode_result = llama_decode(session->target_context, materialized.batch);
    } catch (...) {
        decode_result = -1;
    }
    if (decode_result != 0) {
        poison_session(*session, result, LLAMA_SPECULATIVE_DECODE_FAILED);
        return fail(error, LLAMA_SPECULATIVE_DECODE_FAILED, "target decode failed with an unprovable boundary");
    }
    session->next_position = resulting_position;
    session->has_authoritative_logits = false;
    session->metrics.target_decode_calls++;
    session->metrics.target_decode_tokens += (uint64_t) params->batch->n_tokens;
    bool capture_failed = false;
    try {
        const bool prompt_warmup = params->phase != LLAMA_SPECULATIVE_DECODE_GENERATED_REPLAY;
        capture_failed = common_speculative_on_target_seq_batch(
            session->speculative, session->target_context, materialized.batch, 0, prompt_warmup) != 0;
        if (!capture_failed && session->engine->stage.type == bridge_stage_type::MTP) {
            capture_failed = !common_speculative_mtp_discard_draft_tail(
                session->speculative, 0, session->next_position);
        }
    } catch (...) {
        capture_failed = true;
    }
    if (capture_failed) {
        bool reset = false;
        try {
            reset = reset_session_conditioning(*session, session->next_position);
        } catch (...) {
            reset = false;
        }
        if (!reset) {
            poison_session(*session, result, LLAMA_SPECULATIVE_POISONED);
            return fail(error, LLAMA_SPECULATIVE_POISONED,
                    "target decode committed, but speculative state could not be reset at the committed boundary");
        }
        set_error(error, LLAMA_SPECULATIVE_OK, "target decode committed, but speculative conditioning was cleared");
        return finish_call(result, LLAMA_SPECULATIVE_OK,
                LLAMA_SPECULATIVE_EFFECT_SPEC_STATE_CLEARED_TARGET_VALID,
                LLAMA_SPECULATIVE_BOUNDARY_POST_CALL_COMMITTED,
                (uint64_t) params->batch->n_tokens,
                session->next_position);
    }
    return finish_call(result, LLAMA_SPECULATIVE_OK, LLAMA_SPECULATIVE_EFFECT_UNCHANGED,
            LLAMA_SPECULATIVE_BOUNDARY_POST_CALL_COMMITTED,
            (uint64_t) params->batch->n_tokens,
            session->next_position);
}

llama_speculative_status llama_speculative_session_clear_conditioning(
        llama_speculative_session * session,
        llama_pos expected_position,
        llama_speculative_call_result * result,
        llama_speculative_error ** error) {
    const llama_speculative_status checked = check_session_call(session, result, error);
    if (checked != LLAMA_SPECULATIVE_OK) {
        return checked;
    }
    if (expected_position != session->next_position || session->generation_open || session->round_open) {
        return fail_call(result, error, LLAMA_SPECULATIVE_INVALID_ARGUMENT,
                "conditioning boundary does not match the session", session->next_position);
    }
    bool reset = false;
    try {
        reset = reset_session_conditioning(*session, session->next_position);
    } catch (...) {
        reset = false;
    }
    if (!reset) {
        poison_session(*session, result, LLAMA_SPECULATIVE_POISONED);
        return fail(error, LLAMA_SPECULATIVE_POISONED,
                "speculative conditioning could not be reset at the committed boundary");
    }
    return finish_call(result, LLAMA_SPECULATIVE_OK,
            LLAMA_SPECULATIVE_EFFECT_SPEC_STATE_CLEARED_TARGET_VALID,
            LLAMA_SPECULATIVE_BOUNDARY_POST_CALL_COMMITTED, 0, session->next_position);
}

llama_speculative_status llama_speculative_session_trim(
        llama_speculative_session * session,
        llama_pos expected_position,
        llama_pos new_next_position,
        llama_speculative_call_result * result,
        llama_speculative_error ** error) {
    const llama_speculative_status checked = check_session_call(session, result, error);
    if (checked != LLAMA_SPECULATIVE_OK) {
        return checked;
    }
    if (expected_position != session->next_position || new_next_position < session->sequence_start_position ||
            new_next_position > session->next_position || session->generation_open || session->round_open) {
        return fail_call(result, error, LLAMA_SPECULATIVE_INVALID_ARGUMENT,
                "invalid trim boundary or active generation", session->next_position);
    }
    return fail_call(result, error, LLAMA_SPECULATIVE_UNSUPPORTED,
            "speculative bridge v1 does not support transactional target-context trimming", session->next_position);
}

llama_speculative_status llama_speculative_session_context_shift(
        llama_speculative_session * session,
        llama_pos expected_start_position,
        llama_pos expected_next_position,
        llama_pos keep_position,
        llama_pos discard_count,
        llama_speculative_call_result * result,
        llama_speculative_error ** error) {
    const llama_speculative_status checked = check_session_call(session, result, error);
    if (checked != LLAMA_SPECULATIVE_OK) {
        return checked;
    }
    if (expected_start_position != session->sequence_start_position || expected_next_position != session->next_position ||
            keep_position < session->sequence_start_position || keep_position > session->next_position ||
            discard_count <= 0 || discard_count > session->next_position - keep_position ||
            session->generation_open || session->round_open) {
        return fail_call(result, error, LLAMA_SPECULATIVE_INVALID_ARGUMENT,
                "invalid context-shift boundary or active generation", session->next_position);
    }
    return fail_call(result, error, LLAMA_SPECULATIVE_UNSUPPORTED,
            "speculative bridge v1 does not support transactional target-context shifting", session->next_position);
}

llama_speculative_status llama_speculative_round_begin(
        llama_speculative_session * session,
        const llama_speculative_round_params * params,
        llama_speculative_round ** out_round,
        llama_speculative_call_result * result,
        llama_speculative_error ** error) {
    const llama_speculative_status checked = check_session_call(session, result, error);
    if (out_round != nullptr) {
        *out_round = nullptr;
    }
    if (checked != LLAMA_SPECULATIVE_OK) {
        return checked;
    }
    if (params == nullptr || out_round == nullptr || !session->generation_open || session->sampler == nullptr ||
            session->round_open || session->sampler->probe_open || params->cooperative_token_allowance == 0 ||
            (params->flags & ~LLAMA_SPECULATIVE_ROUND_FORCE_TARGET_ONLY) != 0) {
        return fail_call(result, error, LLAMA_SPECULATIVE_INVALID_ARGUMENT,
                "round params or session state is invalid", session->next_position);
    }

    const llama_pos initial_position = session->next_position;
    std::unique_ptr<llama_speculative_round> prepared;
    const auto fail_pre_call = [&](llama_speculative_status status, const char * message) {
        llama_speculative_effect effect = LLAMA_SPECULATIVE_EFFECT_UNCHANGED;
        if (prepared != nullptr) {
            const companion_tail_result tail_result = round_discard_companion_tail(*prepared);
            if (tail_result == companion_tail_result::FAILED) {
                poison_session(*session, result, LLAMA_SPECULATIVE_INTERNAL_ERROR);
                return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR,
                        "speculative round failed and the MTP draft tail could not be discarded");
            }
            if (tail_result == companion_tail_result::CLEARED) {
                effect = LLAMA_SPECULATIVE_EFFECT_SPEC_STATE_CLEARED_TARGET_VALID;
            }
        }
        finish_call(result, status, effect,
                LLAMA_SPECULATIVE_BOUNDARY_PRE_CALL, 0, initial_position);
        return fail(error, status, message);
    };

    try {
        prepared = std::make_unique<llama_speculative_round>();
        prepared->session = session;
        prepared->params = *params;
        prepared->view.initial_position = initial_position;
        prepared->view.provisional_position = initial_position;
        prepared->view.selected_stage_index = 0;

        const size_t vocabulary_size = session->authoritative_logits.size();
        const float * boundary_logits = session->has_authoritative_logits
            ? session->authoritative_logits.data()
            : llama_get_logits_ith(session->target_context, -1);
        if (boundary_logits == nullptr) {
            return fail_pre_call(LLAMA_SPECULATIVE_INVALID_ARGUMENT, "target boundary logits are unavailable");
        }
        prepared->previous_logits.assign(boundary_logits, boundary_logits + vocabulary_size);

        uint64_t maximum_draft = (uint64_t) session->engine->stage.n_max;
        maximum_draft = std::min(maximum_draft, params->cooperative_token_allowance - 1);
        if (params->generation_token_allowance != 0) {
            maximum_draft = std::min(maximum_draft, params->generation_token_allowance - 1);
        }
        if (params->context_token_allowance != 0) {
            maximum_draft = std::min(maximum_draft, params->context_token_allowance - 1);
        }
        maximum_draft = std::min(maximum_draft, (uint64_t) llama_n_batch(session->target_context) - 1);
        if ((session->sampler->params.flags & LLAMA_SPECULATIVE_SAMPLER_FIXED_SEED) != 0 ||
                (params->flags & LLAMA_SPECULATIVE_ROUND_FORCE_TARGET_ONLY) != 0) {
            // Multi-token verification can use numerically different target kernels. Preserve
            // fixed-seed target-stream identity through the canonical one-token path.
            maximum_draft = 0;
        }
        if (session->engine->stage.type == bridge_stage_type::MTP &&
                (!session->mtp_prefix_complete ||
                 !common_speculative_has_sequence_hidden(session->speculative, 0))) {
            maximum_draft = 0;
        }
        if (maximum_draft > (uint64_t) std::numeric_limits<int32_t>::max()) {
            return fail_pre_call(LLAMA_SPECULATIVE_INTERNAL_ERROR, "draft-token bound exceeds int32_t");
        }

        const size_t maximum_outputs = (size_t) maximum_draft + 1;
        if (!position_add_fits(initial_position, maximum_outputs)) {
            return fail_pre_call(LLAMA_SPECULATIVE_INVALID_ARGUMENT,
                    "speculative round would overflow the target position type");
        }
        session->sampler->canonical.accepted.reserve(
            session->sampler->canonical.accepted.size() + maximum_outputs);
        prepared->provisional = session->sampler->canonical;
        prepared->provisional.accepted.reserve(prepared->provisional.accepted.size() + maximum_outputs);
        prepared->committable_tokens.reserve(maximum_outputs);
        prepared->commit_ids.reserve(maximum_outputs);
        prepared->prefix_rng.reserve(maximum_outputs + 1);
        prepared->prefix_rng.push_back(session->sampler->canonical.rng);

        llama_token first = LLAMA_TOKEN_NULL;
        // A carry stores the post-sample RNG; its one-token accepted-history delta is
        // reconstructed by the unconditional accepted push below.
        if (!llama_speculative_internal::sampler_token_carry_apply(
                    session->private_carry, first, prepared->provisional.rng)) {
            first = sample_token(
                *session->sampler, prepared->provisional, session->target_context, -1, boundary_logits);
        }
        if (first == LLAMA_TOKEN_NULL) {
            return fail_pre_call(LLAMA_SPECULATIVE_INTERNAL_ERROR, "could not sample target boundary logits");
        }
        prepared->view.terminal = candidate_terminal(*prepared, first, 0);
        if (prepared->view.terminal != LLAMA_SPECULATIVE_TERMINAL_NONE) {
            prepared->view.flags = LLAMA_SPECULATIVE_ROUND_ATTEMPTED |
                    LLAMA_SPECULATIVE_ROUND_TARGET_ONLY;
            session->metrics.target_only_rounds++;
            if ((params->flags & LLAMA_SPECULATIVE_ROUND_FORCE_TARGET_ONLY) != 0) {
                prepared->view.flags |= LLAMA_SPECULATIVE_ROUND_RECOVERED_FALLBACK;
                session->metrics.recoverable_fallbacks++;
            }
            prepared->view.tokens = nullptr;
            session->round_open = true;
            session->metrics.rounds_attempted++;
            *out_round = prepared.release();
            session->active_round = *out_round;
            return finish_call(result, LLAMA_SPECULATIVE_OK, LLAMA_SPECULATIVE_EFFECT_UNCHANGED,
                    LLAMA_SPECULATIVE_BOUNDARY_PRE_CALL, 0, initial_position);
        }

        prepared->provisional.accepted.push_back(first);
        prepared->committable_tokens.push_back(first);
        prepared->prefix_rng.push_back(prepared->provisional.rng);

        common_params_speculative common_params = make_common_params(*session->engine, session->target_context);
        common_params.n_max = (int32_t) maximum_draft;
        for (auto & stage : common_params.stages) {
            stage.n_max = (int32_t) maximum_draft;
        }
        common_speculative_draft_result draft_result = {};
        if (maximum_draft != 0) {
            prepared->companion_draft_tail_open =
                session->engine->stage.type == bridge_stage_type::MTP;
            const llama_tokens empty_history;
            draft_result = common_speculative_draft_ex(
                session->speculative,
                session->target_context,
                common_params,
                empty_history,
                first,
                initial_position,
                0);
        }
        llama_tokens & draft = draft_result.tokens;
        prepared->companion_boundary_row_retained =
            prepared->companion_draft_tail_open && !draft.empty();
        if (draft.size() > maximum_draft) {
            return fail_pre_call(LLAMA_SPECULATIVE_INTERNAL_ERROR, "speculative backend returned too many draft tokens");
        }

        const size_t output_count = draft.size() + 1;
        uint64_t logit_floats = 0;
        uint64_t hidden_floats = 0;
        if (!checked_multiply((uint64_t) output_count, (uint64_t) vocabulary_size, logit_floats) ||
                !checked_multiply((uint64_t) output_count, (uint64_t) session->engine->feature_width, hidden_floats) ||
                logit_floats > SIZE_MAX || hidden_floats > SIZE_MAX) {
            return fail_pre_call(LLAMA_SPECULATIVE_OUT_OF_MEMORY, "round output staging size overflows");
        }
        prepared->prefix_logits.resize((size_t) logit_floats);
        prepared->hidden_rows.reserve((size_t) hidden_floats);
        prepared->output_indices.resize(output_count);
        prepared->verify_batch = llama_batch_init((int32_t) output_count, 0, 1);
        prepared->verify_owned_positions = prepared->verify_batch.pos;
        if (prepared->verify_batch.token == nullptr || prepared->verify_batch.pos == nullptr ||
                prepared->verify_batch.n_seq_id == nullptr || prepared->verify_batch.seq_id == nullptr ||
                prepared->verify_batch.logits == nullptr) {
            return fail_pre_call(LLAMA_SPECULATIVE_OUT_OF_MEMORY, "could not allocate target verification batch");
        }
        prepared->verify_scalar_positions.resize(output_count);
        prepared->verify_positions.resize(output_count * 4);
        prepared->verify_batch.pos = prepared->verify_positions.data();
        for (size_t i = 0; i < output_count; ++i) {
            if (prepared->verify_batch.seq_id[i] == nullptr) {
                return fail_pre_call(LLAMA_SPECULATIVE_OUT_OF_MEMORY,
                        "could not allocate target verification sequence IDs");
            }
            prepared->verify_batch.token[i] = i == 0 ? first : draft[i - 1];
            prepared->verify_scalar_positions[i] = initial_position + (llama_pos) i;
            prepared->verify_batch.n_seq_id[i] = 1;
            prepared->verify_batch.seq_id[i][0] = 0;
            prepared->verify_batch.logits[i] = true;
            prepared->output_indices[i] = (int32_t) i;
        }
        round_set_verify_token_count(*prepared, (int32_t) output_count);

        const int checkpoint_mode = llama_spec_ckpt_init(
            session->target_context, session->engine->checkpoint_mode, (int) output_count);
        if (checkpoint_mode == LLAMA_SPEC_CKPT_NONE || !llama_spec_ckpt_save(session->target_context, 0)) {
            llama_spec_ckpt_discard(session->target_context);
            return fail_pre_call(LLAMA_SPECULATIVE_UNSUPPORTED, "target context could not create a speculative checkpoint");
        }
        prepared->checkpoint_saved = true;

        if (llama_decode(session->target_context, prepared->verify_batch) != 0) {
            if (!round_restore_target(*prepared, 0)) {
                poison_session(*session, result, LLAMA_SPECULATIVE_POISONED);
                return fail(error, LLAMA_SPECULATIVE_POISONED, "target verification failed and rollback could not be proven");
            }
            return fail_pre_call(LLAMA_SPECULATIVE_DECODE_FAILED, "target verification decode failed and was rolled back");
        }
        session->metrics.target_decode_calls++;
        session->metrics.target_decode_tokens += output_count;

        if (!round_copy_output_state(*prepared)) {
            if (!round_restore_target(*prepared, 0)) {
                poison_session(*session, result, LLAMA_SPECULATIVE_POISONED);
                return fail(error, LLAMA_SPECULATIVE_POISONED, "target output staging failed and rollback could not be proven");
            }
            return fail_pre_call(LLAMA_SPECULATIVE_INTERNAL_ERROR, "could not stage target verification outputs");
        }

        uint32_t accepted_drafts = 0;
        for (size_t row = 0; row < output_count; ++row) {
            const uint64_t prefix_count = prepared->committable_tokens.size();
            const float * logits = prepared->prefix_logits.data() + row * vocabulary_size;
            const llama_token sampled = sample_token(
                *session->sampler, prepared->provisional, session->target_context, (int32_t) row, logits);
            if (sampled == LLAMA_TOKEN_NULL) {
                if (!round_restore_target(*prepared, 0)) {
                    poison_session(*session, result, LLAMA_SPECULATIVE_POISONED);
                    return fail(error, LLAMA_SPECULATIVE_POISONED, "sampling failed and rollback could not be proven");
                }
                return fail_pre_call(LLAMA_SPECULATIVE_INTERNAL_ERROR, "could not sample target verification logits");
            }

            prepared->view.terminal = candidate_terminal(*prepared, sampled, prefix_count);
            if (prepared->view.terminal != LLAMA_SPECULATIVE_TERMINAL_NONE) {
                break;
            }
            if (row >= draft.size() || sampled != draft[row]) {
                llama_speculative_internal::sampler_token_carry_capture(
                    prepared->outgoing_carry, sampled, prepared->provisional.rng);
                break;
            }
            prepared->provisional.accepted.push_back(sampled);
            prepared->committable_tokens.push_back(sampled);
            prepared->prefix_rng.push_back(prepared->provisional.rng);
            ++accepted_drafts;
        }

        prepared->commit_ids.resize(prepared->committable_tokens.size(), LLAMA_TOKEN_NULL);
        for (size_t i = 1; i < prepared->committable_tokens.size(); ++i) {
            prepared->commit_ids[i - 1] = prepared->committable_tokens[i];
        }
        prepared->view.tokens = prepared->committable_tokens.data();
        prepared->view.committable_count = prepared->committable_tokens.size();
        if (!checked_position_add(initial_position, output_count, prepared->view.provisional_position)) {
            return fail_pre_call(LLAMA_SPECULATIVE_INTERNAL_ERROR,
                    "target verification position exceeded the prevalidated bound");
        }
        prepared->view.proposed_draft_tokens = (uint32_t) draft.size();
        prepared->view.accepted_draft_tokens = accepted_drafts;
        prepared->view.flags = LLAMA_SPECULATIVE_ROUND_ATTEMPTED;
        if (draft.empty()) {
            prepared->view.flags |= LLAMA_SPECULATIVE_ROUND_TARGET_ONLY;
            session->metrics.target_only_rounds++;
        } else {
            prepared->view.flags |= LLAMA_SPECULATIVE_ROUND_USED_SPECULATION;
            session->metrics.speculative_rounds++;
        }
        if ((params->flags & LLAMA_SPECULATIVE_ROUND_FORCE_TARGET_ONLY) != 0) {
            prepared->view.flags |= LLAMA_SPECULATIVE_ROUND_RECOVERED_FALLBACK;
            session->metrics.recoverable_fallbacks++;
        }

        session->metrics.rounds_attempted++;
        session->metrics.proposed_draft_tokens += draft.size();
        session->round_open = true;
        *out_round = prepared.release();
        session->active_round = *out_round;
        return finish_call(result, LLAMA_SPECULATIVE_OK, LLAMA_SPECULATIVE_EFFECT_UNCHANGED,
                LLAMA_SPECULATIVE_BOUNDARY_PRE_CALL, 0, initial_position);
    } catch (const std::bad_alloc &) {
        if (prepared != nullptr && prepared->checkpoint_saved && !round_restore_target(*prepared, 0)) {
            poison_session(*session, result, LLAMA_SPECULATIVE_POISONED);
            return fail(error, LLAMA_SPECULATIVE_POISONED, "round allocation failed and rollback could not be proven");
        }
        return fail_pre_call(LLAMA_SPECULATIVE_OUT_OF_MEMORY, "could not allocate speculative round state");
    } catch (const std::exception & exception) {
        if (prepared != nullptr && prepared->checkpoint_saved && !round_restore_target(*prepared, 0)) {
            poison_session(*session, result, LLAMA_SPECULATIVE_POISONED);
            return fail(error, LLAMA_SPECULATIVE_POISONED, "round preparation failed and rollback could not be proven");
        }
        return fail_pre_call(LLAMA_SPECULATIVE_INTERNAL_ERROR, exception.what());
    } catch (...) {
        if (prepared != nullptr && prepared->checkpoint_saved && !round_restore_target(*prepared, 0)) {
            poison_session(*session, result, LLAMA_SPECULATIVE_POISONED);
            return fail(error, LLAMA_SPECULATIVE_POISONED, "unknown round failure and rollback could not be proven");
        }
        return fail_pre_call(LLAMA_SPECULATIVE_INTERNAL_ERROR, "unknown speculative round failure");
    }
}

llama_speculative_status llama_speculative_round_get_view(
        const llama_speculative_round * round,
        llama_speculative_round_view * view,
        llama_speculative_error ** error) {
    clear_error(error);
    if (round == nullptr || view == nullptr || round->session == nullptr ||
            !session_on_owner_thread(round->session) || !round->session->round_open ||
            round->session->active_round != round) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "open round and view output are required");
    }
    *view = round->view;
    view->tokens = round->committable_tokens.empty() ? nullptr : round->committable_tokens.data();
    return LLAMA_SPECULATIVE_OK;
}

llama_speculative_status llama_speculative_round_commit(
        llama_speculative_round ** round_pointer,
        uint64_t prefix_count,
        llama_speculative_round_commit_result * result,
        llama_speculative_error ** error) {
    clear_error(error);
    if (result != nullptr) {
        *result = {};
        init_call_result(&result->call, 0);
    }
    if (round_pointer == nullptr || *round_pointer == nullptr || result == nullptr) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "open round, commit result, and round pointer are required");
    }
    llama_speculative_round & round = **round_pointer;
    llama_speculative_session & session = *round.session;
    init_call_result(&result->call, session.next_position);
    if (!session_on_owner_thread(&session) || !session.round_open || session.active_round != &round ||
            prefix_count > round.committable_tokens.size()) {
        return fail_call(&result->call, error, LLAMA_SPECULATIVE_INVALID_ARGUMENT,
                "commit prefix or round state is invalid", session.next_position);
    }
    llama_pos committed_position = 0;
    if (!checked_position_add(round.view.initial_position, prefix_count, committed_position)) {
        return fail_call(&result->call, error, LLAMA_SPECULATIVE_INTERNAL_ERROR,
                "committed speculative prefix position overflows", session.next_position);
    }

    if (!round_restore_target(round, prefix_count)) {
        poison_session(session, &result->call, LLAMA_SPECULATIVE_ROLLBACK_FAILED);
        finish_round(round_pointer);
        return fail(error, LLAMA_SPECULATIVE_ROLLBACK_FAILED, "could not restore the requested target prefix");
    }

    llama_speculative_effect effect = LLAMA_SPECULATIVE_EFFECT_UNCHANGED;
    const bool mtp = session.engine->stage.type == bridge_stage_type::MTP;
    const bool keep_companion_boundary = mtp && prefix_count != 0 && round.companion_boundary_row_retained;
    const companion_tail_result tail_result = round_discard_companion_tail(round, keep_companion_boundary);
    if (tail_result == companion_tail_result::FAILED) {
        poison_session(session, &result->call, LLAMA_SPECULATIVE_INTERNAL_ERROR);
        finish_round(round_pointer);
        return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR,
                "target boundary restored, but MTP draft tail could not be discarded");
    }
    if (tail_result == companion_tail_result::CLEARED) {
        effect = LLAMA_SPECULATIVE_EFFECT_SPEC_STATE_CLEARED_TARGET_VALID;
    }
    if (prefix_count != 0) {
        bool conditioning_committed = tail_result != companion_tail_result::CLEARED;
        if (conditioning_committed) {
            try {
                round.hidden_rows.resize((size_t) prefix_count * (size_t) session.engine->feature_width);
                if (mtp) {
                    round.commit_ids.assign(
                        round.committable_tokens.begin(),
                        round.committable_tokens.begin() + (size_t) prefix_count);
                } else {
                    round.commit_ids.resize((size_t) prefix_count);
                }
                conditioning_committed = mtp
                    ? common_speculative_mtp_commit_prefix(
                        session.speculative,
                        0,
                        round.view.initial_position,
                        round.commit_ids,
                        round.hidden_rows,
                        keep_companion_boundary)
                    : common_speculative_commit_accepted_hidden_rows(
                        session.speculative,
                        COMMON_SPECULATIVE_TYPE_DSPARK,
                        0,
                        round.view.initial_position + 1,
                        round.committable_tokens[0],
                        round.commit_ids,
                        round.hidden_rows);
            } catch (...) {
                conditioning_committed = false;
            }
        }
        if (!conditioning_committed && tail_result != companion_tail_result::CLEARED) {
            bool reset = false;
            try {
                reset = reset_session_conditioning(session, committed_position);
            } catch (...) {
                reset = false;
            }
            if (!reset) {
                poison_session(session, &result->call, LLAMA_SPECULATIVE_INTERNAL_ERROR);
                finish_round(round_pointer);
                return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR,
                        "accepted target prefix committed, but speculative conditioning could not be reset");
            }
            effect = LLAMA_SPECULATIVE_EFFECT_SPEC_STATE_CLEARED_TARGET_VALID;
        }
    }

    llama_speculative_sampler & sampler = *session.sampler;
    for (size_t i = 0; i < (size_t) prefix_count; ++i) {
        sampler.canonical.accepted.push_back(round.committable_tokens[i]);
    }
    sampler.canonical.rng = round.prefix_rng[(size_t) prefix_count];
    session.next_position = committed_position;
    session.metrics.accepted_draft_tokens += prefix_count > 0
        ? std::min<uint64_t>(prefix_count - 1, round.view.accepted_draft_tokens)
        : 0;
    session.metrics.committed_tokens += prefix_count;
    try {
        common_speculative_accept(session.speculative,
            (uint16_t) std::min<uint64_t>(prefix_count > 0 ? prefix_count - 1 : 0, UINT16_MAX));
    } catch (...) {
        bool reset = false;
        try {
            reset = reset_session_conditioning(session, session.next_position);
        } catch (...) {
            reset = false;
        }
        if (!reset) {
            poison_session(session, &result->call, LLAMA_SPECULATIVE_INTERNAL_ERROR);
            finish_round(round_pointer);
            return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR,
                    "target prefix committed, but speculative acceptance bookkeeping failed");
        }
        effect = LLAMA_SPECULATIVE_EFFECT_SPEC_STATE_CLEARED_TARGET_VALID;
    }

    const bool full_prefix_committed = prefix_count == round.committable_tokens.size();
    llama_speculative_internal::sampler_token_carry_commit(
        session.private_carry,
        round.outgoing_carry,
        full_prefix_committed && round.view.terminal == LLAMA_SPECULATIVE_TERMINAL_NONE);

    result->terminal = full_prefix_committed
        ? round.view.terminal
        : LLAMA_SPECULATIVE_TERMINAL_NONE;
    finish_call(&result->call, LLAMA_SPECULATIVE_OK, effect,
            LLAMA_SPECULATIVE_BOUNDARY_POST_CALL_COMMITTED, prefix_count, session.next_position);
    finish_round(round_pointer);
    return LLAMA_SPECULATIVE_OK;
}

llama_speculative_status llama_speculative_round_abort(
        llama_speculative_round ** round_pointer,
        llama_speculative_call_result * result,
        llama_speculative_error ** error) {
    clear_error(error);
    if (round_pointer == nullptr || *round_pointer == nullptr || result == nullptr) {
        init_call_result(result, 0);
        return fail_call(result, error, LLAMA_SPECULATIVE_INVALID_ARGUMENT,
                "open round, result, and round pointer are required", 0);
    }
    llama_speculative_round & round = **round_pointer;
    llama_speculative_session & session = *round.session;
    init_call_result(result, session.next_position);
    if (!session_on_owner_thread(&session) || !session.round_open || session.active_round != &round) {
        return fail_call(result, error, LLAMA_SPECULATIVE_INVALID_ARGUMENT,
                "round is not open on its owner thread", session.next_position);
    }
    if (!round_restore_target(round, 0)) {
        poison_session(session, result, LLAMA_SPECULATIVE_ROLLBACK_FAILED);
        finish_round(round_pointer);
        return fail(error, LLAMA_SPECULATIVE_ROLLBACK_FAILED, "could not restore the pre-round target boundary");
    }
    const companion_tail_result tail_result = round_discard_companion_tail(round);
    if (tail_result == companion_tail_result::FAILED) {
        poison_session(session, result, LLAMA_SPECULATIVE_INTERNAL_ERROR);
        finish_round(round_pointer);
        return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR,
                "target boundary restored, but MTP draft tail could not be discarded");
    }
    const llama_speculative_effect effect = tail_result == companion_tail_result::CLEARED
        ? LLAMA_SPECULATIVE_EFFECT_SPEC_STATE_CLEARED_TARGET_VALID
        : LLAMA_SPECULATIVE_EFFECT_UNCHANGED;
    finish_call(result, LLAMA_SPECULATIVE_OK, effect,
            LLAMA_SPECULATIVE_BOUNDARY_PRE_CALL, 0, session.next_position);
    finish_round(round_pointer);
    return LLAMA_SPECULATIVE_OK;
}

llama_speculative_status llama_speculative_session_state_size(
        const llama_speculative_session * const_session,
        uint64_t * required_size,
        llama_speculative_state_info * info,
        llama_speculative_error ** error) {
    clear_error(error);
    if (required_size != nullptr) {
        *required_size = 0;
    }
    if (info != nullptr) {
        *info = {};
    }
    if (const_session == nullptr || required_size == nullptr || info == nullptr ||
            !session_on_owner_thread(const_session) || !const_session->attached || const_session->poisoned ||
            const_session->round_open || const_session->private_carry.ready) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "quiescent attached session and state outputs are required");
    }
    auto & session = *const_cast<llama_speculative_session *>(const_session);
    if (session.engine->stage.type == bridge_stage_type::MTP) {
        try {
            common_speculative_mtp_state_view view;
            std::vector<uint8_t> context_state;
            parsed_mtp_state state;
            if (!mtp_state_from_session(session, view, context_state, state, *required_size)) {
                return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, "could not inspect live MTP canonical state");
            }
            fill_state_info(*session.engine, state, info);
            return LLAMA_SPECULATIVE_OK;
        } catch (const std::bad_alloc &) {
            return fail(error, LLAMA_SPECULATIVE_OUT_OF_MEMORY, "could not allocate MTP canonical state view");
        } catch (const std::exception & exception) {
            return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, exception.what());
        } catch (...) {
            return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, "unknown MTP canonical state inspection failure");
        }
    }
    common_speculative_dflash_state_view view;
    parsed_dspark_state state;
    if (!state_from_session(session, view, state, *required_size)) {
        return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, "could not inspect live DSpark canonical state");
    }
    fill_state_info(*session.engine, state, info);
    return LLAMA_SPECULATIVE_OK;
}

llama_speculative_status llama_speculative_session_state_save(
        const llama_speculative_session * const_session,
        void * destination,
        uint64_t capacity,
        uint64_t * written_size,
        llama_speculative_state_info * info,
        llama_speculative_error ** error) {
    clear_error(error);
    if (written_size != nullptr) {
        *written_size = 0;
    }
    if (info != nullptr) {
        *info = {};
    }
    if (const_session == nullptr || written_size == nullptr || info == nullptr ||
            !session_on_owner_thread(const_session) || !const_session->attached || const_session->poisoned ||
            const_session->round_open || const_session->private_carry.ready) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "quiescent attached session and state outputs are required");
    }
    auto & session = *const_cast<llama_speculative_session *>(const_session);
    if (session.engine->stage.type == bridge_stage_type::MTP) {
        try {
            common_speculative_mtp_state_view view;
            std::vector<uint8_t> context_state;
            parsed_mtp_state state;
            uint64_t required_size = 0;
            if (!mtp_state_from_session(session, view, context_state, state, required_size)) {
                return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, "could not inspect live MTP canonical state");
            }
            *written_size = required_size;
            fill_state_info(*session.engine, state, info);
            if (destination == nullptr || capacity < required_size) {
                return LLAMA_SPECULATIVE_BUFFER_TOO_SMALL;
            }

            auto * bytes = static_cast<uint8_t *>(destination);
            std::memcpy(bytes, STATE_MAGIC.data(), STATE_MAGIC.size());
            write_u32(bytes + 8, STATE_HEADER_SIZE);
            write_u32(bytes + 12, (uint32_t) required_size);
            std::memcpy(bytes + 16, session.engine->state_layout_id.data(), session.engine->state_layout_id.size());
            write_i64(bytes + 48, state.sequence_start_position);
            write_i64(bytes + 56, state.next_position);
            write_u32(bytes + 64, 0);
            write_u32(bytes + 68, 1);

            uint8_t * section = bytes + STATE_HEADER_SIZE;
            write_u32(section, 0);
            write_u32(section + 4, STATE_SECTION_MTP);
            write_u16(section + 8, 1);
            write_u16(section + 10, 0);
            write_u32(section + 12, STATE_SECTION_REQUIRED);
            write_u64(section + 16, STATE_HEADER_SIZE + STATE_SECTION_ENTRY_SIZE);
            write_u64(section + 24, required_size - STATE_HEADER_SIZE - STATE_SECTION_ENTRY_SIZE);

            uint8_t * payload = section + STATE_SECTION_ENTRY_SIZE;
            write_u32(payload, state.feature_width);
            write_u32(payload + 4, state.warmed_heads);
            write_u32(payload + 8, state.hidden_count);
            write_u32(payload + 12, state.prefix_complete ? 1u : 0u);
            write_i64(payload + 16, state.sequence_start_position);
            write_i64(payload + 24, state.next_position);
            write_u64(payload + 32, state.context_state_size);

            uint8_t * cursor = payload + STATE_MTP_PREFIX_SIZE;
            for (uint32_t i = 0; i < state.hidden_count; ++i) {
                write_f32(cursor, view.target_hidden[i]);
                cursor += sizeof(float);
            }
            std::memcpy(cursor, context_state.data(), context_state.size());
            cursor += context_state.size();
            const auto digest = sha256_mtp_payload(payload, (size_t) (cursor - payload));
            std::memcpy(payload + 40, digest.data(), digest.size());
            if ((uint64_t) (cursor - bytes) != required_size) {
                return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, "MTP serialization length invariant failed");
            }
            session.metrics.state_saves++;
            session.metrics.state_save_bytes += required_size;
            return LLAMA_SPECULATIVE_OK;
        } catch (const std::bad_alloc &) {
            return fail(error, LLAMA_SPECULATIVE_OUT_OF_MEMORY, "could not allocate MTP canonical state");
        } catch (const std::exception & exception) {
            return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, exception.what());
        } catch (...) {
            return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, "unknown MTP canonical state save failure");
        }
    }
    common_speculative_dflash_state_view view;
    parsed_dspark_state state;
    uint64_t required_size = 0;
    if (!state_from_session(session, view, state, required_size)) {
        return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, "could not inspect live DSpark canonical state");
    }
    *written_size = required_size;
    fill_state_info(*session.engine, state, info);
    if (destination == nullptr || capacity < required_size) {
        return LLAMA_SPECULATIVE_BUFFER_TOO_SMALL;
    }

    auto * bytes = static_cast<uint8_t *>(destination);
    std::memcpy(bytes, STATE_MAGIC.data(), STATE_MAGIC.size());
    write_u32(bytes + 8, STATE_HEADER_SIZE);
    write_u32(bytes + 12, (uint32_t) required_size);
    std::memcpy(bytes + 16, session.engine->state_layout_id.data(), session.engine->state_layout_id.size());
    write_i64(bytes + 48, state.sequence_start_position);
    write_i64(bytes + 56, state.next_position);
    write_u32(bytes + 64, 0);
    write_u32(bytes + 68, 1);

    uint8_t * section = bytes + STATE_HEADER_SIZE;
    write_u32(section, 0);
    write_u32(section + 4, STATE_SECTION_DSPARK);
    write_u16(section + 8, 1);
    write_u16(section + 10, 0);
    write_u32(section + 12, STATE_SECTION_REQUIRED);
    write_u64(section + 16, STATE_HEADER_SIZE + STATE_SECTION_ENTRY_SIZE);
    write_u64(section + 24, required_size - STATE_HEADER_SIZE - STATE_SECTION_ENTRY_SIZE);

    uint8_t * payload = section + STATE_SECTION_ENTRY_SIZE;
    write_u32(payload, state.row_count);
    write_u32(payload + 4, state.feature_width);
    write_u32(payload + 8, state.cross_ctx);
    write_u32(payload + 12, state.target_layer_count);
    write_i64(payload + 16, state.sequence_start_position);
    write_i64(payload + 24, state.coverage_start_position);
    write_i64(payload + 32, state.window_start_position);
    write_i64(payload + 40, state.next_position);

    uint8_t * cursor = payload + STATE_DSPARK_PREFIX_SIZE;
    for (uint32_t i = 0; i < state.target_layer_count; ++i) {
        write_i32(cursor, view.data.target_layer_ids[i]);
        cursor += sizeof(int32_t);
    }
    for (uint32_t i = 0; i < state.row_count; ++i) {
        write_i64(cursor, view.data.positions[i]);
        cursor += sizeof(int64_t);
    }
    for (uint64_t i = 0; i < state.feature_count; ++i) {
        write_f32(cursor, view.data.features[i]);
        cursor += sizeof(float);
    }
    if ((uint64_t) (cursor - bytes) != required_size) {
        return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, "DSpark serialization length invariant failed");
    }
    session.metrics.state_saves++;
    session.metrics.state_save_bytes += required_size;
    return LLAMA_SPECULATIVE_OK;
}

llama_speculative_status llama_speculative_engine_state_inspect(
        const llama_speculative_engine * engine,
        const llama_speculative_state_inspect_params * params,
        llama_speculative_state_info * info,
        llama_speculative_error ** error) {
    clear_error(error);
    if (info != nullptr) {
        *info = {};
    }
    if (engine == nullptr || params == nullptr || info == nullptr || params->flags != 0 ||
            (params->quality_key_capacity != 0 && params->quality_key == nullptr)) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "engine, state inspection input, and output are required");
    }
    if (engine->stage.type == bridge_stage_type::MTP) {
        parsed_mtp_state state;
        std::string message;
        llama_speculative_status status = LLAMA_SPECULATIVE_INTERNAL_ERROR;
        try {
            status = parse_mtp_state(*engine, params->state, params->state_size, state, message);
        } catch (const std::bad_alloc &) {
            return fail(error, LLAMA_SPECULATIVE_OUT_OF_MEMORY, "could not inspect MTP speculative state");
        } catch (const std::exception & exception) {
            return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, exception.what());
        } catch (...) {
            return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, "unknown MTP state inspection failure");
        }
        if (status != LLAMA_SPECULATIVE_OK) {
            return fail(error, status, message);
        }
        fill_state_info(*engine, state, info);
        if (params->quality_key == nullptr || params->quality_key_capacity < STATE_QUALITY_KEY_SIZE) {
            return LLAMA_SPECULATIVE_BUFFER_TOO_SMALL;
        }
        make_quality_key(state, params->quality_key);
        return LLAMA_SPECULATIVE_OK;
    }
    parsed_dspark_state state;
    std::string message;
    llama_speculative_status status = LLAMA_SPECULATIVE_INTERNAL_ERROR;
    try {
        status = parse_state(*engine, params->state, params->state_size, state, message);
    } catch (const std::bad_alloc &) {
        return fail(error, LLAMA_SPECULATIVE_OUT_OF_MEMORY, "could not inspect speculative state");
    } catch (const std::exception & exception) {
        return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, exception.what());
    } catch (...) {
        return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, "unknown speculative state inspection failure");
    }
    if (status != LLAMA_SPECULATIVE_OK) {
        return fail(error, status, message);
    }
    fill_state_info(*engine, state, info);
    if (params->quality_key == nullptr || params->quality_key_capacity < STATE_QUALITY_KEY_SIZE) {
        return LLAMA_SPECULATIVE_BUFFER_TOO_SMALL;
    }
    make_quality_key(state, params->quality_key);
    return LLAMA_SPECULATIVE_OK;
}

llama_speculative_status llama_speculative_session_state_load(
        llama_speculative_session * session,
        const void * state_bytes,
        uint64_t state_size,
        llama_speculative_call_result * result,
        llama_speculative_error ** error) {
    const llama_speculative_status checked = check_session_call(session, result, error);
    if (checked != LLAMA_SPECULATIVE_OK) {
        return checked;
    }
    if (session->generation_open || session->round_open) {
        return fail_call(result, error, LLAMA_SPECULATIVE_BUSY,
                "state load requires an idle attached session", session->next_position);
    }

    if (session->engine->stage.type == bridge_stage_type::MTP) {
        parsed_mtp_state parsed;
        std::string message;
        llama_speculative_status parsed_status = LLAMA_SPECULATIVE_INTERNAL_ERROR;
        try {
            parsed_status = parse_mtp_state(*session->engine, state_bytes, state_size, parsed, message);
        } catch (const std::bad_alloc &) {
            return fail_call(result, error, LLAMA_SPECULATIVE_OUT_OF_MEMORY,
                    "could not parse MTP speculative state", session->next_position);
        } catch (const std::exception & exception) {
            return fail_call(result, error, LLAMA_SPECULATIVE_INTERNAL_ERROR,
                    exception.what(), session->next_position);
        } catch (...) {
            return fail_call(result, error, LLAMA_SPECULATIVE_INTERNAL_ERROR,
                    "unknown MTP state parsing failure", session->next_position);
        }
        if (parsed_status != LLAMA_SPECULATIVE_OK) {
            finish_call(result, parsed_status, LLAMA_SPECULATIVE_EFFECT_UNCHANGED,
                    LLAMA_SPECULATIVE_BOUNDARY_PRE_CALL, 0, session->next_position);
            return fail(error, parsed_status, message);
        }
        if (parsed.sequence_start_position != session->sequence_start_position ||
                parsed.next_position != session->next_position) {
            finish_call(result, LLAMA_SPECULATIVE_STATE_INCOMPATIBLE, LLAMA_SPECULATIVE_EFFECT_UNCHANGED,
                    LLAMA_SPECULATIVE_BOUNDARY_PRE_CALL, 0, session->next_position);
            return fail(error, LLAMA_SPECULATIVE_STATE_INCOMPATIBLE,
                    "MTP speculative state boundary differs from the target session");
        }

        try {
            std::vector<float> hidden(parsed.hidden_count);
            for (uint32_t i = 0; i < parsed.hidden_count; ++i) {
                hidden[i] = read_f32(parsed.hidden + (uint64_t) i * sizeof(float));
            }
            const common_speculative_mtp_state_import_status imported = common_speculative_mtp_state_import(
                session->speculative,
                0,
                hidden.empty() ? nullptr : hidden.data(),
                hidden.size(),
                (int32_t) parsed.warmed_heads,
                parsed.context_state,
                (size_t) parsed.context_state_size);
            if (imported != COMMON_SPECULATIVE_MTP_STATE_IMPORT_OK) {
                const llama_speculative_effect effect =
                    imported == COMMON_SPECULATIVE_MTP_STATE_IMPORT_FAILED_CLEARED
                    ? LLAMA_SPECULATIVE_EFFECT_SPEC_STATE_CLEARED_TARGET_VALID
                    : LLAMA_SPECULATIVE_EFFECT_UNCHANGED;
                if (effect == LLAMA_SPECULATIVE_EFFECT_SPEC_STATE_CLEARED_TARGET_VALID) {
                    session->mtp_prefix_complete = false;
                }
                finish_call(result, LLAMA_SPECULATIVE_STATE_CORRUPT, effect,
                        effect == LLAMA_SPECULATIVE_EFFECT_UNCHANGED
                        ? LLAMA_SPECULATIVE_BOUNDARY_PRE_CALL
                        : LLAMA_SPECULATIVE_BOUNDARY_POST_CALL_COMMITTED,
                        0, session->next_position);
                return fail(error, LLAMA_SPECULATIVE_STATE_CORRUPT,
                        "MTP companion sequence state could not be restored");
            }
        } catch (const std::bad_alloc &) {
            return fail_call(result, error, LLAMA_SPECULATIVE_OUT_OF_MEMORY,
                    "could not allocate MTP canonical state", session->next_position);
        } catch (const std::exception & exception) {
            return fail_call(result, error, LLAMA_SPECULATIVE_INTERNAL_ERROR,
                    exception.what(), session->next_position);
        } catch (...) {
            return fail_call(result, error, LLAMA_SPECULATIVE_INTERNAL_ERROR,
                    "unknown MTP state import failure", session->next_position);
        }

        session->mtp_prefix_complete = parsed.prefix_complete;
        session->metrics.state_loads++;
        session->metrics.state_load_bytes += state_size;
        return finish_call(result, LLAMA_SPECULATIVE_OK, LLAMA_SPECULATIVE_EFFECT_UNCHANGED,
                LLAMA_SPECULATIVE_BOUNDARY_POST_CALL_COMMITTED, 0, session->next_position);
    }

    parsed_dspark_state parsed;
    std::string message;
    llama_speculative_status parsed_status = LLAMA_SPECULATIVE_INTERNAL_ERROR;
    try {
        parsed_status = parse_state(*session->engine, state_bytes, state_size, parsed, message);
    } catch (const std::bad_alloc &) {
        return fail_call(result, error, LLAMA_SPECULATIVE_OUT_OF_MEMORY,
                "could not parse speculative state", session->next_position);
    } catch (const std::exception & exception) {
        return fail_call(result, error, LLAMA_SPECULATIVE_INTERNAL_ERROR,
                exception.what(), session->next_position);
    } catch (...) {
        return fail_call(result, error, LLAMA_SPECULATIVE_INTERNAL_ERROR,
                "unknown speculative state parsing failure", session->next_position);
    }
    if (parsed_status != LLAMA_SPECULATIVE_OK) {
        finish_call(result, parsed_status, LLAMA_SPECULATIVE_EFFECT_UNCHANGED,
                LLAMA_SPECULATIVE_BOUNDARY_PRE_CALL, 0, session->next_position);
        return fail(error, parsed_status, message);
    }
    if (parsed.sequence_start_position != session->sequence_start_position ||
            parsed.next_position != session->next_position) {
        finish_call(result, LLAMA_SPECULATIVE_STATE_INCOMPATIBLE, LLAMA_SPECULATIVE_EFFECT_UNCHANGED,
                LLAMA_SPECULATIVE_BOUNDARY_PRE_CALL, 0, session->next_position);
        return fail(error, LLAMA_SPECULATIVE_STATE_INCOMPATIBLE, "speculative state boundary differs from the target session");
    }

    try {
        std::vector<int32_t> layer_ids(parsed.target_layer_count);
        std::vector<llama_pos> positions(parsed.row_count);
        std::vector<float> features((size_t) parsed.feature_count);
        for (uint32_t i = 0; i < parsed.target_layer_count; ++i) {
            layer_ids[i] = read_i32(parsed.layer_ids + (uint64_t) i * sizeof(int32_t));
        }
        for (uint32_t i = 0; i < parsed.row_count; ++i) {
            positions[i] = (llama_pos) read_i64(parsed.positions + (uint64_t) i * sizeof(int64_t));
        }
        for (uint64_t i = 0; i < parsed.feature_count; ++i) {
            features[(size_t) i] = read_f32(parsed.features + i * sizeof(float));
        }

        common_speculative_dflash_state_data data;
        data.row_count = (int32_t) parsed.row_count;
        data.feature_width = (int32_t) parsed.feature_width;
        data.cross_ctx = (int32_t) parsed.cross_ctx;
        data.target_layer_count = (int32_t) parsed.target_layer_count;
        data.sequence_start_position = parsed.sequence_start_position;
        data.coverage_start_position = parsed.coverage_start_position;
        data.window_start_position = parsed.window_start_position;
        data.next_position = parsed.next_position;
        data.target_layer_ids = layer_ids.empty() ? nullptr : layer_ids.data();
        data.positions = positions.empty() ? nullptr : positions.data();
        data.features = features.empty() ? nullptr : features.data();
        data.feature_count = features.size();
        if (!common_speculative_dflash_state_import(session->speculative, data)) {
            finish_call(result, LLAMA_SPECULATIVE_OUT_OF_MEMORY, LLAMA_SPECULATIVE_EFFECT_UNCHANGED,
                    LLAMA_SPECULATIVE_BOUNDARY_PRE_CALL, 0, session->next_position);
            return fail(error, LLAMA_SPECULATIVE_OUT_OF_MEMORY, "could not copy DSpark canonical state");
        }
    } catch (const std::bad_alloc &) {
        finish_call(result, LLAMA_SPECULATIVE_OUT_OF_MEMORY, LLAMA_SPECULATIVE_EFFECT_UNCHANGED,
                LLAMA_SPECULATIVE_BOUNDARY_PRE_CALL, 0, session->next_position);
        return fail(error, LLAMA_SPECULATIVE_OUT_OF_MEMORY, "could not allocate DSpark canonical state");
    } catch (const std::exception & exception) {
        finish_call(result, LLAMA_SPECULATIVE_INTERNAL_ERROR, LLAMA_SPECULATIVE_EFFECT_UNCHANGED,
                LLAMA_SPECULATIVE_BOUNDARY_PRE_CALL, 0, session->next_position);
        return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, exception.what());
    } catch (...) {
        finish_call(result, LLAMA_SPECULATIVE_INTERNAL_ERROR, LLAMA_SPECULATIVE_EFFECT_UNCHANGED,
                LLAMA_SPECULATIVE_BOUNDARY_PRE_CALL, 0, session->next_position);
        return fail(error, LLAMA_SPECULATIVE_INTERNAL_ERROR, "unknown DSpark state import failure");
    }

    session->metrics.state_loads++;
    session->metrics.state_load_bytes += state_size;
    return finish_call(result, LLAMA_SPECULATIVE_OK, LLAMA_SPECULATIVE_EFFECT_UNCHANGED,
            LLAMA_SPECULATIVE_BOUNDARY_POST_CALL_COMMITTED, 0, session->next_position);
}

llama_speculative_status llama_speculative_session_get_metrics(
        const llama_speculative_session * session,
        llama_speculative_metrics * metrics,
        llama_speculative_error ** error) {
    clear_error(error);
    if (session == nullptr || metrics == nullptr) {
        return fail(error, LLAMA_SPECULATIVE_INVALID_ARGUMENT, "session and metrics output are required");
    }
    if (!session_on_owner_thread(session) || session->round_open) {
        return fail(error, LLAMA_SPECULATIVE_BUSY, "session is unavailable for a stable metrics snapshot");
    }
    *metrics = session->metrics;
    return LLAMA_SPECULATIVE_OK;
}

llama_speculative_status llama_speculative_session_reset_metrics(
        llama_speculative_session * session,
        llama_speculative_error ** error) {
    clear_error(error);
    if (session == nullptr || !session_on_owner_thread(session) || session->round_open) {
        return fail(error, LLAMA_SPECULATIVE_BUSY, "session is unavailable for metrics reset");
    }
    session->metrics = {};
    return LLAMA_SPECULATIVE_OK;
}

} // extern "C"
