#include "common.h"
#include "llama-dflash.h"
#include "speculative.h"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        std::abort(); \
    } \
} while (0)

static void check_model_requirement(common_speculative_type type) {
    common_params_speculative params;
    params.stages.push_back({ .type = type });

    std::string error;
    CHECK(!common_speculative_validate_chain(params, &error));

    params.model = "draft.gguf";
    CHECK(common_speculative_validate_chain(params, &error));

    params.model.clear();
    alignas(void *) unsigned char borrowed_model_storage = 0;
    params.model_dft = reinterpret_cast<llama_model *>(&borrowed_model_storage);
    CHECK(common_speculative_validate_chain(params, &error));
}

static void check_dflash_runtime_width() {
    const auto dspark_negative = common_speculative_plan_dflash_width(COMMON_SPECULATIVE_TYPE_DSPARK, 5, -1);
    CHECK(dspark_negative.effective_n_max == 0);
    CHECK(dspark_negative.decode_tokens == 5);

    const auto dspark_zero = common_speculative_plan_dflash_width(COMMON_SPECULATIVE_TYPE_DSPARK, 5, 0);
    CHECK(dspark_zero.effective_n_max == 0);
    CHECK(dspark_zero.decode_tokens == 5);

    const auto dspark_three = common_speculative_plan_dflash_width(COMMON_SPECULATIVE_TYPE_DSPARK, 5, 3);
    CHECK(dspark_three.effective_n_max == 3);
    CHECK(dspark_three.decode_tokens == 5);

    const auto dspark_seven = common_speculative_plan_dflash_width(COMMON_SPECULATIVE_TYPE_DSPARK, 5, 7);
    CHECK(dspark_seven.effective_n_max == 7);
    CHECK(dspark_seven.decode_tokens == 7);

    const auto dspark_max = common_speculative_plan_dflash_width(COMMON_SPECULATIVE_TYPE_DSPARK, 5, 1024);
    CHECK(dspark_max.effective_n_max == 1024);
    CHECK(dspark_max.decode_tokens == 1024);

    const auto dflash_seven = common_speculative_plan_dflash_width(COMMON_SPECULATIVE_TYPE_DFLASH, 5, 7);
    CHECK(dflash_seven.effective_n_max == 4);
    CHECK(dflash_seven.decode_tokens == 5);

    const auto dflash_three = common_speculative_plan_dflash_width(COMMON_SPECULATIVE_TYPE_DFLASH, 5, 3);
    CHECK(dflash_three.effective_n_max == 3);
    CHECK(dflash_three.decode_tokens == 5);

    const auto invalid = common_speculative_plan_dflash_width(COMMON_SPECULATIVE_TYPE_DSPARK, 0, 7);
    CHECK(invalid.effective_n_max == 0);
    CHECK(invalid.decode_tokens == 0);

    uint32_t context_size = 0;
    CHECK(common_speculative_dflash_context_size(512, dspark_seven, context_size));
    CHECK(context_size == 519);
    CHECK(!common_speculative_dflash_context_size(
        std::numeric_limits<int32_t>::max(), dspark_seven, context_size));

    CHECK(llama_dflash_runtime_token_capacity(5, 3) == 5);
    CHECK(llama_dflash_runtime_token_capacity(5, 7) == 7);
    CHECK(llama_dflash_default_cross_context(517, 5, 5) == 512);
    CHECK(llama_dflash_default_cross_context(519, 5, 7) == 512);
}

int main() {
    check_model_requirement(COMMON_SPECULATIVE_TYPE_DRAFT);
    check_model_requirement(COMMON_SPECULATIVE_TYPE_DFLASH);
    check_model_requirement(COMMON_SPECULATIVE_TYPE_DSPARK);
    check_dflash_runtime_width();
    return 0;
}
