#include "common.h"

#include <cstdio>
#include <cstdlib>
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

int main() {
    check_model_requirement(COMMON_SPECULATIVE_TYPE_DRAFT);
    check_model_requirement(COMMON_SPECULATIVE_TYPE_DFLASH);
    check_model_requirement(COMMON_SPECULATIVE_TYPE_DSPARK);
    return 0;
}
