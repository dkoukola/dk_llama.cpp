#include "clip.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

static bool check_preprocessed_size(clip_ctx * ctx, int width, int height, int expected_width, int expected_height) {
    std::vector<unsigned char> pixels(static_cast<size_t>(width) * height * 3, 127);
    clip_image_u8 * image = clip_image_u8_init();
    clip_image_f32_batch * batch = clip_image_f32_batch_init();
    if (image == nullptr || batch == nullptr) {
        clip_image_u8_free(image);
        clip_image_f32_batch_free(batch);
        return false;
    }

    clip_build_img_from_pixels(pixels.data(), width, height, image);
    const bool success = clip_image_preprocess(ctx, image, batch) &&
                         clip_image_f32_batch_n_images(batch) == 1 &&
                         clip_image_f32_batch_nx(batch, 0) == static_cast<size_t>(expected_width) &&
                         clip_image_f32_batch_ny(batch, 0) == static_cast<size_t>(expected_height);
    clip_image_u8_free(image);
    clip_image_f32_batch_free(batch);
    return success;
}

static clip_ctx * load(const char * path, int image_min_tokens, int image_max_tokens) {
    const clip_context_params params = {
        /* use_gpu          */ false,
        /* verbosity        */ GGML_LOG_LEVEL_ERROR,
        /* flash_attn_type  */ CLIP_FLASH_ATTN_TYPE_DISABLED,
        /* image_min_tokens */ image_min_tokens,
        /* image_max_tokens */ image_max_tokens,
        /* kq_type          */ GGML_TYPE_F32,
    };
    clip_init_result result = clip_init(path, params);
    if (result.ctx_a != nullptr) {
        clip_free(result.ctx_a);
        clip_free(result.ctx_v);
        return nullptr;
    }
    return result.ctx_v;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        return EXIT_FAILURE;
    }

    clip_ctx * ctx = load(argv[1], -1, -1);
    if (ctx == nullptr || !clip_is_qwen3vl(ctx) || clip_get_hidden_size(ctx) != 8 ||
        clip_n_mmproj_embd(ctx) != 6 || !check_preprocessed_size(ctx, 1, 1, 32, 32) ||
        !check_preprocessed_size(ctx, 1024, 1024, 512, 512)) {
        clip_free(ctx);
        return EXIT_FAILURE;
    }
    clip_free(ctx);

    ctx = load(argv[1], 4, 16);
    if (ctx == nullptr || !check_preprocessed_size(ctx, 1, 1, 8, 8) ||
        !check_preprocessed_size(ctx, 100, 100, 16, 16)) {
        clip_free(ctx);
        return EXIT_FAILURE;
    }
    clip_free(ctx);
    return EXIT_SUCCESS;
}
