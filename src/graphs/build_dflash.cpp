#include "../llama-build-context.h"
#include "../llama-context.h"
#include "../llama-dflash.h"
#include "../llama-model.h"

#include <cmath>

static int64_t dflash_dspark_block_count(const llama_batch & batch) {
    if (batch.n_tokens <= 0 || batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return 1;
    }

    int64_t n_blocks = 1;
    llama_seq_id previous = batch.seq_id[0][0];
    for (int32_t i = 1; i < batch.n_tokens; ++i) {
        GGML_ASSERT(batch.n_seq_id[i] > 0 && batch.seq_id[i] != nullptr);
        const llama_seq_id current = batch.seq_id[i][0];
        if (current != previous) {
            ++n_blocks;
            previous = current;
        }
    }

    GGML_ASSERT(batch.n_tokens % n_blocks == 0);
    const int64_t block_size = batch.n_tokens / n_blocks;
    for (int64_t block = 0; block < n_blocks; ++block) {
        const int64_t begin = block * block_size;
        const llama_seq_id id = batch.seq_id[begin][0];
        for (int64_t i = begin + 1; i < begin + block_size; ++i) {
            GGML_ASSERT(batch.seq_id[i][0] == id);
        }
    }

    return n_blocks;
}

static ggml_tensor * build_dspark_markov_head(
        llm_build_context & llm,
        ggml_cgraph * gf,
        ggml_tensor * base_logits,
        ggml_tensor * confidence_hidden) {
    ggml_context * ctx0 = llm.ctx0;
    const llama_model & model = llm.model;
    llama_context & lctx = llm.lctx;
    const llm_build_cb & cb = llm.cb;

    GGML_ASSERT(model.dflash_markov_w1 != nullptr);
    GGML_ASSERT(model.dflash_markov_w2 != nullptr);
    GGML_ASSERT(model.dflash_conf_proj != nullptr);
    GGML_ASSERT(lctx.inp_tokens != nullptr);

    const int64_t n_vocab = base_logits->ne[0];
    const int64_t n_tok = base_logits->ne[1];
    const int64_t n_blocks = dflash_dspark_block_count(llm.batch);
    const int64_t block_drafts = n_tok / n_blocks;
    GGML_ASSERT(block_drafts > 0 && block_drafts <= (int64_t) llm.cparams.n_ubatch);

    const size_t token_stride = (size_t) block_drafts * lctx.inp_tokens->nb[0];
    const size_t logits_stride = (size_t) block_drafts * base_logits->nb[1];
    ggml_tensor * previous = ggml_view_2d(ctx0, lctx.inp_tokens, 1, n_blocks, token_stride, 0);
    previous = ggml_cont_1d(ctx0, previous, n_blocks);

    ggml_tensor * biased_columns = nullptr;
    ggml_tensor * confidence_columns = nullptr;
    for (int64_t i = 0; i < block_drafts; ++i) {
        ggml_tensor * markov_feature = ggml_get_rows(ctx0, model.dflash_markov_w1, previous);
        ggml_tensor * bias = ggml_mul_mat(ctx0, model.dflash_markov_w2, markov_feature);
        ggml_tensor * base_column = ggml_view_2d(ctx0, base_logits, n_vocab, n_blocks,
                logits_stride, i * base_logits->nb[1]);
        ggml_tensor * biased = ggml_add(ctx0, base_column, bias);
        biased_columns = biased_columns == nullptr
                ? biased
                : ggml_concat(ctx0, biased_columns, biased, 1);

        ggml_tensor * hidden_column = ggml_view_2d(ctx0, confidence_hidden,
                confidence_hidden->ne[0], n_blocks,
                (size_t) block_drafts * confidence_hidden->nb[1],
                i * confidence_hidden->nb[1]);
        ggml_tensor * confidence_feature = ggml_concat(ctx0,
                ggml_cont(ctx0, hidden_column), markov_feature, 0);
        ggml_tensor * confidence = ggml_mul_mat(ctx0, model.dflash_conf_proj, confidence_feature);
        if (model.dflash_conf_proj_b != nullptr) {
            confidence = ggml_add(ctx0, confidence, model.dflash_conf_proj_b);
        }
        confidence = ggml_sigmoid(ctx0, confidence);
        confidence_columns = confidence_columns == nullptr
                ? confidence
                : ggml_concat(ctx0, confidence_columns, confidence, 1);

        if (i + 1 < block_drafts) {
            previous = ggml_argmax(ctx0, biased);
        }
    }

    ggml_tensor * result = ggml_reshape_3d(ctx0, biased_columns, n_vocab, n_blocks, block_drafts);
    result = ggml_cont(ctx0, ggml_permute(ctx0, result, 0, 2, 1, 3));
    result = ggml_reshape_2d(ctx0, result, n_vocab, n_tok);
    cb(result, "result_output", -1);
    ggml_build_forward_expand(gf, result);

    ggml_tensor * confidence = ggml_reshape_3d(ctx0, confidence_columns, 1, n_blocks, block_drafts);
    confidence = ggml_cont(ctx0, ggml_permute(ctx0, confidence, 0, 2, 1, 3));
    confidence = ggml_reshape_1d(ctx0, confidence, n_tok);
    ggml_set_name(confidence, "draft_confidence");
    ggml_build_forward_expand(gf, confidence);
    lctx.dflash.draft_confidence_tensor = confidence;

    return result;
}

ggml_cgraph * llm_build_context::build_dflash_kv_cache() {
    const int64_t n_embd_head_k = hparams.n_embd_head_k(0);
    const int64_t n_embd_head_v = hparams.n_embd_head_v(0);
    const int64_t n_target_features = hparams.dflash_n_target_features;
    const int64_t ctx_len = lctx.dflash.visible_cross_ctx > 0
            ? (int64_t) lctx.dflash.visible_cross_ctx
            : llama_dflash_default_cross_context(
                cparams.n_ctx,
                hparams.dflash_block_size,
                cparams.n_ubatch);
    const int64_t update_rows = std::max<int64_t>(1, lctx.dflash.kv.cache_update_rows > 0 ? lctx.dflash.kv.cache_update_rows : ctx_len);
    const int32_t write_pos = lctx.dflash.kv.cache_write_pos;

    GGML_ASSERT(n_embd_head_k == n_embd_head_v);
    GGML_ASSERT(n_target_features > 0);
    GGML_ASSERT(lctx.ensure_dflash_kv_cache_tensors((int32_t) ctx_len));
    GGML_ASSERT(update_rows > 0 && update_rows <= ctx_len);
    GGML_ASSERT(write_pos >= 0 && write_pos < ctx_len);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes((int) std::max<int64_t>(1, update_rows)) + 24 * n_layer, false);

    lctx.dflash.kv.cache_input_target_features = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_target_features, update_rows);
    ggml_set_input(lctx.dflash.kv.cache_input_target_features);
    cb(lctx.dflash.kv.cache_input_target_features, "dflash_kv_input_target_features", -1);

    lctx.dflash.kv.cache_input_pos_ctx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, update_rows);
    ggml_set_input(lctx.dflash.kv.cache_input_pos_ctx);
    cb(lctx.dflash.kv.cache_input_pos_ctx, "dflash_kv_input_pos_ctx", -1);

    ggml_tensor * target_features = lctx.dflash.kv.cache_input_target_features;
    if (hparams.dflash_laguna) {
        GGML_ASSERT(model.dflash_aux_hidden_norms.size() == hparams.dflash_n_target_layers);
        const int64_t slice_width = n_target_features / hparams.dflash_n_target_layers;
        ggml_tensor * normalized_features = nullptr;
        for (uint32_t i = 0; i < hparams.dflash_n_target_layers; ++i) {
            ggml_tensor * slice = ggml_view_2d(
                    ctx0,
                    target_features,
                    slice_width,
                    update_rows,
                    target_features->nb[1],
                    i * slice_width * target_features->nb[0]);
            slice = llm_build_norm(
                    ctx0,
                    slice,
                    hparams,
                    model.dflash_aux_hidden_norms[i],
                    nullptr,
                    LLM_NORM_RMS,
                    cb,
                    -1);
            cb(slice, "dflash_kv_aux_norm", (int) i);
            normalized_features = normalized_features == nullptr
                    ? slice
                    : ggml_concat(ctx0, normalized_features, slice, 0);
        }
        GGML_ASSERT(normalized_features != nullptr);
        target_features = normalized_features;
        cb(target_features, "dflash_kv_normalized_target_features", -1);
    }

    ggml_tensor * fused_target = llm_build_lora_mm(lctx, ctx0, model.dflash_fc, target_features);
    fused_target = llm_build_norm(ctx0, fused_target, hparams, model.dflash_hidden_norm, nullptr, LLM_NORM_RMS, cb, -1);
    cb(fused_target, "dflash_kv_fused_target", -1);

    for (int il = 0; il < n_layer; ++il) {
        GGML_ASSERT(il < (int32_t) lctx.dflash.kv.k_ctx_cache.size());
        GGML_ASSERT(il < (int32_t) lctx.dflash.kv.v_ctx_cache.size());

        ggml_tensor * Kcur_ctx = nullptr;
        ggml_tensor * Vcur_ctx = nullptr;
        if (hparams.dsv4_hc_mult > 0) {
            GGML_ASSERT(model.layers[il].wkv_latent != nullptr);
            ggml_tensor * kv = llm_build_lora_mm(lctx, ctx0, model.layers[il].wkv_latent, fused_target);
            kv = llm_build_norm(ctx0, kv, hparams, model.layers[il].attn_kv_norm,
                    nullptr, LLM_NORM_RMS, cb, il);
            kv = ggml_reshape_3d(ctx0, kv, n_embd_head_k, 1, update_rows);
            kv = ggml_rope_ext_inplace(ctx0, kv, lctx.dflash.kv.cache_input_pos_ctx, nullptr,
                    n_rot, rope_type, 0, freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            kv->op_params[15] = 1;
            cb(kv, "dflash_kv_injected_rope", il);
            Kcur_ctx = ggml_cont(ctx0, ggml_permute(ctx0, kv, 0, 2, 1, 3));
            Vcur_ctx = Kcur_ctx;
            cb(Kcur_ctx, "dflash_kv_injected_physical", il);
        } else {
            ggml_tensor * layer_target = fused_target;
            if (hparams.dflash_laguna) {
                layer_target = llm_build_norm(
                        ctx0,
                        layer_target,
                        hparams,
                        model.layers[il].attn_norm,
                        nullptr,
                        LLM_NORM_RMS,
                        cb,
                        il);
                cb(layer_target, "dflash_kv_attn_norm", il);
            }

            ggml_tensor * Kcur_ctx_proj = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, layer_target);
            if (model.layers[il].bk) { Kcur_ctx_proj = ggml_add(ctx0, Kcur_ctx_proj, model.layers[il].bk); }
            cb(Kcur_ctx_proj, "dflash_kv_k_proj", il);

            Kcur_ctx = ggml_reshape_3d(ctx0, Kcur_ctx_proj, n_embd_head_k, n_head_kv, update_rows);
            Kcur_ctx = llm_build_norm(ctx0, Kcur_ctx, hparams, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, cb, il);
            cb(Kcur_ctx, "dflash_kv_k_norm", il);
            const float target_freq_base = hparams.dflash_backbone_rotary_base > 0.0f
                ? hparams.dflash_backbone_rotary_base : freq_base;
            Kcur_ctx = ggml_rope_ext(ctx0, Kcur_ctx, lctx.dflash.kv.cache_input_pos_ctx, nullptr,
                    n_rot, rope_type, n_ctx_orig, target_freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur_ctx, "dflash_kv_k_rope", il);
            Kcur_ctx = ggml_cont(ctx0, ggml_permute(ctx0, Kcur_ctx, 0, 2, 1, 3));
            cb(Kcur_ctx, "dflash_kv_k_physical", il);

            Vcur_ctx = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, layer_target);
            if (model.layers[il].bv) { Vcur_ctx = ggml_add(ctx0, Vcur_ctx, model.layers[il].bv); }
            cb(Vcur_ctx, "dflash_kv_v_proj", il);
            if (std::abs(hparams.f_attn_v_scale - 1.0f) > 1e-4f) {
                Vcur_ctx = ggml_scale(ctx0, Vcur_ctx, hparams.f_attn_v_scale);
                cb(Vcur_ctx, "dflash_kv_v_scaled", il);
            }
            Vcur_ctx = ggml_reshape_3d(ctx0, Vcur_ctx, n_embd_head_v, n_head_kv, update_rows);
            Vcur_ctx = ggml_cont(ctx0, ggml_permute(ctx0, Vcur_ctx, 0, 2, 1, 3));
            cb(Vcur_ctx, "dflash_kv_v_physical", il);
        }

        const int32_t first_rows = std::min<int32_t>((int32_t) update_rows, (int32_t) ctx_len - write_pos);
        const int32_t second_rows = (int32_t) update_rows - first_rows;

        if (first_rows > 0) {
            ggml_tensor * Ksrc_first = first_rows == update_rows
                ? Kcur_ctx
                : ggml_view_3d(ctx0, Kcur_ctx,
                    Kcur_ctx->ne[0],
                    first_rows,
                    Kcur_ctx->ne[2],
                    Kcur_ctx->nb[1],
                    Kcur_ctx->nb[2],
                    0);
            ggml_tensor * Vsrc_first = first_rows == update_rows
                ? Vcur_ctx
                : ggml_view_3d(ctx0, Vcur_ctx,
                    Vcur_ctx->ne[0],
                    first_rows,
                    Vcur_ctx->ne[2],
                    Vcur_ctx->nb[1],
                    Vcur_ctx->nb[2],
                    0);
            ggml_tensor * Kdst_first = ggml_view_3d(ctx0, lctx.dflash.kv.k_ctx_cache[il],
                lctx.dflash.kv.k_ctx_cache[il]->ne[0],
                first_rows,
                lctx.dflash.kv.k_ctx_cache[il]->ne[2],
                lctx.dflash.kv.k_ctx_cache[il]->nb[1],
                lctx.dflash.kv.k_ctx_cache[il]->nb[2],
                (size_t) write_pos * lctx.dflash.kv.k_ctx_cache[il]->nb[1]);
            ggml_tensor * Vdst_first = ggml_view_3d(ctx0, lctx.dflash.kv.v_ctx_cache[il],
                lctx.dflash.kv.v_ctx_cache[il]->ne[0],
                first_rows,
                lctx.dflash.kv.v_ctx_cache[il]->ne[2],
                lctx.dflash.kv.v_ctx_cache[il]->nb[1],
                lctx.dflash.kv.v_ctx_cache[il]->nb[2],
                (size_t) write_pos * lctx.dflash.kv.v_ctx_cache[il]->nb[1]);

            ggml_tensor * Kstore_first = ggml_cpy(ctx0, Ksrc_first, Kdst_first);
            cb(Kstore_first, "dflash_kv_k_store", il);
            ggml_build_forward_expand(gf, Kstore_first);

            ggml_tensor * Vstore_first = ggml_cpy(ctx0, Vsrc_first, Vdst_first);
            cb(Vstore_first, "dflash_kv_v_store", il);
            ggml_build_forward_expand(gf, Vstore_first);
        }

        if (second_rows > 0) {
            ggml_tensor * Ksrc_second = ggml_view_3d(ctx0, Kcur_ctx,
                Kcur_ctx->ne[0],
                second_rows,
                Kcur_ctx->ne[2],
                Kcur_ctx->nb[1],
                Kcur_ctx->nb[2],
                (size_t) first_rows * Kcur_ctx->nb[1]);
            ggml_tensor * Vsrc_second = ggml_view_3d(ctx0, Vcur_ctx,
                Vcur_ctx->ne[0],
                second_rows,
                Vcur_ctx->ne[2],
                Vcur_ctx->nb[1],
                Vcur_ctx->nb[2],
                (size_t) first_rows * Vcur_ctx->nb[1]);
            ggml_tensor * Kdst_second = ggml_view_3d(ctx0, lctx.dflash.kv.k_ctx_cache[il],
                lctx.dflash.kv.k_ctx_cache[il]->ne[0],
                second_rows,
                lctx.dflash.kv.k_ctx_cache[il]->ne[2],
                lctx.dflash.kv.k_ctx_cache[il]->nb[1],
                lctx.dflash.kv.k_ctx_cache[il]->nb[2],
                0);
            ggml_tensor * Vdst_second = ggml_view_3d(ctx0, lctx.dflash.kv.v_ctx_cache[il],
                lctx.dflash.kv.v_ctx_cache[il]->ne[0],
                second_rows,
                lctx.dflash.kv.v_ctx_cache[il]->ne[2],
                lctx.dflash.kv.v_ctx_cache[il]->nb[1],
                lctx.dflash.kv.v_ctx_cache[il]->nb[2],
                0);

            ggml_tensor * Kstore_second = ggml_cpy(ctx0, Ksrc_second, Kdst_second);
            cb(Kstore_second, "dflash_kv_k_store", il);
            ggml_build_forward_expand(gf, Kstore_second);

            ggml_tensor * Vstore_second = ggml_cpy(ctx0, Vsrc_second, Vdst_second);
            cb(Vstore_second, "dflash_kv_v_store", il);
            ggml_build_forward_expand(gf, Vstore_second);
        }
    }

    return gf;
}

ggml_cgraph * llm_build_context::build_dflash() {
    const int64_t n_embd_head_k = hparams.n_embd_head_k(0);
    const int64_t n_embd_head_v = hparams.n_embd_head_v(0);
    const int64_t n_target_features = hparams.dflash_n_target_features;
    const int64_t ctx_len = lctx.dflash.visible_cross_ctx > 0
            ? (int64_t) lctx.dflash.visible_cross_ctx
            : llama_dflash_default_cross_context(
                cparams.n_ctx,
                hparams.dflash_block_size,
                cparams.n_ubatch);
    const int64_t n_kv_total = GGML_PAD(ctx_len + n_tokens, (int64_t) llama_kv_cache::get_padding(flash_attn));

    GGML_ASSERT(n_embd_head_k == n_embd_head_v);
    GGML_ASSERT(n_target_features > 0);
    GGML_ASSERT(lctx.ensure_dflash_kv_cache_tensors((int32_t) ctx_len));

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0,
            model.max_nodes((int) std::max<int64_t>(n_tokens, ctx_len)) + 96 * n_layer +
                64 * std::max<int64_t>(hparams.dflash_block_size, n_tokens),
            false);

    lctx.dflash.draft_tokens_tensor = nullptr;
    lctx.dflash.draft_confidence_tensor = nullptr;

    const bool needs_swa_mask = hparams.n_swa > 0 && [&]() {
        for (int il = 0; il < n_layer; ++il) {
            if (hparams.swa_layers[il]) {
                return true;
            }
        }
        return false;
    }();
    const ggml_type mask_type = flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    // The full (non-SWA) mask is only consumed by non-SWA layers. For an all-SWA draft every layer
    // uses kq_mask_swa, leaving the full mask a dead graph node that the scheduler never backs with a
    // buffer (and the unconditional input-set then asserts buf!=NULL). So create each mask only when
    // some layer uses it: full mask iff any non-SWA layer; swa mask iff needs_swa_mask.
    const bool needs_full_mask = !needs_swa_mask || [&]() {
        for (int il = 0; il < n_layer; ++il) {
            if (!hparams.swa_layers[il]) {
                return true;
            }
        }
        return false;
    }();

    lctx.dflash.inputs.kq_mask = nullptr;
    lctx.dflash.kv.kq_mask_tensor = nullptr;
    ggml_tensor * dflash_kq_mask_full = nullptr;
    if (needs_full_mask) {
        lctx.dflash.inputs.kq_mask = ggml_new_tensor_2d(ctx0, mask_type, n_kv_total, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
        lctx.dflash.kv.kq_mask_tensor = lctx.dflash.inputs.kq_mask;
        ggml_set_input(lctx.dflash.inputs.kq_mask);
        cb(lctx.dflash.inputs.kq_mask, "dflash_kq_mask", -1);
        dflash_kq_mask_full = lctx.dflash.inputs.kq_mask;
    }

    lctx.dflash.kv.draft_tail_rows_tensor = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(lctx.dflash.kv.draft_tail_rows_tensor);
    cb(lctx.dflash.kv.draft_tail_rows_tensor, "dflash_draft_tail_rows", -1);

    ggml_tensor * dflash_kq_mask_swa = nullptr;
    lctx.dflash.inputs.kq_mask_swa = nullptr;
    lctx.dflash.kv.kq_mask_swa_tensor = nullptr;
    if (needs_swa_mask) {
        lctx.dflash.inputs.kq_mask_swa = ggml_new_tensor_2d(ctx0, mask_type, n_kv_total, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
        lctx.dflash.kv.kq_mask_swa_tensor = lctx.dflash.inputs.kq_mask_swa;
        ggml_set_input(lctx.dflash.inputs.kq_mask_swa);
        cb(lctx.dflash.inputs.kq_mask_swa, "dflash_kq_mask_swa", -1);
        dflash_kq_mask_swa = lctx.dflash.inputs.kq_mask_swa;
    }

    ggml_tensor * tok_embd = model.tok_embd;
    GGML_ASSERT(tok_embd != nullptr);

    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, tok_embd, cb);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = (n_tokens > 1 && n_outputs < n_tokens) ? build_inp_out_ids() : nullptr;

    const float kq_scale = 1.0f / std::sqrt((float) n_embd_head_k);

    if (hparams.dsv4_hc_mult > 0) {
        GGML_ASSERT(dflash_kq_mask_swa != nullptr);
        inpL = build_dflash_dsv4_layers(gf, inpL, inp_pos, dflash_kq_mask_swa);
    } else {
        for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        ggml_tensor * cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);
        ggml_tensor * input_normed = cur;

        ggml_tensor * Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
        ggml_tensor * Kcur_noise = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
        ggml_tensor * Vcur_noise = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
        if (model.layers[il].bq) { Qcur       = ggml_add(ctx0, Qcur,       model.layers[il].bq); }
        if (model.layers[il].bk) { Kcur_noise = ggml_add(ctx0, Kcur_noise, model.layers[il].bk); }
        if (model.layers[il].bv) { Vcur_noise = ggml_add(ctx0, Vcur_noise, model.layers[il].bv); }
        cb(Qcur, "Qcur", il);
        cb(Kcur_noise, "Kcur_noise", il);
        cb(Vcur_noise, "Vcur_noise", il);
        ggml_build_forward_expand(gf, Qcur);
        ggml_build_forward_expand(gf, Kcur_noise);
        ggml_build_forward_expand(gf, Vcur_noise);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head, n_tokens);
        Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(Qcur, "Qcur_normed", il);
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur_roped", il);

        Kcur_noise = ggml_reshape_3d(ctx0, Kcur_noise, n_embd_head_k, n_head_kv, n_tokens);
        Kcur_noise = llm_build_norm(ctx0, Kcur_noise, hparams, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(Kcur_noise, "Kcur_normed", il);
        Kcur_noise = ggml_rope_ext(ctx0, Kcur_noise, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Kcur_noise, "Kcur_roped", il);

        Vcur_noise = ggml_reshape_3d(ctx0, Vcur_noise, n_embd_head_v, n_head_kv, n_tokens);
        if (std::abs(hparams.f_attn_v_scale - 1.0f) > 1e-4f) {
            Vcur_noise = ggml_scale(ctx0, Vcur_noise, hparams.f_attn_v_scale);
            cb(Vcur_noise, "Vcur_noise_scaled", il);
        }
        cb(Vcur_noise, "Vcur_noise", il);

        GGML_ASSERT(il < (int32_t) lctx.dflash.kv.k_ctx_cache.size());
        GGML_ASSERT(il < (int32_t) lctx.dflash.kv.v_ctx_cache.size());
        GGML_ASSERT(lctx.dflash.kv.k_ctx_cache[il] != nullptr);
        GGML_ASSERT(lctx.dflash.kv.v_ctx_cache[il] != nullptr);
        GGML_ASSERT(lctx.dflash.kv.k_ctx_cache[il]->type == lctx.dflash.kv.v_ctx_cache[il]->type);
        GGML_ASSERT(lctx.dflash.kv.k_ctx_cache[il]->ne[1] >= n_kv_total);
        GGML_ASSERT(lctx.dflash.kv.v_ctx_cache[il]->ne[1] >= n_kv_total);

        //ggml_tensor * Kcur_draft = ggml_cont(ctx0, ggml_permute(ctx0, Kcur_noise, 0, 2, 1, 3));
        //ggml_tensor * Vcur_draft = ggml_cont(ctx0, ggml_permute(ctx0, Vcur_noise, 0, 2, 1, 3));
        ggml_tensor * Kcur_draft = ggml_permute(ctx0, Kcur_noise, 0, 2, 1, 3);
        ggml_tensor * Vcur_draft = ggml_permute(ctx0, Vcur_noise, 0, 2, 1, 3);
        cb(Kcur_draft, "dflash_main_k_perm_cont", il);
        cb(Vcur_draft, "dflash_main_v_perm_cont", il);

        ggml_tensor * Kcur = ggml_set_rows(ctx0, lctx.dflash.kv.k_ctx_cache[il], Kcur_draft, lctx.dflash.kv.draft_tail_rows_tensor);
        ggml_tensor * Vcur = ggml_set_rows(ctx0, lctx.dflash.kv.v_ctx_cache[il], Vcur_draft, lctx.dflash.kv.draft_tail_rows_tensor);
        cb(Kcur, "dflash_main_k_set_tail", il);
        cb(Vcur, "dflash_main_v_set_tail", il);

        if (Kcur->ne[1] != n_kv_total) {
            Kcur = ggml_view_3d(ctx0, Kcur,
                Kcur->ne[0],
                n_kv_total,
                Kcur->ne[2],
                Kcur->nb[1],
                Kcur->nb[2],
                0);
            cb(Kcur, "dflash_main_k_active_view", il);
        }
        if (Vcur->ne[1] != n_kv_total) {
            Vcur = ggml_view_3d(ctx0, Vcur,
                Vcur->ne[0],
                n_kv_total,
                Vcur->ne[2],
                Vcur->nb[1],
                Vcur->nb[2],
                0);
            cb(Vcur, "dflash_main_v_active_view", il);
        }

        if (Kcur->type == GGML_TYPE_F32) {
            Kcur = ggml_cast(ctx0, Kcur, GGML_TYPE_F16);
        }
        if (Vcur->type == GGML_TYPE_F32) {
            Vcur = ggml_cast(ctx0, Vcur, GGML_TYPE_F16);
        }

        cb(Qcur, "Qcur", il);

        ggml_tensor * q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        ggml_tensor * k = Kcur;
        ggml_tensor * v = Vcur;
        bool use_swa = hparams.swa_layers[il] && dflash_kq_mask_swa != nullptr;
        ggml_tensor * dflash_kq_mask_l = use_swa ? dflash_kq_mask_swa : dflash_kq_mask_full;
        cb(q, "q", il);

        cur = ggml_flash_attn_ext(ctx0, q, k, v, dflash_kq_mask_l, kq_scale, hparams.f_max_alibi_bias,
                hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        if (model.layers[il].attn_sinks) {
            ggml_flash_attn_ext_add_sinks(cur, model.layers[il].attn_sinks);
        }
        cb(cur, "flash_attn", il);
        ggml_build_forward_expand(gf, cur);
        // Somethiong goes wrong with thisi optimization.
        // I guess, the cross context does not mingle well with it.
        //if (use_swa) {
        //    cur->op_params[4] = hparams.n_swa;
        //}

        if (hparams.dflash_laguna) {
            GGML_ASSERT(model.layers[il].wqkv_gate != nullptr);
            ggml_tensor * gate = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv_gate, input_normed);
            gate = ggml_softplus(ctx0, gate);
            cb(gate, "attn_gate", il);
            GGML_ASSERT(gate->ne[0] == n_head);
            gate = ggml_reshape_3d(ctx0, gate, 1, n_head, n_tokens);
            cur = ggml_mul(ctx0, cur, gate);
            cb(cur, "attn_gated", il);
        }

        cur = ggml_reshape_2d(ctx0, cur, model.layers[il].wo->ne[0], n_tokens);
        cb(cur, "flash_attn_reshaped", il);

        cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
        if (model.layers[il].bo) { cur = ggml_add(ctx0, cur, model.layers[il].bo); }
        cb(cur, "kqv_out", il);

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        if (inp_out_ids != nullptr && il == n_layer - 1) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            cb(cur, "result_output_rows", -1);
        }

        ggml_tensor * ffn_residual = cur;
        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_post_norm", il);

        cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                model.layers[il].ffn_up, nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, false, false);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "l_out", il);

            inpL = cur;
        }
    }

    GGML_ASSERT(model.output_mtp != nullptr);
    ggml_tensor * result = build_output(lctx, ctx0, inpL, model.output_mtp, model.output_norm, cb);
    if (model.dflash_markov_w1 != nullptr) {
        cb(result, "dspark_base_logits", -1);
        result = build_dspark_markov_head(*this, gf, result, inpL);
    } else {
        cb(result, "result_output", -1);
        ggml_build_forward_expand(gf, result);
    }

    ggml_tensor * draft_tokens = ggml_argmax(ctx0, result);
    ggml_set_name(draft_tokens, "draft_argmax");
    ggml_build_forward_expand(gf, draft_tokens);
    lctx.dflash.draft_tokens_tensor = draft_tokens;

    return gf;
}
