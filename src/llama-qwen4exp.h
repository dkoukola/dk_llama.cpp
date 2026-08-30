#pragma once

#include "llama.h"

struct ggml_tensor;
struct llama_batch;
struct llama_context;

bool llama_qwen4exp_spec_ckpt_prepare(llama_context * ctx, int max_tokens);
bool llama_qwen4exp_spec_ckpt_save(llama_context * ctx, llama_seq_id seq_id);
bool llama_qwen4exp_spec_ckpt_begin_capture(llama_context * ctx, const llama_batch & batch);
void llama_qwen4exp_spec_ckpt_finish_capture(llama_context * ctx, int32_t n_tokens);
enum llama_spec_ckpt_restore_result llama_qwen4exp_spec_ckpt_restore(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_pos n_past,
        int accepted_step);
void llama_qwen4exp_spec_ckpt_discard(llama_context * ctx);
ggml_tensor * llama_qwen4exp_spec_ckpt_ple(llama_context * ctx, int32_t il);
