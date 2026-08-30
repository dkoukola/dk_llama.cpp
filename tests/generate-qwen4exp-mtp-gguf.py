#!/usr/bin/env python3
"""Generate tiny Qwen3.8 embedded and predictor-only MTP GGUFs for tests."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-py"))

import gguf  # noqa: E402


def generate(output: Path, companion: bool = False) -> None:
    rng = np.random.default_rng(1234)
    writer = gguf.GGUFWriter(output, "qwen4exp")

    hidden = 128
    hc_count = 4
    hc_width = hidden * hc_count
    hc_rank = 2
    vocab = 16
    layers = 3
    heads = 2
    kv_heads = 1
    head_dim = 64
    indexer_heads = 1
    indexer_dim = 64
    experts = 2
    ff = 64
    ssm_state = 64
    ssm_inner = 64
    ssm_groups = 1
    ssm_dt_rank = 1
    ssm_conv = 4

    writer.add_name("synthetic-qwen4exp-mtp-companion" if companion else "synthetic-qwen4exp-mtp")
    writer.add_block_count(layers)
    writer.add_context_length(64)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(ff)
    writer.add_head_count(heads)
    writer.add_head_count_kv(kv_heads)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_expert_count(experts)
    writer.add_expert_used_count(1)
    writer.add_expert_feed_forward_length(ff)
    writer.add_expert_shared_feed_forward_length(ff)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_freq_base(10000.0)
    writer.add_rope_dimension_count(32)
    writer.add_rope_dimension_sections([16, 16, 0, 0])
    writer.add_ssm_conv_kernel(ssm_conv)
    writer.add_ssm_state_size(ssm_state)
    writer.add_ssm_group_count(ssm_groups)
    writer.add_ssm_time_step_rank(ssm_dt_rank)
    writer.add_ssm_inner_size(ssm_inner)
    writer.add_full_attention_interval(2)
    writer.add_nextn_predict_layers(1)
    writer.add_hyper_connection_count(hc_count)
    writer.add_hyper_connection_low_rank(hc_rank)
    writer.add_attention_indexer_head_count(indexer_heads)
    writer.add_attention_indexer_key_length(indexer_dim)
    writer.add_attention_indexer_top_k(2)
    writer.add_attention_compress_ratios([0, 2, 2])

    tokens = ["<unk>", "<s>", "</s>"] + [chr(ord("a") + i) for i in range(vocab - 3)]
    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * vocab)
    writer.add_token_types([
        gguf.TokenType.UNKNOWN,
        gguf.TokenType.CONTROL,
        gguf.TokenType.CONTROL,
    ] + [gguf.TokenType.NORMAL] * (vocab - 3))
    writer.add_unk_token_id(0)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)

    def rand(shape: tuple[int, ...]) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    def should_emit(name: str) -> bool:
        if not companion:
            return True
        return name in {"token_embd.weight", "output.weight"} or name.startswith(f"blk.{layers - 1}.")

    def add(name: str, shape: tuple[int, ...], norm: bool = False) -> None:
        tensor = np.ones(shape, dtype=np.float32) if norm else rand(shape)
        if should_emit(name):
            writer.add_tensor(name, tensor)

    token_embd = rand((vocab, hidden))
    writer.add_tensor("token_embd.weight", token_embd)
    writer.add_tensor("output.weight", token_embd.copy())
    add("output_hc_norm.weight", (hc_width,), norm=True)
    add("output_hc_down.weight", (hc_rank, hc_width))
    add("output_hc_up.weight", (hc_width, hc_rank))

    for layer in range(layers):
        prefix = f"blk.{layer}"
        add(f"{prefix}.hc_attn_norm.weight", (hc_width,), norm=True)
        add(f"{prefix}.hc_attn_down.weight", (hc_rank, hc_width))
        add(f"{prefix}.hc_attn_up.weight", (hc_width, hc_rank))
        add(f"{prefix}.hc_attn_inject.weight", (hc_count, hc_width))
        add(f"{prefix}.hc_ffn_norm.weight", (hc_width,), norm=True)
        add(f"{prefix}.hc_ffn_down.weight", (hc_rank, hc_width))
        add(f"{prefix}.hc_ffn_up.weight", (hc_width, hc_rank))
        add(f"{prefix}.hc_ffn_inject.weight", (hc_count, hc_width))

        if layer == 0:
            conv_dim = ssm_state * ssm_groups * 2 + ssm_inner
            add(f"{prefix}.attn_qkv.weight", (conv_dim, hidden))
            add(f"{prefix}.attn_gate.weight", (ssm_inner, hidden))
            add(f"{prefix}.ssm_conv1d.weight", (conv_dim, ssm_conv))
            add(f"{prefix}.ssm_dt.bias", (ssm_dt_rank,))
            add(f"{prefix}.ssm_a", (ssm_dt_rank,))
            add(f"{prefix}.ssm_beta.weight", (1, hidden))
            add(f"{prefix}.ssm_alpha.weight", (1, hidden))
            add(f"{prefix}.ssm_norm.weight", (ssm_inner,), norm=True)
            add(f"{prefix}.ssm_out.weight", (hidden, ssm_inner))
        else:
            add(f"{prefix}.attn_q.weight", (2 * heads * head_dim, hidden))
            add(f"{prefix}.attn_k.weight", (kv_heads * head_dim, hidden))
            add(f"{prefix}.attn_v.weight", (kv_heads * head_dim, hidden))
            add(f"{prefix}.attn_output.weight", (hidden, heads * head_dim))
            add(f"{prefix}.attn_q_norm.weight", (head_dim,), norm=True)
            add(f"{prefix}.attn_k_norm.weight", (head_dim,), norm=True)
            add(f"{prefix}.indexer.q_proj.weight", (indexer_heads * indexer_dim, hidden))
            add(f"{prefix}.indexer.k_proj.weight", (indexer_dim, hidden))
            add(f"{prefix}.indexer.q_norm.weight", (indexer_dim,), norm=True)
            add(f"{prefix}.indexer.k_norm.weight", (indexer_dim,), norm=True)

        add(f"{prefix}.ffn_gate_inp.weight", (experts, hidden))
        add(f"{prefix}.ffn_gate_exps.weight", (experts, ff, hidden))
        add(f"{prefix}.ffn_up_exps.weight", (experts, ff, hidden))
        add(f"{prefix}.ffn_down_exps.weight", (experts, hidden, ff))
        add(f"{prefix}.ffn_gate_inp_shexp.weight", (hidden,))
        add(f"{prefix}.ffn_gate_shexp.weight", (ff, hidden))
        add(f"{prefix}.ffn_up_shexp.weight", (ff, hidden))
        add(f"{prefix}.ffn_down_shexp.weight", (hidden, ff))

    add("blk.2.nextn.e_proj.weight", (hidden, hidden))
    if should_emit("blk.2.nextn.h_proj.weight"):
        writer.add_tensor("blk.2.nextn.h_proj.weight", np.eye(hidden, dtype=np.float32))
    add("blk.2.nextn.enorm.weight", (hidden,), norm=True)
    add("blk.2.nextn.hnorm.weight", (hc_width,), norm=True)
    add("blk.2.nextn.hc_norm.weight", (hc_width,), norm=True)
    add("blk.2.nextn.hc_down.weight", (hc_rank, hc_width))
    add("blk.2.nextn.hc_up.weight", (hc_width, hc_rank))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} EMBEDDED.gguf [COMPANION.gguf]")
    generate(Path(sys.argv[1]))
    if len(sys.argv) == 3:
        generate(Path(sys.argv[2]), companion=True)
