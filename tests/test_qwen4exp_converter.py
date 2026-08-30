import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))

import gguf
from convert_hf_to_gguf import Qwen2MoeModel, Qwen4ExpModel


def minimal_config():
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "image_token_id": 99,
        "text_config": {
            "num_hidden_layers": 4,
            "max_position_embeddings": 1024,
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 4,
            "shared_expert_intermediate_size": 4,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [1, 0, 0],
            },
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 2,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_value_head_dim": 2,
            "full_attention_interval": 4,
            "hc_count": 2,
            "hc_lowrank": 2,
            "indexer_n_heads": 2,
            "indexer_head_dim": 2,
            "indexer_budget": 8,
            "indexer_compress_ratio": 2,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "ple_layer_ids": [2],
            "ngram_size": 3,
            "heads_per_ngram": 1,
            "ple_conv_kernel_size": 4,
            "split_ngram_parts": 2,
            "eos_token_id": 7,
        },
    }


def make_model(tmp_path):
    Qwen4ExpModel.mtp_only = False
    (tmp_path / "config.json").write_text(json.dumps(minimal_config()), encoding="utf-8")
    model = Qwen4ExpModel(
        tmp_path,
        gguf.LlamaFileType.MOSTLY_F16,
        tmp_path / "model.gguf",
    )
    model._ple_row_dim = 8
    model._ple_multipliers = [23703573157769, 20109073645365, 8052911324071]
    model._ple_head_offsets = [0, 2]
    model._ple_head_vocab_sizes = [2, 2]
    return model


def make_mtp_model(tmp_path):
    Qwen4ExpModel.mtp_only = False
    config = minimal_config()
    config["text_config"]["mtp_num_hidden_layers"] = 1
    config["text_config"]["mtp"] = {
        "num_hidden_layers": 1,
        "layer_types": ["full_attention"],
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    model = Qwen4ExpModel(
        tmp_path,
        gguf.LlamaFileType.MOSTLY_F16,
        tmp_path / "model.gguf",
    )
    model._ple_row_dim = 8
    model._ple_multipliers = [23703573157769, 20109073645365, 8052911324071]
    model._ple_head_offsets = [0, 2]
    model._ple_head_vocab_sizes = [2, 2]
    return model


def test_qwen4exp_metadata_matches_runtime_keys(tmp_path):
    model = make_model(tmp_path)
    model.set_gguf_parameters()
    metadata = model.gguf_writer.kv_data[0]

    assert metadata["general.architecture"].value == "qwen4exp"
    assert metadata["qwen4exp.block_count"].value == 4
    assert metadata["qwen4exp.full_attention_interval"].value == 4
    assert metadata["qwen4exp.ssm.group_count"].value == 2
    assert metadata["qwen4exp.hyper_connection.count"].value == 2
    assert metadata["qwen4exp.attention.indexer.head_count"].value == 2
    assert metadata["qwen4exp.attention.compress_ratios"].value == [0, 0, 0, 2]
    assert metadata["qwen4exp.ple.layers"].value == [1]
    assert metadata["qwen4exp.ple.image_token_id"].value == 99
    assert metadata["qwen4exp.ple.layer_multipliers"].sub_type == gguf.GGUFValueType.UINT64
    assert metadata["qwen4exp.ple.layer_multipliers"].value == [
        23703573157769,
        20109073645365,
        8052911324071,
    ]


def test_qwen4exp_reads_hash_constants_without_float_rounding(tmp_path):
    model = make_model(tmp_path)
    expected = [23703573157769, 20109073645365, 8052911324071]
    name = "model.layers.1.ple.ple_embedding.layer_multipliers"
    model._tensor_sources[name] = (name, "unused.safetensors")
    model._load_checkpoint_tensor = lambda name: torch.tensor(expected, dtype=torch.int64)

    assert model._read_ple_constant("ple_embedding.layer_multipliers") == expected


def test_qwen4exp_streams_ple_shards_in_numeric_order(tmp_path):
    model = make_model(tmp_path)
    loaded = []
    chunks = {
        "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": torch.arange(8).reshape(1, 8),
        "model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight": torch.arange(8, 16).reshape(1, 8),
    }

    def load_tensor(name):
        loaded.append(name)
        return chunks[name]

    model._load_checkpoint_tensor = load_tensor
    assert list(model._place_ple_shard(chunks[next(iter(chunks))], next(iter(chunks)))) == []
    second_name = list(chunks)[1]
    [(name, table)] = model._place_ple_shard(chunks[second_name], second_name)

    assert name == "per_layer_token_embd.weight"
    assert loaded == []
    assert isinstance(table, gguf.LazyChunkedTensor)
    output = tmp_path / "ple.bin"
    with output.open("wb") as output_file:
        table.tofile(output_file)
    assert loaded == list(chunks)
    np.testing.assert_array_equal(np.fromfile(output, dtype=np.float32), np.arange(16, dtype=np.float32))


def test_qwen4exp_applies_official_fp8_ple_weight_scale_while_streaming(tmp_path, monkeypatch):
    model = make_model(tmp_path)
    prefix = "model.layers.1.ple.ple_embedding"
    constant_names = [
        f"{prefix}.layer_multipliers",
        f"{prefix}.ngram_heads_offsets",
        f"{prefix}.ngram_heads_vocab_sizes",
    ]
    scale_name = f"{prefix}.ngram_embedding.weight_scale"
    shard_names = [
        f"{prefix}.ngram_embedding.shard_0.weight",
        f"{prefix}.ngram_embedding.shard_1.weight",
    ]
    scale = torch.tensor([0.00019931793212890625], dtype=torch.bfloat16)
    tensors = {
        constant_names[0]: torch.tensor(model._ple_multipliers, dtype=torch.int64),
        constant_names[1]: torch.tensor(model._ple_head_offsets, dtype=torch.int64),
        constant_names[2]: torch.tensor(model._ple_head_vocab_sizes, dtype=torch.int64),
        scale_name: scale,
        shard_names[0]: torch.arange(8, dtype=torch.float32).reshape(1, 8).to(torch.float8_e4m3fn),
        shard_names[1]: torch.arange(8, 16, dtype=torch.float32).reshape(1, 8).to(torch.float8_e4m3fn),
    }
    model._tensor_sources = {name: (name, "unused.safetensors") for name in tensors}
    loaded = []

    def load_tensor(name):
        loaded.append(name)
        return tensors[name]

    emitted = []

    def prepare_base(self):
        assert self._ple_weight_scale == float(scale.float().item())
        assert list(self.modify_tensors(scale, scale_name, 1)) == []
        assert list(self._place_ple_shard(tensors[shard_names[0]], shard_names[0])) == []
        emitted.extend(self._place_ple_shard(tensors[shard_names[1]], shard_names[1]))

    model._load_checkpoint_tensor = load_tensor
    monkeypatch.setattr(Qwen2MoeModel, "prepare_tensors", prepare_base)
    model.prepare_tensors()

    assert loaded == constant_names + [scale_name]
    [(name, table)] = emitted
    assert name == "per_layer_token_embd.weight"
    assert isinstance(table, gguf.LazyChunkedTensor)
    output = tmp_path / "scaled-ple.bin"
    with output.open("wb") as output_file:
        table.tofile(output_file)

    expected = torch.cat([tensors[shard_name].float() for shard_name in shard_names])
    expected.mul_(float(scale.float().item()))
    np.testing.assert_array_equal(np.fromfile(output, dtype=np.float32).reshape(2, 8), expected.numpy())
    assert loaded == constant_names + [scale_name] + shard_names


def test_qwen4exp_splits_indexer_projection_and_reorders_linear_value_heads(tmp_path):
    model = make_model(tmp_path)
    indexer = torch.arange(6 * 8, dtype=torch.float32).reshape(6, 8)
    mapped = list(model.modify_tensors(
        indexer,
        "model.layers.3.self_attn.indexer.index_qk_proj.weight",
        3,
    ))
    assert [(name, tuple(tensor.shape)) for name, tensor in mapped] == [
        ("blk.3.indexer.q_proj.weight", (4, 8)),
        ("blk.3.indexer.k_proj.weight", (2, 8)),
    ]

    qkv = torch.arange(32, dtype=torch.float32).reshape(16, 2)
    [(name, reordered)] = model.modify_tensors(
        qkv,
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        0,
    )
    assert name == "blk.0.attn_qkv.weight"
    expected_v_rows = qkv[8:]
    expected_v_rows = expected_v_rows.reshape(2, 2, 2, 2).permute(1, 0, 2, 3).reshape(8, 2)
    torch.testing.assert_close(reordered[8:], expected_v_rows)


def test_qwen4exp_embeds_one_mtp_tail_in_block_metadata(tmp_path):
    model = make_mtp_model(tmp_path)
    model.set_gguf_parameters()
    metadata = model.gguf_writer.kv_data[0]

    assert model.block_count == 5
    assert metadata["qwen4exp.block_count"].value == 5
    assert metadata["qwen4exp.nextn_predict_layers"].value == 1
    assert metadata["qwen4exp.attention.compress_ratios"].value == [0, 0, 0, 2, 2]


def test_qwen4exp_standalone_mtp_selects_only_io_and_predictor_tensors(tmp_path):
    config = minimal_config()
    config["text_config"]["mtp_num_hidden_layers"] = 1
    config["text_config"]["mtp"] = {
        "num_hidden_layers": 1,
        "layer_types": ["full_attention"],
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    tensors = {
        "model.language_model.embed_tokens.weight": torch.zeros((16, 8)),
        "lm_head.weight": torch.ones((16, 8)),
        "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight": torch.zeros(16),
        "model.mtp.fc_embedding.weight": torch.zeros((8, 8)),
        "model.mtp.layers.0.mlp.experts.0.gate_proj.weight": torch.ones((4, 4)),
        "model.mtp.layers.0.mlp.experts.0.gate_proj.weight_scale_inv": torch.ones((2, 2)),
        "model.mtp.layers.0.mlp.experts.1.gate_proj.weight": torch.ones((4, 4)),
        "model.mtp.layers.0.mlp.experts.1.gate_proj.weight_scale_inv": torch.ones((2, 2)),
        "model.mtp.layers.0.self_attn.q_proj.weight": torch.zeros((16, 8)),
    }
    shard_name = "model.safetensors"
    save_file(tensors, tmp_path / shard_name)

    Qwen4ExpModel.mtp_only = True
    try:
        model = Qwen4ExpModel(
            tmp_path,
            gguf.LlamaFileType.MOSTLY_F16,
            tmp_path / "mtp.gguf",
        )
        selected = []
        mapped = []
        for name, tensor in model.get_tensors():
            selected.append(name)
            mapped.extend(model.modify_tensors(tensor, name, None))
        assert selected == [
            "lm_head.weight",
            "model.embed_tokens.weight",
            "mtp.fc_embedding.weight",
            "mtp.layers.0.mlp.experts.0.gate_proj.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
        ]
        assert sum(name == "blk.4.ffn_gate_exps.weight" for name, _ in mapped) == 1

        model.set_gguf_parameters()
        metadata = model.gguf_writer.kv_data[0]
        assert metadata["qwen4exp.block_count"].value == 5
        assert metadata["qwen4exp.nextn_predict_layers"].value == 1
        assert metadata["qwen4exp.attention.compress_ratios"].value == [0, 0, 0, 2, 2]
        assert "qwen4exp.ple.layers" not in metadata
    finally:
        Qwen4ExpModel.mtp_only = False


def test_qwen4exp_maps_mtp_fusion_and_decoder_weights_to_appended_block(tmp_path):
    model = make_mtp_model(tmp_path)

    special = {
        "mtp.fc_embedding.weight": "blk.4.nextn.e_proj.weight",
        "mtp.fc_hidden.weight": "blk.4.nextn.h_proj.weight",
        "mtp.pre_fc_norm_embedding.weight": "blk.4.nextn.enorm.weight",
        "mtp.pre_fc_norm_hidden.weight": "blk.4.nextn.hnorm.weight",
        "mtp.hyper_connection_mixer.hc_norm.weight": "blk.4.nextn.hc_norm.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight": "blk.4.nextn.hc_down.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight": "blk.4.nextn.hc_up.weight",
    }
    for source_name, expected_name in special.items():
        tensor = torch.zeros(8)
        [(name, converted)] = model.modify_tensors(tensor, source_name, None)
        assert name == expected_name
        if source_name.endswith(("norm_embedding.weight", "norm_hidden.weight", "hc_norm.weight")):
            torch.testing.assert_close(converted, torch.ones_like(tensor))
        else:
            torch.testing.assert_close(converted, tensor)

    indexer = torch.arange(6 * 8, dtype=torch.float32).reshape(6, 8)
    assert [name for name, _ in model.modify_tensors(
        indexer,
        "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
        None,
    )] == [
        "blk.4.indexer.q_proj.weight",
        "blk.4.indexer.k_proj.weight",
    ]

    decoder_names = {
        "mtp.layers.0.attn_hyper_connection.hc_norm.weight": "blk.4.hc_attn_norm.weight",
        "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight": "blk.4.hc_attn_inject.weight",
        "mtp.layers.0.self_attn.q_proj.weight": "blk.4.attn_q.weight",
        "mtp.layers.0.mlp.shared_expert_gate.weight": "blk.4.ffn_gate_inp_shexp.weight",
    }
    for source_name, expected_name in decoder_names.items():
        [(name, _)] = model.modify_tensors(torch.zeros(8), source_name, None)
        assert name == expected_name


def test_qwen4exp_dequantizes_and_stacks_mtp_fp8_experts(tmp_path):
    model = make_mtp_model(tmp_path)
    outputs = []

    for projection in ("down_proj", "gate_proj", "up_proj"):
        for expert_id in range(2):
            name = f"mtp.layers.0.mlp.experts.{expert_id}.{projection}.weight"
            scale_name = name + "_scale_inv"
            model._tensor_sources[scale_name] = (scale_name, "unused.safetensors")

            weight = torch.full((4, 4), float(expert_id + 1))
            scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            assert list(model.modify_tensors(weight, name, None)) == []
            outputs.extend(model.modify_tensors(scale, scale_name, None))

    assert [name for name, _ in outputs] == [
        "blk.4.ffn_down_exps.weight",
        "blk.4.ffn_gate_exps.weight",
        "blk.4.ffn_up_exps.weight",
    ]
    for _, tensor in outputs:
        assert tensor.shape == (2, 4, 4)
        expected_scale = torch.tensor([
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0],
            [3.0, 3.0, 4.0, 4.0],
        ])
        torch.testing.assert_close(tensor[0], expected_scale)
        torch.testing.assert_close(tensor[1], 2 * expected_scale)


def test_qwen4exp_streams_complete_mtp_expert_groups(tmp_path):
    model = make_mtp_model(tmp_path)
    tensors = {}
    for expert_id in range(2):
        weight_name = f"mtp.layers.0.mlp.experts.{expert_id}.gate_proj.weight"
        scale_name = weight_name + "_scale_inv"
        model._tensor_sources[weight_name] = (weight_name, "unused.safetensors")
        model._tensor_sources[scale_name] = (scale_name, "unused.safetensors")
        tensors[weight_name] = torch.full((4, 4), float(expert_id + 1))
        tensors[scale_name] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    loaded = []

    def load_tensor(name):
        loaded.append(name)
        return tensors[name]

    model._load_checkpoint_tensor = load_tensor
    first_name = "mtp.layers.0.mlp.experts.0.gate_proj.weight"
    [(name, table)] = model.modify_tensors(tensors[first_name], first_name, None)
    assert name == "blk.4.ffn_gate_exps.weight"
    assert isinstance(table, gguf.LazyChunkedTensor)
    assert table.shape == (2, 4, 4)
    assert loaded == []

    output = tmp_path / "experts.bin"
    with output.open("wb") as output_file:
        table.tofile(output_file)
    expected_scale = np.array([
        [1.0, 1.0, 2.0, 2.0],
        [1.0, 1.0, 2.0, 2.0],
        [3.0, 3.0, 4.0, 4.0],
        [3.0, 3.0, 4.0, 4.0],
    ], dtype=np.float32)
    expected = np.stack((expected_scale, 2 * expected_scale))
    np.testing.assert_array_equal(np.fromfile(output, dtype=np.float32).reshape(2, 4, 4), expected)
    assert loaded == [
        "mtp.layers.0.mlp.experts.0.gate_proj.weight",
        "mtp.layers.0.mlp.experts.0.gate_proj.weight_scale_inv",
        "mtp.layers.0.mlp.experts.1.gate_proj.weight",
        "mtp.layers.0.mlp.experts.1.gate_proj.weight_scale_inv",
    ]
