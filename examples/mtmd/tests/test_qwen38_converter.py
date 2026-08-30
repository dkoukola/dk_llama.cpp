from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from gguf import GGMLQuantizationType, GGUFReader
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[3]
CONVERTER = (
    REPO_ROOT
    / "examples"
    / "mtmd"
    / "legacy-models"
    / "qwen38-convert-image-encoder-to-gguf.py"
)


def make_config(model_dir: Path) -> None:
    config = {
        "_name_or_path": "synthetic-qwen38",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {"hidden_size": 6, "rms_norm_eps": 1e-6},
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 1,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 8,
            "in_channels": 3,
            "intermediate_size": 12,
            "num_heads": 2,
            "num_position_embeddings": 4,
            "out_hidden_size": 6,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    preprocessor = {
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "size": {"shortest_edge": 1024, "longest_edge": 262144},
    }
    (model_dir / "preprocessor_config.json").write_text(
        json.dumps(preprocessor), encoding="utf-8"
    )


def make_tensors() -> dict[str, torch.Tensor]:
    prefix = "model.visual."
    tensors: dict[str, torch.Tensor] = {
        prefix + "patch_embed.proj.weight": torch.arange(
            8 * 3 * 2 * 2 * 2, dtype=torch.float32
        ).reshape(8, 3, 2, 2, 2),
        prefix + "patch_embed.proj.bias": torch.arange(8, dtype=torch.float32),
        prefix + "pos_embed.weight": torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8),
        prefix + "merger.norm.weight": torch.ones(8),
        prefix + "merger.norm.bias": torch.zeros(8),
        prefix + "merger.linear_fc1.weight": torch.arange(
            32 * 32, dtype=torch.float32
        ).reshape(32, 32),
        prefix + "merger.linear_fc1.bias": torch.arange(32, dtype=torch.float32),
        prefix + "merger.linear_fc2.weight": torch.arange(
            6 * 32, dtype=torch.float32
        ).reshape(6, 32),
        prefix + "merger.linear_fc2.bias": torch.arange(6, dtype=torch.float32),
    }

    block = prefix + "blocks.0."
    tensors.update(
        {
            block + "attn.qkv.weight": torch.arange(24 * 8, dtype=torch.float32).reshape(24, 8),
            block + "attn.qkv.bias": torch.arange(24, dtype=torch.float32),
            block + "attn.proj.weight": torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8),
            block + "attn.proj.bias": torch.arange(8, dtype=torch.float32),
            block + "norm1.weight": torch.ones(8),
            block + "norm1.bias": torch.zeros(8),
            block + "norm2.weight": torch.ones(8),
            block + "norm2.bias": torch.zeros(8),
            block + "mlp.linear_fc1.weight": torch.arange(12 * 8, dtype=torch.float32).reshape(12, 8),
            block + "mlp.linear_fc1.bias": torch.arange(12, dtype=torch.float32),
            block + "mlp.linear_fc2.weight": torch.arange(8 * 12, dtype=torch.float32).reshape(8, 12),
            block + "mlp.linear_fc2.bias": torch.arange(8, dtype=torch.float32),
        }
    )
    return tensors


def write_checkpoint(model_dir: Path, tensors: dict[str, torch.Tensor]) -> None:
    first_names = sorted(tensors)[::2]
    second_names = sorted(tensors)[1::2]
    first_shard = "model-00001-of-00002.safetensors"
    second_shard = "model-00002-of-00002.safetensors"
    first = {name: tensors[name] for name in first_names}
    first["model.layers.0.self_attn.q_proj.weight"] = torch.ones(3, 3)
    second = {name: tensors[name] for name in second_names}
    save_file(first, model_dir / first_shard)
    save_file(second, model_dir / second_shard)

    weight_map = {name: first_shard for name in first}
    weight_map.update({name: second_shard for name in second})
    index = {"metadata": {}, "weight_map": weight_map}
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )


def run_converter(model_dir: Path, output: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "gguf-py")
    return subprocess.run(
        [sys.executable, str(CONVERTER), "-m", str(model_dir), "-o", str(output)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def expected_tensor_names() -> set[str]:
    names = {
        "v.patch_embd.weight",
        "v.patch_embd.weight.1",
        "v.patch_embd.bias",
        "v.position_embd.weight",
        "v.post_ln.weight",
        "v.post_ln.bias",
        "mm.0.weight",
        "mm.0.bias",
        "mm.2.weight",
        "mm.2.bias",
    }
    for suffix in (
        "attn_qkv.weight",
        "attn_qkv.bias",
        "attn_out.weight",
        "attn_out.bias",
        "ln1.weight",
        "ln1.bias",
        "ln2.weight",
        "ln2.bias",
        "ffn_up.weight",
        "ffn_up.bias",
        "ffn_down.weight",
        "ffn_down.bias",
    ):
        names.add(f"v.blk.0.{suffix}")
    return names


EXPECTED_METADATA = {
    "GGUF.version",
    "GGUF.tensor_count",
    "GGUF.kv_count",
    "general.architecture",
    "general.type",
    "general.name",
    "general.description",
    "general.file_type",
    "general.quantization_version",
    "clip.has_text_encoder",
    "clip.has_vision_encoder",
    "clip.has_audio_encoder",
    "clip.projector_type",
    "clip.use_gelu",
    "clip.vision.image_size",
    "clip.vision.patch_size",
    "clip.vision.embedding_length",
    "clip.vision.feed_forward_length",
    "clip.vision.projection_dim",
    "clip.vision.attention.head_count",
    "clip.vision.block_count",
    "clip.vision.attention.layer_norm_epsilon",
    "clip.vision.spatial_merge_size",
    "clip.vision.temporal_patch_size",
    "clip.vision.image_min_pixels",
    "clip.vision.image_max_pixels",
    "clip.vision.image_mean",
    "clip.vision.image_std",
}


def test_converter_writes_qwen3vl_tensor_abi(tmp_path: Path) -> None:
    make_config(tmp_path)
    source_tensors = make_tensors()
    write_checkpoint(tmp_path, source_tensors)
    output = tmp_path / "mmproj.gguf"

    result = run_converter(tmp_path, output)
    assert result.returncode == 0, result.stderr

    reader = GGUFReader(output)
    assert reader.fields["general.type"].contents() == "mmproj"
    assert reader.fields["clip.projector_type"].contents() == "qwen3vl_merger"
    assert reader.fields["clip.vision.image_size"].contents() == 4
    assert reader.fields["clip.vision.embedding_length"].contents() == 8
    assert reader.fields["clip.vision.projection_dim"].contents() == 6
    assert reader.fields["clip.vision.block_count"].contents() == 1
    assert reader.fields["clip.vision.image_min_pixels"].contents() == 1024
    assert reader.fields["clip.vision.image_max_pixels"].contents() == 262144
    assert set(reader.fields) == EXPECTED_METADATA

    tensors = {tensor.name: tensor for tensor in reader.tensors}
    assert set(tensors) == expected_tensor_names()
    assert "model.layers.0.self_attn.q_proj.weight" not in tensors
    assert tensors["v.patch_embd.weight"].tensor_type == GGMLQuantizationType.F16
    assert tensors["v.patch_embd.bias"].tensor_type == GGMLQuantizationType.F32
    assert tensors["v.position_embd.weight"].tensor_type == GGMLQuantizationType.F32
    assert tensors["mm.0.weight"].tensor_type == GGMLQuantizationType.F16

    patch = source_tensors["model.visual.patch_embed.proj.weight"]
    torch.testing.assert_close(
        torch.from_numpy(tensors["v.patch_embd.weight"].data.copy()),
        patch[:, :, 0].half(),
    )
    torch.testing.assert_close(
        torch.from_numpy(tensors["v.patch_embd.weight.1"].data.copy()),
        patch[:, :, 1].half(),
    )


def test_converter_rejects_unsupported_patch_shape(tmp_path: Path) -> None:
    make_config(tmp_path)
    tensors = make_tensors()
    tensors["model.visual.patch_embed.proj.weight"] = torch.zeros(8, 3, 3, 2, 2)
    write_checkpoint(tmp_path, tensors)
    output = tmp_path / "bad.gguf"

    result = run_converter(tmp_path, output)
    assert result.returncode != 0
    assert "unsupported shape for model.visual.patch_embed.proj.weight" in result.stderr


@pytest.mark.parametrize(
    "collision_name",
    ["config.json", "preprocessor_config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors"],
)
def test_converter_rejects_input_output_collision(tmp_path: Path, collision_name: str) -> None:
    make_config(tmp_path)
    write_checkpoint(tmp_path, make_tensors())
    collision = tmp_path / collision_name
    original = collision.read_bytes()

    result = run_converter(tmp_path, collision)
    assert result.returncode != 0
    assert "output path collides with converter input" in result.stderr
    assert collision.read_bytes() == original


def test_converter_rejects_hardlink_alias_to_input(tmp_path: Path) -> None:
    make_config(tmp_path)
    write_checkpoint(tmp_path, make_tensors())
    source = tmp_path / "model-00001-of-00002.safetensors"
    alias = tmp_path / "alias.gguf"
    try:
        os.link(source, alias)
    except OSError as error:
        pytest.skip(f"hard links unavailable: {error}")
    original = source.read_bytes()

    result = run_converter(tmp_path, alias)
    assert result.returncode != 0
    assert "output path aliases converter input" in result.stderr
    assert source.read_bytes() == original


def test_failed_conversion_preserves_existing_output(tmp_path: Path) -> None:
    make_config(tmp_path)
    tensors = make_tensors()
    tensors["model.visual.patch_embed.proj.weight"] = torch.zeros(
        8, 3, 2, 2, 2, dtype=torch.int32
    )
    write_checkpoint(tmp_path, tensors)
    output = tmp_path / "existing.gguf"
    output.write_bytes(b"existing output")

    result = run_converter(tmp_path, output)
    assert result.returncode != 0
    assert "source tensor must have a floating-point dtype" in result.stderr
    assert output.read_bytes() == b"existing output"
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def find_mtmd_smoke() -> Path | None:
    configured = os.environ.get("QWEN38_MTMD_SMOKE")
    if configured:
        return Path(configured)
    for build_dir in sorted(REPO_ROOT.glob("build*")):
        candidate = build_dir / "bin" / "test-qwen38-mmproj-load"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def test_mtmd_load_warmup_and_image_limits(tmp_path: Path) -> None:
    smoke = find_mtmd_smoke()
    if smoke is None:
        pytest.skip("test-qwen38-mmproj-load has not been built")
    make_config(tmp_path)
    write_checkpoint(tmp_path, make_tensors())
    output = tmp_path / "mmproj.gguf"
    result = run_converter(tmp_path, output)
    assert result.returncode == 0, result.stderr

    smoke_result = subprocess.run(
        [str(smoke), str(output)], check=False, capture_output=True, text=True
    )
    assert smoke_result.returncode == 0, smoke_result.stdout + smoke_result.stderr


@pytest.mark.parametrize("key", ["spatial_merge_size", "temporal_patch_size"])
def test_converter_rejects_unsupported_graph_configuration(
    tmp_path: Path, key: str
) -> None:
    make_config(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["vision_config"][key] = 3
    config_path.write_text(json.dumps(config), encoding="utf-8")
    write_checkpoint(tmp_path, make_tensors())

    result = run_converter(tmp_path, tmp_path / "bad.gguf")
    assert result.returncode != 0
    assert f"vision_config.{key} must be 2" in result.stderr
