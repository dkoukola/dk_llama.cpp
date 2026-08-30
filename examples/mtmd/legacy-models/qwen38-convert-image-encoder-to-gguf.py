#!/usr/bin/env python3
"""Convert the Qwen3.8-Flash-Next vision tower to an mtmd GGUF."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from gguf import GGML_QUANT_VERSION, GGUFEndian, GGUFWriter
from safetensors import safe_open


VISION_PREFIX = "model.visual."


@dataclass(frozen=True)
class OutputTensor:
    name: str
    shape: tuple[int, ...]
    temporal_index: int | None = None
    force_f32: bool = False


@dataclass(frozen=True)
class TensorPlan:
    source_name: str
    shard_name: str
    source_shape: tuple[int, ...]
    outputs: tuple[OutputTensor, ...]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        value = json.load(config_file)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def required_int(config: dict[str, Any], key: str, section: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{section}.{key} must be a positive integer")
    return value


def load_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = read_json(index_path)
        raw_weight_map = index.get("weight_map")
        if not isinstance(raw_weight_map, dict):
            raise ValueError(f"weight_map is missing from {index_path}")

        weight_map: dict[str, str] = {}
        for name, shard_name in raw_weight_map.items():
            if not isinstance(name, str) or not isinstance(shard_name, str):
                raise ValueError(f"invalid weight_map entry in {index_path}")
            weight_map[name] = shard_name
        return weight_map

    shards = sorted(model_dir.glob("*.safetensors"))
    if len(shards) != 1:
        raise FileNotFoundError(
            f"expected model.safetensors.index.json or one safetensors file in {model_dir}"
        )

    with safe_open(shards[0], framework="pt", device="cpu") as shard:
        return {name: shards[0].name for name in shard.keys()}


def add_spec(
    specs: dict[str, tuple[tuple[int, ...], tuple[OutputTensor, ...]]],
    source_name: str,
    source_shape: tuple[int, ...],
    output_name: str,
    *,
    force_f32: bool = False,
) -> None:
    specs[source_name] = (
        source_shape,
        (OutputTensor(output_name, source_shape, force_f32=force_f32),),
    )


def build_tensor_specs(
    vision_config: dict[str, Any],
    projection_dim: int,
) -> dict[str, tuple[tuple[int, ...], tuple[OutputTensor, ...]]]:
    hidden_size = required_int(vision_config, "hidden_size", "vision_config")
    intermediate_size = required_int(vision_config, "intermediate_size", "vision_config")
    block_count = required_int(vision_config, "depth", "vision_config")
    input_channels = required_int(vision_config, "in_channels", "vision_config")
    patch_size = required_int(vision_config, "patch_size", "vision_config")
    temporal_patch_size = required_int(
        vision_config, "temporal_patch_size", "vision_config"
    )
    position_count = required_int(
        vision_config, "num_position_embeddings", "vision_config"
    )
    merge_size = required_int(vision_config, "spatial_merge_size", "vision_config")

    if temporal_patch_size != 2:
        raise ValueError(
            "vision_config.temporal_patch_size must be 2 for the qwen3vl mtmd graph"
        )
    if merge_size != 2:
        raise ValueError(
            "vision_config.spatial_merge_size must be 2 for the qwen3vl mtmd graph"
        )

    merged_size = hidden_size * merge_size * merge_size
    specs: dict[str, tuple[tuple[int, ...], tuple[OutputTensor, ...]]] = {}

    patch_source = VISION_PREFIX + "patch_embed.proj.weight"
    patch_shape = (
        hidden_size,
        input_channels,
        temporal_patch_size,
        patch_size,
        patch_size,
    )
    patch_output_shape = (hidden_size, input_channels, patch_size, patch_size)
    specs[patch_source] = (
        patch_shape,
        (
            OutputTensor("v.patch_embd.weight", patch_output_shape, temporal_index=0),
            OutputTensor("v.patch_embd.weight.1", patch_output_shape, temporal_index=1),
        ),
    )
    add_spec(
        specs,
        VISION_PREFIX + "patch_embed.proj.bias",
        (hidden_size,),
        "v.patch_embd.bias",
    )
    add_spec(
        specs,
        VISION_PREFIX + "pos_embed.weight",
        (position_count, hidden_size),
        "v.position_embd.weight",
        force_f32=True,
    )

    add_spec(
        specs,
        VISION_PREFIX + "merger.norm.weight",
        (hidden_size,),
        "v.post_ln.weight",
    )
    add_spec(
        specs,
        VISION_PREFIX + "merger.norm.bias",
        (hidden_size,),
        "v.post_ln.bias",
    )
    add_spec(
        specs,
        VISION_PREFIX + "merger.linear_fc1.weight",
        (merged_size, merged_size),
        "mm.0.weight",
    )
    add_spec(
        specs,
        VISION_PREFIX + "merger.linear_fc1.bias",
        (merged_size,),
        "mm.0.bias",
    )
    add_spec(
        specs,
        VISION_PREFIX + "merger.linear_fc2.weight",
        (projection_dim, merged_size),
        "mm.2.weight",
    )
    add_spec(
        specs,
        VISION_PREFIX + "merger.linear_fc2.bias",
        (projection_dim,),
        "mm.2.bias",
    )

    for block in range(block_count):
        source_prefix = f"{VISION_PREFIX}blocks.{block}."
        output_prefix = f"v.blk.{block}."
        add_spec(
            specs,
            source_prefix + "attn.qkv.weight",
            (3 * hidden_size, hidden_size),
            output_prefix + "attn_qkv.weight",
        )
        add_spec(
            specs,
            source_prefix + "attn.qkv.bias",
            (3 * hidden_size,),
            output_prefix + "attn_qkv.bias",
        )
        add_spec(
            specs,
            source_prefix + "attn.proj.weight",
            (hidden_size, hidden_size),
            output_prefix + "attn_out.weight",
        )
        add_spec(
            specs,
            source_prefix + "attn.proj.bias",
            (hidden_size,),
            output_prefix + "attn_out.bias",
        )
        add_spec(
            specs,
            source_prefix + "norm1.weight",
            (hidden_size,),
            output_prefix + "ln1.weight",
        )
        add_spec(
            specs,
            source_prefix + "norm1.bias",
            (hidden_size,),
            output_prefix + "ln1.bias",
        )
        add_spec(
            specs,
            source_prefix + "norm2.weight",
            (hidden_size,),
            output_prefix + "ln2.weight",
        )
        add_spec(
            specs,
            source_prefix + "norm2.bias",
            (hidden_size,),
            output_prefix + "ln2.bias",
        )
        add_spec(
            specs,
            source_prefix + "mlp.linear_fc1.weight",
            (intermediate_size, hidden_size),
            output_prefix + "ffn_up.weight",
        )
        add_spec(
            specs,
            source_prefix + "mlp.linear_fc1.bias",
            (intermediate_size,),
            output_prefix + "ffn_up.bias",
        )
        add_spec(
            specs,
            source_prefix + "mlp.linear_fc2.weight",
            (hidden_size, intermediate_size),
            output_prefix + "ffn_down.weight",
        )
        add_spec(
            specs,
            source_prefix + "mlp.linear_fc2.bias",
            (hidden_size,),
            output_prefix + "ffn_down.bias",
        )

    return specs


def validate_config(config: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or "Qwen4ExpForConditionalGeneration" not in architectures:
        raise ValueError(
            "config.architectures must contain Qwen4ExpForConditionalGeneration"
        )

    vision_config = config.get("vision_config")
    text_config = config.get("text_config")
    if not isinstance(vision_config, dict) or not isinstance(text_config, dict):
        raise ValueError("config.json must contain vision_config and text_config objects")

    if vision_config.get("deepstack_visual_indexes", []) != []:
        raise ValueError("Qwen3.8 deepstack vision layers are not supported by this converter")

    hidden_size = required_int(vision_config, "hidden_size", "vision_config")
    head_count = required_int(vision_config, "num_heads", "vision_config")
    if hidden_size % head_count != 0 or (hidden_size // head_count) % 4 != 0:
        raise ValueError(
            "vision hidden_size / num_heads must be an integer divisible by 4"
        )

    position_count = required_int(
        vision_config, "num_position_embeddings", "vision_config"
    )
    position_side = math.isqrt(position_count)
    if position_side * position_side != position_count:
        raise ValueError("vision_config.num_position_embeddings must be a square")

    projection_dim = required_int(text_config, "hidden_size", "text_config")
    out_hidden_size = required_int(
        vision_config, "out_hidden_size", "vision_config"
    )
    if out_hidden_size != projection_dim:
        raise ValueError(
            "vision_config.out_hidden_size must match text_config.hidden_size"
        )

    patch_size = required_int(vision_config, "patch_size", "vision_config")
    image_size = position_side * patch_size
    return vision_config, projection_dim, image_size


def build_plan(
    model_dir: Path,
    weight_map: dict[str, str],
    specs: dict[str, tuple[tuple[int, ...], tuple[OutputTensor, ...]]],
) -> list[TensorPlan]:
    found_names = {name for name in weight_map if name.startswith(VISION_PREFIX)}
    expected_names = set(specs)
    missing_names = sorted(expected_names - found_names)
    unexpected_names = sorted(found_names - expected_names)
    if missing_names:
        raise ValueError(f"missing vision tensor: {missing_names[0]}")
    if unexpected_names:
        raise ValueError(f"unsupported vision tensor: {unexpected_names[0]}")

    plan: list[TensorPlan] = []
    shard_names: dict[str, list[str]] = {}
    for source_name in sorted(expected_names):
        shard_name = weight_map[source_name]
        shard_names.setdefault(shard_name, []).append(source_name)

    seen_outputs: set[str] = set()
    for shard_name in sorted(shard_names):
        shard_path = model_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"missing safetensors shard {shard_path}")

        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            shard_keys = set(shard.keys())
            for source_name in shard_names[shard_name]:
                if source_name not in shard_keys:
                    raise ValueError(
                        f"{source_name} is assigned to {shard_name} but is not present"
                    )
                source_shape = tuple(int(value) for value in shard.get_slice(source_name).get_shape())
                expected_shape, outputs = specs[source_name]
                if source_shape != expected_shape:
                    raise ValueError(
                        f"unsupported shape for {source_name}: got {source_shape}, "
                        f"expected {expected_shape}"
                    )
                for output in outputs:
                    if output.name in seen_outputs:
                        raise ValueError(f"duplicate output tensor name {output.name}")
                    seen_outputs.add(output.name)
                plan.append(TensorPlan(source_name, shard_name, source_shape, outputs))

    return plan


def output_dtype(output: OutputTensor, use_f32: bool) -> np.dtype[Any]:
    if use_f32 or output.force_f32 or len(output.shape) <= 1:
        return np.dtype(np.float32)
    return np.dtype(np.float16)


def convert_output_tensor(
    source: torch.Tensor,
    output: OutputTensor,
    use_f32: bool,
) -> np.ndarray[Any, Any]:
    if not source.is_floating_point():
        raise ValueError(f"{output.name} source tensor must have a floating-point dtype")

    selected = source
    if output.temporal_index is not None:
        selected = selected[:, :, output.temporal_index, :, :]
    if tuple(selected.shape) != output.shape:
        raise ValueError(
            f"internal shape mismatch for {output.name}: got {tuple(selected.shape)}, "
            f"expected {output.shape}"
        )

    dtype = output_dtype(output, use_f32)
    if dtype == np.dtype(np.float32):
        selected = selected.float()
    else:
        selected = selected.float().half()
    return selected.contiguous().numpy()


def add_metadata(
    writer: GGUFWriter,
    config: dict[str, Any],
    vision_config: dict[str, Any],
    preprocessor: dict[str, Any],
    image_size: int,
    projection_dim: int,
    use_f32: bool,
) -> None:
    writer.add_type("mmproj")
    writer.add_name(str(config.get("_name_or_path", "Qwen3.8-Flash-Next vision projector")))
    writer.add_description("Qwen3.8-Flash-Next Qwen3-VL vision projector")
    writer.add_file_type(0 if use_f32 else 1)
    writer.add_quantization_version(GGML_QUANT_VERSION)

    writer.add_bool("clip.has_text_encoder", False)
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_bool("clip.has_audio_encoder", False)
    writer.add_string("clip.projector_type", "qwen3vl_merger")
    writer.add_bool("clip.use_gelu", True)

    writer.add_uint32("clip.vision.image_size", image_size)
    writer.add_uint32(
        "clip.vision.patch_size",
        required_int(vision_config, "patch_size", "vision_config"),
    )
    writer.add_uint32(
        "clip.vision.embedding_length",
        required_int(vision_config, "hidden_size", "vision_config"),
    )
    writer.add_uint32(
        "clip.vision.feed_forward_length",
        required_int(vision_config, "intermediate_size", "vision_config"),
    )
    writer.add_uint32("clip.vision.projection_dim", projection_dim)
    writer.add_uint32(
        "clip.vision.attention.head_count",
        required_int(vision_config, "num_heads", "vision_config"),
    )
    writer.add_uint32(
        "clip.vision.block_count",
        required_int(vision_config, "depth", "vision_config"),
    )

    text_config = config["text_config"]
    layer_norm_epsilon = text_config.get("rms_norm_eps", 1e-6)
    if not isinstance(layer_norm_epsilon, (int, float)) or layer_norm_epsilon <= 0:
        raise ValueError("text_config.rms_norm_eps must be positive")
    writer.add_float32(
        "clip.vision.attention.layer_norm_epsilon", float(layer_norm_epsilon)
    )
    writer.add_uint32(
        "clip.vision.spatial_merge_size",
        required_int(vision_config, "spatial_merge_size", "vision_config"),
    )
    writer.add_uint32(
        "clip.vision.temporal_patch_size",
        required_int(vision_config, "temporal_patch_size", "vision_config"),
    )

    patch_size = required_int(vision_config, "patch_size", "vision_config")
    merge_size = required_int(vision_config, "spatial_merge_size", "vision_config")
    pixels_per_token = patch_size * patch_size * merge_size * merge_size
    processor_size = preprocessor.get("size", {})
    if not isinstance(processor_size, dict):
        raise ValueError("preprocessor_config.size must be an object")
    image_min_pixels = preprocessor.get(
        "min_pixels", processor_size.get("shortest_edge", 64 * pixels_per_token)
    )
    image_max_pixels = preprocessor.get(
        "max_pixels", processor_size.get("longest_edge", 16384 * pixels_per_token)
    )
    if (
        isinstance(image_min_pixels, bool)
        or not isinstance(image_min_pixels, int)
        or isinstance(image_max_pixels, bool)
        or not isinstance(image_max_pixels, int)
        or image_min_pixels <= 0
        or image_max_pixels < image_min_pixels
        or image_max_pixels > 0xFFFFFFFF
        or image_min_pixels % pixels_per_token != 0
        or image_max_pixels % pixels_per_token != 0
    ):
        raise ValueError(
            "Qwen3.8 image pixel limits must be positive ordered uint32 multiples "
            "of one merged vision token"
        )
    writer.add_uint32("clip.vision.image_min_pixels", image_min_pixels)
    writer.add_uint32("clip.vision.image_max_pixels", image_max_pixels)
    writer.add_array(
        "clip.vision.image_mean", preprocessor.get("image_mean", [0.5, 0.5, 0.5])
    )
    writer.add_array(
        "clip.vision.image_std", preprocessor.get("image_std", [0.5, 0.5, 0.5])
    )


def iter_plan_by_shard(plan: Iterable[TensorPlan]) -> Iterable[tuple[str, list[TensorPlan]]]:
    current_shard: str | None = None
    current_plan: list[TensorPlan] = []
    for item in plan:
        if current_shard is not None and item.shard_name != current_shard:
            yield current_shard, current_plan
            current_plan = []
        current_shard = item.shard_name
        current_plan.append(item)
    if current_shard is not None:
        yield current_shard, current_plan


def input_paths(model_dir: Path, weight_map: dict[str, str]) -> set[Path]:
    paths = {model_dir / "config.json"}
    for optional_name in ("preprocessor_config.json", "model.safetensors.index.json"):
        optional_path = model_dir / optional_name
        if optional_path.exists():
            paths.add(optional_path)
    paths.update(model_dir / shard_name for shard_name in weight_map.values())
    return {path.resolve() for path in paths}


def validate_output_path(output: Path, sources: set[Path]) -> Path:
    resolved_output = output.resolve()
    if resolved_output in sources:
        raise ValueError(f"output path collides with converter input {resolved_output}")
    if output.exists():
        for source in sources:
            if os.path.samefile(output, source):
                raise ValueError(f"output path aliases converter input {source}")
    return resolved_output


def convert(model_dir: Path, output: Path, use_f32: bool, big_endian: bool) -> None:
    config = read_json(model_dir / "config.json")
    vision_config, projection_dim, image_size = validate_config(config)
    preprocessor_path = model_dir / "preprocessor_config.json"
    preprocessor = read_json(preprocessor_path) if preprocessor_path.exists() else {}

    weight_map = load_weight_map(model_dir)
    specs = build_tensor_specs(vision_config, projection_dim)
    plan = build_plan(model_dir, weight_map, specs)

    output.parent.mkdir(parents=True, exist_ok=True)
    output = validate_output_path(output, input_paths(model_dir, weight_map))
    temporary_file = tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    writer: GGUFWriter | None = None
    try:
        writer = GGUFWriter(
            temporary_path,
            "clip",
            endianess=GGUFEndian.BIG if big_endian else GGUFEndian.LITTLE,
        )
        add_metadata(
            writer,
            config,
            vision_config,
            preprocessor,
            image_size,
            projection_dim,
            use_f32,
        )

        for item in plan:
            for destination in item.outputs:
                dtype = output_dtype(destination, use_f32)
                element_count = math.prod(destination.shape)
                writer.add_tensor_info(
                    destination.name,
                    destination.shape,
                    dtype,
                    element_count * dtype.itemsize,
                )

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for shard_name, shard_plan in iter_plan_by_shard(plan):
            with safe_open(model_dir / shard_name, framework="pt", device="cpu") as shard:
                for item in shard_plan:
                    source = shard.get_tensor(item.source_name)
                    for destination in item.outputs:
                        writer.write_tensor_data(
                            convert_output_tensor(source, destination, use_f32)
                        )
                    del source
        writer.close()
        writer = None
        os.replace(temporary_path, output)
    finally:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Qwen3.8-Flash-Next model.visual weights to qwen3vl mmproj GGUF"
    )
    parser.add_argument(
        "-m", "--model-dir", type=Path, required=True, help="Qwen3.8 HF model directory"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output GGUF path (default: MODEL_DIR/mmproj-qwen38-f16.gguf)",
    )
    parser.add_argument("--use-f32", action="store_true", help="write all tensors as f32")
    parser.add_argument("--bigendian", action="store_true", help="write a big-endian GGUF")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    output = args.output
    if output is None:
        suffix = "f32" if args.use_f32 else "f16"
        output = model_dir / f"mmproj-qwen38-{suffix}.gguf"
    convert(model_dir, output.resolve(), args.use_f32, args.bigendian)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
