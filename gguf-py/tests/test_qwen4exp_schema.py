from gguf import (
    GGUFValueType,
    GGUFWriter,
    Keys,
    MODEL_ARCH,
    MODEL_TENSOR,
    MODEL_TENSORS,
    TENSOR_NAMES,
)


def test_qwen4exp_fused_nextn_tensor_schema():
    expected_names = {
        MODEL_TENSOR.NEXTN_EH_PROJ: "blk.7.nextn.eh_proj",
        MODEL_TENSOR.NEXTN_HC_HEAD_NORM: "blk.7.nextn.hc_head_norm",
        MODEL_TENSOR.NEXTN_HC_HEAD_DOWN: "blk.7.nextn.hc_head_down",
        MODEL_TENSOR.NEXTN_HC_HEAD_UP: "blk.7.nextn.hc_head_up",
    }

    for tensor, expected_name in expected_names.items():
        assert tensor in MODEL_TENSORS[MODEL_ARCH.QWEN4EXP]
        assert TENSOR_NAMES[tensor].format(bid=7) == expected_name


def test_qwen4exp_shared_target_tensor_metadata(tmp_path):
    writer = GGUFWriter(tmp_path / "shared-nextn.gguf", "qwen4exp")
    writer.add_nextn_shared_target_tensors(True)

    key = Keys.LLM.NEXTN_SHARED_TARGET_TENSORS.format(arch="qwen4exp")
    value = writer.kv_data[0][key]
    assert key == "qwen4exp.nextn_shared_target_tensors"
    assert value.type == GGUFValueType.BOOL
    assert value.value is True
