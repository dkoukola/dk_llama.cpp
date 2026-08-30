from typing import Any, cast

import numpy as np

from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter, LazyChunkedTensor


def test_lazy_chunked_tensor_writes_chunks_in_row_order(tmp_path):
    loaded = []

    def chunk(first):
        def load():
            loaded.append(first)
            return np.arange(first, first + 8, dtype=np.float32).reshape(2, 4)
        return load

    tensor = LazyChunkedTensor([chunk(0), chunk(8)], (4, 4), np.float32)
    quantized = tensor.quantize(GGMLQuantizationType.F16)

    assert loaded == []
    output = tmp_path / "chunked.bin"
    with output.open("wb") as output_file:
        quantized.tofile(output_file)

    assert loaded == [0, 8]
    actual = np.fromfile(output, dtype=np.float16).reshape(4, 4)
    expected = np.arange(16, dtype=np.float16).reshape(4, 4)
    np.testing.assert_array_equal(actual, expected)
    assert output.stat().st_size == quantized.nbytes


def test_lazy_chunked_tensor_checks_written_size(tmp_path):
    tensor = LazyChunkedTensor(
        [lambda: np.zeros((1, 4), dtype=np.float32)],
        (2, 4),
        np.float32,
    )

    with (tmp_path / "short.bin").open("wb") as output_file:
        try:
            tensor.tofile(output_file)
        except ValueError as exc:
            assert "wrote 16 bytes, expected 32" in str(exc)
        else:
            raise AssertionError("short chunked tensor write unexpectedly succeeded")


def test_gguf_writer_keeps_chunked_tensor_lazy(tmp_path):
    loaded = []

    def chunk(first):
        def load():
            loaded.append(first)
            return np.arange(first, first + 8, dtype=np.float32).reshape(2, 4)
        return load

    output = tmp_path / "chunked.gguf"
    tensor = LazyChunkedTensor([chunk(0), chunk(8)], (4, 4), np.float32)
    tensor = tensor.quantize(GGMLQuantizationType.F16)
    writer = GGUFWriter(output, "llama")
    writer.add_tensor("chunked.weight", cast(Any, tensor), raw_dtype=GGMLQuantizationType.F16)
    assert loaded == []

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    assert loaded == [0, 8]
    result = GGUFReader(output).tensors[0]
    assert result.name == "chunked.weight"
    assert result.tensor_type == GGMLQuantizationType.F16
    assert result.shape.tolist() == [4, 4]
    np.testing.assert_array_equal(
        result.data.view(np.float16),
        np.arange(16, dtype=np.float16).reshape(4, 4),
    )
