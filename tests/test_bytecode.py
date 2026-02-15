"""
Tests for binary_writer module.
"""

import pytest
from minir import (
    Function,
    Operation,
    Tensor,
    Dense,
    Array,
    Int64,
    to_bytes,
    from_bytes,
)
from minir.bytecode import (
    encode_varint,
    decode_varint,
    BinaryWriter,
    BinaryReader,
    MAGIC_NUMBER,
)
from io import BytesIO


class TestVarInt:
    """Test VarInt encoding/decoding."""

    def test_encode_small_value(self):
        """Small values should use 1 byte."""
        assert encode_varint(0) == b"\x00"
        assert encode_varint(1) == b"\x01"
        assert encode_varint(127) == b"\x7f"

    def test_encode_medium_value(self):
        """Values 128-16383 should use 2 bytes."""
        result = encode_varint(128)
        assert len(result) == 2
        result = encode_varint(16383)
        assert len(result) == 2

    def test_encode_large_value(self):
        """Larger values should use more bytes."""
        result = encode_varint(16384)
        assert len(result) == 3

    def test_roundtrip(self):
        """Encoding and decoding should return the original value."""
        test_values = [0, 1, 127, 128, 255, 256, 16383, 16384, 65535, 1000000]
        for value in test_values:
            encoded = encode_varint(value)
            stream = BytesIO(encoded)
            decoded = decode_varint(stream)
            assert decoded == value, f"Failed for {value}"

    def test_negative_raises(self):
        """Negative values should raise an error."""
        with pytest.raises(ValueError):
            encode_varint(-1)


class TestBinaryRoundtrip:
    """Test serialization/deserialization roundtrip."""

    def test_simple_function(self):
        """Test roundtrip with a simple function."""
        x = Tensor("x", "f32", [2, 3])
        y = Tensor("y", "f32", [2, 3])

        op1 = Operation("onnx.Relu", operands=[x], results=[y])
        op2 = Operation("func.return", operands=[y], results=[])

        func = Function([op1, op2], name="test_func")

        # Serialize
        data = to_bytes(func)
        assert data.startswith(MAGIC_NUMBER)

        # Deserialize
        restored = from_bytes(data)
        assert restored.name == func.name
        assert len(restored.operations) == len(func.operations)

    def test_function_with_attributes(self):
        """Test roundtrip with operations containing attributes."""
        x = Tensor("x", "f32", [1, 3, 224, 224])
        y = Tensor("y", "f32", [1, 3, 112, 112])

        op = Operation(
            "onnx.MaxPool",
            operands=[x],
            results=[y],
            attributes={
                "kernel_shape": [2, 2],
                "strides": [2, 2],
            },
        )
        ret = Operation("func.return", operands=[y], results=[])

        func = Function([op, ret], name="pool_func")

        data = to_bytes(func)
        restored = from_bytes(data)

        assert restored.name == "pool_func"
        assert len(restored.operations) == 2
        assert restored.operations[0].name == "onnx.MaxPool"

    def test_function_with_constant(self):
        """Test roundtrip with dense constant data."""
        import numpy as np

        weight_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        weight = Tensor("w", "f32", [4])

        const_op = Operation(
            "arith.constant",
            operands=[],
            results=[weight],
            attributes={"value": Dense(weight_data.tobytes(), "f32", [4])},
        )

        x = Tensor("x", "f32", [4])
        y = Tensor("y", "f32", [4])

        add_op = Operation("onnx.Add", operands=[x, weight], results=[y])
        ret = Operation("func.return", operands=[y], results=[])

        func = Function([const_op, add_op, ret], name="add_const")

        data = to_bytes(func)
        restored = from_bytes(data)

        assert restored.name == "add_const"
        # Check const op has Dense attribute
        const_restored = restored.operations[0]
        assert "value" in const_restored.attributes
        assert isinstance(const_restored.attributes["value"], Dense)

    def test_function_with_array_attr(self):
        """Test roundtrip with Array attribute."""
        x = Tensor("x", "f32", [2, 3, 4])
        y = Tensor("y", "f32", [4, 3, 2])

        op = Operation(
            "tosa.reshape",
            operands=[x],
            results=[y],
            attributes={"new_shape": Array([4, 3, 2])},
        )
        ret = Operation("func.return", operands=[y], results=[])

        func = Function([op, ret], name="reshape")

        data = to_bytes(func)
        restored = from_bytes(data)

        assert restored.operations[0].name == "tosa.reshape"
        new_shape = restored.operations[0].attributes.get("new_shape")
        assert isinstance(new_shape, Array)
        assert new_shape.data == [4, 3, 2]

    def test_multiple_dtypes(self):
        """Test with different data types."""
        dtypes = ["i32", "i64", "f16", "f32", "f64"]

        for dtype in dtypes:
            x = Tensor("x", dtype, [2, 2])
            y = Tensor("y", dtype, [2, 2])

            op = Operation("test.identity", operands=[x], results=[y])
            ret = Operation("func.return", operands=[y], results=[])

            func = Function([op, ret], name=f"test_{dtype}")

            data = to_bytes(func)
            restored = from_bytes(data)

            assert restored.name == f"test_{dtype}"
            # Check dtype is preserved
            result = restored.operations[0].results[0]
            assert result.dtype == dtype


class TestBinaryWriter:
    """Test BinaryWriter internals."""

    def test_string_deduplication(self):
        """Strings should be deduplicated."""
        writer = BinaryWriter()

        idx1 = writer._add_string("test")
        idx2 = writer._add_string("test")
        idx3 = writer._add_string("other")

        assert idx1 == idx2  # Same string, same index
        assert idx3 != idx1  # Different string, different index
        assert len(writer.strings) == 2

    def test_type_deduplication(self):
        """Types with same signature should be deduplicated."""
        writer = BinaryWriter()

        t1 = Tensor("a", "f32", [2, 3])
        t2 = Tensor("b", "f32", [2, 3])
        t3 = Tensor("c", "f32", [4, 5])

        idx1 = writer._add_type(t1)
        idx2 = writer._add_type(t2)
        idx3 = writer._add_type(t3)

        assert idx1 == idx2  # Same type, same index
        assert idx3 != idx1  # Different shape, different index


class TestBinaryFileIO:
    """Test file I/O operations."""

    def test_save_and_load(self, tmp_path):
        """Test saving and loading from file."""
        x = Tensor("x", "f32", [10])
        y = Tensor("y", "f32", [10])

        op = Operation("onnx.Sigmoid", operands=[x], results=[y])
        ret = Operation("func.return", operands=[y], results=[])

        func = Function([op, ret], name="sigmoid_func")

        # Save
        path = tmp_path / "test.mir"
        to_bytes(func, str(path))

        assert path.exists()

        # Load
        loaded = from_bytes(str(path))

        assert loaded.name == "sigmoid_func"
        assert len(loaded.operations) == 2
        assert loaded.operations[0].name == "onnx.Sigmoid"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_function(self):
        """Test function with only return."""
        ret = Operation("func.return", operands=[], results=[])
        func = Function([ret], name="empty")

        data = to_bytes(func)
        restored = from_bytes(data)

        assert restored.name == "empty"
        assert len(restored.operations) == 1

    def test_invalid_magic_number(self):
        """Invalid magic number should raise an error."""
        data = b"INVALID\x00" + b"\x00" * 100

        with pytest.raises(ValueError, match="Invalid magic number"):
            from_bytes(data)

    def test_int64_attribute(self):
        """Test Int64 attribute type."""
        x = Tensor("x", "f32", [10])
        y = Tensor("y", "f32", [10])

        op = Operation(
            "test.op", operands=[x], results=[y], attributes={"axis": Int64(42)}
        )
        ret = Operation("func.return", operands=[y], results=[])

        func = Function([op, ret], name="int64_test")

        data = to_bytes(func)
        restored = from_bytes(data)

        axis = restored.operations[0].attributes.get("axis")
        assert isinstance(axis, Int64)
        assert axis.value == 42
