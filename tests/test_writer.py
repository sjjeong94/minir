import pytest
import numpy as np
from minir.ir import Value, Operation
from minir.writer import Writer


def test_writer_empty_creates_value():
    writer = Writer()
    value = writer.empty("float32", [2, 3])
    assert isinstance(value, Value)
    assert value.dtype == "float32"
    assert value.shape == [2, 3]
    assert value.data is None


@pytest.mark.parametrize(
    "method,dtype",
    [
        ("int8", "int8"),
        ("int16", "int16"),
        ("int32", "int32"),
        ("int64", "int64"),
        ("uint8", "uint8"),
        ("uint16", "uint16"),
        ("uint32", "uint32"),
        ("uint64", "uint64"),
        ("float16", "float16"),
        ("float32", "float32"),
        ("float64", "float64"),
    ],
)
def test_dtype_shortcuts(method, dtype):
    writer = Writer()
    value = getattr(writer, method)([4, 4])
    assert isinstance(value, Value)
    assert value.dtype == dtype
    assert value.shape == [4, 4]
    assert value.data is None


def test_writer_constant():
    writer = Writer()
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    const_val = writer.constant(data)

    assert isinstance(const_val, Value)
    assert const_val.is_constant()
    assert const_val.to_numpy().tolist() == data.tolist()


def test_writer_write_add_operation():
    writer = Writer("arith")
    a = writer.float32([2, 2])
    b = writer.float32([2, 2])
    c = writer.float32([2, 2])
    writer.write("addf", operands=[a, b], results=[c])

    print(writer)

    assert len(writer.operations) == 2
    op = writer.operations[0]
    assert isinstance(op, Operation)
    assert op.name == "arith.addf"
    assert op.operands == [a, b]
    assert op.results == [c]
