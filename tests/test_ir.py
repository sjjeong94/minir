import numpy as np
from minir import Value, Operation


def test_value_basic_properties():
    val = Value(name="%0", dtype="float32", shape=[2, 3])

    assert val.name == "%0"
    assert val.dtype == "float32"
    assert val.shape == [2, 3]
    assert val.rank == 2
    assert val.numel == 6
    assert val.is_tensor()
    assert not val.is_scalar()
    assert not val.is_constant()
    assert repr(val) == "tensor<2x3xf32>"


def test_value_scalar():
    val = Value(name="%0", dtype="int64", shape=[])

    assert val.rank == 0
    assert val.numel == 1
    assert val.is_scalar()
    assert not val.is_tensor()
    assert repr(val) == "i64"


def test_value_from_numpy_and_to_numpy():
    np_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    val = Value.from_numpy(np_array, name="%0")

    assert val.dtype == "float32"
    assert val.shape == [2, 2]
    assert val.is_constant()
    np_recovered = val.to_numpy()
    np.testing.assert_array_equal(np_array, np_recovered)


def test_operation_repr():
    op = Operation(
        name="transpose",
        operands=[Value(name="%0", dtype="float32", shape=[4, 8, 12])],
        results=[Value(name="%1", dtype="float32", shape=[12, 4, 8])],
        attributes={"perms": [2, 0, 1]},
    )
    assert (
        repr(op)
        == '%1 = "transpose"(%0) {perms=[2, 0, 1]} : (tensor<4x8x12xf32>) -> tensor<12x4x8xf32>'
    )
