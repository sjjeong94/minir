import numpy as np
from minir import Value, Operation, Function


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
        name="arith.mulf",
        operands=[
            Value(name="%0", dtype="float32", shape=[8]),
            Value(name="%1", dtype="float32", shape=[8]),
        ],
        results=[Value(name="%2", dtype="float32", shape=[8])],
    )
    assert (
        repr(op)
        == '%2 = "arith.mulf"(%0, %1) : (tensor<8xf32>, tensor<8xf32>) -> (tensor<8xf32>)'
    )


def test_function_multiple_operations():
    x = Value(name="x", dtype="float32", shape=[8])
    y = Value(name="y", dtype="float32", shape=[8])
    z = Value(name="z", dtype="float32", shape=[8])
    temp1 = Value(name="temp1", dtype="float32", shape=[8])
    temp2 = Value(name="temp2", dtype="float32", shape=[8])
    op1 = Operation(name="arith.mulf", operands=[x, y], results=[temp1])
    op2 = Operation(name="arith.addf", operands=[temp1, z], results=[temp2])
    func = Function(operations=[op1, op2])

    print(repr(func))

    assert x in func.arguments
    assert y in func.arguments
    assert z in func.arguments
    assert temp1 in func.local_values
    assert temp2 in func.results
    assert all(v.owner is not None or v in func.arguments for v in func.values)
