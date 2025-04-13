import numpy as np
from onnx import helper, TensorProto
from onnx.numpy_helper import from_array, to_array

from minir.onnx import (
    onnx_to_dtype,
    dtype_to_onnx,
    get_attributes,
    parse_values,
    parse_constants,
    make_value_info,
    make_initializer,
    make_node,
    to_onnx,
    from_onnx,
)
from minir.ir import Value, Operation, Function


def test_dtype_conversion():
    assert onnx_to_dtype(TensorProto.FLOAT) == "float32"
    assert dtype_to_onnx("float32") == TensorProto.FLOAT


def test_get_attributes():
    node = helper.make_node("Relu", ["x"], ["y"], alpha=1.0)
    attrs = get_attributes(node)
    assert "alpha" in attrs
    assert attrs["alpha"] == 1.0


def test_parse_values():
    vi = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 224, 224])
    values = parse_values([vi])
    assert len(values) == 1
    assert values[0].dtype == "float32"
    assert values[0].shape == [1, 3, 224, 224]


def test_parse_constants():
    arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
    tensor = from_array(arr, name="const")
    values = parse_constants([tensor])
    assert values[0].name == "const"
    np.testing.assert_array_equal(values[0].to_numpy(), arr)


def test_make_value_info_and_initializer():
    val = Value(name="x", dtype="float32", shape=[1, 3, 224, 224])
    vi = make_value_info(val)
    assert vi.name == "x"
    assert vi.type.tensor_type.elem_type == TensorProto.FLOAT

    val.data = np.ones([1, 3, 224, 224], dtype=np.float32)
    tensor_proto = make_initializer(val)
    np.testing.assert_array_equal(to_array(tensor_proto), val.to_numpy())


def test_make_node():
    input_val = Value(name="a", dtype="float32", shape=[1])
    output_val = Value(name="b", dtype="float32", shape=[1])
    op = Operation(name="Relu", operands=[input_val], results=[output_val])
    node = make_node(op)
    assert node.op_type == "Relu"
    assert node.input == ["a"]
    assert node.output == ["b"]


def test_full_roundtrip_conversion():
    x = Value(name="x", dtype="float32", shape=[1])
    y = Value(name="y", dtype="float32", shape=[1])
    op = Operation(name="Relu", operands=[x], results=[y])
    func = Function(operations=[op])
    model = to_onnx(func)
    new_func = from_onnx(model)
    assert len(new_func.operations) == 1
    assert new_func.operations[0].name == "Relu"
