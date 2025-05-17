import onnx
import onnxsim
from typing import List, Dict, Any
from minir.ir import Value, Operation, Function
from minir.utils import generate_unique_name


def onnx_to_dtype(onnx_dtype: int) -> str:
    mapping = {
        onnx.TensorProto.INT8: "int8",
        onnx.TensorProto.INT16: "int16",
        onnx.TensorProto.INT32: "int32",
        onnx.TensorProto.INT64: "int64",
        onnx.TensorProto.UINT8: "uint8",
        onnx.TensorProto.UINT16: "uint16",
        onnx.TensorProto.UINT32: "uint32",
        onnx.TensorProto.UINT64: "uint64",
        onnx.TensorProto.FLOAT16: "float16",
        onnx.TensorProto.FLOAT: "float32",
        onnx.TensorProto.DOUBLE: "float64",
    }
    return mapping[onnx_dtype]


def dtype_to_onnx(dtype: str) -> int:
    mapping = {
        "int8": onnx.TensorProto.INT8,
        "int16": onnx.TensorProto.INT16,
        "int32": onnx.TensorProto.INT32,
        "int64": onnx.TensorProto.INT64,
        "uint8": onnx.TensorProto.UINT8,
        "uint16": onnx.TensorProto.UINT16,
        "uint32": onnx.TensorProto.UINT32,
        "uint64": onnx.TensorProto.UINT64,
        "float16": onnx.TensorProto.FLOAT16,
        "float32": onnx.TensorProto.FLOAT,
        "float64": onnx.TensorProto.DOUBLE,
    }
    return mapping[dtype]


def get_attributes(node: onnx.NodeProto) -> Dict[str, Any]:
    attributes = dict()
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.FLOAT:
            value = attr.f
        elif attr.type == onnx.AttributeProto.INT:
            value = attr.i
        elif attr.type == onnx.AttributeProto.STRING:
            value = attr.s.decode("utf-8")
        elif attr.type == onnx.AttributeProto.FLOATS:
            value = list(attr.floats)
        elif attr.type == onnx.AttributeProto.INTS:
            value = list(attr.ints)
        elif attr.type == onnx.AttributeProto.STRINGS:
            value = [s.decode("utf-8") for s in attr.strings]
        elif attr.type == onnx.AttributeProto.TENSOR:
            value = str(attr.t)
        elif attr.type == onnx.AttributeProto.GRAPH:
            value = str(attr.g)
        else:
            raise ValueError(
                f"Unsupported attribute type: {onnx.AttributeProto.AttributeType.Name(attr.type)}"
            )
        attributes[attr.name] = value
    return attributes


def parse_values(onnx_tensors: List[onnx.ValueInfoProto]) -> List[Value]:
    values = []
    for tensor in onnx_tensors:
        type_proto = tensor.type
        dtype = onnx_to_dtype(type_proto.tensor_type.elem_type)
        shape = [
            dim.dim_value if dim.HasField("dim_value") else "dynamic"
            for dim in type_proto.tensor_type.shape.dim
        ]
        values.append(
            Value(
                name=tensor.name,
                dtype=dtype,
                shape=shape,
            )
        )
    return values


def parse_constants(onnx_tensors: List[onnx.TensorProto]) -> List[Value]:
    constants = []
    for tensor in onnx_tensors:
        data = onnx.numpy_helper.to_array(tensor)
        constants.append(
            Value.from_numpy(
                name=tensor.name,
                data=data,
            )
        )
    return constants


def make_value_info(tensor: Value) -> onnx.ValueInfoProto:
    elem_type = dtype_to_onnx(tensor.dtype)
    return onnx.helper.make_tensor_value_info(
        name=tensor.name, elem_type=elem_type, shape=tensor.shape
    )


def make_initializer(tensor: Value) -> onnx.TensorProto:
    return onnx.numpy_helper.from_array(
        tensor.to_numpy(),
        name=tensor.name,
    )


def make_node(operation: Operation) -> onnx.NodeProto:
    inputs = [tensor.name for tensor in operation.operands]
    outputs = [tensor.name for tensor in operation.results]
    attrs = operation.attributes
    return onnx.helper.make_node(
        op_type=operation.name,
        inputs=inputs,
        outputs=outputs,
        name=generate_unique_name(prefix=operation.name + "_"),
        **attrs,
    )


def make_graph(func: Function) -> onnx.GraphProto:
    graph = onnx.helper.make_graph(
        nodes=[make_node(operation) for operation in func.operations],
        name="graph",
        inputs=[make_value_info(value) for value in func.arguments],
        outputs=[make_value_info(value) for value in func.results],
        initializer=[make_initializer(value) for value in func.constants],
        value_info=[make_value_info(value) for value in func.local_values],
    )
    return graph


def to_onnx(func: Function) -> onnx.ModelProto:
    graph = make_graph(func)
    model = onnx.helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[onnx.helper.make_opsetid("", 20)],
    )
    return model


def from_onnx(model: onnx.ModelProto) -> Function:
    model = onnxsim.simplify(model)[0]
    graph = model.graph

    arguments = parse_values(graph.input)
    results = parse_values(graph.output)
    constants = parse_constants(graph.initializer)
    local_values = parse_values(graph.value_info)

    values = arguments + results + constants + local_values
    values_map = {value.name: value for value in values}

    nodes: List[Operation] = []
    for node in graph.node:
        nodes.append(
            Operation(
                name=node.op_type,
                operands=[values_map[name] for name in node.input],
                results=[values_map[name] for name in node.output],
                attributes=get_attributes(node),
            )
        )

    return Function(nodes)
