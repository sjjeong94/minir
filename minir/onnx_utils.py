import onnx
from typing import List, Dict, Any, Union, Optional
from minir.ir import Operation, Function, Tensor
from minir.utils import generate_unique_name


def onnx_to_dtype(onnx_dtype: int) -> str:
    mapping = {
        onnx.TensorProto.INT8: "i8",
        onnx.TensorProto.INT16: "i16",
        onnx.TensorProto.INT32: "i32",
        onnx.TensorProto.INT64: "i64",
        onnx.TensorProto.UINT8: "ui8",
        onnx.TensorProto.UINT16: "ui16",
        onnx.TensorProto.UINT32: "ui32",
        onnx.TensorProto.UINT64: "ui64",
        onnx.TensorProto.FLOAT16: "f16",
        onnx.TensorProto.FLOAT: "f32",
        onnx.TensorProto.DOUBLE: "f64",
    }
    return mapping[onnx_dtype]


def dtype_to_onnx(dtype: str) -> int:
    mapping = {
        "i8": onnx.TensorProto.INT8,
        "i16": onnx.TensorProto.INT16,
        "i32": onnx.TensorProto.INT32,
        "i64": onnx.TensorProto.INT64,
        "ui8": onnx.TensorProto.UINT8,
        "ui16": onnx.TensorProto.UINT16,
        "ui32": onnx.TensorProto.UINT32,
        "ui64": onnx.TensorProto.UINT64,
        "f16": onnx.TensorProto.FLOAT16,
        "f32": onnx.TensorProto.FLOAT,
        "f64": onnx.TensorProto.DOUBLE,
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
    attributes["onnx_name"] = node.name
    return attributes


def parse_values(onnx_tensors: List[onnx.ValueInfoProto]) -> List[Tensor]:
    values = []
    for tensor in onnx_tensors:
        type_proto = tensor.type
        dtype = onnx_to_dtype(type_proto.tensor_type.elem_type)
        shape = [
            dim.dim_value if dim.HasField("dim_value") else "dynamic"
            for dim in type_proto.tensor_type.shape.dim
        ]
        values.append(
            Tensor(
                name=tensor.name,
                dtype=dtype,
                shape=shape,
            )
        )
    return values


def parse_constants(onnx_tensors: List[onnx.TensorProto]) -> List[Tensor]:
    constants = []
    for tensor in onnx_tensors:
        constants.append(
            Tensor(
                name=tensor.name,
                dtype=onnx_to_dtype(tensor.data_type),
                shape=list(tensor.dims),
            )
        )
    return constants


def make_value_info(tensor: Tensor) -> onnx.ValueInfoProto:
    elem_type = dtype_to_onnx(tensor.dtype)
    return onnx.helper.make_tensor_value_info(
        name=tensor.name, elem_type=elem_type, shape=tensor.shape
    )


def make_initializer(operation: Operation) -> onnx.TensorProto:
    tensor = operation.results[0]
    tensor_proto = onnx.TensorProto()
    tensor_proto.name = tensor.name
    tensor_proto.data_type = dtype_to_onnx(tensor.dtype)
    tensor_proto.dims.extend(tensor.shape)
    tensor_proto.raw_data = operation.attributes.get("value")
    return tensor_proto


def make_node(operation: Operation) -> onnx.NodeProto:
    inputs = [tensor.name for tensor in operation.operands]
    outputs = [tensor.name for tensor in operation.results]
    attrs = operation.attributes
    name = attrs.pop("onnx_name", None)
    if name is None:
        name = generate_unique_name(prefix=operation.name + "_")
    return onnx.helper.make_node(
        op_type=operation.name.split(".")[-1],
        inputs=inputs,
        outputs=outputs,
        name=name,
        **attrs,
    )


def make_graph(func: Function) -> onnx.GraphProto:
    constants = []
    operations = []
    for op in func.operations:
        if op.name == "func.return":
            continue
        elif op.name == "arith.constant":
            constants.append(op)
        else:
            operations.append(op)

    if func.arg_attrs:
        for idx, value in enumerate(func.arguments):
            for k, v in func.arg_attrs[idx].items():
                if k == "onnx_name":
                    value.name = v
    if func.res_attrs:
        for idx, value in enumerate(func.results):
            for k, v in func.res_attrs[idx].items():
                if k == "onnx_name":
                    value.name = v

    graph = onnx.helper.make_graph(
        nodes=[make_node(operation) for operation in operations],
        name=func.name,
        inputs=[make_value_info(value) for value in func.arguments],
        outputs=[make_value_info(value) for value in func.results],
        value_info=[make_value_info(value) for value in func.local_values],
        initializer=[make_initializer(op) for op in constants],
    )
    return graph


def constants_to_initializers(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Convert ONNX Constant nodes to initializers.

    Args:
        model: Input ONNX model

    Returns:
        Modified ONNX model with Constant nodes converted to initializers
    """
    graph = model.graph

    # Find all Constant nodes
    constant_nodes = []
    other_nodes = []

    for node in graph.node:
        if node.op_type == "Constant":
            constant_nodes.append(node)
        else:
            other_nodes.append(node)

    # Convert Constant nodes to initializers
    new_initializers = list(graph.initializer)

    for node in constant_nodes:
        # Get the constant value from attributes
        value_attr = None
        for attr in node.attribute:
            if attr.name == "value":
                value_attr = attr.t
                break

        if value_attr is None:
            continue

        # Create initializer from the constant
        initializer = onnx.TensorProto()
        initializer.CopyFrom(value_attr)
        initializer.name = node.output[0]

        new_initializers.append(initializer)

    # Create new graph with updated nodes and initializers
    new_graph = onnx.helper.make_graph(
        nodes=other_nodes,
        name=graph.name,
        inputs=graph.input,
        outputs=graph.output,
        initializer=new_initializers,
        value_info=graph.value_info,
    )

    # Create new model
    new_model = onnx.helper.make_model(
        new_graph,
        producer_name=model.producer_name,
        ir_version=model.ir_version,
        opset_imports=model.opset_import,
    )

    return new_model


def to_onnx(func: Function, path: Optional[str] = None) -> onnx.ModelProto:
    graph = make_graph(func)
    model = onnx.helper.make_model(
        graph,
        ir_version=11,
        opset_imports=[onnx.helper.make_opsetid("", 23)],
    )
    if path is not None:
        onnx.save(model, path)
    return model


def from_onnx(model: Union[str, onnx.ModelProto]) -> Function:
    if isinstance(model, str):
        model = onnx.load(model)
    model = constants_to_initializers(model)
    model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph

    arguments = parse_values(graph.input)
    results = parse_values(graph.output)
    local_values = parse_values(graph.value_info)
    constants = parse_constants(graph.initializer)

    values = arguments + results + local_values + constants
    values_map = {value.name: value for value in values}

    nodes: List[Operation] = []
    for init in graph.initializer:
        nodes.append(
            Operation(
                name="arith.constant",
                operands=[],
                results=[values_map[init.name]],
                attributes={"value": init.raw_data},
            )
        )
    for node in graph.node:
        nodes.append(
            Operation(
                name=f"onnx.{node.op_type}",
                operands=[values_map[name] for name in node.input],
                results=[values_map[name] for name in node.output],
                attributes=get_attributes(node),
            )
        )
    nodes.append(
        Operation(
            name="func.return",
            operands=[values_map[output.name] for output in graph.output],
            results=[],
        )
    )

    return Function(
        operations=nodes,
        name=graph.name,
        arg_attrs=[{"onnx_name": value.name} for value in graph.input],
        res_attrs=[{"onnx_name": value.name} for value in graph.output],
    )
