import onnx
import numpy as np
from typing import List, Dict, Any, Optional
from functools import reduce


def get_dtype(dtype: str) -> str:
    return {
        "bool": "i1",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "uint8": "ui8",
        "uint16": "ui16",
        "uint32": "ui32",
        "uint64": "ui64",
        "float16": "f16",
        "float32": "f32",
        "float64": "f64",
    }[dtype]


class Value:
    def __init__(
        self,
        name: str,
        dtype: str,
        shape: Optional[List[int]] = None,
        data: Optional[bytes] = None,
    ) -> None:
        self.name = name
        self.dtype = dtype
        self.shape = shape if shape is not None else []
        self.data = data

        self.owner: Operation = None
        self.users: List[Operation] = []

    def __repr__(self):
        shape = "x".join([str(value) for value in self.shape])
        dtype = get_dtype(self.dtype)
        return f"tensor<{shape}x{dtype}>" if not self.is_scalar() else dtype

    def is_scalar(self) -> bool:
        return len(self.shape) == 0

    def is_tensor(self) -> bool:
        return len(self.shape) > 0

    def is_constant(self) -> bool:
        return self.data is not None

    @property
    def rank(self):
        return len(self.shape)

    @property
    def numel(self):
        if self.is_scalar():
            return 1
        return reduce(lambda x, y: x * y, self.shape)

    @classmethod
    def from_numpy(cls, data: np.ndarray, name: str = "") -> "Value":
        return cls(
            name=name,
            dtype=str(data.dtype),
            shape=list(data.shape),
            data=data.tobytes(),
        )

    def to_numpy(self) -> np.ndarray:
        dtype = "int64" if self.dtype == "index" else self.dtype
        return np.frombuffer(self.data, dtype=dtype).reshape(self.shape)


class Operation:
    def __init__(
        self,
        op_type: str,
        operands: List[Value],
        results: List[Value],
        attributes: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> None:
        self.op_type = op_type
        self.operands = operands
        self.results = results
        self.attributes = attributes if attributes is not None else {}
        self.name = name

    def __repr__(self):
        def get_attr(value: Any) -> str:
            if isinstance(value, str):
                return f'"{value}"'
            elif isinstance(value, int):
                return f"{value} : i32"
            elif isinstance(value, float):
                return f"{value} : f32"
            elif isinstance(value, np.ndarray):
                dtype = get_dtype(str(value.dtype))
                return f"array<{dtype}: {str(value.tolist())[1:-1]}>"
            elif isinstance(value, np.dtype):
                return get_dtype(str(value))
            return str(value)

        operands = f", ".join([value.name for value in self.operands])
        results = f", ".join([value.name for value in self.results])
        operands_info = f", ".join([str(value) for value in self.operands])
        results_info = f", ".join([str(value) for value in self.results])
        attrs = f", ".join([f"{k} = {get_attr(v)}" for k, v in self.attributes.items()])
        attrs = " {" + attrs + "}" if attrs else ""
        t = f"{results} = " if results else ""
        t += f'"{self.op_type}"({operands}){attrs} : ({operands_info}) -> ({results_info})'
        return t


class Function:
    def __init__(self, operations: List[Operation], name: Optional[str] = None) -> None:
        self.name = "graph" if name is None else name
        self.operations = operations
        self.set_owner_and_users()

    def __repr__(self, elide: bool = True) -> str:
        args = f", ".join([f"{v.name} : {str(v)}" for v in self.arguments])
        results_info = f", ".join([f"{str(v)}" for v in self.results])
        t = f"func.func @{self.name}({args}) -> ({results_info}) {{\n"
        for constant in self.constants:
            if constant.is_scalar():
                data = f"{constant.to_numpy()}"
            else:
                data = "..." if elide else f'"0x{constant.data.hex().upper()}"'
                data = f"dense<{data}>"
            t += f"  {constant.name} = arith.constant {data} : {constant}\n"
        for operation in self.operations:
            t += f"  {str(operation)}\n"
        t += f"}}"
        return t

    def set_owner_and_users(self) -> "Function":
        for value in self.values:
            value.owner = None
            value.users = []
        for operation in self.operations:
            for value in operation.operands:
                value.users.append(operation)
            for value in operation.results:
                value.owner = operation
        return self

    @property
    def arguments(self) -> List[Value]:
        values = []
        for operation in self.operations:
            for value in operation.operands:
                if value.owner is None and value not in values:
                    values.append(value)
        return values

    @property
    def results(self) -> List[Value]:
        if self.operations[-1].op_type == "func.return":
            return self.operations[-1].operands
        return []

    @property
    def constants(self) -> List[Value]:
        values = []
        for operation in self.operations:
            for value in operation.operands:
                if value.is_constant() and value not in values:
                    values.append(value)
        return values

    @property
    def local_values(self) -> List[Value]:
        results = self.results
        values = []
        for operation in self.operations:
            for value in operation.results:
                if value not in results:
                    values.append(value)
        return values

    @property
    def values(self) -> List[Value]:
        return self.arguments + self.results + self.local_values


def onnx_to_dtype(onnx_dtype: int) -> str:
    mapping = {
        onnx.TensorProto.BOOL: "bool",
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
        "bool": onnx.TensorProto.BOOL,
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
        op_type=operation.op_type,
        inputs=inputs,
        outputs=outputs,
        name=operation.name,
        **attrs,
    )


def make_graph(func: Function) -> onnx.GraphProto:
    graph = onnx.helper.make_graph(
        nodes=[make_node(operation) for operation in func.operations[:-1]],
        name=func.name,
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
                op_type=node.op_type,
                operands=[values_map[name] for name in node.input],
                results=[values_map[name] for name in node.output],
                attributes=get_attributes(node),
                name=node.name,
            )
        )
    nodes.append(
        Operation(
            op_type="func.return",
            operands=[values_map[output.name] for output in graph.output],
            results=[],
        )
    )
    return Function(nodes)
