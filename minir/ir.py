from typing import List, Optional, Dict, Any


class Int64:
    """Wrapper class to represent 64-bit integers in MLIR attributes."""

    def __init__(self, value: int) -> None:
        self.value: int = value

    def __repr__(self) -> str:
        return f"{self.value} : i64"


def get_attr(value: Any) -> str:
    if isinstance(value, str):
        if len(value) > 8:
            value = "..."
        return f'"{value}"'
    elif isinstance(value, Int64):
        return f"{value.value} : i64"
    elif isinstance(value, int):
        return f"{value} : i32"
    elif isinstance(value, float):
        return f"{value} : f32"
    elif isinstance(value, bytes):
        elide = False
        data = "__elided__" if elide else f'"0x{value.hex().upper()}"'
        data = f"dense<{data}>"
        return data
    else:
        return repr(value)


class Value:
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.owner: Optional[Operation] = None
        self.users: List[Operation] = []


class Scalar(Value):
    def __init__(self, name: str, dtype: str) -> None:
        super().__init__(name)
        self.dtype: str = dtype

    def __repr__(self) -> str:
        return self.dtype


class Vector(Scalar):
    def __init__(self, name: str, dtype: str, shape: List[int] = [1]) -> None:
        super().__init__(name, dtype)
        self.shape: List[int] = shape

    def __repr__(self) -> str:
        if len(self.shape) == 0:
            return self.dtype
        shape = "x".join(str(v) for v in self.shape)
        return f"vector<{shape}x{self.dtype}>"


class Tensor(Vector):
    def __init__(
        self,
        name: str,
        dtype: str,
        shape: List[int] = [1],
        encoding: Any = None,
    ) -> None:
        super().__init__(name, dtype, shape)
        self.encoding: Any = encoding

    def __repr__(self) -> str:
        if len(self.shape) == 0:
            return self.dtype
        shape = "x".join(str(v) for v in self.shape)
        encoding = f", {repr(self.encoding)}" if self.encoding is not None else ""
        return f"tensor<{shape}x{self.dtype}{encoding}>"


class Array:
    def __init__(self, data: List[int]) -> None:
        self.data: List[int] = data

    def __repr__(self) -> str:
        data_str = ", ".join(str(v) for v in self.data)
        return f"array<i64: {data_str}>"


class Dense:
    def __init__(self, data: bytes, dtype: str, shape: List[int]) -> None:
        self.data: bytes = data
        self.dtype: str = dtype
        self.shape: List[int] = shape

    def __repr__(self) -> str:
        elide = False
        data = "__elided__" if elide else f'"0x{self.data.hex().upper()}"'
        shape = "x".join(str(v) for v in self.shape)
        return f"dense<{data}> : tensor<{shape}x{self.dtype}>"


class Operation:
    def __init__(
        self,
        name: str,
        operands: Optional[List[Tensor]] = None,
        results: Optional[List[Tensor]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got {type(name)}")
        if operands is not None and not all(isinstance(v, Value) for v in operands):
            raise TypeError("all elements of operands must be Value", operands, name)
        if results is not None and not all(isinstance(v, Value) for v in results):
            raise TypeError("all elements of results must be Value")
        if attributes is not None and not isinstance(attributes, dict):
            raise TypeError(f"attributes must be dict, got {type(attributes)}")

        self.name: str = name
        self.operands: List[Value] = operands if operands is not None else []
        self.results: List[Value] = results if results is not None else []
        self.attributes: Dict[str, Any] = attributes if attributes is not None else {}

    def __repr__(self) -> str:
        opname = f'"{self.name}"'
        operands = ", ".join(v.name for v in self.operands)
        results = ", ".join(v.name for v in self.results)
        operands_info = ", ".join(str(v) for v in self.operands)
        results_info = ", ".join(str(v) for v in self.results)
        attrs = ", ".join(f"{k} = {get_attr(v)}" for k, v in self.attributes.items())
        attrs = f" {{{attrs}}}" if attrs else ""
        t = f"{results} = " if results else ""
        t += f"{opname}({operands}){attrs} : ({operands_info}) -> ({results_info})"
        return t


class Function:
    def __init__(
        self,
        operations: List[Operation],
        name: str = "function",
        arg_attrs: Optional[List[Dict[str, Any]]] = None,
        res_attrs: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.name: str = name
        self.operations: List[Operation] = operations
        self.set_owner_and_users()
        self.rename_values()
        self.arg_attrs: List[Dict[str, Any]] = [] if arg_attrs is None else arg_attrs
        self.res_attrs: List[Dict[str, Any]] = [] if res_attrs is None else res_attrs

    def __repr__(self) -> str:
        # Format arguments with their attributes (indexed)
        args_list = []
        for idx, v in enumerate(self.arguments):
            arg_str = f"{v.name} : {str(v)}"
            if idx < len(self.arg_attrs) and self.arg_attrs[idx]:
                attrs = ", ".join(
                    f"{k} = {get_attr(val)}" for k, val in self.arg_attrs[idx].items()
                )
                arg_str += f" {{{attrs}}}"
            args_list.append(arg_str)
        args = ", ".join(args_list)

        # Format results with their attributes (indexed)
        results_list = []
        for idx, v in enumerate(self.results):
            result_str = str(v)
            if idx < len(self.res_attrs) and self.res_attrs[idx]:
                attrs = ", ".join(
                    f"{k} = {get_attr(val)}" for k, val in self.res_attrs[idx].items()
                )
                result_str += f" {{{attrs}}}"
            results_list.append(result_str)
        results_info = ", ".join(results_list)

        t = f"func.func @{self.name}({args}) -> ({results_info}) {{\n"
        for operation in self.operations:
            t += f"  {str(operation)}\n"
        t += f"}}"
        return t

    def set_owner_and_users(self) -> "Function":
        for op in self.operations:
            for value in op.operands + op.results:
                value.owner = None
                value.users = []
        for operation in self.operations:
            for value in operation.operands:
                value.users.append(operation)
            for value in operation.results:
                value.owner = operation
        return self

    def rename_values(self) -> "Function":
        for idx, value in enumerate(self.values):
            value.name = f"%{idx}"
        return self

    @property
    def arguments(self) -> List[Value]:
        values: List[Value] = []
        for operation in self.operations:
            for value in operation.operands:
                if value.owner is None and value not in values:
                    values.append(value)
        return values

    @property
    def results(self) -> List[Value]:
        if self.operations and self.operations[-1].name == "func.return":
            return self.operations[-1].operands
        return []

    @property
    def local_values(self) -> List[Value]:
        results = self.results
        values: List[Value] = []
        for operation in self.operations:
            for value in operation.results:
                if value not in results:
                    values.append(value)
        return values

    @property
    def values(self) -> List[Value]:
        return self.arguments + self.results + self.local_values
