import numpy as np
from functools import reduce
from typing import List, Optional, Dict, Any


def get_dtype(dtype: str) -> str:
    return {
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
        "index": "index",
    }[dtype]


class Value:
    def __init__(
        self,
        name: str,
        dtype: str,
        shape: List[int],
        data: Optional[bytes] = None,
    ) -> None:
        self.name = name
        self.dtype = dtype
        self.shape = shape
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
        name: str,
        operands: List[Value],
        results: List[Value],
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.operands = operands
        self.results = results
        self.attributes = attributes if attributes is not None else {}

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
        t = f'{results} = "{self.name}"({operands}){attrs} : ({operands_info}) -> {results_info}'
        return t


class Function:
    def __init__(self, operations: List[Operation], name: Optional[str] = None) -> None:
        self.name = "function" if name is None else name
        self.operations = operations
        self.set_owner_and_users()
        self.rename_values()

    def __repr__(self, elide: bool = True) -> str:
        args = f", ".join([f"{v.name} : {str(v)}" for v in self.arguments])
        results = f", ".join([v.name for v in self.results])
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
        t += f"  return {results} : {results_info}\n"
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

    def rename_values(self) -> "Function":
        counter = 0
        for value in self.values:
            value.name = f"%{counter}"
            counter += 1
        return self

    @property
    def arguments(self) -> List[Value]:
        values = []
        for operation in self.operations:
            for value in operation.operands:
                if (
                    not value.is_constant()
                    and value.owner is None
                    and value not in values
                ):
                    values.append(value)
        return values

    @property
    def results(self) -> List[Value]:
        values = []
        for operation in self.operations:
            for value in operation.results:
                if len(value.users) == 0:
                    values.append(value)
        return values

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
        values = []
        for operation in self.operations:
            for value in operation.results:
                if len(value.users) > 0:
                    values.append(value)
        return values

    @property
    def values(self) -> List[Value]:
        return self.arguments + self.results + self.constants + self.local_values
