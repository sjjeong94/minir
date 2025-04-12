import numpy as np
from functools import reduce
from typing import List, Optional, Dict, Any


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
        dtype = {
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
        }[self.dtype]
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
        return np.frombuffer(self.data, dtype=self.dtype).reshape(self.shape)


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
            return str(value)

        operands = f", ".join([value.name for value in self.operands])
        results = f", ".join([value.name for value in self.results])
        operands_info = f", ".join([str(value) for value in self.operands])
        results_info = f", ".join([str(value) for value in self.results])
        attrs = f", ".join([f"{k}={get_attr(v)}" for k, v in self.attributes.items()])
        attrs = " {" + attrs + "}" if attrs else ""
        t = f'{results} = "{self.name}"({operands}){attrs} : ({operands_info}) -> {results_info}'
        return t
