import numpy as np
from typing import Any, Dict, List, Optional
from minir.ir import Operation, Value, Function
from minir.utils import generate_unique_name


class Writer:
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.operations = []

    def __repr__(self):
        return repr(self.to_function())

    def to_function(self, name: Optional[str] = None) -> Function:
        return Function(operations=self.operations, name=name)

    def write(
        self,
        name: str,
        operands: List[Value],
        results: List[Value],
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        name = name if self.name is None else f"{self.name}.{name}"
        self.operations.append(
            Operation(
                name=name,
                operands=operands,
                results=results,
                attributes=attributes,
            )
        )

    def constant(self, data: np.ndarray) -> Value:
        return Value.from_numpy(data, name=generate_unique_name())

    def empty(self, dtype: str, shape: List[int]) -> Value:
        return Value(generate_unique_name(), dtype=dtype, shape=shape)

    def int8(self, shape: List[int]) -> Value:
        return self.empty(dtype="int8", shape=shape)

    def int16(self, shape: List[int]) -> Value:
        return self.empty(dtype="int16", shape=shape)

    def int32(self, shape: List[int]) -> Value:
        return self.empty(dtype="int32", shape=shape)

    def int64(self, shape: List[int]) -> Value:
        return self.empty(dtype="int64", shape=shape)

    def uint8(self, shape: List[int]) -> Value:
        return self.empty(dtype="uint8", shape=shape)

    def uint16(self, shape: List[int]) -> Value:
        return self.empty(dtype="uint16", shape=shape)

    def uint32(self, shape: List[int]) -> Value:
        return self.empty(dtype="uint32", shape=shape)

    def uint64(self, shape: List[int]) -> Value:
        return self.empty(dtype="uint64", shape=shape)

    def float16(self, shape: List[int]) -> Value:
        return self.empty(dtype="float16", shape=shape)

    def float32(self, shape: List[int]) -> Value:
        return self.empty(dtype="float32", shape=shape)

    def float64(self, shape: List[int]) -> Value:
        return self.empty(dtype="float64", shape=shape)
