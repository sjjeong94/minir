import numpy as np
from typing import Any, Dict, List, Optional, Union
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

    def constant(self, data: Union[np.ndarray, int]) -> Value:
        if isinstance(data, np.ndarray):
            value = Value.from_numpy(data, name=generate_unique_name())
        elif isinstance(data, int):
            data = np.int64(data)
            value = Value.from_numpy(data, name=generate_unique_name())
            value.dtype = "index"
        else:
            raise ValueError("Unsupported data type for constant.")
        return value

    def empty(self, dtype: str, shape: Optional[List[int]] = None) -> Value:
        return Value(generate_unique_name(), dtype=dtype, shape=shape)

    def int8(self, shape: Optional[List[int]] = None) -> Value:
        return self.empty(dtype="int8", shape=shape)

    def int16(self, shape: Optional[List[int]] = None) -> Value:
        return self.empty(dtype="int16", shape=shape)

    def int32(self, shape: Optional[List[int]] = None) -> Value:
        return self.empty(dtype="int32", shape=shape)

    def int64(self, shape: Optional[List[int]] = None) -> Value:
        return self.empty(dtype="int64", shape=shape)

    def uint8(self, shape: Optional[List[int]] = None) -> Value:
        return self.empty(dtype="uint8", shape=shape)

    def uint16(self, shape: Optional[List[int]] = None) -> Value:
        return self.empty(dtype="uint16", shape=shape)

    def uint32(self, shape: Optional[List[int]] = None) -> Value:
        return self.empty(dtype="uint32", shape=shape)

    def uint64(self, shape: Optional[List[int]] = None) -> Value:
        return self.empty(dtype="uint64", shape=shape)

    def float16(self, shape: Optional[List[int]] = None) -> Value:
        return self.empty(dtype="float16", shape=shape)

    def float32(self, shape: Optional[List[int]] = None) -> Value:
        return self.empty(dtype="float32", shape=shape)

    def float64(self, shape: Optional[List[int]] = None) -> Value:
        return self.empty(dtype="float64", shape=shape)

    def func_return(self, values: List[Value]) -> None:
        if not values:
            raise ValueError("Return values cannot be empty.")
        self.write(
            name="func.return",
            operands=values,
            results=[],
        )
