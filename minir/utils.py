import uuid
from typing import Iterable
from functools import reduce


def product(iterable: Iterable[int]) -> int:
    return reduce(lambda x, y: x * y, iterable)


def generate_unique_name(prefix: str = "%") -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def dtype_to_numpy(dtype: str) -> str:
    mapping = {
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "ui8": "uint8",
        "ui16": "uint16",
        "ui32": "uint32",
        "ui64": "uint64",
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
    }
    return mapping[dtype]


def numpy_to_dtype(np_dtype: str) -> str:
    if not isinstance(np_dtype, str):
        np_dtype = str(np_dtype)
    mapping = {
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
    }
    return mapping[np_dtype]
