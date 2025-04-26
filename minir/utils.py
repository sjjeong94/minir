import uuid
from typing import Iterable
from functools import reduce


def product(iterable: Iterable[int]) -> int:
    return reduce(lambda x, y: x * y, iterable)


def generate_unique_name(prefix: str = "%") -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}"
