import numpy as np
from typing import Any, Dict, List, Optional, Union
from minir.ir import Function, Operation, Tensor, Array, Dense
from minir.utils import numpy_to_dtype, product, dtype_to_numpy


class TOSAWriter:
    def __init__(self) -> None:
        self.operations: List[Operation] = []

    def __repr__(self) -> str:
        return repr(self.to_function())

    def to_function(self, name: str = "function") -> Function:
        if not self.operations or self.operations[-1].name != "func.return":
            self.ret()
        operations = [op for op in self.operations if op.name == "tosa.const"]
        for op in self.operations:
            if op.name != "tosa.const":
                operations.append(op)
        return Function(operations, name=name)

    def tensor(
        self,
        dtype: str,
        shape: List[int] = [1],
    ) -> Tensor:
        return Tensor(
            name="%0",
            dtype=dtype,
            shape=shape,
        )

    def write(
        self,
        name: str,
        operands: Optional[List[Tensor]] = None,
        results: Optional[List[Tensor]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.operations.append(
            Operation(
                name=name,
                operands=operands,
                results=results,
                attributes=attributes,
            )
        )

    def ret(self, values: List[Tensor] = None) -> None:
        if values is None:
            values = []
        self.write(
            name="func.return",
            operands=values,
            results=[],
        )

    def constant(self, array: np.ndarray) -> Tensor:
        tensor = self.tensor(
            dtype=numpy_to_dtype(array.dtype),
            shape=list(array.shape),
        )
        self.write(
            name="tosa.const",
            operands=[],
            results=[tensor],
            attributes={
                "value": Dense(
                    array.tobytes(),
                    dtype=numpy_to_dtype(array.dtype),
                    shape=list(array.shape),
                )
            },
        )
        return tensor

    def unary_op(self, name: str, x: Tensor, **kwargs) -> Tensor:
        y = self.tensor(dtype=x.dtype, shape=x.shape)
        self.write(name=name, operands=[x], results=[y], attributes=kwargs)
        return y

    # Unary operations
    def abs(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="tosa.abs", x=x, **kwargs)

    def negate(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="tosa.negate", x=x, **kwargs)

    def exp(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="tosa.exp", x=x, **kwargs)

    def log(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="tosa.log", x=x, **kwargs)

    def rsqrt(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="tosa.rsqrt", x=x, **kwargs)

    def reciprocal(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="tosa.reciprocal", x=x, **kwargs)

    def floor(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="tosa.floor", x=x, **kwargs)

    def ceil(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="tosa.ceil", x=x, **kwargs)

    def tanh(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="tosa.tanh", x=x, **kwargs)

    def sigmoid(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="tosa.sigmoid", x=x, **kwargs)

    # Binary operations
    def binary_op(self, name: str, a: Tensor, b: Tensor, **kwargs) -> Tensor:
        ref = a if product(a.shape) >= product(b.shape) else b
        c = self.tensor(dtype=ref.dtype, shape=ref.shape)
        self.write(name=name, operands=[a, b], results=[c], attributes=kwargs)
        return c

    def add(self, a: Tensor, b: Tensor, **kwargs) -> Tensor:
        return self.binary_op(name="tosa.add", a=a, b=b, **kwargs)

    def sub(self, a: Tensor, b: Tensor, **kwargs) -> Tensor:
        return self.binary_op(name="tosa.sub", a=a, b=b, **kwargs)

    def mul(self, a: Tensor, b: Tensor, shift: int = 0, **kwargs) -> Tensor:
        return self.binary_op(name="tosa.mul", a=a, b=b, shift=shift, **kwargs)

    def pow(self, a: Tensor, b: Tensor, **kwargs) -> Tensor:
        return self.binary_op(name="tosa.pow", a=a, b=b, **kwargs)

    def matmul(self, a: Tensor, b: Tensor, **kwargs) -> Tensor:
        # TOSA matmul requires 3D tensors [Batch, M, K] @ [Batch, K, N] = [Batch, M, N]
        if len(a.shape) != 3 or len(b.shape) != 3:
            raise ValueError(
                f"TOSA MatMul requires 3D tensors, got shapes {a.shape} and {b.shape}"
            )

        batch = a.shape[0]
        m = a.shape[1]
        n = b.shape[2]
        y = self.tensor(dtype=a.dtype, shape=[batch, m, n])
        self.write(name="tosa.matmul", operands=[a, b], results=[y], attributes=kwargs)
        return y

    def reshape(self, x: Tensor, new_shape: List[int], **kwargs) -> Tensor:
        y = self.tensor(dtype=x.dtype, shape=new_shape)
        new_shape = Array(new_shape)
        self.write(
            name="tosa.reshape",
            operands=[x],
            results=[y],
            attributes={"new_shape": new_shape, **kwargs},
        )
        return y

    def transpose(self, x: Tensor, perm: List[int], **kwargs) -> Tensor:
        # TOSA transpose requires perm as a constant tensor operand (2nd operand)
        perm_const = self.constant(np.array(perm, dtype=np.int32))
        y_shape = [x.shape[i] for i in perm]
        y = self.tensor(dtype=x.dtype, shape=y_shape)
        self.write(
            name="tosa.transpose",
            operands=[x, perm_const],
            results=[y],
            attributes=kwargs,
        )
        return y


if __name__ == "__main__":
    w = TOSAWriter()
    q = w.tensor("f32", [12, 50, 64])
    k = w.tensor("f32", [12, 64, 50])
    x = w.matmul(q, k)
    x = w.abs(x)
    x = w.negate(x)
    x = w.exp(x)
    x = w.log(x)
    x = w.rsqrt(x)
    x = w.reciprocal(x)
    x = w.floor(x)
    x = w.ceil(x)
    x = w.tanh(x)
    x = w.sigmoid(x)
    x = w.add(x, x)
    x = w.sub(x, x)
    x = w.mul(x, x, shift=0)
    x = w.pow(x, x)
    x = w.reshape(x, [12, 50, 50])
    x = w.transpose(x, [0, 2, 1])

    w.ret([x])

    print(w)
