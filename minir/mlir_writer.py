import numpy as np
from typing import Optional, List
from minir.ir import Value
from minir.writer import Writer


class MLIRWriter(Writer):
    def to_mlir(self, save_path: Optional[str] = None, elide: bool = False) -> str:
        mlir = self.to_function().__repr__(elide=elide)
        if save_path is not None:
            with open(save_path, "w") as f:
                f.write(mlir)
        return mlir

    def unary_op(self, name: str, operand: Value, **kwargs) -> Value:
        result = self.empty(dtype=operand.dtype, shape=operand.shape)
        self.write(name=name, operands=[operand], results=[result], attributes=kwargs)
        return result

    def binary_op(self, name: str, lhs: Value, rhs: Value, **kwargs) -> Value:
        result = self.empty(dtype=lhs.dtype, shape=lhs.shape)
        self.write(name=name, operands=[lhs, rhs], results=[result], attributes=kwargs)
        return result

    def tosa_abs(self, operand: Value) -> Value:
        return self.unary_op("tosa.abs", operand)

    def tosa_ceil(self, operand: Value) -> Value:
        return self.unary_op("tosa.ceil", operand)

    def tosa_cos(self, operand: Value) -> Value:
        return self.unary_op("tosa.cos", operand)

    def tosa_erf(self, operand: Value) -> Value:
        return self.unary_op("tosa.erf", operand)

    def tosa_exp(self, operand: Value) -> Value:
        return self.unary_op("tosa.exp", operand)

    def tosa_floor(self, operand: Value) -> Value:
        return self.unary_op("tosa.floor", operand)

    def tosa_identity(self, operand: Value) -> Value:
        return self.unary_op("tosa.identity", operand)

    def tosa_log(self, operand: Value) -> Value:
        return self.unary_op("tosa.log", operand)

    def tosa_negate(self, operand: Value) -> Value:
        return self.unary_op("tosa.negate", operand)

    def tosa_rsqrt(self, operand: Value) -> Value:
        return self.unary_op("tosa.rsqrt", operand)

    def tosa_sigmoid(self, operand: Value) -> Value:
        return self.unary_op("tosa.sigmoid", operand)

    def tosa_sin(self, operand: Value) -> Value:
        return self.unary_op("tosa.sin", operand)

    def tosa_tanh(self, operand: Value) -> Value:
        return self.unary_op("tosa.tanh", operand)

    def tosa_clamp(self, operand: Value, min_val: float, max_val: float) -> Value:
        return self.unary_op("tosa.clamp", operand, min_val=min_val, max_val=max_val)

    def tosa_add(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op("tosa.add", lhs, rhs)

    def tosa_sub(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op("tosa.sub", lhs, rhs)

    def tosa_pow(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op("tosa.pow", lhs, rhs)

    def tosa_concat(self, operands: List[Value], axis: int) -> Value:
        shape = operands[0].shape.copy()
        for i in range(1, len(operands)):
            shape[axis] += operands[i].shape[axis]
        result = self.empty(dtype=operands[0].dtype, shape=shape)
        self.write(
            name="tosa.concat",
            operands=operands,
            results=[result],
            attributes={"axis": axis},
        )
        return result

    def tosa_matmul(self, lhs: Value, rhs: Value) -> Value:
        shape = lhs.shape[:-1] + rhs.shape[-1:]
        result = self.empty(dtype=lhs.dtype, shape=shape)
        self.write(
            name="tosa.matmul",
            operands=[lhs, rhs],
            results=[result],
        )
        return result

    def tosa_transpose(self, operand: Value, perm: List[int]) -> Value:
        result = self.empty(dtype=operand.dtype, shape=[operand.shape[i] for i in perm])
        perm_value = self.constant(np.int32(perm))
        self.write("tosa.transpose", [operand, perm_value], [result])
        return result

    def tosa_cast(self, operand: Value, dtype: str) -> Value:
        result = self.empty(dtype=dtype, shape=operand.shape)
        self.write(
            name="tosa.cast",
            operands=[operand],
            results=[result],
        )
        return result

    def arith_negf(self, operand: Value) -> Value:
        return self.unary_op("arith.negf", operand)

    def arith_addf(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op("arith.addf", lhs, rhs)

    def arith_subf(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op("arith.subf", lhs, rhs)

    def arith_mulf(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op("arith.mulf", lhs, rhs)

    def arith_divf(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op("arith.divf", lhs, rhs)

    def arith_addi(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op("arith.addi", lhs, rhs)

    def arith_subi(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op("arith.subi", lhs, rhs)

    def arith_muli(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op("arith.muli", lhs, rhs)

    def arith_divsi(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op("arith.divsi", lhs, rhs)

    def arith_divui(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op("arith.divui", lhs, rhs)

    def arith_andi(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op("arith.andi", lhs, rhs)
