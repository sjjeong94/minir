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

    def tosa_conv2d(
        self,
        input: Value,
        weight: Value,
        bias: Value,
        pad: List[int] = [0, 0, 0, 0],
        stride: List[int] = [1, 1],
        dilation: List[int] = [1, 1],
        acc_type: str = "float32",
    ) -> Value:
        shape = [input.shape[0]]
        spatial_dims = len(input.shape) - 2
        for i in range(spatial_dims):
            xs, ws = input.shape[i + 1], weight.shape[i + 1]
            pad_size = pad[i] + pad[i + spatial_dims]
            dil, str = dilation[i], stride[i]
            s = (xs + pad_size - dil * (ws - 1) - 1) // str + 1
            shape.append(s)
        shape.append(weight.shape[0])

        output = self.empty(dtype=input.dtype, shape=shape)
        self.write(
            name="tosa.conv2d",
            operands=[input, weight, bias],
            results=[output],
            attributes={
                "pad": np.int64(pad),
                "stride": np.int64(stride),
                "dilation": np.int64(dilation),
                "acc_type": np.dtype(acc_type),
            },
        )
        return output

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

    def tensor_reshape(self, source: Value, shape: List[int]) -> Value:
        result = self.empty(dtype=source.dtype, shape=shape)
        shape_value = self.constant(np.int32(shape))
        self.write(
            name="tensor.reshape",
            operands=[source, shape_value],
            results=[result],
        )
        return result

    def tensor_concat(self, inputs: List[Value], dim: int) -> Value:
        shape = inputs[0].shape.copy()
        for i in range(1, len(inputs)):
            shape[dim] += inputs[i].shape[dim]
        result = self.empty(dtype=inputs[0].dtype, shape=shape)
        self.write(
            name="tensor.concat",
            operands=inputs,
            results=[result],
            attributes={"dim": np.int64(dim)},
        )
        return result

    def tensor_dim(self, source: Value, index: Value) -> Value:
        result = self.empty(dtype="index", shape=[])
        self.write(
            name="tensor.dim",
            operands=[source, index],
            results=[result],
        )

    def tensor_rank(self, tensor: Value) -> Value:
        result = self.empty(dtype="index", shape=[])
        self.write(
            name="tensor.rank",
            operands=[tensor],
            results=[result],
        )
        return result

    def math_absf(self, operand: Value) -> Value:
        return self.unary_op("math.absf", operand)

    def math_absi(self, operand: Value) -> Value:
        return self.unary_op("math.absi", operand)

    def math_acos(self, operand: Value) -> Value:
        return self.unary_op("math.acos", operand)

    def math_acosh(self, operand: Value) -> Value:
        return self.unary_op("math.acosh", operand)

    def math_asin(self, operand: Value) -> Value:
        return self.unary_op("math.asin", operand)

    def math_asinh(self, operand: Value) -> Value:
        return self.unary_op("math.asinh", operand)

    def math_atan(self, operand: Value) -> Value:
        return self.unary_op("math.atan", operand)

    def math_atan2(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op("math.atan2", lhs, rhs)

    def math_atanh(self, operand: Value) -> Value:
        return self.unary_op("math.atanh", operand)

    def math_cbrt(self, operand: Value) -> Value:
        return self.unary_op("math.cbrt", operand)

    def math_ceil(self, operand: Value) -> Value:
        return self.unary_op("math.ceil", operand)

    def math_copysign(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op("math.copysign", lhs, rhs)

    def math_cos(self, operand: Value) -> Value:
        return self.unary_op("math.cos", operand)

    def math_cosh(self, operand: Value) -> Value:
        return self.unary_op("math.cosh", operand)

    def math_ctlz(self, operand: Value) -> Value:
        return self.unary_op("math.ctlz", operand)

    def math_ctpop(self, operand: Value) -> Value:
        return self.unary_op("math.ctpop", operand)

    def math_cttz(self, operand: Value) -> Value:
        return self.unary_op("math.cttz", operand)

    def math_erf(self, operand: Value) -> Value:
        return self.unary_op("math.erf", operand)

    def math_erfc(self, operand: Value) -> Value:
        return self.unary_op("math.erfc", operand)

    def math_exp(self, operand: Value) -> Value:
        return self.unary_op("math.exp", operand)

    def math_exp2(self, operand: Value) -> Value:
        return self.unary_op("math.exp2", operand)

    def math_expm1(self, operand: Value) -> Value:
        return self.unary_op("math.expm1", operand)

    def math_floor(self, operand: Value) -> Value:
        return self.unary_op("math.floor", operand)

    def math_fma(self, a: Value, b: Value, c: Value) -> Value:
        result = self.empty(dtype=a.dtype, shape=a.shape)
        self.write(
            name="math.fma",
            operands=[a, b, c],
            results=[result],
        )
        return result

    def math_fpowi(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op("math.fpowi", lhs, rhs)

    def math_ipowi(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op("math.ipowi", lhs, rhs)

    def math_isfinite(self, operand: Value) -> Value:
        return self.unary_op("math.isfinite", operand)

    def math_isinf(self, operand: Value) -> Value:
        return self.unary_op("math.isinf", operand)

    def math_isnan(self, operand: Value) -> Value:
        return self.unary_op("math.isnan", operand)

    def math_isnan(self, operand: Value) -> Value:
        return self.unary_op("math.isnan", operand)

    def math_isnormal(self, operand: Value) -> Value:
        return self.unary_op("math.isnormal", operand)

    def math_log(self, operand: Value) -> Value:
        return self.unary_op("math.log", operand)

    def math_log10(self, operand: Value) -> Value:
        return self.unary_op("math.log10", operand)

    def math_log1p(self, operand: Value) -> Value:
        return self.unary_op("math.log1p", operand)

    def math_log2(self, operand: Value) -> Value:
        return self.unary_op("math.log2", operand)

    def math_powf(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op("math.powf", lhs, rhs)

    def math_round(self, operand: Value) -> Value:
        return self.unary_op("math.round", operand)

    def math_roundeven(self, operand: Value) -> Value:
        return self.unary_op("math.roundeven", operand)

    def math_rsqrt(self, operand: Value) -> Value:
        return self.unary_op("math.rsqrt", operand)

    def math_sin(self, operand: Value) -> Value:
        return self.unary_op("math.sin", operand)

    def math_sinh(self, operand: Value) -> Value:
        return self.unary_op("math.sinh", operand)

    def math_sqrt(self, operand: Value) -> Value:
        return self.unary_op("math.sqrt", operand)

    def math_tan(self, operand: Value) -> Value:
        return self.unary_op("math.tan", operand)

    def math_tanh(self, operand: Value) -> Value:
        return self.unary_op("math.tanh", operand)

    def math_trunc(self, operand: Value) -> Value:
        return self.unary_op("math.trunc", operand)
