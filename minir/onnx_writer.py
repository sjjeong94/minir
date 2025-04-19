import onnx
import numpy as np
from typing import Optional, Any, List
from minir.ir import Value
from minir.writer import Writer
from minir.onnx import to_onnx


class ONNXWriter(Writer):
    # opset_version = 20
    def to_onnx(self, save_path: Optional[str] = None) -> onnx.ModelProto:
        model = to_onnx(self.to_function())
        onnx.checker.check_model(model, full_check=True)
        if save_path is not None:
            onnx.save(model, save_path)
        return model

    def unary_op(self, name: str, x: Value, **kwargs) -> Value:
        y = self.empty(dtype=x.dtype, shape=x.shape)
        self.write(name=name, operands=[x], results=[y], attributes=kwargs)
        return y

    def Abs(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Abs", x=x, **kwargs)

    def Neg(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Neg", x=x, **kwargs)

    def Exp(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Exp", x=x, **kwargs)

    def Log(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Log", x=x, **kwargs)

    def Sqrt(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Sqrt", x=x, **kwargs)

    def Cos(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Cos", x=x, **kwargs)

    def Sin(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Sin", x=x, **kwargs)

    def Tan(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Tan", x=x, **kwargs)

    def Acos(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Acos", x=x, **kwargs)

    def Asin(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Asin", x=x, **kwargs)

    def Atan(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Atan", x=x, **kwargs)

    def Cosh(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Cosh", x=x, **kwargs)

    def Sinh(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Sinh", x=x, **kwargs)

    def Tanh(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Tanh", x=x, **kwargs)

    def Acosh(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Acosh", x=x, **kwargs)

    def Asinh(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Asinh", x=x, **kwargs)

    def Atanh(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Atanh", x=x, **kwargs)

    def Reciprocal(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Reciprocal", x=x, **kwargs)

    def Floor(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Floor", x=x, **kwargs)

    def Ceil(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Ceil", x=x, **kwargs)

    def Round(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Round", x=x, **kwargs)

    def Relu(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Relu", x=x, **kwargs)

    def Sigmoid(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Sigmoid", x=x, **kwargs)

    def Elu(self, x: Value, alpha: float = 1.0, **kwargs) -> Value:
        return self.unary_op(name="Elu", x=x, alpha=alpha, **kwargs)

    def Selu(
        self, x: Value, alpha: float = 1.67326, gamma: float = 1.0507, **kwargs
    ) -> Value:
        return self.unary_op(name="Selu", x=x, alpha=alpha, gamma=gamma, **kwargs)

    def ThresholdedRelu(self, x: Value, alpha: float = 1.0, **kwargs) -> Value:
        return self.unary_op(name="ThresholdedRelu", x=x, alpha=alpha, **kwargs)

    def HardSigmoid(
        self, x: Value, alpha: float = 0.2, beta: float = 0.5, **kwargs
    ) -> Value:
        return self.unary_op(name="HardSigmoid", x=x, alpha=alpha, beta=beta, **kwargs)

    def HardSwish(self, x: Value, **kwargs) -> Value:
        # HardSwish(x) = x * HardSigmoid(x)
        return self.Mul(x, self.HardSigmoid(x), **kwargs)

    def Softsign(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Softsign", x=x, **kwargs)

    def Softplus(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Softplus", x=x, **kwargs)

    def LeakyRelu(self, x: Value, alpha: float = 0.01, **kwargs) -> Value:
        return self.unary_op(name="LeakyRelu", x=x, alpha=alpha, **kwargs)

    def Gelu(self, x: Value, approximate: str = "none", **kwargs) -> Value:
        return self.unary_op(name="Gelu", x=x, approximate=approximate, **kwargs)

    def Mish(self, x: Value, **kwargs) -> Value:
        return self.unary_op(name="Mish", x=x, **kwargs)

    def Swish(self, x: Value, **kwargs) -> Value:
        # Swish(x) = x * Sigmoid(x)
        return self.Mul(x, self.Sigmoid(x), **kwargs)

    def Softmax(self, x: Value, axis: int = -1, **kwargs) -> Value:
        return self.unary_op(name="Softmax", x=x, axis=axis, **kwargs)

    def binary_op(self, name: str, a: Value, b: Value, **kwargs) -> Value:
        ref = a if a.numel >= b.numel else b
        c = self.empty(dtype=ref.dtype, shape=ref.shape)
        self.write(name=name, operands=[a, b], results=[c], attributes=kwargs)
        return c

    def Mul(self, a: Value, b: Value, **kwargs) -> Value:
        return self.binary_op(name="Mul", a=a, b=b, **kwargs)

    def Add(self, a: Value, b: Value, **kwargs) -> Value:
        return self.binary_op(name="Add", a=a, b=b, **kwargs)

    def Sub(self, a: Value, b: Value, **kwargs) -> Value:
        return self.binary_op(name="Sub", a=a, b=b, **kwargs)

    def Div(self, a: Value, b: Value, **kwargs) -> Value:
        return self.binary_op(name="Div", a=a, b=b, **kwargs)

    def Pow(self, a: Value, b: Value, **kwargs) -> Value:
        return self.binary_op(name="Pow", a=a, b=b, **kwargs)

    def PRelu(self, x: Value, slope: Value, **kwargs) -> Value:
        return self.binary_op(name="PRelu", a=x, b=slope, **kwargs)

    def reduce_op(
        self,
        name: str,
        x: Value,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Value:
        shape = []
        for i in range(x.rank):
            if i in axes:
                if keepdims:
                    shape.append(1)
            else:
                shape.append(x.shape[i])
        y = self.empty(dtype=x.dtype, shape=shape)
        axes_value = self.constant(np.array(axes, dtype=np.int64))
        self.write(
            name=name,
            operands=[x, axes_value],
            results=[y],
            attributes={"keepdims": keepdims, **kwargs},
        )
        return y

    def ReduceSum(
        self,
        x: Value,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Value:
        return self.reduce_op(
            name="ReduceSum", x=x, axes=axes, keepdims=keepdims, **kwargs
        )

    def ReduceSumSquare(
        self,
        x: Value,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Value:
        return self.reduce_op(
            name="ReduceSumSquare", x=x, axes=axes, keepdims=keepdims, **kwargs
        )

    def ReduceMean(
        self,
        x: Value,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Value:
        return self.reduce_op(
            name="ReduceMean", x=x, axes=axes, keepdims=keepdims, **kwargs
        )

    def ReduceMax(
        self,
        x: Value,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Value:
        return self.reduce_op(
            name="ReduceMax", x=x, axes=axes, keepdims=keepdims, **kwargs
        )

    def ReduceMin(
        self,
        x: Value,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Value:
        return self.reduce_op(
            name="ReduceMin", x=x, axes=axes, keepdims=keepdims, **kwargs
        )

    def ReduceProd(
        self,
        x: Value,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Value:
        return self.reduce_op(
            name="ReduceProd", x=x, axes=axes, keepdims=keepdims, **kwargs
        )

    def ReduceL1(
        self,
        x: Value,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Value:
        return self.reduce_op(
            name="ReduceL1", x=x, axes=axes, keepdims=keepdims, **kwargs
        )

    def ReduceL2(
        self,
        x: Value,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Value:
        return self.reduce_op(
            name="ReduceL2", x=x, axes=axes, keepdims=keepdims, **kwargs
        )

    def ReduceLogSum(
        self,
        x: Value,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Value:
        return self.reduce_op(
            name="ReduceLogSum", x=x, axes=axes, keepdims=keepdims, **kwargs
        )

    def ReduceLogSumExp(
        self,
        x: Value,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Value:
        return self.reduce_op(
            name="ReduceLogSumExp", x=x, axes=axes, keepdims=keepdims, **kwargs
        )

    def Clip(self, x: Value, min: Any, max: Any, **kwargs) -> Value:
        min = self.constant(np.array(min, dtype=x.dtype))
        max = self.constant(np.array(max, dtype=x.dtype))
        y = self.empty(dtype=x.dtype, shape=x.shape)
        self.write(
            name="Clip",
            operands=[x, min, max],
            results=[y],
            attributes=kwargs,
        )
        return y

    def Gemm(
        self,
        a: Value,
        b: Value,
        c: Optional[Value] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        transA: int = 0,
        transB: int = 0,
        **kwargs,
    ) -> Value:
        row = a.shape[0] if transA == 0 else a.shape[1]
        col = b.shape[1] if transB == 0 else b.shape[0]
        y = self.empty(dtype=a.dtype, shape=[row, col])
        self.write(
            name="Gemm",
            operands=[a, b, c] if c is not None else [a, b],
            results=[y],
            attributes={
                "alpha": alpha,
                "beta": beta,
                "transA": transA,
                "transB": transB,
                **kwargs,
            },
        )
        return y

    def MatMul(self, a: Value, b: Value) -> Value:
        y = self.float32(a.shape[:-1] + b.shape[-1:])
        self.write("MatMul", [a, b], [y])
        return y

    def Reshape(self, x: Value, shape: List[int], **kwargs) -> Value:
        shape_value = self.constant(np.array(shape, dtype=np.int64))
        y = self.empty(dtype=x.dtype, shape=shape)
        self.write("Reshape", [x, shape_value], [y], attributes=kwargs)
        return y

    def Transpose(self, x: Value, perm: List[int], **kwargs) -> Value:
        y = self.empty(dtype=x.dtype, shape=[x.shape[i] for i in perm])
        self.write("Transpose", [x], [y], attributes={"perm": perm, **kwargs})
        return y

    def Squeeze(self, x: Value, axes: List[int], **kwargs) -> Value:
        shape = [x.shape[i] for i in range(len(x.shape)) if i not in axes]
        y = self.empty(dtype=x.dtype, shape=shape)
        axes_value = self.constant(np.array(axes, dtype=np.int64))
        self.write("Squeeze", [x, axes_value], [y], attributes={**kwargs})
        return y

    def Unsqueeze(self, x: Value, axes: List[int], **kwargs) -> Value:
        shape = x.shape.copy()
        for axis in axes:
            if axis < 0:
                axis += len(shape) + 1
            shape.insert(axis, 1)
        y = self.empty(dtype=x.dtype, shape=shape)
        axes_value = self.constant(np.array(axes, dtype=np.int64))
        self.write("Unsqueeze", [x, axes_value], [y], attributes={**kwargs})
        return y

    def Concat(
        self,
        inputs: List[Value],
        axis: int,
        **kwargs,
    ) -> Value:
        shape = inputs[0].shape.copy()
        for i in range(1, len(inputs)):
            shape[axis] += inputs[i].shape[axis]
        y = self.empty(dtype=inputs[0].dtype, shape=shape)
        self.write(
            name="Concat",
            operands=inputs,
            results=[y],
            attributes={"axis": axis, **kwargs},
        )
        return y

    def Gather(
        self,
        data: Value,
        indices: Value,
        axis: int = 0,
        **kwargs,
    ) -> Value:
        shape = data.shape.copy()
        shape[axis] = indices.shape[0]
        y = self.empty(dtype=data.dtype, shape=shape)
        self.write(
            name="Gather",
            operands=[data, indices],
            results=[y],
            attributes={"axis": axis, **kwargs},
        )
        return y

    def LayerNormalization(
        self,
        x: Value,
        scale: Value,
        bias: Optional[Value] = None,
        axis: int = -1,
        epsilon: float = 1e-5,
        **kwargs,
    ) -> Value:
        y = self.empty(dtype=x.dtype, shape=x.shape)
        self.write(
            name="LayerNormalization",
            operands=[x, scale, bias] if bias is not None else [x, scale],
            results=[y],
            attributes={
                "axis": axis,
                "epsilon": epsilon,
                **kwargs,
            },
        )
        return y
