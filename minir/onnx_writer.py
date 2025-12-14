import onnx
import numpy as np
from typing import Any, Dict, List, Optional, Union
from minir.ir import Function, Operation, Tensor, Dense
from minir.utils import numpy_to_dtype, product, dtype_to_numpy
from minir.onnx_utils import dtype_to_onnx, to_onnx


class ONNXWriter:
    def __init__(self) -> None:
        self.operations: List[Operation] = []

    def __repr__(self) -> str:
        return repr(self.to_function())

    def to_function(self, name: str = "function") -> Function:
        if not self.operations or self.operations[-1].name != "func.return":
            self.ret()
        operations = [op for op in self.operations if op.name == "arith.constant"]
        for op in self.operations:
            if op.name != "arith.constant":
                operations.append(op)
        return Function(operations, name=name)

    def to_onnx(
        self,
        save_path: Optional[str] = None,
        check_model: bool = True,
    ) -> onnx.ModelProto:
        model = to_onnx(self.to_function())
        if check_model:
            onnx.checker.check_model(model, full_check=True)
        if save_path is not None:
            onnx.save(model, save_path)
        return model

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
            name="arith.constant",
            operands=[],
            results=[tensor],
            attributes={
                "value": Dense(array.tobytes(), dtype=tensor.dtype, shape=tensor.shape)
            },
        )
        return tensor

    def unary_op(self, name: str, x: Tensor, **kwargs) -> Tensor:
        y = self.tensor(dtype=x.dtype, shape=x.shape)
        self.write(name=name, operands=[x], results=[y], attributes=kwargs)
        return y

    def Identity(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Identity", x=x, **kwargs)

    def Abs(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Abs", x=x, **kwargs)

    def Neg(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Neg", x=x, **kwargs)

    def Exp(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Exp", x=x, **kwargs)

    def Log(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Log", x=x, **kwargs)

    def Sqrt(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Sqrt", x=x, **kwargs)

    def Cos(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Cos", x=x, **kwargs)

    def Sin(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Sin", x=x, **kwargs)

    def Tan(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Tan", x=x, **kwargs)

    def Acos(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Acos", x=x, **kwargs)

    def Asin(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Asin", x=x, **kwargs)

    def Atan(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Atan", x=x, **kwargs)

    def Cosh(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Cosh", x=x, **kwargs)

    def Sinh(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Sinh", x=x, **kwargs)

    def Tanh(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Tanh", x=x, **kwargs)

    def Acosh(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Acosh", x=x, **kwargs)

    def Asinh(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Asinh", x=x, **kwargs)

    def Atanh(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Atanh", x=x, **kwargs)

    def Reciprocal(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Reciprocal", x=x, **kwargs)

    def Floor(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Floor", x=x, **kwargs)

    def Ceil(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Ceil", x=x, **kwargs)

    def Round(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Round", x=x, **kwargs)

    def Relu(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Relu", x=x, **kwargs)

    def Sigmoid(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Sigmoid", x=x, **kwargs)

    def Elu(self, x: Tensor, alpha: float = 1.0, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Elu", x=x, alpha=alpha, **kwargs)

    def Selu(
        self, x: Tensor, alpha: float = 1.67326, gamma: float = 1.0507, **kwargs
    ) -> Tensor:
        return self.unary_op(name="onnx.Selu", x=x, alpha=alpha, gamma=gamma, **kwargs)

    def ThresholdedRelu(self, x: Tensor, alpha: float = 1.0, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.ThresholdedRelu", x=x, alpha=alpha, **kwargs)

    def HardSigmoid(
        self, x: Tensor, alpha: float = 0.2, beta: float = 0.5, **kwargs
    ) -> Tensor:
        return self.unary_op(
            name="onnx.HardSigmoid", x=x, alpha=alpha, beta=beta, **kwargs
        )

    def HardSwish(self, x: Tensor, **kwargs) -> Tensor:
        # HardSwish(x) = x * HardSigmoid(x)
        return self.Mul(x, self.HardSigmoid(x), **kwargs)

    def Softsign(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Softsign", x=x, **kwargs)

    def Softplus(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Softplus", x=x, **kwargs)

    def LeakyRelu(self, x: Tensor, alpha: float = 0.01, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.LeakyRelu", x=x, alpha=alpha, **kwargs)

    def Gelu(self, x: Tensor, approximate: str = "none", **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Gelu", x=x, approximate=approximate, **kwargs)

    def Mish(self, x: Tensor, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Mish", x=x, **kwargs)

    def Swish(self, x: Tensor, **kwargs) -> Tensor:
        # Swish(x) = x * Sigmoid(x)
        return self.Mul(x, self.Sigmoid(x), **kwargs)

    def Softmax(self, x: Tensor, axis: int = -1, **kwargs) -> Tensor:
        return self.unary_op(name="onnx.Softmax", x=x, axis=axis, **kwargs)

    def binary_op(self, name: str, a: Tensor, b: Tensor, **kwargs) -> Tensor:
        ref = a if product(a.shape) >= product(b.shape) else b
        c = self.tensor(dtype=ref.dtype, shape=ref.shape)
        self.write(name=name, operands=[a, b], results=[c], attributes=kwargs)
        return c

    def Mul(self, a: Tensor, b: Tensor, **kwargs) -> Tensor:
        return self.binary_op(name="onnx.Mul", a=a, b=b, **kwargs)

    def Add(self, a: Tensor, b: Tensor, **kwargs) -> Tensor:
        return self.binary_op(name="onnx.Add", a=a, b=b, **kwargs)

    def Sub(self, a: Tensor, b: Tensor, **kwargs) -> Tensor:
        return self.binary_op(name="onnx.Sub", a=a, b=b, **kwargs)

    def Div(self, a: Tensor, b: Tensor, **kwargs) -> Tensor:
        return self.binary_op(name="onnx.Div", a=a, b=b, **kwargs)

    def Pow(self, a: Tensor, b: Tensor, **kwargs) -> Tensor:
        return self.binary_op(name="onnx.Pow", a=a, b=b, **kwargs)

    def PRelu(self, x: Tensor, slope: Tensor, **kwargs) -> Tensor:
        return self.binary_op(name="onnx.PRelu", a=x, b=slope, **kwargs)

    def reduce_op(
        self,
        name: str,
        x: Tensor,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Tensor:
        shape = []
        for i in range(x.rank):
            if i in axes:
                if keepdims:
                    shape.append(1)
            else:
                shape.append(x.shape[i])
        y = self.tensor(dtype=x.dtype, shape=shape)
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
        x: Tensor,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Tensor:
        return self.reduce_op(
            name="onnx.ReduceSum", x=x, axes=axes, keepdims=keepdims, **kwargs
        )

    def ReduceSumSquare(
        self,
        x: Tensor,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Tensor:
        return self.reduce_op(
            name="onnx.ReduceSumSquare", x=x, axes=axes, keepdims=keepdims, **kwargs
        )

    def ReduceMean(
        self,
        x: Tensor,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Tensor:
        return self.reduce_op(
            name="onnx.ReduceMean", x=x, axes=axes, keepdims=keepdims, **kwargs
        )

    def ReduceMax(
        self,
        x: Tensor,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Tensor:
        return self.reduce_op(
            name="onnx.ReduceMax", x=x, axes=axes, keepdims=keepdims, **kwargs
        )

    def ReduceMin(
        self,
        x: Tensor,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Tensor:
        return self.reduce_op(
            name="onnx.ReduceMin", x=x, axes=axes, keepdims=keepdims, **kwargs
        )

    def ReduceProd(
        self,
        x: Tensor,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Tensor:
        return self.reduce_op(
            name="onnx.ReduceProd", x=x, axes=axes, keepdims=keepdims, **kwargs
        )

    def ReduceL1(
        self,
        x: Tensor,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Tensor:
        return self.reduce_op(
            name="onnx.ReduceL1", x=x, axes=axes, keepdims=keepdims, **kwargs
        )

    def ReduceL2(
        self,
        x: Tensor,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Tensor:
        return self.reduce_op(
            name="onnx.ReduceL2", x=x, axes=axes, keepdims=keepdims, **kwargs
        )

    def ReduceLogSum(
        self,
        x: Tensor,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Tensor:
        return self.reduce_op(
            name="onnx.ReduceLogSum", x=x, axes=axes, keepdims=keepdims, **kwargs
        )

    def ReduceLogSumExp(
        self,
        x: Tensor,
        axes: List[int],
        keepdims: int = 0,
        **kwargs,
    ) -> Tensor:
        return self.reduce_op(
            name="onnx.ReduceLogSumExp", x=x, axes=axes, keepdims=keepdims, **kwargs
        )

    def global_op(self, name: str, x: Tensor, **kwargs) -> Tensor:
        shape = x.shape.copy()
        for i in range(2, len(shape)):
            shape[i] = 1
        y = self.tensor(dtype=x.dtype, shape=shape)
        self.write(name=name, operands=[x], results=[y], attributes=kwargs)
        return y

    def GlobalAveragePool(self, x: Tensor, **kwargs) -> Tensor:
        return self.global_op(name="onnx.GlobalAveragePool", x=x, **kwargs)

    def GlobalMaxPool(self, x: Tensor, **kwargs) -> Tensor:
        return self.global_op(name="onnx.GlobalMaxPool", x=x, **kwargs)

    def GlobalLpPool(self, x: Tensor, p: int = 2, **kwargs) -> Tensor:
        return self.global_op(name="onnx.GlobalLpPool", x=x, p=p, **kwargs)

    def Clip(self, x: Tensor, min: Any, max: Any, **kwargs) -> Tensor:
        min = self.constant(np.array(min, dtype=dtype_to_numpy(x.dtype)))
        max = self.constant(np.array(max, dtype=dtype_to_numpy(x.dtype)))
        y = self.tensor(dtype=x.dtype, shape=x.shape)
        self.write(
            name="onnx.Clip",
            operands=[x, min, max],
            results=[y],
            attributes=kwargs,
        )
        return y

    def Gemm(
        self,
        a: Tensor,
        b: Tensor,
        c: Optional[Tensor] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        transA: int = 0,
        transB: int = 0,
        **kwargs,
    ) -> Tensor:
        row = a.shape[0] if transA == 0 else a.shape[1]
        col = b.shape[1] if transB == 0 else b.shape[0]
        y = self.tensor(dtype=a.dtype, shape=[row, col])
        self.write(
            name="onnx.Gemm",
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

    def MatMul(self, a: Tensor, b: Tensor, **kwargs) -> Tensor:
        y = self.tensor(dtype=a.dtype, shape=a.shape[:-1] + b.shape[-1:])
        self.write(name="onnx.MatMul", operands=[a, b], results=[y], attributes=kwargs)
        return y

    def Pad(
        self,
        x: Tensor,
        pads: List[int],
        mode: str = "constant",
        constant_value: Optional[Any] = None,
        **kwargs,
    ) -> Tensor:
        shape = []
        for i in range(len(x.shape)):
            shape.append(x.shape[i] + pads[i] + pads[i + len(pads) // 2])
        y = self.tensor(dtype=x.dtype, shape=shape)
        pads_value = self.constant(np.array(pads, dtype=np.int64))
        inputs = [x, pads_value]
        if constant_value is not None:
            inputs.append(
                self.constant(np.array(constant_value, dtype=dtype_to_numpy(x.dtype)))
            )
        self.write(
            name="onnx.Pad",
            operands=inputs,
            results=[y],
            attributes={"mode": mode, **kwargs},
        )
        return y

    def Conv(
        self,
        x: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
        dilations: List[int] = [1, 1],
        group: int = 1,
        pads: List[int] = [0, 0, 0, 0],
        strides: List[int] = [1, 1],
        **kwargs,
    ) -> Tensor:
        shape = [x.shape[0], w.shape[0]]
        spatial_dims = len(x.shape) - 2
        for i in range(spatial_dims):
            xs, ws = x.shape[i + 2], w.shape[i + 2]
            pad_size = pads[i] + pads[i + spatial_dims]
            dilation, stride = dilations[i], strides[i]
            s = (xs + pad_size - dilation * (ws - 1) - 1) // stride + 1
            shape.append(s)

        y = self.tensor(dtype=x.dtype, shape=shape)
        self.write(
            name="onnx.Conv",
            operands=[x, w, b] if b is not None else [x, w],
            results=[y],
            attributes={
                "dilations": dilations,
                "group": group,
                "pads": pads,
                "strides": strides,
                **kwargs,
            },
        )
        return y

    def ConvTranspose(
        self,
        x: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
        dilations: List[int] = [1, 1],
        group: int = 1,
        pads: List[int] = [0, 0, 0, 0],
        strides: List[int] = [1, 1],
        **kwargs,
    ) -> Tensor:
        shape = [x.shape[0], w.shape[1] * group]
        spatial_dims = len(x.shape) - 2
        for i in range(spatial_dims):
            xs, ws = x.shape[i + 2], w.shape[i + 2]
            pad_size = pads[i] + pads[i + spatial_dims]
            dilation, stride = dilations[i], strides[i]
            s = stride * (xs - 1) + dilation * (ws - 1) + 1 - pad_size
            shape.append(s)

        y = self.tensor(dtype=x.dtype, shape=shape)
        self.write(
            name="onnx.ConvTranspose",
            operands=[x, w, b] if b is not None else [x, w],
            results=[y],
            attributes={
                "dilations": dilations,
                "group": group,
                "pads": pads,
                "strides": strides,
                **kwargs,
            },
        )
        return y

    def Reshape(self, x: Tensor, shape: List[int], **kwargs) -> Tensor:
        shape_value = self.constant(np.array(shape, dtype=np.int64))
        y = self.tensor(dtype=x.dtype, shape=shape)
        self.write("onnx.Reshape", [x, shape_value], [y], attributes=kwargs)
        return y

    def Transpose(self, x: Tensor, perm: List[int], **kwargs) -> Tensor:
        y = self.tensor(dtype=x.dtype, shape=[x.shape[i] for i in perm])
        self.write("onnx.Transpose", [x], [y], attributes={"perm": perm, **kwargs})
        return y

    def Squeeze(self, x: Tensor, axes: List[int], **kwargs) -> Tensor:
        shape = [x.shape[i] for i in range(len(x.shape)) if i not in axes]
        y = self.tensor(dtype=x.dtype, shape=shape)
        axes_value = self.constant(np.array(axes, dtype=np.int64))
        self.write("onnx.Squeeze", [x, axes_value], [y], attributes=kwargs)
        return y

    def Unsqueeze(self, x: Tensor, axes: List[int], **kwargs) -> Tensor:
        shape = x.shape.copy()
        for axis in axes:
            if axis < 0:
                axis += len(shape) + 1
            shape.insert(axis, 1)
        y = self.tensor(dtype=x.dtype, shape=shape)
        axes_value = self.constant(np.array(axes, dtype=np.int64))
        self.write("onnx.Unsqueeze", [x, axes_value], [y], attributes=kwargs)
        return y

    def Flatten(self, x: Tensor, axis: int = 1, **kwargs) -> Tensor:
        shape = [product(x.shape[:axis]), product(x.shape[axis:])]
        y = self.tensor(dtype=x.dtype, shape=shape)
        self.write("onnx.Flatten", [x], [y], attributes={"axis": axis, **kwargs})
        return y

    def Concat(
        self,
        inputs: List[Tensor],
        axis: int,
        **kwargs,
    ) -> Tensor:
        shape = inputs[0].shape.copy()
        for i in range(1, len(inputs)):
            shape[axis] += inputs[i].shape[axis]
        y = self.tensor(dtype=inputs[0].dtype, shape=shape)
        self.write(
            name="onnx.Concat",
            operands=inputs,
            results=[y],
            attributes={"axis": axis, **kwargs},
        )
        return y

    def Split(
        self,
        x: Tensor,
        split: List[int],
        axis: int = 0,
        **kwargs,
    ) -> List[Tensor]:
        y = []
        for i in range(len(split)):
            shape = x.shape.copy()
            shape[axis] = split[i]
            y.append(self.tensor(dtype=x.dtype, shape=shape))
        split_value = self.constant(np.array(split, dtype=np.int64))
        self.write(
            name="onnx.Split",
            operands=[x, split_value],
            results=y,
            attributes={"axis": axis, **kwargs},
        )
        return y

    def Gather(
        self,
        data: Tensor,
        indices: List[int],
        axis: int = 0,
        **kwargs,
    ) -> Tensor:
        shape = data.shape.copy()
        shape[axis] = len(indices)
        indices_value = self.constant(np.array(indices, dtype=np.int64))
        y = self.tensor(dtype=data.dtype, shape=shape)
        self.write(
            name="onnx.Gather",
            operands=[data, indices_value],
            results=[y],
            attributes={"axis": axis, **kwargs},
        )
        return y

    def DepthToSpace(
        self,
        x: Tensor,
        blocksize: int = 2,
        mode: str = "DCR",
        **kwargs,
    ) -> Tensor:
        n, c, h, w = x.shape
        shape = [n, c // (blocksize**2), h * blocksize, w * blocksize]
        y = self.tensor(dtype=x.dtype, shape=shape)
        self.write(
            name="onnx.DepthToSpace",
            operands=[x],
            results=[y],
            attributes={"blocksize": blocksize, "mode": mode, **kwargs},
        )
        return y

    def SpaceToDepth(
        self,
        x: Tensor,
        blocksize: int = 2,
        **kwargs,
    ) -> Tensor:
        n, c, h, w = x.shape
        shape = [n, c * (blocksize**2), h // blocksize, w // blocksize]
        y = self.tensor(dtype=x.dtype, shape=shape)
        self.write(
            name="onnx.SpaceToDepth",
            operands=[x],
            results=[y],
            attributes={"blocksize": blocksize, **kwargs},
        )
        return y

    def LayerNormalization(
        self,
        x: Tensor,
        scale: Tensor,
        bias: Optional[Tensor] = None,
        axis: int = -1,
        epsilon: float = 1e-5,
        **kwargs,
    ) -> Tensor:
        y = self.tensor(dtype=x.dtype, shape=x.shape)
        self.write(
            name="onnx.LayerNormalization",
            operands=[x, scale, bias] if bias is not None else [x, scale],
            results=[y],
            attributes={
                "axis": axis,
                "epsilon": epsilon,
                **kwargs,
            },
        )
        return y

    def InstanceNormalization(
        self,
        x: Tensor,
        scale: Tensor,
        bias: Tensor,
        epsilon: float = 1e-5,
        **kwargs,
    ) -> Tensor:
        y = self.tensor(dtype=x.dtype, shape=x.shape)
        self.write(
            name="onnx.InstanceNormalization",
            operands=[x, scale, bias],
            results=[y],
            attributes={
                "epsilon": epsilon,
                **kwargs,
            },
        )
        return y

    def GroupNormalization(
        self,
        x: Tensor,
        scale: Tensor,
        bias: Tensor,
        num_groups: int,
        epsilon: float = 1e-5,
        **kwargs,
    ) -> Tensor:
        y = self.tensor(dtype=x.dtype, shape=x.shape)
        self.write(
            name="onnx.GroupNormalization",
            operands=[x, scale, bias],
            results=[y],
            attributes={
                "num_groups": num_groups,
                "epsilon": epsilon,
                **kwargs,
            },
        )
        return y

    def pool_op(
        self,
        name: str,
        x: Tensor,
        kernel_shape: List[int],
        dilations: List[int] = [1, 1],
        pads: List[int] = [0, 0, 0, 0],
        strides: List[int] = [1, 1],
        **kwargs,
    ) -> Tensor:
        shape = [x.shape[0], x.shape[1]]
        spatial_dims = len(x.shape) - 2
        for i in range(spatial_dims):
            xs = x.shape[i + 2]
            pad_size = pads[i] + pads[i + spatial_dims]
            dilation, stride = dilations[i], strides[i]
            s = (xs + pad_size - dilation * (kernel_shape[i] - 1) - 1) // stride + 1
            shape.append(s)
        y = self.tensor(dtype=x.dtype, shape=shape)
        self.write(
            name=name,
            operands=[x],
            results=[y],
            attributes={
                "kernel_shape": kernel_shape,
                "dilations": dilations,
                "pads": pads,
                "strides": strides,
                **kwargs,
            },
        )
        return y

    def AveragePool(
        self,
        x: Tensor,
        kernel_shape: List[int],
        dilations: List[int] = [1, 1],
        pads: List[int] = [0, 0, 0, 0],
        strides: List[int] = [1, 1],
        **kwargs,
    ) -> Tensor:
        return self.pool_op(
            name="onnx.AveragePool",
            x=x,
            kernel_shape=kernel_shape,
            dilations=dilations,
            pads=pads,
            strides=strides,
            **kwargs,
        )

    def MaxPool(
        self,
        x: Tensor,
        kernel_shape: List[int],
        dilations: List[int] = [1, 1],
        pads: List[int] = [0, 0, 0, 0],
        strides: List[int] = [1, 1],
        **kwargs,
    ) -> Tensor:
        return self.pool_op(
            name="onnx.MaxPool",
            x=x,
            kernel_shape=kernel_shape,
            dilations=dilations,
            pads=pads,
            strides=strides,
            **kwargs,
        )

    def LpPool(
        self,
        x: Tensor,
        kernel_shape: List[int],
        p: int = 2,
        dilations: List[int] = [1, 1],
        pads: List[int] = [0, 0, 0, 0],
        strides: List[int] = [1, 1],
        **kwargs,
    ) -> Tensor:
        return self.pool_op(
            name="onnx.LpPool",
            x=x,
            kernel_shape=kernel_shape,
            dilations=dilations,
            pads=pads,
            strides=strides,
            p=p,
            **kwargs,
        )

    def Cast(
        self,
        x: Tensor,
        to: str,
        **kwargs,
    ) -> Tensor:
        y = self.tensor(dtype=to, shape=x.shape)
        self.write(
            name="onnx.Cast",
            operands=[x],
            results=[y],
            attributes={"to": dtype_to_onnx(to), **kwargs},
        )
        return y
