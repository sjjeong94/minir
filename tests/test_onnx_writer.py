import onnx
from minir.onnx_writer import ONNXWriter


def test_unary_op():
    l = ONNXWriter()
    x = l.float32([3, 4])
    x = l.Identity(x)
    x = l.Abs(x)
    x = l.Neg(x)
    x = l.Exp(x)
    x = l.Log(x)
    x = l.Sqrt(x)
    x = l.Cos(x)
    x = l.Sin(x)
    x = l.Tan(x)
    x = l.Acos(x)
    x = l.Asin(x)
    x = l.Atan(x)
    x = l.Cosh(x)
    x = l.Sinh(x)
    x = l.Tanh(x)
    x = l.Acosh(x)
    x = l.Asinh(x)
    x = l.Atanh(x)
    x = l.Reciprocal(x)
    x = l.Floor(x)
    x = l.Ceil(x)
    x = l.Round(x)
    x = l.Relu(x)
    x = l.Sigmoid(x)
    x = l.Elu(x, alpha=1.0)
    x = l.Selu(x, alpha=1.67326, gamma=1.0507)
    x = l.ThresholdedRelu(x, alpha=1.0)
    x = l.HardSigmoid(x, alpha=0.2, beta=0.5)
    x = l.HardSwish(x)
    x = l.Softplus(x)
    x = l.Softsign(x)
    x = l.LeakyRelu(x, alpha=0.01)
    x = l.Swish(x)
    x = l.Softmax(x, axis=1)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_binary_op():
    l = ONNXWriter()
    a = l.float32([3, 4])
    b = l.float32([3, 4])
    x = l.Add(a, b)
    x = l.Sub(a, b)
    x = l.Mul(a, b)
    x = l.Div(a, b)
    x = l.Pow(a, b)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_reduce_sum():
    l = ONNXWriter()
    x = l.float32([3, 4])
    x = l.ReduceSum(x, axes=[0], keepdims=1)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_reduce_sum_square():
    l = ONNXWriter()
    x = l.float32([3, 4])
    x = l.ReduceSumSquare(x, axes=[0], keepdims=1)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_reduce_mean():
    l = ONNXWriter()
    x = l.float32([3, 4])
    x = l.ReduceMean(x, axes=[0], keepdims=1)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_reduce_max():
    l = ONNXWriter()
    x = l.float32([3, 4])
    x = l.ReduceMax(x, axes=[0], keepdims=1)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_reduce_min():
    l = ONNXWriter()
    x = l.float32([3, 4])
    x = l.ReduceMin(x, axes=[0], keepdims=1)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_reduce_prod():
    l = ONNXWriter()
    x = l.float32([3, 4])
    x = l.ReduceProd(x, axes=[0], keepdims=1)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_reduce_l1():
    l = ONNXWriter()
    x = l.float32([3, 4])
    x = l.ReduceL1(x, axes=[0], keepdims=1)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_reduce_l2():
    l = ONNXWriter()
    x = l.float32([3, 4])
    x = l.ReduceL2(x, axes=[0], keepdims=1)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_reduce_log_sum():
    l = ONNXWriter()
    x = l.float32([3, 4])
    x = l.ReduceLogSum(x, axes=[0], keepdims=1)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_reduce_log_sum_exp():
    l = ONNXWriter()
    x = l.float32([3, 4])
    x = l.ReduceLogSumExp(x, axes=[0], keepdims=1)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_global_average_pool():
    l = ONNXWriter()
    x = l.float32([1, 3, 4, 4])
    x = l.GlobalAveragePool(x)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_global_max_pool():
    l = ONNXWriter()
    x = l.float32([1, 3, 4, 4])
    x = l.GlobalMaxPool(x)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_global_lp_pool():
    l = ONNXWriter()
    x = l.float32([1, 3, 4, 4])
    x = l.GlobalLpPool(x, p=2)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_clip():
    l = ONNXWriter()
    x = l.float32([3, 4])
    x = l.Clip(x, min=0.0, max=1.0)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_gemm():
    l = ONNXWriter()
    a = l.float32([3, 4])
    b = l.float32([4, 5])
    x = l.Gemm(a, b, alpha=1.0, beta=1.0)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_matmul():
    l = ONNXWriter()
    a = l.float32([3, 4])
    b = l.float32([4, 5])
    x = l.MatMul(a, b)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_pad():
    l = ONNXWriter()
    x = l.float32([3, 4])
    x = l.Pad(x, pads=[1, 1, 1, 1], mode="constant", constant_value=0.0)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_conv():
    l = ONNXWriter()
    x = l.float32([1, 3, 4, 4])
    w = l.float32([2, 3, 3, 3])
    b = l.float32([2])
    x = l.Conv(x, w, b, strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_conv_transpose():
    l = ONNXWriter()
    x = l.float32([1, 3, 4, 4])
    w = l.float32([3, 2, 3, 3])
    b = l.float32([2])
    x = l.ConvTranspose(
        x, w, b, strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1
    )
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_reshape():
    l = ONNXWriter()
    x = l.float32([3, 4])
    x = l.Reshape(x, [4, 3])
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_transpose():
    l = ONNXWriter()
    x = l.float32([3, 4])
    x = l.Transpose(x, perm=[1, 0])
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_squeeze():
    l = ONNXWriter()
    x = l.float32([1, 3, 4])
    x = l.Squeeze(x, axes=[0])
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_unsqueeze():
    l = ONNXWriter()
    x = l.float32([3, 4])
    x = l.Unsqueeze(x, axes=[0])
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_concat():
    l = ONNXWriter()
    x1 = l.float32([3, 4])
    x2 = l.float32([3, 4])
    x = l.Concat([x1, x2], axis=0)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_split():
    l = ONNXWriter()
    x = l.float32([3, 4])
    x1, x2 = l.Split(x, split=[2, 1], axis=0)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_gather():
    l = ONNXWriter()
    data = l.float32([3, 4])
    x = l.Gather(data, indices=[2], axis=0)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_layer_norm():
    l = ONNXWriter()
    x = l.float32([3, 4])
    scale = l.float32([4])
    bias = l.float32([4])
    x = l.LayerNormalization(x, scale, bias, axis=-1, epsilon=1e-5)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_instance_norm():
    l = ONNXWriter()
    x = l.float32([3, 4, 8, 8])
    scale = l.float32([4])
    bias = l.float32([4])
    x = l.InstanceNormalization(x, scale, bias, epsilon=1e-5)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_group_norm():
    l = ONNXWriter()
    x = l.float32([3, 4, 8, 8])
    scale = l.float32([2])
    bias = l.float32([2])
    x = l.GroupNormalization(x, scale, bias, num_groups=2, epsilon=1e-5)
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_average_pool():
    l = ONNXWriter()
    x = l.float32([1, 3, 4, 4])
    x = l.AveragePool(x, kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0])
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_max_pool():
    l = ONNXWriter()
    x = l.float32([1, 3, 4, 4])
    x = l.MaxPool(x, kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0])
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)


def test_lp_pool():
    l = ONNXWriter()
    x = l.float32([1, 3, 4, 4])
    x = l.LpPool(x, kernel_shape=[2, 2], p=2, strides=[2, 2], pads=[0, 0, 0, 0])
    model = l.to_onnx()
    assert isinstance(model, onnx.ModelProto)
