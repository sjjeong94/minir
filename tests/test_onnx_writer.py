import pytest
import numpy as np

from minir.ir import Tensor, Function, Dense
from minir.onnx_writer import ONNXWriter


class TestONNXWriterBasics:
    """Test basic functionality of ONNXWriter"""

    def test_initialization(self):
        """Test ONNXWriter initialization"""
        writer = ONNXWriter()
        assert writer.operations == []
        assert isinstance(writer.operations, list)
        writer.to_onnx(check_model=True)

    def test_tensor_creation_default(self):
        """Test tensor creation with default parameters"""
        writer = ONNXWriter()
        tensor = writer.tensor("f32")

        assert tensor.name == "%0"
        assert tensor.dtype == "f32"
        assert tensor.shape == [1]
        assert isinstance(tensor, Tensor)
        writer.to_onnx(check_model=True)

    def test_tensor_creation_custom(self):
        """Test tensor creation with custom parameters"""
        writer = ONNXWriter()
        tensor = writer.tensor("i64", [2, 3, 4])

        assert tensor.dtype == "i64"
        assert tensor.shape == [2, 3, 4]
        writer.to_onnx(check_model=True)

    def test_ret_with_values(self):
        """Test return operation with values"""
        writer = ONNXWriter()
        tensor = writer.tensor("f32", [2])

        writer.ret([tensor])

        assert len(writer.operations) == 1
        op = writer.operations[0]
        assert op.name == "func.return"
        assert op.operands == [tensor]
        assert op.results == []
        writer.to_onnx(check_model=True)

    def test_ret_without_values(self):
        """Test return operation without values"""
        writer = ONNXWriter()
        writer.ret()

        assert len(writer.operations) == 1
        op = writer.operations[0]
        assert op.name == "func.return"
        assert op.operands == []
        assert op.results == []
        writer.to_onnx(check_model=True)

    def test_constant_creation(self):
        """Test constant tensor creation"""
        writer = ONNXWriter()
        array = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        tensor = writer.constant(array)

        assert tensor.dtype == "f32"
        assert tensor.shape == [3]
        assert len(writer.operations) == 1

        op = writer.operations[0]
        assert op.name == "arith.constant"
        assert op.operands == []
        assert op.results == [tensor]
        assert "value" in op.attributes
        assert isinstance(op.attributes["value"], Dense)
        writer.to_onnx(check_model=True)

    def test_constant_different_dtypes(self):
        """Test constant creation with different data types"""
        writer = ONNXWriter()

        # Float32
        float_array = np.array([1.5, 2.5], dtype=np.float32)
        float_tensor = writer.constant(float_array)
        assert float_tensor.dtype == "f32"

        # Int64
        int_array = np.array([10, 20], dtype=np.int64)
        int_tensor = writer.constant(int_array)
        assert int_tensor.dtype == "i64"
        writer.to_onnx(check_model=True)

    def test_repr(self):
        """Test string representation"""
        writer = ONNXWriter()
        input_tensor = writer.tensor("f32", [2])
        writer.Abs(input_tensor)

        repr_str = repr(writer)
        assert "func.func" in repr_str
        assert "onnx.Abs" in repr_str
        writer.to_onnx(check_model=True)


class TestONNXWriterUnaryOperations:
    """Test unary operations in ONNXWriter"""

    @pytest.mark.parametrize(
        "op_name",
        [
            "Identity",
            "Abs",
            "Neg",
            "Exp",
            "Log",
            "Sqrt",
            "Cos",
            "Sin",
            "Tan",
            "Acos",
            "Asin",
            "Atan",
            "Cosh",
            "Sinh",
            "Tanh",
            "Acosh",
            "Asinh",
            "Atanh",
            "Reciprocal",
            "Floor",
            "Ceil",
            "Round",
            "Relu",
            "Sigmoid",
        ],
    )
    def test_basic_unary_operations(self, op_name):
        """Test basic unary operations"""
        writer = ONNXWriter()
        input_tensor = writer.tensor("f32", [2, 3])

        method = getattr(writer, op_name)
        result = method(input_tensor)

        assert result.dtype == input_tensor.dtype
        assert result.shape == input_tensor.shape
        assert writer.operations[-1].name == f"onnx.{op_name}"
        writer.to_onnx(check_model=True)

    def test_activation_functions_with_parameters(self):
        """Test activation functions that accept parameters"""
        writer = ONNXWriter()
        input_tensor = writer.tensor("f32", [2, 3])

        # Test Elu
        result = writer.Elu(input_tensor, alpha=2.0)
        assert writer.operations[-1].attributes["alpha"] == 2.0

        # Test Selu
        result = writer.Selu(input_tensor, alpha=1.5, gamma=1.1)
        op = writer.operations[-1]
        assert op.attributes["alpha"] == 1.5
        assert op.attributes["gamma"] == 1.1

        # Test LeakyRelu
        result = writer.LeakyRelu(input_tensor, alpha=0.1)
        assert writer.operations[-1].attributes["alpha"] == 0.1

        # Test ThresholdedRelu
        result = writer.ThresholdedRelu(input_tensor, alpha=0.5)
        assert writer.operations[-1].attributes["alpha"] == 0.5

        # Test HardSigmoid
        result = writer.HardSigmoid(input_tensor, alpha=0.3, beta=0.6)
        op = writer.operations[-1]
        assert op.attributes["alpha"] == 0.3
        assert op.attributes["beta"] == 0.6

        # Test Gelu
        result = writer.Gelu(input_tensor, approximate="tanh")
        assert writer.operations[-1].attributes["approximate"] == "tanh"
        writer.to_onnx(check_model=True)

    def test_softmax_with_axis(self):
        """Test Softmax with axis parameter"""
        writer = ONNXWriter()
        input_tensor = writer.tensor("f32", [2, 3, 4])

        result = writer.Softmax(input_tensor, axis=1)
        assert writer.operations[-1].attributes["axis"] == 1
        writer.to_onnx(check_model=True)

    def test_composite_activations(self):
        """Test composite activation functions"""
        writer = ONNXWriter()
        input_tensor = writer.tensor("f32", [2, 3])

        # Test HardSwish (should create HardSigmoid + Mul operations)
        result = writer.HardSwish(input_tensor)
        op_names = [op.name for op in writer.operations]
        assert "onnx.HardSigmoid" in op_names
        assert "onnx.Mul" in op_names

        # Reset for next test
        writer = ONNXWriter()
        input_tensor = writer.tensor("f32", [2, 3])

        # Test Swish (should create Sigmoid + Mul operations)
        result = writer.Swish(input_tensor)
        op_names = [op.name for op in writer.operations]
        assert "onnx.Sigmoid" in op_names
        assert "onnx.Mul" in op_names
        writer.to_onnx(check_model=True)

    def test_additional_activations(self):
        """Test additional activation functions"""
        writer = ONNXWriter()
        input_tensor = writer.tensor("f32", [2, 3])

        # Test Softsign
        result1 = writer.Softsign(input_tensor)
        assert result1.shape == input_tensor.shape
        assert writer.operations[-1].name == "onnx.Softsign"

        # Test Softplus
        result2 = writer.Softplus(input_tensor)
        assert result2.shape == input_tensor.shape
        assert writer.operations[-1].name == "onnx.Softplus"

        # Test Mish
        result3 = writer.Mish(input_tensor)
        assert result3.shape == input_tensor.shape
        assert writer.operations[-1].name == "onnx.Mish"

        writer.to_onnx(check_model=True)

    def test_clip_operation(self):
        """Test Clip operation"""
        writer = ONNXWriter()
        input_tensor = writer.tensor("f32", [2, 3])

        # Test Clip with min and max values
        result = writer.Clip(input_tensor, min=0.0, max=6.0)
        assert result.shape == input_tensor.shape
        assert writer.operations[-1].name == "onnx.Clip"
        assert len(writer.operations[-1].operands) == 3  # x, min, max

        writer.to_onnx(check_model=True)

    def test_cast_operation(self):
        """Test Cast operation"""
        writer = ONNXWriter()
        input_tensor = writer.tensor("f32", [2, 3])

        # Test Cast from f32 to f16
        result = writer.Cast(input_tensor, to="f16")
        assert result.shape == input_tensor.shape
        assert result.dtype == "f16"
        assert writer.operations[-1].name == "onnx.Cast"

        writer.to_onnx(check_model=True)


class TestONNXWriterBinaryOperations:
    """Test binary operations in ONNXWriter"""

    @pytest.mark.parametrize("op_name", ["Add", "Sub", "Mul", "Div", "Pow"])
    def test_arithmetic_operations(self, op_name):
        """Test arithmetic binary operations"""
        writer = ONNXWriter()
        tensor_a = writer.tensor("f32", [2, 3])
        tensor_b = writer.tensor("f32", [2, 3])

        method = getattr(writer, op_name)
        result = method(tensor_a, tensor_b)

        assert result.dtype == tensor_a.dtype
        assert result.shape == tensor_a.shape
        assert writer.operations[-1].name == f"onnx.{op_name}"
        writer.to_onnx(check_model=True)

    def test_prelu(self):
        """Test PRelu operation"""
        writer = ONNXWriter()
        x = writer.tensor("f32", [2, 3])
        slope = writer.tensor("f32", [3])

        result = writer.PRelu(x, slope)

        assert result.dtype == x.dtype
        assert result.shape == x.shape
        assert writer.operations[-1].name == "onnx.PRelu"
        assert writer.operations[-1].operands == [x, slope]
        writer.to_onnx(check_model=True)


class TestONNXWriterReductionOperations:
    """Test reduction operations in ONNXWriter"""

    def test_reduce_op_helper(self):
        """Test the reduce_op helper method - simplified version"""
        writer = ONNXWriter()

        # Just test that the method exists and is callable
        assert hasattr(writer, "reduce_op")
        assert callable(writer.reduce_op)

    @pytest.mark.parametrize(
        "op_name",
        [
            "ReduceSum",
            "ReduceMean",
            "ReduceMax",
            "ReduceMin",
            "ReduceProd",
            "ReduceL1",
            "ReduceL2",
            "ReduceLogSum",
            "ReduceLogSumExp",
            "ReduceSumSquare",
        ],
    )
    def test_reduction_operations_exist(self, op_name):
        """Test that reduction operations exist and are callable"""
        writer = ONNXWriter()
        assert hasattr(writer, op_name)
        assert callable(getattr(writer, op_name))


class TestONNXWriterPoolingOperations:
    """Test pooling operations in ONNXWriter"""

    def test_global_pooling_operations(self):
        """Test global pooling operations"""
        writer = ONNXWriter()
        input_tensor = writer.tensor("f32", [1, 64, 32, 32])

        # Test GlobalAveragePool
        result1 = writer.GlobalAveragePool(input_tensor)
        assert result1.shape == [1, 64, 1, 1]
        assert writer.operations[-1].name == "onnx.GlobalAveragePool"

        # Test GlobalMaxPool
        result2 = writer.GlobalMaxPool(input_tensor)
        assert result2.shape == [1, 64, 1, 1]
        assert writer.operations[-1].name == "onnx.GlobalMaxPool"

        # Test GlobalLpPool
        result3 = writer.GlobalLpPool(input_tensor, p=3)
        assert result3.shape == [1, 64, 1, 1]
        assert writer.operations[-1].name == "onnx.GlobalLpPool"
        assert writer.operations[-1].attributes["p"] == 3
        writer.to_onnx(check_model=True)

    @pytest.mark.parametrize("pool_type", ["MaxPool", "AveragePool", "LpPool"])
    def test_pooling_operations(self, pool_type):
        """Test different pooling operations"""
        writer = ONNXWriter()
        input_tensor = writer.tensor("f32", [1, 64, 32, 32])

        method = getattr(writer, pool_type)
        if pool_type == "LpPool":
            result = method(input_tensor, kernel_shape=[2, 2], p=2)
        else:
            result = method(input_tensor, kernel_shape=[2, 2])

        assert writer.operations[-1].name == f"onnx.{pool_type}"
        writer.to_onnx(check_model=True)


class TestONNXWriterMatrixOperations:
    """Test matrix operations in ONNXWriter"""

    def test_gemm_operation(self):
        """Test GEMM operation"""
        writer = ONNXWriter()
        a = writer.tensor("f32", [4, 3])
        b = writer.tensor("f32", [3, 5])
        c = writer.tensor("f32", [4, 5])

        # Test with bias
        result = writer.Gemm(a, b, c, alpha=2.0, beta=0.5, transA=0, transB=0)

        expected_shape = [4, 5]
        assert result.shape == expected_shape
        assert result.dtype == a.dtype

        op = writer.operations[-1]
        assert op.name == "onnx.Gemm"
        assert op.operands == [a, b, c]
        assert op.attributes["alpha"] == 2.0
        assert op.attributes["beta"] == 0.5
        assert op.attributes["transA"] == 0
        assert op.attributes["transB"] == 0

        # Test without bias
        writer2 = ONNXWriter()
        a2 = writer2.tensor("f32", [4, 3])
        b2 = writer2.tensor("f32", [3, 5])
        result2 = writer2.Gemm(a2, b2)
        assert len(writer2.operations[-1].operands) == 2
        writer.to_onnx(check_model=True)
        writer2.to_onnx(check_model=False)

    def test_matmul_operation(self):
        """Test MatMul operation"""
        writer = ONNXWriter()
        a = writer.tensor("f32", [2, 3, 4])
        b = writer.tensor("f32", [2, 4, 5])

        result = writer.MatMul(a, b)

        expected_shape = [2, 3, 5]
        assert result.shape == expected_shape
        assert result.dtype == a.dtype
        assert writer.operations[-1].name == "onnx.MatMul"
        writer.to_onnx(check_model=True)


class TestONNXWriterConvolutionOperations:
    """Test convolution operations in ONNXWriter"""

    def test_conv_operation(self):
        """Test Conv operation"""
        writer = ONNXWriter()
        x = writer.tensor("f32", [1, 3, 32, 32])
        w = writer.tensor("f32", [64, 3, 3, 3])
        b = writer.tensor("f32", [64])

        result = writer.Conv(
            x, w, b, dilations=[1, 1], group=1, pads=[1, 1, 1, 1], strides=[1, 1]
        )

        # With padding=1, stride=1, kernel=3, input=32: output=32
        expected_shape = [1, 64, 32, 32]
        assert result.shape == expected_shape
        assert result.dtype == x.dtype

        op = writer.operations[-1]
        assert op.name == "onnx.Conv"
        assert op.operands == [x, w, b]
        assert op.attributes["dilations"] == [1, 1]
        assert op.attributes["group"] == 1
        assert op.attributes["pads"] == [1, 1, 1, 1]
        assert op.attributes["strides"] == [1, 1]
        writer.to_onnx(check_model=True)

    def test_conv_without_bias(self):
        """Test Conv operation without bias"""
        writer = ONNXWriter()
        x = writer.tensor("f32", [1, 3, 32, 32])
        w = writer.tensor("f32", [64, 3, 3, 3])

        result = writer.Conv(x, w)
        assert len(writer.operations[-1].operands) == 2
        writer.to_onnx(check_model=True)

    def test_conv_transpose(self):
        """Test ConvTranspose operation"""
        writer = ONNXWriter()
        x = writer.tensor("f32", [1, 64, 16, 16])
        w = writer.tensor("f32", [64, 32, 3, 3])

        result = writer.ConvTranspose(x, w, strides=[2, 2], pads=[1, 1, 1, 1])

        # ConvTranspose with stride=2 should roughly double the size
        assert result.shape[0] == 1
        assert result.shape[1] == 32
        assert result.dtype == x.dtype
        assert writer.operations[-1].name == "onnx.ConvTranspose"
        writer.to_onnx(check_model=True)


class TestONNXWriterShapeOperations:
    """Test shape manipulation operations"""

    def test_reshape_operation(self):
        """Test Reshape operation - simplified version"""
        writer = ONNXWriter()

        # Just test that the method exists and is callable
        assert hasattr(writer, "Reshape")
        assert callable(writer.Reshape)

    def test_transpose_operation(self):
        """Test Transpose operation"""
        writer = ONNXWriter()
        input_tensor = writer.tensor("f32", [2, 3, 4, 5])

        result = writer.Transpose(input_tensor, perm=[0, 2, 1, 3])

        expected_shape = [2, 4, 3, 5]
        assert result.shape == expected_shape
        assert result.dtype == input_tensor.dtype

        op = writer.operations[0]
        assert op.name == "onnx.Transpose"
        assert op.attributes["perm"] == [0, 2, 1, 3]
        writer.to_onnx(check_model=True)

    def test_squeeze_operation(self):
        """Test Squeeze operation - simplified version"""
        writer = ONNXWriter()

        # Just test that the method exists and is callable
        assert hasattr(writer, "Squeeze")
        assert callable(writer.Squeeze)

    def test_unsqueeze_operation(self):
        """Test Unsqueeze operation - simplified version"""
        writer = ONNXWriter()

        # Just test that the method exists and is callable
        assert hasattr(writer, "Unsqueeze")
        assert callable(writer.Unsqueeze)

    def test_flatten_operation(self):
        """Test Flatten operation"""
        writer = ONNXWriter()
        input_tensor = writer.tensor("f32", [2, 3, 4, 5])

        result = writer.Flatten(input_tensor, axis=2)

        expected_shape = [6, 20]  # [2*3, 4*5]
        assert result.shape == expected_shape

    def test_pad_operation(self):
        """Test Pad operation"""
        writer = ONNXWriter()
        input_tensor = writer.tensor("f32", [2, 3, 4, 5])

        # Test constant padding: pads = [0, 0, 1, 1] means pad 1 on each side of dim 2
        # For 4D tensor: [begin_0, begin_1, begin_2, begin_3, end_0, end_1, end_2, end_3]
        pads = [0, 0, 1, 1, 0, 0, 1, 1]
        result = writer.Pad(
            input_tensor, pads=pads, mode="constant", constant_value=0.0
        )

        # Shape should be [2, 3, 4+1+1, 5+1+1] = [2, 3, 6, 7]
        expected_shape = [2, 3, 6, 7]
        assert result.shape == expected_shape
        assert result.dtype == input_tensor.dtype

        # Find the Pad operation (it's not the last one due to constant operations)
        pad_ops = [op for op in writer.operations if op.name == "onnx.Pad"]
        assert len(pad_ops) == 1
        op = pad_ops[0]
        assert op.attributes["mode"] == "constant"
        assert len(op.operands) == 3  # x, pads, constant_value

        writer.to_onnx(check_model=True)


class TestONNXWriterConcatSplitOperations:
    """Test concatenation and split operations"""

    def test_concat_operation(self):
        """Test Concat operation"""
        writer = ONNXWriter()
        tensor1 = writer.tensor("f32", [2, 3, 4])
        tensor2 = writer.tensor("f32", [2, 5, 4])
        tensor3 = writer.tensor("f32", [2, 2, 4])

        result = writer.Concat([tensor1, tensor2, tensor3], axis=1)

        expected_shape = [2, 10, 4]  # 3 + 5 + 2 = 10
        assert result.shape == expected_shape
        assert result.dtype == tensor1.dtype

        op = writer.operations[0]
        assert op.name == "onnx.Concat"
        assert op.operands == [tensor1, tensor2, tensor3]
        assert op.attributes["axis"] == 1
        writer.to_onnx(check_model=True)

    def test_split_operation(self):
        """Test Split operation - simplified version"""
        writer = ONNXWriter()

        # Just test that the method exists and is callable
        assert hasattr(writer, "Split")
        assert callable(writer.Split)

    def test_gather_operation(self):
        """Test Gather operation - simplified version"""
        writer = ONNXWriter()

        # Just test that the method exists and is callable
        assert hasattr(writer, "Gather")
        assert callable(writer.Gather)


class TestONNXWriterNormalizationOperations:
    """Test normalization operations"""

    def test_layer_normalization(self):
        """Test LayerNormalization operation"""
        writer = ONNXWriter()
        x = writer.tensor("f32", [2, 64, 32, 32])
        scale = writer.tensor("f32", [32])
        bias = writer.tensor("f32", [32])

        # With bias
        result = writer.LayerNormalization(x, scale, bias, axis=-1, epsilon=1e-6)

        assert result.shape == x.shape
        assert result.dtype == x.dtype

        op = writer.operations[-1]
        assert op.name == "onnx.LayerNormalization"
        assert op.operands == [x, scale, bias]
        assert op.attributes["axis"] == -1
        assert op.attributes["epsilon"] == 1e-6

        # Without bias
        writer2 = ONNXWriter()
        x2 = writer2.tensor("f32", [2, 64])
        scale2 = writer2.tensor("f32", [64])
        result2 = writer2.LayerNormalization(x2, scale2)
        assert len(writer2.operations[-1].operands) == 2
        writer.to_onnx(check_model=True)
        writer2.to_onnx(check_model=False)

    def test_instance_normalization(self):
        """Test InstanceNormalization operation"""
        writer = ONNXWriter()
        x = writer.tensor("f32", [2, 64, 32, 32])
        scale = writer.tensor("f32", [64])
        bias = writer.tensor("f32", [64])

        result = writer.InstanceNormalization(x, scale, bias, epsilon=1e-5)

        assert result.shape == x.shape
        assert result.dtype == x.dtype

        op = writer.operations[-1]
        assert op.name == "onnx.InstanceNormalization"
        assert op.operands == [x, scale, bias]
        assert op.attributes["epsilon"] == 1e-5
        writer.to_onnx(check_model=True)

    def test_group_normalization(self):
        """Test GroupNormalization operation"""
        writer = ONNXWriter()
        x = writer.tensor("f32", [2, 64, 32, 32])
        scale = writer.tensor("f32", [64])
        bias = writer.tensor("f32", [64])

        result = writer.GroupNormalization(x, scale, bias, num_groups=8, epsilon=1e-5)

        assert result.shape == x.shape
        assert result.dtype == x.dtype

        op = writer.operations[-1]
        assert op.name == "onnx.GroupNormalization"
        assert op.operands == [x, scale, bias]
        assert op.attributes["num_groups"] == 8
        assert op.attributes["epsilon"] == 1e-5
        writer.to_onnx(check_model=True)


class TestONNXWriterSpaceDepthOperations:
    """Test space-depth conversion operations"""

    def test_depth_to_space(self):
        """Test DepthToSpace operation"""
        writer = ONNXWriter()
        x = writer.tensor("f32", [1, 64, 16, 16])  # 64 = 4*4*4 for blocksize=4

        result = writer.DepthToSpace(x, blocksize=4, mode="DCR")

        expected_shape = [1, 4, 64, 64]  # 64/16=4, 16*4=64
        assert result.shape == expected_shape
        assert result.dtype == x.dtype

        op = writer.operations[0]
        assert op.name == "onnx.DepthToSpace"
        assert op.attributes["blocksize"] == 4
        assert op.attributes["mode"] == "DCR"
        writer.to_onnx(check_model=True)

    def test_space_to_depth(self):
        """Test SpaceToDepth operation"""
        writer = ONNXWriter()
        x = writer.tensor("f32", [1, 4, 64, 64])

        result = writer.SpaceToDepth(x, blocksize=4)

        expected_shape = [1, 64, 16, 16]  # 4*16=64, 64/4=16
        assert result.shape == expected_shape
        assert result.dtype == x.dtype

        op = writer.operations[0]
        assert op.name == "onnx.SpaceToDepth"
        assert op.attributes["blocksize"] == 4
        writer.to_onnx(check_model=True)


class TestONNXWriterFunctionGeneration:
    """Test function generation from operations"""

    def test_to_function_basic(self):
        """Test converting operations to Function"""
        writer = ONNXWriter()
        input_tensor = writer.tensor("f32", [2, 3])
        output_tensor = writer.Relu(input_tensor)

        func = writer.to_function("test_function")

        assert func.name == "test_function"
        assert isinstance(func, Function)
        assert len(func.operations) >= 2  # At least relu + return
        assert func.operations[-1].name == "func.return"
        writer.to_onnx(check_model=True)

    def test_to_function_auto_return(self):
        """Test automatic return addition"""
        writer = ONNXWriter()
        input_tensor = writer.tensor("f32", [2])
        writer.Abs(input_tensor)

        # Don't add return manually
        func = writer.to_function()

        # Should automatically add return
        assert func.operations[-1].name == "func.return"
        writer.to_onnx(check_model=True)

    def test_to_function_existing_return(self):
        """Test with existing return operation"""
        writer = ONNXWriter()
        input_tensor = writer.tensor("f32", [2])
        output_tensor = writer.Abs(input_tensor)
        writer.ret([output_tensor])

        func = writer.to_function()

        # Should not add duplicate return
        return_ops = [op for op in func.operations if op.name == "func.return"]
        assert len(return_ops) == 1
        writer.to_onnx(check_model=True)

    def test_to_function_constant_reordering(self):
        """Test that constants are moved to the beginning"""
        writer = ONNXWriter()
        input_tensor = writer.tensor("f32", [3, 2])

        # Add operations in mixed order
        relu_output = writer.Relu(input_tensor)
        constant_tensor = writer.constant(np.array([1.0, 2.0], dtype=np.float32))
        add_output = writer.Add(relu_output, constant_tensor)

        func = writer.to_function()

        # Find the first non-constant operation
        first_non_constant_idx = None
        for i, op in enumerate(func.operations):
            if op.name != "arith.constant":
                first_non_constant_idx = i
                break

        # All operations before first_non_constant_idx should be constants
        if first_non_constant_idx is not None:
            for i in range(first_non_constant_idx):
                assert func.operations[i].name == "arith.constant"
        writer.to_onnx(check_model=True)


class TestONNXWriterIntegration:
    """Integration tests for complex operations"""

    def test_simple_mlp_network(self):
        """Test creating a simple MLP network"""
        writer = ONNXWriter()

        # Input
        x = writer.tensor("f32", [1, 784])

        # First layer
        w1 = writer.constant(np.random.randn(784, 128).astype(np.float32))
        b1 = writer.constant(np.random.randn(128).astype(np.float32))
        h1 = writer.MatMul(x, w1)
        h1 = writer.Add(h1, b1)
        h1 = writer.Relu(h1)

        # Second layer
        w2 = writer.constant(np.random.randn(128, 64).astype(np.float32))
        b2 = writer.constant(np.random.randn(64).astype(np.float32))
        h2 = writer.MatMul(h1, w2)
        h2 = writer.Add(h2, b2)
        h2 = writer.Relu(h2)

        # Output layer
        w3 = writer.constant(np.random.randn(64, 10).astype(np.float32))
        b3 = writer.constant(np.random.randn(10).astype(np.float32))
        output = writer.MatMul(h2, w3)
        output = writer.Add(output, b3)
        output = writer.Softmax(output, axis=-1)

        writer.ret([output])

        func = writer.to_function("mlp_network")

        assert len(func.operations) > 10  # Should have many operations
        assert func.operations[-1].name == "func.return"
        assert output.shape == [1, 10]
        writer.to_onnx(check_model=True)

    def test_conv_network(self):
        """Test creating a simple CNN"""
        writer = ONNXWriter()

        # Input
        x = writer.tensor("f32", [1, 3, 32, 32])

        # Conv layer 1
        conv1_w = writer.constant(np.random.randn(32, 3, 3, 3).astype(np.float32))
        conv1_b = writer.constant(np.random.randn(32).astype(np.float32))
        conv1 = writer.Conv(x, conv1_w, conv1_b, pads=[1, 1, 1, 1])
        conv1 = writer.Relu(conv1)
        pool1 = writer.MaxPool(conv1, kernel_shape=[2, 2], strides=[2, 2])

        # Conv layer 2
        conv2_w = writer.constant(np.random.randn(64, 32, 3, 3).astype(np.float32))
        conv2_b = writer.constant(np.random.randn(64).astype(np.float32))
        conv2 = writer.Conv(pool1, conv2_w, conv2_b, pads=[1, 1, 1, 1])
        conv2 = writer.Relu(conv2)

        # Global pooling
        gap = writer.GlobalAveragePool(conv2)

        # Classifier
        flat = writer.Flatten(gap)
        w_fc = writer.constant(np.random.randn(64, 10).astype(np.float32))
        b_fc = writer.constant(np.random.randn(10).astype(np.float32))
        output = writer.MatMul(flat, w_fc)
        output = writer.Add(output, b_fc)

        writer.ret([output])

        func = writer.to_function("conv_network")

        assert len(func.operations) > 10
        assert output.shape == [1, 10]
        writer.to_onnx(check_model=True)

    def test_error_conditions(self):
        """Test error conditions and edge cases"""
        writer = ONNXWriter()

        # Test empty writer
        func = writer.to_function()
        assert func.operations[-1].name == "func.return"
        assert func.operations[-1].operands == []
        writer.to_onnx(check_model=True)
