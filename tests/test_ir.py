import pytest


from minir.ir import Value, Scalar, Vector, Tensor, Operation, Function


class TestValue:
    """Test cases for the Value class"""

    def test_value_initialization(self):
        """Test Value instance creation"""
        value = Value("test_value")
        assert value.name == "test_value"
        assert value.owner is None
        assert value.users == []

    def test_value_attributes(self):
        """Test Value attributes can be modified"""
        value = Value("test")
        value.name = "modified"
        assert value.name == "modified"


class TestScalar:
    """Test cases for the Scalar class"""

    def test_scalar_initialization(self):
        """Test Scalar instance creation"""
        scalar = Scalar("scalar_val", "f32")
        assert scalar.name == "scalar_val"
        assert scalar.dtype == "f32"
        assert scalar.owner is None
        assert scalar.users == []

    def test_scalar_repr(self):
        """Test Scalar string representation"""
        scalar = Scalar("test", "i32")
        assert repr(scalar) == "i32"

        scalar_float = Scalar("test", "f64")
        assert repr(scalar_float) == "f64"


class TestVector:
    """Test cases for the Vector class"""

    def test_vector_default_initialization(self):
        """Test Vector with default shape"""
        vector = Vector("vec", "f32")
        assert vector.name == "vec"
        assert vector.dtype == "f32"
        assert vector.shape == [1]

    def test_vector_custom_shape(self):
        """Test Vector with custom shape"""
        vector = Vector("vec", "i32", [3, 4, 5])
        assert vector.shape == [3, 4, 5]

    def test_vector_repr(self):
        """Test Vector string representation"""
        vector = Vector("test", "f32", [2, 3])
        assert repr(vector) == "vector<2x3xf32>"

        vector_1d = Vector("test", "i64", [10])
        assert repr(vector_1d) == "vector<10xi64>"


class TestTensor:
    """Test cases for the Tensor class"""

    def test_tensor_default_initialization(self):
        """Test Tensor with default parameters"""
        tensor = Tensor("tensor", "f32")
        assert tensor.name == "tensor"
        assert tensor.dtype == "f32"
        assert tensor.shape == [1]
        assert tensor.encoding is None

    def test_tensor_with_encoding(self):
        """Test Tensor with encoding"""
        encoding = {"sparse": True}
        tensor = Tensor("sparse_tensor", "f64", [2, 3], encoding)
        assert tensor.encoding == encoding

    def test_tensor_repr_without_encoding(self):
        """Test Tensor string representation without encoding"""
        tensor = Tensor("test", "f32", [2, 3, 4])
        assert repr(tensor) == "tensor<2x3x4xf32>"

    def test_tensor_repr_with_encoding(self):
        """Test Tensor string representation with encoding"""
        encoding = {"type": "sparse"}
        tensor = Tensor("test", "f32", [2, 3], encoding)
        expected = "tensor<2x3xf32, {'type': 'sparse'}>"
        assert repr(tensor) == expected


class TestOperation:
    """Test cases for the Operation class"""

    def test_operation_basic_initialization(self):
        """Test Operation with minimal parameters"""
        op = Operation("add")
        assert op.name == "add"
        assert op.operands == []
        assert op.results == []
        assert op.attributes == {}

    def test_operation_full_initialization(self):
        """Test Operation with all parameters"""
        operand1 = Tensor("input1", "f32", [2, 3])
        operand2 = Tensor("input2", "f32", [2, 3])
        result = Tensor("output", "f32", [2, 3])
        attrs = {"axis": 0, "keepdims": True}

        op = Operation("add", [operand1, operand2], [result], attrs)
        assert op.name == "add"
        assert len(op.operands) == 2
        assert len(op.results) == 1
        assert op.attributes == attrs

    def test_operation_invalid_name_type(self):
        """Test Operation with invalid name type"""
        with pytest.raises(TypeError, match="name must be str"):
            Operation(123)

    def test_operation_invalid_operands_type(self):
        """Test Operation with invalid operands"""
        with pytest.raises(TypeError, match="all elements of operands must be Value"):
            Operation("test", ["not_a_value"])

    def test_operation_invalid_results_type(self):
        """Test Operation with invalid results"""
        with pytest.raises(TypeError, match="all elements of results must be Value"):
            Operation("test", None, ["not_a_value"])

    def test_operation_invalid_attributes_type(self):
        """Test Operation with invalid attributes type"""
        with pytest.raises(TypeError, match="attributes must be dict"):
            Operation("test", None, None, "not_a_dict")

    def test_operation_repr_simple(self):
        """Test Operation string representation for simple case"""
        op = Operation("test_op")
        assert repr(op) == '"test_op"() : () -> ()'

    def test_operation_repr_with_operands_and_results(self):
        """Test Operation string representation with operands and results"""
        operand = Tensor("input", "f32", [2])
        result = Tensor("output", "f32", [2])
        op = Operation("relu", [operand], [result])

        expected = 'output = "relu"(input) : (tensor<2xf32>) -> (tensor<2xf32>)'
        assert repr(op) == expected

    def test_operation_repr_with_attributes(self):
        """Test Operation string representation with attributes"""
        operand = Scalar("x", "f32")
        result = Scalar("y", "f32")
        attrs = {"alpha": 0.5, "beta": 1, "mode": "linear"}
        op = Operation("custom", [operand], [result], attrs)

        result_str = repr(op)
        assert '"custom"(x)' in result_str
        assert "alpha = 0.5 : f32" in result_str
        assert "beta = 1 : i32" in result_str
        assert 'mode = "linear"' in result_str

    def test_operation_repr_with_bytes_attribute(self):
        """Test Operation string representation with bytes attribute"""
        operand = Scalar("x", "f32")
        result = Scalar("y", "f32")
        attrs = {"weights": b"\x01\x02\x03\x04"}
        op = Operation("const", [operand], [result], attrs)

        result_str = repr(op)
        assert 'dense<"0x01020304">' in result_str


class TestFunction:
    """Test cases for the Function class"""

    def test_function_basic_initialization(self):
        """Test Function with simple operations"""
        arg = Tensor("arg0", "f32", [2])
        result = Tensor("result", "f32", [2])
        return_op = Operation("func.return", [result])

        func = Function([return_op])
        assert func.name == "function"
        assert len(func.operations) == 1

    def test_function_custom_name(self):
        """Test Function with custom name"""
        return_op = Operation("func.return", [])
        func = Function([return_op], "my_function")
        assert func.name == "my_function"

    def test_function_arguments_property(self):
        """Test Function arguments property"""
        arg1 = Tensor("input1", "f32", [2])
        arg2 = Tensor("input2", "f32", [2])
        result = Tensor("output", "f32", [2])

        add_op = Operation("add", [arg1, arg2], [result])
        return_op = Operation("func.return", [result])

        func = Function([add_op, return_op])

        # Arguments should be operands with no owner
        assert len(func.arguments) == 2
        assert all(arg.owner is None for arg in func.arguments)

    def test_function_results_property(self):
        """Test Function results property"""
        arg = Tensor("input", "f32", [2])
        result = Tensor("output", "f32", [2])

        relu_op = Operation("relu", [arg], [result])
        return_op = Operation("func.return", [result])

        func = Function([relu_op, return_op])

        # Results should come from return operation
        assert len(func.results) == 1
        assert func.results[0] == result

    def test_function_results_property_no_return(self):
        """Test Function results property when no return operation"""
        arg = Tensor("input", "f32", [2])
        result = Tensor("output", "f32", [2])

        relu_op = Operation("relu", [arg], [result])
        func = Function([relu_op])

        # No return operation, so no results
        assert func.results == []

    def test_function_local_values_property(self):
        """Test Function local_values property"""
        arg = Tensor("input", "f32", [2])
        intermediate = Tensor("temp", "f32", [2])
        result = Tensor("output", "f32", [2])

        op1 = Operation("relu", [arg], [intermediate])
        op2 = Operation("sigmoid", [intermediate], [result])
        return_op = Operation("func.return", [result])

        func = Function([op1, op2, return_op])

        # Local values should not include arguments or final results
        assert len(func.local_values) == 1
        assert intermediate in func.local_values

    def test_function_values_property(self):
        """Test Function values property (all values)"""
        arg = Tensor("input", "f32", [2])
        intermediate = Tensor("temp", "f32", [2])
        result = Tensor("output", "f32", [2])

        op1 = Operation("relu", [arg], [intermediate])
        op2 = Operation("sigmoid", [intermediate], [result])
        return_op = Operation("func.return", [result])

        func = Function([op1, op2, return_op])

        # All values: arguments + results + local_values
        all_values = func.values
        assert len(all_values) == 3
        assert arg in all_values
        assert intermediate in all_values
        assert result in all_values

    def test_function_set_owner_and_users(self):
        """Test Function properly sets owner and users relationships"""
        arg = Tensor("input", "f32", [2])
        result = Tensor("output", "f32", [2])

        relu_op = Operation("relu", [arg], [result])
        return_op = Operation("func.return", [result])

        func = Function([relu_op, return_op])

        # Check owners
        assert arg.owner is None  # Argument has no owner
        assert result.owner == relu_op  # Result owned by relu_op

        # Check users
        assert relu_op in arg.users  # arg is used by relu_op
        assert return_op in result.users  # result is used by return_op

    def test_function_rename_values(self):
        """Test Function renames values with %index format"""
        arg = Tensor("original_input", "f32", [2])
        result = Tensor("original_output", "f32", [2])

        relu_op = Operation("relu", [arg], [result])
        return_op = Operation("func.return", [result])

        func = Function([relu_op, return_op])

        # Values should be renamed to %0, %1, etc.
        value_names = [v.name for v in func.values]
        assert "%0" in value_names
        assert "%1" in value_names

    def test_function_repr(self):
        """Test Function string representation"""
        arg = Tensor("input", "f32", [2])
        result = Tensor("output", "f32", [2])

        relu_op = Operation("relu", [arg], [result])
        return_op = Operation("func.return", [result])

        func = Function([relu_op, return_op], "test_func")

        func_str = repr(func)
        assert "func.func @test_func" in func_str
        assert "tensor<2xf32>" in func_str
        assert "relu" in func_str
        assert "func.return" in func_str
        assert func_str.startswith("func.func")
        assert func_str.endswith("}")


class TestIntegration:
    """Integration tests for multiple classes working together"""

    def test_complex_function_creation(self):
        """Test creating a complex function with multiple operations"""
        # Create inputs
        input1 = Tensor("x", "f32", [4, 4])
        input2 = Tensor("y", "f32", [4, 4])

        # Create intermediate results
        sum_result = Tensor("sum", "f32", [4, 4])
        relu_result = Tensor("activated", "f32", [4, 4])

        # Create operations
        add_op = Operation("arith.addf", [input1, input2], [sum_result])
        relu_op = Operation("math.relu", [sum_result], [relu_result])
        return_op = Operation("func.return", [relu_result])

        # Create function
        func = Function([add_op, relu_op, return_op], "add_and_relu")

        # Verify the function structure
        assert len(func.arguments) == 2
        assert len(func.results) == 1
        assert len(func.operations) == 3

        # Verify relationships
        assert input1.owner is None
        assert input2.owner is None
        assert sum_result.owner == add_op
        assert relu_result.owner == relu_op

        # Verify users
        assert add_op in input1.users
        assert add_op in input2.users
        assert relu_op in sum_result.users
        assert return_op in relu_result.users

    def test_function_with_attributes(self):
        """Test function with operations that have attributes"""
        input_tensor = Tensor("input", "f32", [1, 3, 32, 32])
        output_tensor = Tensor("output", "f32", [1, 16, 30, 30])

        conv_attrs = {"kernel_shape": [3, 3], "strides": [1, 1], "padding": "valid"}

        conv_op = Operation("conv2d", [input_tensor], [output_tensor], conv_attrs)
        return_op = Operation("func.return", [output_tensor])

        func = Function([conv_op, return_op], "conv_function")

        # Verify attributes are preserved
        assert func.operations[0].attributes == conv_attrs

        # Verify function representation includes attributes
        func_str = repr(func)
        assert "conv2d" in func_str
        assert "kernel_shape" in func_str
