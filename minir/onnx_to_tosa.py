from minir import Function, Tensor, Operation, Array, Int64
from minir.utils import dense_to_numpy, numpy_to_dense
from minir import OpRewritePattern, RewritePatternSet, GreedyPatternRewriter
import numpy as np


# =============================================================================
# Reshape / Transpose / MatMul Patterns
# =============================================================================


class ReshapeToTOSAPattern(OpRewritePattern):
    def match(self, op):
        return op.name == "onnx.Reshape"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]
        shape_operand = op.operands[1].owner.attributes["value"]

        shape_value = dense_to_numpy(shape_operand)
        if shape_value is None:
            return False
        new_shape = shape_value.tolist()

        new_op = Operation(
            "tosa.reshape",
            operands=[x],
            results=[y],
            attributes={"new_shape": Array(new_shape)},
        )

        rewriter.replace_op(op, [new_op])
        return True


class TransposeToTOSAPattern(OpRewritePattern):
    def __init__(self):
        super().__init__(benefit=10)

    def match(self, op):
        return op.name == "onnx.Transpose"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]
        perm = op.attributes.get("perm", list(range(len(x.shape) - 1, -1, -1)))

        perm_tensor = Tensor("", dtype="i32", shape=[len(perm)])
        perm_op = Operation(
            "arith.constant",
            operands=[],
            results=[perm_tensor],
            attributes={"value": numpy_to_dense(np.int32(perm))},
        )

        new_op = Operation(
            name="tosa.transpose",
            operands=[x, perm_tensor],
            results=[y],
        )

        rewriter.replace_op(op, [new_op])
        rewriter.insert_op_before(op, perm_op)
        return True


class MatMulToTOSAPattern(OpRewritePattern):
    def match(self, op):
        return op.name == "onnx.MatMul"

    def rewrite(self, op, rewriter):
        a = op.operands[0]
        b = op.operands[1]
        y = op.results[0]

        new_op = Operation(
            name="tosa.matmul",
            operands=[a, b],
            results=[y],
        )

        rewriter.replace_op(op, [new_op])
        return True


# =============================================================================
# Unary Element-wise Patterns
# =============================================================================


class AbsToTOSAPattern(OpRewritePattern):
    """onnx.Abs -> tosa.abs"""

    def match(self, op):
        return op.name == "onnx.Abs"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]
        new_op = Operation("tosa.abs", operands=[x], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class NegToTOSAPattern(OpRewritePattern):
    """onnx.Neg -> tosa.negate"""

    def match(self, op):
        return op.name == "onnx.Neg"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]
        new_op = Operation("tosa.negate", operands=[x], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class ExpToTOSAPattern(OpRewritePattern):
    """onnx.Exp -> tosa.exp"""

    def match(self, op):
        return op.name == "onnx.Exp"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]
        new_op = Operation("tosa.exp", operands=[x], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class LogToTOSAPattern(OpRewritePattern):
    """onnx.Log -> tosa.log"""

    def match(self, op):
        return op.name == "onnx.Log"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]
        new_op = Operation("tosa.log", operands=[x], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class ReciprocalToTOSAPattern(OpRewritePattern):
    """onnx.Reciprocal -> tosa.reciprocal"""

    def match(self, op):
        return op.name == "onnx.Reciprocal"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]
        new_op = Operation("tosa.reciprocal", operands=[x], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class FloorToTOSAPattern(OpRewritePattern):
    """onnx.Floor -> tosa.floor"""

    def match(self, op):
        return op.name == "onnx.Floor"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]
        new_op = Operation("tosa.floor", operands=[x], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class CeilToTOSAPattern(OpRewritePattern):
    """onnx.Ceil -> tosa.ceil"""

    def match(self, op):
        return op.name == "onnx.Ceil"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]
        new_op = Operation("tosa.ceil", operands=[x], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class TanhToTOSAPattern(OpRewritePattern):
    """onnx.Tanh -> tosa.tanh"""

    def match(self, op):
        return op.name == "onnx.Tanh"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]
        new_op = Operation("tosa.tanh", operands=[x], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class SigmoidToTOSAPattern(OpRewritePattern):
    """onnx.Sigmoid -> tosa.sigmoid"""

    def match(self, op):
        return op.name == "onnx.Sigmoid"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]
        new_op = Operation("tosa.sigmoid", operands=[x], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class SqrtToTOSAPattern(OpRewritePattern):
    """onnx.Sqrt -> tosa.rsqrt + tosa.reciprocal (1/rsqrt(x) = sqrt(x))"""

    def match(self, op):
        return op.name == "onnx.Sqrt"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]
        # sqrt(x) = 1 / rsqrt(x) = reciprocal(rsqrt(x))
        rsqrt_result = Tensor("", dtype=x.dtype, shape=x.shape)
        rsqrt_op = Operation("tosa.rsqrt", operands=[x], results=[rsqrt_result])
        recip_op = Operation("tosa.reciprocal", operands=[rsqrt_result], results=[y])
        # Insert in reverse order since they go to the same position and are sorted in reverse
        rewriter.insert_op_before(op, recip_op)
        rewriter.insert_op_before(op, rsqrt_op)
        rewriter.erase_op(op)
        return True


# =============================================================================
# Binary Element-wise Patterns
# =============================================================================


class AddToTOSAPattern(OpRewritePattern):
    """onnx.Add -> tosa.add"""

    def match(self, op):
        return op.name == "onnx.Add"

    def rewrite(self, op, rewriter):
        a = op.operands[0]
        b = op.operands[1]
        y = op.results[0]
        new_op = Operation("tosa.add", operands=[a, b], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class SubToTOSAPattern(OpRewritePattern):
    """onnx.Sub -> tosa.sub"""

    def match(self, op):
        return op.name == "onnx.Sub"

    def rewrite(self, op, rewriter):
        a = op.operands[0]
        b = op.operands[1]
        y = op.results[0]
        new_op = Operation("tosa.sub", operands=[a, b], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class MulToTOSAPattern(OpRewritePattern):
    """onnx.Mul -> tosa.mul"""

    def match(self, op):
        return op.name == "onnx.Mul"

    def rewrite(self, op, rewriter):
        a = op.operands[0]
        b = op.operands[1]
        y = op.results[0]
        new_op = Operation(
            "tosa.mul", operands=[a, b], results=[y], attributes={"shift": 0}
        )
        rewriter.replace_op(op, [new_op])
        return True


class DivToTOSAPattern(OpRewritePattern):
    """onnx.Div -> tosa.reciprocal + tosa.mul (a / b = a * (1/b))"""

    def match(self, op):
        return op.name == "onnx.Div"

    def rewrite(self, op, rewriter):
        a = op.operands[0]
        b = op.operands[1]
        y = op.results[0]
        # a / b = a * reciprocal(b)
        recip_result = Tensor("", dtype=b.dtype, shape=b.shape)
        recip_op = Operation("tosa.reciprocal", operands=[b], results=[recip_result])
        mul_op = Operation(
            "tosa.mul",
            operands=[a, recip_result],
            results=[y],
            attributes={"shift": 0},
        )
        # Insert in reverse order since they go to the same position and are sorted in reverse
        rewriter.insert_op_before(op, mul_op)
        rewriter.insert_op_before(op, recip_op)
        rewriter.erase_op(op)
        return True


class PowToTOSAPattern(OpRewritePattern):
    """onnx.Pow -> tosa.pow"""

    def match(self, op):
        return op.name == "onnx.Pow"

    def rewrite(self, op, rewriter):
        a = op.operands[0]
        b = op.operands[1]
        y = op.results[0]
        new_op = Operation("tosa.pow", operands=[a, b], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


# =============================================================================
# Comparison Patterns
# =============================================================================


class EqualToTOSAPattern(OpRewritePattern):
    """onnx.Equal -> tosa.equal"""

    def match(self, op):
        return op.name == "onnx.Equal"

    def rewrite(self, op, rewriter):
        a = op.operands[0]
        b = op.operands[1]
        y = op.results[0]
        new_op = Operation("tosa.equal", operands=[a, b], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class GreaterToTOSAPattern(OpRewritePattern):
    """onnx.Greater -> tosa.greater"""

    def match(self, op):
        return op.name == "onnx.Greater"

    def rewrite(self, op, rewriter):
        a = op.operands[0]
        b = op.operands[1]
        y = op.results[0]
        new_op = Operation("tosa.greater", operands=[a, b], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class GreaterOrEqualToTOSAPattern(OpRewritePattern):
    """onnx.GreaterOrEqual -> tosa.greater_equal"""

    def match(self, op):
        return op.name == "onnx.GreaterOrEqual"

    def rewrite(self, op, rewriter):
        a = op.operands[0]
        b = op.operands[1]
        y = op.results[0]
        new_op = Operation("tosa.greater_equal", operands=[a, b], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class LessToTOSAPattern(OpRewritePattern):
    """onnx.Less -> tosa.greater (with swapped operands)"""

    def match(self, op):
        return op.name == "onnx.Less"

    def rewrite(self, op, rewriter):
        a = op.operands[0]
        b = op.operands[1]
        y = op.results[0]
        # a < b is equivalent to b > a
        new_op = Operation("tosa.greater", operands=[b, a], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class LessOrEqualToTOSAPattern(OpRewritePattern):
    """onnx.LessOrEqual -> tosa.greater_equal (with swapped operands)"""

    def match(self, op):
        return op.name == "onnx.LessOrEqual"

    def rewrite(self, op, rewriter):
        a = op.operands[0]
        b = op.operands[1]
        y = op.results[0]
        # a <= b is equivalent to b >= a
        new_op = Operation("tosa.greater_equal", operands=[b, a], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


# =============================================================================
# Logical Patterns
# =============================================================================


class AndToTOSAPattern(OpRewritePattern):
    """onnx.And -> tosa.logical_and"""

    def match(self, op):
        return op.name == "onnx.And"

    def rewrite(self, op, rewriter):
        a = op.operands[0]
        b = op.operands[1]
        y = op.results[0]
        new_op = Operation("tosa.logical_and", operands=[a, b], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class OrToTOSAPattern(OpRewritePattern):
    """onnx.Or -> tosa.logical_or"""

    def match(self, op):
        return op.name == "onnx.Or"

    def rewrite(self, op, rewriter):
        a = op.operands[0]
        b = op.operands[1]
        y = op.results[0]
        new_op = Operation("tosa.logical_or", operands=[a, b], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class XorToTOSAPattern(OpRewritePattern):
    """onnx.Xor -> tosa.logical_xor"""

    def match(self, op):
        return op.name == "onnx.Xor"

    def rewrite(self, op, rewriter):
        a = op.operands[0]
        b = op.operands[1]
        y = op.results[0]
        new_op = Operation("tosa.logical_xor", operands=[a, b], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class NotToTOSAPattern(OpRewritePattern):
    """onnx.Not -> tosa.logical_not"""

    def match(self, op):
        return op.name == "onnx.Not"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]
        new_op = Operation("tosa.logical_not", operands=[x], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


# =============================================================================
# Bitwise Patterns
# =============================================================================


class BitwiseAndToTOSAPattern(OpRewritePattern):
    """onnx.BitwiseAnd -> tosa.bitwise_and"""

    def match(self, op):
        return op.name == "onnx.BitwiseAnd"

    def rewrite(self, op, rewriter):
        a = op.operands[0]
        b = op.operands[1]
        y = op.results[0]
        new_op = Operation("tosa.bitwise_and", operands=[a, b], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class BitwiseOrToTOSAPattern(OpRewritePattern):
    """onnx.BitwiseOr -> tosa.bitwise_or"""

    def match(self, op):
        return op.name == "onnx.BitwiseOr"

    def rewrite(self, op, rewriter):
        a = op.operands[0]
        b = op.operands[1]
        y = op.results[0]
        new_op = Operation("tosa.bitwise_or", operands=[a, b], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class BitwiseXorToTOSAPattern(OpRewritePattern):
    """onnx.BitwiseXor -> tosa.bitwise_xor"""

    def match(self, op):
        return op.name == "onnx.BitwiseXor"

    def rewrite(self, op, rewriter):
        a = op.operands[0]
        b = op.operands[1]
        y = op.results[0]
        new_op = Operation("tosa.bitwise_xor", operands=[a, b], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class BitwiseNotToTOSAPattern(OpRewritePattern):
    """onnx.BitwiseNot -> tosa.bitwise_not"""

    def match(self, op):
        return op.name == "onnx.BitwiseNot"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]
        new_op = Operation("tosa.bitwise_not", operands=[x], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


# =============================================================================
# Activation Patterns
# =============================================================================


class ReluToTOSAPattern(OpRewritePattern):
    """onnx.Relu -> tosa.clamp (with min=0)"""

    def match(self, op):
        return op.name == "onnx.Relu"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]
        # ReLU is clamp with min=0, max=float_max
        # Use float max value (3.40282347E+38) instead of inf for MLIR compatibility
        new_op = Operation(
            "tosa.clamp",
            operands=[x],
            results=[y],
            attributes={
                "min_fp": 0.0,
                "max_fp": 3.40282347e38,
                "min_int": Int64(0),
                "max_int": Int64(9223372036854775807),
            },
        )
        rewriter.replace_op(op, [new_op])
        return True


class ClipToTOSAPattern(OpRewritePattern):
    """onnx.Clip -> tosa.clamp"""

    def match(self, op):
        return op.name == "onnx.Clip"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]

        # Get min/max from operands (if they are constants)
        # Use float max/min values for MLIR compatibility
        min_val = -3.40282347e38
        max_val = 3.40282347e38

        if len(op.operands) > 1 and op.operands[1].owner is not None:
            min_dense = op.operands[1].owner.attributes.get("value")
            if min_dense is not None:
                min_arr = dense_to_numpy(min_dense)
                if min_arr is not None:
                    min_val = float(min_arr.flat[0])

        if len(op.operands) > 2 and op.operands[2].owner is not None:
            max_dense = op.operands[2].owner.attributes.get("value")
            if max_dense is not None:
                max_arr = dense_to_numpy(max_dense)
                if max_arr is not None:
                    max_val = float(max_arr.flat[0])

        new_op = Operation(
            "tosa.clamp",
            operands=[x],
            results=[y],
            attributes={
                "min_fp": min_val,
                "max_fp": max_val,
                "min_int": Int64(
                    int(min_val) if min_val > -3.4e38 else -9223372036854775808
                ),
                "max_int": Int64(
                    int(max_val) if max_val < 3.4e38 else 9223372036854775807
                ),
            },
        )
        rewriter.replace_op(op, [new_op])
        return True


# =============================================================================
# Reduce Patterns
# =============================================================================


class ReduceSumToTOSAPattern(OpRewritePattern):
    """onnx.ReduceSum -> tosa.reduce_sum"""

    def match(self, op):
        return op.name == "onnx.ReduceSum"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]

        # Get axes from operand or attribute
        if len(op.operands) > 1 and op.operands[1].owner is not None:
            axes_dense = op.operands[1].owner.attributes.get("value")
            if axes_dense is not None:
                axes = dense_to_numpy(axes_dense)
                if axes is not None:
                    axis = int(axes.flat[0])
                    new_op = Operation(
                        "tosa.reduce_sum",
                        operands=[x],
                        results=[y],
                        attributes={"axis": axis},
                    )
                    rewriter.replace_op(op, [new_op])
                    return True
        return False


class ReduceMaxToTOSAPattern(OpRewritePattern):
    """onnx.ReduceMax -> tosa.reduce_max"""

    def match(self, op):
        return op.name == "onnx.ReduceMax"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]

        if len(op.operands) > 1 and op.operands[1].owner is not None:
            axes_dense = op.operands[1].owner.attributes.get("value")
            if axes_dense is not None:
                axes = dense_to_numpy(axes_dense)
                if axes is not None:
                    axis = int(axes.flat[0])
                    new_op = Operation(
                        "tosa.reduce_max",
                        operands=[x],
                        results=[y],
                        attributes={"axis": axis},
                    )
                    rewriter.replace_op(op, [new_op])
                    return True
        return False


class ReduceMinToTOSAPattern(OpRewritePattern):
    """onnx.ReduceMin -> tosa.reduce_min"""

    def match(self, op):
        return op.name == "onnx.ReduceMin"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]

        if len(op.operands) > 1 and op.operands[1].owner is not None:
            axes_dense = op.operands[1].owner.attributes.get("value")
            if axes_dense is not None:
                axes = dense_to_numpy(axes_dense)
                if axes is not None:
                    axis = int(axes.flat[0])
                    new_op = Operation(
                        "tosa.reduce_min",
                        operands=[x],
                        results=[y],
                        attributes={"axis": axis},
                    )
                    rewriter.replace_op(op, [new_op])
                    return True
        return False


class ReduceProdToTOSAPattern(OpRewritePattern):
    """onnx.ReduceProd -> tosa.reduce_prod"""

    def match(self, op):
        return op.name == "onnx.ReduceProd"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]

        if len(op.operands) > 1 and op.operands[1].owner is not None:
            axes_dense = op.operands[1].owner.attributes.get("value")
            if axes_dense is not None:
                axes = dense_to_numpy(axes_dense)
                if axes is not None:
                    axis = int(axes.flat[0])
                    new_op = Operation(
                        "tosa.reduce_prod",
                        operands=[x],
                        results=[y],
                        attributes={"axis": axis},
                    )
                    rewriter.replace_op(op, [new_op])
                    return True
        return False


# =============================================================================
# Ternary Patterns
# =============================================================================


class WhereToTOSAPattern(OpRewritePattern):
    """onnx.Where -> tosa.select"""

    def match(self, op):
        return op.name == "onnx.Where"

    def rewrite(self, op, rewriter):
        condition = op.operands[0]
        x = op.operands[1]
        y_input = op.operands[2]
        y = op.results[0]
        # tosa.select(condition, on_true, on_false)
        new_op = Operation("tosa.select", operands=[condition, x, y_input], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


# =============================================================================
# Other Patterns
# =============================================================================


class ConcatToTOSAPattern(OpRewritePattern):
    """onnx.Concat -> tosa.concat"""

    def match(self, op):
        return op.name == "onnx.Concat"

    def rewrite(self, op, rewriter):
        inputs = op.operands
        y = op.results[0]
        axis = op.attributes.get("axis", 0)

        new_op = Operation(
            "tosa.concat",
            operands=inputs,
            results=[y],
            attributes={"axis": axis},
        )
        rewriter.replace_op(op, [new_op])
        return True


class SliceToTOSAPattern(OpRewritePattern):
    """onnx.Slice -> tosa.slice"""

    def match(self, op):
        return op.name == "onnx.Slice"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]

        # ONNX Slice has starts, ends, axes, steps as operands
        if len(op.operands) < 3:
            return False

        starts_dense = (
            op.operands[1].owner.attributes.get("value")
            if op.operands[1].owner
            else None
        )
        ends_dense = (
            op.operands[2].owner.attributes.get("value")
            if op.operands[2].owner
            else None
        )

        if starts_dense is None or ends_dense is None:
            return False

        starts = dense_to_numpy(starts_dense)
        ends = dense_to_numpy(ends_dense)

        if starts is None or ends is None:
            return False

        # Get axes (default: all axes in order)
        axes = None
        if len(op.operands) > 3 and op.operands[3].owner:
            axes_dense = op.operands[3].owner.attributes.get("value")
            if axes_dense is not None:
                axes = dense_to_numpy(axes_dense)
                if axes is not None:
                    axes = axes.tolist()

        if axes is None:
            axes = list(range(len(starts)))

        # Get steps (default: 1 for all)
        steps = None
        if len(op.operands) > 4 and op.operands[4].owner:
            steps_dense = op.operands[4].owner.attributes.get("value")
            if steps_dense is not None:
                steps = dense_to_numpy(steps_dense)
                if steps is not None:
                    steps = steps.tolist()

        # Check if steps are all 1 (TOSA doesn't support steps != 1)
        if steps is not None and any(s != 1 for s in steps):
            return False

        # Build full start and size arrays for all dimensions
        rank = len(x.shape)
        start_list = [0] * rank
        size_list = list(x.shape)

        for i, axis in enumerate(axes):
            if axis < 0:
                axis = rank + axis
            start_val = int(starts[i])
            end_val = int(ends[i])

            # Handle negative indices
            dim_size = x.shape[axis]
            if start_val < 0:
                start_val = max(0, dim_size + start_val)
            if end_val < 0:
                end_val = max(0, dim_size + end_val)

            # Clamp to valid range
            start_val = min(start_val, dim_size)
            end_val = min(end_val, dim_size)

            start_list[axis] = start_val
            size_list[axis] = max(0, end_val - start_val)

        new_op = Operation(
            "tosa.slice",
            operands=[x],
            results=[y],
            attributes={
                "start": Array(start_list),
                "size": Array(size_list),
            },
        )
        rewriter.replace_op(op, [new_op])
        return True


class IdentityToTOSAPattern(OpRewritePattern):
    """onnx.Identity -> tosa.identity"""

    def match(self, op):
        return op.name == "onnx.Identity"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]
        new_op = Operation("tosa.identity", operands=[x], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class CastToTOSAPattern(OpRewritePattern):
    """onnx.Cast -> tosa.cast"""

    def match(self, op):
        return op.name == "onnx.Cast"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]
        new_op = Operation("tosa.cast", operands=[x], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class MaxToTOSAPattern(OpRewritePattern):
    """onnx.Max -> tosa.maximum"""

    def match(self, op):
        return op.name == "onnx.Max"

    def rewrite(self, op, rewriter):
        if len(op.operands) != 2:
            return False
        a = op.operands[0]
        b = op.operands[1]
        y = op.results[0]
        new_op = Operation("tosa.maximum", operands=[a, b], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class MinToTOSAPattern(OpRewritePattern):
    """onnx.Min -> tosa.minimum"""

    def match(self, op):
        return op.name == "onnx.Min"

    def rewrite(self, op, rewriter):
        if len(op.operands) != 2:
            return False
        a = op.operands[0]
        b = op.operands[1]
        y = op.results[0]
        new_op = Operation("tosa.minimum", operands=[a, b], results=[y])
        rewriter.replace_op(op, [new_op])
        return True


class FlattenToTOSAPattern(OpRewritePattern):
    """onnx.Flatten -> tosa.reshape"""

    def match(self, op):
        return op.name == "onnx.Flatten"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]
        axis = op.attributes.get("axis", 1)

        # Calculate new shape
        shape = x.shape
        dim0 = 1
        dim1 = 1
        for i in range(axis):
            dim0 *= shape[i]
        for i in range(axis, len(shape)):
            dim1 *= shape[i]

        new_shape = [dim0, dim1]
        new_op = Operation(
            "tosa.reshape",
            operands=[x],
            results=[y],
            attributes={"new_shape": Array(new_shape)},
        )
        rewriter.replace_op(op, [new_op])
        return True


class SqueezeToTOSAPattern(OpRewritePattern):
    """onnx.Squeeze -> tosa.reshape"""

    def match(self, op):
        return op.name == "onnx.Squeeze"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]

        # Get axes to squeeze
        axes = []
        if len(op.operands) > 1 and op.operands[1].owner is not None:
            axes_dense = op.operands[1].owner.attributes.get("value")
            if axes_dense is not None:
                axes_arr = dense_to_numpy(axes_dense)
                if axes_arr is not None:
                    axes = axes_arr.tolist()

        # Calculate new shape (remove dimensions of size 1)
        new_shape = []
        for i, dim in enumerate(x.shape):
            if axes:
                if i not in axes and (i - len(x.shape)) not in axes:
                    new_shape.append(dim)
            else:
                if dim != 1:
                    new_shape.append(dim)

        new_op = Operation(
            "tosa.reshape",
            operands=[x],
            results=[y],
            attributes={"new_shape": Array(new_shape)},
        )
        rewriter.replace_op(op, [new_op])
        return True


class UnsqueezeToTOSAPattern(OpRewritePattern):
    """onnx.Unsqueeze -> tosa.reshape"""

    def match(self, op):
        return op.name == "onnx.Unsqueeze"

    def rewrite(self, op, rewriter):
        x = op.operands[0]
        y = op.results[0]

        # Get axes to unsqueeze
        if len(op.operands) < 2 or op.operands[1].owner is None:
            return False

        axes_dense = op.operands[1].owner.attributes.get("value")
        if axes_dense is None:
            return False

        axes_arr = dense_to_numpy(axes_dense)
        if axes_arr is None:
            return False

        axes = sorted(axes_arr.tolist())

        # Calculate new shape (insert dimensions of size 1)
        new_shape = list(x.shape)
        for axis in axes:
            if axis < 0:
                axis = len(new_shape) + 1 + axis
            new_shape.insert(axis, 1)

        new_op = Operation(
            "tosa.reshape",
            operands=[x],
            results=[y],
            attributes={"new_shape": Array(new_shape)},
        )
        rewriter.replace_op(op, [new_op])
        return True


# =============================================================================
# Main conversion function
# =============================================================================


def convert_onnx_to_tosa(ir: Function) -> Function:
    patterns = RewritePatternSet()
    patterns.add_patterns(
        [
            # Shape operations
            ReshapeToTOSAPattern(),
            TransposeToTOSAPattern(),
            FlattenToTOSAPattern(),
            SqueezeToTOSAPattern(),
            UnsqueezeToTOSAPattern(),
            # Matrix operations
            MatMulToTOSAPattern(),
            # Unary element-wise
            AbsToTOSAPattern(),
            NegToTOSAPattern(),
            ExpToTOSAPattern(),
            LogToTOSAPattern(),
            ReciprocalToTOSAPattern(),
            FloorToTOSAPattern(),
            CeilToTOSAPattern(),
            TanhToTOSAPattern(),
            SigmoidToTOSAPattern(),
            SqrtToTOSAPattern(),
            # Binary element-wise
            AddToTOSAPattern(),
            SubToTOSAPattern(),
            MulToTOSAPattern(),
            DivToTOSAPattern(),
            PowToTOSAPattern(),
            MaxToTOSAPattern(),
            MinToTOSAPattern(),
            # Comparison
            EqualToTOSAPattern(),
            GreaterToTOSAPattern(),
            GreaterOrEqualToTOSAPattern(),
            LessToTOSAPattern(),
            LessOrEqualToTOSAPattern(),
            # Logical
            AndToTOSAPattern(),
            OrToTOSAPattern(),
            XorToTOSAPattern(),
            NotToTOSAPattern(),
            # Bitwise
            BitwiseAndToTOSAPattern(),
            BitwiseOrToTOSAPattern(),
            BitwiseXorToTOSAPattern(),
            BitwiseNotToTOSAPattern(),
            # Activations
            ReluToTOSAPattern(),
            ClipToTOSAPattern(),
            # Reduce operations
            ReduceSumToTOSAPattern(),
            ReduceMaxToTOSAPattern(),
            ReduceMinToTOSAPattern(),
            ReduceProdToTOSAPattern(),
            # Ternary
            WhereToTOSAPattern(),
            # Other
            ConcatToTOSAPattern(),
            SliceToTOSAPattern(),
            IdentityToTOSAPattern(),
            CastToTOSAPattern(),
        ]
    )
    changed = GreedyPatternRewriter.apply_patterns_and_fold_greedily(ir, patterns)
    return ir
