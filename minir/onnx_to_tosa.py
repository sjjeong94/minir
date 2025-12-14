from minir import Function, Tensor, Operation, Array
from minir.utils import dense_to_numpy, numpy_to_dense
from minir import OpRewritePattern, RewritePatternSet, GreedyPatternRewriter
import numpy as np


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


def convert_onnx_to_tosa(ir: Function) -> Function:
    patterns = RewritePatternSet()
    patterns.add_patterns(
        [
            ReshapeToTOSAPattern(),
            TransposeToTOSAPattern(),
            MatMulToTOSAPattern(),
        ]
    )
    changed = GreedyPatternRewriter.apply_patterns_and_fold_greedily(ir, patterns)
    return ir
