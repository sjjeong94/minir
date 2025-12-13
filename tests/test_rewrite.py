"""
Unit tests for the rewrite pattern system.
"""

import unittest
from minir.ir import Operation, Tensor, Function
from minir.rewrite import (
    OpRewritePattern,
    RewritePatternSet,
    PatternRewriter,
    GreedyPatternRewriter,
    DeadCodeEliminationPattern,
    IdentityEliminationPattern,
    ChainedIdentityPattern,
)


class TestPatternRewriter(unittest.TestCase):
    """Test PatternRewriter functionality."""

    def test_replace_op_with_value(self):
        """Test replacing an operation with a value."""
        input_tensor = Tensor("input", "f32", [1, 256])
        identity_out = Tensor("identity", "f32", [1, 256])
        final_out = Tensor("final", "f32", [1, 256])

        identity_op = Operation("identity", [input_tensor], [identity_out])
        operations = [
            identity_op,
            Operation("process", [identity_out], [final_out]),
            Operation("func.return", [final_out]),
        ]

        function = Function(operations, name="test")
        rewriter = PatternRewriter(function)

        # Replace identity operation with its input
        rewriter.replace_op_with_value(identity_op, input_tensor)
        rewriter.apply_changes()

        # Verify the identity operation is removed
        self.assertNotIn(identity_op, function.operations)
        # Verify the process operation now uses the input directly
        process_op = function.operations[0]
        self.assertEqual(process_op.operands[0], input_tensor)

    def test_erase_op(self):
        """Test erasing an operation."""
        input_tensor = Tensor("input", "f32", [1, 256])
        unused_out = Tensor("unused", "f32", [1, 256])

        unused_op = Operation("unused", [input_tensor], [unused_out])
        operations = [
            unused_op,
            Operation("func.return", [input_tensor]),
        ]

        function = Function(operations, name="test")
        initial_count = len(function.operations)

        rewriter = PatternRewriter(function)
        rewriter.erase_op(unused_op)
        rewriter.apply_changes()

        self.assertEqual(len(function.operations), initial_count - 1)
        self.assertNotIn(unused_op, function.operations)

    def test_replace_op_with_new_op(self):
        """Test replacing an operation with a new operation."""
        input_tensor = Tensor("input", "f32", [1, 256])
        old_out = Tensor("old", "f32", [1, 256])
        new_out = Tensor("new", "f32", [1, 256])

        old_op = Operation("old_op", [input_tensor], [old_out])
        new_op = Operation("new_op", [input_tensor], [new_out])

        operations = [
            old_op,
            Operation("func.return", [old_out]),
        ]

        function = Function(operations, name="test")
        rewriter = PatternRewriter(function)

        rewriter.replace_op_with_new_op(old_op, new_op)
        rewriter.apply_changes()

        self.assertNotIn(old_op, function.operations)
        self.assertIn(new_op, function.operations)

    def test_insert_op_before(self):
        """Test inserting an operation before another."""
        input_tensor = Tensor("input", "f32", [1, 256])
        process_out = Tensor("process", "f32", [1, 256])

        process_op = Operation("process", [input_tensor], [process_out])
        operations = [
            process_op,
            Operation("func.return", [process_out]),
        ]

        function = Function(operations, name="test")
        initial_count = len(function.operations)

        rewriter = PatternRewriter(function)

        prep_out = Tensor("prep", "f32", [1, 256])
        prep_op = Operation("prep", [input_tensor], [prep_out])
        rewriter.insert_op_before(process_op, prep_op)
        rewriter.apply_changes()

        self.assertEqual(len(function.operations), initial_count + 1)


class TestOpRewritePattern(unittest.TestCase):
    """Test OpRewritePattern base functionality."""

    def test_pattern_benefit(self):
        """Test pattern benefit ordering."""
        pattern1 = IdentityEliminationPattern("test", benefit=1)
        pattern2 = IdentityEliminationPattern("test", benefit=5)
        pattern3 = IdentityEliminationPattern("test", benefit=3)

        patterns = [pattern1, pattern2, pattern3]
        sorted_patterns = sorted(patterns, key=lambda p: p.benefit, reverse=True)

        self.assertEqual(sorted_patterns[0].benefit, 5)
        self.assertEqual(sorted_patterns[1].benefit, 3)
        self.assertEqual(sorted_patterns[2].benefit, 1)


class TestDeadCodeEliminationPattern(unittest.TestCase):
    """Test dead code elimination pattern."""

    def test_removes_unused_operations(self):
        """Test that unused operations are removed."""
        input_tensor = Tensor("input", "f32", [1, 256])
        unused_out = Tensor("unused", "f32", [1, 256])
        used_out = Tensor("used", "f32", [1, 256])

        operations = [
            Operation("compute_unused", [input_tensor], [unused_out]),
            Operation("compute_used", [input_tensor], [used_out]),
            Operation("func.return", [used_out]),
        ]

        function = Function(operations, name="test")

        patterns = RewritePatternSet()
        patterns.add_pattern(DeadCodeEliminationPattern())

        changed = GreedyPatternRewriter.apply_patterns_and_fold_greedily(
            function, patterns
        )

        self.assertTrue(changed)
        # Should have removed the unused operation
        self.assertEqual(len(function.operations), 2)
        # Verify the remaining operations
        op_names = [op.name for op in function.operations]
        self.assertIn("compute_used", op_names)
        self.assertNotIn("compute_unused", op_names)

    def test_preserves_used_operations(self):
        """Test that operations with used results are preserved."""
        input_tensor = Tensor("input", "f32", [1, 256])
        intermediate = Tensor("intermediate", "f32", [1, 256])
        final = Tensor("final", "f32", [1, 256])

        operations = [
            Operation("compute", [input_tensor], [intermediate]),
            Operation("process", [intermediate], [final]),
            Operation("func.return", [final]),
        ]

        function = Function(operations, name="test")
        initial_count = len(function.operations)

        patterns = RewritePatternSet()
        patterns.add_pattern(DeadCodeEliminationPattern())

        changed = GreedyPatternRewriter.apply_patterns_and_fold_greedily(
            function, patterns
        )

        self.assertFalse(changed)
        self.assertEqual(len(function.operations), initial_count)


class TestIdentityEliminationPattern(unittest.TestCase):
    """Test identity elimination pattern."""

    def test_eliminates_identity_ops(self):
        """Test that identity operations are eliminated."""
        input_tensor = Tensor("input", "f32", [1, 256])
        identity_out = Tensor("identity", "f32", [1, 256])

        operations = [
            Operation("identity", [input_tensor], [identity_out]),
            Operation("func.return", [identity_out]),
        ]

        function = Function(operations, name="test")

        patterns = RewritePatternSet()
        patterns.add_pattern(IdentityEliminationPattern("identity"))

        changed = GreedyPatternRewriter.apply_patterns_and_fold_greedily(
            function, patterns
        )

        self.assertTrue(changed)
        # Only func.return should remain
        self.assertEqual(len(function.operations), 1)
        self.assertEqual(function.operations[0].name, "func.return")


class TestChainedIdentityPattern(unittest.TestCase):
    """Test chained identity pattern."""

    def test_eliminates_chained_identities(self):
        """Test that chained identity operations are eliminated."""
        input_tensor = Tensor("input", "f32", [1, 256])
        identity1_out = Tensor("identity1", "f32", [1, 256])
        identity2_out = Tensor("identity2", "f32", [1, 256])
        identity3_out = Tensor("identity3", "f32", [1, 256])

        operations = [
            Operation("identity", [input_tensor], [identity1_out]),
            Operation("identity", [identity1_out], [identity2_out]),
            Operation("identity", [identity2_out], [identity3_out]),
            Operation("func.return", [identity3_out]),
        ]

        function = Function(operations, name="test")

        patterns = RewritePatternSet()
        patterns.add_pattern(ChainedIdentityPattern())
        patterns.add_pattern(IdentityEliminationPattern("identity"))

        changed = GreedyPatternRewriter.apply_patterns_and_fold_greedily(
            function, patterns
        )

        self.assertTrue(changed)
        # All identity operations should be eliminated
        self.assertEqual(len(function.operations), 1)
        self.assertEqual(function.operations[0].name, "func.return")
        # The return should use the original input
        self.assertEqual(function.operations[0].operands[0], input_tensor)


class TestRewritePatternSet(unittest.TestCase):
    """Test RewritePatternSet functionality."""

    def test_add_pattern(self):
        """Test adding patterns to the set."""
        pattern_set = RewritePatternSet()
        pattern = DeadCodeEliminationPattern()

        pattern_set.add_pattern(pattern)

        self.assertEqual(len(pattern_set.patterns), 1)
        self.assertIn(pattern, pattern_set.patterns)

    def test_add_patterns(self):
        """Test adding multiple patterns to the set."""
        pattern_set = RewritePatternSet()
        patterns = [
            DeadCodeEliminationPattern(),
            IdentityEliminationPattern("identity"),
        ]

        pattern_set.add_patterns(patterns)

        self.assertEqual(len(pattern_set.patterns), 2)

    def test_pattern_priority(self):
        """Test that patterns are applied in benefit order."""
        input_tensor = Tensor("input", "f32", [1, 256])
        identity_out = Tensor("identity", "f32", [1, 256])

        operations = [
            Operation("identity", [input_tensor], [identity_out]),
            Operation("func.return", [identity_out]),
        ]

        function = Function(operations, name="test")

        # Create patterns with different benefits
        low_benefit = IdentityEliminationPattern("identity", benefit=1)
        high_benefit = IdentityEliminationPattern("identity", benefit=10)

        pattern_set = RewritePatternSet()
        pattern_set.add_pattern(low_benefit)
        pattern_set.add_pattern(high_benefit)

        # Both patterns should work, but high benefit should be tried first
        rewriter = PatternRewriter(function)
        changed = pattern_set.apply_patterns(rewriter)

        self.assertTrue(changed)


class TestGreedyPatternRewriter(unittest.TestCase):
    """Test greedy pattern rewriter."""

    def test_converges_to_fixpoint(self):
        """Test that rewriting converges to a fixpoint."""
        input_tensor = Tensor("input", "f32", [1, 256])
        id1 = Tensor("id1", "f32", [1, 256])
        id2 = Tensor("id2", "f32", [1, 256])
        id3 = Tensor("id3", "f32", [1, 256])

        operations = [
            Operation("identity", [input_tensor], [id1]),
            Operation("identity", [id1], [id2]),
            Operation("identity", [id2], [id3]),
            Operation("func.return", [id3]),
        ]

        function = Function(operations, name="test")

        patterns = RewritePatternSet()
        patterns.add_pattern(ChainedIdentityPattern())
        patterns.add_pattern(IdentityEliminationPattern("identity"))

        changed = GreedyPatternRewriter.apply_patterns_and_fold_greedily(
            function, patterns, max_iterations=10
        )

        self.assertTrue(changed)
        # Should converge to just the return statement
        self.assertEqual(len(function.operations), 1)
        self.assertEqual(function.operations[0].name, "func.return")


class TestCommonSubexpressionElimination(unittest.TestCase):
    """Test Common Subexpression Elimination pattern."""

    def test_cse_eliminates_duplicate_operations(self):
        """Test that CSE eliminates duplicate computations."""
        from minir.rewrite import CSEPass

        input_a = Tensor("a", "f32", [1, 256])
        input_b = Tensor("b", "f32", [1, 256])
        add1_out = Tensor("add1", "f32", [1, 256])
        add2_out = Tensor("add2", "f32", [1, 256])
        mul_out = Tensor("mul", "f32", [1, 256])

        operations = [
            Operation("add", [input_a, input_b], [add1_out]),
            Operation("add", [input_a, input_b], [add2_out]),  # Duplicate
            Operation("mul", [add1_out, add2_out], [mul_out]),
            Operation("func.return", [mul_out]),
        ]

        function = Function(operations, name="test_cse")
        initial_count = len(function.operations)

        # Run CSE
        changed = CSEPass.run(function)

        self.assertTrue(changed)
        # One add operation should be eliminated
        self.assertEqual(len(function.operations), initial_count - 1)

        # Verify the remaining operations
        op_names = [op.name for op in function.operations]
        self.assertEqual(op_names.count("add"), 1)

    def test_cse_with_different_operands(self):
        """Test that CSE doesn't eliminate operations with different operands."""
        from minir.rewrite import CSEPass

        input_a = Tensor("a", "f32", [1, 256])
        input_b = Tensor("b", "f32", [1, 256])
        input_c = Tensor("c", "f32", [1, 256])
        add1_out = Tensor("add1", "f32", [1, 256])
        add2_out = Tensor("add2", "f32", [1, 256])

        operations = [
            Operation("add", [input_a, input_b], [add1_out]),
            Operation("add", [input_a, input_c], [add2_out]),  # Different operands
            Operation("func.return", [add1_out, add2_out]),
        ]

        function = Function(operations, name="test")
        initial_count = len(function.operations)

        changed = CSEPass.run(function)

        self.assertFalse(changed)
        self.assertEqual(len(function.operations), initial_count)

    def test_cse_with_attributes(self):
        """Test that CSE considers operation attributes."""
        from minir.rewrite import CSEPass

        input_tensor = Tensor("input", "f32", [1, 256])
        scale1_out = Tensor("scale1", "f32", [1, 256])
        scale2_out = Tensor("scale2", "f32", [1, 256])
        scale3_out = Tensor("scale3", "f32", [1, 256])

        operations = [
            Operation("scale", [input_tensor], [scale1_out], {"factor": 2.0}),
            Operation("scale", [input_tensor], [scale2_out], {"factor": 2.0}),  # Same
            Operation(
                "scale", [input_tensor], [scale3_out], {"factor": 3.0}
            ),  # Different
            Operation("func.return", [scale1_out, scale2_out, scale3_out]),
        ]

        function = Function(operations, name="test")

        changed = CSEPass.run(function)

        self.assertTrue(changed)
        # Only one scale with factor 2.0 should be eliminated
        op_names = [op.name for op in function.operations]
        self.assertEqual(op_names.count("scale"), 2)

    def test_cse_multiple_results(self):
        """Test CSE with operations that have multiple results."""
        from minir.rewrite import CSEPass

        input_tensor = Tensor("input", "f32", [1, 256])
        out1_a = Tensor("out1_a", "f32", [1, 128])
        out1_b = Tensor("out1_b", "f32", [1, 128])
        out2_a = Tensor("out2_a", "f32", [1, 128])
        out2_b = Tensor("out2_b", "f32", [1, 128])

        operations = [
            Operation("split", [input_tensor], [out1_a, out1_b], {"axis": 1}),
            Operation(
                "split", [input_tensor], [out2_a, out2_b], {"axis": 1}
            ),  # Duplicate
            Operation("func.return", [out1_a, out2_b]),
        ]

        function = Function(operations, name="test")

        changed = CSEPass.run(function)

        self.assertTrue(changed)
        # One split should be eliminated
        op_names = [op.name for op in function.operations]
        self.assertEqual(op_names.count("split"), 1)

    def test_cse_chain_of_operations(self):
        """Test CSE with a chain of operations."""
        from minir.rewrite import CSEPass

        a = Tensor("a", "f32", [1, 256])
        b = Tensor("b", "f32", [1, 256])
        add1 = Tensor("add1", "f32", [1, 256])
        mul1 = Tensor("mul1", "f32", [1, 256])
        add2 = Tensor("add2", "f32", [1, 256])
        mul2 = Tensor("mul2", "f32", [1, 256])
        final = Tensor("final", "f32", [1, 256])

        operations = [
            Operation("add", [a, b], [add1]),
            Operation("mul", [add1, b], [mul1]),
            Operation("add", [a, b], [add2]),  # Duplicate of first add
            Operation("mul", [add2, b], [mul2]),  # Duplicate of first mul
            Operation("add", [mul1, mul2], [final]),
            Operation("func.return", [final]),
        ]

        function = Function(operations, name="test")

        changed = CSEPass.run(function)

        self.assertTrue(changed)
        # Two operations should be eliminated
        op_names = [op.name for op in function.operations]
        self.assertEqual(op_names.count("add"), 2)  # 2 adds: 1 original + 1 final
        self.assertEqual(op_names.count("mul"), 1)  # 1 mul remaining

    def test_cse_does_not_eliminate_side_effects(self):
        """Test that CSE doesn't eliminate operations with side effects."""
        from minir.rewrite import CSEPass

        input_tensor = Tensor("input", "f32", [1, 256])
        call1_out = Tensor("call1", "f32", [1, 256])
        call2_out = Tensor("call2", "f32", [1, 256])

        operations = [
            Operation("func.call", [input_tensor], [call1_out], {"callee": "foo"}),
            Operation("func.call", [input_tensor], [call2_out], {"callee": "foo"}),
            Operation("func.return", [call1_out, call2_out]),
        ]

        function = Function(operations, name="test")
        initial_count = len(function.operations)

        changed = CSEPass.run(function)

        self.assertFalse(changed)
        self.assertEqual(len(function.operations), initial_count)


if __name__ == "__main__":
    unittest.main()
