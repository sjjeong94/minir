"""
MLIR-like op rewrite pattern implementation for minir.

This module provides pattern matching and rewriting capabilities similar to MLIR's
RewritePattern system. It allows you to define patterns that match specific operations
or operation sequences and replace them with optimized alternatives.

Example usage:
    from minir.ir import Operation, Tensor, Function
    from minir.rewrite import OpRewritePattern, RewritePatternSet, PatternRewriter

    # Define a pattern to fold identity operations
    class FoldIdentityPattern(OpRewritePattern):
        def match(self, op: Operation) -> bool:
            return op.name == "identity" and len(op.operands) == 1

        def rewrite(self, op: Operation, rewriter: PatternRewriter) -> bool:
            rewriter.replace_op_with_value(op, op.operands[0])
            return True

    # Apply patterns to a function
    patterns = RewritePatternSet()
    patterns.add_pattern(FoldIdentityPattern())
    rewriter = PatternRewriter(function)
    patterns.apply_patterns(rewriter)
"""

from typing import List, Optional, Dict, Any, Callable
from abc import ABC, abstractmethod
from minir.ir import Operation, Value, Function


class PatternRewriter:
    """
    Rewriter context that tracks changes and provides rewriting utilities.
    Similar to MLIR's PatternRewriter.
    """

    def __init__(self, function: Function):
        self.function = function
        self.operations_to_remove: List[Operation] = []
        self.operations_to_add: List[tuple[int, Operation]] = []  # (position, op)
        self.value_replacements: Dict[Value, Value] = {}

    def replace_op(self, old_op: Operation, new_ops: List[Operation]) -> None:
        """Replace an operation with one or more new operations."""
        old_idx = self.function.operations.index(old_op)
        self.operations_to_remove.append(old_op)
        for i, new_op in enumerate(new_ops):
            self.operations_to_add.append((old_idx + i, new_op))

    def replace_op_with_value(self, op: Operation, replacement: Value) -> None:
        """Replace all uses of an operation's results with a value."""
        for result in op.results:
            self.value_replacements[result] = replacement
        self.erase_op(op)

    def replace_op_with_new_op(self, old_op: Operation, new_op: Operation) -> None:
        """Replace an operation with a new operation."""
        old_idx = self.function.operations.index(old_op)

        # Map old results to new results
        for old_result, new_result in zip(old_op.results, new_op.results):
            self.value_replacements[old_result] = new_result

        self.operations_to_remove.append(old_op)
        self.operations_to_add.append((old_idx, new_op))

    def erase_op(self, op: Operation) -> None:
        """Mark an operation for removal."""
        self.operations_to_remove.append(op)

    def insert_op_before(self, reference_op: Operation, new_op: Operation) -> None:
        """Insert a new operation before a reference operation."""
        ref_idx = self.function.operations.index(reference_op)
        self.operations_to_add.append((ref_idx, new_op))

    def insert_op_after(self, reference_op: Operation, new_op: Operation) -> None:
        """Insert a new operation after a reference operation."""
        ref_idx = self.function.operations.index(reference_op)
        self.operations_to_add.append((ref_idx + 1, new_op))

    def replace_all_uses_with(self, old_value: Value, new_value: Value) -> None:
        """Replace all uses of a value with another value."""
        self.value_replacements[old_value] = new_value

    def apply_changes(self) -> bool:
        """Apply all pending changes to the function. Returns True if changes were made."""
        if (
            not self.operations_to_remove
            and not self.operations_to_add
            and not self.value_replacements
        ):
            return False

        # Apply value replacements
        for op in self.function.operations:
            for i, operand in enumerate(op.operands):
                if operand in self.value_replacements:
                    op.operands[i] = self.value_replacements[operand]

        # Remove operations
        for op in self.operations_to_remove:
            if op in self.function.operations:
                self.function.operations.remove(op)

        # Add operations (sort by position in reverse to maintain indices)
        self.operations_to_add.sort(key=lambda x: x[0], reverse=True)
        for pos, op in self.operations_to_add:
            # Clamp position to valid range
            pos = min(pos, len(self.function.operations))
            self.function.operations.insert(pos, op)

        # Rebuild function metadata
        self.function.set_owner_and_users()
        self.function.rename_values()

        # Clear pending changes
        self.operations_to_remove.clear()
        self.operations_to_add.clear()
        self.value_replacements.clear()

        return True


class OpRewritePattern(ABC):
    """
    Base class for operation rewrite patterns.
    Similar to MLIR's OpRewritePattern.
    """

    def __init__(self, benefit: int = 1):
        """
        Initialize the pattern.

        Args:
            benefit: Pattern benefit score. Higher benefit patterns are applied first.
        """
        self.benefit = benefit

    @abstractmethod
    def match(self, op: Operation) -> bool:
        """
        Check if this pattern matches the given operation.

        Args:
            op: The operation to match against.

        Returns:
            True if the pattern matches, False otherwise.
        """
        pass

    @abstractmethod
    def rewrite(self, op: Operation, rewriter: PatternRewriter) -> bool:
        """
        Rewrite the matched operation.

        Args:
            op: The operation to rewrite.
            rewriter: The pattern rewriter context.

        Returns:
            True if rewriting was successful, False otherwise.
        """
        pass

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> bool:
        """
        Convenience method that combines match and rewrite.

        Args:
            op: The operation to match and rewrite.
            rewriter: The pattern rewriter context.

        Returns:
            True if the operation was matched and rewritten, False otherwise.
        """
        if self.match(op):
            return self.rewrite(op, rewriter)
        return False


class RewritePatternSet:
    """
    A collection of rewrite patterns that can be applied to a function.
    Similar to MLIR's RewritePatternSet.
    """

    def __init__(self):
        self.patterns: List[OpRewritePattern] = []

    def add_pattern(self, pattern: OpRewritePattern) -> "RewritePatternSet":
        """Add a pattern to the set."""
        self.patterns.append(pattern)
        return self

    def add_patterns(self, patterns: List[OpRewritePattern]) -> "RewritePatternSet":
        """Add multiple patterns to the set."""
        self.patterns.extend(patterns)
        return self

    def apply_patterns(
        self, rewriter: PatternRewriter, max_iterations: int = 10
    ) -> bool:
        """
        Apply all patterns to the function until no more changes occur.

        Args:
            rewriter: The pattern rewriter context.
            max_iterations: Maximum number of iterations to prevent infinite loops.

        Returns:
            True if any patterns were applied, False otherwise.
        """
        # Sort patterns by benefit (higher first)
        sorted_patterns = sorted(self.patterns, key=lambda p: p.benefit, reverse=True)

        changed = False
        for iteration in range(max_iterations):
            iteration_changed = False

            # Try to apply patterns to each operation
            # Iterate over a copy since we may modify the operations list
            ops_snapshot = list(rewriter.function.operations)

            for op in ops_snapshot:
                if op not in rewriter.function.operations:
                    # Operation was already removed
                    continue

                for pattern in sorted_patterns:
                    if pattern.match_and_rewrite(op, rewriter):
                        iteration_changed = True
                        # Apply changes immediately after each successful rewrite
                        if rewriter.apply_changes():
                            changed = True
                        break  # Move to next operation after first successful match

            # If no changes in this iteration, we're done
            if not iteration_changed:
                break

        return changed


class GreedyPatternRewriter:
    """
    Applies patterns greedily to maximize rewrites.
    Similar to MLIR's applyPatternsAndFoldGreedily.
    """

    @staticmethod
    def apply_patterns_and_fold_greedily(
        function: Function, patterns: RewritePatternSet, max_iterations: int = 10
    ) -> bool:
        """
        Apply patterns greedily until convergence or max iterations reached.

        Args:
            function: The function to transform.
            patterns: The pattern set to apply.
            max_iterations: Maximum number of iterations.

        Returns:
            True if any changes were made, False otherwise.
        """
        rewriter = PatternRewriter(function)
        return patterns.apply_patterns(rewriter, max_iterations)


# Common utility patterns


class DeadCodeEliminationPattern(OpRewritePattern):
    """Remove operations whose results are never used."""

    def match(self, op: Operation) -> bool:
        # Don't remove terminator operations or operations with side effects
        if op.name in ["func.return"]:
            return False

        # Check if all results have no users
        return all(len(result.users) == 0 for result in op.results)

    def rewrite(self, op: Operation, rewriter: PatternRewriter) -> bool:
        rewriter.erase_op(op)
        return True


class ConstantFoldingPattern(OpRewritePattern):
    """Base class for constant folding patterns."""

    def __init__(self, op_name: str, fold_fn: Callable, benefit: int = 2):
        super().__init__(benefit)
        self.op_name = op_name
        self.fold_fn = fold_fn

    def match(self, op: Operation) -> bool:
        if op.name != self.op_name:
            return False
        # Check if all operands have constant values
        # This is a simplified check; in practice, you'd need to track constant ops
        return all(hasattr(operand, "constant_value") for operand in op.operands)

    def rewrite(self, op: Operation, rewriter: PatternRewriter) -> bool:
        # Extract constant values and fold
        values = [operand.constant_value for operand in op.operands]
        result_value = self.fold_fn(*values)

        # This would create a constant operation with the folded value
        # Implementation depends on your IR's constant representation
        # For now, we just demonstrate the pattern
        return False  # Placeholder


class OpCanonicalizationPattern(OpRewritePattern):
    """Base class for canonicalization patterns that simplify operations."""

    def __init__(self, op_name: str, benefit: int = 1):
        super().__init__(benefit)
        self.op_name = op_name

    def match(self, op: Operation) -> bool:
        return op.name == self.op_name


class IdentityEliminationPattern(OpRewritePattern):
    """Eliminate identity operations that just pass through their input."""

    def __init__(self, op_name: str, benefit: int = 1):
        super().__init__(benefit)
        self.op_name = op_name

    def match(self, op: Operation) -> bool:
        return (
            op.name == self.op_name and len(op.operands) == 1 and len(op.results) == 1
        )

    def rewrite(self, op: Operation, rewriter: PatternRewriter) -> bool:
        rewriter.replace_op_with_value(op, op.operands[0])
        return True


# Example patterns for common optimizations


class ChainedIdentityPattern(OpRewritePattern):
    """Replace chained identity operations with a single identity."""

    def __init__(self, op_name: str = "identity", benefit: int = 2):
        super().__init__(benefit)
        self.op_name = op_name

    def match(self, op: Operation) -> bool:
        if op.name != self.op_name or len(op.operands) != 1:
            return False

        # Check if operand comes from another identity op
        operand = op.operands[0]
        if operand.owner and operand.owner.name == self.op_name:
            return True
        return False

    def rewrite(self, op: Operation, rewriter: PatternRewriter) -> bool:
        # Get the input of the first identity operation
        first_identity = op.operands[0].owner
        original_input = first_identity.operands[0]

        # Replace the second identity with the original input
        rewriter.replace_op_with_value(op, original_input)
        return True


class AttributeBasedPattern(OpRewritePattern):
    """Pattern that matches based on operation attributes."""

    def __init__(self, op_name: str, attr_conditions: Dict[str, Any], benefit: int = 1):
        super().__init__(benefit)
        self.op_name = op_name
        self.attr_conditions = attr_conditions

    def match(self, op: Operation) -> bool:
        if op.name != self.op_name:
            return False

        for attr_name, expected_value in self.attr_conditions.items():
            if attr_name not in op.attributes:
                return False
            if callable(expected_value):
                if not expected_value(op.attributes[attr_name]):
                    return False
            elif op.attributes[attr_name] != expected_value:
                return False

        return True

    @abstractmethod
    def rewrite(self, op: Operation, rewriter: PatternRewriter) -> bool:
        pass


class CommonSubexpressionEliminationPattern(OpRewritePattern):
    """
    Common Subexpression Elimination (CSE) pattern.

    Eliminates redundant computations by identifying operations that compute
    the same result and reusing the first computation.

    Two operations are considered equivalent if they:
    1. Have the same operation name
    2. Have the same operands (in the same order)
    3. Have the same attributes

    Example:
        %1 = add(%a, %b)
        %2 = mul(%1, %c)
        %3 = add(%a, %b)  # Redundant - same as %1
        %4 = mul(%3, %c)  # Will use %2 after CSE

    After CSE:
        %1 = add(%a, %b)
        %2 = mul(%1, %c)
        %4 = mul(%1, %c)  # Uses %1 instead of %3
    """

    def __init__(self, benefit: int = 5):
        super().__init__(benefit)
        self.seen_operations: Dict[tuple, Operation] = {}

    def match(self, op: Operation) -> bool:
        # Don't CSE terminator operations or operations with side effects
        if op.name in ["func.return", "func.call"]:
            return False

        # Don't CSE operations without results
        if not op.results:
            return False

        # Create a signature for this operation
        signature = self._compute_signature(op)

        # Check if we've seen an equivalent operation before
        if signature in self.seen_operations:
            previous_op = self.seen_operations[signature]
            # Make sure the previous operation is still in the function
            # and dominates this operation (comes before it)
            if previous_op in op.owner.operations if hasattr(op, "owner") else True:
                return True

        return False

    def rewrite(self, op: Operation, rewriter: PatternRewriter) -> bool:
        signature = self._compute_signature(op)

        if signature in self.seen_operations:
            previous_op = self.seen_operations[signature]

            # Replace all results of the current operation with results from the previous one
            for old_result, new_result in zip(op.results, previous_op.results):
                rewriter.replace_all_uses_with(old_result, new_result)

            # Erase the redundant operation
            rewriter.erase_op(op)
            return True

        # If we get here, record this operation for future CSE
        # This happens during the match phase
        return False

    def _compute_signature(self, op: Operation) -> tuple:
        """
        Compute a signature for an operation that can be used to identify
        equivalent operations.

        Returns:
            A tuple representing the operation's signature.
        """
        # Create tuple of operand identities (using id() for object identity)
        operand_ids = tuple(id(operand) for operand in op.operands)

        # Create tuple of attributes (handling unhashable types like lists)
        attr_items = self._make_hashable_attrs(op.attributes)

        # Combine operation name, operands, and attributes into signature
        return (op.name, operand_ids, attr_items)

    def _make_hashable_attrs(self, attributes: Dict[str, Any]) -> tuple:
        """
        Convert attributes to a hashable tuple.
        Handles lists and other unhashable types by converting them to tuples.
        """
        if not attributes:
            return ()

        hashable_items = []
        for key, value in sorted(attributes.items()):
            if isinstance(value, list):
                # Convert list to tuple for hashing
                hashable_value = tuple(value)
            elif isinstance(value, dict):
                # Recursively handle nested dicts
                hashable_value = self._make_hashable_attrs(value)
            else:
                hashable_value = value
            hashable_items.append((key, hashable_value))

        return tuple(hashable_items)


class CSEPass:
    """
    A dedicated CSE pass that processes the entire function.

    This is more efficient than the pattern-based approach for CSE
    as it can do a single pass over the function and build a complete
    hash table of seen operations.
    """

    @staticmethod
    def run(function: Function, max_iterations: int = 10) -> bool:
        """
        Run CSE on the entire function.

        Args:
            function: The function to optimize.
            max_iterations: Maximum iterations to handle chained CSE opportunities.

        Returns:
            True if any changes were made, False otherwise.
        """
        overall_changed = False

        for _ in range(max_iterations):
            seen_operations: Dict[tuple, Operation] = {}
            value_map: Dict[int, Value] = {}  # Map value IDs to canonical values
            iteration_changed = False

            ops_to_remove = []
            value_replacements: Dict[Value, Value] = {}

            for op in function.operations:
                # Skip operations that shouldn't be CSE'd
                if op.name in ["func.return", "func.call"] or not op.results:
                    continue

                # Get canonical operands (follow value replacements)
                canonical_operands = []
                for operand in op.operands:
                    operand_id = id(operand)
                    canonical = value_map.get(operand_id, operand)
                    canonical_operands.append(canonical)

                # Compute signature with canonical operands
                signature = CSEPass._compute_signature_with_operands(
                    op.name, canonical_operands, op.attributes
                )

                if signature in seen_operations:
                    previous_op = seen_operations[signature]

                    # Replace all uses of current op with previous op
                    for old_result, new_result in zip(op.results, previous_op.results):
                        value_replacements[old_result] = new_result
                        value_map[id(old_result)] = new_result

                    ops_to_remove.append(op)
                    iteration_changed = True
                else:
                    # Record this operation with its canonical form
                    seen_operations[signature] = op
                    # Map this operation's results to themselves (as canonical)
                    for result in op.results:
                        value_map[id(result)] = result

            # Apply changes for this iteration
            if iteration_changed:
                rewriter = PatternRewriter(function)
                for old_val, new_val in value_replacements.items():
                    rewriter.replace_all_uses_with(old_val, new_val)
                for op in ops_to_remove:
                    rewriter.erase_op(op)
                rewriter.apply_changes()
                overall_changed = True
            else:
                # No more changes, exit early
                break

        return overall_changed

    @staticmethod
    def _compute_signature(op: Operation) -> tuple:
        """Compute a signature for an operation."""
        operand_ids = tuple(id(operand) for operand in op.operands)
        attr_items = CSEPass._make_hashable_attrs(op.attributes)
        return (op.name, operand_ids, attr_items)

    @staticmethod
    def _compute_signature_with_operands(
        op_name: str, operands: List[Value], attributes: Dict[str, Any]
    ) -> tuple:
        """Compute a signature for an operation with given operands."""
        operand_ids = tuple(id(operand) for operand in operands)
        attr_items = CSEPass._make_hashable_attrs(attributes)
        return (op_name, operand_ids, attr_items)

    @staticmethod
    def _make_hashable_attrs(attributes: Dict[str, Any]) -> tuple:
        """
        Convert attributes to a hashable tuple.
        Handles lists and other unhashable types by converting them to tuples.
        """
        if not attributes:
            return ()

        hashable_items = []
        for key, value in sorted(attributes.items()):
            if isinstance(value, list):
                # Convert list to tuple for hashing
                hashable_value = tuple(value)
            elif isinstance(value, dict):
                # Recursively handle nested dicts
                hashable_value = CSEPass._make_hashable_attrs(value)
            else:
                hashable_value = value
            hashable_items.append((key, hashable_value))

        return tuple(hashable_items)
