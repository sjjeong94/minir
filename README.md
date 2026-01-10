# minir
A minimal implementation of an intermediate representation with MLIR-like rewrite patterns

## Features

- **IR Representation**: Define operations, values (scalars, vectors, tensors), and functions
- **Op Rewrite Patterns**: MLIR-inspired pattern matching and rewriting system for IR transformations
- **ONNX Support**: Convert between ONNX and minir IR
- **TOSA Support**: TOSA (Tensor Operator Set Architecture) writer support

## Op Rewrite Patterns

minir includes a powerful rewrite pattern system inspired by MLIR's pattern rewriting infrastructure. This allows you to define transformation patterns that can match and rewrite operations in your IR.

### Basic Usage

```python
from minir.ir import Operation, Tensor, Function
from minir.rewrite import (
    OpRewritePattern,
    RewritePatternSet,
    PatternRewriter,
    GreedyPatternRewriter,
)

# Define a custom pattern
class MyOptimizationPattern(OpRewritePattern):
    def match(self, op: Operation) -> bool:
        # Return True if this operation should be rewritten
        return op.name == "my_op" and some_condition(op)
    
    def rewrite(self, op: Operation, rewriter: PatternRewriter) -> bool:
        # Perform the rewrite
        rewriter.replace_op_with_value(op, op.operands[0])
        return True

# Create a function with operations
operations = [
    Operation("my_op", [input_tensor], [output_tensor]),
    Operation("func.return", [output_tensor]),
]
function = Function(operations, name="my_function")

# Apply the pattern
patterns = RewritePatternSet()
patterns.add_pattern(MyOptimizationPattern())
changed = GreedyPatternRewriter.apply_patterns_and_fold_greedily(function, patterns)
```

### Built-in Patterns

- **DeadCodeEliminationPattern**: Removes operations whose results are never used
- **IdentityEliminationPattern**: Eliminates identity operations that just pass through their input
- **ChainedIdentityPattern**: Collapses chains of identity operations
- **CommonSubexpressionEliminationPattern**: Eliminates redundant computations (CSE)
- **CSEPass**: Dedicated Common Subexpression Elimination pass for efficient optimization
- **ConstantFoldingPattern**: Base class for constant folding transformations
- **AttributeBasedPattern**: Pattern matching based on operation attributes

### Pattern Features

- **Pattern Benefits**: Assign priority scores to patterns for controlled application order
- **Greedy Rewriting**: Apply patterns iteratively until convergence
- **Value Replacement**: Track and apply value substitutions throughout the IR
- **Operation Management**: Insert, replace, and remove operations with automatic metadata updates

### Example: Dead Code Elimination

```python
from minir.rewrite import DeadCodeEliminationPattern

patterns = RewritePatternSet()
patterns.add_pattern(DeadCodeEliminationPattern())

# Automatically removes unused operations
GreedyPatternRewriter.apply_patterns_and_fold_greedily(function, patterns)
```

### Example: Common Subexpression Elimination

```python
from minir.rewrite import CSEPass

# Create a function with redundant computations
operations = [
    Operation("add", [a, b], [add1_out]),
    Operation("mul", [add1_out, c], [mul1_out]),
    Operation("add", [a, b], [add2_out]),  # Duplicate computation
    Operation("mul", [add2_out, c], [mul2_out]),  # Duplicate computation
    Operation("func.return", [mul1_out, mul2_out]),
]
function = Function(operations, name="test_cse")

# Run CSE to eliminate redundant computations
changed = CSEPass.run(function)
# After CSE: both add2_out and mul2_out are replaced with add1_out and mul1_out
```

### Example: Custom Attribute-Based Pattern

```python
class RemoveIdentityScalePattern(OpRewritePattern):
    """Remove scale operations with scale factor of 1.0."""
    
    def match(self, op: Operation) -> bool:
        return (op.name == "scale" and 
                op.attributes.get("factor") == 1.0)
    
    def rewrite(self, op: Operation, rewriter: PatternRewriter) -> bool:
        # Scale by 1.0 is identity, replace with input
        rewriter.replace_op_with_value(op, op.operands[0])
        return True
```
