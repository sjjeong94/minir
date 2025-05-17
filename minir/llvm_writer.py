from typing import Optional
from minir.ir import Value
from minir.writer import Writer


class LLVMWriter(Writer):
    def to_mlir(self, save_path: Optional[str] = None, elide: bool = False) -> str:
        mlir = self.to_function().__repr__(elide=elide)
        if save_path is not None:
            with open(save_path, "w") as f:
                f.write(mlir)
        return mlir

    def unary_op(self, name: str, operand: Value, **kwargs) -> Value:
        result = self.empty(dtype=operand.dtype, shape=operand.shape)
        self.write(name=name, operands=[operand], results=[result], attributes=kwargs)
        return result

    def binary_op(self, name: str, lhs: Value, rhs: Value, **kwargs) -> Value:
        result = self.empty(dtype=lhs.dtype, shape=lhs.shape)
        self.write(name=name, operands=[lhs, rhs], results=[result], attributes=kwargs)
        return result

    def fadd(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op(name="llvm.fadd", lhs=lhs, rhs=rhs)

    def fdiv(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op(name="llvm.fdiv", lhs=lhs, rhs=rhs)

    def fmul(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op(name="llvm.fmul", lhs=lhs, rhs=rhs)

    def fsub(self, lhs: Value, rhs: Value) -> Value:
        return self.binary_op(name="llvm.fsub", lhs=lhs, rhs=rhs)
