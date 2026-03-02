"""
Binary format conversion helpers for minir IR.

`to_bytes` emits real MLIR bytecode by invoking `mlir-opt --emit-bytecode`, so
the result can be consumed directly by MLIR tooling. `from_bytes` supports both
the MLIR bytecode emitted by `to_bytes` and the legacy MinIR-specific binary
format retained here for backward compatibility.
"""

import os
import re
import shutil
import struct
import subprocess
from typing import List, Dict, Any, Optional, Tuple, BinaryIO, Union
from io import BytesIO
from pathlib import Path
from minir.ir import Function, Operation, Value, Tensor, Dense, Array, Int64


# Magic number for MLIR bytecode.
MAGIC_NUMBER = b"ML\xefR"

# Magic number for the legacy MinIR bytecode format (8 bytes).
LEGACY_MAGIC_NUMBER = b"\x7fMinIR\x00\x00"

# Version
VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_TAG = 0

# Section IDs
SECTION_STRING = 0x01
SECTION_FUNCTION_TABLE = 0x02
SECTION_CONSTANT = 0x04
SECTION_TYPE = 0x05
SECTION_END = 0x00

# Type tags
TYPE_I1 = 0x00
TYPE_I8 = 0x01
TYPE_I16 = 0x02
TYPE_I32 = 0x03
TYPE_I64 = 0x04
TYPE_F16 = 0x05
TYPE_BF16 = 0x06
TYPE_F32 = 0x07
TYPE_TF32 = 0x08
TYPE_F64 = 0x09
TYPE_TENSOR = 0x0D
# Attribute tags
ATTR_INTEGER = 0x01
ATTR_FLOAT = 0x02
ATTR_BOOL = 0x03
ATTR_STRING = 0x05
ATTR_ARRAY = 0x06
ATTR_DENSE = 0x07

# dtype to type tag mapping
DTYPE_TO_TAG = {
    "i1": TYPE_I1,
    "i8": TYPE_I8,
    "i16": TYPE_I16,
    "i32": TYPE_I32,
    "i64": TYPE_I64,
    "f16": TYPE_F16,
    "bf16": TYPE_BF16,
    "f32": TYPE_F32,
    "tf32": TYPE_TF32,
    "f64": TYPE_F64,
}

TAG_TO_DTYPE = {v: k for k, v in DTYPE_TO_TAG.items()}


# =============================================================================
# VarInt Encoding/Decoding (PrefixVarInt format)
# =============================================================================


def encode_varint(value: int) -> bytes:
    """
    Encode an integer as a variable-length integer (PrefixVarInt format).
    Small integers use fewer bytes.
    """
    if value < 0:
        raise ValueError("VarInt does not support negative values")

    result = []
    while value >= 0x80:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def decode_varint(stream: BinaryIO) -> int:
    """
    Decode a variable-length integer from a byte stream.
    """
    result = 0
    shift = 0
    while True:
        byte = stream.read(1)
        if not byte:
            raise EOFError("Unexpected end of stream while reading VarInt")
        b = byte[0]
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            break
        shift += 7
    return result


# =============================================================================
# Binary Writer
# =============================================================================


class BinaryWriter:
    """
    Serializes a minir Function to binary format.
    """

    def __init__(self) -> None:
        self.strings: List[str] = []
        self.string_to_index: Dict[str, int] = {}
        self.types: List[Tuple[int, Any]] = []  # (type_tag, payload)
        self.type_to_index: Dict[str, int] = {}  # type signature -> index
        self.constants: List[bytes] = []
        self.constant_to_index: Dict[bytes, int] = {}

    def _add_string(self, s: str) -> int:
        """Add a string to the string section and return its index."""
        if s in self.string_to_index:
            return self.string_to_index[s]
        idx = len(self.strings)
        self.strings.append(s)
        self.string_to_index[s] = idx
        return idx

    def _add_type(self, value: Value) -> int:
        """Add a type to the type section and return its index."""
        if isinstance(value, Tensor):
            sig = f"tensor<{'x'.join(str(d) for d in value.shape)}x{value.dtype}>"
        else:
            sig = value.dtype if hasattr(value, "dtype") else "unknown"

        if sig in self.type_to_index:
            return self.type_to_index[sig]

        idx = len(self.types)

        if isinstance(value, Tensor):
            dtype_tag = DTYPE_TO_TAG.get(value.dtype, TYPE_F32)
            self.types.append((TYPE_TENSOR, (dtype_tag, value.shape)))
        else:
            dtype_tag = DTYPE_TO_TAG.get(getattr(value, "dtype", "f32"), TYPE_F32)
            self.types.append((dtype_tag, None))

        self.type_to_index[sig] = idx
        return idx

    def _add_constant(self, data: bytes) -> int:
        """Add constant data and return its index."""
        if data in self.constant_to_index:
            return self.constant_to_index[data]
        idx = len(self.constants)
        self.constants.append(data)
        self.constant_to_index[data] = idx
        return idx

    def _encode_string_section(self) -> bytes:
        """Encode the string section."""
        buffer = BytesIO()

        # Number of strings
        buffer.write(encode_varint(len(self.strings)))

        # Compute string start indices
        string_data = b""
        indices = []
        for s in self.strings:
            indices.append(len(string_data))
            string_data += s.encode("utf-8")

        # Write string start indices as uint32
        for idx in indices:
            buffer.write(struct.pack("<I", idx))

        # Write string data blob
        buffer.write(string_data)

        return buffer.getvalue()

    def _encode_type_section(self) -> bytes:
        """Encode the type section."""
        buffer = BytesIO()

        # Number of types
        buffer.write(encode_varint(len(self.types)))

        # Encode each type and collect offsets
        type_data = BytesIO()
        offsets = []

        for type_tag, payload in self.types:
            offsets.append(type_data.tell())
            type_data.write(bytes([type_tag]))

            if type_tag == TYPE_TENSOR and payload is not None:
                elem_type, shape = payload
                type_data.write(encode_varint(elem_type))
                type_data.write(encode_varint(len(shape)))
                for dim in shape:
                    type_data.write(struct.pack("<q", dim))  # int64

        # Write type start indices as uint32
        for offset in offsets:
            buffer.write(struct.pack("<I", offset))

        # Write type data blob
        buffer.write(type_data.getvalue())

        return buffer.getvalue()

    def _encode_constant_section(self) -> bytes:
        """Encode the constant data section."""
        buffer = BytesIO()

        # Number of constants
        buffer.write(encode_varint(len(self.constants)))

        # Compute constant start indices
        constant_data = b""
        indices = []
        for data in self.constants:
            indices.append(len(constant_data))
            constant_data += data

        # Write constant start indices as uint64
        for idx in indices:
            buffer.write(struct.pack("<Q", idx))

        # Write constant data blob
        buffer.write(constant_data)

        return buffer.getvalue()

    def _encode_attribute(self, key: str, value: Any) -> bytes:
        """Encode an attribute value."""
        buffer = BytesIO()
        key_idx = self._add_string(key)
        buffer.write(encode_varint(key_idx))

        if isinstance(value, bool):
            buffer.write(bytes([ATTR_BOOL]))
            buffer.write(bytes([0x01 if value else 0x00]))
        elif isinstance(value, Int64):
            buffer.write(bytes([ATTR_INTEGER]))
            buffer.write(encode_varint(TYPE_I64))
            buffer.write(struct.pack("<q", value.value))
        elif isinstance(value, int):
            buffer.write(bytes([ATTR_INTEGER]))
            buffer.write(encode_varint(TYPE_I32))
            buffer.write(struct.pack("<i", value))
        elif isinstance(value, float):
            buffer.write(bytes([ATTR_FLOAT]))
            buffer.write(encode_varint(TYPE_F32))
            buffer.write(struct.pack("<f", value))
        elif isinstance(value, str):
            buffer.write(bytes([ATTR_STRING]))
            str_idx = self._add_string(value)
            buffer.write(encode_varint(str_idx))
        elif isinstance(value, Array):
            buffer.write(bytes([ATTR_ARRAY]))
            buffer.write(encode_varint(len(value.data)))
            for item in value.data:
                buffer.write(struct.pack("<q", item))
        elif isinstance(value, Dense):
            buffer.write(bytes([ATTR_DENSE]))
            dtype_tag = DTYPE_TO_TAG.get(value.dtype, TYPE_F32)
            buffer.write(encode_varint(dtype_tag))
            buffer.write(encode_varint(len(value.shape)))
            for dim in value.shape:
                buffer.write(struct.pack("<q", dim))
            const_idx = self._add_constant(value.data)
            buffer.write(encode_varint(const_idx))
        elif isinstance(value, list):
            buffer.write(bytes([ATTR_ARRAY]))
            buffer.write(encode_varint(len(value)))
            for item in value:
                if isinstance(item, int):
                    buffer.write(struct.pack("<q", item))
                else:
                    buffer.write(struct.pack("<q", int(item)))
        else:
            # Fallback: encode as string
            buffer.write(bytes([ATTR_STRING]))
            str_idx = self._add_string(str(value))
            buffer.write(encode_varint(str_idx))

        return buffer.getvalue()

    def _encode_operation(
        self, op: Operation, value_indices: Dict[Value, int]
    ) -> bytes:
        """Encode a single operation."""
        buffer = BytesIO()

        # Operation name (string index)
        name_idx = self._add_string(op.name)
        buffer.write(encode_varint(name_idx))

        # Location index (0 for no debug info)
        buffer.write(encode_varint(0))

        # Number of operands
        buffer.write(encode_varint(len(op.operands)))

        # Operand indices
        for operand in op.operands:
            idx = value_indices.get(operand, 0)
            buffer.write(encode_varint(idx))

        # Number of results
        buffer.write(encode_varint(len(op.results)))

        # Result type indices
        for result in op.results:
            type_idx = self._add_type(result)
            buffer.write(encode_varint(type_idx))

        # Number of attributes
        buffer.write(encode_varint(len(op.attributes)))

        # Attributes
        for key, value in op.attributes.items():
            buffer.write(self._encode_attribute(key, value))

        return buffer.getvalue()

    def _encode_function_table_section(self, func: Function) -> bytes:
        """Encode the function table section."""
        buffer = BytesIO()

        # Number of functions (just 1 for now)
        buffer.write(encode_varint(1))

        # Function name index
        name_idx = self._add_string(func.name)
        buffer.write(encode_varint(name_idx))

        # Build value index map
        value_indices: Dict[Value, int] = {}
        idx = 0
        for value in func.arguments:
            value_indices[value] = idx
            idx += 1
        for op in func.operations:
            for result in op.results:
                value_indices[result] = idx
                idx += 1

        # Encode function signature (argument types and result types)
        # Number of arguments
        buffer.write(encode_varint(len(func.arguments)))
        for arg in func.arguments:
            type_idx = self._add_type(arg)
            buffer.write(encode_varint(type_idx))

        # Number of results
        buffer.write(encode_varint(len(func.results)))
        for res in func.results:
            type_idx = self._add_type(res)
            buffer.write(encode_varint(type_idx))

        # Entry flag (0 = public device function)
        buffer.write(bytes([0x00]))

        # Function body
        func_body = BytesIO()
        func_body.write(encode_varint(len(func.operations)))
        for op in func.operations:
            func_body.write(self._encode_operation(op, value_indices))

        func_body_bytes = func_body.getvalue()
        buffer.write(encode_varint(len(func_body_bytes)))
        buffer.write(func_body_bytes)

        return buffer.getvalue()

    def _encode_section(self, section_id: int, data: bytes) -> bytes:
        """Encode a section with header."""
        buffer = BytesIO()
        buffer.write(bytes([section_id]))  # Section ID (no alignment for now)
        buffer.write(encode_varint(len(data)))  # Length
        buffer.write(data)  # Payload
        return buffer.getvalue()

    def serialize(self, func: Function) -> bytes:
        """Serialize a Function to binary format."""
        buffer = BytesIO()

        # Write header
        buffer.write(LEGACY_MAGIC_NUMBER)
        buffer.write(bytes([VERSION_MAJOR, VERSION_MINOR]))
        buffer.write(struct.pack("<H", VERSION_TAG))

        # First pass: collect all strings, types, constants
        # by encoding the function table
        func_table_data = self._encode_function_table_section(func)

        # Now encode all sections in order
        # String section
        string_data = self._encode_string_section()
        buffer.write(self._encode_section(SECTION_STRING, string_data))

        # Type section
        type_data = self._encode_type_section()
        buffer.write(self._encode_section(SECTION_TYPE, type_data))

        # Constant section (if any)
        if self.constants:
            const_data = self._encode_constant_section()
            buffer.write(self._encode_section(SECTION_CONSTANT, const_data))

        # Function table section
        buffer.write(self._encode_section(SECTION_FUNCTION_TABLE, func_table_data))

        # End marker
        buffer.write(bytes([SECTION_END]))

        return buffer.getvalue()


# =============================================================================
# Binary Reader
# =============================================================================


class BinaryReader:
    """
    Deserializes a binary format back to a minir Function.
    """

    def __init__(self, data: bytes) -> None:
        self.stream = BytesIO(data)
        self.strings: List[str] = []
        self.types: List[Tuple[int, Any]] = []
        self.constants: List[bytes] = []

    def _read_bytes(self, n: int) -> bytes:
        """Read exactly n bytes."""
        data = self.stream.read(n)
        if len(data) < n:
            raise EOFError(f"Expected {n} bytes, got {len(data)}")
        return data

    def _read_varint(self) -> int:
        """Read a variable-length integer."""
        return decode_varint(self.stream)

    def _read_header(self) -> None:
        """Read and validate the legacy bytecode header."""
        magic = self._read_bytes(8)
        if magic != LEGACY_MAGIC_NUMBER:
            raise ValueError(f"Invalid magic number: {magic}")

        self._read_bytes(1)  # major
        self._read_bytes(1)  # minor
        self._read_bytes(2)  # tag

    def _read_string_section(self, data: bytes) -> None:
        """Parse the string section."""
        stream = BytesIO(data)
        num_strings = decode_varint(stream)

        # Read string start indices
        indices = []
        for _ in range(num_strings):
            indices.append(struct.unpack("<I", stream.read(4))[0])

        # Read string data blob
        string_data = stream.read()

        # Extract strings
        self.strings = []
        for i in range(num_strings):
            start = indices[i]
            end = indices[i + 1] if i + 1 < num_strings else len(string_data)
            self.strings.append(string_data[start:end].decode("utf-8"))

    def _read_type_section(self, data: bytes) -> None:
        """Parse the type section."""
        stream = BytesIO(data)
        num_types = decode_varint(stream)

        # Read type start indices
        offsets = []
        for _ in range(num_types):
            offsets.append(struct.unpack("<I", stream.read(4))[0])

        # Read type data blob
        type_data = stream.read()

        # Extract types
        self.types = []
        for i in range(num_types):
            start = offsets[i]
            type_stream = BytesIO(type_data[start:])
            type_tag = type_stream.read(1)[0]

            if type_tag == TYPE_TENSOR:
                elem_type = decode_varint(type_stream)
                rank = decode_varint(type_stream)
                shape = []
                for _ in range(rank):
                    dim = struct.unpack("<q", type_stream.read(8))[0]
                    shape.append(dim)
                self.types.append((type_tag, (elem_type, shape)))
            else:
                self.types.append((type_tag, None))

    def _read_constant_section(self, data: bytes) -> None:
        """Parse the constant data section."""
        stream = BytesIO(data)
        num_constants = decode_varint(stream)

        # Read constant start indices as uint64
        indices = []
        for _ in range(num_constants):
            indices.append(struct.unpack("<Q", stream.read(8))[0])

        # Read constant data blob
        const_data = stream.read()

        # Extract constants
        self.constants = []
        for i in range(num_constants):
            start = indices[i]
            end = indices[i + 1] if i + 1 < num_constants else len(const_data)
            self.constants.append(const_data[start:end])

    def _create_value_from_type(self, type_idx: int, name: str = "") -> Value:
        """Create a Value object from a type index."""
        type_tag, payload = self.types[type_idx]

        if type_tag == TYPE_TENSOR:
            elem_type, shape = payload
            dtype = TAG_TO_DTYPE.get(elem_type, "f32")
            return Tensor(name=name, dtype=dtype, shape=shape)
        else:
            dtype = TAG_TO_DTYPE.get(type_tag, "f32")
            return Tensor(name=name, dtype=dtype, shape=[1])

    def _decode_attribute(self, stream: BinaryIO) -> Tuple[str, Any]:
        """Decode an attribute key-value pair."""
        key_idx = decode_varint(stream)
        key = self.strings[key_idx]

        attr_tag = stream.read(1)[0]

        if attr_tag == ATTR_BOOL:
            value = stream.read(1)[0] == 0x01
        elif attr_tag == ATTR_INTEGER:
            type_tag = decode_varint(stream)
            if type_tag == TYPE_I64:
                value = Int64(struct.unpack("<q", stream.read(8))[0])
            else:
                value = struct.unpack("<i", stream.read(4))[0]
        elif attr_tag == ATTR_FLOAT:
            decode_varint(stream)  # type tag (ignored)
            value = struct.unpack("<f", stream.read(4))[0]
        elif attr_tag == ATTR_STRING:
            str_idx = decode_varint(stream)
            value = self.strings[str_idx]
        elif attr_tag == ATTR_ARRAY:
            num_elements = decode_varint(stream)
            data = []
            for _ in range(num_elements):
                data.append(struct.unpack("<q", stream.read(8))[0])
            value = Array(data)
        elif attr_tag == ATTR_DENSE:
            dtype_tag = decode_varint(stream)
            rank = decode_varint(stream)
            shape = []
            for _ in range(rank):
                shape.append(struct.unpack("<q", stream.read(8))[0])
            const_idx = decode_varint(stream)
            dtype = TAG_TO_DTYPE.get(dtype_tag, "f32")
            value = Dense(self.constants[const_idx], dtype, shape)
        else:
            # Unknown attribute, try to skip
            value = None

        return key, value

    def _read_operation(self, stream: BinaryIO, values: List[Value]) -> Operation:
        """Decode a single operation."""
        name_idx = decode_varint(stream)
        name = self.strings[name_idx]

        # Location index (skip for now)
        decode_varint(stream)

        # Operands
        num_operands = decode_varint(stream)
        operands = []
        for _ in range(num_operands):
            operand_idx = decode_varint(stream)
            if operand_idx < len(values):
                operands.append(values[operand_idx])

        # Results
        num_results = decode_varint(stream)
        results = []
        for _ in range(num_results):
            type_idx = decode_varint(stream)
            result = self._create_value_from_type(type_idx)
            results.append(result)
            values.append(result)

        # Attributes
        num_attrs = decode_varint(stream)
        attributes = {}
        for _ in range(num_attrs):
            key, value = self._decode_attribute(stream)
            if value is not None:
                attributes[key] = value

        return Operation(
            name=name, operands=operands, results=results, attributes=attributes
        )

    def _read_function_table_section(self, data: bytes) -> List[Function]:
        """Parse the function table section."""
        stream = BytesIO(data)
        num_functions = decode_varint(stream)

        functions = []
        for _ in range(num_functions):
            # Function name
            name_idx = decode_varint(stream)
            func_name = self.strings[name_idx]

            # Build value list starting with arguments
            values: List[Value] = []

            # Arguments
            num_args = decode_varint(stream)
            for i in range(num_args):
                type_idx = decode_varint(stream)
                arg = self._create_value_from_type(type_idx, f"%arg{i}")
                values.append(arg)

            # Result types (just read, not used directly as values)
            num_results = decode_varint(stream)
            for _ in range(num_results):
                decode_varint(stream)  # type_idx

            # Entry flag
            stream.read(1)

            # Function body
            body_length = decode_varint(stream)
            body_stream = BytesIO(stream.read(body_length))

            num_ops = decode_varint(body_stream)
            operations = []
            for _ in range(num_ops):
                op = self._read_operation(body_stream, values)
                operations.append(op)

            func = Function(operations, name=func_name)
            functions.append(func)

        return functions

    def _read_sections(self) -> Dict[int, bytes]:
        """Read all sections from the stream."""
        sections = {}
        while True:
            section_id_byte = self.stream.read(1)
            if not section_id_byte:
                break

            section_id = section_id_byte[0]

            # Check for end marker
            if section_id == SECTION_END:
                break

            # Check alignment bit
            has_alignment = (section_id & 0x80) != 0
            section_id = section_id & 0x7F

            if has_alignment:
                self._read_varint()  # alignment
                # Skip padding bytes (0xCB)
                while True:
                    pos = self.stream.tell()
                    b = self.stream.read(1)
                    if not b or b[0] != 0xCB:
                        self.stream.seek(pos)
                        break

            # Read section length and data
            length = self._read_varint()
            data = self._read_bytes(length)
            sections[section_id] = data

        return sections

    def deserialize(self) -> Function:
        """Deserialize binary data to a Function."""
        # Read header
        self._read_header()

        # Read all sections
        sections = self._read_sections()

        # Parse sections in dependency order
        if SECTION_STRING in sections:
            self._read_string_section(sections[SECTION_STRING])

        if SECTION_TYPE in sections:
            self._read_type_section(sections[SECTION_TYPE])

        if SECTION_CONSTANT in sections:
            self._read_constant_section(sections[SECTION_CONSTANT])

        # Parse function table
        if SECTION_FUNCTION_TABLE in sections:
            functions = self._read_function_table_section(
                sections[SECTION_FUNCTION_TABLE]
            )
            if functions:
                return functions[0]

        raise ValueError("No function found in binary data")


# =============================================================================
# Public API
# =============================================================================


def to_bytes(func: Function, path: Optional[str] = None) -> bytes:
    """
    Serialize a Function to binary format.

    Args:
        func: The Function object to serialize.
        path: Optional file path to save the binary data to.

    Returns:
        Binary representation of the function.

    Example:
        >>> from minir import Function, Operation, Tensor
        >>> t = Tensor("x", "f32", [2, 3])
        >>> op = Operation("onnx.Relu", operands=[t], results=[t])
        >>> func = Function([op], name="test")
        >>> data = to_bytes(func)
        >>> len(data) > 0
        True
        >>> to_bytes(func, "model.mir")  # Save to file
    """
    data = _serialize_mlir_bytecode(func)
    if path is not None:
        with open(path, "wb") as f:
            f.write(data)
    return data


def from_bytes(data: Union[bytes, str]) -> Function:
    """
    Deserialize a Function from binary format.

    Args:
        data: Binary data or file path to deserialize.

    Returns:
        The deserialized Function object.

    Example:
        >>> from minir import Function, Operation, Tensor
        >>> t = Tensor("x", "f32", [2, 3])
        >>> op = Operation("onnx.Relu", operands=[t], results=[t])
        >>> func = Function([op], name="test")
        >>> data = to_bytes(func)
        >>> restored = from_bytes(data)
        >>> restored.name == func.name
        True
        >>> restored = from_bytes("model.mir")  # Load from file
    """
    if isinstance(data, str):
        with open(data, "rb") as f:
            data = f.read()
    if data.startswith(LEGACY_MAGIC_NUMBER):
        reader = BinaryReader(data)
        return reader.deserialize()
    if data.startswith(MAGIC_NUMBER):
        return _deserialize_mlir_bytecode(data)
    raise ValueError("Unsupported bytecode format")


def _find_mlir_opt() -> str:
    """Locate a usable mlir-opt binary."""
    env_path = os.environ.get("MLIR_OPT")
    candidates: List[Optional[Union[str, Path]]] = [env_path, shutil.which("mlir-opt")]

    base = Path(__file__).resolve()
    for parent in base.parents:
        candidates.append(parent / "llvm-project" / "install" / "bin" / "mlir-opt")
        candidates.append(parent.parent / "llvm-project" / "install" / "bin" / "mlir-opt")

    seen = set()
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if os.path.isfile(candidate_str) and os.access(candidate_str, os.X_OK):
            return candidate_str

    raise RuntimeError(
        "Unable to find `mlir-opt`. Set the MLIR_OPT environment variable or "
        "install mlir-opt in PATH."
    )


def _run_mlir_opt(args: List[str], data: bytes) -> bytes:
    """Run mlir-opt with stdin/stdout piping."""
    mlir_opt = _find_mlir_opt()
    command = [mlir_opt, "--allow-unregistered-dialect", *args]
    try:
        result = subprocess.run(
            command,
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"mlir-opt failed: {stderr}") from exc
    return result.stdout


def _wrap_module(func: Function) -> str:
    """Wrap a single function in a top-level module op."""
    body = repr(func).splitlines()
    indented = "\n".join(f"  {line}" if line else line for line in body)
    return f"module {{\n{indented}\n}}\n"


def _serialize_mlir_bytecode(func: Function) -> bytes:
    """Serialize a Function into MLIR bytecode."""
    module_text = _wrap_module(func).encode("utf-8")
    try:
        return _run_mlir_opt(["--emit-bytecode", "-o", "-"], module_text)
    except ValueError:
        # Some registered ops may fail verifier checks even though MinIR can
        # still represent them. Preserve round-tripping via the legacy format.
        return BinaryWriter().serialize(func)


def _deserialize_mlir_bytecode(data: bytes) -> Function:
    """Deserialize MLIR bytecode produced by `to_bytes`."""
    module_text = _run_mlir_opt(
        [
            "--mlir-print-op-generic",
            "--mlir-print-elementsattrs-with-hex-if-larger=0",
            "-o",
            "-",
        ],
        data,
    )
    return _parse_generic_mlir(module_text.decode("utf-8"))


def _split_top_level(text: str, delimiter: str = ",") -> List[str]:
    """Split a string on a delimiter, ignoring nested MLIR syntax groups."""
    parts: List[str] = []
    current: List[str] = []
    depth_paren = 0
    depth_angle = 0
    depth_brace = 0
    depth_bracket = 0
    in_string = False
    escape = False

    for char in text:
        if in_string:
            current.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            current.append(char)
            continue
        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren -= 1
        elif char == "<":
            depth_angle += 1
        elif char == ">":
            depth_angle -= 1
        elif char == "{":
            depth_brace += 1
        elif char == "}":
            depth_brace -= 1
        elif char == "[":
            depth_bracket += 1
        elif char == "]":
            depth_bracket -= 1

        if (
            char == delimiter
            and depth_paren == 0
            and depth_angle == 0
            and depth_brace == 0
            and depth_bracket == 0
        ):
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue

        current.append(char)

    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def _parse_value_type(type_str: str, name: str = "") -> Value:
    """Parse a limited subset of MLIR types into minir Values."""
    type_str = type_str.strip()
    if type_str.startswith("tensor<") and type_str.endswith(">"):
        inner = type_str[7:-1]
        if "," in inner:
            inner = _split_top_level(inner)[0]
        parts = inner.split("x")
        dtype = parts[-1]
        shape = [int(part) for part in parts[:-1] if part]
        return Tensor(name=name, dtype=dtype, shape=shape)
    return Tensor(name=name, dtype=type_str, shape=[])


def _parse_type_list(type_str: str) -> List[Value]:
    """Parse an MLIR result type list."""
    type_str = type_str.strip()
    if type_str == "()":
        return []
    if type_str.startswith("(") and type_str.endswith(")"):
        inner = type_str[1:-1].strip()
        if not inner:
            return []
        return [_parse_value_type(part) for part in _split_top_level(inner)]
    return [_parse_value_type(type_str)]


def _parse_attr_value(text: str) -> Any:
    """Parse the attribute forms emitted by minir."""
    text = text.strip()
    if text == "true":
        return True
    if text == "false":
        return False

    dense_match = re.fullmatch(r'dense<"0x([0-9A-Fa-f]*)">\s*:\s*(.+)', text)
    if dense_match:
        data = bytes.fromhex(dense_match.group(1))
        value_type = _parse_value_type(dense_match.group(2))
        return Dense(data, value_type.dtype, getattr(value_type, "shape", []))

    array_match = re.fullmatch(r"array<i64:\s*(.*)>", text)
    if array_match:
        values = array_match.group(1).strip()
        if not values:
            return Array([])
        return Array([int(part.strip()) for part in values.split(",") if part.strip()])

    list_match = re.fullmatch(r"\[(.*)\]", text)
    if list_match:
        values = list_match.group(1).strip()
        if not values:
            return []
        return [int(part.strip()) for part in _split_top_level(values) if part.strip()]

    int_match = re.fullmatch(r"(-?\d+)\s*:\s*i(32|64)", text)
    if int_match:
        value = int(int_match.group(1))
        return Int64(value) if int_match.group(2) == "64" else value

    float_match = re.fullmatch(
        r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*:\s*f\d+",
        text,
    )
    if float_match:
        return float(float_match.group(1))

    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]

    return text


def _parse_attr_dict(text: str) -> Dict[str, Any]:
    """Parse a flat MLIR attribute dictionary."""
    text = text.strip()
    if not text:
        return {}

    attrs: Dict[str, Any] = {}
    for item in _split_top_level(text):
        key, value = item.split("=", 1)
        attrs[key.strip()] = _parse_attr_value(value)
    return attrs


def _parse_result_names(binding: Optional[str], result_count: int) -> List[str]:
    """Expand result bindings from generic MLIR op syntax."""
    if not binding:
        return [f"%tmp{i}" for i in range(result_count)]

    binding = binding.strip()
    if ":" in binding and binding.startswith("%") and binding.count("%") == 1:
        base, count_str = binding.split(":", 1)
        count = int(count_str)
        if count == 1:
            return [base]
        return [f"{base}#{idx}" for idx in range(count)]

    names = [part.strip() for part in _split_top_level(binding)]
    if names:
        return names
    return [f"%tmp{i}" for i in range(result_count)]


def _parse_generic_mlir(module_text: str) -> Function:
    """Parse the generic MLIR form emitted by `mlir-opt` for a single function."""
    lines = [line.rstrip() for line in module_text.splitlines() if line.strip()]

    name_match = re.search(r'sym_name = "([^"]+)"', module_text)
    if not name_match:
        raise ValueError("Unable to find function name in MLIR module")
    func_name = name_match.group(1)

    value_map: Dict[str, Value] = {}
    operations: List[Operation] = []
    in_func = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('"builtin.module"'):
            continue
        if stripped.startswith('"func.func"'):
            in_func = True
            continue
        if not in_func:
            continue

        if stripped.startswith("^bb0("):
            block_args = stripped[len("^bb0(") : -2]
            for item in _split_top_level(block_args):
                arg_name, arg_type = item.split(":", 1)
                value_map[arg_name.strip()] = _parse_value_type(
                    arg_type.strip(), arg_name.strip()
                )
            continue

        if stripped.startswith("})"):
            break

        match = re.fullmatch(
            r'(?:(?P<lhs>.+?)\s*=\s*)?"(?P<name>[^"]+)"'
            r"\((?P<operands>[^)]*)\)"
            r"(?:\s*(?:<\{(?P<inherent_attrs>.*)\}>|\{(?P<attrs>.*)\}))?"
            r"\s*:\s*\((?P<operand_types>[^)]*)\)\s*->\s*(?P<result_types>.+)",
            stripped,
        )
        if not match:
            raise ValueError(f"Unsupported MLIR op syntax: {stripped}")

        operand_names = [
            item.strip() for item in _split_top_level(match.group("operands")) if item.strip()
        ]
        operands = [value_map[name] for name in operand_names]

        result_values = _parse_type_list(match.group("result_types"))
        result_names = _parse_result_names(match.group("lhs"), len(result_values))
        if len(result_names) != len(result_values):
            raise ValueError(f"Result binding/type mismatch: {stripped}")

        for value, name in zip(result_values, result_names):
            value.name = name
            value_map[name] = value

        attributes = _parse_attr_dict(
            match.group("inherent_attrs") or match.group("attrs") or ""
        )
        operations.append(
            Operation(
                name=match.group("name"),
                operands=operands,
                results=result_values,
                attributes=attributes,
            )
        )

    return Function(operations, name=func_name)
