"""
Binary format converter for minir IR based on NVIDIA Tile IR bytecode format.

This module provides serialization and deserialization of minir Function objects
to/from a binary format inspired by NVIDIA Tile IR bytecode specification.

File Structure:
    - Magic number: \x7fMinIR\x00\x00 (8 bytes)
    - Version: major(1), minor(1), tag(2) - little endian
    - Sections: String, Type, Constant, Function Table

Reference: https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html
"""

import struct
from typing import List, Dict, Any, Optional, Tuple, BinaryIO, Union
from io import BytesIO
from minir.ir import Function, Operation, Value, Tensor, Dense, Array, Int64


# Magic number for minir bytecode (8 bytes)
MAGIC_NUMBER = b"\x7fMinIR\x00\x00"

# Version
VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_TAG = 0

# Section IDs
SECTION_STRING = 0x01
SECTION_FUNCTION_TABLE = 0x02
SECTION_DEBUG = 0x03
SECTION_CONSTANT = 0x04
SECTION_TYPE = 0x05
SECTION_GLOBAL = 0x06
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
TYPE_FUNCTION = 0x10

# Attribute tags
ATTR_INTEGER = 0x01
ATTR_FLOAT = 0x02
ATTR_BOOL = 0x03
ATTR_TYPE = 0x04
ATTR_STRING = 0x05
ATTR_ARRAY = 0x06
ATTR_DENSE = 0x07
ATTR_DICT = 0x0A

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
        buffer.write(MAGIC_NUMBER)
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

    def _read_header(self) -> Tuple[int, int, int]:
        """Read and validate the header. Returns (major, minor, tag)."""
        magic = self._read_bytes(8)
        if magic != MAGIC_NUMBER:
            raise ValueError(f"Invalid magic number: {magic}")

        major = self._read_bytes(1)[0]
        minor = self._read_bytes(1)[0]
        tag = struct.unpack("<H", self._read_bytes(2))[0]

        return major, minor, tag

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
                alignment = self._read_varint()
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
        major, minor, tag = self._read_header()

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
    writer = BinaryWriter()
    data = writer.serialize(func)
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
    reader = BinaryReader(data)
    return reader.deserialize()
