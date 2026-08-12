# vyre Wire Format (VIR0)

Applies to Vyre 0.7.2.

This document specifies the binary serialization of a `vyre::ir::Program`. The
format is called VIR0. `Program::to_wire` produces it and `Program::from_wire`
consumes it.

Every constant and tag below is taken from the encoder in
`vyre-foundation/src/serial/wire/`. If you change a tag there, change it here in
the same commit.

## Design axioms

1. **Deterministic.** Two encoders on the same program produce byte-identical
   output. This is what makes content-addressed caching and cross-machine
   certificate comparison possible.
2. **Extensible.** Downstream crates can add IR constructs without editing
   `vyre`. Extensions encode as a kind string plus an opaque payload, and a
   decoder that does not know the kind preserves and reports it.
3. **Versioned.** Every encoded program carries a format version, so a decoder
   can tell you that a payload is newer than it understands instead of failing
   with an arbitrary parse error.
4. **Round-trip complete.** Every program that passes `validate_program`
   survives `to_wire` then `from_wire` unchanged.

## Byte layout

A VIR0 blob is a fixed 40-byte header followed by the body:

```
offset  size  field
0       4     magic, the ASCII bytes "VYRE"
4       2     format version, u16 little-endian
6       2     flags, u16 little-endian
8       32    BLAKE3 digest of the body
40      ...   body
```

The magic is `b"VYRE"`, not `VIR0`. VIR0 is the name of the format; the four
bytes on the wire spell `VYRE`. The constants live in
`vyre-foundation/src/serial/wire/framing/magic.rs`:

| Constant | Value | Meaning |
|---|---|---|
| `MAGIC` | `b"VYRE"` | Envelope tag |
| `WIRE_FORMAT_VERSION` | `5` | Version this encoder writes |
| `MIN_SUPPORTED_WIRE_FORMAT_VERSION` | `4` | Oldest version this decoder reads |

`wire_format_version_is_supported(v)` is the single place the accepted range is
decided. It returns true for `4..=5`.

Version history, from the same file: rev 5 adds expression tag 22
(`Expr::BufferRef`). Rev 4 preserves program-level composition-safety flags in
metadata, so parser and stateful kernels do not become fusible after a round
trip. Rev 3 introduces structured version-mismatch errors and a reserved
dialect-manifest section after the header. Rev 2 was never released, so versions
go 1 to 3 directly.

### Flags

The flag word is a bitset defined in `vyre-foundation/src/serial/wire/tags.rs`:

| Bit | Value | Name | Meaning |
|---|---|---|---|
| 0 | `1` | `FLAG_COMPRESSED` | Payload is compressed |
| 1 | `2` | `FLAG_SEALED` | Payload is sealed |
| 2 | `4` | `FLAG_OPAQUE_ENDIAN_FIXED` | Every opaque payload is endian-fixed |

`to_wire` currently writes exactly `FLAG_OPAQUE_ENDIAN_FIXED`, so the flag word
on a freshly encoded program is `4`.

### Digest

Bytes 8 through 39 are `blake3::hash(body)`, where `body` is everything from
offset 40 to the end. The digest covers the body only, not the header, so it does
not depend on itself.

### Body

The body is three sections in this order, written by
`vyre-foundation/src/serial/wire/encode/to_wire.rs`:

1. The nodes section, which carries the program metadata header, the buffer
   table, and the node tree.
2. The memory-regions section, derived from the buffer table.
3. The output set.

The nodes section opens with the metadata header, which begins with the
length-prefixed string `vyre.program.metadata`.

You can confirm the whole layout on any program:

```rust
let wire = program.to_wire()?;
assert_eq!(&wire[0..4], b"VYRE");
assert_eq!(u16::from_le_bytes([wire[4], wire[5]]), 5);
assert_eq!(u16::from_le_bytes([wire[6], wire[7]]), 4);
assert_eq!(&wire[8..40], blake3::hash(&wire[40..]).as_bytes());
```

### Metadata header

The metadata header encodes program-level facts:

- `entry_op_id`: `Option<String>`, the op this program implements, if any.
- `workgroup_size`: `[u32; 3]`.
- `buffers`: `Vec<BufferDecl>`, each with name, binding, access, element type,
  count, output flag, optional output byte range, and memory hints.
- `metadata`: a string-to-bytes map for attached data such as provenance,
  hashes, and certificates.

Each field uses a one-byte discriminant followed by its payload. A discriminant
in `0x00..=0x7F` is a core variant. `0x80..=0xFF` is an extension variant.

## Expr and Node tree

Nodes are serialized depth-first. Every `Node` and every `Expr` begins with a
one-byte tag. Tags in `0x00..=0x7F` are core variants; `0x80` is the extension
escape.

The tag values are not in alphabetical or definition order, because tags are
append-only: a variant added later takes the next free number rather than
shifting its neighbours. Read the tables, do not guess.

### Expr tags

Source: `vyre-foundation/src/serial/wire/encode/put_expr.rs`.

| Tag | Variant | Payload |
|---:|---|---|
| 0 | `LitU32` | `u32` value |
| 1 | `LitI32` | `i32` value, written as its little-endian `u32` bits |
| 2 | `LitBool` | `u8`, 0 or 1 |
| 3 | `Var` | string name |
| 4 | `Load` | string buffer, `Expr` index |
| 5 | `BufLen` | string buffer |
| 6 | `InvocationId` | `u8` axis |
| 7 | `WorkgroupId` | `u8` axis |
| 8 | `LocalId` | `u8` axis |
| 9 | `BinOp` | `u8` op tag, `Expr` left, `Expr` right |
| 10 | `UnOp` | `u8` op tag, `Expr` operand |
| 11 | `Call` | string op_id, `u32` argc, then argc `Expr` values |
| 12 | `Select` | `Expr` cond, `Expr` if_true, `Expr` if_false |
| 13 | `Cast` | `DataType` target, `Expr` value |
| 14 | `Atomic` | `u8` op tag, string buffer, `Expr` index, `u8` has_expected, optional `Expr` expected, `Expr` value, ordering |
| 15 | `LitF32` | `u32`, the canonicalized IEEE 754 bit pattern |
| 16 | `Fma` | `Expr` a, `Expr` b, `Expr` c |
| 17 | `SubgroupReduce` | `u8` op tag, `Expr` value |
| 18 | `SubgroupShuffle` | `Expr` value, `Expr` lane |
| 19 | `SubgroupBallot` | `Expr` cond |
| 20 | `SubgroupLocalId` | none |
| 21 | `SubgroupSize` | none |
| 22 | `BufferRef` | string buffer. Added in format version 5 |
| `0x80` | `Opaque` | string extension kind, length-prefixed payload |

Two details are easy to get wrong. `LitF32` is tag 15, not tag 2, because the
float literal was added after the integer and boolean literals. `Atomic` is tag
14 and `Fma` is tag 16, so they are not adjacent to the arithmetic tags.

Tags 23 through `0x7F` are unallocated core slots.

In `BinOp`, `UnOp`, and `Atomic`, an op tag byte of `0x80` means the operator
itself is an extension, and a `u32` extension id follows.

The `Opaque` payload carries a **string** extension kind, not a `u32`
extension id.

### Node tags

Source: `vyre-foundation/src/serial/wire/encode/put_node.rs`.

| Tag | Variant | Payload |
|---:|---|---|
| 0 | `Let` | string name, `Expr` value |
| 1 | `Assign` | string name, `Expr` value |
| 2 | `Store` | string buffer, `Expr` index, `Expr` value |
| 3 | `If` | `Expr` cond, node list then, node list otherwise |
| 4 | `Loop` | string var, `Expr` from, `Expr` to, node list body |
| 5 | `Return` | none |
| 6 | `Block` | node list |
| 7 | `Barrier` | `u8` ordering tag |
| 8 | `IndirectDispatch` | string count_buffer, `u32` count_offset little-endian |
| 9 | `AsyncLoad` | string source, string destination, `Expr` offset, and the remaining async fields |
| 10 | `AsyncWait` | string tag |
| 11 | `Region` | string generator, `u8` has_source_region, optional string region name, node list body |
| 12 | `AsyncStore` | string source, string destination, `Expr` offset, and the remaining async fields |
| 13 | `Trap` | `Expr` address, string tag |
| 14 | `Resume` | string tag |
| 15 | `AllReduce` | string buffer, `u8` op tag, `u32` group |
| 16 | `AllGather` | string input, string output, `u32` group |
| 17 | `ReduceScatter` | string input, string output, `u8` op tag, `u32` group |
| 18 | `Broadcast` | string buffer, `u32` root, `u32` group |
| `0x80` | `Opaque` | string extension kind, length-prefixed payload |

There is no `For` node. Counted iteration is `Node::Loop { var, from, to, body }`,
which is tag 4. `Return` is tag 5, not tag 8.

`Node::forever` is a constructor, not an encoded variant. It builds a
`Node::Loop` whose bound is `u32::MAX`, so it serializes as tag 4 like any other
loop.

### Operator sub-tags

`BinOp`, `UnOp`, and `AtomicOp` each encode as one `u8`. These tables are
append-only: a new variant takes the next free code, and adding one bumps the
format version. Decoder dispatch is owned by
`vyre-foundation/src/serial/wire/tags/op_tag_decode.rs`.

`AtomicOp`:

| Tag | Variant | | Tag | Variant |
|---:|---|---|---:|---|
| `0x01` | `Add` | | `0x07` | `Exchange` |
| `0x02` | `Or` | | `0x08` | `CompareExchange` |
| `0x03` | `And` | | `0x09` | `CompareExchangeWeak` |
| `0x04` | `Xor` | | `0x0A` | `FetchNand` |
| `0x05` | `Min` | | `0x0B` | `LruUpdate` |
| `0x06` | `Max` | | | |

`UnOp`:

| Tag | Variant | | Tag | Variant |
|---:|---|---|---:|---|
| `0x01` | `Negate` | | `0x13` | `IsFinite` |
| `0x02` | `BitNot` | | `0x14` | `Exp` |
| `0x03` | `LogicalNot` | | `0x15` | `Log` |
| `0x04` | `Popcount` | | `0x16` | `Log2` |
| `0x05` | `Clz` | | `0x17` | `Exp2` |
| `0x06` | `Ctz` | | `0x18` | `Tan` |
| `0x07` | `ReverseBits` | | `0x19` | `Acos` |
| `0x08` | `Cos` | | `0x1A` | `Asin` |
| `0x09` | `Sin` | | `0x1B` | `Atan` |
| `0x0A` | `Abs` | | `0x1C` | `Tanh` |
| `0x0B` | `Sqrt` | | `0x1D` | `Sinh` |
| `0x0C` | `Floor` | | `0x1E` | `Cosh` |
| `0x0D` | `Ceil` | | `0x1F` | `InverseSqrt` |
| `0x0E` | `Round` | | `0x20` | `Unpack4Low` |
| `0x0F` | `Trunc` | | `0x21` | `Unpack4High` |
| `0x10` | `Sign` | | `0x22` | `Unpack8Low` |
| `0x11` | `IsNan` | | `0x23` | `Unpack8High` |
| `0x12` | `IsInf` | | `0x24` | `Reciprocal` |

`BinOp`:

| Range | Variants |
|-------|----------|
| `0x01`-`0x05` | `Add`, `Sub`, `Mul`, `Div`, `Mod` |
| `0x06`-`0x0A` | `BitAnd`, `BitOr`, `BitXor`, `Shl`, `Shr` |
| `0x0B`-`0x12` | `Eq`, `Ne`, `Lt`, `Gt`, `Le`, `Ge`, `And`, `Or` |
| `0x13`-`0x18` | `AbsDiff`, `Min`, `Max`, `SaturatingAdd`, `SaturatingSub`, `SaturatingMul` |
| `0x19`-`0x1C` | `Shuffle`, `Ballot`, `WaveReduce`, `WaveBroadcast` |
| `0x1D`-`0x20` | `RotateLeft`, `RotateRight`, `WrappingAdd`, `WrappingSub` |
| `0x21` | `MulHigh`, the upper 32 bits of a widening `u32` multiply |

## DataType encoding

Source: `vyre-foundation/src/serial/wire/tags/data_type_tag.rs`. Note that
`DataType` tags start at `0x01`, not `0x00`.

| Tag | Type | | Tag | Type |
|---:|---|---|---:|---|
| `0x01` | `U32` | | `0x11` | `I16` |
| `0x02` | `I32` | | `0x12` | `I64` |
| `0x03` | `U64` | | `0x13` | `Handle` |
| `0x04` | `Vec2U32` | | `0x14` | `Vec` |
| `0x05` | `Vec4U32` | | `0x15` | `TensorShaped` |
| `0x06` | `Bool` | | `0x16` | `SparseCsr` |
| `0x07` | `Bytes` | | `0x17` | `SparseCoo` |
| `0x08` | `Array` | | `0x18` | `SparseBsr` |
| `0x09` | `F16` | | `0x19` | `F8E4M3` |
| `0x0A` | `BF16` | | `0x1A` | `F8E5M2` |
| `0x0B` | `F32` | | `0x1B` | `I4` |
| `0x0C` | `F64` | | `0x1C` | `FP4` |
| `0x0D` | `Tensor` | | `0x1D` | `NF4` |
| `0x0E` | `U8` | | `0x1E` | `DeviceMesh` |
| `0x0F` | `U16` | | `0x1F` | `Quantized` |
| `0x10` | `I8` | | `0x80` | `Opaque` |

`Array` (tag `0x08`) is followed by its `element_size` as a little-endian `u32`.
The composite types (`Vec`, `TensorShaped`, the sparse layouts, `DeviceMesh`,
`Quantized`) carry further payload; see `put_data_type` for the exact shape of
each. `Opaque` (tag `0x80`) is followed by a `u32` extension id.

## Ident encoding

Idents are length-prefixed UTF-8: `u32 length | bytes`. No null termination. No interior NUL bytes (validator rejects them).

## Extension extensibility

The `Opaque` tags (`0x80` on both Expr and Node) encode:

```
tag:            0x80           (1 byte)
extension_id:   u32 LE         stable extension namespace ID
payload_len:    u32 LE
payload:        payload_len bytes
```

Extension IDs are registered via `inventory::submit! { ExtensionRegistration { id, kind, decoder } }`. A decoder that does not know an extension returns `DecodeError::UnknownExtension { extension_id, kind }` with the ID preserved. The consumer installs an extension crate and re-decodes.

This is the forward-compatibility story: a new IR node is introduced as
an `Opaque` extension in a downstream crate. Old decoders preserve it
(round-trip preserves the extension bytes) but can't introspect. New
decoders that link the extension crate decode it to its native form.

Extension IDs in the range `[0x0000_0000, 0x7FFF_FFFF]` are reserved for vendor-assigned core extensions. `[0x8000_0000, 0xFFFF_FFFF]` is community-assigned (registered at `https://vyre.dev/registry/extensions/` when that exists).

## Accepted versions

The encoder writes `WIRE_FORMAT_VERSION`, currently 6. The decoder accepts
versions from `MIN_SUPPORTED_WIRE_FORMAT_VERSION`, currently 4, through the
current version. Both constants live in
`vyre_foundation::serial::wire::framing`.

The decoder validates the magic, version, flags, digest, lengths, nesting
limits, and discriminants before constructing a `Program`. An unsupported
version or unknown discriminant is rejected instead of being reinterpreted.

## Versioning policy

- Patch releases do not change wire bytes.
- Any encoding change increments `WIRE_FORMAT_VERSION`.
- Decoders accept only the explicit inclusive version range.
- A value outside that range must be reserialized by a compatible producer.

Every discriminant-table change updates the framing version and adds canonical
round-trip plus stale-version rejection coverage.

## Round-trip invariant

For every program `p` that passes `validate_program`:

```
let bytes = to_wire(&p)?;
let p2 = from_wire(&bytes)?;
assert_eq!(p, p2);              // PartialEq
assert_eq!(to_wire(&p2)?, bytes); // stability under re-encoding
```

The `wire_roundtrip` test suite enforces this invariant on every KAT program and on fuzz-generated programs (proptest) with shrinking.

## Certificate compatibility

The conform certificate includes `wire_format_version` and a `blake3` of the program's canonical wire bytes. Two certificates with matching `wire_format_version + program_hash + witness_set_hash + backend_id` identify exchangeable artifacts.

## Binding conformance

A conformant non-Rust binding MUST:

1. Parse the header and reject any `version` it does not support.
2. Preserve unknown extension payloads (`0x80` tags) verbatim when re-encoding.
3. Surface unknown extensions as structured errors, never silent drops.
4. Produce byte-identical output when re-encoding a program it consumed.
