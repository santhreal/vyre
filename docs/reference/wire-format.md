# Program wire format

```rust
let bytes = program.to_wire()?;
let decoded = Program::from_wire(&bytes)?;
assert_eq!(decoded, program);
```

`Program::to_wire` and `Program::from_wire` are the stable binary IR
encoding. Encoding is deterministic: encoding a decoded program returns the
same bytes, which is what lets a digest over the encoding identify a
program.

## Envelope

The blob opens with the four magic bytes `VYRE` followed by a
little-endian `u16` schema version. `from_wire` validates the magic, then
the version, before it decodes any payload, so a version mismatch reports
itself instead of surfacing as an arbitrary parse failure further in.

| Constant | Value |
|---|---|
| `MAGIC` | `b"VYRE"` |
| `WIRE_FORMAT_VERSION` | 8 |
| `MIN_SUPPORTED_WIRE_FORMAT_VERSION` | 4 |

This decoder reads versions 4 through 8 and writes 8. Version 2 was never
released.

## Bounds are checked before allocation

Every variable-length payload is length-prefixed with a `u32`, and every
length is compared against a ceiling before anything is allocated. A
hostile blob is rejected on the bound, not on the allocator.

| Bound | Value |
|---|---|
| `MAX_PROGRAM_BYTES` | 64 MiB |
| `MAX_BUFFERS` | 16384 |
| `MAX_NODES` | 1000000 per node list |
| `MAX_ARGS` | 4096 per call expression |
| `MAX_STRING_LEN` | 1 MiB |
| `MAX_OPAQUE_PAYLOAD_LEN` | `MAX_ARGS * 1024` |
| `MAX_TENSOR_RANK` | 4096 |
| `MAX_MESH_AXES` | 4096 |
| `MAX_SHAPE_PREDICATE_DEPTH` | 32 |
| `MAX_DECODE_DEPTH` | 64 |

`MAX_NODES` applies to each nested node list as it is decoded, not once to
the program.

`MAX_DECODE_DEPTH` is applied to a single recursion counter that both node
decoding and expression decoding increment on entry and decrement on exit.
A blob cannot evade the cap by alternating statement nesting with
expression nesting, because both spend the same budget, and the depth is
rejected before the stack frame is pushed.

Lengths are written as `u32` rather than `usize` so the blob does not
depend on the pointer width of the host that produced it.

## Tile values and nodes

Version 7 introduces first-class tile values and dedicated tile nodes (`TileLoad`,
`TileStore`, `TileMatmul`, `TileReduce`, `TileElementwise`, `TileDecl`). Tile
extents, layout swizzle permutation vectors, origin vectors, and elementwise
input lists are bounds-checked symmetrically in the encoder and decoder against
`MAX_TENSOR_RANK` and `MAX_ARGS` before serialization or allocation.

## Logical execution markers

Version 8 adds schedule-free logical domain, tile, and within-tile identities
plus logical barriers. Selected-schedule lowering replaces every logical marker
with its physical invocation, workgroup, local, or barrier form before
descriptor construction. A logical marker at physical lowering is rejected.

## What is proved

`vyre/tests/wire_v1_round_trip.rs` covers the round trip, encoder
determinism and a non-empty encoding.
`vyre-foundation/tests/terminal_wire_round_trip.rs` is the exhaustive
property coverage over IR variants. `vyre/tests/wire_malformed_adversarial.rs`
covers rejection. The fuzz corpus under
`vyre-foundation/fuzz/corpus/program_wire/` is tracked and replayed.

A public enum variant with no registered stable wire tag is an encode
error, not a silent omission: adding an IR variant without a tag fails at
encode time.
