# Value contracts

```rust
ValueContract {
    dtype: DataType::F32,
    shape: vec![ShapeDim::Symbol("batch".into()), ShapeDim::Known(768)],
    access: BufferAccess::ReadOnly,
    lifetime: ValueLifetime::Constant,
}
```

A `ValueContract` is the whole semantic description of one connected graph
value: element representation, ordered dimensions, the access a bound
program buffer requires, and a lifetime class. A graph value carries one
contract; every consumer port that binds it declares the contract it
expects, and the compiler compares the two.

## Shape

`ShapeDim::Known(u64)` is an exact element extent. `ShapeDim::Symbol(name)`
is a configuration symbol such as `batch`, `sequence` or `hidden`, bound by
graph configuration rather than by the topology. A shape is a `Vec` in
declaration order; rank is its length.

At the logical compiler boundary, a positive known dimension becomes a static
logical extent. A symbolic dimension becomes a graph-value extent containing
the value identity, axis, symbol and compile-request bound. A zero extent marks
an unresolved runtime buffer and rejects compilation until caller evidence
specializes it. The enclosing logical region records its row-major index map
and layout, reduction axes, aliases, effects, producer dependencies and
overflow-checked positive point bound.

## Access

| Variant | Buffer |
|---|---|
| `ReadOnly` | read-only storage |
| `ReadWrite` | read-write storage |
| `Uniform` | small, read-only, fast path |
| `WriteOnly` | write-only storage |
| `Workgroup` | workgroup-local shared memory |

`BufferAccess` is `#[non_exhaustive]`: match it with a wildcard arm.

## Lifetime

| Variant | Meaning |
|---|---|
| `Constant` | immutable data shared by every invocation |
| `Invocation` | temporary, valid for one invocation |
| `Retained` | mutable, retained across submissions |
| `Output` | caller-visible graph result |

`Retained` is what makes a stateful graph expressible without a host loop.
A `GraphOutput` may name `retained_successor_of`, which is the prior
retained value the output replaces.

## Element type

`DataType` is the frozen element vocabulary in `vyre-spec`. It is wide and
it grows, so the authority is machine-readable rather than a table here:
`docs/generated/OP_SCHEMA.json` carries the current set together with the
operations that accept each. `DataType::TensorShaped` carries a
rank-limited shape, and `DataType::DeviceMesh` carries mesh axes.

## Identity

`GraphValueId(u32)` and `GraphNodeId(u32)` are graph-local. They are dense
indices into one graph, not stable identifiers across graphs, and nothing
outside a single `ProgramGraph` should hold one.

`ProgramGraph` is `Clone`. A clone is a distinct graph with the same dense
indices, so an identity taken from one graph stays valid against its clone and
against nothing else.
