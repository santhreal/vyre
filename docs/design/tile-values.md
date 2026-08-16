# Tile values in the IR

Status: implemented.

## What is missing

`Expr` produces scalars. `Node::Store` writes one element to a buffer at a
scalar index. A buffer is the only medium two operations can use to exchange
data, including two operations inside the same fused `Node::Region`: a fusion
group binds its members to named artifact resources, and a resource is global
memory.

Three consequences follow.

Every operation boundary that survives to lowering is a global-memory round
trip. Fusion joins control flow and leaves the data in HBM.

No instruction that consumes a register fragment is expressible. Matrix
instructions (`mma`, `wgmma`, `ldmatrix` and their equivalents on other targets)
read operands from a fragment held across the invocations of a subgroup, in a
layout the instruction defines. Scalar loads and stores cannot name that
operand, so lowering cannot select the instruction, and matrix work runs on the
general ALU path.

An algorithm whose whole point is residency cannot be written. Fused attention
keeps a score tile and the running maximum and sum of the online softmax in
registers across three matrix operations. Expressed over buffers it is correct
and bandwidth-bound.

The cost is measurable on the simplest program in the tree. A predicate count
over three `u32` columns reads 12.58 MB and dispatches in 26.7 us, which is
471 GB/s against roughly 1.8 TB/s of device bandwidth. A streaming count with
no reuse should approach the bandwidth limit. It does not, and every program
with reuse to exploit starts from that same floor.

## The addition

A tile is a value. It has an element type, static extents, a layout, and a
residency. It is produced, consumed, and passed between operations without a
buffer.

### Type

```
Tile {
    element: DataType,
    extents: [u32; RANK],
    layout: Layout,
    residency: Residency,
}
```

`Layout` names how the logical index maps to the storage index: row-major,
column-major, and the swizzles a target needs to make shared-memory access
bank-conflict free. A layout is data, not a backend name: it states the
permutation and the swizzle period, and each backend reads it.

`Residency` names where the tile lives: `Register`, `Subgroup`, `Workgroup`, or
`Global`. `Register` is private to one invocation. `Subgroup` is the distributed
fragment a matrix instruction expects, held across the invocations of one
subgroup. `Workgroup` is shared memory. `Global` is a buffer view and is the
only residency that survives the end of a dispatch.

### Nodes

- `TileLoad { tile, buffer, origin, layout }` reads a tile from a buffer,
  applying the layout transform on the way in.
- `TileStore { buffer, origin, tile }` writes a tile back.
- `TileMatmul { acc, a, b }` accumulates `a x b` into `acc`. Operand residency
  must be `Subgroup` and the layouts must be the pair the target's matrix
  instruction admits, which validation checks against the target profile rather
  than assuming.
- `TileReduce { out, tile, op, axis }` reduces along one axis, producing a tile
  of lower rank or a scalar.
- `TileElementwise { out, inputs, body }` applies a scalar body to
  corresponding elements, which is how existing scalar `Expr` reaches tile data
  without a second expression language.

`Node::Region` gains a tile-typed interface. A region declares the tiles it
consumes and produces, and two regions fused into one group pass a tile
directly when the producer's residency and the consumer's expectation agree. A
buffer appears only where a tile must outlive the dispatch.

### Validation

A program carrying tiles is admitted only when the target profile can hold
them. The checks are:

- total `Workgroup` residency in bytes against the profile's shared-memory
  limit,
- live `Register` and `Subgroup` residency against the profile's register
  budget per invocation, so an occupancy collapse is a refusal rather than a
  slow kernel,
- every `TileMatmul` operand shape and layout against the instruction shapes
  the profile declares,
- a tile whose residency is `Subgroup` against the profile's subgroup size.

Each refusal names the limit, the measured requirement, and the operation.

### Reference parity

`vyre-reference` executes a tile as a loop nest over its extents with the
layout applied as an index function. A tile program and the scalar program it
replaces must produce byte-identical output on the reference, which is what
keeps the existing parity claim true through this change.

### Wire format

`Program` gains tile-typed values and four node kinds, so the wire schema
version rises by one and the decoder refuses the previous version by version
rather than by malformed payload. Existing scalar programs encode unchanged and
their hashes do not move.

## What this does not do

It does not choose tile sizes, workgroup extents, pipeline depth, or stage
count. Those are lowering decisions and belong to the companion specification
on geometry. This work makes the decision expressible; it does not make it.

It does not add a scheduling language. Warp specialization and producer-consumer
pipelines over tiles are a later addition on top of this one.

It does not delete the buffer path. A program with no tiles is still a valid
program and still lowers exactly as it does today.

## Acceptance criteria

- The four node kinds and the tile type exist in `vyre-foundation`, with the
  validation rules above and a diagnostic per refusal that names the profile
  limit it failed.
- `vyre-reference` executes every one of them, and a tile program and its
  scalar equivalent agree byte for byte on the reference for a fixture per node
  kind.
- One real composition is rewritten to prove the interface carries weight:
  fused attention in `vyre-libs` holds its score tile and its online-softmax
  statistics in `Register` and `Subgroup` residency across the two matrix
  operations, with no intermediate buffer between them.
- At least one backend lowers `TileMatmul` to its matrix instruction, and the
  emitted kernel is byte-exact against the reference within the documented
  window for the element type.
- A gate refuses a program whose declared residency exceeds the target
  profile, proved red by injecting a tile that overruns shared memory.
- The wire schema version rises, a stale bundle is refused by version, and a
  scalar program's hash is unchanged.
