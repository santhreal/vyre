# Launch geometry is a lowering decision

Status: implemented.

## What is wrong

An operation declares its own launch geometry when it builds its program, before
any backend exists. `multi_block_prefix_scan` declared a 1024-invocation
workgroup as a constant in `vyre-libs`. On a target whose profile admits 256 the
program was refused with `target workgroup extent 1024 exceeds profile limit
256`, and the repair available at that layer was to declare 256 instead, which
narrows the cooperative block on every target including the ones that admit
1024.

Both numbers are wrong in the same way. A constant in library code cannot be
right for two targets, and the choice does not belong to the operation at all.

Everything that follows from geometry is decided in the same wrong place:
invocations per workgroup, tile size, elements per invocation, register budget,
occupancy, shared-memory footprint, and how many stages an asynchronous copy
pipeline carries. The compiler receives a program whose entire tuning space has
already been collapsed to one point by an author who could not see the device.

The plan search compounds it. Candidate plans are compiled and measured on the
device, so search can only explore what the program leaves open. When the
program fixes geometry, search ranks one candidate.

## The change

An operation declares requirements and invariants. A backend decides numbers.

### What an operation may declare

```
GeometryRequirements {
    cooperative_width: CooperativeWidth,
    min_shared_bytes: u32,
    per_invocation_elements: ElementPolicy,
    subgroup_uniformity: Uniformity,
}
```

- `CooperativeWidth` is `Agnostic` when the algorithm is correct at any width,
  or `AtLeast(n)` when a cooperative step needs at least `n` invocations to
  exchange data, or `Exactly(n)` when the algorithm is written around one width.
  A scan that reduces in a tree is `Agnostic` above its radix; saying so is what
  lets a backend pick.
- `min_shared_bytes` is what the algorithm needs, not what a device has.
- `ElementPolicy` states whether elements per invocation may be raised by the
  backend, and any divisibility the algorithm requires.
- `Uniformity` states which values must be uniform across a subgroup, so a
  backend that widens a workgroup does not break a subgroup-uniform assumption.

No operation names an invocation count, a tile size, or a stage count. A gate
enforces that.

### What a backend decides

A `LaunchGeometry` is produced by the backend's lowering strategy from three
inputs: the operation's requirements, the authenticated target profile, and the
program's declared element count.

```
LaunchGeometry {
    workgroup: [u32; 3],
    grid: [u32; 3],
    elements_per_invocation: u32,
    pipeline_stages: u32,
    shared_bytes: u32,
}
```

The decision is a function of the profile, so the same program on two devices
lowers to two geometries and neither is a portable compromise. A geometry that
the profile does not admit is a compiler defect, not a runtime refusal, and the
strategy is the only place that can produce one.

### Where the code lives

`vyre-foundation` owns `GeometryRequirements`, `LaunchGeometry`, and a neutral
`GeometryStrategy` trait. It contains no device names, no instruction names, and
no numbers taken from a device.

Each concrete backend crate implements the trait and owns its numbers: maximum
invocations per workgroup, shared memory per workgroup, register file per
invocation, subgroup size, asynchronous copy depth, and whatever dual-issue or
matrix-instruction constraint the target has. A shared crate never learns a
concrete limit, and a backend never raises a profile limit to admit a geometry.

`vyre-libs` and `vyre-primitives` stop declaring geometry and declare
requirements instead.

### Search

Geometry becomes a ranked dimension. The lowering strategy returns candidate
geometries in preference order rather than one answer, and the existing plan
search compiles and measures the top candidates. Ranking is by the strategy's
own model of the target, so the search explores a real space and the measurement
picks the winner rather than confirming the only option.

A candidate that measures slower than the preference order predicted is
recorded, because a strategy whose ranking never disagrees with the device is a
strategy nobody has validated.

## Migration

Every operation that names a geometry constant today is converted. The
conversion is mechanical per operation and its correctness is the parity suite:
a converted operation must produce byte-identical reference output, and its
emitted kernel must stay byte-exact on each backend within the documented window
for its element type.

`PORTABLE_WORKGROUP_INVOCATIONS` disappears at the end of this work. It exists
because there was nowhere else to make the decision. A portable floor is the
right answer only while the compiler cannot choose, and after this change it
would be a second owner of a decision that has one.

## Acceptance criteria

- `GeometryRequirements`, `LaunchGeometry` and the `GeometryStrategy` trait
  exist in `vyre-foundation` with no concrete device number in any of them.
- Each backend crate implements the trait from its authenticated profile, and
  the numbers appear only there.
- No operation in `vyre-libs` or `vyre-primitives` names an invocation count, a
  tile size, an elements-per-invocation constant, or a stage count. A gate
  refuses one, proved red by reintroducing a workgroup constant into an
  operation.
- The strategy returns ranked candidates and the plan search measures more than
  one, demonstrated by a recorded search whose winner is not the first ranked
  candidate for at least one operation.
- `multi_block_prefix_scan` runs at the width its target admits: 1024 where the
  profile allows it, 256 where it does not, from one program with no constant on
  either side. The measured dispatch time at both widths is recorded, and if the
  wider block is not faster the ranking says so rather than the source.
- `PORTABLE_WORKGROUP_INVOCATIONS` is deleted, with every caller converted in
  the same change.
- Reference parity is unchanged for every converted operation, and every
  backend's emitted-kernel golden that moves is regenerated by this change with
  the reason stated.
