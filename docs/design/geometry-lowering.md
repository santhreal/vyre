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
    subgroup_width: CooperativeWidth,
    min_shared_bytes: u32,
    per_invocation_elements: ElementPolicy,
    subgroup_uniformity: Uniformity,
    requires_cooperative_launch: bool,
    memory_ordering: Option<MemoryOrdering>,
}
```

- `cooperative_width` constrains total workgroup width. `Agnostic` permits every
  admitted width, `AtLeast(n)` sets a floor, and `Exactly(n)` fixes a
  semantics-dependent width.
- `subgroup_width` applies the same constraint lattice to subgroup width.
- `min_shared_bytes` states the workgroup scratch required by semantics.
- `per_invocation_elements` states scalar or divisibility constraints.
- `subgroup_uniformity` states the scope that must remain uniform.
- `requires_cooperative_launch` admits schedules with grid-wide barriers.
- `memory_ordering` states the strongest atomic or barrier ordering required.

Constraint composition takes the stronger compatible value in each dimension.
Conflicting exact widths and an exact width below a required minimum produce
stable `GeometryConstraintConflict` variants before schedule search.

An exact width is a semantic invariant, not a preferred target policy. The
registry derives observable lane geometry from the canonical program and
rejects a conflicting declaration.

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

Every operation registration selects an explicitly unconstrained constructor or
records stronger requirements with `with_geometry_requirements`. The registry
derives semantic constraints from the canonical program and composes them with
that decision. `docs/generated/OP_SCHEMA.json` records the effective result for
every linked operation.

### Deriving the span a launch must cover

A program states no launch, so the span one covers is derived from the program.
`vyre-foundation::geometry` publishes that analysis.

`guarded_logical_span` returns the largest axis-0 logical index a program can
affect when every effect it performs is dominated by a constant bound on that
index, and `None` when an effect escapes every such guard, because an unbounded
effect leaves high lanes observable. A program with no effect returns `Some(0)`:
the launch minimum belongs to whoever sizes the launch.

`admitted_logical_span` narrows a resource-derived span to that guarded domain.
A resource span takes the widest declared buffer, which a scatter makes far
larger than the domain its guard admits. The result is at least one.

`launch_covers_full_input_span` states when narrowing is illegal whatever the
guards admit. An atomic, a subgroup collective and a workgroup-scoped buffer all
make the result depend on how many invocations ran rather than only on which
elements each one touched. The last is the shared-memory reduction: every lane of
a group contributes a partial, so a launch narrowed to a one-element output
leaves the rest of the input unreduced.

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
