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

A `LaunchGeometry` is built by `vyre-megakernel` from three inputs: the
operation's requirements, the widths the authenticated target profile admits,
and the program's declared element count.

```
LaunchGeometry {
    workgroup: [u32; 3],
    grid: [u32; 3],
    elements_per_invocation: u32,
    pipeline_stages: u32,
    shared_bytes: u32,
}
```

Admissibility is a function of the profile, so the same program on two devices
admits two width sets and neither is a portable compromise. A backend states
which widths are legal and never places them in preference order: a second
order is a second cost model.

### Where the code lives

`vyre-foundation` defines `GeometryRequirements` and `LaunchGeometry`. It
contains no device names, no instruction names, and no numbers taken from a
device, and it selects nothing.

Each concrete backend crate reports its own numbers as facts: maximum
invocations per workgroup, shared memory per workgroup, register file per
invocation, subgroup size, asynchronous copy depth, and whatever dual-issue or
matrix-instruction constraint the target has.
`vyre_driver::DeviceProfile::admissible_workgroup_widths` returns every
workgroup width the profile admits for one requirement set, ascending, and an
empty list when no width satisfies it. A shared crate never learns a concrete
limit, and a backend never raises a profile limit to admit a geometry.

`vyre-megakernel` is the single schedule-selection owner. It receives admitted
widths as facts, builds the candidate geometries, and orders them under the
compile objective. The `schedule-ownership` gate rejects a second route.

`vyre-libs` and `vyre-primitives` declare requirements. A cooperative block
whose width is load-bearing for the algorithm is a semantic width, declared
through the one target-neutral portable extent rather than a crate-local
constant.

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

Four guard shapes prove that bound: the index itself, a local proven equal to
it, a predicate bound to a local, and a comparison against a sum carrying the
index as an addend. The last is the chunked form, where one invocation handles
several cells and the guard reads `chunk * lanes + index`; every addend is a
`u32`, so a bound on the sum is a bound on the index.

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

Geometry is a ranked dimension in `vyre-megakernel`. Candidates are built from
the widths the profile admits and ordered by the compile objective; measurement
decides between the leading candidates when the search budget allows it. A
backend contributes legality and measurement, never an order.

A candidate that measures slower than the objective predicted is recorded,
because a cost model whose ranking never disagrees with the device is a cost
model nobody has validated.

## Migration

Every operation that names a geometry constant today is converted. The
conversion is mechanical per operation and its correctness is the parity suite:
a converted operation must produce byte-identical reference output, and its
emitted kernel must stay byte-exact on each backend within the documented window
for its element type.

`vyre_foundation::ir::PORTABLE_WORKGROUP_INVOCATIONS` is the one target-neutral
extent an operation may read when its cooperative block width is semantics. A
crate-local geometry constant is rejected by a gate; a second portable floor
would be a second definition of a width that has one.

## Acceptance criteria

- `GeometryRequirements` and `LaunchGeometry` exist in `vyre-foundation` with no
  concrete device number in either of them and no selection API beside them.
- Each backend crate reports admitted widths from its authenticated profile, and
  the numbers appear only there.
- No operation in `vyre-libs` or `vyre-primitives` names an invocation count, a
  tile size, an elements-per-invocation constant, or a stage count. A gate
  refuses one, proved red by reintroducing a workgroup constant into an
  operation.
- `vyre-megakernel` orders the admitted candidates and the plan search measures
  more than one, demonstrated by a recorded search whose winner is not the first
  ranked candidate for at least one operation.
- `multi_block_prefix_scan` runs at the width its target admits: 1024 where the
  profile allows it, 256 where it does not, from one program with no constant on
  either side. The measured dispatch time at both widths is recorded, and if the
  wider block is not faster the ranking says so rather than the source.
- No operation defines a crate-local geometry constant, and the single portable
  extent has exactly one definition.
- Reference parity is unchanged for every converted operation, and every
  backend's emitted-kernel golden that moves is regenerated by this change with
  the reason stated.
