# vyre-alloc-probe

Layer `standalone-tooling`. Owner `benchmarks`.

## Owns

Count heap traffic per thread behind a `GlobalAlloc` wrapper, so an allocation
budget measures the code under test rather than the test schedule. Depends on no
vyre crate, so a harness binary links it at run time and a driver suite links it
as a dev-dependency.

The chapter is [crate boundaries](../docs/architecture/crates.md#vyre-alloc-probe).

## Must never contain

A second allocation counter, and any measurement that is not heap traffic. A
process-wide counter charges one measured region for every allocation the
threads beside it made, which reports the schedule under a budget's name.

## What crosses its edges

This crate depends on no other workspace member.

Into this crate, from:

- `vyre-bench` over the `benchmarks` seam, private.
- `vyre-driver-wgpu` over the `benchmarks` seam, private, as a dev-dependency.

## Direction that may not reverse

`vyre-alloc-probe` takes no workspace dependency. A binary that installs a
global allocator links exactly one, so a crate that reaches back into the
workspace would make the probe unavailable to whichever binary the cycle
excludes.

## Invariants

- Counters are thread-local. A region reports only the traffic its own thread
  charged between the two snapshots.
- Gross allocations and deallocations are reported separately from the net, so
  churn is distinguishable from retention.
- A realloc counts once in `reallocations` and once in each of `allocations`
  and `deallocations`, with the new size charged and the old size released.
- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares no feature beyond `default`, so every build of it is the
  same build.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `test-material-placement`,
`file-size`.
