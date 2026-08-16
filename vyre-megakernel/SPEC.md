# vyre-megakernel

Layer `compiler-boundary`. Owner `megakernel-compiler`.

## Owns

Explore and select legal whole-ProgramGraph fusion schedules under explicit
SearchBudget bounds, emit a megakernel Artifact and TargetPayloads, and never
claim a measured winner that no clock produced. Does not own admission,
execution, or lifecycle policy.

The chapter is [crate boundaries](../docs/architecture/crates.md#vyre-megakernel).

## Must never contain

Device admission, submission, queues, residency, recovery. Those consume the
artifact and must not alter its identity, because identity is what makes two
routes comparable and a cache sound. Also not here: any claim of a measured
winner that no clock produced.

## What crosses its edges

Out of this crate, into:

- `vyre-foundation` over the `foundation-ir` seam, public: typed IR, graph,
  diagnostics, validation, and semantic optimization contracts. Built when:
  always.
- `vyre-lower` over the `lowering` seam, private: single verified
  selected-module representation lowering. Built when: always.

Into this crate, from:

- `vyre` over the `megakernel-compiler` seam, private.
- `vyre-aot` over the `megakernel-compiler` seam, public.
- `vyre-conform` over the `megakernel-compiler` seam, private.
- `vyre-driver` over the `megakernel-compiler` seam, public.
- `vyre-driver-cuda` over the `megakernel-compiler` seam, private.
- `vyre-driver-metal` over the `megakernel-compiler` seam, private.
- `vyre-driver-spirv` over the `megakernel-compiler` seam, private.
- `vyre-driver-wgpu` over the `megakernel-compiler` seam, private.
- `vyre-runtime` over the `megakernel-compiler` seam, public.
- `xtask-registry` over the `megakernel-compiler` seam, private.

## Direction that may not reverse

`vyre-foundation`, `vyre-lower` must never depend on `vyre-megakernel`. The
edge is one way: a cycle back into this crate makes the two crates one crate
that cannot be built, reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares no feature beyond `default`, so every build of it is the
  same build.
- The public surface is recorded in `docs/public-api/vyre-megakernel.txt`. An
  item added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`public-api-snapshot`, `public-api-paths`.
