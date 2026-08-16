# vyre-foundation

Layer `foundation`. Owner `foundation-ir`.

## Owns

Own typed IR and ProgramGraph contracts, validation, diagnostics,
serialization, semantic operation registration, and backend-neutral
optimization.

The chapter is [crate boundaries](docs/architecture/crates.md#vyre-foundation).

## Must never contain

Application semantics. No operation in this crate knows what a neural network
is. Almost every crate depends on this one, so a domain concept admitted here
is one every crate inherits and none can escape.

## What crosses its edges

Out of this crate, into:

- `vyre-macros` over the `registration-macros` seam, private: compile-time
  registration generation. Built when: always.
- `vyre-spec` over the `specification` seam, public: stable cross-engine
  schemas and operation definitions. Built when: always.

Into this crate, from:

- `vyre` over the `foundation-ir` seam, private.
- `vyre-aot` over the `foundation-ir` seam, public.
- `vyre-bench` over the `foundation-ir` seam, private.
- `vyre-conform` over the `foundation-ir` seam, private.
- `vyre-debug` over the `foundation-ir` seam, public.
- `vyre-driver` over the `foundation-ir` seam, public.
- `vyre-driver-cuda` over the `foundation-ir` seam, public.
- `vyre-driver-metal` over the `foundation-ir` seam, private.
- `vyre-driver-reference` over the `foundation-ir` seam, public.
- `vyre-driver-spirv` over the `foundation-ir` seam, public.
- `vyre-driver-wgpu` over the `foundation-ir` seam, public.
- `vyre-emit-metal` over the `foundation-ir` seam, private.
- `vyre-emit-naga` over the `foundation-ir` seam, public.
- `vyre-emit-ptx` over the `foundation-ir` seam, private.
- `vyre-libs` over the `foundation-ir` seam, public.
- `vyre-lower` over the `foundation-ir` seam, public.
- `vyre-megakernel` over the `foundation-ir` seam, public.
- `vyre-pass-engine` over the `foundation-ir` seam, public.
- `vyre-primitives` over the `foundation-ir` seam, private.
- `vyre-reference` over the `foundation-ir` seam, public.
- `vyre-registry-link` over the `foundation-ir` seam, private.
- `vyre-runtime` over the `foundation-ir` seam, public.
- `vyre-test-support` over the `foundation-ir` seam, private.
- `xtask-evidence` over the `foundation-ir` seam, private.
- `xtask-registry` over the `foundation-ir` seam, private.

## Direction that may not reverse

`vyre-macros`, `vyre-spec` must never depend on `vyre-foundation`. The edge is
one way: a cycle back into this crate makes the two crates one crate that
cannot be built, reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares 1 feature beyond `default`: `serde`. It builds alone.
- The public surface is recorded in `docs/public-api/vyre-foundation.txt`. An
  item added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`public-api-snapshot`, `public-api-paths`, `feature-isolation`,
`feature-matrix`.
