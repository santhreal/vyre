# vyre-registry-link

Layer `registry-link`. Owner `registry-link`.

## Owns

Own every inventory registry link anchor, report which sources a build links,
and assert that each linked source reached the registry it submits into.

The chapter is [crate boundaries](../docs/architecture/crates.md#vyre-registry-link).

## Must never contain

Anything but linking and the floor.

## What crosses its edges

Out of this crate, into:

- `vyre-driver` over the `backend-contract` seam, private: backend registry
  contracts. Built when: always.
- `vyre-foundation` over the `foundation-ir` seam, private: operation registry
  contracts. Built when: always.
- `vyre-driver-cuda` over the `cuda-driver` seam, private: native accelerator
  backend registration. Built when: always.
- `vyre-driver-metal` over the `metal-driver` seam, private: native Apple
  backend registration. Built when: always.
- `vyre-driver-reference` over the `reference-driver` seam, private: reference
  backend registration. Built when: always.
- `vyre-driver-spirv` over the `spirv-driver` seam, private: SPIR-V backend
  registration. Built when: always.
- `vyre-driver-wgpu` over the `portable-driver` seam, private: portable backend
  registration. Built when: always.
- `vyre-libs` over the `product-libraries` seam, private: product operation
  registrations. Built when: always.
- `vyre-primitives` over the `primitive-library` seam, private: primitive
  operation registrations. Built when: always.

Into this crate, from:

- `vyre-bench` over the `registry-link` seam, private.
- `vyre-conform` over the `registry-link` seam, private.
- `xtask-evidence` over the `registry-link` seam, private.
- `xtask-registry` over the `registry-link` seam, private.

## Direction that may not reverse

`vyre-driver`, `vyre-foundation`, `vyre-driver-cuda`, `vyre-driver-metal`,
`vyre-driver-reference`, `vyre-driver-spirv`, `vyre-driver-wgpu`, `vyre-libs`,
`vyre-primitives` must never depend on `vyre-registry-link`. The edge is one
way: a cycle back into this crate makes the two crates one crate that cannot be
built, reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares 6 features beyond `default`: `cuda`, `metal`,
  `operations`, `reference`, `spirv`, `wgpu`. Each builds alone.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`feature-isolation`, `feature-matrix`.
