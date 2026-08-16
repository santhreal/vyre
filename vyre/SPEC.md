# vyre

Layer `facade`. Owner `public-facade`.

## Owns

Public facade. Re-export IR, driver, runtime, and the artifact compiler. Own no
logic.

The chapter is [crate boundaries](docs/architecture/crates.md#vyre).

## Must never contain

Logic. A facade with behaviour is a fourth place to look for a bug.

## What crosses its edges

Out of this crate, into:

- `vyre-driver` over the `backend-contract` seam, private: backend-neutral
  target, materialization, submission, and completion contracts. Built when:
  always.
- `vyre-driver-cuda` over the `cuda-driver` seam, private: native accelerator
  backend execution. Built when: always.
- `vyre-driver-wgpu` over the `portable-driver` seam, private: portable backend
  execution. Built when: always.
- `vyre-foundation` over the `foundation-ir` seam, private: typed IR, graph,
  diagnostics, validation, and semantic optimization contracts. Built when:
  always.
- `vyre-megakernel` over the `megakernel-compiler` seam, private: whole-graph
  compilation and immutable artifact contracts. Built when: always.
- `vyre-runtime` over the `runtime` seam, private: artifact admission,
  residency, submission, recovery, and readback lifecycle. Built when: always.
- `vyre-spec` over the `specification` seam, private: stable cross-engine
  schemas and operation definitions. Built when: always.

Into this crate, from:

- `vyre-bench` over the `public-facade` seam, private.
- `vyre-conform` over the `public-facade` seam, private.
- `vyre-debug` over the `public-facade` seam, private.
- `xtask-registry` over the `public-facade` seam, private.

## Direction that may not reverse

`vyre-driver`, `vyre-driver-cuda`, `vyre-driver-wgpu`, `vyre-foundation`,
`vyre-megakernel`, `vyre-runtime`, `vyre-spec` must never depend on `vyre`. The
edge is one way: a cycle back into this crate makes the two crates one crate
that cannot be built, reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares 3 features beyond `default`: `cpu-parity`, `cuda`, `wgpu`.
  Each builds alone.
- The public surface is recorded in `docs/public-api/vyre.txt`. An item added,
  removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`public-api-snapshot`, `public-api-paths`, `feature-isolation`,
`feature-matrix`.
