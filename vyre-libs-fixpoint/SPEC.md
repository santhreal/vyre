# vyre-libs-fixpoint

Layer `libraries`. Owner `libs-fixpoint`.

## Owns

Deterministic fixpoint iteration kernels and the grid synchronization barriers they require, including the convergence contract that makes a fixpoint reproducible.

## Must never contain

1. A concrete backend name, an emitter crate, or a shader dialect string. Lowering asks the target what it can do; it never asks which vendor made it.
2. A host-side reimplementation of what the IR already expresses. A CPU evaluation of a composition belongs in `vyre-reference` and nowhere else.
3. Execution-level schedule decisions inside a builder: hardcoded `InvocationId` or `LocalId` indices, explicit workgroup geometry, or a serial-versus-parallel predicate. Iteration domains are abstract here and are scheduled by `vyre-megakernel` and `vyre-foundation`.
4. Host dispatch orchestration, scratch allocation loops, or runtime state caching. Those belong to `vyre-driver` and `vyre-runtime`.
5. A convergence test that depends on scheduling order. A fixpoint whose result changes with lane order is not deterministic and is a defect here.
6. A grid-sync implementation. The barrier is requested as IR; the backend provides it.

## What crosses its edges

Out of this crate, into:

- `vyre-foundation` over the `foundation-ir` seam.
- `vyre-libs-builder` over the `libs-builder` seam.
- `vyre-primitives` over the `primitive-library` seam.

Under test only, into:

- `vyre-driver` over the `backend-contract` seam.
- `vyre-reference` over the `reference-semantics` seam.

## Direction that may not reverse

`vyre-foundation`, `vyre-primitives` and `vyre-spec` must never depend on `vyre-libs-fixpoint`. The edge is one way: a cycle back into this crate makes the two crates one crate that cannot be built, reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in `Cargo.toml` that the registry does not carry, and a registry row no manifest declares, both fail.
- The public surface is recorded in `docs/public-api/vyre-libs-fixpoint.txt`. An item added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `file-size`, `feature-isolation`, `crate-readmes`, `crate-pages`.
