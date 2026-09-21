# vyre-libs-vfs

Layer `libraries`. Owner `libs-vfs`.

## Owns

Virtual filesystem compositions: asynchronous DMA block loads and asset resolution over a resident block table.

## Must never contain

1. A concrete backend name, an emitter crate, or a shader dialect string. Lowering asks the target what it can do; it never asks which vendor made it.
2. A host-side reimplementation of what the IR already expresses. A CPU evaluation of a composition belongs in `vyre-reference` and nowhere else.
3. Execution-level schedule decisions inside a builder: hardcoded `InvocationId` or `LocalId` indices, explicit workgroup geometry, or a serial-versus-parallel predicate. Iteration domains are abstract here and are scheduled by `vyre-megakernel` and `vyre-foundation`.
4. Host dispatch orchestration, scratch allocation loops, or runtime state caching. Those belong to `vyre-driver` and `vyre-runtime`.
5. A filesystem call, a path resolution against the host filesystem, or an IO handle. This crate composes the transfer; `vyre-runtime` performs it.
6. A cache policy. Residency and retention are runtime mechanisms.

## What crosses its edges

Out of this crate, into:

- `vyre-foundation` over the `foundation-ir` seam.
- `vyre-libs-builder` over the `libs-builder` seam.

## Direction that may not reverse

`vyre-foundation`, `vyre-primitives` and `vyre-spec` must never depend on `vyre-libs-vfs`. The edge is one way: a cycle back into this crate makes the two crates one crate that cannot be built, reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in `Cargo.toml` that the registry does not carry, and a registry row no manifest declares, both fail.
- The public surface is recorded in `docs/public-api/vyre-libs-vfs.txt`. An item added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `file-size`, `feature-isolation`, `crate-readmes`, `crate-pages`.
