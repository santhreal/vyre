# vyre-libs-analysis

Layer `libraries`. Owner `libs-analysis`.

## Owns

Compiler-internal static analysis over programs already built: cost models, dataflow fixpoint solutions, and the diagnostics derived from them. Every result is a `Program` or a fact about one.

## Must never contain

1. A concrete backend name, an emitter crate, or a shader dialect string. Lowering asks the target what it can do; it never asks which vendor made it.
2. A host-side reimplementation of what the IR already expresses. A CPU evaluation of a composition belongs in `vyre-reference` and nowhere else.
3. Execution-level schedule decisions inside a builder: hardcoded `InvocationId` or `LocalId` indices, explicit workgroup geometry, or a serial-versus-parallel predicate. Iteration domains are abstract here and are scheduled by `vyre-megakernel` and `vyre-foundation`.
4. Host dispatch orchestration, scratch allocation loops, or runtime state caching. Those belong to `vyre-driver` and `vyre-runtime`.
5. A pass that mutates the program it analyses. Analysis reports; the optimizer decides.
6. A cost number tied to one device model. A cost model reads a capability record, never a chip name.

## What crosses its edges

Out of this crate, into:

- `vyre-foundation` over the `foundation-ir` seam.
- `vyre-libs-builder` over the `libs-builder` seam.
- `vyre-libs-fixpoint` over the `libs-fixpoint` seam.
- `vyre-libs-graph` over the `libs-graph` seam.
- `vyre-libs-math` over the `libs-math` seam.
- `vyre-megakernel` over the `megakernel-compiler` seam.
- `vyre-primitives` over the `primitive-library` seam.
- `vyre-spec` over the `specification` seam.

Under test only, into:

- `vyre-driver-reference` over the `reference-driver` seam.
- `vyre-libs-encoding` over the `libs-encoding` seam.
- `vyre-reference` over the `reference-semantics` seam.
- `vyre-test-support` over the `test-support` seam.

## Direction that may not reverse

`vyre-foundation`, `vyre-primitives` and `vyre-spec` must never depend on `vyre-libs-analysis`. The edge is one way: a cycle back into this crate makes the two crates one crate that cannot be built, reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in `Cargo.toml` that the registry does not carry, and a registry row no manifest declares, both fail.
- The public surface is recorded in `docs/public-api/vyre-libs-analysis.txt`. An item added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `file-size`, `feature-isolation`, `crate-readmes`, `crate-pages`.
