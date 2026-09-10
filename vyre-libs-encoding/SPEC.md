# vyre-libs-encoding

Layer `libraries`. Owner `libs-encoding`.

## Owns

Compiler-internal encoding compositions: bitset encodings, provenance records, matroid structures, and fingerprints.

## Must never contain

1. A concrete backend name, an emitter crate, or a shader dialect string. Lowering asks the target what it can do; it never asks which vendor made it.
2. A host-side reimplementation of what the IR already expresses. A CPU evaluation of a composition belongs in `vyre-reference` and nowhere else.
3. Execution-level schedule decisions inside a builder: hardcoded `InvocationId` or `LocalId` indices, explicit workgroup geometry, or a serial-versus-parallel predicate. Iteration domains are abstract here and are scheduled by `vyre-megakernel` and `vyre-foundation`.
4. Host dispatch orchestration, scratch allocation loops, or runtime state caching. Those belong to `vyre-driver` and `vyre-runtime`.
5. A cryptographic guarantee. Fingerprints here identify structure; `vyre-libs-hash` and `vyre-libs-security` own integrity and trust.
6. An encoding whose width is read from a target profile.

## What crosses its edges

Out of this crate, into:

- `vyre-foundation` over the `foundation-ir` seam.
- `vyre-libs-bitset` over the `libs-bitset` seam.
- `vyre-libs-builder` over the `libs-builder` seam.
- `vyre-libs-graph` over the `libs-graph` seam.
- `vyre-libs-hash` over the `libs-hash` seam.
- `vyre-libs-math` over the `libs-math` seam.
- `vyre-libs-nn` over the `libs-nn` seam.
- `vyre-libs-parsing` over the `libs-parsing` seam.
- `vyre-libs-pattern` over the `libs-pattern` seam.
- `vyre-libs-reduce` over the `libs-reduce` seam.
- `vyre-megakernel` over the `megakernel-compiler` seam.
- `vyre-primitives` over the `primitive-library` seam.

Under test only, into:

- `vyre-driver-reference` over the `reference-driver` seam.
- `vyre-reference` over the `reference-semantics` seam.
- `vyre-test-support` over the `test-support` seam.

## Direction that may not reverse

`vyre-foundation`, `vyre-primitives` and `vyre-spec` must never depend on `vyre-libs-encoding`. The edge is one way: a cycle back into this crate makes the two crates one crate that cannot be built, reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in `Cargo.toml` that the registry does not carry, and a registry row no manifest declares, both fail.
- The public surface is recorded in `docs/public-api/vyre-libs-encoding.txt`. An item added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `file-size`, `feature-isolation`, `crate-readmes`, `crate-pages`.
