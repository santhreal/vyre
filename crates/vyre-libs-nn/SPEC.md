# vyre-libs-nn

Layer `libraries`. Owner `libs-nn`.

## Owns

Neural network compositions: activations, linear layers, normalization, attention, mixture-of-experts routing, and the inference compositions assembled from them.

## Must never contain

1. A concrete backend name, an emitter crate, or a shader dialect string. Lowering asks the target what it can do; it never asks which vendor made it.
2. A host-side reimplementation of what the IR already expresses. A CPU evaluation of a composition belongs in `vyre-reference` and nowhere else.
3. Execution-level schedule decisions inside a builder: hardcoded `InvocationId` or `LocalId` indices, explicit workgroup geometry, or a serial-versus-parallel predicate. Iteration domains are abstract here and are scheduled by `vyre-megakernel` and `vyre-foundation`.
4. Host dispatch orchestration, scratch allocation loops, or runtime state caching. Those belong to `vyre-driver` and `vyre-runtime`.
5. A named model architecture, a checkpoint convention, a tokenizer, or a configuration file for a published model. Those are application concerns.
6. A weight layout tied to one serialization format. `vyre-safetensors` maps a container into typed transfers.

## What crosses its edges

Out of this crate, into:

- `vyre-foundation` over the `foundation-ir` seam.
- `vyre-libs-builder` over the `libs-builder` seam.
- `vyre-libs-math` over the `libs-math` seam.
- `vyre-libs-reduce` over the `libs-reduce` seam.
- `vyre-megakernel` over the `megakernel-compiler` seam.
- `vyre-primitives` over the `primitive-library` seam.
- `vyre-spec` over the `specification` seam.

Under test only, into:

- `vyre-lower` over the `lowering` seam.
- `vyre-reference` over the `reference-semantics` seam.
- `vyre-test-support` over the `test-support` seam.

## Direction that may not reverse

`vyre-foundation`, `vyre-primitives` and `vyre-spec` must never depend on `vyre-libs-nn`. The edge is one way: a cycle back into this crate makes the two crates one crate that cannot be built, reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in `Cargo.toml` that the registry does not carry, and a registry row no manifest declares, both fail.
- The public surface is recorded in `docs/public-api/vyre-libs-nn.txt`. An item added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `file-size`, `feature-isolation`, `crate-readmes`, `crate-pages`.
