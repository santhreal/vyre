# vyre-conform

Layer `conformance`. Owner `conformance`.

## Owns

Execute production artifacts against independent reference semantics, minimize
counterexamples, check algebraic laws, and issue versioned certificates and
replay records through one library and thin CLI.

The chapter is [crate boundaries](docs/architecture/crates.md#vyre-conform-and-vyre-conform-spec).

## Must never contain

A pass a backend can satisfy without producing the reference's bytes. A soft
pass is worse than no suite.

## What crosses its edges

Out of this crate, into:

- `vyre` over the `public-facade` seam, private: public lifecycle facade. Built
  when: always.
- `vyre-conform-spec` over the `conformance` seam, private: versioned
  conformance schemas. Built when: always.
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
- `vyre-libs` over the `product-libraries` seam, private: product operation
  builders. Built when: always.
- `vyre-megakernel` over the `megakernel-compiler` seam, private: whole-graph
  compilation and immutable artifact contracts. Built when: always.
- `vyre-primitives` over the `primitive-library` seam, private: reusable
  semantic Program builders. Built when: always.
- `vyre-reference` over the `reference-semantics` seam, private: independent
  semantic oracle execution. Built when: always.
- `vyre-runtime` over the `runtime` seam, private: artifact admission,
  residency, submission, recovery, and readback lifecycle. Built when: always.
- `vyre-spec` over the `specification` seam, private: stable cross-engine
  schemas and operation definitions. Built when: always.
- `vyre-registry-link` over the `registry-link` seam, private: linked inventory
  registry sources and the per-source floor. Built when: always.

No workspace member depends on this crate.

## Direction that may not reverse

`vyre`, `vyre-conform-spec`, `vyre-driver`, `vyre-driver-cuda`,
`vyre-driver-wgpu`, `vyre-foundation`, `vyre-libs`, `vyre-megakernel`,
`vyre-primitives`, `vyre-reference`, `vyre-runtime`, `vyre-spec`,
`vyre-registry-link` must never depend on `vyre-conform`. The edge is one way:
a cycle back into this crate makes the two crates one crate that cannot be
built, reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares 1 feature beyond `default`: `gpu`. It builds alone.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`feature-isolation`, `feature-matrix`.
