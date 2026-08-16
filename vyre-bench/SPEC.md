# vyre-bench

Layer `tooling`. Owner `benchmarks`.

## Owns

Own reproducible workload benchmarks against the best available native baseline
for each class, not against vyre's own unfused output.

The chapter is [crate boundaries](../docs/architecture/crates.md#vyre-bench).

## Must never contain

A benchmark whose baseline is vyre's own unfused output. Beating your own slow
path is not a result. The baseline is the best available native implementation
for that class.

## What crosses its edges

Out of this crate, into:

- `vyre` over the `public-facade` seam, private: public lifecycle facade. Built
  when: always.
- `vyre-driver` over the `backend-contract` seam, private: backend-neutral
  target, materialization, submission, and completion contracts. Built when:
  always.
- `vyre-driver-cuda` over the `cuda-driver` seam, private: native accelerator
  backend execution. Built when: cfg(not(target_os = "macos")).
- `vyre-driver-reference` over the `reference-driver` seam, private: reference
  backend adaptation. Built when: always.
- `vyre-driver-wgpu` over the `portable-driver` seam, private: portable backend
  execution. Built when: always.
- `vyre-emit-ptx` over the `primary-binary-emitter` seam, private: primary
  binary backend text emission. Built when: always.
- `vyre-foundation` over the `foundation-ir` seam, private: typed IR, graph,
  diagnostics, validation, and semantic optimization contracts. Built when:
  always.
- `vyre-libs` over the `product-libraries` seam, private: product operation
  builders. Built when: always.
- `vyre-lower` over the `lowering` seam, private: verified backend-neutral
  representation lowering. Built when: always.
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
- `xtask` over the `release-tooling` seam, private: the one producer of the
  source fingerprint a recorded artifact names its tree with. Built when:
  always.

Into this crate, from:

- `xtask-evidence` over the `benchmarks` seam, private.

## Direction that may not reverse

`vyre`, `vyre-driver`, `vyre-driver-cuda`, `vyre-driver-reference`,
`vyre-driver-wgpu`, `vyre-emit-ptx`, `vyre-foundation`, `vyre-libs`,
`vyre-lower`, `vyre-primitives`, `vyre-reference`, `vyre-runtime`, `vyre-spec`,
`vyre-registry-link`, `xtask` must never depend on `vyre-bench`. The edge is
one way: a cycle back into this crate makes the two crates one crate that
cannot be built, reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares 1 feature beyond `default`: `cli`. It builds alone.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`feature-isolation`, `feature-matrix`.
