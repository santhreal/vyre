# xtask-registry

Layer `tooling`. Owner `release-tooling`.

## Owns

Own the xtask subcommands that must observe the live operation registry, the
primitive catalog behind it, or a linked backend driver.

The chapter is [crate boundaries](../docs/architecture/crates.md#xtask-xtask-registry-xtask-evidence).

## Must never contain

An exemption. A gate that is allowed to be red is not a gate.

## What crosses its edges

Out of this crate, into:

- `xtask` over the `release-tooling` seam, private: subcommand registry,
  bounded readers, and release manifests. Built when: always.
- `vyre` over the `public-facade` seam, private: public lifecycle facade. Built
  when: always.
- `vyre-foundation` over the `foundation-ir` seam, private: typed IR, graph,
  diagnostics, validation, and semantic optimization contracts. Built when:
  always.
- `vyre-megakernel` over the `megakernel-compiler` seam, private: neutral
  artifact compilation and target payload contracts. Built when: always.
- `vyre-driver` over the `backend-contract` seam, private: backend-neutral
  target, materialization, submission, and completion contracts. Built when:
  always.
- `vyre-libs` over the `product-libraries` seam, private: product operation
  builders. Built when: always.
- `vyre-primitives` over the `primitive-library` seam, private: reusable
  semantic Program builders. Built when: always.
- `vyre-reference` over the `reference-semantics` seam, private: independent
  semantic oracle execution. Built when: always.
- `vyre-spec` over the `specification` seam, private: stable cross-engine
  schemas and operation definitions. Built when: always.
- `vyre-registry-link` over the `registry-link` seam, private: linked inventory
  registry sources and the per-source floor. Built when: always.

No workspace member depends on this crate.

## Direction that may not reverse

`xtask`, `vyre`, `vyre-foundation`, `vyre-megakernel`, `vyre-driver`,
`vyre-libs`, `vyre-primitives`, `vyre-reference`, `vyre-spec`,
`vyre-registry-link` must never depend on `xtask-registry`. The edge is one
way: a cycle back into this crate makes the two crates one crate that cannot be
built, reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares no feature beyond `default`, so every build of it is the
  same build.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`.
