# vyre-pass-engine

Layer `pass-engine`. Owner `pass-engine`.

## Owns

Execute optimizer passes as Vyre Programs through compiler-owned semantic
compilation and admitted artifact submission.

The chapter is [crate boundaries](../docs/architecture/crates.md#vyre-pass-engine).

## Must never contain

Host reimplementations of passes that exist as compositions. That is the
composition rule applied to the compiler itself, and this crate is the proof
the rule is livable.

## What crosses its edges

Out of this crate, into:

- `vyre-foundation` over the `foundation-ir` seam, public: typed IR, graph,
  diagnostics, validation, and semantic optimization contracts. Built when:
  always.
- `vyre-libs` over the `product-libraries` seam, private: product operation
  builders. Built when: always.
- `vyre-megakernel` over the `megakernel-compiler` seam, private: whole-graph
  compilation and immutable artifact contracts. Built when: always.
- `vyre-primitives` over the `primitive-library` seam, public: reusable
  semantic Program builders. Built when: always.

Into this crate, from:

- `vyre-driver-cuda` over the `pass-engine` seam, public.
- `vyre-driver-wgpu` over the `pass-engine` seam, public.

## Direction that may not reverse

`vyre-foundation`, `vyre-libs`, `vyre-megakernel`, and `vyre-primitives` must
never depend on `vyre-pass-engine`. The edge is one way. A reverse dependency
creates a cycle that prevents separate builds, review, and publication.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares 2 features beyond `default`: `all-solvers`,
  `optimizer`. Each builds alone.
- The public surface is recorded in `docs/public-api/vyre-pass-engine.txt`. An
  item added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`public-api-snapshot`, `public-api-paths`, `feature-isolation`,
`feature-matrix`.
