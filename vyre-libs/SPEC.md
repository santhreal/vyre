# vyre-libs

Layer `libraries`. Owner `product-libraries`.

## Owns

Own every composition in the workspace: consumer dialects and compiler-internal
solvers, encoding, analysis, scheduling, and reasoning. Returns Programs. No
backend, no emitter, no host rewrite of IR.

The chapter is [crate boundaries](docs/architecture/crates.md#vyre-libs).

## Must never contain

Anything that names a concrete backend, links an emitter crate, or reimplements
in host Rust what IR expresses. The first two invert the dependency, because a
composition states what it needs and the driver decides who provides it. The
third is the failure the crate exists to prevent.

## What crosses its edges

Out of this crate, into:

- `vyre-foundation` over the `foundation-ir` seam, public: typed IR, graph,
  diagnostics, validation, and semantic optimization contracts. Built when:
  always.
- `vyre-primitives` over the `primitive-library` seam, public: the wire format,
  guarded IR construction, the launch-geometry helper, the marker types, and
  the intrinsic registrations. Built when: always.
- `vyre-spec` over the `specification` seam, public: stable cross-engine
  schemas and operation definitions. Built when: always.

Into this crate, from:

- `vyre-bench` over the `product-libraries` seam, private.
- `vyre-conform` over the `product-libraries` seam, private.
- `vyre-debug` over the `product-libraries` seam, private.
- `vyre-driver` over the `product-libraries` seam, private.
- `vyre-driver-cuda` over the `product-libraries` seam, private.
- `vyre-driver-wgpu` over the `product-libraries` seam, private.
- `vyre-pass-engine` over the `product-libraries` seam, private.
- `vyre-reference` over the `product-libraries` seam, public.
- `vyre-registry-link` over the `product-libraries` seam, private.
- `vyre-runtime` over the `product-libraries` seam, private.
- `xtask-registry` over the `product-libraries` seam, private.

## Direction that may not reverse

`vyre-foundation`, `vyre-primitives`, `vyre-spec` must never depend on
`vyre-libs`. The edge is one way: a cycle back into this crate makes the two
crates one crate that cannot be built, reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares 59 features beyond `default`: `analysis`, `bitset`,
  `cat-a-builder-options`, `cpu-parity`, `crypto`, `crypto-blake3`, `decode`,
  `device`, `encoding`, `fixpoint`, `full`, `geom`, `go-parser`, `graph`,
  `graph-dispatch`, `hash`, `intern`, `label`, `logical`, `matching`,
  `matching-dfa`, `matching-kernels`, `matching-nfa`, `matching-regex`,
  `matching-substring`, `math`, `math-algebra`, `math-broadcast`,
  `math-dialect`, `math-kernels`, `math-linalg`, `math-scan`, `math-succinct`,
  `nfa`, `nn`, `nn-activation`, `nn-attention`, `nn-inference`, `nn-kernels`,
  `nn-linear`, `nn-linear-4bit`, `nn-moe`, `nn-norm`, `opt`, `parsing`,
  `parsing-kernels`, `predicate`, `python-parser`, `reasoning`, `reduce`,
  `rule`, `scheduling`, `security`, `solvers`, `telemetry`, `test-fixtures`,
  `text`, `topology`, `visual`. Each builds alone.
- The public surface is recorded in `docs/public-api/vyre-libs.txt`. An item
  added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`public-api-snapshot`, `public-api-paths`, `feature-isolation`,
`feature-matrix`.
