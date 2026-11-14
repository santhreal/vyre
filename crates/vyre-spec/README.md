# vyre-spec

Stable cross-engine data schemas, operation descriptors, algebraic law declarations, and IR level definitions.

## Purpose

- Defines `OpDef`, `AlgebraicLaw`, `Category`, `IrLevel`, intrinsic tables, and versioned compatibility schemas.
- Provides schema digests and canonical serialization shared across the compiler, emitters, runtime, and conformance harnesses.

## Boundaries

- Depends on zero workspace crates (`serde`, `serde_json`, `smallvec` only).
- Contains pure data definitions and schema validation; must not contain IR graph builders, lowering passes, target emitters, or runtime state.
- May be depended on by every workspace layer (`vyre-foundation`, `vyre-reference`, `vyre-compiler`, `vyre-libs`, `vyre-runtime`, `vyre-registry-link`, `vyre`, `vyre-test-support`, `vyre-bench`).
- Must never add a production dependency on another workspace crate.
