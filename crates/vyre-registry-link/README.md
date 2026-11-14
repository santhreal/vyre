# vyre-registry-link

Link-anchor crate retaining global `inventory` registrations across static linking boundaries.

## Purpose

- References explicit link-anchor symbols in every registration-submitting crate (`vyre-libs`, `vyre-primitives`, `vyre-compiler`, and `vyre-runtime` with its optional `cuda`, `metal`, `spirv`, and `wgpu` backend modules).
- Enforces per-source registration floors so static linkers cannot silently discard object files containing `inventory::submit!` entries.

## Boundaries

- Depends on `vyre-foundation`, `vyre-spec`, `vyre-compiler`, `vyre-runtime`, and optional `vyre-primitives` and `vyre-libs`.
- Unpublished internal linkage crate (`publish = false`). Contains link anchors, source enumerations, and floor assertions only; must not define operations, lowering passes, emitters, or runtime execution logic.
- May be depended on only by binaries and harnesses that inspect populated registries (`vyre-test-support`, `vyre-bench`); library crates must never depend on it.
