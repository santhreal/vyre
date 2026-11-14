# vyre-primitives

Marker types, launch-geometry helpers, and irreducible hardware intrinsics that cannot be expressed by composing existing vyre IR operations.

## Purpose

- Defines fixed-layout scalar and vector marker types (`bytemuck`-compatible) with zero default workspace dependencies.
- Registers irreducible Category C hardware intrinsics (`hardware` and `inventory-registry` features) that require dedicated emitter and reference-oracle arms.
- Provides guarded IR construction and launch-geometry helpers shared above the foundation layer.

## Boundaries

- Depends only on optional `vyre-foundation` among workspace crates (`bytemuck` and optional `inventory` externally).
- Any operation expressible as a composition of existing IR nodes belongs in `vyre-libs`, never in `vyre-primitives`.
- Contains no target code generation, no device dispatch, and no host CPU fallback execution.
- May be depended on by `vyre-reference`, `vyre-libs`, `vyre-pass-engine`, `vyre-registry-link`, `vyre-test-support`, and `vyre-bench`.
