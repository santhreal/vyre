# CODEX-IMPL-2 Report

## BUILD 1

- `vyre-primitives` is the canonical open-IR primitive crate.
- The twenty canonical primitives have concrete `NodeKind` structs and
  inventory registrations.
- `cargo check -p vyre-primitives` passes.

## BUILD 2

- `vyre-reference::primitives` defines `ReferenceEvaluator` and `EvalError`.
- All twenty canonical primitives have pure Rust CPU reference evaluators.
- Evaluators use deterministic little-endian byte payloads via
  `workgroup::Memory::from_bytes`.

## BUILD 3

- `vyre-conform::exhaustive` enumerates bounded scalar domains and emits
  coverage reports with BLAKE3 commitments.
- 16-bit binary domains are partitioned into 256 deterministic shards.

## BUILD 4

- `vyre-conform::laws::exhaustive` enumerates the full 8-bit domain for
  declared associativity, commutativity, and idempotence laws.
- Broken laws emit exact witnesses and effective `LawStatus::Broken` reports.

## BUILD 5

- `vyre-std` exports no wrappers.
- Source files under `vyre-std/src` are retained as orchestrator markers and
  contain no implementation.
- Callers must use `vyre-primitives` or compose canonical primitives directly.

## Verification

- Passed: `cargo check -p vyre-primitives`.
- Blocked: `cargo check -p vyre-reference`, `cargo check -p vyre-conform`,
  and `cargo check --workspace` currently fail in the pre-existing dirty
  `vyre-core` open-IR refactor before compiling the CODEX-IMPL-2 crates.
- Current blocker observed: the dirty core open-IR refactor no longer exposes
  `crate::ir::model::expr::{Expr, ExprNode, Ident}` and no longer declares
  `model::node_kind`, while many core modules still import those paths.
