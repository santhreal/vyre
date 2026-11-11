# tests/SKILL.md: vyre facade

## Purpose

The `vyre` package is the stable public facade for semantic IR, whole-program
compilation, authenticated artifact sessions, typed submission, and scan
workflows.

## Critical invariants

- Production compilation starts from a validated `ProgramGraph` and produces an
  immutable compiler `Artifact`.
- Production execution materializes an authenticated `TargetPayload` into an
  `ArtifactInstance` before typed submission.
- The facade does not select concrete backends or lower directly to target code.
- Raw `Program` execution remains limited to explicit reference and conformance
  paths.
- Every public item has one documented stable path.

## Adversarial surface

Tests cover malformed graphs, incompatible payloads, invalid bindings, stale
materializer generations, and unavailable registered targets through the public
facade.

## Cross-crate contracts

This crate proves that the canonical owners compose through one public route:

- `vyre::ir` comes from `vyre-foundation`.
- `vyre::compiler` comes from `vyre-megakernel`.
- `vyre::ArtifactSession` comes from `vyre-runtime`.
- Concrete target compilation and materialization stay in concrete drivers.

## Bench and fuzz targets

Performance and codec fuzzing remain with the owning compiler, runtime, driver,
and product crates.

## What NOT to test here

- Owner-local implementation details.
- Source text or private module layout.
- Concrete backend semantics outside an end-to-end facade workflow.

## Running

```bash
./cargo_full test -p vyre
```
