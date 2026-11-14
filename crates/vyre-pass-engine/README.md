# vyre-pass-engine

Executes compiler analysis and optimization passes as dispatched vyre `Program` graphs.

## Purpose

- Marshals compiler IR analysis tables into device-ready byte buffers (`dispatch_buffers`) and executes pass `Program` graphs through `vyre-compiler` semantic execution contracts.
- Exposes optional solver suites (`all-solvers`) over `vyre-libs` graph, analysis, encoding, reasoning, solver, and scheduling compositions.

## Boundaries

- Depends on `vyre-foundation`, `vyre-primitives`, `vyre-compiler`, and `vyre-libs` (`default-features = false`).
- Contains pass execution orchestration and buffer marshalling only; IR data structures belong in `vyre-foundation`, pass `Program` compositions belong in `vyre-libs`, and hardware dispatch belongs in `vyre-runtime`.
- Unpublished internal crate (`publish = false`).
- May be depended on only by workspace tooling and benchmarks (`vyre-bench`); `vyre-foundation`, `vyre-compiler`, `vyre-emit`, `vyre-runtime`, and `vyre` must not depend on it.
