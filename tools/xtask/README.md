# xtask

Unified workspace structural verifier, duplication scanner, coverage object pruner, and IR diagnostic CLI (`tools/xtask`).

## Purpose

- Runs fast source-tree structural checks (`cargo run -p xtask -- check`) covering architecture layer rules (`CRATE_OWNERSHIP.toml`), documentation register, single-test-binary layout, safety invariants, and duplicate code ceilings without compiling the compiler.
- Provides `prune_coverage_objects` for `cargo llvm-cov` object selection and optional `vyre-dbg` (`--features debug`) for inspecting lowered IR and emitted shaders.

## Boundaries

- Unpublished tooling crate (`publish = false`, rank 8 `tooling` layer).
- Default build depends on zero workspace crates (`syn`, `proc-macro2`, `toml`, `serde`, `serde_json`, `walkdir`, `blake3`, `sha2` only) so `xtask check` runs even when compiler crates fail to build.
- Workspace compiler dependencies (`vyre`, `vyre-foundation`, `vyre-compiler`, `vyre-emit`, `vyre-libs`, `vyre-test-support`) are restricted to the optional `debug` feature.
- No workspace crate may depend on `xtask`.
