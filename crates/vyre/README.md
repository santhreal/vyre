# vyre

Public consumer facade for the vyre GPU megakernel compiler and runtime.

## Purpose

- Re-exports the validated IR, compiler, runtime, and schema contracts under a single dependency (`vyre::ir`, `vyre::compiler`, `vyre::runtime`, `vyre::spec`).
- Exposes optional `cuda` and `wgpu` features forwarding concrete backend selection to `vyre-runtime`.
- Publishes stable consumer types without adding facade-local execution or compilation logic.

## Boundaries

- Depends on `vyre-foundation`, `vyre-spec`, `vyre-compiler`, and `vyre-runtime`.
- Contains re-exports and consumer surface tests only. IR validation, lowering, target code generation, device execution, and domain compositions belong in lower crates.
- Within the workspace, only conformance harnesses (`vyre-test-support`), benchmark harnesses (`vyre-bench`), and diagnostic tooling (`xtask`) may depend on `vyre`.
- Lower crates (`vyre-spec`, `vyre-foundation`, `vyre-primitives`, `vyre-compiler`, `vyre-emit`, `vyre-runtime`, `vyre-libs`) must not depend on `vyre`.
