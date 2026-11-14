# vyre-reference

Independent pure-Rust semantic interpreter and floating-point ULP distance oracle for vyre IR.

## Purpose

- Evaluates `Program` IR deterministically on the host under explicit execution budgets to produce reference outputs for parity verification.
- Computes canonical ULP distance and verifies composition witnesses across scalar, vector, and tile operations.
- Serves as the sole crate permitted to evaluate vyre IR semantics on the CPU.

## Boundaries

- Depends on `vyre-foundation`, `vyre-primitives`, and `vyre-spec` only among workspace crates.
- Never acts as a production execution backend or a silent CPU fallback when a GPU device is absent.
- Contains no lowering schedules, no target shader emission, and no hardware backend state.
- May be depended on only by conformance harnesses (`vyre-test-support`), benchmarks (`vyre-bench`), and dev-dependency test suites; production compiler and runtime crates (`vyre-compiler`, `vyre-emit`, `vyre-runtime`, `vyre`) must never depend on it.
