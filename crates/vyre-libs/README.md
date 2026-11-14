# vyre-libs

Single workspace library of generic `Program` compositions built from vyre IR.

## Purpose

- Constructs reusable `Program` graphs across graph traversal, pattern scanning, parsing, text processing, linear algebra, neural operators (including safetensors metadata loading), hashing, decoding, bitsets, reductions, fixpoint iteration, visual kernels, and compiler-internal analysis/scheduling/solver compositions.
- Exposes domain modules behind Cargo features so consumers compile only the compositions they select.
- Submits `OperationRegistration` descriptors for every public composition into the operation inventory.

## Boundaries

- Depends on `vyre-foundation`, `vyre-spec`, `vyre-primitives`, and `vyre-compiler` among workspace crates.
- Returns `Program` values and host buffer descriptors only; must not contain target code emitters, device backends, host-side IR rewrite passes, or CPU fallback compute loops.
- Domain-specific helpers are restricted to generic compositions and consumer-backed utilities; separate per-domain library crates are prohibited.
- May be depended on by `vyre-pass-engine`, `vyre-runtime` (optional `libs-compositions`), `vyre-registry-link`, `vyre-test-support`, `vyre-bench`, and `xtask` (`debug`); `vyre-foundation`, `vyre-compiler`, and `vyre-emit` must not depend on it.
