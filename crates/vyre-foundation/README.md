# vyre-foundation

Typed intermediate representation, memory model, zero-copy wire format, validation, and semantic optimization for vyre programs.

## Purpose

- Defines `Program`, `ProgramGraph`, logical program domains, schedule IR, types, memory spaces, and layout rules.
- Implements zero-copy `VIR0` wire serialization and deserialization (`wire`), structural and type validation (`validate`), and diagnostics.
- Provides semantic operation registration records and backend-neutral algebraic and polyhedral optimization passes (`optimizer`, `pass_math`).

## Boundaries

- Depends on `vyre-spec` and `vyre-macros` as its only workspace dependencies.
- Contains zero application or domain compositions, zero target source emission, and zero device or runtime calls.
- May be depended on by every higher layer (`vyre-primitives`, `vyre-reference`, `vyre-compiler`, `vyre-emit`, `vyre-libs`, `vyre-pass-engine`, `vyre-runtime`, `vyre-registry-link`, `vyre`, `vyre-test-support`, `vyre-bench`, `xtask`).
- Must never depend on `vyre-primitives`, `vyre-compiler`, `vyre-emit`, `vyre-runtime`, `vyre-libs`, or `vyre-reference` in production dependencies.
