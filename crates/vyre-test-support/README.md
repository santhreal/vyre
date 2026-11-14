# vyre-test-support

Private verification fixtures, per-thread allocation probe, and conformance harness for the vyre workspace.

## Purpose

- Provides deterministic IR fixtures (`ir-fixtures`), golden corpus readers (`golden-corpus`), semantic execution requests (`semantic-requests`), and `proptest` strategies (`spec-strategies`).
- Defines the global-allocator per-thread heap counter (`alloc_probe`) used by allocation budget tests and `vyre-bench`.
- Hosts the conformance schema (`conform-spec`) and differential parity, algebraic law, and counterexample minimization engine (`conform`, `conform-gpu`).
- Exposes `vyre_test_support::oracle` (`ReferenceSemanticExecutor`, `CpuRefEvaluator`, `ORACLE_EXECUTOR_ID`, `is_oracle_executor_id`, and `target_profile`) behind the `oracle` Cargo feature as the CPU reference semantic executor for differential tests.

## Boundaries

- Unpublished workspace test crate (`publish = false`, rank 7 `test-tooling` layer).
- Base build depends only on `vyre-spec`, `inventory`, `serde`, `serde_json`, and `toml`; compiler, library, runtime backend (`vyre-runtime/{cuda,wgpu,spirv,metal}`), and reference dependencies are strictly opt-in via Cargo features to keep test build graphs minimal.
- Contains test harnesses, probes, and fixtures only; must never be a normal production dependency of any publishable crate (`vyre`, `vyre-foundation`, `vyre-spec`, `vyre-macros`, `vyre-primitives`, `vyre-compiler`, `vyre-emit`, `vyre-runtime`, `vyre-libs`, `vyre-reference`).
- Reached only via `[dev-dependencies]` or by unpublished tooling crates (`vyre-bench`, `xtask`, `vyre-alloc-probe`, `vyre-conform`, `vyre-conform-spec`).
