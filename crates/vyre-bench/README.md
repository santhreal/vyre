# vyre-bench

Reproducible GPU workload benchmark and reference-parity harness for vyre (`vyre-bench`).

## Purpose

- Measures megakernel, micro, macro, and release workloads (`--suite megakernel`) across `vyre-runtime` hardware backends (`cuda`, `wgpu`, `spirv`, `metal`) against native baselines.
- Reports per-workload wall median (`p50`), device active time, heap allocation count and bytes (`alloc_probe`), batch per-item latency, compile time, and output parity against `vyre-reference`.
- Enforces zero avoidable host copies and zero per-dispatch steady-state heap allocations on resident execution paths.

## Boundaries

- Unpublished tooling crate (`publish = false`, rank 8 `tooling` layer).
- May depend on `vyre`, `vyre-foundation`, `vyre-spec`, `vyre-primitives`, `vyre-libs`, `vyre-compiler`, `vyre-pass-engine`, `vyre-emit`, `vyre-runtime`, `vyre-registry-link`, `vyre-reference`, and `vyre-test-support`.
- Contains benchmark suites, native baseline runners, and measurement reporting only; compiler passes, target emitters, runtime scheduling, and reusable test fixtures belong in lower crates.
- No workspace crate may depend on `vyre-bench`.
