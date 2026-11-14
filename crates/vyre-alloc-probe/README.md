# vyre-alloc-probe

Compatibility re-export crate forwarding per-thread heap allocation counters from `vyre_test_support::alloc_probe`.

## Purpose

- Re-exports `vyre_test_support::alloc_probe::*` (`CountingAlloc`, `Region`, `ThreadAllocStats`, `snapshot`) for existing `[dev-dependencies]` suites during workspace consolidation.

## Boundaries

- Unpublished internal crate (`publish = false`, `test-tooling` layer) depending solely on `vyre-test-support` (`default-features = false`).
- Contains a single re-export line; all allocator tracking logic is defined in `vyre-test-support`.
- Must never be a production dependency of any publishable crate; new callers import `vyre_test_support::alloc_probe` directly.
