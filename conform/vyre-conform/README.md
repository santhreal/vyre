# vyre-conform

Compatibility re-export crate forwarding the differential conformance engine from `vyre_test_support::conform`.

## Purpose

- Re-exports `vyre_test_support::conform::*` and forwards `gpu` and `device-tests` features to `vyre-test-support`.

## Boundaries

- Unpublished internal crate (`publish = false`, `conformance` layer) depending solely on `vyre-test-support` (`features = ["conform"]`).
- Contains re-exports only; differential parity lenses, counterexample minimization, and algebraic law checks are defined in `vyre-test-support`.
- Must never be depended on by production crates; new callers depend on `vyre-test-support` with feature `conform` or `conform-gpu`.
