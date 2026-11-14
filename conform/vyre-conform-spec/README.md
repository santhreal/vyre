# vyre-conform-spec

Compatibility re-export crate forwarding conformance witness and law schemas from `vyre_test_support::conform_spec`.

## Purpose

- Re-exports `vyre_test_support::conform_spec::*` for conformance suites transitioning to `vyre-test-support`.

## Boundaries

- Unpublished internal crate (`publish = false`, `conformance` layer) depending solely on `vyre-test-support` (`features = ["conform-spec"]`).
- Contains a single re-export line; all conformance schema types are defined in `vyre-test-support`.
- Must never be depended on by production crates; new callers depend on `vyre-test-support` with feature `conform-spec`.
