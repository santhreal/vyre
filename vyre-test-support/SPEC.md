# vyre-test-support

Layer `test-tooling`. Owner `test-support`.

## Owns

Provide shared deterministic fixtures and assertions for workspace tests.

The chapter is [crate boundaries](../docs/architecture/crates.md#vyre-test-support).

## Must never contain

Production code, and a fixture that exists in a crate that already owns it.
Fixture duplication is how two suites end up testing two different programs
under one name.

## What crosses its edges

Out of this crate, into:

- `structure-gate` over the `release-tooling` seam, private: resolve the
  checkout a gate reports on from the working directory at run time. Built
  when: always.
- `vyre-foundation` over the `foundation-ir` seam, private: IR statement
  fixtures for the run-time variant enumeration, behind the ir-fixtures
  feature. Built when: always.
- `vyre-spec` over the `specification` seam, private: DataType and declared
  operation signatures for fixture tables, without gating a leaf crate behind
  ir-fixtures. Built when: always.

No workspace member depends on this crate.

## Direction that may not reverse

`structure-gate`, `vyre-foundation`, `vyre-spec` must never depend on
`vyre-test-support`. The edge is one way: a cycle back into this crate makes
the two crates one crate that cannot be built, reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares 1 feature beyond `default`: `ir-fixtures`. It builds
  alone.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`feature-isolation`, `feature-matrix`.
