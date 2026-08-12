# Test layout convention

Applies to Vyre 0.7.2.

Tests in the vyre workspace live in exactly one of three places.

## Unit tests

Inside the source file they test, in a `#[cfg(test)] mod tests` block.
One module per file. Import `super::*`. No external crate deps.

```rust
// vyre-foundation/src/validate/typecheck.rs
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn and_or_type_is_bool() { /* … */ }
}
```

Unit tests prove the **contract of a function**. They run fast, have no
setup, and assert the postcondition directly.

## Integration tests

Under `<crate>/tests/`. Use one integration-test file per observable feature
area. The file exercises the crate through its public boundary.

```
vyre/tests/artifact_workflow.rs
vyre-runtime/tests/artifact_admission_contract.rs
```

Integration tests prove the **contract of a module from outside the
crate**. They exercise public APIs, they can touch the filesystem and the
GPU, and they encode golden-vector / KAT-style assertions.

## Adversarial / property / fuzz

- Adversarial (`<crate>/tests/<behavior>.rs`): hand-written boundary cases
  designed to fail the broken behavior. Each affected contract needs one.
- Property (`proptest`): random inputs, invariants as assertions. Live
  alongside the unit/integration tests that own the function under test.
- Fuzz (`<crate>/fuzz/`): `cargo-fuzz` targets for wire-format and other
  attacker-controlled inputs. Separate crate per workspace member.

Bench files live under `<crate>/benches/` and are wired via `[[bench]]`
entries in that crate's `Cargo.toml`. Bench baselines are committed at
`<crate>/benches/baselines/<bench>.json` so CI can diff.

## What NOT to do

- Don't mix unit and integration tests in `tests/`. `tests/` is external.
- Don't invent new top-level test locations. Add the test to the owning
  crate's `tests/` directory unless an existing inline test must change with
  the contract.
- Don't add `[[test]]` entries in `Cargo.toml` unless the test needs a bespoke
  entrypoint.

Covers ARCH-019 and NEW-TEST-001 scope expectations.
