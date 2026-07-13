# vyre-test-support

Test-only harness helpers shared across the vyre workspace. **Dev-dependency only** 
never a runtime dependency of any crate.

## `assert_registry_closure`

The ONE canonical **registry/coverage closure gate**. Each vyre crate that ships
`pub fn ... -> Program` builders keeps a thin `tests/adversarial_registry_closure.rs`:

```rust
const COVERAGE_WAIVER: &[&str] = &[/* builder, // reason */];

#[test]
fn every_program_builder_is_tested_registered_or_explicitly_waived() {
    vyre_test_support::assert_registry_closure(env!("CARGO_MANIFEST_DIR"), COVERAGE_WAIVER, 4);
}
```

The helper source-enumerates every `pub fn NAME(...) -> Program` builder under the crate's
`src/` (excluding `&self` methods and `Program`-first-param transform passes) and asserts
each is registered in an `inventory::submit!` block, pinned by a `tests/` parity test, or
listed in `COVERAGE_WAIVER` with a reason. Stale / now-covered / unwaived guards keep the
waiver honest and only-shrinkable; the `floor` argument fails loudly if the enumeration
silently breaks.

Enumeration reads source as text, so the gate is **feature-independent** and matches CI
under any `--features` selection. Replaces the 26 drifting per-crate copies (4 real gates +
22 tautology stubs) tracked in `BACKLOG.md` WIRING-tautology-closure-25crates.
