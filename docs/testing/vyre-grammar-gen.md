# Testing `vyre-grammar-gen`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-grammar-gen
```

Generate host-side grammar tables consumed by frontend and parsing crates.

The crate lives at `vyre-grammar-gen`. The `grammar-generation` owner maintains its
`tooling` testing contract.

## Commands

```console
./cargo_full test -p vyre-grammar-gen
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bin` | `vyre-grammar-gen` | `vyre-grammar-gen/src/main.rs` | None | `./cargo_full test -p vyre-grammar-gen --bin vyre-grammar-gen` |
| `example` | `lex_c_source` | `vyre-grammar-gen/examples/lex_c_source.rs` | None | `./cargo_full test -p vyre-grammar-gen --example lex_c_source` |
| `example` | `pack_lexer_blob` | `vyre-grammar-gen/examples/pack_lexer_blob.rs` | None | `./cargo_full test -p vyre-grammar-gen --example pack_lexer_blob` |
| `lib` | `vyre_grammar_gen` | `vyre-grammar-gen/src/lib.rs` | None | `./cargo_full test -p vyre-grammar-gen` |
| `test` | `adversarial` | `vyre-grammar-gen/tests/adversarial.rs` | None | `./cargo_full test -p vyre-grammar-gen --test adversarial` |
| `test` | `adversarial_contracts` | `vyre-grammar-gen/tests/adversarial_contracts.rs` | None | `./cargo_full test -p vyre-grammar-gen --test adversarial_contracts` |
| `test` | `conformance_contracts` | `vyre-grammar-gen/tests/conformance_contracts.rs` | None | `./cargo_full test -p vyre-grammar-gen --test conformance_contracts` |
| `test` | `corpus_smoke` | `vyre-grammar-gen/tests/corpus_smoke.rs` | None | `./cargo_full test -p vyre-grammar-gen --test corpus_smoke` |
| `test` | `gap` | `vyre-grammar-gen/tests/gap.rs` | None | `./cargo_full test -p vyre-grammar-gen --test gap` |
| `test` | `gap_contracts` | `vyre-grammar-gen/tests/gap_contracts.rs` | None | `./cargo_full test -p vyre-grammar-gen --test gap_contracts` |
| `test` | `gen_lex_hash` | `vyre-grammar-gen/tests/gen_lex_hash.rs` | None | `./cargo_full test -p vyre-grammar-gen --test gen_lex_hash` |
| `test` | `hello_max_munch_golden` | `vyre-grammar-gen/tests/hello_max_munch_golden.rs` | None | `./cargo_full test -p vyre-grammar-gen --test hello_max_munch_golden` |
| `test` | `integration` | `vyre-grammar-gen/tests/integration.rs` | None | `./cargo_full test -p vyre-grammar-gen --test integration` |
| `test` | `integration_contracts` | `vyre-grammar-gen/tests/integration_contracts.rs` | None | `./cargo_full test -p vyre-grammar-gen --test integration_contracts` |
| `test` | `property` | `vyre-grammar-gen/tests/property.rs` | None | `./cargo_full test -p vyre-grammar-gen --test property` |
| `test` | `property_contracts` | `vyre-grammar-gen/tests/property_contracts.rs` | None | `./cargo_full test -p vyre-grammar-gen --test property_contracts` |
| `test` | `unit` | `vyre-grammar-gen/tests/unit.rs` | None | `./cargo_full test -p vyre-grammar-gen --test unit` |
| `test` | `unit_contracts` | `vyre-grammar-gen/tests/unit_contracts.rs` | None | `./cargo_full test -p vyre-grammar-gen --test unit_contracts` |

## Test classes

- Command and policy behavior
- Evidence schema and regeneration contracts
- Failure diagnostics and repository boundaries

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
