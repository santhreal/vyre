# Testing `vyre-libs-parsing`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-libs-parsing
```

Own lexer drivers, LR(1) table walkers, and language-specific AST construction IR compositions. Does not own semantic analysis passes, backend lowering, or runtime execution.

The crate lives at `vyre-libs-parsing`. The `product-libraries` owner maintains its
`libraries` testing contract.

## Commands

```console
./cargo_full test -p vyre-libs-parsing
```

```console
./cargo_full test -p vyre-libs-parsing --all-features
```

## Feature sets

- Default feature members: `parsing`
- Available manifest features: `default`, `go-parser`, `parsing`, `parsing-kernels`, `python-parser`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_libs_parsing` | `vyre-libs-parsing/src/lib.rs` | None | `./cargo_full test -p vyre-libs-parsing` |
| `test` | `all_tests` | `vyre-libs-parsing/tests/all_tests.rs` | None | `./cargo_full test -p vyre-libs-parsing --test all_tests` |

## Test classes

- Product-library exact behavior
- Primitive-to-library composition
- Reference and backend parity

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
