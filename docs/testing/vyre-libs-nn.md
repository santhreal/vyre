# Testing `vyre-libs-nn`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-libs-nn
```

Own neural network activations, linear transformations, normalization, attention, mixture-of-experts, and language model inference IR compositions. Does not own general matrix kernels, backend lowering, or runtime execution.

The crate lives at `vyre-libs-nn` and owns the `libs-nn` seam in the `libraries` layer.

## Commands

```console
./cargo_full test -p vyre-libs-nn
```

```console
./cargo_full test -p vyre-libs-nn --all-features
```

## Feature sets

- Default feature members: `nn`, `nn-inference`, `llm`
- Available manifest features: `default`, `llm`, `nn`, `nn-activation`, `nn-attention`, `nn-inference`, `nn-kernels`, `nn-linear`, `nn-linear-4bit`, `nn-moe`, `nn-norm`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_libs_nn` | `vyre-libs-nn/src/lib.rs` | None | `./cargo_full test -p vyre-libs-nn` |
| `test` | `all_tests` | `vyre-libs-nn/tests/all_tests.rs` | None | `./cargo_full test -p vyre-libs-nn --test all_tests` |

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
