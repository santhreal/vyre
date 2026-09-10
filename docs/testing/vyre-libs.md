# Testing `vyre-libs`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-libs
```

Own every composition in the workspace: consumer dialects and compiler-internal solvers, encoding, analysis, scheduling, and reasoning. Returns Programs. No backend, no emitter, no host rewrite of IR.

The crate lives at `vyre-libs` and owns the `semantic-library` seam in the `libraries` layer.

## Commands

```console
./cargo_full test -p vyre-libs
```

```console
./cargo_full test -p vyre-libs --all-features
```

## Feature sets

- Default feature members: `math-linalg`, `math-scan`, `math-broadcast`, `nn-activation`, `nn-linear`, `nn-norm`, `pattern-substring`, `pattern-dfa`, `hash`, `decode`
- Available manifest features: `analysis`, `bitset`, `builder`, `builder-ops`, `cat-a-builder-options`, `crypto`, `crypto-blake3`, `decode`, `default`, `device`, `encoding`, `fixpoint`, `full`, `geom`, `go-parser`, `graph`, `graph-dispatch`, `hash`, `interactive-graphics`, `label`, `llm`, `logical`, `math`, `math-algebra`, `math-broadcast`, `math-dialect`, `math-kernels`, `math-linalg`, `math-scan`, `math-succinct`, `nfa`, `nn`, `nn-activation`, `nn-attention`, `nn-inference`, `nn-kernels`, `nn-linear`, `nn-linear-4bit`, `nn-moe`, `nn-norm`, `opt`, `parsing`, `parsing-kernels`, `pattern`, `pattern-dfa`, `pattern-kernels`, `pattern-nfa`, `pattern-regex`, `pattern-substring`, `predicate`, `python-parser`, `reasoning`, `reduce`, `representation`, `rule`, `scheduling`, `security`, `solvers`, `telemetry`, `test-fixtures`, `text`, `topology`, `vfs`, `visual`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `dominator_tree_e2e` | `vyre-libs/examples/dominator_tree_e2e.rs` | None | `./cargo_full test -p vyre-libs --example dominator_tree_e2e` |
| `example` | `jacobi_workgroup_perf` | `vyre-libs/examples/jacobi_workgroup_perf.rs` | None | `./cargo_full test -p vyre-libs --example jacobi_workgroup_perf` |
| `example` | `prefix_sum_megakernel` | `vyre-libs/examples/prefix_sum_megakernel.rs` | None | `./cargo_full test -p vyre-libs --example prefix_sum_megakernel` |
| `example` | `select1_optimizer_parity` | `vyre-libs/examples/select1_optimizer_parity.rs` | `bitset` | `./cargo_full test -p vyre-libs --example select1_optimizer_parity` |
| `lib` | `vyre_libs` | `vyre-libs/src/lib.rs` | None | `./cargo_full test -p vyre-libs` |
| `test` | `all_tests` | `vyre-libs/tests/all_tests.rs` | None | `./cargo_full test -p vyre-libs --test all_tests` |
| `test` | `all_tests_fixpoint_graph_math_kernels` | `vyre-libs/tests/all_tests_fixpoint_graph_math_kernels.rs` | `fixpoint`, `graph`, `math-kernels` | `./cargo_full test -p vyre-libs --test all_tests_fixpoint_graph_math_kernels` |
| `test` | `all_tests_fixpoint_graph_math_kernels_parsing_kernels` | `vyre-libs/tests/all_tests_fixpoint_graph_math_kernels_parsing_kernels.rs` | `fixpoint`, `graph`, `math-kernels`, `parsing-kernels` | `./cargo_full test -p vyre-libs --test all_tests_fixpoint_graph_math_kernels_parsing_kernels` |
| `test` | `all_tests_fixpoint_math_kernels` | `vyre-libs/tests/all_tests_fixpoint_math_kernels.rs` | `fixpoint`, `math-kernels` | `./cargo_full test -p vyre-libs --test all_tests_fixpoint_math_kernels` |
| `test` | `all_tests_hash` | `vyre-libs/tests/all_tests_hash.rs` | `hash` | `./cargo_full test -p vyre-libs --test all_tests_hash` |
| `test` | `all_tests_hash_math_nn_activation_nn_linear_pattern` | `vyre-libs/tests/all_tests_hash_math_nn_activation_nn_linear_pattern.rs` | `hash`, `math`, `nn-activation`, `nn-linear`, `pattern` | `./cargo_full test -p vyre-libs --test all_tests_hash_math_nn_activation_nn_linear_pattern` |
| `test` | `all_tests_hash_nn_activation_nn_attention_nn_linear_nn_norm` | `vyre-libs/tests/all_tests_hash_nn_activation_nn_attention_nn_linear_nn_norm.rs` | `hash`, `nn-activation`, `nn-attention`, `nn-linear`, `nn-norm` | `./cargo_full test -p vyre-libs --test all_tests_hash_nn_activation_nn_attention_nn_linear_nn_norm` |
| `test` | `all_tests_hash_nn_attention_nn_norm` | `vyre-libs/tests/all_tests_hash_nn_attention_nn_norm.rs` | `hash`, `nn-attention`, `nn-norm` | `./cargo_full test -p vyre-libs --test all_tests_hash_nn_attention_nn_norm` |
| `test` | `all_tests_math_math_linalg_nn_activation_nn_attention_nn_linear_nn_norm` | `vyre-libs/tests/all_tests_math_math_linalg_nn_activation_nn_attention_nn_linear_nn_norm.rs` | `math`, `math-linalg`, `nn-activation`, `nn-attention`, `nn-linear`, `nn-norm` | `./cargo_full test -p vyre-libs --test all_tests_math_math_linalg_nn_activation_nn_attention_nn_linear_nn_norm` |
| `test` | `all_tests_math_nn_attention` | `vyre-libs/tests/all_tests_math_nn_attention.rs` | `math`, `nn-attention` | `./cargo_full test -p vyre-libs --test all_tests_math_nn_attention` |

## Test classes

- Product-library exact behavior
- Primitive-to-library composition
- Reference and backend parity

## Hardware requirements

Reference and builder suites are host-capable. Tests that request concrete backend parity require that device on the execution host and fail visibly when unavailable.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
