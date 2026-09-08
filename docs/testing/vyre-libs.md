# Testing `vyre-libs`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-libs
```

Own every composition in the workspace: consumer dialects and compiler-internal solvers, encoding, analysis, scheduling, and reasoning. Returns Programs. No backend, no emitter, no host rewrite of IR.

The crate lives at `vyre-libs`. The `product-libraries` owner maintains its
`libraries` testing contract.

## Commands

```console
./cargo_full test -p vyre-libs
```

```console
./cargo_full test -p vyre-libs --all-features
```

## Feature sets

- Default feature members: `math-linalg`, `math-scan`, `math-broadcast`, `nn-activation`, `nn-linear`, `nn-norm`, `pattern-substring`, `pattern-dfa`, `hash`, `decode`
- Available manifest features: `analysis`, `bitset`, `builder`, `builder-ops`, `cat-a-builder-options`, `crypto`, `crypto-blake3`, `decode`, `default`, `device`, `encoding`, `fixpoint`, `full`, `geom`, `go-parser`, `graph`, `graph-dispatch`, `hash`, `label`, `llm`, `logical`, `math`, `math-algebra`, `math-broadcast`, `math-dialect`, `math-kernels`, `math-linalg`, `math-scan`, `math-succinct`, `nfa`, `nn`, `nn-activation`, `nn-attention`, `nn-inference`, `nn-kernels`, `nn-linear`, `nn-linear-4bit`, `nn-moe`, `nn-norm`, `opt`, `parsing`, `parsing-kernels`, `pattern`, `pattern-dfa`, `pattern-kernels`, `pattern-nfa`, `pattern-regex`, `pattern-substring`, `predicate`, `python-parser`, `reasoning`, `reduce`, `representation`, `rule`, `scheduling`, `security`, `solvers`, `telemetry`, `test-fixtures`, `text`, `topology`, `vfs`, `visual`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `dominator_tree_e2e` | `vyre-libs/examples/dominator_tree_e2e.rs` | `graph` | `./cargo_full test -p vyre-libs --example dominator_tree_e2e` |
| `example` | `jacobi_workgroup_perf` | `vyre-libs/examples/jacobi_workgroup_perf.rs` | None | `./cargo_full test -p vyre-libs --example jacobi_workgroup_perf` |
| `example` | `prefix_sum_megakernel` | `vyre-libs/examples/prefix_sum_megakernel.rs` | `math-scan` | `./cargo_full test -p vyre-libs --example prefix_sum_megakernel` |
| `example` | `select1_optimizer_parity` | `vyre-libs/examples/select1_optimizer_parity.rs` | `bitset` | `./cargo_full test -p vyre-libs --example select1_optimizer_parity` |
| `lib` | `vyre_libs` | `vyre-libs/src/lib.rs` | None | `./cargo_full test -p vyre-libs` |
| `test` | `all_tests` | `vyre-libs/tests/all_tests.rs` | None | `./cargo_full test -p vyre-libs --test all_tests` |
| `test` | `all_tests_analysis` | `vyre-libs/tests/all_tests_analysis.rs` | `analysis` | `./cargo_full test -p vyre-libs --test all_tests_analysis` |
| `test` | `all_tests_analysis_encoding` | `vyre-libs/tests/all_tests_analysis_encoding.rs` | `analysis`, `encoding` | `./cargo_full test -p vyre-libs --test all_tests_analysis_encoding` |
| `test` | `all_tests_bitset` | `vyre-libs/tests/all_tests_bitset.rs` | `bitset` | `./cargo_full test -p vyre-libs --test all_tests_bitset` |
| `test` | `all_tests_decode` | `vyre-libs/tests/all_tests_decode.rs` | `decode` | `./cargo_full test -p vyre-libs --test all_tests_decode` |
| `test` | `all_tests_decode_parsing` | `vyre-libs/tests/all_tests_decode_parsing.rs` | `decode`, `parsing` | `./cargo_full test -p vyre-libs --test all_tests_decode_parsing` |
| `test` | `all_tests_encoding` | `vyre-libs/tests/all_tests_encoding.rs` | `encoding` | `./cargo_full test -p vyre-libs --test all_tests_encoding` |
| `test` | `all_tests_fixpoint` | `vyre-libs/tests/all_tests_fixpoint.rs` | `fixpoint` | `./cargo_full test -p vyre-libs --test all_tests_fixpoint` |
| `test` | `all_tests_fixpoint_graph_math_kernels` | `vyre-libs/tests/all_tests_fixpoint_graph_math_kernels.rs` | `fixpoint`, `graph`, `math-kernels` | `./cargo_full test -p vyre-libs --test all_tests_fixpoint_graph_math_kernels` |
| `test` | `all_tests_fixpoint_graph_math_kernels_parsing_kernels` | `vyre-libs/tests/all_tests_fixpoint_graph_math_kernels_parsing_kernels.rs` | `fixpoint`, `graph`, `math-kernels`, `parsing-kernels` | `./cargo_full test -p vyre-libs --test all_tests_fixpoint_graph_math_kernels_parsing_kernels` |
| `test` | `all_tests_fixpoint_math_kernels` | `vyre-libs/tests/all_tests_fixpoint_math_kernels.rs` | `fixpoint`, `math-kernels` | `./cargo_full test -p vyre-libs --test all_tests_fixpoint_math_kernels` |
| `test` | `all_tests_graph` | `vyre-libs/tests/all_tests_graph.rs` | `graph` | `./cargo_full test -p vyre-libs --test all_tests_graph` |
| `test` | `all_tests_graph_dispatch` | `vyre-libs/tests/all_tests_graph_dispatch.rs` | `graph-dispatch` | `./cargo_full test -p vyre-libs --test all_tests_graph_dispatch` |
| `test` | `all_tests_hash` | `vyre-libs/tests/all_tests_hash.rs` | `hash` | `./cargo_full test -p vyre-libs --test all_tests_hash` |
| `test` | `all_tests_hash_math_nn_activation_nn_linear_pattern` | `vyre-libs/tests/all_tests_hash_math_nn_activation_nn_linear_pattern.rs` | `hash`, `math`, `nn-activation`, `nn-linear`, `pattern` | `./cargo_full test -p vyre-libs --test all_tests_hash_math_nn_activation_nn_linear_pattern` |
| `test` | `all_tests_hash_nn_activation_nn_attention_nn_linear_nn_norm` | `vyre-libs/tests/all_tests_hash_nn_activation_nn_attention_nn_linear_nn_norm.rs` | `hash`, `nn-activation`, `nn-attention`, `nn-linear`, `nn-norm` | `./cargo_full test -p vyre-libs --test all_tests_hash_nn_activation_nn_attention_nn_linear_nn_norm` |
| `test` | `all_tests_hash_nn_attention_nn_norm` | `vyre-libs/tests/all_tests_hash_nn_attention_nn_norm.rs` | `hash`, `nn-attention`, `nn-norm` | `./cargo_full test -p vyre-libs --test all_tests_hash_nn_attention_nn_norm` |
| `test` | `all_tests_label` | `vyre-libs/tests/all_tests_label.rs` | `label` | `./cargo_full test -p vyre-libs --test all_tests_label` |
| `test` | `all_tests_llm` | `vyre-libs/tests/all_tests_llm.rs` | `llm` | `./cargo_full test -p vyre-libs --test all_tests_llm` |
| `test` | `all_tests_logical` | `vyre-libs/tests/all_tests_logical.rs` | `logical` | `./cargo_full test -p vyre-libs --test all_tests_logical` |
| `test` | `all_tests_math` | `vyre-libs/tests/all_tests_math.rs` | `math` | `./cargo_full test -p vyre-libs --test all_tests_math` |
| `test` | `all_tests_math_kernels` | `vyre-libs/tests/all_tests_math_kernels.rs` | `math-kernels` | `./cargo_full test -p vyre-libs --test all_tests_math_kernels` |
| `test` | `all_tests_math_math_linalg_nn_activation_nn_attention_nn_linear_nn_norm` | `vyre-libs/tests/all_tests_math_math_linalg_nn_activation_nn_attention_nn_linear_nn_norm.rs` | `math`, `math-linalg`, `nn-activation`, `nn-attention`, `nn-linear`, `nn-norm` | `./cargo_full test -p vyre-libs --test all_tests_math_math_linalg_nn_activation_nn_attention_nn_linear_nn_norm` |
| `test` | `all_tests_math_nn_attention` | `vyre-libs/tests/all_tests_math_nn_attention.rs` | `math`, `nn-attention` | `./cargo_full test -p vyre-libs --test all_tests_math_nn_attention` |
| `test` | `all_tests_math_scan` | `vyre-libs/tests/all_tests_math_scan.rs` | `math-scan` | `./cargo_full test -p vyre-libs --test all_tests_math_scan` |
| `test` | `all_tests_nfa` | `vyre-libs/tests/all_tests_nfa.rs` | `nfa` | `./cargo_full test -p vyre-libs --test all_tests_nfa` |
| `test` | `all_tests_nn_activation` | `vyre-libs/tests/all_tests_nn_activation.rs` | `nn-activation` | `./cargo_full test -p vyre-libs --test all_tests_nn_activation` |
| `test` | `all_tests_nn_activation_nn_linear` | `vyre-libs/tests/all_tests_nn_activation_nn_linear.rs` | `nn-activation`, `nn-linear` | `./cargo_full test -p vyre-libs --test all_tests_nn_activation_nn_linear` |
| `test` | `all_tests_nn_attention` | `vyre-libs/tests/all_tests_nn_attention.rs` | `nn-attention` | `./cargo_full test -p vyre-libs --test all_tests_nn_attention` |
| `test` | `all_tests_nn_attention_nn_linear_nn_norm` | `vyre-libs/tests/all_tests_nn_attention_nn_linear_nn_norm.rs` | `nn-attention`, `nn-linear`, `nn-norm` | `./cargo_full test -p vyre-libs --test all_tests_nn_attention_nn_linear_nn_norm` |
| `test` | `all_tests_nn_attention_nn_norm` | `vyre-libs/tests/all_tests_nn_attention_nn_norm.rs` | `nn-attention`, `nn-norm` | `./cargo_full test -p vyre-libs --test all_tests_nn_attention_nn_norm` |
| `test` | `all_tests_nn_inference` | `vyre-libs/tests/all_tests_nn_inference.rs` | `nn-inference` | `./cargo_full test -p vyre-libs --test all_tests_nn_inference` |
| `test` | `all_tests_nn_linear` | `vyre-libs/tests/all_tests_nn_linear.rs` | `nn-linear` | `./cargo_full test -p vyre-libs --test all_tests_nn_linear` |
| `test` | `all_tests_nn_norm` | `vyre-libs/tests/all_tests_nn_norm.rs` | `nn-norm` | `./cargo_full test -p vyre-libs --test all_tests_nn_norm` |
| `test` | `all_tests_parsing` | `vyre-libs/tests/all_tests_parsing.rs` | `parsing` | `./cargo_full test -p vyre-libs --test all_tests_parsing` |
| `test` | `all_tests_parsing_kernels` | `vyre-libs/tests/all_tests_parsing_kernels.rs` | `parsing-kernels` | `./cargo_full test -p vyre-libs --test all_tests_parsing_kernels` |
| `test` | `all_tests_pattern` | `vyre-libs/tests/all_tests_pattern.rs` | `pattern` | `./cargo_full test -p vyre-libs --test all_tests_pattern` |
| `test` | `all_tests_pattern_kernels` | `vyre-libs/tests/all_tests_pattern_kernels.rs` | `pattern-kernels` | `./cargo_full test -p vyre-libs --test all_tests_pattern_kernels` |
| `test` | `all_tests_pattern_regex` | `vyre-libs/tests/all_tests_pattern_regex.rs` | `pattern-regex` | `./cargo_full test -p vyre-libs --test all_tests_pattern_regex` |
| `test` | `all_tests_pattern_substring` | `vyre-libs/tests/all_tests_pattern_substring.rs` | `pattern-substring` | `./cargo_full test -p vyre-libs --test all_tests_pattern_substring` |
| `test` | `all_tests_predicate` | `vyre-libs/tests/all_tests_predicate.rs` | `predicate` | `./cargo_full test -p vyre-libs --test all_tests_predicate` |
| `test` | `all_tests_reasoning` | `vyre-libs/tests/all_tests_reasoning.rs` | `reasoning` | `./cargo_full test -p vyre-libs --test all_tests_reasoning` |
| `test` | `all_tests_reduce` | `vyre-libs/tests/all_tests_reduce.rs` | `reduce` | `./cargo_full test -p vyre-libs --test all_tests_reduce` |
| `test` | `all_tests_reduce_telemetry` | `vyre-libs/tests/all_tests_reduce_telemetry.rs` | `reduce`, `telemetry` | `./cargo_full test -p vyre-libs --test all_tests_reduce_telemetry` |
| `test` | `all_tests_rule` | `vyre-libs/tests/all_tests_rule.rs` | `rule` | `./cargo_full test -p vyre-libs --test all_tests_rule` |
| `test` | `all_tests_scheduling` | `vyre-libs/tests/all_tests_scheduling.rs` | `scheduling` | `./cargo_full test -p vyre-libs --test all_tests_scheduling` |
| `test` | `all_tests_security` | `vyre-libs/tests/all_tests_security.rs` | `security` | `./cargo_full test -p vyre-libs --test all_tests_security` |
| `test` | `all_tests_solvers` | `vyre-libs/tests/all_tests_solvers.rs` | `solvers` | `./cargo_full test -p vyre-libs --test all_tests_solvers` |
| `test` | `all_tests_text` | `vyre-libs/tests/all_tests_text.rs` | `text` | `./cargo_full test -p vyre-libs --test all_tests_text` |

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
