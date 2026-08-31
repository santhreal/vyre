# Testing `vyre-megakernel`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-megakernel
```

Construct foundation-owned selected schedules through bounded whole-ProgramGraph search, and own immutable Artifact identity and authenticated TargetPayload construction. Does not own logical semantics, schedule schemas, physical-kernel lowering, admission, execution, or lifecycle policy.

The crate lives at `vyre-megakernel`. The `megakernel-compiler` owner maintains its
`compiler-boundary` testing contract.

## Commands

```console
./cargo_full test -p vyre-megakernel
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_megakernel` | `vyre-megakernel/src/lib.rs` | None | `./cargo_full test -p vyre-megakernel` |
| `test` | `algebraic_equivalence` | `vyre-megakernel/tests/algebraic_equivalence.rs` | None | `./cargo_full test -p vyre-megakernel --test algebraic_equivalence` |
| `test` | `artifact_contract` | `vyre-megakernel/tests/artifact_contract.rs` | None | `./cargo_full test -p vyre-megakernel --test artifact_contract` |
| `test` | `candidate_budget_and_dependency_endpoints` | `vyre-megakernel/tests/candidate_budget_and_dependency_endpoints.rs` | None | `./cargo_full test -p vyre-megakernel --test candidate_budget_and_dependency_endpoints` |
| `test` | `compile_objective` | `vyre-megakernel/tests/compile_objective.rs` | None | `./cargo_full test -p vyre-megakernel --test compile_objective` |
| `test` | `compile_portfolio` | `vyre-megakernel/tests/compile_portfolio.rs` | None | `./cargo_full test -p vyre-megakernel --test compile_portfolio` |
| `test` | `frontier_topology_selection` | `vyre-megakernel/tests/frontier_topology_selection.rs` | None | `./cargo_full test -p vyre-megakernel --test frontier_topology_selection` |
| `test` | `graph_result_values` | `vyre-megakernel/tests/graph_result_values.rs` | None | `./cargo_full test -p vyre-megakernel --test graph_result_values` |
| `test` | `law_derived_candidates` | `vyre-megakernel/tests/law_derived_candidates.rs` | None | `./cargo_full test -p vyre-megakernel --test law_derived_candidates` |
| `test` | `level_stage_verdict` | `vyre-megakernel/tests/level_stage_verdict.rs` | None | `./cargo_full test -p vyre-megakernel --test level_stage_verdict` |
| `test` | `measurement_protocol` | `vyre-megakernel/tests/measurement_protocol.rs` | None | `./cargo_full test -p vyre-megakernel --test measurement_protocol` |
| `test` | `mesh_topology_contract` | `vyre-megakernel/tests/mesh_topology_contract.rs` | None | `./cargo_full test -p vyre-megakernel --test mesh_topology_contract` |
| `test` | `multi_fidelity_ladder` | `vyre-megakernel/tests/multi_fidelity_ladder.rs` | None | `./cargo_full test -p vyre-megakernel --test multi_fidelity_ladder` |
| `test` | `numeric_budget_legality` | `vyre-megakernel/tests/numeric_budget_legality.rs` | None | `./cargo_full test -p vyre-megakernel --test numeric_budget_legality` |
| `test` | `schedule_grammar_contract` | `vyre-megakernel/tests/schedule_grammar_contract.rs` | None | `./cargo_full test -p vyre-megakernel --test schedule_grammar_contract` |
| `test` | `selected_geometry_authority` | `vyre-megakernel/tests/selected_geometry_authority.rs` | None | `./cargo_full test -p vyre-megakernel --test selected_geometry_authority` |
| `test` | `selection_cost_contract` | `vyre-megakernel/tests/selection_cost_contract.rs` | None | `./cargo_full test -p vyre-megakernel --test selection_cost_contract` |
| `test` | `shared_tile_cost` | `vyre-megakernel/tests/shared_tile_cost.rs` | None | `./cargo_full test -p vyre-megakernel --test shared_tile_cost` |
| `test` | `specialization_contract` | `vyre-megakernel/tests/specialization_contract.rs` | None | `./cargo_full test -p vyre-megakernel --test specialization_contract` |
| `test` | `specialization_portfolio` | `vyre-megakernel/tests/specialization_portfolio.rs` | None | `./cargo_full test -p vyre-megakernel --test specialization_portfolio` |
| `test` | `target_payload_contract` | `vyre-megakernel/tests/target_payload_contract.rs` | None | `./cargo_full test -p vyre-megakernel --test target_payload_contract` |
| `test` | `topology_contract` | `vyre-megakernel/tests/topology_contract.rs` | None | `./cargo_full test -p vyre-megakernel --test topology_contract` |

## Test classes

- Megakernel artifact compilation contracts
- Static and persistent plan validation
- Invalid program and boundary rejection

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
