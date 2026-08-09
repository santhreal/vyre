# Testing `vyre-spec`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec
```

Own stable schemas, operation definitions, and compatibility contracts without runtime dependencies.

The crate lives at `vyre-spec`. The `specification` owner maintains its
`foundation` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_spec_release_surface` | `vyre-spec/examples/vyre_spec_release_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --example vyre_spec_release_surface` |
| `lib` | `vyre_spec` | `vyre-spec/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec` |
| `test` | `algebraic_law_surface` | `vyre-spec/tests/algebraic_law_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test algebraic_law_surface` |
| `test` | `capability_id_property_contracts` | `vyre-spec/tests/capability_id_property_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test capability_id_property_contracts` |
| `test` | `catalog_completeness` | `vyre-spec/tests/catalog_completeness.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test catalog_completeness` |
| `test` | `category_property_contracts` | `vyre-spec/tests/category_property_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test category_property_contracts` |
| `test` | `collective_op_contracts` | `vyre-spec/tests/collective_op_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test collective_op_contracts` |
| `test` | `collective_op_property_contracts` | `vyre-spec/tests/collective_op_property_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test collective_op_property_contracts` |
| `test` | `collective_property_contracts` | `vyre-spec/tests/collective_property_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test collective_property_contracts` |
| `test` | `comm_group_property_contracts` | `vyre-spec/tests/comm_group_property_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test comm_group_property_contracts` |
| `test` | `cuda_resident_dispatch_hot_path_waivers` | `vyre-spec/tests/cuda_resident_dispatch_hot_path_waivers.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test cuda_resident_dispatch_hot_path_waivers` |
| `test` | `data_type_generated_edge_matrix` | `vyre-spec/tests/data_type_generated_edge_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test data_type_generated_edge_matrix` |
| `test` | `data_type_generated_edge_matrix_support` | `vyre-spec/tests/data_type_generated_edge_matrix_support.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test data_type_generated_edge_matrix_support` |
| `test` | `data_type_layout_matrix` | `vyre-spec/tests/data_type_layout_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test data_type_layout_matrix` |
| `test` | `data_type_min_bytes_property_contracts` | `vyre-spec/tests/data_type_min_bytes_property_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test data_type_min_bytes_property_contracts` |
| `test` | `data_type_packed_size_adversarial` | `vyre-spec/tests/data_type_packed_size_adversarial.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test data_type_packed_size_adversarial` |
| `test` | `data_type_property_contracts` | `vyre-spec/tests/data_type_property_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test data_type_property_contracts` |
| `test` | `data_type_surface` | `vyre-spec/tests/data_type_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test data_type_surface` |
| `test` | `data_type_wire_payload_invariance_generated` | `vyre-spec/tests/data_type_wire_payload_invariance_generated.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test data_type_wire_payload_invariance_generated` |
| `test` | `extension_collective_category_contracts` | `vyre-spec/tests/extension_collective_category_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test extension_collective_category_contracts` |
| `test` | `extension_id_contracts` | `vyre-spec/tests/extension_id_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test extension_id_contracts` |
| `test` | `extension_id_generated_matrix` | `vyre-spec/tests/extension_id_generated_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test extension_id_generated_matrix` |
| `test` | `extension_id_property_contracts` | `vyre-spec/tests/extension_id_property_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test extension_id_property_contracts` |
| `test` | `frozen_discriminants` | `vyre-spec/tests/frozen_discriminants.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test frozen_discriminants` |
| `test` | `generated_surface_matrix` | `vyre-spec/tests/generated_surface_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test generated_surface_matrix` |
| `test` | `intrinsic_descriptor_surface` | `vyre-spec/tests/intrinsic_descriptor_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test intrinsic_descriptor_surface` |
| `test` | `invariant_catalog_generated_matrix` | `vyre-spec/tests/invariant_catalog_generated_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test invariant_catalog_generated_matrix` |
| `test` | `invariant_catalog_surface` | `vyre-spec/tests/invariant_catalog_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test invariant_catalog_surface` |
| `test` | `invariant_property_contracts` | `vyre-spec/tests/invariant_property_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test invariant_property_contracts` |
| `test` | `op_contract_surface` | `vyre-spec/tests/op_contract_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test op_contract_surface` |
| `test` | `op_signature_contract_generated` | `vyre-spec/tests/op_signature_contract_generated.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test op_signature_contract_generated` |
| `test` | `op_signature_property_contracts` | `vyre-spec/tests/op_signature_property_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test op_signature_property_contracts` |
| `test` | `op_wire_property_contracts` | `vyre-spec/tests/op_wire_property_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test op_wire_property_contracts` |
| `test` | `operation_contract_property_contracts` | `vyre-spec/tests/operation_contract_property_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test operation_contract_property_contracts` |
| `test` | `semiring_property_contracts` | `vyre-spec/tests/semiring_property_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test semiring_property_contracts` |
| `test` | `semiring_surface` | `vyre-spec/tests/semiring_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test semiring_surface` |
| `test` | `serde_contract_surface` | `vyre-spec/tests/serde_contract_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test serde_contract_surface` |
| `test` | `soundness_contracts` | `vyre-spec/tests/soundness_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test soundness_contracts` |
| `test` | `spec_contract_errors` | `vyre-spec/tests/spec_contract_errors.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test spec_contract_errors` |
| `test` | `static_vector_identity_contracts` | `vyre-spec/tests/static_vector_identity_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test static_vector_identity_contracts` |
| `test` | `sweep_wire_roundtrip_oracle_matrix` | `vyre-spec/tests/sweep_wire_roundtrip_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test sweep_wire_roundtrip_oracle_matrix` |
| `test` | `sweep_wire_u32_volume_oracle_matrix` | `vyre-spec/tests/sweep_wire_u32_volume_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test sweep_wire_u32_volume_oracle_matrix` |
| `test` | `test_descriptor_property_contracts` | `vyre-spec/tests/test_descriptor_property_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test test_descriptor_property_contracts` |
| `test` | `test_descriptor_surface` | `vyre-spec/tests/test_descriptor_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test test_descriptor_surface` |
| `test` | `wire_tag_reservation_manifest` | `vyre-spec/tests/wire_tag_reservation_manifest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test wire_tag_reservation_manifest` |
| `test` | `wire_tag_surface` | `vyre-spec/tests/wire_tag_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-spec --test wire_tag_surface` |

## Test classes

- IR construction and serialization contracts
- Validation and optimizer semantics
- Adversarial, property, and compatibility tests

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
