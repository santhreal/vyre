//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/data_type_edge_matrix_cases/mod.rs`.
#[path = "data_type_edge_matrix_cases/mod.rs"]
pub mod data_type_edge_matrix_cases;

/// Shared fixture module from `tests/spec_variants/mod.rs`.
#[path = "spec_variants/mod.rs"]
pub mod spec_variants;

/// Integration tests from `tests/algebraic_law_surface.rs`.
#[path = "algebraic_law_surface.rs"]
pub mod algebraic_law_surface;

/// Integration tests from `tests/bin_op_result_class.rs`.
#[path = "bin_op_result_class.rs"]
pub mod bin_op_result_class;

/// Integration tests from `tests/capability_id_property_contracts.rs`.
#[path = "capability_id_property_contracts.rs"]
pub mod capability_id_property_contracts;

/// Integration tests from `tests/catalog_completeness.rs`.
#[path = "catalog_completeness.rs"]
pub mod catalog_completeness;

/// Integration tests from `tests/category_property_contracts.rs`.
#[path = "category_property_contracts.rs"]
pub mod category_property_contracts;

/// Integration tests from `tests/collective_op_contracts.rs`.
#[path = "collective_op_contracts.rs"]
pub mod collective_op_contracts;

/// Integration tests from `tests/collective_op_property_contracts.rs`.
#[path = "collective_op_property_contracts.rs"]
pub mod collective_op_property_contracts;

/// Integration tests from `tests/collective_property_contracts.rs`.
#[path = "collective_property_contracts.rs"]
pub mod collective_property_contracts;

/// Integration tests from `tests/comm_group_property_contracts.rs`.
#[path = "comm_group_property_contracts.rs"]
pub mod comm_group_property_contracts;

/// Integration tests from `tests/data_type_generated_edge_matrix.rs`.
#[path = "data_type_generated_edge_matrix.rs"]
pub mod data_type_generated_edge_matrix;

/// Integration tests from `tests/data_type_layout_matrix.rs`.
#[path = "data_type_layout_matrix.rs"]
pub mod data_type_layout_matrix;

/// Integration tests from `tests/data_type_min_bytes_property_contracts.rs`.
#[path = "data_type_min_bytes_property_contracts.rs"]
pub mod data_type_min_bytes_property_contracts;

/// Integration tests from `tests/data_type_packed_size_adversarial.rs`.
#[path = "data_type_packed_size_adversarial.rs"]
pub mod data_type_packed_size_adversarial;

/// Integration tests from `tests/data_type_property_contracts.rs`.
#[path = "data_type_property_contracts.rs"]
pub mod data_type_property_contracts;

/// Integration tests from `tests/data_type_surface.rs`.
#[path = "data_type_surface.rs"]
pub mod data_type_surface;

/// Integration tests from `tests/data_type_wire_payload_invariance_generated.rs`.
#[path = "data_type_wire_payload_invariance_generated.rs"]
pub mod data_type_wire_payload_invariance_generated;

/// Integration tests from `tests/extension_collective_category_contracts.rs`.
#[path = "extension_collective_category_contracts.rs"]
pub mod extension_collective_category_contracts;

/// Integration tests from `tests/extension_id_contracts.rs`.
#[path = "extension_id_contracts.rs"]
pub mod extension_id_contracts;

/// Integration tests from `tests/extension_id_generated_matrix.rs`.
#[path = "extension_id_generated_matrix.rs"]
pub mod extension_id_generated_matrix;

/// Integration tests from `tests/extension_id_property_contracts.rs`.
#[path = "extension_id_property_contracts.rs"]
pub mod extension_id_property_contracts;

/// Integration tests from `tests/frozen_discriminants.rs`.
#[path = "frozen_discriminants.rs"]
pub mod frozen_discriminants;

/// Integration tests from `tests/generated_surface_matrix.rs`.
#[path = "generated_surface_matrix.rs"]
pub mod generated_surface_matrix;

/// Integration tests from `tests/intrinsic_descriptor_surface.rs`.
#[path = "intrinsic_descriptor_surface.rs"]
pub mod intrinsic_descriptor_surface;

/// Integration tests from `tests/invariant_catalog_generated_matrix.rs`.
#[path = "invariant_catalog_generated_matrix.rs"]
pub mod invariant_catalog_generated_matrix;

/// Integration tests from `tests/invariant_catalog_surface.rs`.
#[path = "invariant_catalog_surface.rs"]
pub mod invariant_catalog_surface;

/// Integration tests from `tests/invariant_property_contracts.rs`.
#[path = "invariant_property_contracts.rs"]
pub mod invariant_property_contracts;

/// Integration tests from `tests/numeric_semantics_contracts.rs`.
#[path = "numeric_semantics_contracts.rs"]
pub mod numeric_semantics_contracts;

/// Integration tests from `tests/op_contract_surface.rs`.
#[path = "op_contract_surface.rs"]
pub mod op_contract_surface;

/// Integration tests from `tests/op_signature_contract_generated.rs`.
#[path = "op_signature_contract_generated.rs"]
pub mod op_signature_contract_generated;

/// Integration tests from `tests/op_signature_property_contracts.rs`.
#[path = "op_signature_property_contracts.rs"]
pub mod op_signature_property_contracts;

/// Integration tests from `tests/op_wire_property_contracts.rs`.
#[path = "op_wire_property_contracts.rs"]
pub mod op_wire_property_contracts;

/// Integration tests from `tests/operation_contract_property_contracts.rs`.
#[path = "operation_contract_property_contracts.rs"]
pub mod operation_contract_property_contracts;

/// Integration tests from `tests/semiring_property_contracts.rs`.
#[path = "semiring_property_contracts.rs"]
pub mod semiring_property_contracts;

/// Integration tests from `tests/semiring_surface.rs`.
#[path = "semiring_surface.rs"]
pub mod semiring_surface;

/// Integration tests from `tests/serde_contract_surface.rs`.
#[path = "serde_contract_surface.rs"]
pub mod serde_contract_surface;

/// Integration tests from `tests/soundness_contracts.rs`.
#[path = "soundness_contracts.rs"]
pub mod soundness_contracts;

/// Integration tests from `tests/spec_contract_errors.rs`.
#[path = "spec_contract_errors.rs"]
pub mod spec_contract_errors;

/// Integration tests from `tests/spec_variant_tables_cover_the_frozen_surface.rs`.
#[path = "spec_variant_tables_cover_the_frozen_surface.rs"]
pub mod spec_variant_tables_cover_the_frozen_surface;

/// Integration tests from `tests/static_vector_identity_contracts.rs`.
#[path = "static_vector_identity_contracts.rs"]
pub mod static_vector_identity_contracts;

/// Integration tests from `tests/sweep_wire_roundtrip_oracle_matrix.rs`.
#[path = "sweep_wire_roundtrip_oracle_matrix.rs"]
pub mod sweep_wire_roundtrip_oracle_matrix;

/// Integration tests from `tests/sweep_wire_u32_volume_oracle_matrix.rs`.
#[path = "sweep_wire_u32_volume_oracle_matrix.rs"]
pub mod sweep_wire_u32_volume_oracle_matrix;

/// Integration tests from `tests/test_descriptor_property_contracts.rs`.
#[path = "test_descriptor_property_contracts.rs"]
pub mod test_descriptor_property_contracts;

/// Integration tests from `tests/test_descriptor_surface.rs`.
#[path = "test_descriptor_surface.rs"]
pub mod test_descriptor_surface;

/// Integration tests from `tests/token_ids_have_one_owner.rs`.
#[path = "token_ids_have_one_owner.rs"]
pub mod token_ids_have_one_owner;

/// Integration tests from `tests/operation_law_contract_records.rs`.
#[path = "operation_law_contract_records.rs"]
pub mod operation_law_contract_records;
/// Integration tests from `tests/protocol_compatibility_matrix.rs`.
#[path = "protocol_compatibility_matrix.rs"]
pub mod protocol_compatibility_matrix;
/// Integration tests from `tests/resource_capability_contracts.rs`.
#[path = "resource_capability_contracts.rs"]
pub mod resource_capability_contracts;
/// Integration tests from `tests/wire_tag_surface.rs`.
#[path = "wire_tag_surface.rs"]
pub mod wire_tag_surface;
