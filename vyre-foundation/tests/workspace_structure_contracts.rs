//! Workspace-structure contract checks. Implementation lives in two
//! `include!`-d chunks under `contract_cases/`.
include!("contract_cases/workspace_structure_contracts__foundation_no_nested_tests_in_src.rs");
include!(
    "contract_cases/workspace_structure_contracts__op_id_literals_match_their_catalog_tier.rs"
);
