//! Test: c ast property typedef annotation contracts.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
include!(
    "contract_cases/c_ast_property_typedef_annotation_contracts__run_gpu_typedef_annotation.rs"
);
include!("contract_cases/c_ast_property_typedef_annotation_contracts__arb_atom.rs");
