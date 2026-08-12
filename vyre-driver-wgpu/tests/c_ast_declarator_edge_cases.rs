//! GPU/CPU parity tests for difficult C declarator edge cases.
//! Implementation lives in `contract_cases/` chunks.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_declarator_edge_cases_suite {
    include!("contract_cases/c_ast_declarator_edge_cases_support.rs");
    mod c_ast_declarator_edge_cases_cpu_array_of_function_pointers_kinds {
        include!(
            "contract_cases/c_ast_declarator_edge_cases__cpu_array_of_function_pointers_kinds.rs"
        );
    }
    mod c_ast_declarator_edge_cases_gpu_parity_classifier_abstract_declarator_cast {
        include!("contract_cases/c_ast_declarator_edge_cases__gpu_parity_classifier_abstract_declarator_cast.rs");
    }
}
