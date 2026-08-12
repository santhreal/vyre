//! CPU-only reference tests for Linux-grade C constructs not covered
//! by the existing GNU extension test suite.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
include!("contract_cases/c_ast_linux_grade_gnu_and_c11_construct_coverage__new.rs");
include!("contract_cases/c_ast_linux_grade_gnu_and_c11_construct_coverage__cpu_reference_attribute_constructor_parses.rs");
