//! CPU-only reference tests for Linux-grade C constructs not covered
//! by the existing GNU extension test suite.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]

#[path = "contract_cases/c_ast_linux_grade_gnu_and_c11_construct_coverage__new.rs"]
mod c_ast_linux_grade_gnu_and_c11_construct_coverage_new;
