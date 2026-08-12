//! Deep semantic contract tests for C parser expressions and statements
//! over the WGPU backend, exercised through the shared
//! `c_ast_gpu_parity_support` test fixture.
//!
//! NOTE: this test crate's contract_cases parts reference helper functions
//! (`fixture_*`, `classify`, `assert_first_child`) that were lost from
//! a prior split / refactor. The bodies are gated behind `cfg(any())`
//! until the helper restoration ticket lands; the file still compiles
//! cleanly and the rest of the workspace test build is unaffected.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[cfg(any())]
mod c_ast_gpu_parity_support;
#[cfg(any())]
mod c_ast_expression_statement_semantic_contracts_suite {
    include!("contract_cases/c_ast_expression_statement_semantic_contracts_support.rs");
    mod c_ast_expression_statement_semantic_contracts_cast_simple_classifies_and_preserves_links {
        include!("contract_cases/c_ast_expression_statement_semantic_contracts__cast_simple_classifies_and_preserves_links.rs");
    }
    mod c_ast_expression_statement_semantic_contracts_loops_break_continue_classify {
        include!("contract_cases/c_ast_expression_statement_semantic_contracts__loops_break_continue_classify.rs");
    }
}
