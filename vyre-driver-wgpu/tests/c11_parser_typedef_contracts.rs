//! Generated wrapper test crate for c11 parser typedef contracts.
//!
//! Implementation lives in `contract_cases/` chunks.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
include!("contract_cases/c11_parser_typedef_contracts__assert_kind.rs");
include!(
    "contract_cases/c11_parser_typedef_contracts__pg_lower_preserves_typedef_cast_vs_expr_kinds.rs"
);
