//! Contract tests for c ast asm extended operand goto label pg lowering contracts.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "c_ast_asm_extended_operand_goto_label_pg_lowering_contracts/classify.rs"]
mod classify;
#[path = "c_ast_asm_extended_operand_goto_label_pg_lowering_contracts/pg_lower_preserves_asm_symbolic_names_and_earlyclobber.rs"]
mod pg_lower_preserves_asm_symbolic_names_and_earlyclobber;
