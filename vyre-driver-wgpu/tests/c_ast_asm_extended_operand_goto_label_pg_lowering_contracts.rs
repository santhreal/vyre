//! Property-graph lowering of GNU extended asm: symbolic operand names, earlyclobber, clobber
//! lists, and asm goto labels.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
mod c_ast_gpu_parity_support;
#[path = "c_ast_asm_extended_operand_goto_label_pg_lowering_contracts/classify.rs"]
mod classify;
#[path = "c_ast_asm_extended_operand_goto_label_pg_lowering_contracts/pg_lowering_and_gpu_parity.rs"]
mod pg_lowering_and_gpu_parity;
