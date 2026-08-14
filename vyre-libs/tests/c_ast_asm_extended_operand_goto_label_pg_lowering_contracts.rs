//! Property-graph lowering of GNU extended asm: symbolic operand names, earlyclobber, clobber
//! lists, and asm goto labels.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/asm_extended_operands.rs"]
mod asm_extended_operands;
#[path = "c_ast_asm_extended_operand_goto_label_pg_lowering_contracts/classify.rs"]
mod classify;
