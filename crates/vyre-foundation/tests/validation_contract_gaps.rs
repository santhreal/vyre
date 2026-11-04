//! Failure-oriented tests for validator gaps not covered by other suites.
//!
//! Each test constructs a single malformed program and asserts that the
//! validator emits exactly the expected contract-error message. No silent
//! fake paths are allowed.

use vyre_foundation::ir::MemoryOrdering;
use vyre_foundation::ir::{AtomicOp, BinOp, BufferDecl, DataType, Expr, Node, Program, UnOp};
use vyre_foundation::validate::validate;

fn output_program(nodes: Vec<Node>) -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(4)],
        [1, 1, 1],
        nodes,
    )
}

#[path = "contract_cases/validation_contract_gaps__unop_logical_not_on_f32_is_rejected.rs"]
mod validation_contract_gaps_unop_logical_not_on_f32_is_rejected;
#[path = "contract_cases/validation_contract_gaps__workgroup_size_zero_is_rejected.rs"]
mod validation_contract_gaps_workgroup_size_zero_is_rejected;
