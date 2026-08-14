//! Adversarial tests that expose real semantic gaps in the vyre-reference CPU
//! interpreter. Every assertion documents behavior that was previously untested.

mod flat_expr_eval;

use vyre_foundation::ir::{BinOp, BufferDecl, DataType, Expr, Node, Program, UnOp};
use vyre_reference::{
    execution::expr as eval_expr, execution::expr::Buffer, reference_eval, value::Value,
    workgroup::Memory,
};

use flat_expr_eval::{empty_program, eval_expr_value, float_bits, zero_invocation};

#[path = "contract_cases/adversarial_gaps__program_with_no_buffers_executes_pure_nodes.rs"]
mod adversarial_gaps_program_with_no_buffers_executes_pure_nodes;
#[path = "contract_cases/adversarial_gaps__subnormal_sqrt_sin_cos_produce_canonical_results.rs"]
mod adversarial_gaps_subnormal_sqrt_sin_cos_produce_canonical_results;
