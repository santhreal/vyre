//! Adversarial tests that expose real semantic gaps in the vyre-reference CPU
//! interpreter. Every assertion documents behavior that was previously untested.

use crate::flat_expr_eval;

use vyre_foundation::ir::{BinOp, BufferDecl, DataType, Expr, Node, Program, UnOp};
use vyre_reference::expr::Buffer;
use vyre_reference::{expr as eval_expr, reference_eval, value::Value, workgroup::Memory, ReferenceBudget, ReferenceRequest, ReferenceResponse};

use flat_expr_eval::{empty_program, eval_expr_value, float_bits, zero_invocation};

/// Helper for evaluating a program using the standard ReferenceRequest.
pub fn eval_program(
    program: &Program,
    inputs: &[Value],
) -> Result<ReferenceResponse, vyre_reference::ReferenceError> {
    let req = ReferenceRequest::new(program, inputs, ReferenceBudget::standard());
    reference_eval(&req)
}
#[path = "contract_cases/adversarial_gaps__program_with_no_buffers_executes_pure_nodes.rs"]
mod adversarial_gaps_program_with_no_buffers_executes_pure_nodes;
#[path = "contract_cases/adversarial_gaps__subnormal_sqrt_sin_cos_produce_canonical_results.rs"]
mod adversarial_gaps_subnormal_sqrt_sin_cos_produce_canonical_results;
