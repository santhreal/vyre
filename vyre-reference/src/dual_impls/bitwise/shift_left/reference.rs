use crate::{dual_impls::evaluator, workgroup::Memory};
use vyre_primitives::ShiftLeft;

impl evaluator::ReferenceEvaluator for ShiftLeft {
    fn evaluate(&self, inputs: &[Memory]) -> Result<Memory, evaluator::EvalError> {
        evaluator::binary_u32_scalar(inputs, "shift_left", |left, right| left << (right & 31))
    }
}
