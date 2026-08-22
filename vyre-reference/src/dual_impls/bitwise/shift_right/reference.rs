use crate::{dual_impls::evaluator, workgroup::Memory};
use vyre_primitives::ShiftRight;

impl evaluator::ReferenceEvaluator for ShiftRight {
    fn evaluate(&self, inputs: &[Memory]) -> Result<Memory, evaluator::EvalError> {
        evaluator::binary_u32_scalar(inputs, "shift_right", |left, right| left >> (right & 31))
    }
}
