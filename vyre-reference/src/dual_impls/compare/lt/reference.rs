use crate::{dual_impls::evaluator, workgroup::Memory};
use vyre_primitives::CompareLt;

impl evaluator::ReferenceEvaluator for CompareLt {
    fn evaluate(&self, inputs: &[Memory]) -> Result<Memory, evaluator::EvalError> {
        evaluator::binary_u32_predicate(inputs, "compare_lt", |left, right| left < right)
    }
}
