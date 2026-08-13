use crate::{dual_impls::evaluator, workgroup::Memory};
use vyre_primitives::CompareEq;

impl evaluator::ReferenceEvaluator for CompareEq {
    fn evaluate(&self, inputs: &[Memory]) -> Result<Memory, evaluator::EvalError> {
        evaluator::binary_u32_predicate(inputs, "compare_eq", |left, right| left == right)
    }
}
