use crate::{dual_impls::evaluator, workgroup::Memory};
use vyre_primitives::HashBlake3;

impl evaluator::ReferenceEvaluator for HashBlake3 {
    fn evaluate(&self, inputs: &[Memory]) -> Result<Memory, evaluator::EvalError> {
        let input = evaluator::one_input(inputs, "hash_blake3")?;
        Ok(Memory::from_bytes(blake3::hash(&input).as_bytes().to_vec()))
    }
}
