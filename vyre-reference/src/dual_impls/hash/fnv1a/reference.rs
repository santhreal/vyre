use crate::{dual_impls::evaluator, workgroup::Memory};
use vyre_libs::hash::fnv1a::fnv1a32;
use vyre_primitives::HashFnv1a;

impl evaluator::ReferenceEvaluator for HashFnv1a {
    fn evaluate(&self, inputs: &[Memory]) -> Result<Memory, evaluator::EvalError> {
        let input = evaluator::one_input(inputs, "hash_fnv1a")?;
        let hash = fnv1a32(&input);
        Ok(Memory::from_bytes(hash.to_le_bytes().to_vec()))
    }
}
