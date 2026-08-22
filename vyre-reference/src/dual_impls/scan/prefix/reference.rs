use crate::{dual_impls::evaluator, workgroup::Memory};
use vyre_primitives::Scan;

impl evaluator::ReferenceEvaluator for Scan {
    fn evaluate(&self, inputs: &[Memory]) -> Result<Memory, evaluator::EvalError> {
        let words = evaluator::u32_words(evaluator::one_input(inputs, "scan")?, "scan")?;
        let mut iter = words.into_iter();
        let Some(first) = iter.next() else {
            return Ok(Memory::from_bytes(Vec::new()));
        };
        let mut acc = first;
        let mut output = vec![acc];
        for value in iter {
            acc = evaluator::combine(self.combine, acc, value)?;
            output.push(acc);
        }
        Ok(evaluator::write_u32s(output))
    }
}
