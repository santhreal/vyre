use crate::{dual_impls::evaluator, workgroup::Memory};
use vyre_primitives::PatternMatchLiteral;

impl evaluator::ReferenceEvaluator for PatternMatchLiteral {
    fn evaluate(&self, inputs: &[Memory]) -> Result<Memory, evaluator::EvalError> {
        let haystack = evaluator::one_input(inputs, "scan_literal")?;
        if self.literal.is_empty() {
            return Err(evaluator::EvalError::new(
                "primitive `scan_literal` has empty literal. Fix: pass a non-empty literal.",
            ));
        }
        let mut offsets = Vec::new();
        for offset in 0..=haystack.len().saturating_sub(self.literal.len()) {
            if haystack[offset..].starts_with(&self.literal) {
                offsets.push(u32::try_from(offset).map_err(|_| {
                    evaluator::EvalError::new(
                        "primitive `scan_literal` offset exceeds u32. Fix: split haystacks before 4 GiB.",
                    )
                })?);
            }
        }
        Ok(evaluator::write_u32s(offsets))
    }
}
