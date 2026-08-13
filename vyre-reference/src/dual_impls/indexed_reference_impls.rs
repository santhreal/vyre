use crate::{dual_impls::evaluator, workgroup::Memory};
use vyre_primitives::{Gather, Shuffle};

macro_rules! impl_indexed_select_reference {
    ($type:ty, $name:literal) => {
        impl evaluator::ReferenceEvaluator for $type {
            fn evaluate(&self, inputs: &[Memory]) -> Result<Memory, evaluator::EvalError> {
                indexed_select_u32(inputs, $name)
            }
        }
    };
}

fn indexed_select_u32(
    inputs: &[Memory],
    op_name: &'static str,
) -> Result<Memory, evaluator::EvalError> {
    let (values, indices) = evaluator::two_inputs(inputs, op_name)?;
    let values = evaluator::u32_words(values, op_name)?;
    let indices = evaluator::u32_words(indices, op_name)?;
    let mut output = Vec::with_capacity(indices.len());
    for index in indices {
        output.push(values[evaluator::checked_index(index, values.len(), op_name)?]);
    }
    Ok(evaluator::write_u32s(output))
}

impl_indexed_select_reference!(Gather, "gather");
impl_indexed_select_reference!(Shuffle, "shuffle");
