//! Shared per-word unary bitset kernel builder.

use crate::builder::elementwise::ElementwiseComposer;
use vyre_foundation::ir::{BufferAccess, DataType, Expr, Program, UnOp};

pub(crate) fn bitset_unary_word_program(
    op_id: &'static str,
    input: &str,
    output: &str,
    words: u32,
    op: UnOp,
) -> Program {
    ElementwiseComposer::new(op_id, words)
        .with_workgroup_size([256, 1, 1])
        .add_input_storage(input, BufferAccess::ReadOnly, DataType::U32, words)
        .add_output_storage(output, BufferAccess::ReadWrite, DataType::U32, words)
        .build_pointwise(output, |i| Expr::UnOp {
            op,
            operand: Box::new(Expr::load(input, i)),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_unary_word_program_lengths_are_declared_exactly() {
        let mut cases = 0usize;
        for words in 0..=2048 {
            for op in [UnOp::BitNot, UnOp::Popcount] {
                let program =
                    bitset_unary_word_program("vyre-libs::bitset::test", "in", "out", words, op);
                assert_eq!(program.buffers().len(), 2);
                let output = program
                    .buffers()
                    .iter()
                    .find(|buffer| buffer.name() == "out")
                    .expect("Fix: bitset unary program must declare output buffer.");
                assert_eq!(output.count(), words);
                cases += 1;
            }
        }
        assert_eq!(cases, 4_098);
    }
}
