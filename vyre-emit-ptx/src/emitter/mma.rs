use std::fmt::Write as _;

use vyre_lower::{
    FragmentOperand, KernelOp, MatrixMmaElement, MatrixMmaLayout, MatrixMmaSpec, MatrixTileShape,
};

use super::BodyCtx;
use crate::reg::{PtxType, Reg};
use crate::EmitError;

/// The one tile this target has a native multiply-accumulate for.
const NATIVE_TILE: MatrixTileShape = MatrixTileShape { m: 16, n: 8, k: 16 };

/// Invocations the native form distributes a fragment across.
const NATIVE_LANES: u16 = 32;

impl BodyCtx<'_> {
    pub(super) fn emit_matrix_mma(
        &mut self,
        op: &KernelOp,
        spec: &MatrixMmaSpec,
    ) -> Result<Vec<Reg>, EmitError> {
        let words = spec.operand_words().map_err(|reason| {
            EmitError::InvalidDescriptor(format!(
                "MatrixMma declares fragments that cannot be carried: {reason}. Fix: state a tile that distributes across its lanes in whole 32-bit words."
            ))
        })?;
        let required = words.iter().sum::<u32>() as usize;
        if op.operands.len() != required {
            return Err(EmitError::InvalidDescriptor(format!(
                "MatrixMma declares {required} operand words but the op provides {}. Fix: pass the left, right and accumulator fragment words the specification derives.",
                op.operands.len()
            )));
        }
        if !self.supports_native_form(spec) {
            return Err(EmitError::UnsupportedOp(KernelOp {
                kind: op.kind.clone(),
                operands: op.operands.clone(),
                result: op.result,
            }));
        }

        let mut cursor = 0usize;
        let mut fragments: [Vec<Reg>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (slot, word_count) in words.into_iter().enumerate() {
            for _ in 0..word_count {
                fragments[slot].push(self.lookup_operand(op.operands[cursor])?);
                cursor += 1;
            }
        }
        let [a, b, c] = fragments;
        for reg in a.iter().chain(b.iter()) {
            if reg.0 != PtxType::U32 {
                return Err(EmitError::InvalidDescriptor(format!(
                    "MatrixMma f16 fragments must be packed u32 registers; got {reg}. Fix: pack two f16 lanes per u32 fragment operand."
                )));
            }
        }
        for reg in &c {
            if reg.0 != PtxType::F32 {
                return Err(EmitError::InvalidDescriptor(format!(
                    "MatrixMma f32 accumulators must be f32 registers; got {reg}. Fix: pass f32 accumulator operands."
                )));
            }
        }

        let d: Vec<Reg> = (0..c.len()).map(|_| self.alloc(PtxType::F32)).collect();
        let _ = writeln!(
            self.text,
            "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32    {{{}}}, {{{}}}, {{{}}}, {{{}}};",
            join(&d),
            join(&a),
            join(&b),
            join(&c),
        );
        Ok(d)
    }

    /// Whether the declared facts equal the one native form this target emits.
    ///
    /// Instruction selection belongs to the target: nothing outside this module
    /// states which tile, orientation or element types the substrate has an
    /// instruction for, and a declaration outside that set is rejected rather
    /// than reinterpreted.
    fn supports_native_form(&self, spec: &MatrixMmaSpec) -> bool {
        let lanes_match = [
            FragmentOperand::Left,
            FragmentOperand::Right,
            FragmentOperand::Accumulator,
        ]
        .into_iter()
        .all(|operand| {
            let fragment = spec.fragment(operand);
            fragment.lanes == NATIVE_LANES && fragment.is_register_resident()
        });

        lanes_match
            && spec.tile == NATIVE_TILE
            && spec.left.layout == MatrixMmaLayout::RowMajor
            && spec.right.layout == MatrixMmaLayout::ColMajor
            && spec.left.element == MatrixMmaElement::F16
            && spec.right.element == MatrixMmaElement::F16
            && spec.accumulator.element == MatrixMmaElement::F32
            && self.options.target.supports_wmma_f16()
    }
}

fn join(regs: &[Reg]) -> String {
    let mut text = String::new();
    for (index, reg) in regs.iter().enumerate() {
        if index > 0 {
            text.push_str(", ");
        }
        let _ = write!(text, "{reg}");
    }
    text
}
