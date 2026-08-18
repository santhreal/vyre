//! Pure lookup tables: vyre IR ops → naga BinaryOperator / UnaryOperator
//! / MathFunction handles, plus literal and barrier-flag conversion.
//! No state  -  the contents are direct enum mappings.

use crate::EmitError;
use naga::{BinaryOperator, Literal, ScalarKind, UnaryOperator};
use vyre_foundation::ir::MemoryOrdering;
use vyre_foundation::ir::{BinOp, DataType, UnOp};
use vyre_lower::{KernelBody, KernelOpKind, LiteralValue};

pub(super) fn naga_literal(literal: &LiteralValue) -> Result<Literal, EmitError> {
    match literal {
        LiteralValue::U32(value) => Ok(Literal::U32(*value)),
        LiteralValue::I32(value) => Ok(Literal::I32(*value)),
        LiteralValue::F32(value) if value.is_finite() => Ok(Literal::F32(*value)),
        LiteralValue::F32(value) => Err(EmitError::InvalidDescriptor(format!(
            "f32 literal {value:?} is not finite; Naga literals cannot represent NaN/Inf"
        ))),
        LiteralValue::Bool(value) => Ok(Literal::Bool(*value)),
    }
}

pub(super) fn binary_operator(op: BinOp) -> Result<BinaryOperator, EmitError> {
    Ok(match op {
        BinOp::Add | BinOp::WrappingAdd => BinaryOperator::Add,
        BinOp::Sub | BinOp::WrappingSub => BinaryOperator::Subtract,
        BinOp::Mul => BinaryOperator::Multiply,
        BinOp::Div => BinaryOperator::Divide,
        BinOp::Mod => BinaryOperator::Modulo,
        BinOp::Eq => BinaryOperator::Equal,
        BinOp::Ne => BinaryOperator::NotEqual,
        BinOp::Lt => BinaryOperator::Less,
        BinOp::Le => BinaryOperator::LessEqual,
        BinOp::Gt => BinaryOperator::Greater,
        BinOp::Ge => BinaryOperator::GreaterEqual,
        // BinOp::And / Or always emit bitwise. WGSL bitwise And/Or on
        // bool operands returns bool with the LogicalAnd/Or truth
        // table (no short-circuit, but vyre IR doesn't model
        // short-circuit anyway). This is the only mapping that's
        // accepted by naga across the bool+bool, u32+u32, and mixed
        // bool+u32 (post-widen) operand shapes the BinOpKind arm
        // produces. LogicalAnd/Or would reject u32 operands outright.
        BinOp::And => BinaryOperator::And,
        BinOp::Or => BinaryOperator::InclusiveOr,
        BinOp::BitAnd => BinaryOperator::And,
        BinOp::BitOr => BinaryOperator::InclusiveOr,
        BinOp::BitXor => BinaryOperator::ExclusiveOr,
        BinOp::Shl => BinaryOperator::ShiftLeft,
        BinOp::Shr => BinaryOperator::ShiftRight,
        other => {
            return Err(EmitError::NagaConstructionFailed(format!(
                "binary op `{other:?}` has no direct Naga operator"
            )))
        }
    })
}

pub(super) fn unary_operator(op: &UnOp) -> Result<UnaryOperator, EmitError> {
    Ok(match op {
        UnOp::Negate => UnaryOperator::Negate,
        UnOp::LogicalNot => UnaryOperator::LogicalNot,
        UnOp::BitNot => UnaryOperator::BitwiseNot,
        other => {
            return Err(EmitError::NagaConstructionFailed(format!(
                "unary op `{other:?}` has no direct Naga unary operator"
            )))
        }
    })
}

/// Map BinOps that compile to `Expression::Math` (WGSL builtin
/// functions) instead of the basic `BinaryOperator` enum. Returns
/// `None` for ops that already have a direct binary-operator form.
pub(super) fn binary_math_function(op: BinOp) -> Option<naga::MathFunction> {
    Some(match op {
        BinOp::Min => naga::MathFunction::Min,
        BinOp::Max => naga::MathFunction::Max,
        // Saturating arithmetic + AbsDiff are emitted via the same
        // builtin path; Naga lowers them to wgsl `min(max(...))` etc.
        BinOp::SaturatingAdd | BinOp::SaturatingSub | BinOp::SaturatingMul | BinOp::AbsDiff => {
            return None;
        }
        _ => return None,
    })
}

/// Map UnOps that compile to `Expression::Math` (WGSL builtin
/// functions) instead of the basic `UnaryOperator` enum.
pub(super) fn unary_math_function(op: &UnOp) -> Option<naga::MathFunction> {
    Some(match op {
        UnOp::Sqrt => naga::MathFunction::Sqrt,
        UnOp::InverseSqrt => naga::MathFunction::InverseSqrt,
        UnOp::Abs => naga::MathFunction::Abs,
        UnOp::Sin => naga::MathFunction::Sin,
        UnOp::Cos => naga::MathFunction::Cos,
        UnOp::Tan => naga::MathFunction::Tan,
        UnOp::Asin => naga::MathFunction::Asin,
        UnOp::Acos => naga::MathFunction::Acos,
        UnOp::Atan => naga::MathFunction::Atan,
        UnOp::Sinh => naga::MathFunction::Sinh,
        UnOp::Cosh => naga::MathFunction::Cosh,
        UnOp::Tanh => naga::MathFunction::Tanh,
        UnOp::Exp => naga::MathFunction::Exp,
        UnOp::Exp2 => naga::MathFunction::Exp2,
        UnOp::Log => naga::MathFunction::Log,
        UnOp::Log2 => naga::MathFunction::Log2,
        UnOp::Floor => naga::MathFunction::Floor,
        UnOp::Ceil => naga::MathFunction::Ceil,
        UnOp::Round => naga::MathFunction::Round,
        UnOp::Trunc => naga::MathFunction::Trunc,
        UnOp::Sign => naga::MathFunction::Sign,
        UnOp::Popcount => naga::MathFunction::CountOneBits,
        UnOp::Clz => naga::MathFunction::CountLeadingZeros,
        UnOp::Ctz => naga::MathFunction::CountTrailingZeros,
        UnOp::ReverseBits => naga::MathFunction::ReverseBits,
        _ => return None,
    })
}

/// The `(right_shift, and_mask)` lowering for the nibble/byte unpack UnOps.
/// WGSL/Naga has no pack/unpack intrinsic, so these compile to an explicit
/// `(value >> shift) & mask` on u32. The semantics mirror `ir_eval` exactly:
/// `Unpack4Low = v & 0x0F`, `Unpack4High = (v >> 4) & 0x0F`,
/// `Unpack8Low = v & 0xFF`, `Unpack8High = (v >> 24) & 0xFF`.
pub(super) fn unpack_shift_mask(op: &UnOp) -> Option<(u32, u32)> {
    Some(match op {
        UnOp::Unpack4Low => (0, 0x0F),
        UnOp::Unpack4High => (4, 0x0F),
        UnOp::Unpack8Low => (0, 0xFF),
        UnOp::Unpack8High => (24, 0xFF),
        _ => return None,
    })
}

pub(super) fn scalar_cast_target(target: &DataType) -> Result<(ScalarKind, u8), EmitError> {
    match target {
        DataType::Bool => Ok((ScalarKind::Bool, 1)),
        DataType::U8 | DataType::U16 | DataType::U32 => Ok((ScalarKind::Uint, 4)),
        DataType::I8 | DataType::I16 | DataType::I32 => Ok((ScalarKind::Sint, 4)),
        DataType::F32 => Ok((ScalarKind::Float, 4)),
        // `Bytes` is a packed-byte buffer-element marker, NOT a castable scalar.
        // The foundation cast table (`validate::cast::cast_is_valid`) rejects
        // every cast to/from `Bytes`, but the Program-compatibility `emit_module`
        // path does NOT run full validation, so a `Cast { target: Bytes }` can
        // reach here. Mapping it to a 32-bit uint would SILENTLY reinterpret a
        // byte target as a word (Law 10). Fail closed and name the fix.
        DataType::Bytes => Err(EmitError::NagaConstructionFailed(
            "cast target `Bytes` is not a scalar: `Bytes` is a packed-byte buffer \
             element and must be unpacked via a pack-to-u32 pre-pass before \
             emission, never reinterpreted as a u32 word"
                .to_owned(),
        )),
        other => Err(EmitError::NagaConstructionFailed(format!(
            "cast target `{other:?}` is not supported by the scalar Naga emitter"
        ))),
    }
}

/// Address spaces one barrier orders, read off the body it sits in.
///
/// The first field is storage, the second workgroup scratch. Descent covers
/// `child_bodies`, so a barrier whose fenced accesses live inside a sibling
/// loop or conditional is measured from those accesses and not from the empty
/// sibling list around it.
///
/// `LoadConstant` and `BufferLength` contribute nothing. A constant or uniform
/// binding is read-only for the whole dispatch and a buffer length reads
/// binding metadata rather than buffer contents, so no barrier can order a
/// write against either of them.
fn barrier_body_spaces(body: &KernelBody) -> (bool, bool) {
    let mut storage = false;
    let mut workgroup = false;
    for op in &body.ops {
        match op.kind {
            KernelOpKind::LoadGlobal
            | KernelOpKind::StoreGlobal
            | KernelOpKind::VectorLoadGlobal { .. }
            | KernelOpKind::VectorStoreGlobal { .. }
            | KernelOpKind::Atomic { .. } => storage = true,
            KernelOpKind::LoadShared | KernelOpKind::StoreShared => workgroup = true,
            _ => {}
        }
    }
    for child in &body.child_bodies {
        let (child_storage, child_workgroup) = barrier_body_spaces(child);
        storage |= child_storage;
        workgroup |= child_workgroup;
    }
    (storage, workgroup)
}

/// Barrier flags for one IR memory ordering in the body that carries it.
///
/// The four strong orderings used to collapse onto `STORAGE | WORK_GROUP`, so a
/// workgroup-scratch reduction round paid a storage fence it never needed and
/// `SeqCst` was indistinguishable from `AcqRel` in the emitted shader.
///
/// `Acquire`, `Release` and `AcqRel` name global-memory visibility, so they
/// lower to `STORAGE` alone. WGSL has no fence without convergence, so
/// `storageBarrier()` also converges the workgroup; that is stronger than the
/// ordering asks for and never weaker.
///
/// `SeqCst` is a full barrier within the issuing workgroup, so its flags are
/// the address spaces the barrier actually orders. A body that touches only
/// scratch emits `WORK_GROUP`, a body that touches only storage emits
/// `STORAGE`, and a body that touches both, or neither, keeps the full fence.
/// Narrowing only on a demonstrated single address space keeps every existing
/// storage fence intact.
///
/// `GridSync` is device wide. WGSL has no whole-grid barrier and no cooperative
/// launch, so it is a planner cut (`vyre_megakernel::grid_sync`) that splits the
/// program into sequential dispatches before emission, never an instruction.
pub(super) fn barrier_flags(
    ordering: MemoryOrdering,
    body: &KernelBody,
) -> Result<naga::Barrier, EmitError> {
    match ordering {
        MemoryOrdering::Acquire | MemoryOrdering::Release | MemoryOrdering::AcqRel => {
            Ok(naga::Barrier::STORAGE)
        }
        MemoryOrdering::SeqCst => Ok(match barrier_body_spaces(body) {
            (false, true) => naga::Barrier::WORK_GROUP,
            (true, false) => naga::Barrier::STORAGE,
            _ => naga::Barrier::STORAGE | naga::Barrier::WORK_GROUP,
        }),
        MemoryOrdering::Relaxed => Err(EmitError::InvalidDescriptor(
            "relaxed barrier has no synchronization semantics".to_owned(),
        )),
        MemoryOrdering::GridSync => Err(EmitError::NagaConstructionFailed(
            "Fix: grid synchronization requires dispatch splitting before Naga emission".to_owned(),
        )),
        _ => Err(EmitError::NagaConstructionFailed(
            "future memory ordering is not mapped by the Naga emitter".to_owned(),
        )),
    }
}

/// WHY: the narrowing this module performs was covered only by two backend
/// golden corpora, and one of them was left stale for a release: the WGSL golden
/// carried the narrowed `storageBarrier()` while the SPIR-V golden still pinned
/// the memory-semantics word for the wider fence, and nothing said the two
/// disagreed. Reading the flags here judges the decision itself, so a regression
/// that weakens or widens a fence is red at the mapping rather than at whichever
/// pinned artifact somebody remembered to regenerate.
///
/// What this does not catch: whether a backend turns these flags into the right
/// instruction. That is the emitted-artifact goldens' job, and they only mean
/// something while every one of them is regenerated by the change that moves it.
// Inline: covers `barrier_flags`, which no integration test can name.
#[cfg(test)]
mod tests {
    use super::{barrier_flags, MemoryOrdering};
    use vyre_lower::{KernelBody, KernelOp, KernelOpKind, LiteralValue};

    fn body(kinds: &[KernelOpKind], children: Vec<KernelBody>) -> KernelBody {
        KernelBody {
            ops: kinds
                .iter()
                .cloned()
                .map(|kind| KernelOp {
                    kind,
                    operands: vec![],
                    result: None,
                })
                .collect(),
            child_bodies: children,
            literals: vec![LiteralValue::U32(0)],
        }
    }

    #[test]
    fn a_seqcst_barrier_orders_only_the_address_spaces_its_body_touches() {
        let storage_only = body(&[KernelOpKind::StoreGlobal], vec![]);
        assert_eq!(
            barrier_flags(MemoryOrdering::SeqCst, &storage_only).expect("storage-only body"),
            naga::Barrier::STORAGE
        );
        let scratch_only = body(&[KernelOpKind::StoreShared], vec![]);
        assert_eq!(
            barrier_flags(MemoryOrdering::SeqCst, &scratch_only).expect("scratch-only body"),
            naga::Barrier::WORK_GROUP
        );
        let both = body(
            &[KernelOpKind::LoadShared, KernelOpKind::LoadGlobal],
            vec![],
        );
        assert_eq!(
            barrier_flags(MemoryOrdering::SeqCst, &both).expect("both spaces"),
            naga::Barrier::STORAGE | naga::Barrier::WORK_GROUP
        );
    }

    /// A body that names no memory access keeps the full fence: the barrier is
    /// there for accesses the scan cannot see, so narrowing it would be a
    /// guess that weakens synchronization.
    #[test]
    fn a_seqcst_barrier_over_no_measured_access_keeps_the_full_fence() {
        assert_eq!(
            barrier_flags(MemoryOrdering::SeqCst, &body(&[], vec![])).expect("empty body"),
            naga::Barrier::STORAGE | naga::Barrier::WORK_GROUP
        );
    }

    /// The scan descends, so a barrier is measured from the accesses inside a
    /// nested loop or conditional rather than from the empty sibling list
    /// around it. This is the shape the `adv_loop_barrier` corpus case has.
    #[test]
    fn a_barrier_is_measured_from_the_accesses_in_child_bodies() {
        let nested = body(&[], vec![body(&[KernelOpKind::StoreShared], vec![])]);
        assert_eq!(
            barrier_flags(MemoryOrdering::SeqCst, &nested).expect("nested scratch access"),
            naga::Barrier::WORK_GROUP
        );
    }

    /// Acquire, Release and AcqRel name global-memory visibility, so the body
    /// does not move them: a scratch-only body still gets the storage fence.
    #[test]
    fn the_release_orderings_do_not_read_the_body() {
        for ordering in [
            MemoryOrdering::Acquire,
            MemoryOrdering::Release,
            MemoryOrdering::AcqRel,
        ] {
            assert_eq!(
                barrier_flags(ordering, &body(&[KernelOpKind::StoreShared], vec![]))
                    .expect("a release ordering maps without reading the body"),
                naga::Barrier::STORAGE,
                "{ordering:?} must not depend on the body"
            );
        }
    }

    /// Relaxed has no synchronization to emit and GridSync is a planner cut, so
    /// neither may quietly become an instruction.
    #[test]
    fn relaxed_and_gridsync_are_refused_rather_than_emitted() {
        assert!(barrier_flags(MemoryOrdering::Relaxed, &body(&[], vec![])).is_err());
        assert!(barrier_flags(MemoryOrdering::GridSync, &body(&[], vec![])).is_err());
    }
}
