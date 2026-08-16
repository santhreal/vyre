//! Lowered op stream behavior: hashing for the float-bearing op and literal
//! records, and the constructors emitters build them with.

use super::{KernelOp, KernelOpKind, LiteralValue};

impl Eq for LiteralValue {}

impl std::hash::Hash for LiteralValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::U32(v) => {
                0u8.hash(state);
                v.hash(state);
            }
            Self::I32(v) => {
                1u8.hash(state);
                v.hash(state);
            }
            // Hash f32 by its bit pattern so NaN-with-different-payloads
            // hash distinctly. Equality uses bit pattern too via PartialEq
            // on the `==` of f32  -  note this means two NaNs are not equal,
            // which is correct for caching purposes (they CAN be different
            // NaNs).
            Self::F32(v) => {
                2u8.hash(state);
                v.to_bits().hash(state);
            }
            Self::Bool(v) => {
                3u8.hash(state);
                v.hash(state);
            }
        }
    }
}

impl Eq for KernelOp {}

impl std::hash::Hash for KernelOp {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
        self.operands.hash(state);
        self.result.hash(state);
    }
}

impl KernelOp {
    /// Number of result ids this op defines.
    #[must_use]
    pub fn result_id_count(&self) -> u32 {
        match self.kind {
            KernelOpKind::MatrixMma { .. } => 4,
            _ => u32::from(self.result.is_some()),
        }
    }

    /// Every result id produced by this op.
    ///
    /// Most descriptor ops produce zero or one id. Matrix MMA produces a
    /// compact four-id accumulator tuple starting at `result`.
    pub fn result_ids(&self) -> impl Iterator<Item = u32> + '_ {
        let base = self.result;
        (0..self.result_id_count())
            .filter_map(move |offset| base.and_then(|id| id.checked_add(offset)))
    }
}

impl Eq for KernelOpKind {}

impl std::hash::Hash for KernelOpKind {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::BinOpKind(op) => op.hash(state),
            Self::UnOpKind(op) => op.hash(state),
            Self::MatrixMma {
                shape,
                a_layout,
                b_layout,
                a_type,
                b_type,
                accum_type,
            } => {
                shape.hash(state);
                a_layout.hash(state);
                b_layout.hash(state);
                a_type.hash(state);
                b_type.hash(state);
                accum_type.hash(state);
            }
            Self::Cast { target } => target.hash(state),
            Self::Atomic { op, ordering } => {
                op.hash(state);
                ordering.hash(state);
            }
            Self::StructuredForLoop { loop_var } => loop_var.hash(state),
            Self::LoopIndex { loop_var } => loop_var.hash(state),
            Self::Barrier { ordering } => ordering.hash(state),
            Self::Region { generator } => generator.hash(state),
            Self::AsyncLoad { tag }
            | Self::AsyncStore { tag }
            | Self::AsyncWait { tag }
            | Self::Trap { tag }
            | Self::Resume { tag } => tag.hash(state),
            Self::LoopCarrierInit { name }
            | Self::LoopCarrier { name }
            | Self::LoopCarrierEnd { name } => name.hash(state),
            Self::IndirectDispatch { count_offset } => count_offset.hash(state),
            Self::Call { op_id } => op_id.hash(state),
            Self::OpaqueExpr(data) => {
                data.extension_id.hash(state);
                data.extension_kind.hash(state);
                data.payload.hash(state);
            }
            Self::OpaqueNode(data) => {
                data.extension_kind.hash(state);
                data.payload.hash(state);
            }
            _ => {}
        }
    }
}

// Inline: covers items in the crate-private `descriptor` module, which no integration test can reach.
#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::MemoryOrdering;
    use vyre_foundation::ir::{AtomicOp, BinOp, DataType, UnOp};

    #[test]
    fn binop_kind_carries_full_vyre_spec_op() {
        let op = KernelOp {
            kind: KernelOpKind::BinOpKind(BinOp::SaturatingAdd),
            operands: vec![0, 1],
            result: Some(2),
        };
        let json = serde_json::to_string(&op).unwrap();
        let parsed: KernelOp = serde_json::from_str(&json).unwrap();
        assert_eq!(op, parsed);
        // Confirm the variant survives  -  serde_json round-trip preserves it.
        match parsed.kind {
            KernelOpKind::BinOpKind(BinOp::SaturatingAdd) => {}
            other => panic!("lost BinOp variant: {other:?}"),
        }
    }

    #[test]
    fn unop_kind_carries_full_vyre_spec_op() {
        let op = KernelOp {
            kind: KernelOpKind::UnOpKind(UnOp::InverseSqrt),
            operands: vec![5],
            result: Some(6),
        };
        let json = serde_json::to_string(&op).unwrap();
        let parsed: KernelOp = serde_json::from_str(&json).unwrap();
        assert_eq!(op, parsed);
        match parsed.kind {
            KernelOpKind::UnOpKind(UnOp::InverseSqrt) => {}
            other => panic!("lost UnOp variant: {other:?}"),
        }
    }

    #[test]
    fn atomic_carries_op_and_ordering() {
        let op = KernelOp {
            kind: KernelOpKind::Atomic {
                op: AtomicOp::CompareExchange,
                ordering: MemoryOrdering::AcqRel,
            },
            operands: vec![0, 1, 2, 3],
            result: Some(4),
        };
        let json = serde_json::to_string(&op).unwrap();
        let parsed: KernelOp = serde_json::from_str(&json).unwrap();
        assert_eq!(op, parsed);
    }

    #[test]
    fn cast_op_preserves_target_dtype() {
        let op = KernelOp {
            kind: KernelOpKind::Cast {
                target: DataType::F16,
            },
            operands: vec![3],
            result: Some(4),
        };
        let json = serde_json::to_string(&op).unwrap();
        let parsed: KernelOp = serde_json::from_str(&json).unwrap();
        match parsed.kind {
            KernelOpKind::Cast {
                target: DataType::F16,
            } => {}
            other => panic!("lost cast target: {other:?}"),
        }
    }

    #[test]
    fn literal_value_eq_treats_nan_as_distinct_via_bits() {
        let nan1 = LiteralValue::F32(f32::NAN);
        let nan2 = LiteralValue::F32(f32::NAN);
        // PartialEq for f32 treats NaN as not equal to itself; our derive
        // inherits that, so two NaNs are never equal.
        assert_ne!(nan1, nan2);
    }

    #[test]
    fn region_op_round_trips_with_generator_name() {
        let op = KernelOp {
            kind: KernelOpKind::Region {
                generator: "vyre.libs.nn.gqa_attention".into(),
            },
            operands: vec![0],
            result: None,
        };
        let json = serde_json::to_string(&op).unwrap();
        let parsed: KernelOp = serde_json::from_str(&json).unwrap();
        assert_eq!(op, parsed);
    }
}
