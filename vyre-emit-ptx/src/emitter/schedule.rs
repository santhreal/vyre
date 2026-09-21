use vyre_lower::{KernelOp, KernelOpKind};

pub(super) fn is_latency_load(op: &KernelOp) -> bool {
    matches!(
        op.kind,
        KernelOpKind::LoadGlobal
            | KernelOpKind::LoadShared
            | KernelOpKind::LoadConstant
            | KernelOpKind::VectorLoadGlobal { .. }
    ) && op.result.is_some()
}

pub(crate) fn is_scheduling_fence(op: &KernelOp) -> bool {
    matches!(
        op.kind,
        KernelOpKind::StoreGlobal
            | KernelOpKind::StoreShared
            | KernelOpKind::VectorStoreGlobal { .. }
            | KernelOpKind::Atomic { .. }
            | KernelOpKind::Barrier { .. }
            | KernelOpKind::Return
            | KernelOpKind::Region { .. }
            | KernelOpKind::StructuredBlock
            | KernelOpKind::StructuredIfThen
            | KernelOpKind::StructuredIfThenElse
            | KernelOpKind::StructuredForLoop { .. }
            | KernelOpKind::AsyncLoad(_)
            | KernelOpKind::AsyncStore(_)
            | KernelOpKind::AsyncWait(_)
            | KernelOpKind::Trap { .. }
    )
}

pub(crate) fn is_schedulable_pure_op(op: &KernelOp) -> bool {
    matches!(
        op.kind,
        KernelOpKind::Literal
            | KernelOpKind::LocalInvocationId
            | KernelOpKind::GlobalInvocationId
            | KernelOpKind::WorkgroupId
            | KernelOpKind::BinOpKind(_)
            | KernelOpKind::UnOpKind(_)
            | KernelOpKind::Fma
            | KernelOpKind::MatrixMma(_)
            | KernelOpKind::Cast { .. }
            | KernelOpKind::Select
            | KernelOpKind::BufferLength
            | KernelOpKind::SubgroupLocalId
            | KernelOpKind::SubgroupSize
            | KernelOpKind::SubgroupBallot
            | KernelOpKind::SubgroupShuffle
            | KernelOpKind::SubgroupBroadcast
            | KernelOpKind::SubgroupReduce { .. }
            | KernelOpKind::ExtractLane { .. }
    ) && op.result.is_some()
}

pub(super) fn operand_is_immediate(op: &KernelOp, _operand: u32) -> bool {
    matches!(
        op.kind,
        KernelOpKind::Literal
            | KernelOpKind::LocalInvocationId
            | KernelOpKind::GlobalInvocationId
            | KernelOpKind::WorkgroupId
            | KernelOpKind::BufferLength
    )
}

pub(super) fn op_reads_operand(op: &KernelOp, operand: u32) -> bool {
    op.operands
        .iter()
        .any(|candidate| *candidate == operand && !operand_is_immediate(op, *candidate))
}

// Inline: covers the private `is_schedulable_pure_op`, `op_reads_operand` and `operand_is_immediate`, which no integration test can reach.
#[cfg(test)]
mod tests {
    use super::*;
    use vyre_lower::descriptor_builder::mma_f16_m16n8k16;

    fn op(kind: KernelOpKind, operands: Vec<u32>, result: Option<u32>) -> KernelOp {
        KernelOp {
            kind,
            operands,
            result,
        }
    }

    #[test]
    fn fma_and_mma_are_schedulable_compute_fillers() {
        assert!(is_schedulable_pure_op(&op(
            KernelOpKind::Fma,
            vec![1, 2, 3],
            Some(4)
        )));
        assert!(is_schedulable_pure_op(&op(
            mma_f16_m16n8k16(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            Some(11)
        )));
    }

    #[test]
    fn buffer_length_slot_operand_is_immediate_for_scheduling() {
        let length = op(KernelOpKind::BufferLength, vec![7], Some(12));
        assert!(is_schedulable_pure_op(&length));
        assert!(operand_is_immediate(&length, 7));
        assert!(!op_reads_operand(&length, 7));
    }

    #[test]
    fn unsupported_and_resultless_ops_are_not_latency_fillers() {
        assert!(!is_schedulable_pure_op(&op(
            KernelOpKind::Fma,
            vec![1, 2, 3],
            None
        )));
        assert!(!is_schedulable_pure_op(&op(
            KernelOpKind::Copy,
            vec![1],
            Some(2)
        )));
        assert!(!is_schedulable_pure_op(&op(
            KernelOpKind::StoreGlobal,
            vec![0, 1, 2],
            None
        )));
    }
}
