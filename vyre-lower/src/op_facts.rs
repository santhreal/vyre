//! The one per-kind fact table for `KernelOpKind`.
//!
//! Three questions used to each carry their own list of the same variant
//! universe: which operand of an op names a child body, whether an op must be
//! kept when its results are unused, and, in the PTX backend, whether an op can
//! sit under an instruction predicate. Three lists of one enum is three answers
//! that drift apart, and the drift is silent: a list that omits a variant reads
//! as "this variant is ordinary" rather than as an omission.
//!
//! The match below is the only place the variants are enumerated, and it has no
//! wildcard arm. Adding a `KernelOpKind` fails to compile until someone states
//! both facts for it.

use crate::KernelOpKind;

/// What every consumer of an op kind needs to know about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpFacts {
    /// Index of the first operand that names a child body, or `None` when the
    /// op names no child body.
    ///
    /// Child indices run to the end of the operand list, so the start is the
    /// whole answer.
    pub child_body_start: Option<usize>,
    /// True when the op must be kept even if every result id it produces is
    /// unused, because its nested bodies, its memory effects or its backend
    /// contract carry observable behavior.
    pub retained_effect: bool,
}

/// The facts for one op kind.
#[must_use]
pub fn facts_for(kind: &KernelOpKind) -> OpFacts {
    let (child_body_start, retained_effect) = match kind {
        // Structured control flow: child indices follow the condition or the
        // loop bounds, and the nested body is the observable behavior.
        KernelOpKind::StructuredIfThen | KernelOpKind::StructuredIfThenElse => (Some(1), true),
        KernelOpKind::StructuredForLoop { .. } => (Some(2), true),
        KernelOpKind::StructuredBlock | KernelOpKind::Region { .. } => (Some(0), true),

        // No child body, but kept: a memory effect, a protocol step, a
        // control-flow exit, or a body this crate cannot see through.
        KernelOpKind::StoreGlobal
        | KernelOpKind::StoreShared
        | KernelOpKind::LoopCarrierInit { .. }
        | KernelOpKind::LoopCarrierEnd { .. }
        | KernelOpKind::Atomic { .. }
        | KernelOpKind::Barrier { .. }
        | KernelOpKind::AsyncLoad { .. }
        | KernelOpKind::AsyncStore { .. }
        | KernelOpKind::AsyncWait { .. }
        | KernelOpKind::Trap { .. }
        | KernelOpKind::Resume { .. }
        | KernelOpKind::IndirectDispatch { .. }
        | KernelOpKind::Return
        | KernelOpKind::Call { .. }
        | KernelOpKind::OpaqueExpr(_)
        | KernelOpKind::OpaqueNode(_) => (None, true),

        // No child body and removable when unused: a value the op computes or
        // reads, and nothing else.
        KernelOpKind::Literal
        | KernelOpKind::Copy
        | KernelOpKind::LocalInvocationId
        | KernelOpKind::GlobalInvocationId
        | KernelOpKind::WorkgroupId
        | KernelOpKind::SubgroupLocalId
        | KernelOpKind::SubgroupSize
        | KernelOpKind::LoopIndex { .. }
        | KernelOpKind::LoopCarrier { .. }
        | KernelOpKind::LoadGlobal
        | KernelOpKind::LoadShared
        | KernelOpKind::LoadConstant
        | KernelOpKind::BufferLength
        | KernelOpKind::BinOpKind(_)
        | KernelOpKind::UnOpKind(_)
        | KernelOpKind::Fma
        | KernelOpKind::MatrixMma { .. }
        | KernelOpKind::Select
        | KernelOpKind::Cast { .. }
        | KernelOpKind::SubgroupBallot
        | KernelOpKind::SubgroupShuffle
        | KernelOpKind::SubgroupBroadcast
        | KernelOpKind::SubgroupReduce { .. } => (None, false),
    };
    OpFacts {
        child_body_start,
        retained_effect,
    }
}

/// Return true when a result-producing op can be removed if all of its result
/// ids are unused.
///
/// Stricter than "writes memory": structural control ops, async protocol ops,
/// calls, regions, and opaque nodes are kept even when they expose no directly
/// used result id, because their nested bodies or backend contracts may carry
/// observable behavior. Descriptor-level side-effect reporting is a different
/// question and lives on `KernelDescriptor`.
#[must_use]
pub(crate) fn kernel_op_kind_is_dce_pure(kind: &KernelOpKind) -> bool {
    !facts_for(kind).retained_effect
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OpaqueNodeData;
    use vyre_foundation::{
        ir::{AtomicOp, BinOp},
        memory_model::MemoryOrdering,
    };

    #[test]
    fn arithmetic_and_literals_are_dead_eliminable_when_unused() {
        assert!(kernel_op_kind_is_dce_pure(&KernelOpKind::Literal));
        assert!(kernel_op_kind_is_dce_pure(&KernelOpKind::BinOpKind(
            BinOp::Add
        )));
    }

    #[test]
    fn side_effecting_and_structural_ops_are_not_dead_eliminable() {
        assert!(!kernel_op_kind_is_dce_pure(&KernelOpKind::StoreGlobal));
        assert!(!kernel_op_kind_is_dce_pure(&KernelOpKind::Atomic {
            op: AtomicOp::Add,
            ordering: MemoryOrdering::SeqCst,
        }));
        assert!(!kernel_op_kind_is_dce_pure(&KernelOpKind::AsyncLoad {
            tag: "copy".into(),
        }));
        assert!(!kernel_op_kind_is_dce_pure(&KernelOpKind::Barrier {
            ordering: MemoryOrdering::SeqCst,
        }));
        assert!(!kernel_op_kind_is_dce_pure(&KernelOpKind::Return));
        assert!(!kernel_op_kind_is_dce_pure(
            &KernelOpKind::StructuredForLoop {
                loop_var: "i".into(),
            }
        ));
        assert!(!kernel_op_kind_is_dce_pure(&KernelOpKind::OpaqueNode(
            Box::new(OpaqueNodeData {
                extension_kind: "backend-specific".into(),
                payload: Vec::new(),
            })
        )));
    }

    /// Every kind that names a child body is also retained.
    ///
    /// WHY: the two facts were separate lists, and a kind present in one and
    /// absent from the other is the shape both defects here took. A child body
    /// is observable behavior by definition, so an op that carries one and is
    /// reported removable would have its body deleted with it.
    #[test]
    fn a_kind_that_names_a_child_body_is_never_removable() {
        for kind in [
            KernelOpKind::StructuredIfThen,
            KernelOpKind::StructuredIfThenElse,
            KernelOpKind::StructuredForLoop {
                loop_var: "i".into(),
            },
            KernelOpKind::StructuredBlock,
            KernelOpKind::Region {
                generator: "g".into(),
            },
        ] {
            let facts = facts_for(&kind);
            assert!(
                facts.child_body_start.is_some() && facts.retained_effect,
                "Fix: {kind:?} carries a child body, so it must report both a child-body start and a retained effect."
            );
        }
    }
}
