//! The one per-kind fact table for `KernelOpKind`.
//!
//! Three questions used to each carry their own list of the same variant
//! universe: which operand of an op names a child body, whether an op must be
//! kept when its results are unused, and, where a backend has them, whether an op can
//! sit under an instruction predicate. Three lists of one enum is three answers
//! that drift apart, and the drift is silent: a list that omits a variant reads
//! as "this variant is ordinary" rather than as an omission.
//!
//! The match below is the only place the variants are enumerated, and it has no
//! wildcard arm. Adding a `KernelOpKind` fails to compile until someone states
//! every fact for it.

use crate::lower::KernelOpKind;

/// How often, and under what condition, an op runs the bodies it names.
///
/// A dispatch-level transform reads this to decide whether a nested body can be
/// promoted to its container's level. A body that runs exactly once, with no
/// operand selecting it, carries the same execution conditions as the sequence
/// it sits in; a branch arm and a loop body do not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NestedBodyControl {
    /// Every named body runs once, in operand order, whenever the op runs.
    Unconditional,
    /// A condition operand selects which named body runs.
    Conditional,
    /// A named body runs once per iteration.
    Repeated,
}

/// The child bodies an op names, and the condition it runs them under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NestedBodies {
    /// Index of the first operand that names a child body.
    ///
    /// Child indices run to the end of the operand list, so the start is the
    /// whole answer.
    pub start: usize,
    /// Execution condition the op imposes on the bodies it names.
    pub control: NestedBodyControl,
}

/// What every consumer of an op kind needs to know about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpFacts {
    /// The child bodies the op names, or `None` when it names none.
    ///
    /// One field rather than an index beside a separate control enum: an op
    /// that names a body always runs it under some condition, and a pair of
    /// independent fields admits the two states that contradict each other.
    pub nested_bodies: Option<NestedBodies>,
    /// True when the op must be kept even if every result id it produces is
    /// unused, because its nested bodies, its memory effects or its backend
    /// contract carry observable behavior.
    pub retained_effect: bool,
}

/// The facts for one op kind.
#[must_use]
pub fn facts_for(kind: &KernelOpKind) -> OpFacts {
    let (nested_bodies, retained_effect) = match kind {
        // Structured control flow: child indices follow the condition or the
        // loop bounds, and the nested body is the observable behavior.
        KernelOpKind::StructuredIfThen | KernelOpKind::StructuredIfThenElse => (
            Some(NestedBodies {
                start: 1,
                control: NestedBodyControl::Conditional,
            }),
            true,
        ),
        KernelOpKind::StructuredForLoop { .. } => (
            Some(NestedBodies {
                start: 2,
                control: NestedBodyControl::Repeated,
            }),
            true,
        ),
        KernelOpKind::StructuredBlock | KernelOpKind::Region { .. } => (
            Some(NestedBodies {
                start: 0,
                control: NestedBodyControl::Unconditional,
            }),
            true,
        ),

        // No child body, but kept: a memory effect, a protocol step, a
        // control-flow exit, or a body this crate cannot see through.
        KernelOpKind::StoreGlobal
        | KernelOpKind::StoreShared
        | KernelOpKind::VectorStoreGlobal { .. }
        | KernelOpKind::LoopCarrierInit { .. }
        | KernelOpKind::LoopCarrierEnd { .. }
        | KernelOpKind::Atomic { .. }
        | KernelOpKind::Barrier { .. }
        | KernelOpKind::AsyncLoad(_)
        | KernelOpKind::AsyncStore(_)
        | KernelOpKind::AsyncWait(_)
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
        | KernelOpKind::VectorLoadGlobal { .. }
        | KernelOpKind::ExtractLane { .. }
        | KernelOpKind::BufferLength
        | KernelOpKind::BinOpKind(_)
        | KernelOpKind::UnOpKind(_)
        | KernelOpKind::Fma
        | KernelOpKind::MatrixMma(_)
        | KernelOpKind::Select
        | KernelOpKind::Cast { .. }
        | KernelOpKind::SubgroupBallot
        | KernelOpKind::SubgroupShuffle
        | KernelOpKind::SubgroupBroadcast
        | KernelOpKind::SubgroupReduce { .. } => (None, false),
    };
    OpFacts {
        nested_bodies,
        retained_effect,
    }
}

/// True when the op kind represents an ambient thread ID, subgroup property, or constant literal.
#[must_use]
pub(crate) fn is_ambient_or_literal(kind: &KernelOpKind) -> bool {
    matches!(
        kind,
        KernelOpKind::Literal
            | KernelOpKind::LocalInvocationId
            | KernelOpKind::GlobalInvocationId
            | KernelOpKind::WorkgroupId
            | KernelOpKind::SubgroupLocalId
            | KernelOpKind::SubgroupSize
    )
}

// Inline: covers `facts_for` across the retained/removable split.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::OpaqueNodeData;
    use vyre_foundation::ir::MemoryOrdering;
    use vyre_foundation::ir::{AtomicOp, BinOp};

    fn retained(kind: &KernelOpKind) -> bool {
        facts_for(kind).retained_effect
    }

    #[test]
    fn arithmetic_and_literals_are_dead_eliminable_when_unused() {
        assert!(!retained(&KernelOpKind::Literal));
        assert!(!retained(&KernelOpKind::BinOpKind(BinOp::Add)));
    }

    #[test]
    fn side_effecting_and_structural_ops_are_not_dead_eliminable() {
        assert!(retained(&KernelOpKind::StoreGlobal));
        assert!(retained(&KernelOpKind::Atomic {
            op: AtomicOp::Add,
            ordering: MemoryOrdering::SeqCst,
        }));
        assert!(retained(&KernelOpKind::async_load("copy".into())));
        assert!(retained(&KernelOpKind::Barrier {
            ordering: MemoryOrdering::SeqCst,
        }));
        assert!(retained(&KernelOpKind::Return));
        assert!(retained(&KernelOpKind::StructuredForLoop {
            loop_var: "i".into(),
        }));
        assert!(retained(&KernelOpKind::OpaqueNode(Box::new(
            OpaqueNodeData {
                extension_kind: "backend-specific".into(),
                payload: Vec::new(),
            }
        ))));
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
                facts.nested_bodies.is_some() && facts.retained_effect,
                "Fix: {kind:?} carries a child body, so it must report both its nested bodies and a retained effect."
            );
        }
    }
}
