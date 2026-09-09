//! Effects-typed lower pipeline (P-1.0-V1.3).
//!
//! [`compute_program_effects`] walks a `Program` and computes the
//! [`ProgramEffects`] row  -  the union of every effect kind any
//! Region in the program produces. The lowering pipeline can route
//! handler discharges (P-1.0-V1.1, P-1.0-V1.2) against the row to
//! prove that a backend's emitted code respects the declared effect
//! discipline.
//!
//! [`ProgramEffects`] is the one list of effect kinds. An analysis that needs
//! a program's effects reads this row rather than restating the kinds.

use crate::ir_inner::model::expr::Expr;
use crate::ir_inner::model::node::Node;
use crate::ir_inner::model::program::Program;

/// Set of effect kinds a `Program` produces. Each backend lowering pass may
/// require, discharge, or forbid specific kinds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ProgramEffects(u32);

impl ProgramEffects {
    /// Empty row.
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }
    /// Buffer write  -  `Node::Store`, `Node::AsyncStore`.
    pub const BUFFER_WRITE: Self = Self(1 << 0);
    /// Atomic read-modify-write  -  `Expr::Atomic`.
    pub const ATOMIC: Self = Self(1 << 1);
    /// Host-visible I/O effect used by host-bridge extensions.
    pub const HOST_IO: Self = Self(1 << 2);
    /// Nested GPU dispatch  -  `Node::IndirectDispatch`.
    pub const GPU_DISPATCH: Self = Self(1 << 3);
    /// Physical or logical synchronization barrier.
    pub const BARRIER: Self = Self(1 << 4);
    /// Async load fetching from streaming storage  -
    /// `Node::AsyncLoad`.
    pub const ASYNC_LOAD: Self = Self(1 << 5);
    /// Trap or abort  -  `Node::Trap`.
    pub const TRAP: Self = Self(1 << 6);

    /// Whether this row contains every bit set in `other`.
    #[must_use]
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Whether this row has no effects.
    #[must_use]
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether every effect in `self` is also present in `other`.
    #[must_use]
    #[inline]
    pub const fn is_subset_of(self, other: Self) -> bool {
        (self.0 & !other.0) == 0
    }

    /// Effects present in `self` but absent from `previous`.
    #[must_use]
    #[inline]
    pub const fn introduced_since(self, previous: Self) -> Self {
        Self(self.0 & !previous.0)
    }

    /// Raw bitmask.
    #[must_use]
    #[inline]
    pub const fn bits(self) -> u32 {
        self.0
    }
}

impl core::ops::BitOr for ProgramEffects {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for ProgramEffects {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Compute the union of every effect kind reachable from
/// `program.entry()`.
#[must_use]
pub fn compute_program_effects(program: &Program) -> ProgramEffects {
    let mut effects = ProgramEffects::empty();
    crate::visit::for_each_node(program.entry(), |node| {
        effects |= node_effects(node);
    });
    crate::visit::for_each_expr(program.entry(), |expr| {
        if matches!(expr, Expr::Atomic { .. }) {
            effects |= ProgramEffects::ATOMIC;
        }
    });
    effects
}

/// The effects one node contributes, ignoring the nodes and expressions below
/// it.
///
/// Descent belongs to [`crate::visit`], so this states only the per-variant
/// decision. A copy of the walk stood here and restated every `Node` and every
/// `Expr` variant to reach its children.
fn node_effects(node: &Node) -> ProgramEffects {
    match node {
        Node::Store { .. } | Node::AsyncStore { .. } | Node::TileStore { .. } => {
            ProgramEffects::BUFFER_WRITE
        }
        Node::AllReduce { .. }
        | Node::AllGather { .. }
        | Node::ReduceScatter { .. }
        | Node::Broadcast { .. } => ProgramEffects::BUFFER_WRITE | ProgramEffects::BARRIER,
        Node::IndirectDispatch { .. } => ProgramEffects::GPU_DISPATCH,
        Node::AsyncLoad { .. } => ProgramEffects::ASYNC_LOAD,
        Node::Trap { .. } => ProgramEffects::TRAP,
        Node::Barrier { .. } | Node::LogicalBarrier { .. } => ProgramEffects::BARRIER,
        Node::Let { .. }
        | Node::Assign { .. }
        | Node::If { .. }
        | Node::Loop { .. }
        | Node::TileLoad { .. }
        | Node::TileMatmul { .. }
        | Node::TileReduce { .. }
        | Node::TileDecl { .. }
        | Node::TileElementwise { .. }
        | Node::AsyncWait { .. }
        | Node::Resume { .. }
        | Node::Return
        | Node::Opaque(_)
        | Node::Block(_)
        | Node::Region { .. } => ProgramEffects::empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufferAccess, BufferDecl, DataType, Expr as IrExpr, Node as IrNode, Program};

    fn program_with(body: Vec<IrNode>, buffers: Vec<BufferDecl>) -> Program {
        Program::wrapped(buffers, [1, 1, 1], body)
    }

    #[test]
    fn empty_program_has_no_effects() {
        let prog = program_with(vec![IrNode::Return], vec![]);
        assert_eq!(compute_program_effects(&prog), ProgramEffects::empty());
    }

    #[test]
    fn store_records_buffer_write() {
        let prog = program_with(
            vec![IrNode::store("out", IrExpr::u32(0), IrExpr::u32(7))],
            vec![
                BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1),
            ],
        );
        let e = compute_program_effects(&prog);
        assert!(e.contains(ProgramEffects::BUFFER_WRITE));
        assert!(!e.contains(ProgramEffects::ATOMIC));
        assert!(!e.contains(ProgramEffects::BARRIER));
    }

    #[test]
    fn barrier_records_barrier() {
        let prog = program_with(vec![IrNode::barrier(), IrNode::Return], vec![]);
        let e = compute_program_effects(&prog);
        assert!(e.contains(ProgramEffects::BARRIER));
    }

    #[test]
    fn atomic_records_atomic() {
        let prog = program_with(
            vec![IrNode::store(
                "out",
                IrExpr::u32(0),
                IrExpr::atomic_add("out", IrExpr::u32(0), IrExpr::u32(1)),
            )],
            vec![
                BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1),
            ],
        );
        let e = compute_program_effects(&prog);
        assert!(e.contains(ProgramEffects::ATOMIC));
        assert!(e.contains(ProgramEffects::BUFFER_WRITE));
    }

    #[test]
    fn nested_in_if_branches_collects_effects() {
        let prog = program_with(
            vec![IrNode::If {
                cond: IrExpr::bool(true),
                then: vec![IrNode::barrier()],
                otherwise: vec![IrNode::store("o", IrExpr::u32(0), IrExpr::u32(1))],
            }],
            vec![BufferDecl::storage("o", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        );
        let e = compute_program_effects(&prog);
        assert!(e.contains(ProgramEffects::BARRIER));
        assert!(e.contains(ProgramEffects::BUFFER_WRITE));
    }

    #[test]
    fn pure_arithmetic_program_has_no_effects() {
        // Var + Lit binop with no Store still has zero effects.
        let prog = program_with(
            vec![IrNode::let_bind("x", IrExpr::u32(7)), IrNode::Return],
            vec![],
        );
        let e = compute_program_effects(&prog);
        assert_eq!(e, ProgramEffects::empty());
    }

    #[test]
    fn region_traversal_descends_into_body() {
        let prog = program_with(
            vec![IrNode::Region {
                generator: "test.r".into(),
                source_region: None,
                body: std::sync::Arc::new(vec![IrNode::barrier()]),
            }],
            vec![],
        );
        assert!(compute_program_effects(&prog).contains(ProgramEffects::BARRIER));
    }

    /// Every collective writes its destination and orders the participants, so
    /// a scheduler reading only `BUFFER_WRITE` would reorder a program across
    /// the synchronization the collective performs. One variant carried the
    /// write flag alone and no test observed the missing barrier.
    #[test]
    fn every_collective_records_a_buffer_write_and_a_barrier() {
        use crate::ir::{CollectiveOp, CommGroup};

        let collectives = [
            IrNode::AllReduce {
                buffer: "o".into(),
                op: CollectiveOp::Sum,
                group: CommGroup::WORLD,
            },
            IrNode::AllGather {
                input: "i".into(),
                output: "o".into(),
                group: CommGroup::WORLD,
            },
            IrNode::ReduceScatter {
                input: "i".into(),
                output: "o".into(),
                op: CollectiveOp::Sum,
                group: CommGroup::WORLD,
            },
            IrNode::Broadcast {
                buffer: "o".into(),
                root: 0,
                group: CommGroup::WORLD,
            },
        ];
        for node in collectives {
            let effects = node_effects(&node);
            assert!(
                effects.contains(ProgramEffects::BUFFER_WRITE),
                "Fix: {node:?} must record BUFFER_WRITE"
            );
            assert!(
                effects.contains(ProgramEffects::BARRIER),
                "Fix: {node:?} must record BARRIER"
            );
        }
    }

    /// A node contributes its own effects and nothing its children carry, so
    /// the traversal above is the only thing that aggregates.
    #[test]
    fn node_effects_ignores_the_bodies_below_a_node() {
        let nested = IrNode::Block(vec![IrNode::barrier()]);
        assert_eq!(node_effects(&nested), ProgramEffects::empty());
        assert!(compute_program_effects(&program_with(vec![nested], vec![]))
            .contains(ProgramEffects::BARRIER));
    }

    #[test]
    fn effects_form_a_stable_set() {
        // Order of nodes does not change the row.
        let buffer =
            BufferDecl::storage("o", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1);
        let p1 = program_with(
            vec![
                IrNode::barrier(),
                IrNode::store("o", IrExpr::u32(0), IrExpr::u32(1)),
            ],
            vec![buffer.clone()],
        );
        let p2 = program_with(
            vec![
                IrNode::store("o", IrExpr::u32(0), IrExpr::u32(1)),
                IrNode::barrier(),
            ],
            vec![buffer],
        );
        assert_eq!(compute_program_effects(&p1), compute_program_effects(&p2));
    }

    #[test]
    fn introduced_since_reports_only_new_effects() {
        let before = ProgramEffects::BUFFER_WRITE | ProgramEffects::ATOMIC;
        let after = before | ProgramEffects::BARRIER;
        let introduced = after.introduced_since(before);

        assert!(introduced.contains(ProgramEffects::BARRIER));
        assert!(!introduced.contains(ProgramEffects::BUFFER_WRITE));
        assert!(introduced.is_subset_of(ProgramEffects::BARRIER));
        assert!(ProgramEffects::empty().is_empty());
    }
}
