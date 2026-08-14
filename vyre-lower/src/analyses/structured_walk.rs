//! ONE owner for the structured control-flow walk over a body tree.
//!
//! Every analysis that reasons about branches, loops, blocks, or regions has
//! to do the same four things before it can say anything interesting: iterate
//! a body's ops, resolve the child bodies a structured op names, assign each
//! site a flattened op index, and index that body's ops by the result id each
//! one publishes. Written per analysis that is four chances to derive the
//! child-body offsets differently, and the copies do drift.
//!
//! This module owns the traversal. It owns nothing about what a site means:
//! the judgment stays with the analysis, in the crate that has a reason to
//! make it.
//!
//! Child bodies are resolved through [`child_body_operands`], the one owner of
//! the per-kind operand layout, so a walk cannot invent its own idea of where
//! a body index lives.

use crate::analyses::{child_body_operands, producer_map, AccessKind, ProducerMap};
use crate::{KernelBody, KernelOp, KernelOpKind};

/// Whether a walk enters the arms of a structured branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmDescent {
    /// Walk into branch arms as ordinary bodies.
    Enter,
    /// Report the branch and treat its arms as opaque. An analysis that
    /// judges an arm as a single unit uses this so nested sites do not also
    /// surface as independent findings.
    Skip,
}

/// Receives the sites a [`walk_structured`] pass reaches.
///
/// Every hook defaults to doing nothing, so an analysis implements only the
/// granularity it needs: whole-body for window scans, per-op for site
/// classification.
pub trait StructuredVisitor<'a> {
    /// Called once on entering `body`, before any of its ops.
    ///
    /// `op_index_offset` is the flattened index of the body's first op.
    fn enter_body(&mut self, body: &'a KernelBody, op_index_offset: usize) {
        let _ = (body, op_index_offset);
    }

    /// Called for each op of `body` in order, before the walk descends into
    /// any child body that op names.
    ///
    /// `producers` indexes `body`'s own ops by the result id each publishes.
    /// The walk owns it because a child body is walked in the MIDDLE of its
    /// parent's op stream: a visitor holding one map of its own would still be
    /// holding a nested arm's when the parent's next op arrived. All three
    /// analyses on this walk had their own answer to that, two by hand-rolling
    /// the descent outright.
    fn visit_op(
        &mut self,
        body: &'a KernelBody,
        producers: &ProducerMap<'a>,
        op_index: usize,
        op: &'a KernelOp,
    ) {
        let _ = (body, producers, op_index, op);
    }
}

/// Walk `root` and every body structured control flow can reach from it.
///
/// A child body is visited immediately after the op that names it, so a
/// visitor observes parent and nested sites in source order.
pub fn walk_structured<'a, V>(root: &'a KernelBody, arms: ArmDescent, visitor: &mut V)
where
    V: StructuredVisitor<'a>,
{
    walk_body(root, arms, 0, visitor);
}

fn walk_body<'a, V>(body: &'a KernelBody, arms: ArmDescent, op_index_offset: usize, visitor: &mut V)
where
    V: StructuredVisitor<'a>,
{
    visitor.enter_body(body, op_index_offset);
    // Held on this frame, so descending into a child cannot displace it and
    // returning from one cannot leave the parent reading the child's map.
    let producers = producer_map(body);
    // Every child of this body shares one offset: the flattened index just
    // past the parent's own ops.
    let child_offset = op_index_offset + body.ops.len();
    for (local_index, op) in body.ops.iter().enumerate() {
        visitor.visit_op(body, &producers, op_index_offset + local_index, op);
        if skips_arms(arms, &op.kind) {
            continue;
        }
        for child_index in child_body_operands(&op.kind, &op.operands) {
            if let Some(child) = body.child_bodies.get(child_index as usize) {
                walk_body(child, arms, child_offset, visitor);
            }
        }
    }
}

fn skips_arms(arms: ArmDescent, kind: &KernelOpKind) -> bool {
    arms == ArmDescent::Skip
        && matches!(
            kind,
            KernelOpKind::StructuredIfThen | KernelOpKind::StructuredIfThenElse
        )
}

/// One memory access the walk reached, resolved against its own body.
///
/// `'a` is the descriptor's lifetime; `'p` is the walk frame that holds the
/// body's producer map.
pub(crate) struct AccessRef<'p, 'a> {
    /// Body that holds the access op.
    pub(crate) body: &'a KernelBody,
    /// Producer map of that body.
    pub(crate) producers: &'p ProducerMap<'a>,
    /// Flattened index of the access op.
    pub(crate) op_index: usize,
    /// Direction of the access.
    pub(crate) kind: AccessKind,
    /// Binding slot the access targets, operand 0.
    pub(crate) binding_slot: u32,
    /// Result id of the index expression, operand 1.
    pub(crate) index_operand_id: u32,
}

/// Slot and index operand positions, identical on every buffer access kind.
const SLOT_POS: usize = 0;
const INDEX_POS: usize = 1;

/// Call `judge` for every op of `root` whose kind is `load` or `store`.
///
/// WHY: bank-conflict and coalescing classification differ in which pair of op
/// kinds they read (shared or global) and in how they classify an index, not in
/// how they find the accesses. Each selected its pair inside its own visitor,
/// which left the whole visitor plus the malformed-operand guard restated on
/// both sides. An op carrying fewer than two operands is malformed and never
/// reaches `judge`, so a caller reads the slot and the index unconditionally.
pub(crate) fn walk_accesses<'a, F>(
    root: &'a KernelBody,
    load: &KernelOpKind,
    store: &KernelOpKind,
    judge: F,
) where
    F: FnMut(AccessRef<'_, 'a>),
{
    let mut selector = AccessSelector { load, store, judge };
    walk_structured(root, ArmDescent::Enter, &mut selector);
}

struct AccessSelector<'k, F> {
    load: &'k KernelOpKind,
    store: &'k KernelOpKind,
    judge: F,
}

impl<'a, F> StructuredVisitor<'a> for AccessSelector<'_, F>
where
    F: FnMut(AccessRef<'_, 'a>),
{
    fn visit_op(
        &mut self,
        body: &'a KernelBody,
        producers: &ProducerMap<'a>,
        op_index: usize,
        op: &'a KernelOp,
    ) {
        let kind = if &op.kind == self.load {
            AccessKind::Load
        } else if &op.kind == self.store {
            AccessKind::Store
        } else {
            return;
        };
        // Bounds check the operand list so a malformed descriptor does not
        // panic the analysis.
        if op.operands.len() <= INDEX_POS {
            return;
        }
        (self.judge)(AccessRef {
            body,
            producers,
            op_index,
            kind,
            binding_slot: op.operands[SLOT_POS],
            index_operand_id: op.operands[INDEX_POS],
        });
    }
}

/// Shape of a structured branch op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchForm {
    /// `if (cond) { then }`.
    IfThen,
    /// `if (cond) { then } else { otherwise }`.
    IfThenElse,
}

/// A decoded structured branch: its condition and its resolved arms.
#[derive(Debug, Clone, Copy)]
pub struct StructuredBranch<'a> {
    /// Whether the op carries an else arm.
    pub form: BranchForm,
    /// Result id of the condition operand.
    pub cond_operand_id: u32,
    /// True arm, `None` when the operand names no existing child body.
    pub then_body: Option<&'a KernelBody>,
    /// False arm, always `None` for [`BranchForm::IfThen`].
    pub else_body: Option<&'a KernelBody>,
}

/// Decode `op` as a structured branch of `body`, or `None` when it is not one.
///
/// Arms resolve against `body.child_bodies`; an operand naming no existing
/// child yields `None` for that arm rather than dropping the whole site, so a
/// caller can still see the condition of a malformed branch.
#[must_use]
pub fn branch_at<'a>(body: &'a KernelBody, op: &KernelOp) -> Option<StructuredBranch<'a>> {
    let form = match op.kind {
        KernelOpKind::StructuredIfThen => BranchForm::IfThen,
        KernelOpKind::StructuredIfThenElse => BranchForm::IfThenElse,
        _ => return None,
    };
    let arm = |pos: usize| {
        op.operands
            .get(pos)
            .and_then(|index| body.child_bodies.get(*index as usize))
    };
    Some(StructuredBranch {
        form,
        cond_operand_id: *op.operands.first()?,
        then_body: arm(1),
        else_body: match form {
            BranchForm::IfThen => None,
            BranchForm::IfThenElse => arm(2),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor_builder::{body, for_loop, if_then, if_then_else, lit};
    use crate::LiteralValue;

    #[derive(Default)]
    struct Trace {
        bodies: Vec<usize>,
        branches: Vec<usize>,
    }

    impl<'a> StructuredVisitor<'a> for Trace {
        fn enter_body(&mut self, _body: &'a KernelBody, op_index_offset: usize) {
            self.bodies.push(op_index_offset);
        }

        fn visit_op(
            &mut self,
            body: &'a KernelBody,
            _producers: &ProducerMap<'a>,
            op_index: usize,
            op: &'a KernelOp,
        ) {
            if branch_at(body, op).is_some() {
                self.branches.push(op_index);
            }
        }
    }

    /// `if A { if B {} }; if C {}` at the top level. A nested site must be
    /// reported between its parent and the parent's next sibling.
    fn nested_then_sibling() -> KernelBody {
        let inner = body().literal(LiteralValue::Bool(true)).op(lit(0, 20));
        let arm = body()
            .literal(LiteralValue::Bool(true))
            .op(lit(0, 10))
            .op(if_then(10, 0))
            .child(inner);
        body()
            .literal(LiteralValue::Bool(true))
            .op(lit(0, 0))
            .op(if_then(0, 0))
            .op(if_then(0, 1))
            .child(arm)
            .child(body())
            .build()
    }

    /// A child body is walked in the middle of its parent's op stream, so the
    /// map an op is judged against has to be its own body's. This fixture is
    /// exactly that interleaving: outer op 2 follows the arm, and its map must
    /// be the outer body's again rather than the arm's or the arm's own nest.
    #[test]
    fn each_op_is_visited_with_its_own_body_producer_map() {
        #[derive(Default)]
        struct Seen {
            per_op: Vec<(usize, Vec<u32>)>,
        }
        impl<'a> StructuredVisitor<'a> for Seen {
            fn visit_op(
                &mut self,
                _body: &'a KernelBody,
                producers: &ProducerMap<'a>,
                op_index: usize,
                _op: &'a KernelOp,
            ) {
                let mut ids: Vec<u32> = producers.keys().copied().collect();
                ids.sort_unstable();
                self.per_op.push((op_index, ids));
            }
        }

        let mut seen = Seen::default();
        walk_structured(&nested_then_sibling(), ArmDescent::Enter, &mut seen);
        assert_eq!(
            seen.per_op,
            vec![
                (0, vec![0]),  // outer body
                (1, vec![0]),  // outer body's first branch
                (3, vec![10]), // that arm
                (4, vec![10]), // the arm's own branch
                (5, vec![20]), // the nested arm
                (2, vec![0]),  // outer body resumes on its own map
            ],
            "Fix: every op must be visited with the producer map of the body that holds it, or a site after a nested arm is classified against the arm's producers."
        );
    }

    #[test]
    fn nested_sites_are_reported_in_source_order() {
        let mut trace = Trace::default();
        walk_structured(&nested_then_sibling(), ArmDescent::Enter, &mut trace);
        // op 1 is the outer if, op 3+1 = 4 is the branch inside its arm, and
        // op 2 is the outer sibling that follows.
        assert_eq!(trace.branches, vec![1, 4, 2]);
    }

    #[test]
    fn skip_leaves_arms_unvisited() {
        let mut trace = Trace::default();
        walk_structured(&nested_then_sibling(), ArmDescent::Skip, &mut trace);
        assert_eq!(trace.branches, vec![1, 2]);
        assert_eq!(trace.bodies, vec![0], "no arm body was entered");
    }

    #[test]
    fn skip_still_enters_loop_and_block_bodies() {
        let loop_body = body()
            .literal(LiteralValue::Bool(true))
            .op(lit(0, 30))
            .op(if_then(30, 0))
            .child(body());
        let root = body()
            .literals([LiteralValue::U32(0), LiteralValue::U32(4)])
            .op(lit(0, 0))
            .op(lit(1, 1))
            .op(for_loop("i", 0, 1, 0))
            .child(loop_body)
            .build();
        let mut trace = Trace::default();
        walk_structured(&root, ArmDescent::Skip, &mut trace);
        assert_eq!(trace.branches, vec![4], "the in-loop branch is still seen");
    }

    #[test]
    fn both_arms_of_an_if_else_are_entered() {
        let root = body()
            .literal(LiteralValue::Bool(true))
            .op(lit(0, 0))
            .op(if_then_else(0, 0, 1))
            .child(body().literal(LiteralValue::U32(1)).op(lit(0, 10)))
            .child(body().literal(LiteralValue::U32(2)).op(lit(0, 20)))
            .build();
        let mut trace = Trace::default();
        walk_structured(&root, ArmDescent::Enter, &mut trace);
        assert_eq!(
            trace.bodies.len(),
            3,
            "root plus BOTH arms; a walk that takes only the last child body \
             misses the then arm"
        );
    }

    #[test]
    fn branch_at_decodes_arms_and_ignores_other_ops() {
        let root = body()
            .literal(LiteralValue::Bool(true))
            .op(lit(0, 0))
            .op(if_then_else(0, 0, 1))
            .child(body().literal(LiteralValue::U32(1)).op(lit(0, 10)))
            .child(body())
            .build();
        assert!(branch_at(&root, &root.ops[0]).is_none());
        let branch = branch_at(&root, &root.ops[1]).expect("if-else decodes");
        assert_eq!(branch.form, BranchForm::IfThenElse);
        assert_eq!(branch.cond_operand_id, 0);
        assert_eq!(branch.then_body.map(|b| b.ops.len()), Some(1));
        assert_eq!(branch.else_body.map(|b| b.ops.len()), Some(0));
    }

    #[test]
    fn an_if_then_never_reports_an_else_arm() {
        let root = body()
            .literal(LiteralValue::Bool(true))
            .op(lit(0, 0))
            .op(if_then(0, 0))
            .child(body())
            .child(body().literal(LiteralValue::U32(1)).op(lit(0, 10)))
            .build();
        let branch = branch_at(&root, &root.ops[1]).expect("if-then decodes");
        assert_eq!(branch.form, BranchForm::IfThen);
        assert!(branch.else_body.is_none());
    }
}
