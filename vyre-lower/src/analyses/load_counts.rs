//! Shared recursive load-site counting for memory-placement analyses.
//!
//! Texture promotion, AoS-to-SoA layout rewrites, and later cache-placement
//! analyses all need the same conservative traversal over structured kernel
//! bodies. Keeping the traversal here prevents each analysis from guessing
//! which operands are child-body IDs.

use rustc_hash::FxHashMap;

use super::child_body_operands;
use crate::{KernelBody, KernelOpKind};

pub(crate) fn count_global_loads_by_slot<F>(
    body: &KernelBody,
    is_eligible_slot: &F,
    counts: &mut FxHashMap<u32, u32>,
) where
    F: Fn(u32) -> bool,
{
    for op in &body.ops {
        if matches!(op.kind, KernelOpKind::LoadGlobal) {
            if let Some(slot) = op.operands.first() {
                if is_eligible_slot(*slot) {
                    *counts.entry(*slot).or_insert(0) += 1;
                }
            }
        }
        for child_id in child_body_operands(&op.kind, &op.operands) {
            if let Some(child) = body.child_bodies.get(child_id as usize) {
                count_global_loads_by_slot(child, is_eligible_slot, counts);
            }
        }
    }
}

/// Descriptor fixtures for the analyses whose precondition is a load count.
///
/// WHY: texture promotion and AoS-to-SoA layout each wrote out the same two
/// helpers, doc comment included, so the op shape this traversal reads was
/// stated twice and could drift on one side only. The shape belongs with the
/// traversal that defines it.
#[cfg(test)]
pub(crate) mod fixtures {
    use crate::descriptor_builder::{body, descriptor, lit, load_global};
    use crate::{BindingSlot, KernelBody, KernelDescriptor, LiteralValue};

    /// One literal at pool index 0 followed by `count` loads of slot 0. This is
    /// the whole op shape the precondition reads, so every eligibility case
    /// differs only in its binding and its load count.
    pub(crate) fn literal_then_loads(count: u32) -> KernelBody {
        body()
            .op(lit(0, 0))
            .ops((1..=count).map(|result| load_global(0, 0, result)))
            .literal(LiteralValue::U32(0))
            .build()
    }

    /// A 32-thread kernel over one binding, loaded `load_count` times.
    pub(crate) fn kernel(binding: BindingSlot, load_count: u32) -> KernelDescriptor {
        descriptor("k")
            .slot(binding)
            .dispatch(32, 1, 1)
            .body(literal_then_loads(load_count))
            .build()
    }
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap;

    use super::*;
    use crate::{KernelOp, LiteralValue};

    fn body_with_load(slot: u32) -> KernelBody {
        KernelBody {
            ops: vec![KernelOp {
                kind: KernelOpKind::LoadGlobal,
                operands: vec![slot, 0],
                result: Some(slot),
            }],
            child_bodies: vec![],
            literals: vec![],
        }
    }

    #[test]
    fn counts_if_else_children_and_ignores_for_loop_bound_operands() {
        let body = KernelBody {
            ops: vec![
                KernelOp {
                    kind: KernelOpKind::StructuredForLoop {
                        loop_var: "i".into(),
                    },
                    operands: vec![0, 1, 2],
                    result: None,
                },
                KernelOp {
                    kind: KernelOpKind::StructuredIfThenElse,
                    operands: vec![99, 3, 4],
                    result: None,
                },
            ],
            child_bodies: vec![
                body_with_load(7),
                body_with_load(7),
                body_with_load(7),
                body_with_load(7),
                body_with_load(7),
            ],
            literals: vec![LiteralValue::U32(0)],
        };

        let mut counts = FxHashMap::default();
        count_global_loads_by_slot(&body, &|slot| slot == 7, &mut counts);

        assert_eq!(
            counts.get(&7).copied(),
            Some(3),
            "Fix: load counting must include real structured child bodies without treating loop bound operands as child indices."
        );
    }

    /// Both memory-placement analyses read their precondition through this
    /// traversal, so they must agree on the count for the same descriptor,
    /// nested bodies included.
    ///
    /// WHY: shared-memory promotion and constant-buffer promotion each carried
    /// their own copy of this walk, differing only in the eligibility filter.
    /// Two copies is two answers to "how many times is this binding read", and
    /// a rewrite acts on whichever it asked. This goes red the moment either
    /// one grows its own traversal again and drifts.
    ///
    /// It does not check the eligibility filters themselves; each analysis owns
    /// its own and tests it.
    #[test]
    fn both_promotion_analyses_read_the_same_nested_load_count() {
        use crate::descriptor_builder::{body, descriptor, effect, global_ro, lit, SlotCount};
        use crate::KernelOpKind;
        use vyre_foundation::ir::DataType;

        let desc = descriptor("k")
            .slot(global_ro(0, DataType::F32, "ro0").with_count(16))
            .dispatch(32, 1, 1)
            .body(
                body()
                    .op(lit(0, 0))
                    .op(effect(KernelOpKind::StructuredIfThen, [0, 0]))
                    .child(super::fixtures::literal_then_loads(3))
                    .literal(LiteralValue::Bool(true)),
            )
            .build();

        let shared = crate::analyses::analyze_shared_mem_promote(&desc);
        let constant = crate::analyses::analyze_const_buffer_promote(&desc);
        assert_eq!(
            (
                shared.candidates[0].access_count,
                constant.candidates[0].load_count
            ),
            (3, 3),
            "Fix: both placement analyses count loads through count_global_loads_by_slot, so a nested binding read three times must read as three on each side."
        );
    }
}
