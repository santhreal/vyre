//! Analysis pass: walk a `KernelDescriptor`'s op stream, classify
//! each `LoadGlobal` and `StoreGlobal` access pattern, build a
//! `CoalescenceReport`.
//!
//! ## Algorithm
//!
//! For each `LoadGlobal` / `StoreGlobal` op, the index operand is a
//! `LiteralValue` reference. We trace it backward through the body's
//! ops to determine which of these forms it has:
//!
//! 1. `LocalInvocationId.x` / `GlobalInvocationId.x` → CoalescedUnitStride
//! 2. `Add(invocation_id.x, <const>)`                → CoalescedUnitStride
//! 3. `Mul(invocation_id.x, <const k>)`              → Strided { stride: k }
//! 4. `Add(Mul(invocation_id.x, k), c)`              → Strided { stride: k }
//! 5. literal constant                         → Broadcast
//! 6. anything else                            → Scattered
//!
//! Conservative cases that cannot be proven constant-stride classify
//! as `Scattered`, which is the rewrite-safe direction.

use super::report::{AccessPattern, AccessSite, CoalescenceReport};
use crate::analyses::structured_walk::walk_accesses;
use crate::analyses::{constant_u32_operand, ProducerMap};
use crate::{KernelBody, KernelDescriptor, KernelOpKind};
use vyre_foundation::ir::BinOp;

/// Run coalescence analysis on a kernel.
#[must_use]
pub fn analyze(desc: &KernelDescriptor) -> CoalescenceReport {
    let mut sites = Vec::new();
    walk_accesses(
        &desc.body,
        &KernelOpKind::LoadGlobal,
        &KernelOpKind::StoreGlobal,
        |access| {
            sites.push(AccessSite {
                op_index: access.op_index,
                kind: access.kind,
                binding_slot: access.binding_slot,
                pattern: classify_index(access.body, access.producers, access.index_operand_id),
            });
        },
    );
    CoalescenceReport {
        kernel_id: desc.id.clone(),
        sites,
    }
}

/// Classify an index expression by its access pattern across threads.
///
/// `index_operand_id` is the `result` of some op in `body.ops`. We
/// trace backward to find that op and determine its shape.
fn classify_index(
    body: &KernelBody,
    producers: &ProducerMap<'_>,
    index_operand_id: u32,
) -> AccessPattern {
    let producer = producers.get(&index_operand_id).copied();
    let Some(producer) = producer else {
        // Not a body-local result  -  could be a literal pool ref. Look
        // it up there.
        return classify_pool_operand(body, index_operand_id);
    };

    match &producer.kind {
        KernelOpKind::LocalInvocationId | KernelOpKind::GlobalInvocationId => {
            classify_invocation_id(producer)
        }
        KernelOpKind::Literal => AccessPattern::Broadcast,
        KernelOpKind::BinOpKind(BinOp::Add | BinOp::WrappingAdd) => {
            classify_add(body, producers, &producer.operands)
        }
        KernelOpKind::BinOpKind(BinOp::Mul) => classify_mul(body, producers, &producer.operands),
        _ => AccessPattern::Scattered,
    }
}

fn classify_invocation_id(op: &crate::KernelOp) -> AccessPattern {
    match op.operands.first().copied().unwrap_or(0) {
        0 => AccessPattern::CoalescedUnitStride,
        _ => AccessPattern::Scattered,
    }
}

fn classify_add(body: &KernelBody, producers: &ProducerMap<'_>, operands: &[u32]) -> AccessPattern {
    if operands.len() != 2 {
        return AccessPattern::Scattered;
    }
    let lhs = classify_index(body, producers, operands[0]);
    let rhs = classify_index(body, producers, operands[1]);
    // Broadcast (constant) + CoalescedUnitStride = still coalesced
    // unit-stride (just at base + const).
    match (lhs, rhs) {
        (AccessPattern::CoalescedUnitStride, AccessPattern::Broadcast)
        | (AccessPattern::Broadcast, AccessPattern::CoalescedUnitStride) => {
            AccessPattern::CoalescedUnitStride
        }
        // Strided + constant offset preserves stride.
        (AccessPattern::Strided { stride }, AccessPattern::Broadcast)
        | (AccessPattern::Broadcast, AccessPattern::Strided { stride }) => {
            AccessPattern::Strided { stride }
        }
        _ => AccessPattern::Scattered,
    }
}

fn classify_mul(body: &KernelBody, producers: &ProducerMap<'_>, operands: &[u32]) -> AccessPattern {
    if operands.len() != 2 {
        return AccessPattern::Scattered;
    }
    // We're looking for k * LocalInvocationId.x or LocalInvocationId.x * k.
    let const_operand = {
        let l = classify_index(body, producers, operands[0]);
        let r = classify_index(body, producers, operands[1]);
        match (l, r) {
            (AccessPattern::CoalescedUnitStride, AccessPattern::Broadcast) => operands[1],
            (AccessPattern::Broadcast, AccessPattern::CoalescedUnitStride) => operands[0],
            _ => return AccessPattern::Scattered,
        }
    };

    let stride = constant_u32_operand(body, producers, const_operand);

    match stride {
        Some(0) => AccessPattern::Broadcast,
        Some(1) => AccessPattern::CoalescedUnitStride,
        Some(k) if k > 1 => AccessPattern::Strided { stride: k },
        _ => AccessPattern::Scattered,
    }
}

fn classify_pool_operand(body: &KernelBody, operand_id: u32) -> AccessPattern {
    if body.literals.get(operand_id as usize).is_some() {
        AccessPattern::Broadcast
    } else {
        AccessPattern::Scattered
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyses::AccessKind;
    use crate::descriptor_builder::{
        binop, body, descriptor, effect, global_ro, global_rw, if_then, lit, load_global, op,
        store_global,
    };
    use crate::{KernelBody, KernelDescriptor, KernelOp, LiteralValue};
    use vyre_foundation::ir::{BinOp, DataType};

    /// A 64-thread kernel over a single read-write `u32` global.
    fn one_buffer_kernel(body: impl Into<KernelBody>) -> KernelDescriptor {
        descriptor("k")
            .slot(global_rw(0, DataType::U32, "buf"))
            .dispatch(64, 1, 1)
            .body(body)
            .build()
    }

    /// `LocalInvocationId` on the given axis.
    fn tid(axis: impl Into<Vec<u32>>, result: u32) -> KernelOp {
        op(KernelOpKind::LocalInvocationId, axis, result)
    }

    fn pattern_of(body: impl Into<KernelBody>) -> AccessPattern {
        analyze(&one_buffer_kernel(body)).sites[0].pattern
    }

    /// `load(buf, factor * tid)` with the multiply operands in the given
    /// order, which is the shape every stride classification is read from.
    fn scaled_load(lhs: u32, rhs: u32, factor: u32) -> KernelBody {
        body()
            .op(tid([], 0))
            .op(lit(0, 1))
            .op(binop(BinOp::Mul, lhs, rhs, 2))
            .op(load_global(0, 2, 3))
            .literal(LiteralValue::U32(factor))
            .build()
    }

    // Positive truth (coalesced detected)

    #[test]
    fn positive_load_at_local_invocation_id_is_coalesced() {
        // tid = LocalInvocationId; load(buf, tid)
        let r = analyze(&one_buffer_kernel(
            body().op(tid([], 0)).op(load_global(0, 0, 1)),
        ));
        assert_eq!(r.sites.len(), 1);
        assert_eq!(r.sites[0].pattern, AccessPattern::CoalescedUnitStride);
        assert_eq!(r.sites[0].kind, AccessKind::Load);
    }

    #[test]
    fn positive_store_at_local_invocation_id_is_coalesced() {
        let r = analyze(&one_buffer_kernel(
            body()
                .op(tid([], 0))
                .op(lit(0, 1))
                .op(store_global(0, 0, 1))
                .literal(LiteralValue::U32(7)),
        ));
        assert_eq!(r.sites.len(), 1);
        assert_eq!(r.sites[0].pattern, AccessPattern::CoalescedUnitStride);
        assert_eq!(r.sites[0].kind, AccessKind::Store);
    }

    #[test]
    fn positive_load_at_tid_plus_constant_is_coalesced() {
        // load(buf, tid + 16)  -  still coalesced unit stride
        let pattern = pattern_of(
            body()
                .op(tid([], 0))
                .op(lit(0, 1))
                .op(binop(BinOp::Add, 0, 1, 2))
                .op(load_global(0, 2, 3))
                .literal(LiteralValue::U32(16)),
        );
        assert_eq!(pattern, AccessPattern::CoalescedUnitStride);
    }

    #[test]
    fn positive_load_at_global_invocation_id_treated_as_coalesced() {
        let pattern = pattern_of(
            body()
                .op(op(KernelOpKind::GlobalInvocationId, [0], 0))
                .op(load_global(0, 0, 1)),
        );
        assert_eq!(pattern, AccessPattern::CoalescedUnitStride);
    }

    #[test]
    fn global_invocation_y_axis_is_not_unit_stride_x_coalesced() {
        let pattern = pattern_of(
            body()
                .op(op(KernelOpKind::GlobalInvocationId, [1], 0))
                .op(load_global(0, 0, 1)),
        );
        assert_eq!(pattern, AccessPattern::Scattered);
    }

    // Strided detection

    #[test]
    fn strided_4_detected_as_stride_4() {
        // load(buf, 4 * tid)  -  stride 4
        assert_eq!(
            pattern_of(scaled_load(1, 0, 4)),
            AccessPattern::Strided { stride: 4 }
        );
    }

    #[test]
    fn strided_8_with_offset_preserves_stride() {
        // load(buf, 8 * tid + 3)
        let pattern = pattern_of(
            body()
                .op(tid([], 0))
                .op(lit(0, 1))
                .op(binop(BinOp::Mul, 1, 0, 2))
                .op(lit(1, 3))
                .op(binop(BinOp::Add, 2, 3, 4))
                .op(load_global(0, 4, 5))
                .literals([LiteralValue::U32(8), LiteralValue::U32(3)]),
        );
        assert_eq!(pattern, AccessPattern::Strided { stride: 8 });
    }

    #[test]
    fn strided_with_tid_on_left_of_mul_also_detected() {
        // load(buf, tid * 4)  -  same as 4 * tid
        assert_eq!(
            pattern_of(scaled_load(0, 1, 4)),
            AccessPattern::Strided { stride: 4 }
        );
    }

    // Broadcast (constant index)

    #[test]
    fn constant_index_is_broadcast() {
        let pattern = pattern_of(
            body()
                .op(lit(0, 0))
                .op(load_global(0, 0, 1))
                .literal(LiteralValue::U32(0)),
        );
        assert_eq!(pattern, AccessPattern::Broadcast);
    }

    // Negative precision (rule does NOT fire)

    #[test]
    fn negative_load_index_from_unrelated_op_is_scattered() {
        // load(buf, sub(tid, tid))  -  not a recognized pattern
        let pattern = pattern_of(
            body()
                .op(tid([], 0))
                .op(binop(BinOp::Sub, 0, 0, 1))
                .op(load_global(0, 1, 2)),
        );
        assert_eq!(pattern, AccessPattern::Scattered);
    }

    #[test]
    fn negative_load_index_from_indirect_load_is_scattered() {
        // load(buf, load(idx_buf, tid))  -  indirect addressing
        let k = descriptor("k")
            .slots([
                global_ro(0, DataType::U32, "idx_buf"),
                global_ro(1, DataType::U32, "buf"),
            ])
            .dispatch(64, 1, 1)
            .body(
                body()
                    .op(tid([], 0))
                    .op(load_global(0, 0, 1))
                    .op(load_global(1, 1, 2)),
            )
            .build();
        let r = analyze(&k);
        // Two access sites; outer one is scattered (indirect).
        assert_eq!(r.sites.len(), 2);
        assert_eq!(r.sites[1].pattern, AccessPattern::Scattered);
    }

    #[test]
    fn negative_no_global_accesses_yields_empty_report() {
        let r = analyze(&one_buffer_kernel(
            body()
                .op(tid([], 0))
                .op(lit(0, 1))
                .op(binop(BinOp::Add, 0, 1, 2))
                .literal(LiteralValue::U32(1)),
        ));
        assert!(r.sites.is_empty());
    }

    // Adversarial

    #[test]
    fn adversarial_mul_by_one_is_coalesced_not_strided() {
        assert_eq!(
            pattern_of(scaled_load(1, 0, 1)),
            AccessPattern::CoalescedUnitStride
        );
    }

    #[test]
    fn adversarial_mul_by_zero_is_broadcast_or_scattered() {
        // 0 * tid = 0, which is a broadcast access rather than an
        // unstructured scatter.
        assert_eq!(pattern_of(scaled_load(1, 0, 0)), AccessPattern::Broadcast);
    }

    #[test]
    fn adversarial_malformed_op_with_too_few_operands_skipped_safely() {
        // A LoadGlobal with no operands shouldn't panic.
        let r = analyze(&one_buffer_kernel(
            body().op(effect(KernelOpKind::LoadGlobal, [])),
        ));
        // Malformed ops produce no coalescence site and the analysis
        // stays robust to bad input rather than panicking.
        assert!(r.sites.is_empty());
    }

    #[test]
    fn adversarial_strided_with_constant_on_both_sides_classifies_as_coalesced_for_shadow_constant()
    {
        // tid * 1 (mul by one) plus another constant = still coalesced.
        // Verifies the Add classifier sees CoalescedUnitStride + Broadcast.
        let pattern = pattern_of(
            body()
                .op(tid([], 0))
                .op(lit(0, 1))
                .op(binop(BinOp::Mul, 0, 1, 2))
                .op(lit(1, 3))
                .op(binop(BinOp::Add, 2, 3, 4))
                .op(load_global(0, 4, 5))
                .literals([LiteralValue::U32(1), LiteralValue::U32(99)]),
        );
        assert_eq!(pattern, AccessPattern::CoalescedUnitStride);
    }

    // Report aggregation

    #[test]
    fn waste_score_reflects_mixed_kernel() {
        // One coalesced, one strided 4. Expected waste: 0 + 0.75 = 0.75.
        let r = analyze(&one_buffer_kernel(
            body()
                .op(tid([], 0))
                .op(load_global(0, 0, 1))
                .op(lit(0, 2))
                .op(binop(BinOp::Mul, 2, 0, 3))
                .op(load_global(0, 3, 4))
                .literal(LiteralValue::U32(4)),
        ));
        assert_eq!(r.sites.len(), 2);
        assert!((r.waste_score() - 0.75).abs() < 1e-5);
        assert_eq!(r.problematic_count(), 1);
    }

    #[test]
    fn report_kernel_id_echoes_descriptor_id() {
        let r = analyze(&one_buffer_kernel(body()));
        assert_eq!(r.kernel_id, "k");
    }

    /// The walk reports a nested site between its parent's branch and the
    /// parent's next op, and each site is classified against the producer map
    /// of the body that owns it. A single map carried across bodies would
    /// classify the post-branch parent load `Scattered`, because its `Mul`
    /// producer lives in the parent and not in the arm.
    #[test]
    fn a_site_after_a_branch_is_classified_against_the_parent_body() {
        let r = analyze(&one_buffer_kernel(
            body()
                .op(tid([], 0))
                .op(lit(0, 1))
                .op(binop(BinOp::Mul, 0, 1, 2))
                .op(if_then(2, 0))
                .op(load_global(0, 2, 3))
                .child(body().op(tid([], 10)).op(load_global(0, 10, 11)))
                .literal(LiteralValue::U32(4)),
        ));
        assert_eq!(
            r.sites
                .iter()
                .map(|site| (site.op_index, site.pattern))
                .collect::<Vec<_>>(),
            vec![
                (6, AccessPattern::CoalescedUnitStride),
                (4, AccessPattern::Strided { stride: 4 }),
            ],
            "Fix: the arm's site must be reported before the parent's next op, and the parent's site must classify against the parent's own producers."
        );
    }
}
