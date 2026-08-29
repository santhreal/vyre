//! Analysis pass: walk a `KernelDescriptor`, classify each
//! `LoadShared`/`StoreShared` op for bank conflicts.
//!
//! ## Algorithm
//!
//! For each shared-memory access op, examine the index expression's
//! relationship to `LocalInvocationId.x` / `GlobalInvocationId.x`:
//!
//! 1. Index = `tid`                        → addresses are
//!    `0, 1, 2, ..., warp_size-1`. They map to banks
//!    `0 % B, 1 % B, ...`. Distinct banks since `gcd(1, B) == 1`.
//!    Result: **NoConflict**.
//! 2. Index = `tid + const`                → same as above with shift.
//!    NoConflict.
//! 3. Index = `tid * k` for constant k     → addresses are
//!    `0, k, 2k, ...`. Bank pattern depends on `gcd(k, B)`. If
//!    `gcd(k, B) == 1`, NoConflict. Otherwise, way-count is `gcd(k, B)`.
//!    For B=32, k=2 → 2-way. k=4 → 4-way. k=32 → 32-way (worst).
//! 4. Index = `tid * k + const`            → same as case 3.
//! 5. Index = constant                     → all threads read same
//!    address → BroadcastSafe (for read; conflict for write but we
//!    flag NoConflict for now since broadcast-write is a different
//!    correctness concern, not a bank conflict).
//! 6. Anything else                        → Unknown.

use super::report::{BankAccessSite, BankConflictKind, BankConflictReport};
use crate::analyses::constant_u32_operand;
use crate::analyses::gcd_u32;
use crate::analyses::structured_walk::walk_accesses;
use crate::analyses::ProducerMap;
use crate::{KernelBody, KernelDescriptor, KernelOpKind, MemoryClass};
use std::num::NonZeroU32;
use vyre_foundation::ir::BinOp;

/// Run bank-conflict analysis for a stated shared-memory bank count.
///
/// The bank count is a device fact and has no neutral value, so a caller
/// passes the count the target reported. A caller with no reported count runs
/// no bank analysis rather than assuming a layout.
#[must_use]
pub fn analyze(desc: &KernelDescriptor, banks: NonZeroU32) -> BankConflictReport {
    let bank_count = banks.get();
    let mut sites = Vec::new();
    walk_accesses(
        &desc.body,
        &KernelOpKind::LoadShared,
        &KernelOpKind::StoreShared,
        |access| {
            // We only flag accesses whose target binding is in the Shared
            // memory class  -  guards against a future emitter using
            // LoadShared on a non-shared binding (which would be invalid
            // but the analysis stays robust).
            let is_shared = desc.bindings.slots.iter().any(|b| {
                b.slot == access.binding_slot && matches!(b.memory_class, MemoryClass::Shared)
            });
            if !is_shared {
                return;
            }
            let pattern = classify_index(
                access.body,
                access.producers,
                access.index_operand_id,
                bank_count,
            );
            sites.push(BankAccessSite {
                op_index: access.op_index,
                kind: access.kind,
                binding_slot: access.binding_slot,
                conflict: pattern.conflict,
            });
        },
    );
    BankConflictReport {
        kernel_id: desc.id.clone(),
        bank_count,
        sites,
    }
}

/// What classification proved about one shared access's index.
///
/// The conflict class alone is not enough to build a mitigation profile: two
/// strides that collide by the same gcd rank differently once a candidate
/// rewrites the stride, so the stride travels with the class. `None` means the
/// analysis proved no stride, which is not the same as a stride of zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct IndexPattern {
    /// Conflict class for the access.
    pub(super) conflict: BankConflictKind,
    /// Element stride between consecutive lanes, when one was proven.
    pub(super) stride_elements: Option<u32>,
}

impl IndexPattern {
    /// A classified access with a proven stride.
    const fn new(conflict: BankConflictKind, stride_elements: u32) -> Self {
        Self {
            conflict,
            stride_elements: Some(stride_elements),
        }
    }

    /// An access no rule classified.
    const fn unknown() -> Self {
        Self {
            conflict: BankConflictKind::Unknown,
            stride_elements: None,
        }
    }

    /// Every lane reads one address.
    const fn broadcast() -> Self {
        Self::new(BankConflictKind::BroadcastSafe, 0)
    }

    /// Consecutive lanes read consecutive elements.
    const fn unit() -> Self {
        Self::new(BankConflictKind::NoConflict, 1)
    }

    /// Whether the class is a lane-varying one a stride can be carried through.
    const fn lane_varying(self) -> bool {
        matches!(
            self.conflict,
            BankConflictKind::NoConflict | BankConflictKind::Conflict { .. }
        )
    }
}

pub(super) fn classify_index(
    body: &KernelBody,
    producers: &ProducerMap<'_>,
    index_operand_id: u32,
    bank_count: u32,
) -> IndexPattern {
    let producer = match producers.get(&index_operand_id).copied() {
        Some(producer) => producer,
        None => {
            return if body.literals.get(index_operand_id as usize).is_some() {
                IndexPattern::broadcast()
            } else {
                IndexPattern::unknown()
            };
        }
    };

    match &producer.kind {
        KernelOpKind::LocalInvocationId | KernelOpKind::GlobalInvocationId => {
            classify_invocation_id(producer)
        }
        KernelOpKind::Literal => IndexPattern::broadcast(),
        KernelOpKind::BinOpKind(BinOp::Add | BinOp::WrappingAdd) => {
            classify_add(body, producers, &producer.operands, bank_count)
        }
        KernelOpKind::BinOpKind(BinOp::Mul) => {
            classify_mul(body, producers, &producer.operands, bank_count)
        }
        // `x << c` is `x * 2^c`. The optimizer decomposes a constant multiply
        // into a shift before any backend sees it, so a classifier that only
        // reads the multiply form states no stride for the very access a
        // strength reduction just rewrote.
        KernelOpKind::BinOpKind(BinOp::Shl) => {
            classify_shl(body, producers, &producer.operands, bank_count)
        }
        _ => IndexPattern::unknown(),
    }
}

fn classify_invocation_id(op: &crate::KernelOp) -> IndexPattern {
    match op.operands.first().copied().unwrap_or(0) {
        0 => IndexPattern::unit(),
        _ => IndexPattern::unknown(),
    }
}

fn classify_add(
    body: &KernelBody,
    producers: &ProducerMap<'_>,
    operands: &[u32],
    bank_count: u32,
) -> IndexPattern {
    if operands.len() != 2 {
        return IndexPattern::unknown();
    }
    let lhs = classify_index(body, producers, operands[0], bank_count);
    let rhs = classify_index(body, producers, operands[1], bank_count);
    // Adding a lane-invariant offset shifts every address by the same amount,
    // so the bank pattern is the lane-varying side's, stride included.
    match (lhs.conflict, rhs.conflict) {
        (BankConflictKind::BroadcastSafe, _) if rhs.lane_varying() => rhs,
        (_, BankConflictKind::BroadcastSafe) if lhs.lane_varying() => lhs,
        _ => IndexPattern::unknown(),
    }
}

fn classify_mul(
    body: &KernelBody,
    producers: &ProducerMap<'_>,
    operands: &[u32],
    bank_count: u32,
) -> IndexPattern {
    if operands.len() != 2 {
        return IndexPattern::unknown();
    }
    let l = classify_index(body, producers, operands[0], bank_count);
    let r = classify_index(body, producers, operands[1], bank_count);
    let const_operand = match (l.conflict, r.conflict) {
        (BankConflictKind::NoConflict, BankConflictKind::BroadcastSafe) => operands[1],
        (BankConflictKind::BroadcastSafe, BankConflictKind::NoConflict) => operands[0],
        _ => return IndexPattern::unknown(),
    };

    let stride = constant_u32_operand(body, producers, const_operand);

    let stride = match stride {
        Some(s) => s,
        None => return IndexPattern::unknown(),
    };

    if stride == 0 {
        // tid * 0 = 0  -  all threads same address → broadcast.
        return IndexPattern::broadcast();
    }
    let g = gcd_u32(stride, bank_count);
    if g == 1 {
        IndexPattern::new(BankConflictKind::NoConflict, stride)
    } else {
        IndexPattern::new(BankConflictKind::Conflict { way_count: g }, stride)
    }
}

/// `x << c` for a lane-varying `x` and a constant `c`.
///
/// Only the shift amount may be constant: a lane-varying shift amount is not a
/// stride, and a constant shifted by a lane-varying amount is not lane-varying
/// by a fixed step either.
fn classify_shl(
    body: &KernelBody,
    producers: &ProducerMap<'_>,
    operands: &[u32],
    bank_count: u32,
) -> IndexPattern {
    if operands.len() != 2 {
        return IndexPattern::unknown();
    }
    let base = classify_index(body, producers, operands[0], bank_count);
    if !matches!(base.conflict, BankConflictKind::NoConflict) {
        return IndexPattern::unknown();
    }
    let Some(shift) = constant_u32_operand(body, producers, operands[1]) else {
        return IndexPattern::unknown();
    };
    // A shift at or past the word width is not a stride this analysis states.
    if shift >= 32 {
        return IndexPattern::unknown();
    }
    let Some(stride) = base.stride_elements.and_then(|s| s.checked_shl(shift)) else {
        return IndexPattern::unknown();
    };
    if stride == 0 {
        return IndexPattern::broadcast();
    }
    let g = gcd_u32(stride, bank_count);
    if g == 1 {
        IndexPattern::new(BankConflictKind::NoConflict, stride)
    } else {
        IndexPattern::new(BankConflictKind::Conflict { way_count: g }, stride)
    }
}

// Inline: covers the crate-private `analyze` and `gcd_u32`, which no integration test can reach.
#[cfg(test)]
mod tests {
    /// Bank count the fixtures state. The analysis has no default, so every
    /// case below names the layout it classifies against.
    const BANKS: NonZeroU32 = match NonZeroU32::new(32) {
        Some(banks) => banks,
        None => unreachable!(),
    };

    fn banks(count: u32) -> NonZeroU32 {
        NonZeroU32::new(count).expect("a stated bank count is nonzero")
    }

    use super::*;
    use crate::descriptor_builder::{
        binop, body, descriptor, effect, for_loop, global_ro, if_then, lit, op, shared_rw,
    };
    use crate::{BindingSlot, KernelBody, KernelDescriptor, KernelOp, KernelOpKind, LiteralValue};
    use vyre_foundation::ir::{BinOp, DataType};

    fn shared_binding(slot: u32) -> BindingSlot {
        shared_rw(slot, DataType::F32, 1024, &format!("shared{slot}"))
    }

    /// `LocalInvocationId` on the given axis.
    fn tid(axis: impl Into<Vec<u32>>, result: u32) -> KernelOp {
        op(KernelOpKind::LocalInvocationId, axis, result)
    }

    /// A 32-thread kernel over one shared binding.
    fn k(slots: Vec<BindingSlot>, body: impl Into<KernelBody>) -> KernelDescriptor {
        descriptor("k")
            .slots(slots)
            .dispatch(32, 1, 1)
            .body(body)
            .build()
    }

    /// The canonical strided access `shared[tid * stride]`, which is the
    /// only shape that distinguishes the conflict classifications below.
    fn strided_load(stride: u32) -> KernelDescriptor {
        k(
            vec![shared_binding(0)],
            body()
                .op(tid([], 0))
                .op(lit(0, 1))
                .op(binop(BinOp::Mul, 0, 1, 2))
                .op(op(KernelOpKind::LoadShared, [0, 2], 3))
                .literal(LiteralValue::U32(stride)),
        )
    }

    /// `shared[tid << shift]`, the form a constant multiply is decomposed into
    /// before any backend reads the descriptor.
    fn shifted_load(shift: u32) -> KernelDescriptor {
        k(
            vec![shared_binding(0)],
            body()
                .op(tid([], 0))
                .op(lit(0, 1))
                .op(binop(BinOp::Shl, 0, 1, 2))
                .op(op(KernelOpKind::LoadShared, [0, 2], 3))
                .literal(LiteralValue::U32(shift)),
        )
    }

    fn conflict_of(stride: u32) -> BankConflictKind {
        analyze(&strided_load(stride), BANKS).sites[0].conflict
    }

    // Positive truth (no conflict detected)

    #[test]
    fn positive_load_at_tid_no_conflict() {
        let kk = k(
            vec![shared_binding(0)],
            body()
                .op(tid([], 0))
                .op(op(KernelOpKind::LoadShared, [0, 0], 1)),
        );
        let r = analyze(&kk, BANKS);
        assert_eq!(r.sites.len(), 1);
        assert_eq!(r.sites[0].conflict, BankConflictKind::NoConflict);
    }

    #[test]
    fn local_invocation_y_axis_is_unknown_not_x_lane_no_conflict() {
        let kk = k(
            vec![shared_binding(0)],
            body()
                .op(tid([1], 0))
                .op(op(KernelOpKind::LoadShared, [0, 0], 1)),
        );
        let r = analyze(&kk, BANKS);
        assert_eq!(r.sites.len(), 1);
        assert_eq!(r.sites[0].conflict, BankConflictKind::Unknown);
    }

    #[test]
    fn positive_load_at_tid_plus_const_no_conflict() {
        let kk = k(
            vec![shared_binding(0)],
            body()
                .op(tid([], 0))
                .op(lit(0, 1))
                .op(binop(BinOp::Add, 0, 1, 2))
                .op(op(KernelOpKind::LoadShared, [0, 2], 3))
                .literal(LiteralValue::U32(99)),
        );
        let r = analyze(&kk, BANKS);
        assert_eq!(r.sites[0].conflict, BankConflictKind::NoConflict);
    }

    #[test]
    fn positive_constant_index_is_broadcast_safe() {
        let kk = k(
            vec![shared_binding(0)],
            body()
                .op(lit(0, 0))
                .op(op(KernelOpKind::LoadShared, [0, 0], 1))
                .literal(LiteralValue::U32(0)),
        );
        let r = analyze(&kk, BANKS);
        assert_eq!(r.sites[0].conflict, BankConflictKind::BroadcastSafe);
    }

    // Conflict detection (the headline)

    #[test]
    fn conflict_stride_2_is_2_way() {
        assert_eq!(conflict_of(2), BankConflictKind::Conflict { way_count: 2 });
    }

    #[test]
    fn conflict_stride_4_is_4_way() {
        assert_eq!(conflict_of(4), BankConflictKind::Conflict { way_count: 4 });
    }

    #[test]
    fn conflict_stride_32_is_32_way_critical() {
        // The classic shared-mem matmul column-major worst case.
        let r = analyze(&strided_load(32), BANKS);
        assert_eq!(
            r.sites[0].conflict,
            BankConflictKind::Conflict { way_count: 32 }
        );
        assert_eq!(r.problematic_count(), 1);
        assert_eq!(r.critical_count(), 1);
    }

    #[test]
    fn no_conflict_for_stride_coprime_to_bank_count() {
        // gcd(3, 32) == 1 → no conflict.
        assert_eq!(conflict_of(3), BankConflictKind::NoConflict);
    }

    #[test]
    fn stride_1_is_no_conflict() {
        // gcd(1, 32) == 1.
        assert_eq!(conflict_of(1), BankConflictKind::NoConflict);
    }

    #[test]
    fn stride_0_is_broadcast_safe() {
        assert_eq!(conflict_of(0), BankConflictKind::BroadcastSafe);
    }

    // Negative precision (rule does NOT fire)

    #[test]
    fn negative_global_load_not_analyzed() {
        // LoadGlobal  -  not LoadShared. Bank-conflict analysis is
        // only for shared memory.
        let kk = k(
            vec![global_ro(0, DataType::F32, "buf")],
            body()
                .op(tid([], 0))
                .op(op(KernelOpKind::LoadGlobal, [0, 0], 1)),
        );
        let r = analyze(&kk, BANKS);
        assert!(r.sites.is_empty());
    }

    #[test]
    fn negative_load_shared_against_global_binding_skipped() {
        // Robustness: an emitter bug that emits LoadShared against a
        // Global-class binding shouldn't be analyzed as bank conflict.
        // We skip it.
        let kk = k(
            vec![global_ro(0, DataType::F32, "buf")],
            body()
                .op(tid([], 0))
                .op(op(KernelOpKind::LoadShared, [0, 0], 1)),
        );
        let r = analyze(&kk, BANKS);
        assert!(r.sites.is_empty());
    }

    // Adversarial / boundary

    #[test]
    fn adversarial_load_inside_loop_body_counted() {
        let kk = k(
            vec![shared_binding(0)],
            body()
                .op(lit(0, 0))
                .op(lit(0, 1))
                .op(for_loop("", 0, 1, 0))
                .child(
                    body()
                        .op(tid([], 0))
                        .op(lit(0, 1))
                        .op(binop(BinOp::Mul, 0, 1, 2))
                        .op(op(KernelOpKind::LoadShared, [0, 2], 3))
                        .literal(LiteralValue::U32(8)),
                )
                .literal(LiteralValue::U32(0)),
        );
        let r = analyze(&kk, BANKS);
        // gcd(8, 32) == 8 → 8-way conflict.
        assert_eq!(r.sites.len(), 1);
        assert_eq!(
            r.sites[0].conflict,
            BankConflictKind::Conflict { way_count: 8 }
        );
    }

    #[test]
    fn adversarial_unrecognized_index_pattern_is_unknown() {
        let kk = k(
            vec![shared_binding(0)],
            body().op(tid([], 0)).op(binop(BinOp::Sub, 0, 0, 1)).op(op(
                KernelOpKind::LoadShared,
                [0, 1],
                2,
            )),
        );
        let r = analyze(&kk, BANKS);
        assert_eq!(r.sites[0].conflict, BankConflictKind::Unknown);
    }

    #[test]
    fn adversarial_malformed_load_shared_skipped_safely() {
        let kk = k(
            vec![shared_binding(0)],
            body().op(effect(KernelOpKind::LoadShared, [])),
        );
        let r = analyze(&kk, BANKS);
        assert!(r.sites.is_empty());
    }

    // Bank-count override

    #[test]
    fn analyze_with_16_banks_changes_classification() {
        // Stride 3 is coprime with 32 banks but shares a factor of 3 with
        // 6, so the bank count decides the classification.
        let kk = strided_load(3);
        let r32 = analyze(&kk, banks(32));
        assert_eq!(r32.sites[0].conflict, BankConflictKind::NoConflict);
        let r6 = analyze(&kk, banks(6));
        assert_eq!(
            r6.sites[0].conflict,
            BankConflictKind::Conflict { way_count: 3 }
        );
    }

    /// The walk reports a nested site between its parent's branch and the
    /// parent's next op, and each site is classified against the producer map
    /// of the body that owns it. A single map carried across bodies would
    /// classify the post-branch parent load `Unknown`, because its `Mul`
    /// producer lives in the parent and not in the arm.
    #[test]
    fn a_site_after_a_branch_is_classified_against_the_parent_body() {
        let kk = k(
            vec![shared_binding(0)],
            body()
                .op(tid([], 0))
                .op(lit(0, 1))
                .op(binop(BinOp::Mul, 0, 1, 2))
                .op(if_then(2, 0))
                .op(op(KernelOpKind::LoadShared, [0, 2], 3))
                .child(
                    body()
                        .op(tid([], 10))
                        .op(op(KernelOpKind::LoadShared, [0, 10], 11)),
                )
                .literal(LiteralValue::U32(4)),
        );
        let r = analyze(&kk, BANKS);
        assert_eq!(
            r.sites
                .iter()
                .map(|site| (site.op_index, site.conflict))
                .collect::<Vec<_>>(),
            vec![
                (6, BankConflictKind::NoConflict),
                (4, BankConflictKind::Conflict { way_count: 4 }),
            ],
            "Fix: the arm's site must be reported before the parent's next op, and the parent's site must classify against the parent's own producers."
        );
    }

    // gcd helper

    #[test]
    fn gcd_basic_cases() {
        assert_eq!(super::gcd_u32(8, 32), 8);
        assert_eq!(super::gcd_u32(7, 32), 1);
        assert_eq!(super::gcd_u32(1, 1), 1);
        assert_eq!(super::gcd_u32(0, 5), 5);
        assert_eq!(super::gcd_u32(5, 0), 5);
        assert_eq!(super::gcd_u32(12, 18), 6);
    }

    // Shift form of a constant multiply

    /// Strength reduction rewrites `tid * 2^c` into `tid << c` in the neutral
    /// optimizer, so the two forms have to classify identically. Reading only
    /// the multiply form states no stride for the access a pass just rewrote,
    /// and the mitigation it authorizes is refused for the wrong reason.
    #[test]
    fn a_constant_shift_classifies_as_the_multiply_it_replaces() {
        for shift in 0..6_u32 {
            let stride = 1_u32 << shift;
            assert_eq!(
                analyze(&shifted_load(shift), BANKS).sites[0].conflict,
                conflict_of(stride),
                "shared[tid << {shift}] must classify as shared[tid * {stride}]"
            );
        }
    }

    /// A lane-varying shift amount is not a stride. `tid << tid` moves every
    /// lane by a different power of two, so no fixed step describes it.
    #[test]
    fn a_lane_varying_shift_amount_is_not_a_stride() {
        let kk = k(
            vec![shared_binding(0)],
            body().op(tid([], 0)).op(binop(BinOp::Shl, 0, 0, 1)).op(op(
                KernelOpKind::LoadShared,
                [0, 1],
                2,
            )),
        );
        assert_eq!(
            analyze(&kk, BANKS).sites[0].conflict,
            BankConflictKind::Unknown
        );
    }

    /// A shift at or past the word width leaves the element range, so no
    /// stride is stated for it rather than a wrapped one.
    #[test]
    fn a_shift_past_the_word_width_states_no_stride() {
        for shift in [32_u32, 33, 64] {
            assert_eq!(
                analyze(&shifted_load(shift), BANKS).sites[0].conflict,
                BankConflictKind::Unknown,
                "shared[tid << {shift}] states no stride"
            );
        }
    }
}
