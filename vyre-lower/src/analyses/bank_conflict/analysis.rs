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
use super::DEFAULT_BANK_COUNT;
use crate::analyses::constant_u32_operand;
use crate::analyses::structured_walk::walk_accesses;
use crate::analyses::ProducerMap;
use crate::{KernelBody, KernelDescriptor, KernelOpKind, MemoryClass};
use vyre_foundation::ir::BinOp;

/// Run bank-conflict analysis using the default 32-bank layout.
#[must_use]
pub fn analyze(desc: &KernelDescriptor) -> BankConflictReport {
    analyze_with_bank_count(desc, DEFAULT_BANK_COUNT)
}

/// Run bank-conflict analysis with an explicit bank count.
#[must_use]
pub fn analyze_with_bank_count(desc: &KernelDescriptor, bank_count: u32) -> BankConflictReport {
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
            sites.push(BankAccessSite {
                op_index: access.op_index,
                kind: access.kind,
                binding_slot: access.binding_slot,
                conflict: classify_index(
                    access.body,
                    access.producers,
                    access.index_operand_id,
                    bank_count,
                ),
            });
        },
    );
    BankConflictReport {
        kernel_id: desc.id.clone(),
        bank_count,
        sites,
    }
}

fn classify_index(
    body: &KernelBody,
    producers: &ProducerMap<'_>,
    index_operand_id: u32,
    bank_count: u32,
) -> BankConflictKind {
    let producer = match producers.get(&index_operand_id).copied() {
        Some(producer) => producer,
        None => {
            return if body.literals.get(index_operand_id as usize).is_some() {
                BankConflictKind::BroadcastSafe
            } else {
                BankConflictKind::Unknown
            };
        }
    };

    match &producer.kind {
        KernelOpKind::LocalInvocationId | KernelOpKind::GlobalInvocationId => {
            classify_invocation_id(producer)
        }
        KernelOpKind::Literal => BankConflictKind::BroadcastSafe,
        KernelOpKind::BinOpKind(BinOp::Add | BinOp::WrappingAdd) => {
            classify_add(body, producers, &producer.operands, bank_count)
        }
        KernelOpKind::BinOpKind(BinOp::Mul) => {
            classify_mul(body, producers, &producer.operands, bank_count)
        }
        _ => BankConflictKind::Unknown,
    }
}

fn classify_invocation_id(op: &crate::KernelOp) -> BankConflictKind {
    match op.operands.first().copied().unwrap_or(0) {
        0 => BankConflictKind::NoConflict,
        _ => BankConflictKind::Unknown,
    }
}

fn classify_add(
    body: &KernelBody,
    producers: &ProducerMap<'_>,
    operands: &[u32],
    bank_count: u32,
) -> BankConflictKind {
    if operands.len() != 2 {
        return BankConflictKind::Unknown;
    }
    let lhs = classify_index(body, producers, operands[0], bank_count);
    let rhs = classify_index(body, producers, operands[1], bank_count);
    match (lhs, rhs) {
        (BankConflictKind::NoConflict, BankConflictKind::BroadcastSafe)
        | (BankConflictKind::BroadcastSafe, BankConflictKind::NoConflict) => {
            BankConflictKind::NoConflict
        }
        (BankConflictKind::Conflict { way_count }, BankConflictKind::BroadcastSafe)
        | (BankConflictKind::BroadcastSafe, BankConflictKind::Conflict { way_count }) => {
            BankConflictKind::Conflict { way_count }
        }
        _ => BankConflictKind::Unknown,
    }
}

fn classify_mul(
    body: &KernelBody,
    producers: &ProducerMap<'_>,
    operands: &[u32],
    bank_count: u32,
) -> BankConflictKind {
    if operands.len() != 2 {
        return BankConflictKind::Unknown;
    }
    let l = classify_index(body, producers, operands[0], bank_count);
    let r = classify_index(body, producers, operands[1], bank_count);
    let const_operand = match (l, r) {
        (BankConflictKind::NoConflict, BankConflictKind::BroadcastSafe) => operands[1],
        (BankConflictKind::BroadcastSafe, BankConflictKind::NoConflict) => operands[0],
        _ => return BankConflictKind::Unknown,
    };

    let stride = constant_u32_operand(body, producers, const_operand);

    let stride = match stride {
        Some(s) => s,
        None => return BankConflictKind::Unknown,
    };

    if stride == 0 {
        // tid * 0 = 0  -  all threads same address → broadcast.
        return BankConflictKind::BroadcastSafe;
    }
    let g = gcd_u32(stride, bank_count);
    if g == 1 {
        BankConflictKind::NoConflict
    } else {
        BankConflictKind::Conflict { way_count: g }
    }
}

fn gcd_u32(a: u32, b: u32) -> u32 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor_builder::{
        binop, body, descriptor, effect, global_ro, lit, op, shared_rw,
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

    fn conflict_of(stride: u32) -> BankConflictKind {
        analyze(&strided_load(stride)).sites[0].conflict
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
        let r = analyze(&kk);
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
        let r = analyze(&kk);
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
        let r = analyze(&kk);
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
        let r = analyze(&kk);
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
        let r = analyze(&strided_load(32));
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
        let r = analyze(&kk);
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
        let r = analyze(&kk);
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
                .op(effect(
                    KernelOpKind::StructuredForLoop {
                        loop_var: "".into(),
                    },
                    [0, 1, 0],
                ))
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
        let r = analyze(&kk);
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
        let r = analyze(&kk);
        assert_eq!(r.sites[0].conflict, BankConflictKind::Unknown);
    }

    #[test]
    fn adversarial_malformed_load_shared_skipped_safely() {
        let kk = k(
            vec![shared_binding(0)],
            body().op(effect(KernelOpKind::LoadShared, [])),
        );
        let r = analyze(&kk);
        assert!(r.sites.is_empty());
    }

    // Bank-count override

    #[test]
    fn analyze_with_16_banks_changes_classification() {
        // Stride 3 is coprime with 32 banks but shares a factor of 3 with
        // 6, so the bank count decides the classification.
        let kk = strided_load(3);
        let r32 = analyze_with_bank_count(&kk, 32);
        assert_eq!(r32.sites[0].conflict, BankConflictKind::NoConflict);
        let r6 = analyze_with_bank_count(&kk, 6);
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
                .op(effect(KernelOpKind::StructuredIfThen, [2, 0]))
                .op(op(KernelOpKind::LoadShared, [0, 2], 3))
                .child(
                    body()
                        .op(tid([], 10))
                        .op(op(KernelOpKind::LoadShared, [0, 10], 11)),
                )
                .literal(LiteralValue::U32(4)),
        );
        let r = analyze(&kk);
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
}
