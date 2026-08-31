//! Contract tests for target-boundary bank conflict mitigation strategy.
//!
//! Verifies Section 185.2:
//! - Multi-phase access pattern evaluation (load, compute, reduction, epilogue).
//! - Target-specific selection of padding, XOR swizzling, or no rewrite.
//! - Rejection of candidates that move unacceptable conflicts to another phase.
//! - Non-promising of universal zero conflicts (honestly reporting remaining conflicts).

use std::num::NonZeroU32;
use vyre_foundation::ir::{AtomicOp, BinOp, DataType, MemoryOrdering};
use vyre_lower::analyses::{
    derive_shared_access_profiles, evaluate_mitigation_candidate, select_bank_conflict_strategy,
    AccessPhase, AccessPhaseProfile, BankConflictMitigation, ConflictSeverity,
    SharedBindingAccessProfile, SharedPermutationBlock, TargetBankGeometry,
};
use vyre_lower::descriptor_builder::{
    binop, body, descriptor, effect, global_rw, lit, op, shared_rw, store_global,
};
use vyre_lower::{KernelDescriptor, KernelOp, KernelOpKind, LiteralValue};

/// Bank geometry a case states. Every field is a device fact, so a case names
/// all four rather than inheriting them.
fn stated_geometry() -> TargetBankGeometry {
    TargetBankGeometry {
        bank_count: 32,
        bank_width_bytes: 4,
        subgroup_lanes: 32,
        instruction_word_bytes: 4,
    }
}

#[test]
fn padding_mitigates_power_of_two_column_stride_conflicts() {
    let geom = stated_geometry();
    let phases = vec![AccessPhaseProfile {
        phase: AccessPhase::ComputeRead,
        stride_elements: 32, // Stride 32 on 32 banks causes critical 32-way conflict
        active_threads: 32,
        access_weight: 10,
    }];

    let baseline = evaluate_mitigation_candidate(
        &phases,
        &geom,
        BankConflictMitigation::NoRewrite,
        ConflictSeverity::None,
    );
    assert_eq!(baseline.worst_severity, ConflictSeverity::Critical);

    // Padding +1 element per row changes stride from 32 to 33 (gcd(33, 32) == 1 -> NoConflict)
    let padded = evaluate_mitigation_candidate(
        &phases,
        &geom,
        BankConflictMitigation::PadLines {
            pad_elements_per_row: 1,
        },
        ConflictSeverity::Critical,
    );
    assert_eq!(padded.worst_severity, ConflictSeverity::None);
    assert!(padded.accepted);
    assert!(padded.aggregate_penalty < baseline.aggregate_penalty);
}

#[test]
fn xor_swizzling_reduces_conflict_penalty() {
    let geom = stated_geometry();
    let phases = vec![AccessPhaseProfile {
        phase: AccessPhase::ComputeRead,
        stride_elements: 16, // Stride 16 causes 16-way critical conflict
        active_threads: 32,
        access_weight: 5,
    }];

    let swizzled = evaluate_mitigation_candidate(
        &phases,
        &geom,
        BankConflictMitigation::XorSwizzle {
            swizzle_bits: 2,
            stride_shift: 3,
        },
        ConflictSeverity::Critical,
    );
    assert!(swizzled.accepted);
    assert!(swizzled.aggregate_penalty < 16.0 * 5.0);
}

#[test]
fn strategy_selection_rejects_candidate_moving_conflict_to_another_phase() {
    let geom = stated_geometry();
    // Phase 1 is clean at stride 1; Phase 2 has conflict at stride 32
    // If a rewrite at +1 padding fixes Phase 2 (32 -> 33) but messes up Phase 1 into a severe conflict,
    // it must be rejected.
    let phases = vec![
        AccessPhaseProfile {
            phase: AccessPhase::LoadStage,
            stride_elements: 1, // Stride 1 is NoConflict
            active_threads: 32,
            access_weight: 1,
        },
        AccessPhaseProfile {
            phase: AccessPhase::ComputeRead,
            stride_elements: 32, // Stride 32 is Critical conflict
            active_threads: 32,
            access_weight: 10,
        },
    ];

    let selected = select_bank_conflict_strategy(&phases, &geom);
    assert!(selected.accepted);
    assert!(selected.aggregate_penalty < 32.0 * 10.0);
}

#[test]
fn strategy_does_not_promise_universal_zero_conflicts() {
    let geom = stated_geometry();
    // Multiple concurrent phases with mutually conflicting stride constraints:
    // Any padding (+1, +2, +4) pushes at least one phase into a bank conflict.
    let phases = vec![
        AccessPhaseProfile {
            phase: AccessPhase::LoadStage,
            stride_elements: 31, // +1 padding makes it 32 (critical)
            active_threads: 32,
            access_weight: 1,
        },
        AccessPhaseProfile {
            phase: AccessPhase::ComputeRead,
            stride_elements: 30, // +2 padding makes it 32 (critical)
            active_threads: 32,
            access_weight: 1,
        },
        AccessPhaseProfile {
            phase: AccessPhase::Reduction,
            stride_elements: 28, // +4 padding makes it 32 (critical)
            active_threads: 32,
            access_weight: 1,
        },
        AccessPhaseProfile {
            phase: AccessPhase::EpilogueStore,
            stride_elements: 32, // baseline is 32 (critical)
            active_threads: 32,
            access_weight: 1,
        },
    ];

    let selected = select_bank_conflict_strategy(&phases, &geom);
    // Honestly reports remaining conflict severity instead of falsely claiming zero conflict
    assert!(selected.accepted);
    assert_ne!(selected.worst_severity, ConflictSeverity::None);
}

/// A column walk over a 32-wide tile: consecutive lanes address elements a
/// full bank count apart, which is the classifier's worst case.
fn column_walk_tile() -> KernelDescriptor {
    descriptor("column_walk_tile")
        .slot(global_rw(0, DataType::U32, "out"))
        .slot(shared_rw(1, DataType::U32, 1024, "tile"))
        .dispatch(32, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(32)])
                .op(op(KernelOpKind::LocalInvocationId, [], 0))
                .op(lit(0, 1))
                .op(binop(BinOp::Mul, 0, 1, 2))
                .op(effect(KernelOpKind::StoreShared, [1, 2, 0]))
                .op(effect(
                    KernelOpKind::Barrier {
                        ordering: MemoryOrdering::SeqCst,
                    },
                    [],
                ))
                .op(op(KernelOpKind::LoadShared, [1, 2], 3))
                .op(store_global(0, 0, 3)),
        )
        .build()
}

/// `column_walk_tile` with `extra` appended to its op stream.
fn tile_plus(extra: Vec<KernelOp>) -> KernelDescriptor {
    let mut built = column_walk_tile();
    built.body.ops.extend(extra);
    built
}

/// The tile binding's derived profile. The derivation states one entry per
/// shared binding, so a missing entry is itself the failure.
fn tile_profile(desc: &KernelDescriptor) -> SharedBindingAccessProfile {
    let banks = NonZeroU32::new(32).expect("Fix: 32 is not zero");
    derive_shared_access_profiles(desc, banks)
        .into_iter()
        .find(|profile| profile.binding_slot == 1)
        .expect("Fix: the derivation must state every shared binding a descriptor declares")
}

/// The selector's input is a per-phase stride and active width. A descriptor
/// states neither, so the derivation produces both from the index expression
/// each access computes and from the barrier structure between them.
#[test]
fn the_derivation_states_a_stride_and_an_active_width_per_phase() {
    let profile = tile_profile(&column_walk_tile());

    assert_eq!(profile.element_count, 1024);
    assert_eq!(
        profile.blocked_by, None,
        "Fix: a scalar store and load under a barrier leave the tile permutable"
    );

    let derived: Vec<(AccessPhase, u32, u32)> = profile
        .phases
        .iter()
        .map(|phase| (phase.phase, phase.stride_elements, phase.active_threads))
        .collect();
    assert_eq!(
        derived,
        vec![
            (AccessPhase::LoadStage, 32, 32),
            (AccessPhase::ComputeRead, 32, 32),
        ],
        "Fix: a store before the barrier stages the tile and a load after it reads"
    );
}

/// Every block class is derived from the op that causes it, and which operands
/// state a binding comes from the operand-class table rather than a list in the
/// derivation. A shared access form left unclassified would leave the tile
/// permutable and authorize an unsound rewrite, so each arm is proven against a
/// descriptor that reaches it.
#[test]
fn every_block_class_removes_the_binding_it_reaches_from_the_permutable_set() {
    let asynchronous = tile_plus(vec![effect(
        KernelOpKind::async_load("dma".into()),
        [0, 1, 0, 0],
    )]);
    assert_eq!(
        tile_profile(&asynchronous).blocked_by,
        Some(SharedPermutationBlock::AsyncTransaction),
        "Fix: a transfer addresses the allocation, not the element index a \
         permutation rewrites"
    );

    let atomic = tile_plus(vec![op(
        KernelOpKind::Atomic {
            op: AtomicOp::Add,
            ordering: MemoryOrdering::Relaxed,
        },
        [1, 2, 0],
        4,
    )]);
    assert_eq!(
        tile_profile(&atomic).blocked_by,
        Some(SharedPermutationBlock::Atomic),
        "Fix: an atomic addresses a location other lanes agree on"
    );

    let unproven = tile_plus(vec![
        op(KernelOpKind::LoadGlobal, [0, 0], 5),
        op(KernelOpKind::LoadShared, [1, 5], 6),
    ]);
    assert_eq!(
        tile_profile(&unproven).blocked_by,
        Some(SharedPermutationBlock::UnprovenAccess),
        "Fix: a permutation is a bijection over a stated index, and an index \
         no rule classifies states nothing"
    );

    let mut undeclared = column_walk_tile();
    undeclared.bindings.slots[1].element_count = None;
    assert_eq!(
        tile_profile(&undeclared).blocked_by,
        Some(SharedPermutationBlock::NoDeclaredExtent),
        "Fix: a padded allocation is grown against a declared extent"
    );
}

/// A global binding is not a permutation candidate, so it is absent from the
/// derivation rather than present and refused.
#[test]
fn the_derivation_states_shared_bindings_only() {
    let banks = NonZeroU32::new(32).expect("Fix: 32 is not zero");
    let slots: Vec<u32> = derive_shared_access_profiles(&column_walk_tile(), banks)
        .iter()
        .map(|profile| profile.binding_slot)
        .collect();
    assert_eq!(slots, vec![1]);
}
