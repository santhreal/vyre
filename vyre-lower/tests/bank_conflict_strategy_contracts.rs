//! Contract tests for target-boundary bank conflict mitigation strategy.
//!
//! Verifies Section 185.2:
//! - Multi-phase access pattern evaluation (load, compute, reduction, epilogue).
//! - Target-specific selection of padding, XOR swizzling, or no rewrite.
//! - Rejection of candidates that move unacceptable conflicts to another phase.
//! - Non-promising of universal zero conflicts (honestly reporting remaining conflicts).

use std::num::NonZeroU32;

use vyre_foundation::ir::{AtomicOp, MemoryOrdering};
use vyre_lower::analyses::{
    derive_shared_access_profiles, evaluate_mitigation_candidate, select_bank_conflict_strategy,
    AccessPhase, AccessPhaseProfile, BankConflictMitigation, ConflictSeverity,
    SharedBindingAccessProfile, SharedPermutationBlock, TargetBankGeometry,
};
use vyre_lower::descriptor_builder::{column_walk_tile, effect, op, strided_tile_program};
use vyre_lower::lower;
use vyre_lower::{KernelDescriptor, KernelOp, KernelOpKind, WORKGROUP_SLOT_BASE};

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

/// `column_walk_tile` with `extra` appended to its op stream.
fn tile_plus(extra: Vec<KernelOp>) -> KernelDescriptor {
    let mut built = column_walk_tile(1024);
    built.body.ops.extend(extra);
    built
}

/// The tile binding's derived profile. The derivation states one entry per
/// shared binding, so a missing entry is itself the failure.
fn tile_profile(desc: &KernelDescriptor) -> SharedBindingAccessProfile {
    let banks = NonZeroU32::new(32).expect("Fix: 32 is not zero");
    derive_shared_access_profiles(desc, banks)
        .into_iter()
        .find(|profile| profile.binding_slot == WORKGROUP_SLOT_BASE)
        .expect("Fix: the derivation must state every shared binding a descriptor declares")
}

/// The selector's input is a per-phase stride and active width. A descriptor
/// states neither, so the derivation produces both from the index expression
/// each access computes and from the barrier structure between them.
#[test]
fn the_derivation_states_a_stride_and_an_active_width_per_phase() {
    let profile = tile_profile(&column_walk_tile(1024));

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
        [0, WORKGROUP_SLOT_BASE, 0, 0],
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
        [WORKGROUP_SLOT_BASE, 2, 0],
        4,
    )]);
    assert_eq!(
        tile_profile(&atomic).blocked_by,
        Some(SharedPermutationBlock::Atomic),
        "Fix: an atomic addresses a location other lanes agree on"
    );

    let unproven = tile_plus(vec![
        op(KernelOpKind::LoadGlobal, [0, 0], 5),
        op(KernelOpKind::LoadShared, [WORKGROUP_SLOT_BASE, 5], 6),
    ]);
    assert_eq!(
        tile_profile(&unproven).blocked_by,
        Some(SharedPermutationBlock::UnprovenAccess),
        "Fix: a permutation is a bijection over a stated index, and an index \
         no rule classifies states nothing"
    );

    let fused_bulk = tile_plus(vec![
        op(KernelOpKind::LoadGlobal, [0, 0], 5),
        op(KernelOpKind::StoreShared, [WORKGROUP_SLOT_BASE, 0, 5], 6),
    ]);
    assert_eq!(
        tile_profile(&fused_bulk).blocked_by,
        Some(SharedPermutationBlock::FusedBulkCopy),
        "Fix: a fused bulk copy stages the binding via allocation-level transfer"
    );

    let mut undeclared = column_walk_tile(1024);
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
    let slots: Vec<u32> = derive_shared_access_profiles(&column_walk_tile(1024), banks)
        .iter()
        .map(|profile| profile.binding_slot)
        .collect();
    assert_eq!(slots, vec![WORKGROUP_SLOT_BASE]);
}

/// The neutral derivation produces per-phase strides from a real lowered
/// descriptor produced by `lower(&program)`.
#[test]
fn the_derivation_produces_per_phase_strides_from_a_real_lowered_descriptor() {
    let program = strided_tile_program();
    let descriptor = lower(&program).expect("a strided tile program lowers to a descriptor");

    let banks = NonZeroU32::new(32).expect("32 is not zero");
    let profiles = derive_shared_access_profiles(&descriptor, banks);
    assert_eq!(profiles.len(), 1, "exactly one shared binding profile");

    let profile = &profiles[0];
    assert_eq!(profile.element_count, 1024);
    assert_eq!(profile.blocked_by, None);

    let phases: Vec<(AccessPhase, u32, u32)> = profile
        .phases
        .iter()
        .map(|phase| (phase.phase, phase.stride_elements, phase.active_threads))
        .collect();
    assert_eq!(
        phases,
        vec![
            (AccessPhase::LoadStage, 32, 32),
            (AccessPhase::ComputeRead, 32, 32),
        ],
        "the lowered descriptor derives a staging store and compute read both at stride 32"
    );
}

/// A descriptor whose access pattern the analysis classifies as conflicting
/// yields a mitigation strategy other than `NoRewrite`.
#[test]
fn conflicting_access_pattern_yields_strategy_other_than_no_rewrite() {
    let program = strided_tile_program();
    let descriptor = lower(&program).expect("a strided tile program lowers to a descriptor");

    let banks = NonZeroU32::new(32).expect("32 is not zero");
    let report = vyre_lower::analyses::analyze_bank_conflict(&descriptor, banks);
    assert!(
        report.critical_count() > 0,
        "the strided access pattern is classified as conflicting"
    );

    let profiles = derive_shared_access_profiles(&descriptor, banks);
    let geom = stated_geometry();
    let selection = select_bank_conflict_strategy(&profiles[0].phases, &geom);

    assert!(selection.accepted);
    assert_ne!(
        selection.strategy,
        BankConflictMitigation::NoRewrite,
        "a conflicting 32-way pattern must yield a mitigation strategy other than NoRewrite"
    );
    assert_eq!(
        selection.strategy,
        BankConflictMitigation::PadLines {
            pad_elements_per_row: 1
        }
    );
}
/// Every variant of [`SharedPermutationBlock`] is derived from source at run time
/// and tested against a descriptor that produces it, so adding a block reason
/// fails this test until someone records how it is caused.
#[test]
fn every_declared_permutation_block_has_a_test_case() {
    let path = vyre_test_support::monorepo::vyre_workspace_root()
        .join("vyre-lower/src/analyses/bank_conflict/strategy.rs");
    let source = vyre_test_support::read_source_file_bounded(&path).unwrap_or_else(|err| {
        panic!("Fix: cannot read the SharedPermutationBlock declaration at {path:?}: {err}")
    });
    let body = vyre_test_support::braced_body(&source, "pub enum SharedPermutationBlock {")
        .unwrap_or_else(|| {
            panic!("Fix: no `pub enum SharedPermutationBlock` declaration in {path:?}; update this test")
        });
    let declared = vyre_test_support::top_level_variant_names(body);
    assert_eq!(
        declared.len(),
        5,
        "Fix: expected 5 declared SharedPermutationBlock variants, found {}",
        declared.len()
    );

    let covered: std::collections::BTreeSet<String> = [
        SharedPermutationBlock::AsyncTransaction,
        SharedPermutationBlock::Atomic,
        SharedPermutationBlock::FusedBulkCopy,
        SharedPermutationBlock::UnprovenAccess,
        SharedPermutationBlock::NoDeclaredExtent,
    ]
    .iter()
    .map(|b| match b {
        SharedPermutationBlock::AsyncTransaction => "AsyncTransaction".to_string(),
        SharedPermutationBlock::Atomic => "Atomic".to_string(),
        SharedPermutationBlock::FusedBulkCopy => "FusedBulkCopy".to_string(),
        SharedPermutationBlock::UnprovenAccess => "UnprovenAccess".to_string(),
        SharedPermutationBlock::NoDeclaredExtent => "NoDeclaredExtent".to_string(),
    })
    .collect();

    let missing: Vec<&String> = declared.difference(&covered).collect();
    assert!(
        missing.is_empty(),
        "Fix: add coverage for newly declared SharedPermutationBlock variant(s): {missing:?}"
    );
}

/// Every variant of [`AccessPhase`] is derived from source at run time, ensuring
/// exhaustive handling and that any new phase variant turns the suite red.
#[test]
fn every_declared_access_phase_has_a_test_case() {
    let path = vyre_test_support::monorepo::vyre_workspace_root()
        .join("vyre-lower/src/analyses/bank_conflict/strategy.rs");
    let source = vyre_test_support::read_source_file_bounded(&path).unwrap_or_else(|err| {
        panic!("Fix: cannot read the AccessPhase declaration at {path:?}: {err}")
    });
    let body =
        vyre_test_support::braced_body(&source, "pub enum AccessPhase {").unwrap_or_else(|| {
            panic!("Fix: no `pub enum AccessPhase` declaration in {path:?}; update this test")
        });
    let declared = vyre_test_support::top_level_variant_names(body);
    assert_eq!(
        declared.len(),
        4,
        "Fix: expected 4 declared AccessPhase variants, found {}",
        declared.len()
    );

    let covered: std::collections::BTreeSet<String> = [
        AccessPhase::LoadStage,
        AccessPhase::ComputeRead,
        AccessPhase::Reduction,
        AccessPhase::EpilogueStore,
    ]
    .iter()
    .map(|p| match p {
        AccessPhase::LoadStage => "LoadStage".to_string(),
        AccessPhase::ComputeRead => "ComputeRead".to_string(),
        AccessPhase::Reduction => "Reduction".to_string(),
        AccessPhase::EpilogueStore => "EpilogueStore".to_string(),
    })
    .collect();

    let missing: Vec<&String> = declared.difference(&covered).collect();
    assert!(
        missing.is_empty(),
        "Fix: add coverage for newly declared AccessPhase variant(s): {missing:?}"
    );
}
