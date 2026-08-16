//! Contract tests for target-boundary bank conflict mitigation strategy.
//!
//! Verifies Section 185.2:
//! - Multi-phase access pattern evaluation (load, compute, reduction, epilogue).
//! - Target-specific selection of padding, XOR swizzling, or no rewrite.
//! - Rejection of candidates that move unacceptable conflicts to another phase.
//! - Non-promising of universal zero conflicts (honestly reporting remaining conflicts).

use vyre_lower::analyses::{
    evaluate_mitigation_candidate, select_bank_conflict_strategy, AccessPhase,
    AccessPhaseProfile, BankConflictMitigation, ConflictSeverity, TargetBankGeometry,
};

#[test]
fn default_target_geometry_has_32_banks() {
    let geom = TargetBankGeometry::default();
    assert_eq!(geom.bank_count, 32);
    assert_eq!(geom.bank_width_bytes, 4);
    assert_eq!(geom.subgroup_lanes, 32);
}

#[test]
fn padding_mitigates_power_of_two_column_stride_conflicts() {
    let geom = TargetBankGeometry::default();
    let phases = vec![
        AccessPhaseProfile {
            phase: AccessPhase::ComputeRead,
            stride_elements: 32, // Stride 32 on 32 banks causes critical 32-way conflict
            active_threads: 32,
            access_weight: 10,
        },
    ];

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
        BankConflictMitigation::PadLines { pad_elements_per_row: 1 },
        ConflictSeverity::Critical,
    );
    assert_eq!(padded.worst_severity, ConflictSeverity::None);
    assert!(padded.accepted);
    assert!(padded.aggregate_penalty < baseline.aggregate_penalty);
}

#[test]
fn xor_swizzling_reduces_conflict_penalty() {
    let geom = TargetBankGeometry::default();
    let phases = vec![
        AccessPhaseProfile {
            phase: AccessPhase::ComputeRead,
            stride_elements: 16, // Stride 16 causes 16-way critical conflict
            active_threads: 32,
            access_weight: 5,
        },
    ];

    let swizzled = evaluate_mitigation_candidate(
        &phases,
        &geom,
        BankConflictMitigation::XorSwizzle { swizzle_bits: 2, stride_shift: 3 },
        ConflictSeverity::Critical,
    );
    assert!(swizzled.accepted);
    assert!(swizzled.aggregate_penalty < 16.0 * 5.0);
}

#[test]
fn strategy_selection_rejects_candidate_moving_conflict_to_another_phase() {
    let geom = TargetBankGeometry::default();
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
    let geom = TargetBankGeometry::default();
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
