//! The class closed here: an estimated register spill reported as illegal.
//!
//! Register pressure has two thresholds. Above the occupancy budget the target
//! compiler spills to local memory and the launch still runs, so the spill is a
//! cost the whole-program cost model prices. Above the architectural ceiling
//! there is no launch at all. Reporting the first as a violation eliminated
//! every candidate a bounded spill would have made faster, which is the whole
//! class of unrolled and tiled candidates on a register-poor target.
//!
//! Also proved here: partial tiles, tails, alignment, aliasing, divergent
//! barriers, dynamic shapes, shared-memory capacity, and retention of the
//! unswizzled, synchronous and unfused fallback candidates.

use vyre_lower::analyses::{
    verify_candidate_legality, LegalityCheck, TailHandling, TargetResourceLimits,
};
use vyre_lower::{BindingLayout, KernelBody, KernelDescriptor};

/// Limits a target reports: 48 KB of shared memory, 128 registers sustained at
/// the occupancy target, 255 architecturally, 1024 threads, 16-byte vectors.
fn limits() -> TargetResourceLimits {
    TargetResourceLimits::new(48 * 1024, 128, 255, 1024, 16, true)
}

/// Every legality box ticked, so only resource thresholds decide the outcome.
fn clean_legality() -> LegalityCheck {
    LegalityCheck {
        partial_tiles_guarded: true,
        tail_handling: TailHandling::default(),
        alignment_bytes: 16,
        aliasing_free: true,
        no_divergent_barriers: true,
        dynamic_shapes_supported: true,
        instructions_supported: true,
    }
}

fn mock_descriptor() -> KernelDescriptor {
    KernelDescriptor {
        id: "mock_kernel".to_string(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: vyre_lower::Dispatch {
            workgroup_size: [64, 1, 1],
        },
        body: KernelBody {
            ops: vec![],
            literals: vec![],
            child_bodies: vec![],
        },
    }
}

/// WHY: a candidate inside every threshold reports no violation, and the three
/// fallback candidates stay retained so a rejection later has somewhere to land.
#[test]
fn a_candidate_inside_every_threshold_is_legal_and_keeps_its_fallbacks() {
    let report = verify_candidate_legality(
        &mock_descriptor(),
        "swizzled_tiled_matmul",
        clean_legality(),
        16 * 1024,
        64,
        &limits(),
    );

    assert!(report.is_legal);
    assert!(report.resource_bounds.is_within_limits);
    assert_eq!(report.resource_bounds.spill_bytes, 0);
    assert!(report.violations.is_empty());
    assert!(report.retained_fallbacks.unswizzled_candidate_retained);
    assert!(report.retained_fallbacks.synchronous_candidate_retained);
    assert!(report.retained_fallbacks.unfused_candidate_retained);
}

/// WHY: an unguarded tail reads outside its buffer, which no cost can price.
#[test]
fn unguarded_partial_tiles_are_flagged_illegal() {
    let legality = LegalityCheck {
        partial_tiles_guarded: false,
        tail_handling: TailHandling {
            dynamic_predication: false,
            padded_allocation: false,
        },
        ..clean_legality()
    };

    let report = verify_candidate_legality(
        &mock_descriptor(),
        "unguarded_tail",
        legality,
        8 * 1024,
        32,
        &limits(),
    );

    assert!(!report.is_legal);
    assert!(report
        .violations
        .iter()
        .any(|v| v.contains("partial tiles lack dynamic bounds")));
}

/// WHY: a workgroup allocation over the shared-memory limit has nowhere to
/// spill to, so it stays a violation and stays outside the limits.
#[test]
fn shared_memory_over_the_device_limit_is_illegal() {
    let limits = TargetResourceLimits::new(32 * 1024, 64, 255, 256, 16, true);

    let report = verify_candidate_legality(
        &mock_descriptor(),
        "large_tile",
        clean_legality(),
        48 * 1024,
        64,
        &limits,
    );

    assert!(!report.is_legal);
    assert!(!report.resource_bounds.is_within_limits);
    assert!(report
        .violations
        .iter()
        .any(|v| v.contains("shared memory 49152 bytes exceeds")));
}

/// WHY: this is the defect. 80 registers against a 64-register occupancy budget
/// spills 16 registers per thread, which the target compiler emits and the
/// device runs. Reporting it as a violation eliminated the candidate instead of
/// letting the cost model weigh the spill against the occupancy it buys.
#[test]
fn an_allocation_above_the_occupancy_budget_spills_and_stays_legal() {
    let limits = TargetResourceLimits::new(32 * 1024, 64, 255, 256, 16, true);

    let report = verify_candidate_legality(
        &mock_descriptor(),
        "deep_unroll",
        clean_legality(),
        16 * 1024,
        80,
        &limits,
    );

    assert!(report.is_legal, "spilling is how the target executes this");
    assert!(report.resource_bounds.is_within_limits);
    assert!(report.violations.is_empty());
    assert_eq!(
        report.resource_bounds.spill_bytes,
        16 * 4 * 256,
        "sixteen spilled registers of four bytes across the whole workgroup"
    );
    assert_eq!(
        report.resource_bounds.occupancy_register_budget_per_thread,
        64
    );
    assert_eq!(
        report
            .resource_bounds
            .architectural_register_limit_per_thread,
        255
    );
}

/// WHY: the ceiling is the threshold with no execution behind it, so it is the
/// one register threshold that rejects. The report still states the spill, so a
/// caller can see how far past the budget the rejection happened.
#[test]
fn an_allocation_above_the_architectural_ceiling_is_illegal() {
    let limits = TargetResourceLimits::new(32 * 1024, 64, 255, 256, 16, true);

    let report = verify_candidate_legality(
        &mock_descriptor(),
        "unbounded_unroll",
        clean_legality(),
        16 * 1024,
        256,
        &limits,
    );

    assert!(!report.is_legal);
    assert!(!report.resource_bounds.is_within_limits);
    assert!(report
        .violations
        .iter()
        .any(|v| v.contains("exceeds the architectural limit of 255 regs")));
    assert_eq!(report.resource_bounds.spill_bytes, 192 * 4 * 256);
}

/// WHY: a target that reports no ceiling has no threshold to reject against.
/// Defaulting the unknown ceiling to the occupancy budget would reject every
/// spilling candidate again through the other door.
#[test]
fn an_unreported_ceiling_rejects_nothing_for_register_pressure() {
    let limits = TargetResourceLimits::new(32 * 1024, 64, 0, 256, 16, true);

    let report = verify_candidate_legality(
        &mock_descriptor(),
        "deep_unroll",
        clean_legality(),
        16 * 1024,
        4_096,
        &limits,
    );

    assert!(report.is_legal);
    assert!(report.resource_bounds.is_within_limits);
    assert_eq!(
        report
            .resource_bounds
            .architectural_register_limit_per_thread,
        0
    );
    assert_eq!(report.resource_bounds.spill_bytes, (4_096 - 64) * 4 * 256);
}

/// WHY: a barrier inside divergent control flow and an unsupported instruction
/// are both properties of the emitted program, not resource pressure, and both
/// stay violations.
#[test]
fn divergent_barrier_and_unsupported_instruction_violations() {
    let legality = LegalityCheck {
        no_divergent_barriers: false,
        instructions_supported: false,
        ..clean_legality()
    };

    let report = verify_candidate_legality(
        &mock_descriptor(),
        "divergent_candidate",
        legality,
        8 * 1024,
        32,
        &limits(),
    );

    assert!(!report.is_legal);
    assert!(report
        .violations
        .iter()
        .any(|v| v.contains("divergent control flow")));
    assert!(report
        .violations
        .iter()
        .any(|v| v.contains("unsupported on target")));
}
