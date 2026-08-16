//! Contract tests for optimization legality verification and resource bounds.
//!
//! Verifies Section 185.4:
//! - Partial tiles, tails, alignment, aliasing, divergent barriers, dynamic shapes.
//! - Shared-memory capacity limits and register-pressure spill bounds.
//! - Retention of unswizzled, synchronous, and unfused fallback candidates.

use vyre_lower::analyses::resource_bounds::{
    verify_candidate_legality, LegalityCheck, TailHandling, TargetResourceLimits,
};
use vyre_lower::{BindingLayout, KernelBody, KernelDescriptor};

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

#[test]
fn legal_candidate_passes_all_checks_and_retains_fallbacks() {
    let desc = mock_descriptor();
    let limits = TargetResourceLimits::default();

    let legality = LegalityCheck {
        partial_tiles_guarded: true,
        tail_handling: TailHandling::default(),
        alignment_bytes: 16,
        aliasing_free: true,
        no_divergent_barriers: true,
        dynamic_shapes_supported: true,
        instructions_supported: true,
    };

    let report = verify_candidate_legality(
        &desc,
        "swizzled_tiled_matmul",
        legality,
        16 * 1024, // 16 KB shared memory
        64,        // 64 registers per thread
        &limits,
    );

    assert!(report.is_legal);
    assert!(report.resource_bounds.is_within_limits);
    assert_eq!(report.resource_bounds.spill_bytes, 0);
    assert!(report.violations.is_empty());

    // Retained fallbacks are verified
    assert!(report.retained_fallbacks.unswizzled_candidate_retained);
    assert!(report.retained_fallbacks.synchronous_candidate_retained);
    assert!(report.retained_fallbacks.unfused_candidate_retained);
}

#[test]
fn unguarded_partial_tiles_are_flagged_illegal() {
    let desc = mock_descriptor();
    let limits = TargetResourceLimits::default();

    let legality = LegalityCheck {
        partial_tiles_guarded: false, // Violation
        tail_handling: TailHandling {
            dynamic_predication: false,
            padded_allocation: false,
        },
        alignment_bytes: 16,
        aliasing_free: true,
        no_divergent_barriers: true,
        dynamic_shapes_supported: true,
        instructions_supported: true,
    };

    let report = verify_candidate_legality(&desc, "unguarded_tail", legality, 8 * 1024, 32, &limits);

    assert!(!report.is_legal);
    assert!(report.violations.iter().any(|v| v.contains("partial tiles lack dynamic bounds")));
}

#[test]
fn shared_memory_limit_and_spill_bounds_are_enforced() {
    let desc = mock_descriptor();
    let limits = TargetResourceLimits {
        max_shared_memory_bytes: 32 * 1024, // 32 KB limit
        max_registers_per_thread: 64,
        max_threads_per_workgroup: 256,
        required_vector_alignment_bytes: 16,
        supports_dynamic_shapes: true,
    };

    let legality = LegalityCheck {
        partial_tiles_guarded: true,
        tail_handling: TailHandling::default(),
        alignment_bytes: 16,
        aliasing_free: true,
        no_divergent_barriers: true,
        dynamic_shapes_supported: true,
        instructions_supported: true,
    };

    // Shared memory overflow (48 KB > 32 KB)
    let smem_overflow_report = verify_candidate_legality(
        &desc,
        "large_tile",
        legality.clone(),
        48 * 1024,
        64,
        &limits,
    );
    assert!(!smem_overflow_report.is_legal);
    assert!(!smem_overflow_report.resource_bounds.is_within_limits);
    assert!(smem_overflow_report.violations.iter().any(|v| v.contains("shared memory 49152 bytes exceeds")));

    // Register pressure overflow (80 regs > 64 regs limit) inducing spill
    let reg_overflow_report = verify_candidate_legality(
        &desc,
        "deep_unroll",
        legality,
        16 * 1024,
        80,
        &limits,
    );
    assert!(!reg_overflow_report.is_legal);
    assert!(reg_overflow_report.resource_bounds.spill_bytes > 0);
    assert!(reg_overflow_report.violations.iter().any(|v| v.contains("local memory spill")));
}

#[test]
fn divergent_barrier_and_unsupported_instruction_violations() {
    let desc = mock_descriptor();
    let limits = TargetResourceLimits::default();

    let legality = LegalityCheck {
        partial_tiles_guarded: true,
        tail_handling: TailHandling::default(),
        alignment_bytes: 16,
        aliasing_free: true,
        no_divergent_barriers: false, // Divergent barrier violation
        dynamic_shapes_supported: true,
        instructions_supported: false, // Unsupported instruction violation
    };

    let report = verify_candidate_legality(&desc, "divergent_candidate", legality, 8 * 1024, 32, &limits);

    assert!(!report.is_legal);
    assert!(report.violations.iter().any(|v| v.contains("divergent control flow")));
    assert!(report.violations.iter().any(|v| v.contains("unsupported on target")));
}
