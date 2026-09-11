//! Contracts for `vyre_driver::async_copy_overlap`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::arm_independence::ArmBindingSummary;
use vyre_driver::async_copy_overlap::{can_overlap_copy_with_kernel, CopyOverlapDecision};

fn arm(reads: &[u32], writes: &[u32]) -> ArmBindingSummary {
    ArmBindingSummary {
        reads: reads.iter().copied().collect(),
        writes: writes.iter().copied().collect(),
    }
}

#[test]
fn copy_to_unread_slot_overlaps() {
    let kernel = arm(&[0, 1], &[2]);
    assert_eq!(
        can_overlap_copy_with_kernel(7, &kernel),
        CopyOverlapDecision::Overlap
    );
}

#[test]
fn copy_to_kernel_read_slot_serialises() {
    let kernel = arm(&[0, 1], &[2]);
    assert_eq!(
        can_overlap_copy_with_kernel(0, &kernel),
        CopyOverlapDecision::Serialize
    );
}

#[test]
fn copy_to_kernel_write_slot_serialises() {
    // Defensive: copying onto kernel's output buffer is suspect,
    // but if the runtime plans it the substrate must say
    // Serialize so the kernel sees the copied bytes.
    let kernel = arm(&[0], &[5]);
    assert_eq!(
        can_overlap_copy_with_kernel(5, &kernel),
        CopyOverlapDecision::Serialize
    );
}

#[test]
fn copy_with_empty_kernel_overlaps() {
    let kernel = arm(&[], &[]);
    assert_eq!(
        can_overlap_copy_with_kernel(0, &kernel),
        CopyOverlapDecision::Overlap
    );
}
