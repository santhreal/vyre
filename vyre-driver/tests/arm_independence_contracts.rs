//! Contracts for `vyre_driver::arm_independence`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::arm_independence::{
    can_dispatch_concurrently, ArmBindingSummary, ArmConflict, ArmIndependenceVerdict,
};

fn summary(reads: &[u32], writes: &[u32]) -> ArmBindingSummary {
    ArmBindingSummary {
        reads: reads.iter().copied().collect(),
        writes: writes.iter().copied().collect(),
    }
}

#[test]
fn fully_disjoint_arms_are_independent() {
    let a = summary(&[0, 1], &[2]);
    let b = summary(&[3, 4], &[5]);
    assert_eq!(
        can_dispatch_concurrently(&a, &b),
        ArmIndependenceVerdict::Independent
    );
}

#[test]
fn empty_arms_are_independent() {
    let a = summary(&[], &[]);
    let b = summary(&[], &[]);
    assert_eq!(
        can_dispatch_concurrently(&a, &b),
        ArmIndependenceVerdict::Independent
    );
}

#[test]
fn shared_read_only_slot_is_independent() {
    let a = summary(&[7], &[1]);
    let b = summary(&[7], &[2]);
    // Both READ slot 7; neither writes it  -  no race.
    assert_eq!(
        can_dispatch_concurrently(&a, &b),
        ArmIndependenceVerdict::Independent
    );
}

#[test]
fn write_write_conflict_serialises() {
    let a = summary(&[], &[3]);
    let b = summary(&[], &[3]);
    assert_eq!(
        can_dispatch_concurrently(&a, &b),
        ArmIndependenceVerdict::SerializeRequired {
            reason: ArmConflict::WriteWriteConflict,
        }
    );
}

#[test]
fn read_after_write_serialises() {
    let a = summary(&[0], &[5]);
    let b = summary(&[5], &[1]);
    assert_eq!(
        can_dispatch_concurrently(&a, &b),
        ArmIndependenceVerdict::SerializeRequired {
            reason: ArmConflict::ReadAfterWrite,
        }
    );
}

#[test]
fn write_after_read_serialises() {
    let a = summary(&[5], &[1]);
    let b = summary(&[0], &[5]);
    assert_eq!(
        can_dispatch_concurrently(&a, &b),
        ArmIndependenceVerdict::SerializeRequired {
            reason: ArmConflict::WriteAfterRead,
        }
    );
}

#[test]
fn write_write_takes_precedence_over_other_conflicts() {
    // Both write slot 3 AND a writes 1 / b reads 1. Verdict
    // names the strongest conflict (write-write).
    let a = summary(&[], &[1, 3]);
    let b = summary(&[1], &[3]);
    assert_eq!(
        can_dispatch_concurrently(&a, &b),
        ArmIndependenceVerdict::SerializeRequired {
            reason: ArmConflict::WriteWriteConflict,
        }
    );
}

#[test]
fn verdict_is_symmetric_for_writes_and_reads() {
    let a = summary(&[], &[10]);
    let b = summary(&[], &[10]);
    // ww conflict reported the same regardless of arg order.
    let verdict_ab = can_dispatch_concurrently(&a, &b);
    let verdict_ba = can_dispatch_concurrently(&b, &a);
    assert_eq!(verdict_ab, verdict_ba);
}

#[test]
fn one_empty_arm_leaves_independent_when_other_alone() {
    let a = summary(&[1, 2, 3], &[4]);
    let b = ArmBindingSummary::new();
    assert_eq!(
        can_dispatch_concurrently(&a, &b),
        ArmIndependenceVerdict::Independent
    );
    assert_eq!(
        can_dispatch_concurrently(&b, &a),
        ArmIndependenceVerdict::Independent
    );
}
