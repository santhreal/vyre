//! The legacy combined memory ordering model and the closed atomic model
//! encode into one wire schema and describe one lattice.
//!
//! WHY: both types carried their own tag table and their own join table for
//! the five orderings they share. Nothing tied the copies together, so a tag
//! reassigned in one model and not the other would have produced a value that
//! encodes under one type and decodes as a different ordering under the other,
//! with every round-trip test in the crate still passing. The variant space is
//! derived from `AtomicOrdering::ALL` and `MemoryOrdering::ALL` at run time, so
//! a new ordering added to either model is covered without editing this file.
//!
//! Not covered here: whether a tag value is the right one for the wire schema.
//! That is what the round-trip suites assert.

use vyre_foundation::ir::{AtomicOrdering, MemoryOrdering};

/// Every ordering both models name carries the same wire tag under each.
#[test]
fn a_shared_ordering_has_one_wire_tag() {
    for atomic in AtomicOrdering::ALL {
        let combined = MemoryOrdering::from_atomic_ordering(atomic);
        assert_eq!(
            combined.wire_tag(),
            atomic.wire_tag(),
            "{atomic:?} encodes differently under the two models"
        );
        assert_eq!(
            MemoryOrdering::from_wire_tag(atomic.wire_tag()),
            Ok(combined),
            "tag {} decodes to a different ordering",
            atomic.wire_tag()
        );
    }
}

/// The grid-scope barrier is the only ordering the combined model adds, and
/// its tag collides with no atomic ordering.
#[test]
fn the_grid_barrier_tag_is_outside_the_atomic_set() {
    let grid_tag = MemoryOrdering::GridSync.wire_tag();
    assert!(
        AtomicOrdering::from_wire_tag(grid_tag).is_err(),
        "tag {grid_tag} is assigned to an atomic ordering as well"
    );
    for atomic in AtomicOrdering::ALL {
        assert_ne!(atomic.wire_tag(), grid_tag);
    }
    assert_eq!(
        MemoryOrdering::ALL.len(),
        AtomicOrdering::ALL.len() + 1,
        "the combined model adds exactly the grid barrier"
    );
}

/// Joining two shared orderings gives the same answer under both models.
#[test]
fn a_shared_join_has_one_answer() {
    for left in AtomicOrdering::ALL {
        for right in AtomicOrdering::ALL {
            assert_eq!(
                MemoryOrdering::from_atomic_ordering(left)
                    .join(MemoryOrdering::from_atomic_ordering(right)),
                MemoryOrdering::from_atomic_ordering(left.join(right)),
                "{left:?} joined with {right:?} differs between the two models"
            );
        }
    }
}

/// Grid synchronization absorbs every other ordering from either side.
#[test]
fn the_grid_barrier_absorbs_every_join() {
    for ordering in MemoryOrdering::ALL {
        assert_eq!(
            ordering.join(MemoryOrdering::GridSync),
            MemoryOrdering::GridSync
        );
        assert_eq!(
            MemoryOrdering::GridSync.join(ordering),
            MemoryOrdering::GridSync
        );
    }
}

/// A tag past both models is rejected by both, with an actionable message.
#[test]
fn an_unassigned_tag_is_rejected_by_both_models() {
    let past_both = u8::try_from(MemoryOrdering::ALL.len()).expect("the model is small");
    for tag in [past_both, u8::MAX] {
        let combined = MemoryOrdering::from_wire_tag(tag).expect_err("tag is unassigned");
        assert!(combined.contains("InvalidDiscriminant"), "{combined}");
        assert!(combined.contains(&tag.to_string()), "{combined}");
        assert!(AtomicOrdering::from_wire_tag(tag).is_err());
    }
}
