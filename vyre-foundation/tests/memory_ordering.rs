//! Tests for `MemoryOrdering` wire-tag round-trip and validity predicates.
//!
//! Memory ordering is part of the atomic and barrier contracts; a
//! mis-mapped tag silently changes synchronization semantics.

use vyre_foundation::ir::MemoryOrdering;

#[test]
fn relaxed_wire_tag_roundtrips() {
    let tag = MemoryOrdering::Relaxed.wire_tag();
    assert_eq!(
        MemoryOrdering::from_wire_tag(tag).unwrap(),
        MemoryOrdering::Relaxed
    );
}

#[test]
fn acquire_wire_tag_roundtrips() {
    let tag = MemoryOrdering::Acquire.wire_tag();
    assert_eq!(
        MemoryOrdering::from_wire_tag(tag).unwrap(),
        MemoryOrdering::Acquire
    );
}

#[test]
fn release_wire_tag_roundtrips() {
    let tag = MemoryOrdering::Release.wire_tag();
    assert_eq!(
        MemoryOrdering::from_wire_tag(tag).unwrap(),
        MemoryOrdering::Release
    );
}

#[test]
fn acq_rel_wire_tag_roundtrips() {
    let tag = MemoryOrdering::AcqRel.wire_tag();
    assert_eq!(
        MemoryOrdering::from_wire_tag(tag).unwrap(),
        MemoryOrdering::AcqRel
    );
}

#[test]
fn seq_cst_wire_tag_roundtrips() {
    let tag = MemoryOrdering::SeqCst.wire_tag();
    assert_eq!(
        MemoryOrdering::from_wire_tag(tag).unwrap(),
        MemoryOrdering::SeqCst
    );
}

#[test]
fn grid_sync_wire_tag_roundtrips() {
    let tag = MemoryOrdering::GridSync.wire_tag();
    assert_eq!(tag, 5);
    assert_eq!(
        MemoryOrdering::from_wire_tag(tag).unwrap(),
        MemoryOrdering::GridSync
    );
}

#[test]
fn from_wire_tag_rejects_unknown() {
    let err = MemoryOrdering::from_wire_tag(255).unwrap_err();
    assert!(err.contains("Fix:"));
}

#[test]
fn all_tags_are_unique() {
    let orderings = [
        MemoryOrdering::Relaxed,
        MemoryOrdering::Acquire,
        MemoryOrdering::Release,
        MemoryOrdering::AcqRel,
        MemoryOrdering::SeqCst,
        MemoryOrdering::GridSync,
    ];
    let mut tags: Vec<u8> = orderings.iter().map(|o| o.wire_tag()).collect();
    tags.sort_unstable();
    tags.dedup();
    assert_eq!(
        tags.len(),
        orderings.len(),
        "every MemoryOrdering must have a unique wire tag"
    );
}

#[test]
fn relaxed_valid_for_atomic_rmw() {
    assert!(MemoryOrdering::Relaxed.is_valid_for_atomic_rmw());
}

#[test]
fn acquire_valid_for_atomic_rmw() {
    assert!(MemoryOrdering::Acquire.is_valid_for_atomic_rmw());
}

#[test]
fn release_valid_for_atomic_rmw() {
    assert!(MemoryOrdering::Release.is_valid_for_atomic_rmw());
}

#[test]
fn acq_rel_valid_for_atomic_rmw() {
    assert!(MemoryOrdering::AcqRel.is_valid_for_atomic_rmw());
}

#[test]
fn seq_cst_valid_for_atomic_rmw() {
    assert!(MemoryOrdering::SeqCst.is_valid_for_atomic_rmw());
}

#[test]
fn grid_sync_not_valid_for_atomic_rmw() {
    assert!(!MemoryOrdering::GridSync.is_valid_for_atomic_rmw());
}

#[test]
fn relaxed_not_valid_for_barrier() {
    assert!(!MemoryOrdering::Relaxed.is_valid_for_barrier());
}

#[test]
fn acquire_valid_for_barrier() {
    assert!(MemoryOrdering::Acquire.is_valid_for_barrier());
}

#[test]
fn release_valid_for_barrier() {
    assert!(MemoryOrdering::Release.is_valid_for_barrier());
}

#[test]
fn acq_rel_valid_for_barrier() {
    assert!(MemoryOrdering::AcqRel.is_valid_for_barrier());
}

#[test]
fn seq_cst_valid_for_barrier() {
    assert!(MemoryOrdering::SeqCst.is_valid_for_barrier());
}

#[test]
fn grid_sync_valid_for_barrier() {
    assert!(MemoryOrdering::GridSync.is_valid_for_barrier());
}

#[test]
fn only_grid_sync_requires_grid_sync() {
    assert!(!MemoryOrdering::Relaxed.requires_grid_sync());
    assert!(!MemoryOrdering::Acquire.requires_grid_sync());
    assert!(!MemoryOrdering::Release.requires_grid_sync());
    assert!(!MemoryOrdering::AcqRel.requires_grid_sync());
    assert!(!MemoryOrdering::SeqCst.requires_grid_sync());
    assert!(MemoryOrdering::GridSync.requires_grid_sync());
}

#[test]
fn default_memory_ordering_is_seq_cst() {
    assert_eq!(MemoryOrdering::default(), MemoryOrdering::SeqCst);
}

#[test]
fn memory_ordering_join_contracts() {
    let all = [
        MemoryOrdering::Relaxed,
        MemoryOrdering::Acquire,
        MemoryOrdering::Release,
        MemoryOrdering::AcqRel,
        MemoryOrdering::SeqCst,
        MemoryOrdering::GridSync,
    ];

    // Identity element: join with Relaxed returns self.
    for o in all {
        assert_eq!(o.join(MemoryOrdering::Relaxed), o);
        assert_eq!(MemoryOrdering::Relaxed.join(o), o);
        // Idempotence: join with self returns self.
        assert_eq!(o.join(o), o);
        // Top element: join with GridSync returns GridSync.
        assert_eq!(o.join(MemoryOrdering::GridSync), MemoryOrdering::GridSync);
        assert_eq!(MemoryOrdering::GridSync.join(o), MemoryOrdering::GridSync);
    }

    // Commutativity across all pairs.
    for a in all {
        for b in all {
            assert_eq!(a.join(b), b.join(a));
        }
    }

    // Acquire and Release join to AcqRel.
    assert_eq!(
        MemoryOrdering::Acquire.join(MemoryOrdering::Release),
        MemoryOrdering::AcqRel
    );
    assert_eq!(
        MemoryOrdering::Release.join(MemoryOrdering::Acquire),
        MemoryOrdering::AcqRel
    );

    // SeqCst dominates everything below GridSync.
    assert_eq!(
        MemoryOrdering::Acquire.join(MemoryOrdering::SeqCst),
        MemoryOrdering::SeqCst
    );
    assert_eq!(
        MemoryOrdering::Release.join(MemoryOrdering::SeqCst),
        MemoryOrdering::SeqCst
    );
    assert_eq!(
        MemoryOrdering::AcqRel.join(MemoryOrdering::SeqCst),
        MemoryOrdering::SeqCst
    );
}
