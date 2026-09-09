//! Substrate-neutral memory model contracts, closed orthogonal concurrency types,
//! and explicit obligation tracking.
//!
//! Concurrency and memory semantics are modeled using separate, closed, orthogonal types
//! for atomic ordering, memory scope, execution scope, storage domain, fence semantics,
//! barrier participation, asynchronous transaction lifecycle, collective communication groups,
//! and failure/cancellation behavior.
//!
//! There are no default orderings or scopes. Semantic IR carries ownership, borrow, alias,
//! capability, state-epoch, and effect tokens; every load, store, atomic, collective, and
//! state transition consumes and produces explicit obligations.

pub(crate) mod async_tx;
pub(crate) mod atomic;
pub(crate) mod collective;
pub(crate) mod failure;
pub(crate) mod fence;
pub(crate) mod legacy;
pub(crate) mod obligations;
pub(crate) mod scope;
pub(crate) mod storage;

pub use async_tx::{exhaustiveness_check_async_transaction_lifecycle, AsyncTransactionLifecycle};
pub use atomic::{exhaustiveness_check_atomic_ordering, AtomicOrdering};
pub use collective::{exhaustiveness_check_collective_group, CollectiveGroup};
pub use failure::{
    exhaustiveness_check_failure_cancellation_behavior, FailureCancellationBehavior,
};
pub use fence::{
    exhaustiveness_check_barrier_participation, exhaustiveness_check_fence_semantics,
    BarrierParticipation, FenceSemantics,
};
pub use legacy::MemoryOrdering;
pub use obligations::{
    verify_program_obligations, AliasDiscipline, AliasToken, BorrowKind, BorrowToken,
    CapabilityToken, EffectKind, EffectToken, MemoryCapability, Obligation, ObligationError,
    ObligationKind, ObligationTracker, OwnershipKind, OwnershipToken, StateEpoch,
};
pub use scope::{
    exhaustiveness_check_execution_scope, exhaustiveness_check_memory_scope, ExecutionScope,
    MemoryScope,
};
pub use storage::{exhaustiveness_check_storage_domain, StorageDomain};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_memory_ordering_variants_are_distinct() {
        let variants = [
            MemoryOrdering::Relaxed,
            MemoryOrdering::Acquire,
            MemoryOrdering::Release,
            MemoryOrdering::AcqRel,
            MemoryOrdering::SeqCst,
            MemoryOrdering::GridSync,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn wire_tags_round_trip() {
        for ordering in [
            MemoryOrdering::Relaxed,
            MemoryOrdering::Acquire,
            MemoryOrdering::Release,
            MemoryOrdering::AcqRel,
            MemoryOrdering::SeqCst,
            MemoryOrdering::GridSync,
        ] {
            let tag = ordering.wire_tag();
            let decoded = MemoryOrdering::from_wire_tag(tag).unwrap();
            assert_eq!(ordering, decoded);
        }
    }

    #[test]
    fn invalid_wire_tag_rejected() {
        assert!(MemoryOrdering::from_wire_tag(6).is_err());
        assert!(MemoryOrdering::from_wire_tag(255).is_err());
    }

    #[test]
    fn atomic_ordering_wire_tags_round_trip() {
        for ordering in AtomicOrdering::ALL {
            let tag = ordering.wire_tag();
            let decoded = AtomicOrdering::from_wire_tag(tag).unwrap();
            assert_eq!(ordering, decoded);
        }
        assert!(AtomicOrdering::from_wire_tag(5).is_err());
    }

    #[test]
    fn memory_scope_wire_tags_round_trip() {
        for scope in MemoryScope::ALL {
            let tag = scope.wire_tag();
            let decoded = MemoryScope::from_wire_tag(tag).unwrap();
            assert_eq!(scope, decoded);
        }
        assert!(MemoryScope::from_wire_tag(6).is_err());
    }

    #[test]
    fn execution_scope_wire_tags_round_trip() {
        for scope in ExecutionScope::ALL {
            let tag = scope.wire_tag();
            let decoded = ExecutionScope::from_wire_tag(tag).unwrap();
            assert_eq!(scope, decoded);
        }
        assert!(ExecutionScope::from_wire_tag(6).is_err());
    }

    #[test]
    fn storage_domain_wire_tags_round_trip() {
        for domain in StorageDomain::ALL {
            let tag = domain.wire_tag();
            let decoded = StorageDomain::from_wire_tag(tag).unwrap();
            assert_eq!(domain, decoded);
        }
        assert!(StorageDomain::from_wire_tag(8).is_err());
    }

    #[test]
    fn fence_semantics_wire_tags_round_trip() {
        for fence in FenceSemantics::ALL {
            let tag = fence.wire_tag();
            let decoded = FenceSemantics::from_wire_tag(tag).unwrap();
            assert_eq!(fence, decoded);
        }
        assert!(FenceSemantics::from_wire_tag(4).is_err());
    }

    #[test]
    fn barrier_participation_wire_tags_round_trip() {
        for part in BarrierParticipation::ALL {
            let tag = part.wire_tag();
            let decoded = BarrierParticipation::from_wire_tag(tag).unwrap();
            assert_eq!(part, decoded);
        }
        assert!(BarrierParticipation::from_wire_tag(6).is_err());
    }

    #[test]
    fn async_transaction_lifecycle_wire_tags_round_trip() {
        for lifecycle in AsyncTransactionLifecycle::ALL {
            let tag = lifecycle.wire_tag();
            let decoded = AsyncTransactionLifecycle::from_wire_tag(tag).unwrap();
            assert_eq!(lifecycle, decoded);
        }
        assert!(AsyncTransactionLifecycle::from_wire_tag(6).is_err());
    }

    #[test]
    fn collective_group_wire_tags_round_trip() {
        for group in CollectiveGroup::ALL {
            let tag = group.wire_tag();
            let decoded = CollectiveGroup::from_wire_tag(tag).unwrap();
            assert_eq!(group, decoded);
        }
        assert!(CollectiveGroup::from_wire_tag(6).is_err());
    }

    #[test]
    fn failure_cancellation_behavior_wire_tags_round_trip() {
        for behavior in FailureCancellationBehavior::ALL {
            let tag = behavior.wire_tag();
            let decoded = FailureCancellationBehavior::from_wire_tag(tag).unwrap();
            assert_eq!(behavior, decoded);
        }
        assert!(FailureCancellationBehavior::from_wire_tag(5).is_err());
    }

    #[test]
    fn obligation_tracker_catches_unconsumed_async_wait() {
        let mut tracker = ObligationTracker::new();
        tracker.produce(
            "async_wait:tag_0",
            ObligationKind::PendingAsyncWait {
                tag: "tag_0".to_string(),
                source: "src".to_string(),
                destination: "dst".to_string(),
            },
            0,
        );
        let result = tracker.check_all_consumed();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ObligationError::UnconsumedObligation { .. }));
        assert!(err.to_string().contains("async_wait:tag_0"));
    }

    #[test]
    fn obligation_tracker_passes_when_consumed() {
        let mut tracker = ObligationTracker::new();
        let id = tracker.produce(
            "async_wait:tag_0",
            ObligationKind::PendingAsyncWait {
                tag: "tag_0".to_string(),
                source: "src".to_string(),
                destination: "dst".to_string(),
            },
            0,
        );
        assert!(tracker.consume(id, 1).is_ok());
        assert!(tracker.check_all_consumed().is_ok());
    }
}
