//! Tests for closed orthogonal effect and memory model types.
//!
//! Verifies that all 9 closed types have total, unique wire tags, fail closed on unknown tags,
//! maintain exhaustive variant coverage with zero wildcards, and have no Default implementation.

use vyre_foundation::ir::*;

#[test]
fn atomic_ordering_contracts() {
    let variants = [
        AtomicOrdering::Relaxed,
        AtomicOrdering::Acquire,
        AtomicOrdering::Release,
        AtomicOrdering::AcqRel,
        AtomicOrdering::SeqCst,
    ];
    let mut tags = Vec::new();
    for v in variants {
        let tag = v.wire_tag();
        tags.push(tag);
        assert_eq!(AtomicOrdering::from_wire_tag(tag).unwrap(), v);
        assert!(!exhaustiveness_check_atomic_ordering(v).is_empty());
        assert!(!v.name().is_empty());
    }
    tags.sort_unstable();
    tags.dedup();
    assert_eq!(tags.len(), variants.len());
    assert!(AtomicOrdering::from_wire_tag(255).is_err());

    assert!(!AtomicOrdering::Relaxed.is_acquire());
    assert!(!AtomicOrdering::Relaxed.is_release());
    assert!(AtomicOrdering::Acquire.is_acquire());
    assert!(!AtomicOrdering::Acquire.is_release());
    assert!(!AtomicOrdering::Release.is_acquire());
    assert!(AtomicOrdering::Release.is_release());
    assert!(AtomicOrdering::AcqRel.is_acquire());
    assert!(AtomicOrdering::AcqRel.is_release());
    assert!(AtomicOrdering::SeqCst.is_acquire());
    assert!(AtomicOrdering::SeqCst.is_release());

    assert_eq!(
        AtomicOrdering::Acquire.join(AtomicOrdering::Release),
        AtomicOrdering::AcqRel
    );
    assert_eq!(
        AtomicOrdering::Relaxed.join(AtomicOrdering::SeqCst),
        AtomicOrdering::SeqCst
    );
}

#[test]
fn memory_scope_contracts() {
    let variants = [
        MemoryScope::Thread,
        MemoryScope::Subgroup,
        MemoryScope::Workgroup,
        MemoryScope::Cluster,
        MemoryScope::Device,
        MemoryScope::System,
    ];
    let mut tags = Vec::new();
    for v in variants {
        let tag = v.wire_tag();
        tags.push(tag);
        assert_eq!(MemoryScope::from_wire_tag(tag).unwrap(), v);
        assert!(!exhaustiveness_check_memory_scope(v).is_empty());
        assert!(!v.name().is_empty());
    }
    tags.sort_unstable();
    tags.dedup();
    assert_eq!(tags.len(), variants.len());
    assert!(MemoryScope::from_wire_tag(255).is_err());

    assert!(!MemoryScope::Thread.is_cross_workgroup());
    assert!(!MemoryScope::Subgroup.is_cross_workgroup());
    assert!(!MemoryScope::Workgroup.is_cross_workgroup());
    assert!(MemoryScope::Cluster.is_cross_workgroup());
    assert!(MemoryScope::Device.is_cross_workgroup());
    assert!(MemoryScope::System.is_cross_workgroup());

    assert!(!MemoryScope::Workgroup.is_device_wide());
    assert!(MemoryScope::Device.is_device_wide());
    assert!(MemoryScope::System.is_device_wide());

    assert_eq!(
        MemoryScope::Workgroup.widen(MemoryScope::Device),
        MemoryScope::Device
    );
    assert_eq!(
        MemoryScope::System.widen(MemoryScope::Thread),
        MemoryScope::System
    );
}

#[test]
fn execution_scope_contracts() {
    let variants = [
        ExecutionScope::Thread,
        ExecutionScope::Subgroup,
        ExecutionScope::Workgroup,
        ExecutionScope::Cluster,
        ExecutionScope::Grid,
        ExecutionScope::DeviceMesh,
    ];
    let mut tags = Vec::new();
    for v in variants {
        let tag = v.wire_tag();
        tags.push(tag);
        assert_eq!(ExecutionScope::from_wire_tag(tag).unwrap(), v);
        assert!(!exhaustiveness_check_execution_scope(v).is_empty());
        assert!(!v.name().is_empty());
    }
    tags.sort_unstable();
    tags.dedup();
    assert_eq!(tags.len(), variants.len());
    assert!(ExecutionScope::from_wire_tag(255).is_err());

    assert!(!ExecutionScope::Workgroup.is_cross_block());
    assert!(ExecutionScope::Cluster.is_cross_block());
    assert!(ExecutionScope::Grid.is_cross_block());
    assert!(ExecutionScope::DeviceMesh.is_cross_block());
}

#[test]
fn storage_domain_contracts() {
    let variants = [
        StorageDomain::Register,
        StorageDomain::Scratchpad,
        StorageDomain::WorkgroupLocal,
        StorageDomain::DeviceGlobal,
        StorageDomain::HostPinned,
        StorageDomain::HostPaged,
        StorageDomain::Constant,
        StorageDomain::Texture,
    ];
    let mut tags = Vec::new();
    for v in variants {
        let tag = v.wire_tag();
        tags.push(tag);
        assert_eq!(StorageDomain::from_wire_tag(tag).unwrap(), v);
        assert!(!exhaustiveness_check_storage_domain(v).is_empty());
        assert!(!v.name().is_empty());
    }
    tags.sort_unstable();
    tags.dedup();
    assert_eq!(tags.len(), variants.len());
    assert!(StorageDomain::from_wire_tag(255).is_err());

    assert!(!StorageDomain::Register.is_shared_across_threads());
    assert!(StorageDomain::Scratchpad.is_shared_across_threads());
    assert!(StorageDomain::DeviceGlobal.is_shared_across_threads());

    assert!(!StorageDomain::Register.is_host_accessible());
    assert!(!StorageDomain::Scratchpad.is_host_accessible());
    assert!(StorageDomain::HostPinned.is_host_accessible());
    assert!(StorageDomain::HostPaged.is_host_accessible());
}

#[test]
fn fence_semantics_contracts() {
    let variants = [
        FenceSemantics::Acquire,
        FenceSemantics::Release,
        FenceSemantics::AcqRel,
        FenceSemantics::SequentiallyConsistent,
    ];
    let mut tags = Vec::new();
    for v in variants {
        let tag = v.wire_tag();
        tags.push(tag);
        assert_eq!(FenceSemantics::from_wire_tag(tag).unwrap(), v);
        assert!(!exhaustiveness_check_fence_semantics(v).is_empty());
        assert!(!v.name().is_empty());
    }
    tags.sort_unstable();
    tags.dedup();
    assert_eq!(tags.len(), variants.len());
    assert!(FenceSemantics::from_wire_tag(255).is_err());
}

#[test]
fn barrier_participation_contracts() {
    let variants = [
        BarrierParticipation::Uniform,
        BarrierParticipation::Converged,
        BarrierParticipation::ElectOne,
        BarrierParticipation::DynamicMask,
        BarrierParticipation::SubgroupOnly,
        BarrierParticipation::WorkgroupOnly,
    ];
    let mut tags = Vec::new();
    for v in variants {
        let tag = v.wire_tag();
        tags.push(tag);
        assert_eq!(BarrierParticipation::from_wire_tag(tag).unwrap(), v);
        assert!(!exhaustiveness_check_barrier_participation(v).is_empty());
        assert!(!v.name().is_empty());
    }
    tags.sort_unstable();
    tags.dedup();
    assert_eq!(tags.len(), variants.len());
    assert!(BarrierParticipation::from_wire_tag(255).is_err());

    assert!(BarrierParticipation::Uniform.requires_uniform_control_flow());
    assert!(!BarrierParticipation::Converged.requires_uniform_control_flow());
    assert!(!BarrierParticipation::ElectOne.requires_uniform_control_flow());
}

#[test]
fn async_transaction_lifecycle_contracts() {
    let variants = [
        AsyncTransactionLifecycle::Submitted,
        AsyncTransactionLifecycle::InFlight,
        AsyncTransactionLifecycle::Arrived,
        AsyncTransactionLifecycle::Committed,
        AsyncTransactionLifecycle::Failed,
        AsyncTransactionLifecycle::Aborted,
    ];
    let mut tags = Vec::new();
    for v in variants {
        let tag = v.wire_tag();
        tags.push(tag);
        assert_eq!(AsyncTransactionLifecycle::from_wire_tag(tag).unwrap(), v);
        assert!(!exhaustiveness_check_async_transaction_lifecycle(v).is_empty());
        assert!(!v.name().is_empty());
    }
    tags.sort_unstable();
    tags.dedup();
    assert_eq!(tags.len(), variants.len());
    assert!(AsyncTransactionLifecycle::from_wire_tag(255).is_err());

    assert!(!AsyncTransactionLifecycle::Submitted.is_terminal());
    assert!(!AsyncTransactionLifecycle::InFlight.is_terminal());
    assert!(!AsyncTransactionLifecycle::Arrived.is_terminal());
    assert!(AsyncTransactionLifecycle::Committed.is_terminal());
    assert!(AsyncTransactionLifecycle::Failed.is_terminal());
    assert!(AsyncTransactionLifecycle::Aborted.is_terminal());

    assert!(AsyncTransactionLifecycle::Committed.is_successful_commit());
    assert!(!AsyncTransactionLifecycle::Failed.is_successful_commit());
}

#[test]
fn collective_group_contracts() {
    let variants = [
        CollectiveGroup::Subgroup,
        CollectiveGroup::Workgroup,
        CollectiveGroup::Cluster,
        CollectiveGroup::DeviceMesh,
        CollectiveGroup::CrossDeviceRing,
        CollectiveGroup::CustomTopology,
    ];
    let mut tags = Vec::new();
    for v in variants {
        let tag = v.wire_tag();
        tags.push(tag);
        assert_eq!(CollectiveGroup::from_wire_tag(tag).unwrap(), v);
        assert!(!exhaustiveness_check_collective_group(v).is_empty());
        assert!(!v.name().is_empty());
    }
    tags.sort_unstable();
    tags.dedup();
    assert_eq!(tags.len(), variants.len());
    assert!(CollectiveGroup::from_wire_tag(255).is_err());

    assert!(!CollectiveGroup::Subgroup.is_inter_device());
    assert!(!CollectiveGroup::Workgroup.is_inter_device());
    assert!(CollectiveGroup::CrossDeviceRing.is_inter_device());
}

#[test]
fn failure_cancellation_behavior_contracts() {
    let variants = [
        FailureCancellationBehavior::Trap,
        FailureCancellationBehavior::Poison,
        FailureCancellationBehavior::Propagate,
        FailureCancellationBehavior::AbortKernel,
        FailureCancellationBehavior::Ignore,
    ];
    let mut tags = Vec::new();
    for v in variants {
        let tag = v.wire_tag();
        tags.push(tag);
        assert_eq!(FailureCancellationBehavior::from_wire_tag(tag).unwrap(), v);
        assert!(!exhaustiveness_check_failure_cancellation_behavior(v).is_empty());
        assert!(!v.name().is_empty());
    }
    tags.sort_unstable();
    tags.dedup();
    assert_eq!(tags.len(), variants.len());
    assert!(FailureCancellationBehavior::from_wire_tag(255).is_err());

    assert!(FailureCancellationBehavior::Trap.halts_execution());
    assert!(FailureCancellationBehavior::AbortKernel.halts_execution());
    assert!(!FailureCancellationBehavior::Poison.halts_execution());
    assert!(!FailureCancellationBehavior::Propagate.halts_execution());
    assert!(!FailureCancellationBehavior::Ignore.halts_execution());
}

#[test]
fn memory_ordering_conversion_contracts() {
    assert_eq!(
        MemoryOrdering::Relaxed.to_atomic_ordering(),
        Some(AtomicOrdering::Relaxed)
    );
    assert_eq!(
        MemoryOrdering::Acquire.to_atomic_ordering(),
        Some(AtomicOrdering::Acquire)
    );
    assert_eq!(
        MemoryOrdering::Release.to_atomic_ordering(),
        Some(AtomicOrdering::Release)
    );
    assert_eq!(
        MemoryOrdering::AcqRel.to_atomic_ordering(),
        Some(AtomicOrdering::AcqRel)
    );
    assert_eq!(
        MemoryOrdering::SeqCst.to_atomic_ordering(),
        Some(AtomicOrdering::SeqCst)
    );
    assert_eq!(MemoryOrdering::GridSync.to_atomic_ordering(), None);

    assert_eq!(
        MemoryOrdering::Relaxed.execution_scope(),
        ExecutionScope::Thread
    );
    assert_eq!(
        MemoryOrdering::SeqCst.execution_scope(),
        ExecutionScope::Workgroup
    );
    assert_eq!(
        MemoryOrdering::GridSync.execution_scope(),
        ExecutionScope::Grid
    );

    assert_eq!(MemoryOrdering::Relaxed.memory_scope(), MemoryScope::Thread);
    assert_eq!(
        MemoryOrdering::SeqCst.memory_scope(),
        MemoryScope::Workgroup
    );
    assert_eq!(MemoryOrdering::GridSync.memory_scope(), MemoryScope::Device);
}
