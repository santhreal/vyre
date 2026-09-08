//! Tests for generation-scoped namespaces, rolling upgrades, crash interruption, and rollback atomicity.
//!
//! WHY: proves Row 120:
//! - Generation-scoped cache namespaces prevent cross-version key contamination.
//! - Rolling upgrade allows concurrent old/new generation operation during transition.
//! - Commit cleanly drains and reclaims resources from the old generation.
//! - Rollback atomically evicts new generation resources and restores the prior generation.
//! - Incompatible version transitions fail before expensive state mutation.

use vyre_runtime::generation_namespace::{
    GenerationScopedNamespace, RollingUpgradeCoordinator, UpgradePhase,
};
use vyre_foundation::{ProtocolDomain, ProtocolVersion};

#[test]
fn generation_scoped_namespace_prevents_key_collisions() {
    let ns_v1 = GenerationScopedNamespace::new(
        ProtocolDomain::Artifact,
        ProtocolVersion::V1_0_0,
        1,
    );
    let ns_v2 = GenerationScopedNamespace::new(
        ProtocolDomain::Artifact,
        ProtocolVersion::V1_1_0,
        2,
    );

    let key1 = ns_v1.scoped_key("megakernel_relu_f32");
    let key2 = ns_v2.scoped_key("megakernel_relu_f32");

    assert_ne!(
        key1, key2,
        "Fix: generation-scoped namespaces must produce distinct keys for different generations/versions."
    );
    assert!(key1.contains("gen_1") && key1.contains("1.0.0"));
    assert!(key2.contains("gen_2") && key2.contains("1.1.0"));
}

#[test]
fn rolling_upgrade_lifecycle_and_resource_reclamation() {
    let mut coordinator = RollingUpgradeCoordinator::new(
        ProtocolDomain::RuntimeProtocol,
        ProtocolVersion::V1_0_0,
    );
    assert_eq!(coordinator.phase(), UpgradePhase::Active);

    // Register resources in generation 1
    coordinator.register_resource("res_gen1_buffer_a");
    coordinator.register_resource("res_gen1_buffer_b");
    assert_eq!(coordinator.active_resource_count(), 2);

    // Begin rolling upgrade to v1.1.0 (generation 2)
    let ns_gen2 = coordinator
        .begin_upgrade(ProtocolVersion::V1_1_0)
        .expect("Fix: rolling upgrade to compatible v1.1.0 must succeed.");

    assert_eq!(ns_gen2.generation_id, 2);
    assert!(matches!(
        coordinator.phase(),
        UpgradePhase::RollingUpgrade {
            draining_generation: 1,
            target_generation: 2
        }
    ));

    // Register resource in generation 2
    coordinator.register_resource("res_gen2_buffer_c");
    assert_eq!(coordinator.active_resource_count(), 3);

    // Commit upgrade: drains and removes generation 1 resources
    coordinator
        .commit_upgrade()
        .expect("Fix: committing upgrade must succeed.");

    assert_eq!(coordinator.phase(), UpgradePhase::Completed);
    // Only generation 2 resource survives
    assert_eq!(
        coordinator.active_resource_count(),
        1,
        "Fix: commit_upgrade must reclaim all resources belonging to the drained generation."
    );
}

#[test]
fn rolling_upgrade_rollback_atomically_cleans_target_generation() {
    let mut coordinator = RollingUpgradeCoordinator::new(
        ProtocolDomain::RuntimeProtocol,
        ProtocolVersion::V1_0_0,
    );

    coordinator.register_resource("stable_resource_1");

    // Begin upgrade to v1.1.0
    coordinator
        .begin_upgrade(ProtocolVersion::V1_1_0)
        .unwrap();

    // Partial resource allocated under target generation
    coordinator.register_resource("incomplete_new_resource");
    assert_eq!(coordinator.active_resource_count(), 2);

    // Simulate fault and trigger rollback to v1.0.0
    coordinator
        .rollback(ProtocolVersion::V1_0_0, "Crash interruption during rolling upgrade")
        .expect("Fix: rollback must succeed.");

    assert!(matches!(
        coordinator.phase(),
        UpgradePhase::RolledBack {
            restored_generation: 1,
            ..
        }
    ));

    // Partial resource from target generation was evicted; stable resource from gen 1 restored
    assert_eq!(
        coordinator.active_resource_count(),
        1,
        "Fix: rollback must evict incomplete resources from aborted target generation."
    );
    assert_eq!(coordinator.current_namespace().generation_id, 1);
    assert_eq!(
        coordinator.current_namespace().version,
        ProtocolVersion::V1_0_0
    );
}

#[test]
fn incompatible_target_version_is_rejected_before_state_change() {
    let mut coordinator = RollingUpgradeCoordinator::new(
        ProtocolDomain::RuntimeProtocol,
        ProtocolVersion::V1_0_0,
    );

    let err = coordinator
        .begin_upgrade(ProtocolVersion::V2_0_0)
        .expect_err("Fix: upgrading to incompatible major version must be rejected.");

    assert!(err.contains("Fix:"));
    assert_eq!(
        coordinator.phase(),
        UpgradePhase::Active,
        "Fix: rejected upgrade must leave coordinator in clean active phase without mutating state."
    );
}
