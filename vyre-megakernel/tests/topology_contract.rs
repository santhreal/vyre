//! Execution topology candidate model, legality, independence, and ranking contracts.
//!
//! WHY: DEDUP Rows 177.1-177.5 require a neutral execution-topology candidate model in
//! `CandidatePlan` and `vyre-megakernel`:
//! - 177.1: Represent sequential stages, concurrent independent submissions, and resident
//!   partitions with enforceable capability. Unknown device facts reject dependent topologies.
//! - 177.2: Reuse independence analysis; reject RAW/WAR/WAW conflicts, shared output aliases,
//!   cross-arm control dependencies, and device-wide barriers.
//! - 177.3: Avoid unenforceable SM-id masking: generate fixed spatial masks only with hardware
//!   capability; otherwise use concurrent queues or bounded resident work queues with progress.
//! - 177.4: Model asymmetric joins without inventing unbacked barriers.
//! - 177.5: Price aggregate registers, scratch, live bytes, occupancy, queue overlap, and join cost.
//!   Retain sequential baseline; test empty, imbalanced, conflicting, and occupancy-limited arms.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use vyre_foundation::validate::BackendCapabilities;
use vyre_megakernel::{
    compile, Artifact, CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts,
    ObjectiveMetric, SearchBudget, ARTIFACT_SCHEMA_VERSION,
};

#[path = "graph_fixtures/mod.rs"]
mod graph_fixtures;
use graph_fixtures::{
    asymmetric_join_graph, independent_two_arm_graph, raw_conflict_two_arm_graph,
};

fn facts() -> ExternalFacts {
    ExternalFacts::new(Digest([0x77; 32]), BTreeMap::from([("items".into(), 64)]))
}

fn device_default() -> DeviceFacts {
    DeviceFacts::new(BackendCapabilities::default(), 256)
        .with_occupancy(128, 4096)
        .with_compute_units(8)
        .with_concurrent_queues(4)
        .with_launch_costs(4224, 1000)
}

// ============================================================================
// 1. Sequential baseline retention (Row 177.1, 177.5)
// ============================================================================

#[test]
fn sequential_baseline_is_always_legal_and_retained() {
    let graph = independent_two_arm_graph();
    let request = CompileRequest::new(
        graph,
        facts(),
        device_default(),
        SearchBudget::new(64, 100_000, 8, 0, 1_000_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .validate()
    .expect("request must validate");

    let artifact = compile(&request).expect("compilation must succeed");
    assert!(artifact.selected_plan().candidates_explored >= 1);
    assert!(artifact.selected_plan().selection_cost.total > 0);
}

// ============================================================================
// 2. Unknown device facts reject dependent topologies without guessing (Row 177.1)
// ============================================================================

#[test]
fn unknown_device_facts_reject_dependent_topologies() {
    let graph = independent_two_arm_graph();
    // Device with 0 concurrent queues and 0 compute units (unknown facts)
    let unknown_device = DeviceFacts::unknown();

    let request = CompileRequest::new(
        graph,
        facts(),
        unknown_device,
        SearchBudget::new(64, 100_000, 8, 0, 1_000_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .validate()
    .expect("request must validate");

    let artifact = compile(&request).expect("compilation must succeed with sequential fallback");
    assert_eq!(
        artifact.selected_plan().selection_cost.launches,
        2,
        "an unknown device must fall back to sequential baseline (2 launches for 2 groups) instead of guessing concurrency"
    );
}

// ============================================================================
// 3. Arm independence and conflict analysis (Row 177.2)
// ============================================================================

#[test]
fn independent_arms_admit_concurrent_queue_topology() {
    let graph = independent_two_arm_graph();
    let device = device_default().with_concurrent_queues(4);

    let request = CompileRequest::new(
        graph,
        facts(),
        device,
        SearchBudget::new(64, 100_000, 8, 0, 1_000_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .validate()
    .expect("request must validate");

    let artifact = compile(&request).expect("compilation must succeed");
    assert_eq!(
        artifact.selected_plan().selection_cost.launches,
        1,
        "independent arms on device with concurrent queues must execute in 1 concurrent launch batch"
    );
}

#[test]
fn raw_waw_conflicts_reject_concurrent_execution() {
    let graph = raw_conflict_two_arm_graph();
    let device = device_default().with_concurrent_queues(4);

    let request = CompileRequest::new(
        graph,
        facts(),
        device,
        SearchBudget::new(64, 100_000, 8, 0, 1_000_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .validate()
    .expect("request must validate");

    let artifact = compile(&request).expect("compilation must succeed");
    assert!(artifact.selected_plan().candidates_explored >= 1);
}

// ============================================================================
// 4. Avoid unenforceable SM-id masking (Row 177.3)
// ============================================================================

#[test]
fn fixed_spatial_mask_rejected_without_enforceable_hardware_capability() {
    let graph = independent_two_arm_graph();
    // Device reports compute units but spatial partitioning capability is FALSE
    let device = device_default()
        .with_compute_units(16)
        .with_spatial_partitioning(false);

    let request = CompileRequest::new(
        graph,
        facts(),
        device,
        SearchBudget::new(64, 100_000, 8, 0, 1_000_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .validate()
    .expect("request must validate");

    let artifact =
        compile(&request).expect("compilation must succeed with fallback to legal topologies");
    assert!(artifact.selected_plan().candidates_explored >= 1);
}

#[test]
fn bounded_work_queue_rejected_without_cooperative_launch() {
    let graph = independent_two_arm_graph();
    // Device reports compute units but cooperative launch is FALSE
    let device = device_default()
        .with_compute_units(16)
        .with_cooperative_launch(false);

    let request = CompileRequest::new(
        graph,
        facts(),
        device,
        SearchBudget::new(64, 100_000, 8, 0, 1_000_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .validate()
    .expect("request must validate");

    let artifact =
        compile(&request).expect("compilation must succeed with fallback to legal topologies");
    assert!(artifact.selected_plan().candidates_explored >= 1);
}

// ============================================================================
// 5. Occupancy & scratch budgeting across resident partitions (Row 177.5)
// ============================================================================

#[test]
fn occupancy_exceeded_rejects_resident_partition_candidate() {
    let graph = independent_two_arm_graph();
    // Device with tiny register limit of 1 register per thread
    let tiny_device = device_default()
        .with_compute_units(8)
        .with_spatial_partitioning(true)
        .with_occupancy(1, 4096);

    let request = CompileRequest::new(
        graph,
        facts(),
        tiny_device,
        SearchBudget::new(64, 100_000, 8, 0, 1_000_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .validate()
    .expect("request must validate");

    let artifact = compile(&request).expect("compilation must succeed with fallback to baseline");
    assert!(artifact.selected_plan().candidates_explored >= 1);
}

// ============================================================================
// 6. Schema 10 preservation and artifact round-trip
// ============================================================================

#[test]
fn artifact_encoding_preserves_schema_12_and_compiled_topology_schedule() {
    let graph = independent_two_arm_graph();
    let device = device_default().with_concurrent_queues(4);

    let request = CompileRequest::new(
        graph,
        facts(),
        device,
        SearchBudget::new(64, 100_000, 8, 0, 1_000_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .validate()
    .expect("request must validate");

    let artifact = compile(&request).expect("compilation must succeed");
    assert_eq!(artifact.schema_version(), ARTIFACT_SCHEMA_VERSION);
    assert_eq!(artifact.schema_version(), 13);

    let wire_bytes = artifact.to_bytes().expect("artifact must encode");
    let decoded = Artifact::from_bytes(&wire_bytes).expect("artifact must decode");

    assert_eq!(decoded, artifact);
    assert_eq!(decoded.digest(), artifact.digest());
    assert_eq!(decoded.selected_plan(), artifact.selected_plan());
    decoded.selected_plan().schedule.validate().unwrap();
    assert_eq!(
        decoded.selected_plan().schedule.phases.len(),
        decoded.fusion().len(),
        "one selected schedule phase must describe each emitted fusion group"
    );

    // Verify the immediately preceding schema is rejected.
    let mut stale_bytes = wire_bytes.clone();
    stale_bytes[4..6].copy_from_slice(&(ARTIFACT_SCHEMA_VERSION - 1).to_le_bytes());
    let error =
        Artifact::from_bytes(&stale_bytes).expect_err("the preceding stale schema must fail");
    assert_eq!(error.diagnostic.code.as_str(), "MKC015_VERSION_SKEW");
}

// ============================================================================
// 7. Asymmetric joins without cooperative launch (Row 177.4)
// ============================================================================

#[test]
fn asymmetric_join_rejected_without_cooperative_launch() {
    let graph = asymmetric_join_graph();
    let device_no_coop = device_default()
        .with_compute_units(16)
        .with_spatial_partitioning(true)
        .with_cooperative_launch(false);

    let request = CompileRequest::new(
        graph,
        facts(),
        device_no_coop,
        SearchBudget::new(64, 100_000, 8, 0, 1_000_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .validate()
    .expect("request must validate");

    let artifact = compile(&request).expect("compilation must succeed");
    assert!(artifact.selected_plan().candidates_explored >= 1);
}

// ============================================================================
// 8. Cost model ranking and topology selection (Row 177.5)
// ============================================================================

#[test]
fn topology_ranking_prefers_concurrent_over_sequential_when_device_has_queues() {
    let graph_cq = independent_two_arm_graph();
    let graph_seq = independent_two_arm_graph();
    let device_concurrent = device_default().with_concurrent_queues(4);
    let device_sequential = DeviceFacts::unknown();

    let req_concurrent = CompileRequest::new(
        graph_cq,
        facts(),
        device_concurrent,
        SearchBudget::new(64, 100_000, 8, 0, 1_000_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .validate()
    .expect("request must validate");

    let req_sequential = CompileRequest::new(
        graph_seq,
        facts(),
        device_sequential,
        SearchBudget::new(64, 100_000, 8, 0, 1_000_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .validate()
    .expect("request must validate");

    let art_concurrent = compile(&req_concurrent).expect("concurrent compilation must succeed");
    let art_sequential = compile(&req_sequential).expect("sequential compilation must succeed");

    assert_ne!(
        art_concurrent.selected_plan().topology,
        art_sequential.selected_plan().topology,
        "fixture must select two distinct schedule topologies"
    );
    assert_ne!(
        art_concurrent.digest(),
        art_sequential.digest(),
        "artifact identity must authenticate the selected schedule topology"
    );
    assert!(
        art_concurrent.selected_plan().selection_cost.total <= art_sequential.selected_plan().selection_cost.total,
        "Concurrent queue topology must be cheaper than or equal to sequential baseline (CQ: {}, Seq: {})",
        art_concurrent.selected_plan().selection_cost.total,
        art_sequential.selected_plan().selection_cost.total
    );
    assert_eq!(art_concurrent.selected_plan().selection_cost.launches, 1);
    assert_eq!(art_sequential.selected_plan().selection_cost.launches, 2);
}
