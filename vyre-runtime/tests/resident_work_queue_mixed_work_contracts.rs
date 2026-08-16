//! Contracts for `vyre_runtime::resident_work_queue::mixed_work`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_runtime::resident_work_queue::mixed_work::{
    mixed_work_protocol_evidence, validate_mixed_work_protocol, MixedWorkProtocolError,
    MixedWorkProtocolPlan, MixedWorkQueueClass, MixedWorkUnit, MixedWorkUnitType, OutputSlabId,
    ResidentArtifactId, MIXED_WORK_PROTOCOL_SCHEMA_VERSION,
};

use vyre_runtime::resident_work_queue::mixed_work::{
    mixed_work_protocol_evidence, validate_mixed_work_protocol, MixedWorkProtocolError,
    MixedWorkProtocolPlan, MixedWorkQueueClass, MixedWorkUnit, MixedWorkUnitType, OutputSlabId,
    ResidentArtifactId, MIXED_WORK_PROTOCOL_SCHEMA_VERSION,
};

fn unit(
    sequence: u64,
    queue_class: MixedWorkQueueClass,
    unit_type: MixedWorkUnitType,
) -> MixedWorkUnit {
    MixedWorkUnit::new(
        sequence,
        queue_class,
        unit_type,
        ResidentArtifactId(100 + sequence as u32),
        OutputSlabId(200 + sequence as u32),
        10,
        0xfeed_0000 + sequence,
    )
}

#[test]
fn mixed_scan_graph_parser_flow_work_emits_deterministic_bounded_drain_evidence() {
    let units = [
        unit(1, MixedWorkQueueClass::Scan, MixedWorkUnitType::ScanChunk),
        unit(
            2,
            MixedWorkQueueClass::Graph,
            MixedWorkUnitType::GraphFrontier,
        ),
        unit(
            3,
            MixedWorkQueueClass::Parser,
            MixedWorkUnitType::ParserShard,
        ),
        unit(
            4,
            MixedWorkQueueClass::Flow,
            MixedWorkUnitType::FlowRelationDelta,
        ),
        unit(
            5,
            MixedWorkQueueClass::Control,
            MixedWorkUnitType::DrainSentinel,
        ),
    ];
    let plan = MixedWorkProtocolPlan::new(&units, 64);

    let first = mixed_work_protocol_evidence(&plan)
        .expect("Fix: valid mixed-work plan should emit evidence");
    let second = validate_mixed_work_protocol(&plan)
        .expect("Fix: valid mixed-work plan should emit stable evidence");

    assert_eq!(first, second);
    assert_eq!(first.schema_version, MIXED_WORK_PROTOCOL_SCHEMA_VERSION);
    assert!(first.is_complete());
    assert!(first.covers_scan_graph_parser_flow());
    assert!(first.bounded_drain);
    assert_eq!(first.hidden_host_loop_count, 0);
    assert_eq!(first.unit_count, 5);
    assert_eq!(first.total_watchdog_budget_ticks, 50);
    assert_eq!(first.max_watchdog_budget_ticks, 10);
    assert_ne!(first.deterministic_output_digest, 0);
}

#[test]
fn zero_watchdog_budget_is_rejected() {
    let units = [MixedWorkUnit::new(
        7,
        MixedWorkQueueClass::Scan,
        MixedWorkUnitType::ScanChunk,
        ResidentArtifactId(1),
        OutputSlabId(1),
        0,
        9,
    )];
    let plan = MixedWorkProtocolPlan::new(&units, 1);

    assert!(matches!(
        validate_mixed_work_protocol(&plan),
        Err(MixedWorkProtocolError::ZeroUnitWatchdogBudget { sequence: 7 })
    ));
}

#[test]
fn class_unit_mismatch_is_rejected() {
    let units = [MixedWorkUnit::new(
        9,
        MixedWorkQueueClass::Parser,
        MixedWorkUnitType::FlowFixpointStep,
        ResidentArtifactId(1),
        OutputSlabId(1),
        1,
        9,
    )];
    let plan = MixedWorkProtocolPlan::new(&units, 1);

    assert!(matches!(
        validate_mixed_work_protocol(&plan),
        Err(MixedWorkProtocolError::QueueClassMismatch {
            sequence: 9,
            queue_class: "parser",
            unit_type: "flow_fixpoint_step"
        })
    ));
}

#[test]
fn drain_budget_must_bound_all_units() {
    let units = [
        unit(1, MixedWorkQueueClass::Scan, MixedWorkUnitType::ScanChunk),
        unit(
            2,
            MixedWorkQueueClass::Flow,
            MixedWorkUnitType::FlowRelationDelta,
        ),
    ];
    let plan = MixedWorkProtocolPlan::new(&units, 19);

    assert!(matches!(
        validate_mixed_work_protocol(&plan),
        Err(MixedWorkProtocolError::WatchdogBudgetExceeded {
            total_watchdog_budget_ticks: 20,
            drain_watchdog_budget_ticks: 19
        })
    ));
}

#[test]
fn resident_artifact_and_output_slab_ids_are_required() {
    let bad_artifact = [MixedWorkUnit::new(
        1,
        MixedWorkQueueClass::Scan,
        MixedWorkUnitType::ScanChunk,
        ResidentArtifactId(0),
        OutputSlabId(1),
        1,
        1,
    )];
    assert!(matches!(
        validate_mixed_work_protocol(&MixedWorkProtocolPlan::new(&bad_artifact, 1)),
        Err(MixedWorkProtocolError::ZeroResidentArtifactId { sequence: 1 })
    ));

    let bad_slab = [MixedWorkUnit::new(
        2,
        MixedWorkQueueClass::Scan,
        MixedWorkUnitType::ScanChunk,
        ResidentArtifactId(1),
        OutputSlabId(0),
        1,
        1,
    )];
    assert!(matches!(
        validate_mixed_work_protocol(&MixedWorkProtocolPlan::new(&bad_slab, 1)),
        Err(MixedWorkProtocolError::ZeroOutputSlabId { sequence: 2 })
    ));
}
