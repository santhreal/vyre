//! WHY: a resource name carries descriptor binding identity, so an artifact that
//! gives one name to two values lets a consumer bind the wrong buffer. Before
//! this check the two consumers of the name lookup disagreed: the target
//! compiler refused the collision while the runtime's `program_outputs` built a
//! map that silently kept whichever value came last and projected it as the
//! program output. Both now read `Artifact::canonical_value_by_name`, and decode
//! refuses the bytes outright so neither has to be trusted to check.
//!
//! Does not catch: a collision reaching a consumer through some path that builds
//! an `Artifact` without `from_bytes`. Nothing in the workspace does, because the
//! compiler derives resources from graph value names and `ProgramGraph` already
//! refuses a duplicate name, but a future constructor would need the same check.

use super::*;
use crate::allocation::{
    AddressSpace, AliasClass, AllocationRegion, DevicePeak, DeviceSlot, PlacementLayout,
    PlacementPermits, RegionOwner, ValuePlacement, ALLOCATION_SCHEMA_VERSION, REGION_ALIGNMENT,
};
use crate::cost::CostBreakdown;
use crate::identity::{ArtifactNodeId, FusionGroupId};
use crate::measure::{
    CandidateMeasurement, DeviceState, MeasurementEnvironment, MeasurementProtocol,
    MeasurementRecord, SampleEstimate,
};
use crate::mesh::{PartitionKind, RegionPartition, ShardAssignment};
use crate::request::{SearchBudget, SearchWork};
use vyre_foundation::schedule::{SchedulePhaseId, ScheduleTransform};

fn payload(resources: Vec<ResourceRecord>) -> ArtifactPayload {
    let allocation = caller_bound(&resources);
    ArtifactPayload {
        schema_version: ARTIFACT_SCHEMA_VERSION,
        nodes: Vec::new(),
        dependencies: Vec::new(),
        selected_plan: SelectedPlan {
            topology: crate::ExecutionTopology::Sequential,
            frontier_topology: crate::FrontierTopology::SparseFrontier,
            schedule: vyre_test_support::selected_schedules::synthetic(1),
            derivation: Vec::new(),
            certificate: crate::SearchCertificate::new(crate::SCHEDULE_GRAMMAR_VERSION),
            fusion: Vec::new(),
            barriers: Vec::new(),
            materializations: Vec::new(),
            candidates_explored: 1,
            pareto_frontier: 1,
            search_budget: SearchBudget::new(1, 1, 0, 0, 1),
            search_work: SearchWork {
                candidates_explored: 1,
                ..SearchWork::default()
            },
            selection_cost: CostBreakdown::default(),
            pruned_fusions: Vec::new(),
            execution: ExecutionMode::Static,
            measurement: PlanMeasurement::Unbudgeted,
            numeric_budget: crate::NumericRecord::exact(),
        },
        abi: ArtifactAbi {
            resources: Vec::new(),
            entries: Vec::new(),
        },
        resources,
        resource_envelope: ResourceEnvelope { total_bytes: 0 },
        geometry: Vec::new(),
        allocation,
        topology: MeshTopologyPlan::single_device(
            Digest([0; 32]),
            DeviceSlot(0),
            vec![RegionPartition {
                node: ArtifactNodeId(0),
                kind: PartitionKind::Replicated,
                axis: None,
                region_points: 1,
                shards: vec![ShardAssignment {
                    shard: 0,
                    device: DeviceSlot(0),
                    coordinate: vec![0],
                    points: 1,
                }],
            }],
        ),
        provenance: Provenance {
            source_graph: Digest([0; 32]),
            semantic_graph: Digest([0; 32]),
            request: Digest([0; 32]),
            objective: crate::objective::CompileObjective::minimize_latency(),
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
        },
    }
}

fn resource(name: &str, value: u32) -> ResourceRecord {
    ResourceRecord {
        value: ArtifactValueId(value),
        name: name.to_string(),
        element_count: 1,
        byte_count: 4,
        lifetime: ResourceLifetime::Output,
        retained_predecessor: None,
        first_stage: 0,
        last_stage: 0,
    }
}

/// Caller-bound storage for every fixture resource, one region each.
///
/// A decoded artifact states where every value it carries lives, so a fixture
/// that skipped the plan would be refused for the plan rather than for the
/// record each case is about.
fn caller_bound(resources: &[ResourceRecord]) -> AllocationPlan {
    let mut placed: Vec<&ResourceRecord> = resources
        .iter()
        .filter(|resource| resource.byte_count > 0)
        .collect();
    placed.sort_by_key(|resource| resource.value);
    placed.dedup_by_key(|resource| resource.value);
    if placed.is_empty() {
        return AllocationPlan::empty();
    }
    let regions: Vec<AllocationRegion> = placed
        .iter()
        .map(|resource| AllocationRegion {
            device: DeviceSlot(0),
            address_space: AddressSpace::Device,
            owner: RegionOwner::Caller,
            offset: 0,
            bytes: resource.byte_count,
            alignment: REGION_ALIGNMENT,
            padding_bytes: 0,
            placements: vec![ValuePlacement {
                value: resource.value,
                byte_offset: 0,
                bytes: resource.byte_count,
                lifetime: resource.lifetime,
                alias_class: AliasClass(resource.value.0),
                first_stage: resource.first_stage,
                last_stage: resource.last_stage,
                synchronized: false,
                layout: PlacementLayout {
                    element_bytes: 4,
                    storage_order: vec![0],
                    strides: vec![1],
                    contiguous: true,
                },
                permits: PlacementPermits::default(),
            }],
        })
        .collect();
    let peak = placed
        .iter()
        .fold(0u64, |total, resource| total + resource.byte_count);
    AllocationPlan {
        schema_version: ALLOCATION_SCHEMA_VERSION,
        regions,
        device_peaks: vec![DevicePeak {
            device: DeviceSlot(0),
            peak_bytes: peak,
            allocated_bytes: 0,
        }],
        aggregate_peak_bytes: peak,
    }
}

fn decode_payload(payload: ArtifactPayload) -> Result<Artifact, CompileError> {
    let framed = encode_payload(&payload).expect("fixture payload must frame");
    Artifact::from_bytes(&framed.bytes)
}

fn decode(resources: Vec<ResourceRecord>) -> Result<Artifact, CompileError> {
    decode_payload(payload(resources))
}

/// WHY: topology cardinality is part of the selected schedule. A zero
/// cardinality used to deserialize successfully and reached target lowering
/// as an execution topology that could not launch.
#[test]
fn selected_schedule_rejects_every_zero_topology_cardinality() {
    for topology in [
        crate::ExecutionTopology::ConcurrentQueue { queues: 0 },
        crate::ExecutionTopology::ResidentPartition {
            partitions: 0,
            mode: crate::ResidentPartitionMode::FixedSpatialMask,
        },
    ] {
        let mut invalid = payload(Vec::new());
        invalid.selected_plan.topology = topology;
        let error = decode_payload(invalid).expect_err("zero topology cardinality must not decode");
        assert!(
            error
                .diagnostic
                .location
                .as_ref()
                .and_then(|location| location.path.as_deref())
                .is_some_and(|path| path.starts_with("artifact.body.selected_plan.topology")),
            "diagnostic must identify selected topology: {error}"
        );
    }
}

/// WHY: selected phase state and transform proof rows are authenticated
/// together. A stale or edited phase must not reach physical lowering even
/// when the surrounding artifact frame was recomputed.
#[test]
fn selected_schedule_rejects_mutated_phase_and_transform_provenance() {
    let mut valid = payload(Vec::new());
    valid
        .selected_plan
        .schedule
        .apply(ScheduleTransform::SetWorkgroup {
            phase: SchedulePhaseId(0),
            shape: [32, 1, 1],
        })
        .unwrap();

    let mut phase_mutation = valid.clone();
    phase_mutation.selected_plan.schedule.phases[0].workgroup[0] = 64;
    let phase_error =
        decode_payload(phase_mutation).expect_err("mutated phase geometry must fail replay");
    assert!(phase_error.diagnostic.message.contains("replay"));

    let mut provenance_mutation = valid;
    provenance_mutation.selected_plan.schedule.transforms[0]
        .provenance
        .inverse
        .previous_identity[0] ^= 1;
    let provenance_error = decode_payload(provenance_mutation)
        .expect_err("mutated transform provenance must fail replay");
    assert!(provenance_error.diagnostic.message.contains("evidence"));
}

/// WHY: duplicated candidate accounting can authenticate two incompatible
/// answers for the same bounded search. Decode must reject the disagreement
/// before any target consumes the schedule.
#[test]
fn selected_schedule_rejects_inconsistent_candidate_accounting() {
    let mut invalid = payload(Vec::new());
    invalid.selected_plan.search_work.candidates_explored = 2;
    let error = decode_payload(invalid).expect_err("candidate accounting disagreement must fail");
    assert!(
        error
            .diagnostic
            .message
            .contains("records 1 explored candidates"),
        "diagnostic must state both accounting owners: {error}"
    );
}

/// WHY: the recorded frontier is what tells a reader whether the objective's
/// tie breakers decided the winner or whether one plan dominated every
/// other. A frontier of zero states the selected plan was dominated by
/// something, and a frontier wider than the explored set states a choice the
/// search never had; both are plans this selector cannot have produced.
#[test]
fn selected_schedule_rejects_a_frontier_the_search_cannot_support() {
    for frontier in [0, 2] {
        let mut invalid = payload(Vec::new());
        invalid.selected_plan.pareto_frontier = frontier;
        let error = decode_payload(invalid)
            .expect_err("a frontier the search cannot support must not decode");
        assert!(
            error
                .diagnostic
                .location
                .as_ref()
                .and_then(|location| location.path.as_deref())
                .is_some_and(|path| path == "artifact.body.selected_plan.pareto_frontier"),
            "diagnostic must identify the recorded frontier: {error}"
        );
    }
}

/// WHY: measurement samples are charged per target compilation actually
/// performed. Unused compilation budget cannot authenticate samples for a
/// finalist that was never compiled.
#[test]
fn selected_schedule_rejects_measurements_without_compiled_finalists() {
    let mut invalid = payload(Vec::new());
    invalid.selected_plan.search_budget = SearchBudget::new(1, 1, 2, 1, 1);
    invalid.selected_plan.search_work.target_compilations = 1;
    invalid.selected_plan.search_work.measurements = 2;
    let error = decode_payload(invalid).expect_err("samples without a compiled finalist must fail");
    assert!(
        error
            .diagnostic
            .location
            .as_ref()
            .and_then(|location| location.path.as_deref())
            .is_some_and(|path| path.ends_with("search_work.measurements")),
        "diagnostic must identify measurement accounting: {error}"
    );
}

/// WHY: a measured schedule whose evidence does not authenticate its own
/// winner is not measurement evidence and must not deserialize as one. The
/// artifact path has to reject it, not only the record constructor, because
/// decoded bytes never went through that constructor.
#[test]
fn selected_schedule_rejects_measured_evidence_that_authenticates_nothing() {
    for (measurement, expected) in [
        (
            measured_evidence(|record| record.candidates.clear()),
            "retains no candidate samples",
        ),
        (
            measured_evidence(|record| record.winner = 7),
            "names none of the",
        ),
        (
            measured_evidence(|record| record.rounds = 0),
            "round(s) under a protocol ending after",
        ),
        (
            measured_evidence(|record| record.candidates[0].estimate.estimate_ns = 0),
            "zero device time",
        ),
        (
            measured_evidence(|record| record.candidates[0].samples.push(9)),
            "but its estimate covers",
        ),
        (
            measured_evidence(|record| record.protocol.version = 0),
            "records no protocol version",
        ),
    ] {
        let mut invalid = payload(Vec::new());
        invalid.selected_plan.search_budget = SearchBudget::new(1, 1, 1, 1, 1);
        invalid.selected_plan.search_work.target_compilations = 1;
        invalid.selected_plan.search_work.measurements = 1;
        invalid.selected_plan.measurement = measurement;
        let error = decode_payload(invalid)
            .expect_err("evidence that authenticates nothing must not decode");
        assert!(
            error.diagnostic.message.contains(expected),
            "diagnostic must identify invalid measurement evidence, wanted `{expected}`: {error}"
        );
    }
}

/// One measured record that decodes, with `mutate` applied to break exactly
/// one of the rules the record has to satisfy.
fn measured_evidence(mutate: impl FnOnce(&mut MeasurementRecord)) -> PlanMeasurement {
    let protocol = MeasurementProtocol::V1.fitted(1);
    let mut record = MeasurementRecord {
        protocol,
        environment: MeasurementEnvironment {
            warmup_launches: protocol.warmup_launches,
            facts_calibration_version: 3,
            first_round_ns: 128,
            last_round_ns: 128,
            state: DeviceState::unreported(),
        },
        rounds: 1,
        candidates: vec![CandidateMeasurement {
            identity: Digest([0; 32]),
            analytic_rank: 0,
            predicted_ns: 100,
            samples: vec![128],
            estimate: SampleEstimate {
                estimate_ns: 128,
                uncertainty_ns: 0,
                kept: 1,
                trimmed: 0,
            },
        }],
        winner: 0,
    };
    mutate(&mut record);
    PlanMeasurement::Measured(record)
}

/// WHY: unbudgeted and untimed plans cannot authenticate device samples
/// that their measurement state states were never used.
#[test]
fn selected_schedule_rejects_samples_on_every_unmeasured_state() {
    for measurement in [PlanMeasurement::Unbudgeted, PlanMeasurement::UntimedDevice] {
        let mut invalid = payload(Vec::new());
        invalid.selected_plan.search_budget = SearchBudget::new(1, 1, 1, 1, 1);
        invalid.selected_plan.search_work.target_compilations = 1;
        invalid.selected_plan.search_work.measurements = 1;
        invalid.selected_plan.measurement = measurement;
        let error = decode_payload(invalid)
            .expect_err("unmeasured schedules with device samples must not decode");
        assert!(
            error
                .diagnostic
                .message
                .contains("unmeasured selection records 1 on-device measurements"),
            "diagnostic must identify contradictory measurement state: {error}"
        );
    }
}

#[test]
fn one_name_for_two_values_is_refused_at_decode() {
    let error = decode(vec![resource("out", 1), resource("out", 2)])
        .expect_err("a name claimed by two values must not decode");

    assert_eq!(
        error.diagnostic.code.as_str(),
        CompilerFailureKind::MalformedArtifact.as_str()
    );
    let message = error.diagnostic.message.as_ref();
    assert!(
        message.contains("`out`") && message.contains("value 1") && message.contains("value 2"),
        "the diagnostic must name the reused name and both values it claims: {message}"
    );
}

#[test]
fn one_name_repeated_for_one_value_is_not_a_collision() {
    let artifact = decode(vec![resource("out", 1), resource("out", 1)])
        .expect("a repeated record for one value names one binding");

    let by_name = artifact
        .canonical_value_by_name()
        .expect("no name claims two values");
    assert_eq!(by_name.get("out").copied(), Some(ArtifactValueId(1)));
}

#[test]
fn every_resource_name_resolves_to_its_own_value() {
    let names = ["a", "b", "c"];
    let records = names
        .iter()
        .enumerate()
        .map(|(index, name)| resource(name, index as u32 + 7))
        .collect::<Vec<_>>();
    let artifact = decode(records.clone()).expect("distinct names must decode");

    let by_name = artifact
        .canonical_value_by_name()
        .expect("distinct names cannot collide");
    assert_eq!(by_name.len(), records.len());
    for record in &records {
        assert_eq!(
            by_name.get(record.name.as_str()).copied(),
            Some(record.value),
            "`{}` must resolve to its own value",
            record.name
        );
    }
}

fn entry_node(id: u32) -> NodeRecord {
    NodeRecord {
        id: ArtifactNodeId(id),
        name: format!("n{id}"),
        program: Vec::new(),
    }
}

fn launch(node: u32) -> GeometryRecord {
    crate::geometry_fixtures::geometry(node, 0, [32, 1, 1])
}

fn launchable() -> ArtifactPayload {
    let mut payload = payload(vec![ResourceRecord {
        value: ArtifactValueId(1),
        name: "scratch".to_string(),
        element_count: 64,
        byte_count: 256,
        lifetime: ResourceLifetime::Invocation,
        retained_predecessor: None,
        first_stage: 0,
        last_stage: 0,
    }]);
    payload.nodes = vec![entry_node(0)];
    payload.geometry = vec![launch(0)];
    payload.allocation = AllocationPlan {
        schema_version: crate::allocation::ALLOCATION_SCHEMA_VERSION,
        regions: vec![AllocationRegion {
            device: DeviceSlot(0),
            address_space: AddressSpace::Device,
            owner: RegionOwner::Artifact,
            offset: 0,
            bytes: 256,
            alignment: crate::allocation::REGION_ALIGNMENT,
            padding_bytes: 0,
            placements: vec![ValuePlacement {
                value: ArtifactValueId(1),
                byte_offset: 0,
                bytes: 256,
                lifetime: ResourceLifetime::Invocation,
                alias_class: AliasClass(1),
                first_stage: 0,
                last_stage: 0,
                synchronized: false,
                layout: PlacementLayout {
                    element_bytes: 4,
                    storage_order: vec![0],
                    strides: vec![1],
                    contiguous: true,
                },
                permits: PlacementPermits::default(),
            }],
        }],
        device_peaks: vec![DevicePeak {
            device: DeviceSlot(0),
            peak_bytes: 256,
            allocated_bytes: 256,
        }],
        aggregate_peak_bytes: 256,
    };
    payload
}

fn rejection_path(payload: ArtifactPayload, case: &str) -> String {
    let error = decode_payload(payload).unwrap_err();
    error
        .diagnostic
        .location
        .as_ref()
        .and_then(|location| location.path.clone())
        .unwrap_or_else(|| panic!("{case}: diagnostic carries no path: {error}"))
}

/// WHY: recording geometry is what stops a consumer from computing one. A
/// node with no record leaves exactly the hole a consumer fills in itself,
/// and two records for one node let two consumers launch different shapes
/// out of one authenticated artifact.
#[test]
fn decode_rejects_a_geometry_set_that_does_not_cover_every_node_exactly_once() {
    decode_payload(launchable()).expect("a covered node set decodes");

    let mut missing = launchable();
    missing.geometry.clear();
    let error = decode_payload(missing).expect_err("a node without geometry must not decode");
    assert!(
        error
            .diagnostic
            .message
            .contains("node 0 carries no selected geometry"),
        "diagnostic must name the uncovered node: {error}"
    );

    let mut doubled = launchable();
    doubled.geometry.push(launch(0));
    let error = decode_payload(doubled).expect_err("two records for one node must not decode");
    assert!(
        error
            .diagnostic
            .message
            .contains("node 0 carries two geometry records"),
        "diagnostic must name the doubly recorded node: {error}"
    );
}

/// WHY: field-level launchability is checked beside the records, and decode
/// is the boundary that has to apply it. Bytes that reached a consumer with
/// an unlaunchable record would be launched, because the consumer no longer
/// has geometry of its own to fall back on.
#[test]
fn decode_refuses_an_unlaunchable_record_before_a_consumer_reads_it() {
    let mut invalid = launchable();
    invalid.geometry[0].grid = [1, 1, 1];
    assert_eq!(
        rejection_path(invalid, "grid that does not cover the recorded points"),
        "artifact.geometry[0].grid"
    );
}

/// WHY: a predecessor names the entry point a submission waits on. One the
/// artifact does not carry is a dependency the runtime would drop, which
/// reorders the submission instead of failing it.
#[test]
fn decode_rejects_a_predecessor_the_artifact_does_not_carry() {
    let mut invalid = launchable();
    invalid.geometry[0].predecessors = vec![ArtifactNodeId(9)];
    let error = decode_payload(invalid).expect_err("an unknown predecessor must not decode");
    assert!(
        error
            .diagnostic
            .message
            .contains("depends on entry point 9 the artifact does not carry"),
        "diagnostic must name the missing entry point: {error}"
    );
}

/// WHY: the recorded order is the submission order. A consumer submits the
/// geometry set in the order the artifact lists it and runs each group at the
/// stage the artifact states, so a list that places an entry point before one
/// it depends on, or a stage that lets a consuming group run no later than
/// its producer, runs a consumer of a value before its producer wrote it.
/// Recording that wrong is refused here rather than repaired by a topological
/// sort in every consumer, because two consumers sorting independently is how
/// the order stopped being the compiler's in the first place.
#[test]
fn decode_rejects_a_recorded_order_that_contradicts_the_dependency_dag() {
    // Node 1 produces for node 0, recorded in dependency order.
    let ordered = |dependent_first: bool, consumer_stage: u32, producer_stage: u32| {
        let mut payload = launchable();
        payload.nodes = vec![entry_node(0), entry_node(1)];
        let mut dependent = launch(0);
        dependent.predecessors = vec![ArtifactNodeId(1)];
        payload.geometry = if dependent_first {
            vec![dependent, launch(1)]
        } else {
            vec![launch(1), dependent]
        };
        let group = |id: u32, node: u32, stage: u32| FusionRecord {
            id: FusionGroupId(id),
            members: vec![ArtifactNodeId(node)],
            stage,
            legality: Vec::new(),
        };
        // Producer group listed first, so vec position never carries the
        // ordering: only the recorded stage does.
        payload.selected_plan.fusion =
            vec![group(1, 1, producer_stage), group(0, 0, consumer_stage)];
        payload.selected_plan.barriers = (1..=producer_stage.max(consumer_stage))
            .map(|after| BarrierRecord {
                before_stage: after - 1,
                after_stage: after,
                dependencies: Vec::new(),
            })
            .collect();
        payload
    };

    decode_payload(ordered(false, 1, 0)).expect("a dependency-ordered artifact decodes");

    let error = decode_payload(ordered(true, 1, 0))
        .expect_err("a dependent recorded first must not decode");
    assert!(
        error
            .diagnostic
            .message
            .contains("node 0 is recorded before entry point 1 it depends on"),
        "diagnostic must name the misordered pair: {error}"
    );

    for (case, consumer_stage, producer_stage) in [
        ("consumer group sharing its producer's stage", 0, 0),
        ("consumer group staged before its producer", 0, 1),
    ] {
        assert_eq!(
            rejection_path(ordered(false, consumer_stage, producer_stage), case),
            "artifact.body.selected_plan",
            "{case}"
        );
    }
}

/// WHY: the runtime allocates the recorded plan and binds its offsets
/// verbatim, so a region the resource set does not back is a bind of a
/// buffer that has no owner, and an unbindable plan must not survive decode.
#[test]
fn decode_rejects_storage_the_resource_set_does_not_back() {
    let cases: Vec<(&str, fn(&mut ArtifactPayload), &str)> = vec![
        (
            "placement for a value the artifact does not carry",
            |payload| payload.allocation.regions[0].placements[0].value = ArtifactValueId(7),
            "artifact.body.allocation",
        ),
        (
            "placement disagreeing with its resource lifetime",
            |payload| {
                payload.allocation.regions[0].placements[0].lifetime = ResourceLifetime::Retained;
            },
            "artifact.body.allocation",
        ),
        (
            "placement disagreeing with its resource live range",
            |payload| payload.allocation.regions[0].placements[0].last_stage = 4,
            "artifact.body.allocation",
        ),
        (
            "resource row disagreeing with the placement byte count",
            |payload| payload.resources[0].byte_count = 128,
            "artifact.body.allocation",
        ),
        (
            "value occupying bytes the plan places nowhere",
            |payload| {
                payload.allocation = AllocationPlan::empty();
            },
            "artifact.body.allocation",
        ),
        (
            "misaligned region",
            |payload| payload.allocation.regions[0].offset = 8,
            "artifact.allocation.regions[0].offset",
        ),
    ];
    for (case, mutate, expected) in cases {
        let mut invalid = launchable();
        mutate(&mut invalid);
        assert_eq!(rejection_path(invalid, case), expected, "{case}");
    }
}

/// WHY: the plan is what the runtime allocates and what lowering verifies
/// against, so every field of it has to be inside what the digest seals.
/// A field outside identity would let two different physical plans present
/// the same artifact, and the one admitted would not be the one measured.
/// Decode refuses the fields it can cross-check against the resource set;
/// the rest are proven here to change the artifact they belong to.
#[test]
fn every_allocation_field_participates_in_artifact_identity() {
    let canonical = encode_payload(&launchable()).expect("the fixture payload frames");
    let cases: Vec<(&str, fn(&mut ArtifactPayload))> = vec![
        ("lifetime", |payload| {
            payload.allocation.regions[0].placements[0].lifetime = ResourceLifetime::Retained;
        }),
        ("alias class", |payload| {
            payload.allocation.regions[0].placements[0].alias_class = AliasClass(9);
        }),
        ("offset", |payload| {
            payload.allocation.regions[0].placements[0].byte_offset = 256;
        }),
        ("layout", |payload| {
            payload.allocation.regions[0].placements[0].layout.strides = vec![2];
        }),
        ("size", |payload| {
            payload.allocation.regions[0].placements[0].bytes = 512;
        }),
        ("device", |payload| {
            payload.allocation.regions[0].device = DeviceSlot(3);
        }),
        ("synchronization", |payload| {
            payload.allocation.regions[0].placements[0].synchronized = true;
        }),
        ("reuse permit", |payload| {
            payload.allocation.regions[0].placements[0].permits.in_place = true;
        }),
    ];
    for (field, mutate) in cases {
        let mut mutated = launchable();
        mutate(&mut mutated);
        let framed = encode_payload(&mutated).expect("a mutated payload frames");
        assert_ne!(
            framed.digest, canonical.digest,
            "{field} must participate in artifact identity"
        );
    }
}

/// WHY: a region on a device the peaks do not account for states bytes on
/// hardware nothing sized, so the runtime would allocate against a total no
/// ranking ever priced.
#[test]
fn decode_rejects_a_region_the_device_peaks_do_not_account_for() {
    let mut invalid = launchable();
    invalid.allocation.regions[0].device = DeviceSlot(3);
    assert_eq!(
        rejection_path(invalid, "region on an unaccounted device"),
        "artifact.allocation.device_peaks[0].allocated_bytes"
    );
}

/// WHY: a body written by the previous schema carries no geometry set and no
/// allocation plan, so a reader that accepted one would submit an artifact
/// with no recorded launch. The version is read before the body, so stale
/// bytes are refused instead of partially interpreted.
#[test]
fn bytes_stamped_with_the_previous_schema_are_refused() {
    let framed = encode_payload(&launchable()).expect("fixture payload must frame");
    Artifact::from_bytes(&framed.bytes).expect("the current schema decodes");

    let mut stale = framed.bytes.clone();
    stale[4..6].copy_from_slice(&(17u16).to_le_bytes());
    let error = Artifact::from_bytes(&stale).expect_err("a stale schema must not decode");
    assert_eq!(
        error.diagnostic.code.as_str(),
        CompilerFailureKind::VersionSkew.as_str()
    );
    assert!(error
        .to_string()
        .contains("schema 17 is unsupported; expected 18"));
}

#[test]
fn frontier_topology_participates_in_artifact_identity() {
    let canonical = encode_payload(&launchable()).expect("the fixture payload frames");
    let mut mutated = launchable();
    mutated.selected_plan.frontier_topology = crate::FrontierTopology::BlockDenseFrontier;
    let framed = encode_payload(&mutated).expect("a mutated payload frames");
    assert_ne!(
        framed.digest, canonical.digest,
        "frontier_topology must participate in artifact identity"
    );
}

/// One stage assignment with a data edge from the first stage into the second.
fn staged_groups() -> BTreeMap<ArtifactNodeId, (FusionGroupId, u32)> {
    BTreeMap::from([
        (ArtifactNodeId(0), (FusionGroupId(0), 0)),
        (ArtifactNodeId(1), (FusionGroupId(1), 1)),
    ])
}

fn staged_edges() -> Vec<DependencyEdge> {
    vec![DependencyEdge {
        from: DependencyEndpoint::Node(ArtifactNodeId(0)),
        to: DependencyEndpoint::Node(ArtifactNodeId(1)),
        kind: crate::identity::DependencyKind::Data,
        value: None,
    }]
}

fn one_boundary() -> Vec<BarrierRecord> {
    vec![BarrierRecord {
        before_stage: 0,
        after_stage: 1,
        dependencies: vec![0],
    }]
}

/// WHY: a barrier record is what a backend submits between two stages. Every
/// mutation of the pair it states, of the edges it admits, and of the set of
/// records itself is a device ordering the plan was not selected under, so each
/// one has to be refused rather than decoded.
#[test]
fn every_barrier_mutation_is_refused() {
    let groups = staged_groups();
    let edges = staged_edges();
    validate_barrier_records(&one_boundary(), &edges, &groups, 1)
        .expect("the derived boundary set is admitted");

    let cases: Vec<(&str, fn(&mut Vec<BarrierRecord>))> = vec![
        ("the boundary is removed", |barriers| barriers.clear()),
        ("a second boundary is added", |barriers| {
            barriers.push(BarrierRecord {
                before_stage: 1,
                after_stage: 2,
                dependencies: Vec::new(),
            });
        }),
        ("the stage pair is inverted", |barriers| {
            barriers[0].before_stage = 1;
            barriers[0].after_stage = 0;
        }),
        ("the pair is no longer consecutive", |barriers| {
            barriers[0].after_stage = 2;
        }),
        ("the crossing edge is dropped", |barriers| {
            barriers[0].dependencies.clear();
        }),
        ("an edge that does not cross is admitted", |barriers| {
            barriers[0].dependencies.push(7);
        }),
        ("the admitted edges are reordered", |barriers| {
            barriers[0].dependencies = vec![1, 0];
        }),
    ];
    for (what, mutate) in cases {
        let mut barriers = one_boundary();
        mutate(&mut barriers);
        let error = validate_barrier_records(&barriers, &edges, &groups, 1).expect_err(what);
        assert_eq!(
            error.diagnostic.code.as_str(),
            CompilerFailureKind::MalformedArtifact.as_str(),
            "{what} must be refused as a malformed artifact"
        );
        assert_eq!(
            error
                .diagnostic
                .location
                .as_ref()
                .and_then(|location| location.path.as_deref()),
            Some("artifact.body.selected_plan.barriers"),
            "{what} must name the barrier records"
        );
    }
}

/// Two groups on consecutive stages, carrying the data edge that crosses
/// between them and the boundary the compiler derives for it.
fn staged_payload() -> ArtifactPayload {
    let mut payload = launchable();
    payload.nodes = vec![entry_node(0), entry_node(1)];
    let mut dependent = launch(1);
    dependent.predecessors = vec![ArtifactNodeId(0)];
    payload.geometry = vec![launch(0), dependent];
    payload.dependencies = staged_edges();
    payload.selected_plan.fusion = vec![
        FusionRecord {
            id: FusionGroupId(0),
            members: vec![ArtifactNodeId(0)],
            stage: 0,
            legality: Vec::new(),
        },
        FusionRecord {
            id: FusionGroupId(1),
            members: vec![ArtifactNodeId(1)],
            stage: 1,
            legality: Vec::new(),
        },
    ];
    payload.selected_plan.barriers = one_boundary();
    payload
}

/// WHY: `every_barrier_mutation_is_refused` proves the derivation and would
/// stay green with the check unreachable from decode. This one proves decode
/// reaches it with the stage count the plan actually runs, so dropping the
/// call, or passing a stage count that makes the record set look complete,
/// turns it red.
#[test]
fn decode_refuses_a_barrier_set_the_recorded_stages_contradict() {
    decode_payload(staged_payload()).expect("the derived boundary set decodes");

    let cases: Vec<(&str, fn(&mut ArtifactPayload))> = vec![
        ("the only boundary is dropped", |payload| {
            payload.selected_plan.barriers.clear();
        }),
        ("a boundary past the last stage is admitted", |payload| {
            payload.selected_plan.barriers.push(BarrierRecord {
                before_stage: 1,
                after_stage: 2,
                dependencies: Vec::new(),
            });
        }),
        ("the crossing edge is no longer admitted", |payload| {
            payload.selected_plan.barriers[0].dependencies.clear();
        }),
        ("the boundary states the wrong stage pair", |payload| {
            payload.selected_plan.barriers[0].before_stage = 1;
            payload.selected_plan.barriers[0].after_stage = 2;
        }),
    ];
    for (what, mutate) in cases {
        let mut payload = staged_payload();
        mutate(&mut payload);
        assert_eq!(
            rejection_path(payload, what),
            "artifact.body.selected_plan.barriers",
            "{what} must be refused where the boundaries are recorded"
        );
    }
}
