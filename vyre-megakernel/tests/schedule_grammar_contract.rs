//! Candidate generation contracts: the grammar, the constraint classes, and the
//! certificate one bounded search records.
//!
//! WHY: candidate generation used to be a catalog. It proposed the baseline, one
//! fused edge, one greedy grouping, four launch widths, and two topologies, so a
//! kernel organization nobody had written by hand was unreachable. These tests
//! defend the replacement: every production of a versioned grammar contributes,
//! every eliminated family carries a stable reason, the same graph derives
//! different plans on different devices, and an exhausted budget still returns a
//! proved candidate with the certificate that reproduces the search.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, DataType, GraphInput, GraphOutput, ProgramGraph, ShapeDim, ValueContract,
    ValueLifetime,
};
use vyre_foundation::schedule::{
    MappingLevel, MemoryPlacement, PipelineRole, PipelineRoleGroup, ScheduleAxis, SchedulePhaseId,
    ScheduleTransform, SynchronizationScope,
};
use vyre_foundation::validate::BackendCapabilities;
use vyre_megakernel::{
    compile, Artifact, CompileRequest, DeviceFacts, Digest, ExternalFacts, PruneReason,
    ScheduleProduction, SearchBudget, SCHEDULE_GRAMMAR_VERSION,
};
use vyre_test_support::pass_programs::{add_program, copy_program};

/// A two-dimensional `u32` value, so a phase carries more than one axis.
fn matrix(lifetime: ValueLifetime) -> ValueContract {
    ValueContract {
        dtype: DataType::U32,
        shape: vec![
            ShapeDim::Symbol("rows".into()),
            ShapeDim::Symbol("cols".into()),
        ],
        access: BufferAccess::ReadWrite,
        lifetime,
    }
}

/// The same value read by a consumer that only loads it.
fn read_matrix(lifetime: ValueLifetime) -> ValueContract {
    ValueContract {
        access: BufferAccess::ReadOnly,
        ..matrix(lifetime)
    }
}

/// One stage, so no production has anything to fuse, overlap, or cut.
fn single_stage_graph() -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let input = graph
        .add_external_value("in_a", matrix(ValueLifetime::Invocation))
        .expect("external input");
    graph
        .add_node(
            "n0",
            copy_program("in_a", "out"),
            vec![GraphInput {
                buffer: "in_a".into(),
                value: input,
                contract: matrix(ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "out".into(),
                name: "out".into(),
                contract: matrix(ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .expect("only stage");
    graph
}

/// A chain, an independent arm, and a fan-in.
///
/// `n0 -> n1` is a single-consumer chain the grammar may fuse, `n2` is an
/// independent arm, and `n3` reads both `n1` and `n2`, so one phase carries two
/// predecessors and the joining productions have an operand.
fn joined_graph() -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let in_a = graph
        .add_external_value("in_a", matrix(ValueLifetime::Invocation))
        .expect("external input");
    let in_b = graph
        .add_external_value("in_b", matrix(ValueLifetime::Invocation))
        .expect("external input");
    let (_, mid_a) = graph
        .add_node(
            "n0",
            copy_program("in_a", "mid_a"),
            vec![GraphInput {
                buffer: "in_a".into(),
                value: in_a,
                contract: matrix(ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "mid_a".into(),
                name: "mid_a".into(),
                contract: matrix(ValueLifetime::Invocation),
                retained_successor_of: None,
            }],
        )
        .expect("first stage");
    let (_, mid_b) = graph
        .add_node(
            "n1",
            copy_program("mid_a", "mid_b"),
            vec![GraphInput {
                buffer: "mid_a".into(),
                value: mid_a[0],
                contract: matrix(ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "mid_b".into(),
                name: "mid_b".into(),
                contract: matrix(ValueLifetime::Invocation),
                retained_successor_of: None,
            }],
        )
        .expect("second stage");
    let (_, mid_c) = graph
        .add_node(
            "n2",
            copy_program("in_b", "mid_c"),
            vec![GraphInput {
                buffer: "in_b".into(),
                value: in_b,
                contract: matrix(ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "mid_c".into(),
                name: "mid_c".into(),
                contract: matrix(ValueLifetime::Invocation),
                retained_successor_of: None,
            }],
        )
        .expect("independent arm");
    graph
        .add_node(
            "n3",
            add_program("mid_b", "mid_c", "out"),
            vec![
                GraphInput {
                    buffer: "mid_b".into(),
                    value: mid_b[0],
                    contract: read_matrix(ValueLifetime::Invocation),
                },
                GraphInput {
                    buffer: "mid_c".into(),
                    value: mid_c[0],
                    contract: read_matrix(ValueLifetime::Invocation),
                },
            ],
            vec![GraphOutput {
                buffer: "out".into(),
                name: "out".into(),
                contract: matrix(ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .expect("fan-in stage");
    graph
}

fn facts() -> ExternalFacts {
    ExternalFacts::new(
        Digest([0x5a; 32]),
        BTreeMap::from([("rows".into(), 64), ("cols".into(), 64)]),
    )
}

/// A device that grants every capability the grammar can ask an operand for.
fn rich_device() -> DeviceFacts {
    DeviceFacts::new(
        BackendCapabilities {
            supports_subgroup_ops: true,
            ..BackendCapabilities::default()
        },
        256,
    )
    .with_occupancy(128, 64 * 1024)
    .with_compute_units(8)
    .with_concurrent_queues(4)
    .with_spatial_partitioning(true)
    .with_cooperative_launch(true)
    .with_subgroup_size(32)
    .with_launch_costs(4224, 1000)
    .with_bandwidth_facts(3788, 3788)
}

/// A device that reports nothing beyond a launch limit.
fn bare_device() -> DeviceFacts {
    DeviceFacts::new(BackendCapabilities::default(), 256)
}

/// A device where issuing one launch dominates every other cost.
fn launch_bound_device() -> DeviceFacts {
    DeviceFacts::new(BackendCapabilities::default(), 256).with_launch_costs(200_000, 1_000)
}

/// A device where launches are nearly free and resident state is scarce.
fn occupancy_bound_device() -> DeviceFacts {
    DeviceFacts::new(BackendCapabilities::default(), 256)
        .with_launch_costs(1, 1)
        .with_occupancy(1, 1)
        .with_bandwidth_facts(1, 1)
}

/// A device that partitions but guarantees no forward progress.
fn no_progress_device() -> DeviceFacts {
    DeviceFacts::new(BackendCapabilities::default(), 256)
        .with_compute_units(8)
        .with_spatial_partitioning(true)
        .with_launch_costs(4224, 1000)
}

fn budget() -> SearchBudget {
    SearchBudget::new(512, 200_000, 4, 0, 1_000_000_000)
}

fn compiled(device: DeviceFacts, budget: SearchBudget) -> Artifact {
    let request = CompileRequest::new(joined_graph(), facts(), device, budget, 4_000_000)
        .validate()
        .expect("request must validate");
    compile(&request).expect("compilation must succeed")
}

/// WHY: a production the grammar declares and never proposes is a family the
/// compiler claims to search and does not. The variant space comes from
/// `ScheduleProduction::ALL` at run time, so a production added without an
/// expansion turns this red instead of going stale in silence.
#[test]
fn every_production_derives_a_candidate() {
    let artifact = compiled(rich_device(), budget());
    let certificate = &artifact.selected_plan().certificate;

    assert_eq!(certificate.grammar_version, SCHEDULE_GRAMMAR_VERSION);
    for production in ScheduleProduction::ALL {
        assert!(
            certificate.derived_by(*production) > 0,
            "production {} derived no candidate",
            production.code()
        );
    }
    assert!(
        certificate.depth >= 2,
        "a fused phase is reachable only at depth two, got depth {}",
        certificate.depth
    );
}

/// WHY: every schedule transform must belong to exactly one production. The
/// match inside `deriving` is exhaustive, so a new transform variant cannot
/// compile until a production derives it; this test proves the mapping reaches
/// every production rather than collapsing several onto one.
#[test]
fn every_transform_maps_onto_its_own_production() {
    let phase = SchedulePhaseId(0);
    let axis = ScheduleAxis {
        region: 0,
        axis: 0,
        extent: 64,
    };
    let transforms = [
        ScheduleTransform::Fuse {
            phases: vec![SchedulePhaseId(0), SchedulePhaseId(1)],
        },
        ScheduleTransform::PhaseFission {
            phase,
            split_after_region: 0,
        },
        ScheduleTransform::SetWorkgroup {
            phase,
            shape: [64, 1, 1],
        },
        ScheduleTransform::SpatialPartition {
            phase,
            partitions: 2,
            level: MappingLevel::ComputeUnitPartition,
        },
        ScheduleTransform::PersistentQueue { phase, capacity: 2 },
        ScheduleTransform::Pipeline {
            producer: SchedulePhaseId(0),
            consumer: SchedulePhaseId(1),
            ring_slots: 2,
            roles: vec![
                PipelineRoleGroup {
                    role: PipelineRole::Producer,
                    workers: 1,
                },
                PipelineRoleGroup {
                    role: PipelineRole::Consumer,
                    workers: 1,
                },
            ],
        },
        ScheduleTransform::DispatchCut {
            before: SchedulePhaseId(0),
            after: SchedulePhaseId(1),
        },
        ScheduleTransform::AsymmetricJoin {
            producers: vec![SchedulePhaseId(0), SchedulePhaseId(1)],
            consumer: SchedulePhaseId(2),
        },
        ScheduleTransform::Synchronize {
            phases: vec![phase],
            scope: SynchronizationScope::Workgroup,
        },
        ScheduleTransform::PlaceMemory {
            phase,
            value: 0,
            placement: MemoryPlacement::Workgroup,
            bytes: 256,
        },
        ScheduleTransform::Prefetch {
            phase,
            value: 0,
            distance: 1,
            bytes: 256,
        },
        ScheduleTransform::Recompute {
            phase,
            values: vec![0],
        },
        ScheduleTransform::Tile {
            phase,
            tiles: vec![(axis, 2)],
        },
        ScheduleTransform::Split {
            phase,
            axis,
            factor: 2,
        },
        ScheduleTransform::Vectorize {
            phase,
            axis,
            width: 2,
        },
        ScheduleTransform::Map {
            phase,
            axis,
            level: MappingLevel::Lane,
        },
        ScheduleTransform::Reorder {
            phase,
            axes: vec![axis],
        },
    ];

    let mut derived = transforms
        .iter()
        .map(ScheduleProduction::deriving)
        .collect::<Vec<_>>();
    derived.sort_unstable();
    derived.dedup();
    let mut all = ScheduleProduction::ALL.to_vec();
    all.sort_unstable();
    assert_eq!(
        derived, all,
        "one transform per production, and one production per transform"
    );
}

/// WHY: a candidate the compiler cannot execute must disappear with a reason a
/// reader can act on, and the reason must be recorded against the family that
/// proposed it. A reason counted across the whole certificate would let one
/// production's check answer for another, so every pair below names the
/// production and the facts that reach its elimination.
#[test]
fn an_eliminated_family_records_a_stable_reason() {
    let artifact = compiled(bare_device(), budget());
    let certificate = &artifact.selected_plan().certificate;
    let resident = compiled(no_progress_device(), budget());
    let resident_certificate = &resident.selected_plan().certificate;

    for (certificate, production, reason, why) in [
        (
            certificate,
            ScheduleProduction::SpatialPartition,
            PruneReason::TargetFacts,
            "a device reporting no partitioning cannot run a partitioned phase",
        ),
        (
            certificate,
            ScheduleProduction::AxisMapping,
            PruneReason::TargetFacts,
            "a device reporting no partition level cannot map an axis onto one",
        ),
        (
            resident_certificate,
            ScheduleProduction::PersistentQueue,
            PruneReason::Progress,
            "a device without cooperative launch guarantees no forward progress",
        ),
        (
            certificate,
            ScheduleProduction::Recomputation,
            PruneReason::Representation,
            "an artifact assigns one node to exactly one fusion group",
        ),
    ] {
        let count = certificate
            .pruned
            .iter()
            .filter(|family| family.production == production && family.reason == reason)
            .fold(0, |total: u32, family| total + family.count);
        assert!(
            count > 0,
            "{} must be eliminated for {}: {why}",
            production.code(),
            reason.code()
        );
    }
    for family in &certificate.pruned {
        assert!(
            family.count > 0,
            "family {} records no eliminated candidate",
            family.production.code()
        );
        assert!(
            !family.reason.code().is_empty(),
            "family {} records no reason code",
            family.production.code()
        );
    }
}

/// WHY: the artifact must carry the derivation that produced it, and the
/// schedule must replay from its immutable source phases. A recorded plan whose
/// derivation names a transform the schedule never applied would be a claim with
/// no proof behind it.
#[test]
fn the_selected_plan_carries_a_replayable_derivation() {
    let artifact = compiled(rich_device(), budget());
    let plan = artifact.selected_plan();

    plan.schedule
        .validate()
        .expect("the selected schedule must replay from its source phases");
    for step in &plan.derivation {
        for transform in &step.transforms {
            assert!(
                plan.schedule
                    .transforms
                    .iter()
                    .any(|record| record.transform == *transform),
                "derivation step {} names a transform the schedule did not apply",
                step.production.code()
            );
            assert_eq!(
                ScheduleProduction::deriving(transform),
                step.production,
                "a step must record the production that derives its transform"
            );
        }
    }
}

/// WHY: one graph and two devices must be able to produce structurally different
/// kernel organizations, not the same plan with different numbers. That is the
/// difference between a compiler and a catalog. The two devices differ only in
/// measured facts: one pays for every launch, the other pays for resident state.
#[test]
fn different_device_facts_derive_different_plans() {
    let launch_bound = compiled(launch_bound_device(), budget());
    let occupancy_bound = compiled(occupancy_bound_device(), budget());

    let launch_plan = launch_bound.selected_plan();
    let occupancy_plan = occupancy_bound.selected_plan();
    assert!(
        launch_plan.fusion.len() < occupancy_plan.fusion.len(),
        "a device that pays for every launch must organize the graph into fewer \
         generated kernels than one that pays for resident state: {} against {}",
        launch_plan.fusion.len(),
        occupancy_plan.fusion.len()
    );
    assert_ne!(
        launch_plan.derivation, occupancy_plan.derivation,
        "two devices must derive different organizations, not the same one"
    );
    let rich_admitted = rich_device_admissions();
    let bare_admitted = compiled(bare_device(), budget())
        .selected_plan()
        .certificate
        .admitted_total();
    assert!(
        rich_admitted > bare_admitted,
        "a device that grants more capability must admit more candidates: \
         {rich_admitted} against {bare_admitted}"
    );
}

/// Candidates a fully capable device admits for the shared graph.
fn rich_device_admissions() -> u32 {
    compiled(rich_device(), budget())
        .selected_plan()
        .certificate
        .admitted_total()
}

/// WHY: an exhausted budget must return the best proved candidate and say that
/// it stopped early. Silently returning the baseline without recording the bound
/// would make an unfinished search look like a finished one.
#[test]
fn an_exhausted_budget_records_the_bound_it_hit() {
    let artifact = compiled(rich_device(), SearchBudget::new(3, 8, 1, 0, 1_000_000));
    let plan = artifact.selected_plan();

    assert!(
        plan.certificate.budget_exhausted,
        "a search stopped by its bound must record that it was"
    );
    assert!(plan.candidates_explored >= 1);
    plan.schedule
        .validate()
        .expect("the plan a bounded search returns is still proved");
}

/// WHY: a certificate that changed between two identical compilations would make
/// the artifact digest a property of the run instead of the request.
#[test]
fn one_request_records_one_certificate() {
    let first = compiled(rich_device(), budget());
    let second = compiled(rich_device(), budget());

    assert_eq!(
        first.selected_plan().certificate,
        second.selected_plan().certificate
    );
    assert_eq!(
        first.selected_plan().derivation,
        second.selected_plan().derivation
    );
    assert_eq!(first.digest(), second.digest());
}

/// WHY: the unfused, unspecialized baseline has to stay in the candidate set and
/// win when nothing beats it, so an accepted production is always one that paid
/// for itself. On a one-node graph priced only by its single launch, no
/// production can improve the plan, and the proved bound says so rather than the
/// search quietly returning the first thing it derived.
#[test]
fn the_baseline_wins_when_no_production_pays() {
    let request = CompileRequest::new(
        single_stage_graph(),
        facts(),
        bare_device(),
        budget(),
        4_000_000,
    )
    .validate()
    .expect("request must validate");
    let artifact = compile(&request).expect("compilation must succeed");
    let plan = artifact.selected_plan();

    assert_eq!(plan.fusion.len(), 1, "one node is one generated kernel");
    assert!(
        plan.derivation.is_empty(),
        "the winning plan applied {} productions over the baseline",
        plan.derivation.len()
    );
    assert!(
        plan.certificate.pruned_for(PruneReason::ObjectiveDominated) > 0,
        "a candidate whose proved bound cannot beat the incumbent must be \
         eliminated against the objective, not ranked behind it"
    );
}
