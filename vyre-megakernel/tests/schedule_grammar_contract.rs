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

use vyre_foundation::schedule::{
    MappingLevel, MemoryPlacement, PipelineRole, PipelineRoleGroup, ScheduleAxis, SchedulePhaseId,
    ScheduleTransform, SynchronizationScope,
};
use vyre_megakernel::{
    compile, is_required_schedule_unreachable, CompileObjective, CompileRequest, ObjectiveMetric,
    PruneReason, RequiredSchedule, ScheduleProduction, SearchBudget, REQUIRED_SCHEDULE_UNREACHABLE,
    SCHEDULE_GRAMMAR_VERSION,
};

#[path = "support/search_fixtures.rs"]
mod search_fixtures;

use search_fixtures::{
    bare_device, budget, compiled, facts, joined_graph, launch_bound_device, no_progress_device,
    occupancy_bound_device, refused_field, rich_device, single_stage_graph,
};

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
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 4_000_000),
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

/// The fixture graph and facts, compiled under one required schedule family.
fn compiled_requiring(
    required: RequiredSchedule,
) -> Result<vyre_megakernel::Artifact, vyre_megakernel::CompileError> {
    let request = CompileRequest::new(
        joined_graph(),
        facts(),
        rich_device(),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 4_000_000),
    )
    .requiring_schedule(required)
    .validate()
    .expect("request must validate");
    compile(&request)
}

/// WHY: conformance has to run one semantic graph under a named schedule family,
/// not under whichever family the objective ranked first. Without a requirement
/// the fused, tiled, partitioned and resident plans are all derived and all
/// discarded, so a case that claims to check a tiled schedule checks the
/// baseline. This ranges over every family the grammar declares, so a production
/// added to `ScheduleProduction::ALL` is required here before it can go
/// unexercised in silence.
#[test]
fn a_required_family_that_the_graph_reaches_is_the_family_selected() {
    let mut reached = 0_u32;
    for production in ScheduleProduction::ALL {
        let required = RequiredSchedule::Production(*production);
        let artifact = match compiled_requiring(required) {
            Ok(artifact) => artifact,
            Err(error) if is_required_schedule_unreachable(&error) => continue,
            Err(error) => panic!(
                "requiring family {} failed for a reason other than \
                 unreachability: {error}",
                production.code()
            ),
        };
        reached += 1;
        let plan = artifact.selected_plan();
        assert!(
            plan.derivation
                .iter()
                .any(|step| step.production == *production),
            "the plan selected under required family {} applied {:?}",
            production.code(),
            plan.derivation
                .iter()
                .map(|step| step.production)
                .collect::<Vec<_>>()
        );
    }
    assert!(
        reached >= 4,
        "the fixture graph reached only {reached} of the declared families; a \
         graph that reaches almost none proves nothing about the requirement"
    );
}

/// WHY: the baseline is the family a conformance case starts from, and it is the
/// one family stated by the absence of a production rather than the presence of
/// one. A requirement that read an empty derivation as unconstrained would
/// silently accept any plan for the single-invocation case.
#[test]
fn the_required_baseline_selects_a_plan_that_applied_no_production() {
    let artifact = compiled_requiring(RequiredSchedule::Baseline)
        .expect("the baseline is always in the candidate set");
    let plan = artifact.selected_plan();

    assert!(
        plan.derivation.is_empty(),
        "the plan selected under a required baseline applied {} production(s)",
        plan.derivation.len()
    );
    assert!(
        plan.certificate
            .pruned_for(PruneReason::ScheduleRequirement)
            > 0,
        "every derived candidate outside the required family must be eliminated \
         with its own reason, so a reader can tell a constrained selection from \
         an unconstrained one"
    );
}

/// WHY: a family no legal plan reaches is the one case where a fallback would be
/// invisible. A one-node graph cannot be fused; served the baseline instead, a
/// conformance case would record a passing fused schedule it never ran. The
/// refusal carries its own code so a caller iterating families tells it apart
/// from a compile that failed.
#[test]
fn a_required_family_no_legal_plan_reaches_is_refused_and_not_replaced() {
    let request = CompileRequest::new(
        single_stage_graph(),
        facts(),
        bare_device(),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 4_000_000),
    )
    .requiring_schedule(RequiredSchedule::Production(ScheduleProduction::Fusion))
    .validate()
    .expect("request must validate");

    let error =
        compile(&request).expect_err("a one-node graph has no producer-consumer pair to contract");
    assert!(
        is_required_schedule_unreachable(&error),
        "the refusal must carry {REQUIRED_SCHEDULE_UNREACHABLE}, got {}",
        error.diagnostic.code
    );
    assert_eq!(
        refused_field(&error),
        Some("request.required_schedule"),
        "the refusal must name the field the caller stated"
    );
}

/// WHY: a requirement must narrow what may be selected and never what may be
/// considered. If it pruned before derivation, the certificate would stop
/// recording the families it eliminated and the artifact would no longer state
/// why a plan was legal, which is what reproduces a selection.
#[test]
fn a_requirement_narrows_selection_without_narrowing_the_search() {
    let unconstrained = compiled(rich_device(), budget());
    let constrained = compiled_requiring(RequiredSchedule::Baseline)
        .expect("the baseline is always in the candidate set");

    for production in ScheduleProduction::ALL {
        assert_eq!(
            constrained
                .selected_plan()
                .certificate
                .derived_by(*production),
            unconstrained
                .selected_plan()
                .certificate
                .derived_by(*production),
            "production {} derived a different number of candidates under a \
             requirement",
            production.code()
        );
    }
}
