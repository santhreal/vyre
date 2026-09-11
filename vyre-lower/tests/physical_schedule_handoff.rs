//! Contracts for the one handoff from a selected schedule to a target.
//!
//! WHY: neutral lowering used to pass a workgroup shape and drop every other
//! selected fact, so a backend that needed logical coverage, a vector width, an
//! axis mapping level, a role assignment, a ring depth or a resource ceiling
//! either rediscovered it from the op stream or chose its own. A chosen value is
//! a schedule the search never selected and never priced. These tests defend the
//! projection that replaced that gap: it carries every frozen fact, it refuses
//! to state a fact the schedule did not select, it is absent rather than
//! defaulted when no schedule was selected, and it cannot disagree with the
//! dispatch geometry the same lowering produced.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use vyre_foundation::ir::Program;
use vyre_foundation::schedule::{
    MappingLevel, PipelineRole, SchedulePhaseId, SelectedSchedule, SynchronizationScope,
    SCHEDULE_IR_VERSION,
};
use vyre_lower::{lower_physical, lower_scheduled, PhysicalSchedule, PHYSICAL_SCHEDULE_VERSION};

/// Two-phase schedule whose first phase carries every projectable fact: an
/// exact shape, a mapped axis, a pipeline it produces into, two synchronization
/// boundaries and a persistent queue.
fn rich_schedule() -> SelectedSchedule {
    vyre_test_support::selected_schedules::richly_transformed_two_phase()
}

/// WHY: the projection is the whole handoff, so a fact it drops is a fact a
/// backend invents. Every selected term of the phase has to arrive, and the
/// serialized key set is compared against the recorded one so a new field turns
/// this test red until someone states what a target may rely on.
#[test]
fn the_projection_carries_every_fact_the_schedule_froze() {
    let schedule = rich_schedule();
    let projected = PhysicalSchedule::project(&schedule, SchedulePhaseId(0))
        .expect("a validated schedule phase must project");

    assert_eq!(projected.version, PHYSICAL_SCHEDULE_VERSION);
    assert_eq!(projected.schedule_version, SCHEDULE_IR_VERSION);
    assert_eq!(projected.logical_identity, schedule.logical_identity);
    assert_eq!(projected.phase, 0);
    assert_eq!(projected.workgroup, [32, 2, 1]);
    assert_eq!(projected.invocations_per_workgroup(), 64);
    assert_eq!(projected.logical_coverage, [1, 1, 1]);
    assert_eq!(projected.vector_width, 1);
    assert_eq!(
        projected
            .mappings
            .iter()
            .map(|mapping| mapping.level)
            .collect::<Vec<_>>(),
        vec![MappingLevel::Subgroup],
        "the level an axis was mapped to is a selected fact"
    );
    assert_eq!(projected.ring_slots, 3);
    assert!(projected.is_pipelined());
    assert_eq!(projected.role_workers(PipelineRole::Producer), 1);
    assert_eq!(projected.role_workers(PipelineRole::Consumer), 2);
    assert_eq!(
        projected.barriers.len(),
        2,
        "both boundaries covering the phase must arrive: {:?}",
        projected.barriers
    );
    assert_eq!(projected.barriers[0].scope, SynchronizationScope::Workgroup);
    assert_eq!(projected.barriers[1].scope, SynchronizationScope::Device);
    assert!(
        !projected.barriers[0].parity() && projected.barriers[1].parity(),
        "barrier parity has to alternate so a target can double-buffer state"
    );
    assert_eq!(projected.queue_capacity, 128);
    assert!(projected.is_persistent());
    assert_eq!(
        projected.resources, schedule.phases[0].resources,
        "the checked resource ceiling is what a launch tactic is bounded by"
    );

    let encoded = serde_json::to_value(&projected).expect("the projection must serialize");
    let fields = encoded
        .as_object()
        .expect("the projection is a record")
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    let recorded = [
        "barriers",
        "logical_coverage",
        "logical_identity",
        "mappings",
        "phase",
        "queue_capacity",
        "resources",
        "ring_slots",
        "roles",
        "schedule_version",
        "vector_width",
        "version",
        "workgroup",
    ]
    .into_iter()
    .map(String::from)
    .collect::<BTreeSet<_>>();
    assert_eq!(
        fields, recorded,
        "a projected field is a promise to every target: state it here and in the projection docs"
    );
}

/// WHY: a phase the artifact names but the schedule does not contain is a
/// mismatch between selection and emission. Projecting a workgroup shape from a
/// neighbouring phase would emit a launch nothing selected.
#[test]
fn projecting_a_phase_the_schedule_does_not_contain_fails() {
    let schedule = rich_schedule();
    let error = PhysicalSchedule::project(&schedule, SchedulePhaseId(9))
        .expect_err("an absent phase must not project");

    assert!(
        error.message().contains("no phase 9"),
        "the failure must name the phase: {}",
        error.message()
    );
    assert!(error.message().contains("Fix:"));
}

/// WHY: every rule here is a fact a backend is allowed to rely on without
/// checking. A projection that states a zero extent, a zero vector width, or a
/// pipeline with a missing side is worse than an absent projection, because a
/// target reads it and emits a launch nothing selected. Each rule is broken on
/// its own so no single check can cover for another.
#[test]
fn a_projection_that_states_an_unselected_fact_is_refused() {
    let valid = PhysicalSchedule::project(&rich_schedule(), SchedulePhaseId(0))
        .expect("the fixture must project");
    valid.validate().expect("the projected facts must check");

    let cases: Vec<(&str, &str, Box<dyn Fn(&mut PhysicalSchedule)>)> = vec![
        (
            "version",
            "is not",
            Box::new(|projection| projection.version = PHYSICAL_SCHEDULE_VERSION + 1),
        ),
        (
            "workgroup",
            "zero extent",
            Box::new(|projection| projection.workgroup = [32, 0, 1]),
        ),
        (
            "coverage",
            "zero extent",
            Box::new(|projection| projection.logical_coverage = [1, 1, 0]),
        ),
        (
            "vector width",
            "vector width is zero",
            Box::new(|projection| projection.vector_width = 0),
        ),
        (
            "ring without roles",
            "role groups",
            Box::new(|projection| projection.roles.clear()),
        ),
        (
            "roles without a ring",
            "role groups",
            Box::new(|projection| projection.ring_slots = 0),
        ),
        (
            "pipeline missing a consumer",
            "no producer or no consumer",
            Box::new(|projection| {
                projection
                    .roles
                    .retain(|group| group.role != PipelineRole::Consumer);
            }),
        ),
    ];

    for (why, expected, mutate) in cases {
        let mut projection = valid.clone();
        mutate(&mut projection);
        let error = projection
            .validate()
            .expect_err(&format!("{why} must be refused"));
        assert!(
            error.message().contains(expected),
            "{why} must report `{expected}`, got: {}",
            error.message()
        );
        assert!(error.message().contains("Fix:"), "{why} must state a fix");
    }
}

/// WHY: the projection reaches a target through the verified lowering boundary
/// and nowhere else. A schedule-free program states no schedule at all, which a
/// target must be able to tell from a schedule that selected defaults, and a
/// lowering whose dispatch disagrees with the frozen workgroup is a second
/// geometry authority.
#[test]
fn the_lowering_boundary_attaches_the_projection_and_nothing_else_states_one() {
    let program = Program::wrapped(Vec::new(), [64, 1, 1], Vec::new());

    let unscheduled = lower_physical(&program).expect("a physical program must lower");
    assert!(
        unscheduled.schedule().is_none(),
        "a program lowered without a selected schedule must claim no frozen facts"
    );
    assert_eq!(unscheduled.descriptor().dispatch.workgroup_size, [64, 1, 1]);

    let schedule = rich_schedule();
    let scheduled = lower_scheduled(&program, &schedule, SchedulePhaseId(0))
        .expect("a validated schedule phase must lower");
    let projected = scheduled
        .schedule()
        .expect("a scheduled lowering must carry its frozen facts");

    assert_eq!(
        projected.workgroup,
        scheduled.descriptor().dispatch.workgroup_size,
        "the frozen workgroup and the lowered dispatch are one fact, not two"
    );
    assert_eq!(projected.phase, 0);
    assert_eq!(projected.ring_slots, 3);
}
