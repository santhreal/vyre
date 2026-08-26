//! Contracts for the bounded multi-fidelity evaluation ladder.
//!
//! WHY: ranking used to end the search. The analytic winner was assembled and
//! returned, and the measured path handed one plan to the target compiler, so a
//! plan the target could not build failed the whole compilation and a plan the
//! device ran slower than its rival still won. These tests defend the ladder
//! that replaced it: emission is a fidelity level whose rejection eliminates a
//! candidate family with a stable reason and lets the search continue, device
//! measurement decides among what emitted, the counts spent at each level are
//! recorded, and a selection nothing measured is never reported as measured.

#![forbid(unsafe_code)]

use std::sync::Mutex;

use vyre_megakernel::{
    compile_measured, compile_selected_modules, Artifact, CompileRequest, DeviceFacts,
    EmittedTargetModule, FinalistEvaluator, PlanMeasurement, PruneReason, SearchBudget,
    TargetCompileError, TargetCompiler, TargetPayload, TargetPayloadFormat, TargetProfile,
};

#[path = "support/search_fixtures.rs"]
mod search_fixtures;

use search_fixtures::{facts, joined_graph, launch_bound_device};

/// Launches the fixture times against every surviving finalist.
const LAUNCHES: u32 = 3;

/// Finalists the fixture budget lets emission attempt.
const FINALISTS: u32 = 4;

/// Which finalists the fixture target refuses to build.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Unbuildable {
    /// Every module the target is asked for is built.
    None,
    /// The plan the analytic model ranked first is refused, which is the class
    /// of target whose emitter rejects one organization it cannot express.
    FirstRanked,
    /// Nothing the target is asked for is built.
    Every,
}

/// Device time the fixture reports for a plan of `groups` generated kernels.
///
/// More generated kernels measure faster, which is the opposite of how the
/// launch-paying device ranks them, so a plan can only win here by being
/// measured.
fn device_time(groups: usize) -> u64 {
    100_000 - 1_000 * groups as u64
}

/// One emission attempt the fixture target answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Attempt {
    /// Generated kernels in the artifact the target was asked to build.
    groups: usize,
    /// Whether the target built it.
    built: bool,
}

struct LadderCompiler {
    format: TargetPayloadFormat,
    profile: TargetProfile,
    unbuildable: Unbuildable,
    attempts: Mutex<Vec<Attempt>>,
}

impl LadderCompiler {
    fn new(unbuildable: Unbuildable) -> Self {
        Self {
            format: TargetPayloadFormat::new("test.ladder-target", 1)
                .expect("fixture format must be valid"),
            profile: TargetProfile::new("test.ladder-target", 1, [256, 1, 1], 256, 64 * 1024, 0)
                .expect("fixture profile must be valid"),
            unbuildable,
            attempts: Mutex::new(Vec::new()),
        }
    }
}

impl TargetCompiler for LadderCompiler {
    fn format(&self) -> &TargetPayloadFormat {
        &self.format
    }

    fn profile(&self) -> &TargetProfile {
        &self.profile
    }

    fn compile(&self, artifact: &Artifact) -> Result<TargetPayload, TargetCompileError> {
        let groups = artifact.selected_plan().fusion.len();
        let mut attempts = self.attempts.lock().expect("fixture state is not poisoned");
        let refused = match self.unbuildable {
            Unbuildable::None => false,
            Unbuildable::FirstRanked => attempts.is_empty(),
            Unbuildable::Every => true,
        };
        attempts.push(Attempt {
            groups,
            built: !refused,
        });
        drop(attempts);
        if refused {
            return Err(TargetCompileError::Unsupported(format!(
                "fixture target refuses this plan, which organizes the graph into \
                 {groups} generated kernels"
            )));
        }
        compile_selected_modules(
            artifact,
            self.format.clone(),
            self.profile.clone(),
            |selected, _| {
                Ok(EmittedTargetModule {
                    entry_point: format!("group{}", selected.group.0),
                    resource_bindings: selected.canonical_bindings.clone(),
                    bytes: vec![0x0f, u8::try_from(selected.group.0).unwrap_or(u8::MAX)],
                })
            },
        )
    }
}

struct LadderEvaluator {
    compiler: LadderCompiler,
    /// Generated-kernel counts measured, in the order they were launched.
    measured: Mutex<Vec<usize>>,
}

impl LadderEvaluator {
    fn new(unbuildable: Unbuildable) -> Self {
        Self {
            compiler: LadderCompiler::new(unbuildable),
            measured: Mutex::new(Vec::new()),
        }
    }

    /// Emission attempts the target answered.
    fn attempts(&self) -> Vec<Attempt> {
        self.compiler
            .attempts
            .lock()
            .expect("fixture state is not poisoned")
            .clone()
    }

    /// Emission attempts the target refused.
    fn refused(&self) -> usize {
        self.attempts()
            .iter()
            .filter(|attempt| !attempt.built)
            .count()
    }

    /// Generated-kernel counts measured, in launch order.
    fn measured(&self) -> Vec<usize> {
        self.measured
            .lock()
            .expect("fixture state is not poisoned")
            .clone()
    }
}

impl FinalistEvaluator for LadderEvaluator {
    fn target_compiler(&self) -> &dyn TargetCompiler {
        &self.compiler
    }

    fn measure(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<u64, TargetCompileError> {
        assert_eq!(
            payload.neutral_artifact(),
            artifact.digest(),
            "a finalist must be measured against the payload built for it"
        );
        let groups = artifact.selected_plan().fusion.len();
        self.measured
            .lock()
            .expect("fixture state is not poisoned")
            .push(groups);
        Ok(device_time(groups))
    }
}

fn budget(measurements: u32) -> SearchBudget {
    SearchBudget::new(512, 200_000, FINALISTS, measurements, 60_000_000_000)
}

/// The launch-paying device with timestamped launches, so measurement runs.
fn timed_device() -> DeviceFacts {
    launch_bound_device().with_device_timestamps(true)
}

fn measured_compile(
    device: DeviceFacts,
    budget: SearchBudget,
    evaluator: &LadderEvaluator,
) -> Result<Artifact, vyre_megakernel::CompileError> {
    let request = CompileRequest::new(joined_graph(), facts(), device, budget, 4_000_000)
        .validate()
        .expect("request must validate");
    compile_measured(&request, evaluator)
}

/// WHY: emission is a fidelity level, not a last step that can only succeed. A
/// finalist the target compiler refuses has to be eliminated with a reason and
/// the ladder has to continue, because a plan ranked behind it may be
/// buildable. The fixture target refuses the plan the analytic model ranked
/// first, so the winner is a plan only the levels above emission could reach.
#[test]
fn an_unbuildable_finalist_is_eliminated_and_the_ladder_continues() {
    let evaluator = LadderEvaluator::new(Unbuildable::FirstRanked);
    let artifact = measured_compile(timed_device(), budget(LAUNCHES), &evaluator)
        .expect("a buildable finalist must still win the compilation");
    let plan = artifact.selected_plan();
    let attempts = evaluator.attempts();
    let refused = evaluator.refused();
    let measured = evaluator.measured();

    assert_eq!(refused, 1, "the fixture refuses one finalist: {attempts:?}");
    assert_eq!(
        attempts.len(),
        FINALISTS as usize,
        "the refusal must not stop emission of the plans ranked behind it"
    );
    assert_eq!(
        plan.fusion.len(),
        measured
            .iter()
            .copied()
            .max()
            .expect("a survivor must be measured"),
        "the winner must be one of the finalists that emitted and was measured"
    );
    assert_eq!(
        plan.certificate.pruned_for(PruneReason::Emission),
        u32::try_from(refused).expect("refusal count fits u32"),
        "every refused finalist must be eliminated for emission"
    );
    assert_eq!(
        plan.search_work.target_compilations,
        u32::try_from(attempts.len()).expect("attempt count fits u32"),
        "the recorded target compilations must be the emissions performed"
    );
    assert_eq!(
        plan.search_work.measurements,
        u32::try_from(evaluator.measured().len()).expect("measurement count fits u32"),
        "the recorded measurements must be the launches performed"
    );
    assert_eq!(
        evaluator.measured().len(),
        (attempts.len() - refused) * LAUNCHES as usize,
        "only the finalists that emitted may be measured"
    );
    assert!(
        matches!(
            plan.measurement,
            PlanMeasurement::Measured {
                launches: LAUNCHES,
                ..
            }
        ),
        "a measured selection must record its launches, got {:?}",
        plan.measurement
    );
    assert!(
        plan.certificate
            .pruned
            .windows(2)
            .all(|pair| pair[0] <= pair[1]),
        "eliminations recorded by the ladder must stay in canonical order"
    );
}

/// WHY: a compilation whose finalists the target all refuse must fail with the
/// refusal, not return the analytic winner as though it were buildable. The
/// diagnostic names the finalist that was refused last, so the caller can see
/// how far the ladder got.
#[test]
fn a_compilation_no_finalist_can_build_fails_with_the_refusal() {
    let evaluator = LadderEvaluator::new(Unbuildable::Every);
    let error = measured_compile(timed_device(), budget(LAUNCHES), &evaluator)
        .expect_err("a plan the target cannot build must not be returned");
    let attempts = evaluator.attempts();

    assert_eq!(attempts.len(), FINALISTS as usize);
    assert!(
        evaluator.measured().is_empty(),
        "nothing emitted to measure"
    );
    assert_eq!(error.diagnostic.code.as_str(), "MKC026_FINALIST_EVALUATION");
    assert_eq!(
        error
            .diagnostic
            .location
            .as_ref()
            .and_then(|location| location.path.as_deref()),
        Some(format!("search.finalists[{}]", attempts.len() - 1).as_str())
    );
    assert!(error
        .to_string()
        .contains("fixture target refuses this plan"));
}

/// WHY: measurement has to be able to overturn the ranking, or the level is
/// decoration. The fixture device times a plan faster the more kernels it
/// generates, which is the reverse of how the launch-paying device ranks them,
/// so the winner is a finalist the analytic model placed behind another.
#[test]
fn device_measurement_selects_a_finalist_the_ranking_placed_behind() {
    let evaluator = LadderEvaluator::new(Unbuildable::None);
    let artifact = measured_compile(timed_device(), budget(LAUNCHES), &evaluator)
        .expect("every finalist builds, so one must win");
    let plan = artifact.selected_plan();
    let measured = evaluator.measured();
    let fastest = *measured
        .iter()
        .max()
        .expect("the ladder must measure at least one finalist");

    assert_eq!(
        plan.fusion.len(),
        fastest,
        "the winner must be the finalist with the lowest median device time"
    );
    assert_eq!(
        plan.measurement,
        PlanMeasurement::Measured {
            launches: LAUNCHES,
            median_ns: device_time(fastest),
        }
    );
    assert_ne!(
        measured.first().copied(),
        Some(fastest),
        "the first-ranked finalist already being fastest would prove nothing \
         about measurement deciding: measured {measured:?}"
    );
}

/// WHY: a plan nothing measured must never be recorded as measured. Both ways a
/// measurement can be unavailable are stated in the artifact, and neither spends
/// an emission or a launch, so a caller reading the plan can tell an analytic
/// selection from a timed one.
#[test]
fn a_selection_no_device_timed_is_recorded_as_unmeasured() {
    for (device, budget, expected, why) in [
        (
            timed_device(),
            budget(0),
            PlanMeasurement::Unbudgeted,
            "a zero measurement budget times nothing",
        ),
        (
            launch_bound_device(),
            budget(LAUNCHES),
            PlanMeasurement::UntimedDevice,
            "a device with no launch timestamps reports no device time",
        ),
    ] {
        let evaluator = LadderEvaluator::new(Unbuildable::None);
        let artifact = measured_compile(device, budget, &evaluator)
            .expect("the analytic winner is still a compiled plan");
        let plan = artifact.selected_plan();

        assert_eq!(plan.measurement, expected, "{why}");
        assert_eq!(plan.search_work.target_compilations, 0, "{why}");
        assert_eq!(plan.search_work.measurements, 0, "{why}");
        assert!(evaluator.attempts().is_empty(), "{why}");
        assert!(evaluator.measured().is_empty(), "{why}");
    }
}
