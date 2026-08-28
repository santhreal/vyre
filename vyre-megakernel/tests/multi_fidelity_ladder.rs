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

use vyre_megakernel::measure::{DeviceState, MeasurementProtocol, MeasurementRecord};
use vyre_megakernel::{
    compile_measured, compile_selected_modules, Artifact, DeviceFacts, EmittedResources,
    EmittedTargetModule, FinalistEvaluator, PlanMeasurement, PruneReason, SearchBudget,
    TargetCompileError, TargetCompiler, TargetPayload, TargetPayloadFormat, TargetProfile,
};

#[path = "support/search_fixtures.rs"]
mod search_fixtures;

use search_fixtures::{fixture_request, latency_objective, launch_bound_device, validated};

/// Launches the fixture times against every surviving finalist.
const LAUNCHES: u32 = 3;

/// Finalists the fixture budget lets emission attempt.
const FINALISTS: u32 = 4;

/// Samples the fitted protocol counts per candidate under [`LAUNCHES`], which is
/// the launch budget less the warmup the protocol spends first.
fn counted_samples() -> u32 {
    let protocol = MeasurementProtocol::V1.fitted(LAUNCHES);
    protocol.max_rounds * protocol.repetitions_per_round
}

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

/// What the fixture device reports the analytically first-ranked finalist
/// allocated once its entry points were emitted and loaded.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reported {
    /// The device reports nothing, so every finalist keeps its estimate.
    Nothing,
    /// It spills, which is legal and costs traffic the estimate never saw.
    SpillOnRankedFirst,
    /// It allocates more registers than the device has, which no launch can
    /// run.
    OverCeilingOnRankedFirst,
    /// It holds one byte less than the plan requires, which is a device that is
    /// not running the plan the compiler selected.
    ResidentBelowPlan,
    /// It holds exactly the bytes the plan requires.
    ResidentAtPlan,
}

/// Local-memory bytes per invocation the fixture reports for a spilling plan.
/// Priced against the fixture's one byte per nanosecond, this outweighs every
/// launch the plan saves.
const SPILL_BYTES_PER_INVOCATION: u32 = 65_536;

/// Registers per invocation the fixture reports above the device ceiling.
const OVER_CEILING_REGISTERS: u32 = 512;

/// Architectural register ceiling of the fixture device.
const REGISTER_CEILING: u32 = 255;

/// Device time the fixture reports for a plan of `groups` generated kernels.
///
/// More generated kernels measure faster, which is the opposite of how the
/// launch-paying device ranks them, so a plan can only win here by being
/// measured. Each additional kernel is worth 20 percent of the base time, which
/// clears the protocol's equivalence band by a wide margin: a fixture whose
/// candidates differed by less than the band would prove that the band holds,
/// not that measurement can overturn a ranking.
fn device_time(groups: usize) -> u64 {
    100_000 - 20_000 * groups as u64
}

/// Device time for a fixture whose plans are all within the protocol's
/// equivalence band of one another: one percent apart, where the band is two.
fn indistinguishable_device_time(groups: usize) -> u64 {
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
    /// What the fixture device reports for the first plan it is asked about.
    reported: Reported,
    /// Device time the fixture reports for a plan of `groups` kernels.
    timing: fn(usize) -> u64,
    /// Generated-kernel counts whose emitted resources were read, in the order
    /// the compiler asked for them, which is its analytic ranking.
    inspected: Mutex<Vec<usize>>,
    /// Generated-kernel counts measured, in the order they were launched.
    measured: Mutex<Vec<usize>>,
}

impl LadderEvaluator {
    fn new(unbuildable: Unbuildable) -> Self {
        Self {
            compiler: LadderCompiler::new(unbuildable),
            reported: Reported::Nothing,
            timing: device_time,
            inspected: Mutex::new(Vec::new()),
            measured: Mutex::new(Vec::new()),
        }
    }

    /// An evaluator whose device builds every plan and reports `reported`.
    fn reporting(reported: Reported) -> Self {
        Self {
            reported,
            ..Self::new(Unbuildable::None)
        }
    }

    /// An evaluator whose device times every plan within the protocol's
    /// equivalence band of every other.
    fn indistinguishable() -> Self {
        Self {
            timing: indistinguishable_device_time,
            ..Self::new(Unbuildable::None)
        }
    }

    /// Generated-kernel counts whose resources were read, in ranking order.
    fn inspected(&self) -> Vec<usize> {
        self.inspected
            .lock()
            .expect("fixture state is not poisoned")
            .clone()
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

    fn resources(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<Vec<EmittedResources>, TargetCompileError> {
        let groups = artifact.selected_plan().fusion.len();
        let mut inspected = self
            .inspected
            .lock()
            .expect("fixture state is not poisoned");
        let ranked_first = inspected.is_empty();
        inspected.push(groups);
        let record = match self.reported {
            Reported::Nothing => EmittedResources::default(),
            Reported::SpillOnRankedFirst if ranked_first => EmittedResources {
                spill_bytes_per_invocation: SPILL_BYTES_PER_INVOCATION,
                ..EmittedResources::default()
            },
            Reported::OverCeilingOnRankedFirst if ranked_first => EmittedResources {
                registers_per_invocation: OVER_CEILING_REGISTERS,
                ..EmittedResources::default()
            },
            Reported::ResidentBelowPlan => EmittedResources {
                resident_device_bytes: artifact.allocation().aggregate_peak_bytes.saturating_sub(1),
                ..EmittedResources::default()
            },
            Reported::ResidentAtPlan => EmittedResources {
                resident_device_bytes: artifact.allocation().aggregate_peak_bytes,
                ..EmittedResources::default()
            },
            Reported::SpillOnRankedFirst | Reported::OverCeilingOnRankedFirst => {
                EmittedResources::default()
            }
        };
        Ok(vec![record; payload.entries().len()])
    }

    /// The fixture device exposes no management interface, which is the case the
    /// protocol has to cover with observed drift alone.
    fn device_state(&self) -> DeviceState {
        DeviceState::unreported()
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
        Ok((self.timing)(groups))
    }
}

fn budget(measurements: u32) -> SearchBudget {
    SearchBudget::new(512, 200_000, FINALISTS, measurements, 60_000_000_000)
}

/// The launch-paying device with timestamped launches, so measurement runs.
fn timed_device() -> DeviceFacts {
    launch_bound_device().with_device_timestamps(true)
}

/// The timed device with a byte rate and a register ceiling, so a reported
/// spill has a price and a reported allocation has a limit.
fn priced_device() -> DeviceFacts {
    timed_device()
        .with_bandwidth_facts(1, 1)
        .with_occupancy(64, 32 * 1024)
        .with_architectural_register_limit(REGISTER_CEILING)
}

fn measured_compile(
    device: DeviceFacts,
    budget: SearchBudget,
    evaluator: &LadderEvaluator,
) -> Result<Artifact, vyre_megakernel::CompileError> {
    let request = validated(device, budget, latency_objective());
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
            &plan.measurement,
            PlanMeasurement::Measured(evidence) if evidence.winning_launches() == counted_samples()
        ),
        "a measured selection must record the samples the protocol counted, got {:?}",
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
    let PlanMeasurement::Measured(evidence) = &plan.measurement else {
        panic!(
            "a timed device must record measured evidence: {:?}",
            plan.measurement
        );
    };
    assert_eq!(evidence.winning_launches(), counted_samples());
    assert_eq!(evidence.winning_estimate_ns(), device_time(fastest));
    assert_eq!(
        evidence.protocol,
        MeasurementProtocol::V1.fitted(LAUNCHES),
        "the recorded protocol must be the one the launch budget affords"
    );
    assert_eq!(
        evidence.candidates.len(),
        FINALISTS as usize,
        "every finalist that emitted must retain its samples"
    );
    for candidate in &evidence.candidates {
        assert_eq!(
            candidate.samples.len(),
            counted_samples() as usize,
            "every measured candidate retains exactly the counted samples"
        );
    }
    assert_ne!(
        measured.first().copied(),
        Some(fastest),
        "the first-ranked finalist already being fastest would prove nothing \
         about measurement deciding: measured {measured:?}"
    );
}

/// WHY: the same search on the same device has to select the same artifact
/// twice. When the finalists measure within the protocol's equivalence band of
/// each other, the difference is the device, so the selection stays with the
/// canonical lower-risk finalist the analytic ranking put first. Without the
/// band, whichever candidate the noise favoured that minute would win and the
/// artifact would stop being reproducible.
#[test]
fn finalists_inside_the_equivalence_band_keep_the_canonical_selection() {
    let evaluator = LadderEvaluator::indistinguishable();
    let artifact = measured_compile(timed_device(), budget(LAUNCHES), &evaluator)
        .expect("every finalist builds, so one must win");
    let plan = artifact.selected_plan();
    let measured = evaluator.measured();
    let ranked_first = *measured
        .first()
        .expect("the ladder must measure at least one finalist");
    let fastest = *measured
        .iter()
        .max()
        .expect("the ladder must measure at least one finalist");

    assert_ne!(
        ranked_first, fastest,
        "the fixture must offer a faster finalist, or the band is untested: measured {measured:?}"
    );
    assert_eq!(
        plan.fusion.len(),
        ranked_first,
        "a measured difference inside the band must not move the selection"
    );
    let PlanMeasurement::Measured(evidence) = &plan.measurement else {
        panic!(
            "a timed device must record measured evidence: {:?}",
            plan.measurement
        );
    };
    assert_eq!(
        evidence.winning_estimate_ns(),
        indistinguishable_device_time(ranked_first),
        "the retained estimate must be the winner's own measured time"
    );
    assert!(
        evidence
            .candidates
            .iter()
            .any(|candidate| candidate.estimate.estimate_ns
                == indistinguishable_device_time(fastest)),
        "the faster finalist the band declined must still retain its samples"
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

/// WHY: emission answers the register, spill and shared question the analytic
/// model could only estimate, and a measurement spent in the estimate's order
/// is a measurement spent on the wrong plan first. The fixture device reports
/// that the plan ranking placed first spills, which the estimate never saw, so
/// the reported price has to move it to the back of the measurement queue
/// without dropping it from the ladder.
#[test]
fn a_reported_spill_reorders_the_finalists_before_they_are_measured() {
    let baseline = LadderEvaluator::reporting(Reported::Nothing);
    measured_compile(priced_device(), budget(LAUNCHES), &baseline)
        .expect("every finalist builds, so one must win");
    let ranked_first = baseline
        .inspected()
        .first()
        .copied()
        .expect("emission must read what it built");
    let baseline_order = baseline.measured();

    assert_eq!(
        baseline_order.first().copied(),
        Some(ranked_first),
        "with nothing reported the analytic order must stand: {baseline_order:?}"
    );

    let spilling = LadderEvaluator::reporting(Reported::SpillOnRankedFirst);
    measured_compile(priced_device(), budget(LAUNCHES), &spilling)
        .expect("a spilling plan is legal, so the compilation must still succeed");
    let spilling_order = spilling.measured();

    assert_eq!(
        spilling.inspected().first().copied(),
        Some(ranked_first),
        "both compilations must rank the same plan first analytically"
    );
    assert_eq!(
        spilling_order.last().copied(),
        Some(ranked_first),
        "the plan the device reported spilling must be measured last: {spilling_order:?}"
    );
    assert_eq!(
        distinct(&spilling_order),
        distinct(&baseline_order),
        "re-ranking must reorder the finalists, not drop any of them"
    );
}

/// WHY: a register allocation above what the device has is not a price, it is a
/// launch that cannot run. Such a finalist must be eliminated with the emission
/// reason before anything is measured, and the winner must come from what
/// survived.
#[test]
fn a_finalist_over_the_register_ceiling_is_eliminated_before_measurement() {
    let evaluator = LadderEvaluator::reporting(Reported::OverCeilingOnRankedFirst);
    let artifact = measured_compile(priced_device(), budget(LAUNCHES), &evaluator)
        .expect("a finalist within the ceiling must still win");
    let plan = artifact.selected_plan();
    let ranked_first = evaluator
        .inspected()
        .first()
        .copied()
        .expect("emission must read what it built");
    let measured = evaluator.measured();

    assert!(
        !measured.contains(&ranked_first),
        "a plan allocating more registers than the device has must not be launched: {measured:?}"
    );
    assert_ne!(
        plan.fusion.len(),
        ranked_first,
        "the winner cannot be the plan the device could not run"
    );
    assert_eq!(
        plan.certificate.pruned_for(PruneReason::Emission),
        1,
        "the over-ceiling finalist must be eliminated for emission"
    );
    assert_eq!(
        plan.search_work.measurements,
        u32::try_from(measured.len()).expect("measurement count fits u32"),
        "only the finalists that survived emission may be measured"
    );
}

/// WHY: an authenticated winner is what makes a compilation reproducible across
/// sessions on one device. A re-run under the same protocol and the same priced
/// fact set must return the recorded winner even when this session's samples
/// favour a rival inside the equivalence band, and a record priced by another
/// fact set must be set aside, because the figures the ranking paid with
/// changed.
///
/// The stale case stamps another fact-set version onto a record whose candidate
/// identities are this session's own, so the fact-set comparison is the only
/// thing that can refuse it. A record from a differently calibrated device would
/// also carry candidate identities this session never emitted, and would be set
/// aside by the identity lookup whether or not the fact-set version was ever
/// read.
#[test]
fn a_recorded_winner_stands_until_the_priced_fact_set_is_recalibrated() {
    const CALIBRATION: u16 = 4;

    let evaluator = LadderEvaluator::indistinguishable();
    let device = timed_device().with_calibration_version(CALIBRATION);
    let first = measured_compile(device, budget(LAUNCHES), &evaluator)
        .expect("every finalist builds, so one must win");
    let PlanMeasurement::Measured(evidence) = &first.selected_plan().measurement else {
        panic!("a timed device must record measured evidence");
    };
    assert_eq!(
        evidence.environment.facts_calibration_version, CALIBRATION,
        "the record must state which fact set priced the ranking"
    );

    // Authenticate a candidate this session did not select, so retaining it is
    // observable. Every finalist here is inside the band, so no rival can be
    // measured decisively faster.
    let mut recorded = evidence.clone();
    let canonical = evidence
        .winner()
        .expect("a record names its winner")
        .identity;
    let alternative = evidence
        .candidates
        .iter()
        .position(|candidate| candidate.identity != canonical)
        .expect("the fixture must measure more than one finalist");
    recorded.winner = u32::try_from(alternative).expect("candidate index fits u32");
    let authenticated = evidence.candidates[alternative].identity;

    let unchanged = measured_recompile(device, &evaluator, recorded.clone())
        .expect("a re-run must still compile");
    assert_eq!(
        measured_winner(&unchanged),
        authenticated,
        "an unchanged protocol and fact set must return the authenticated winner"
    );

    let mut stale = recorded;
    stale.environment.facts_calibration_version = CALIBRATION - 1;
    let recalibrated =
        measured_recompile(device, &evaluator, stale).expect("a stale record must still compile");
    assert_eq!(
        measured_winner(&recalibrated),
        canonical,
        "a record priced by another fact set must let this session's own selection stand"
    );
}

/// One measured compilation that carries `recorded` as the authenticated winner.
fn measured_recompile(
    device: DeviceFacts,
    evaluator: &LadderEvaluator,
    recorded: MeasurementRecord,
) -> Result<Artifact, vyre_megakernel::CompileError> {
    let request = fixture_request(device, budget(LAUNCHES), latency_objective())
        .with_recorded_measurement(recorded)
        .validate()
        .expect("request must validate");
    compile_measured(&request, evaluator)
}

/// Identity of the candidate a measured artifact selected.
fn measured_winner(artifact: &Artifact) -> vyre_megakernel::Digest {
    let PlanMeasurement::Measured(evidence) = &artifact.selected_plan().measurement else {
        panic!("a timed device must record measured evidence");
    };
    evidence
        .winner()
        .expect("a record names its winner")
        .identity
}

/// The distinct generated-kernel counts in `order`, ascending.
fn distinct(order: &[usize]) -> Vec<usize> {
    let mut counts = order.to_vec();
    counts.sort_unstable();
    counts.dedup();
    counts
}

/// WHY: the allocation plan states the bytes that must be resident for the
/// artifact to run, and a measurement is only evidence about the plan that ran.
/// A device reporting fewer bytes than the plan requires is holding something
/// else, so timing it would rank a schedule nobody compiled. The reconciliation
/// is one-directional on purpose: a device holds the caller's buffers and other
/// work besides, so more bytes than planned is normal and fewer is impossible.
#[test]
fn a_device_holding_fewer_bytes_than_the_plan_requires_refuses_the_compile() {
    let planned = measured_compile(
        priced_device(),
        budget(LAUNCHES),
        &LadderEvaluator::reporting(Reported::ResidentAtPlan),
    )
    .expect("a device holding the planned bytes compiles")
    .allocation()
    .aggregate_peak_bytes;
    assert!(
        planned > 0,
        "the fixture plan must require bytes for the reconciliation to have a subject"
    );

    let error = measured_compile(
        priced_device(),
        budget(LAUNCHES),
        &LadderEvaluator::reporting(Reported::ResidentBelowPlan),
    )
    .expect_err("a device holding fewer bytes than the plan requires must not be measured");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC041_UNRECONCILED_RESIDENT_BYTES"
    );
    assert_eq!(
        error
            .diagnostic
            .location
            .as_ref()
            .and_then(|location| location.path.as_deref()),
        Some("measurement.resident_device_bytes")
    );
    let figures: Vec<u64> = error
        .diagnostic
        .message
        .split_whitespace()
        .filter_map(|word| word.parse().ok())
        .collect();
    assert_eq!(
        figures.len(),
        2,
        "the diagnostic must state the observed and the planned bytes: {error}"
    );
    assert_eq!(
        figures[0] + 1,
        figures[1],
        "the refused pair must be the pair the device reported: {error}"
    );
}

/// WHY: a backend with no memory query reports zero, and zero is an absent fact
/// rather than a device holding nothing. Refusing it would make the measured
/// path unusable on every backend that cannot answer the question.
#[test]
fn a_backend_that_reports_no_resident_bytes_still_measures() {
    let artifact = measured_compile(
        priced_device(),
        budget(LAUNCHES),
        &LadderEvaluator::reporting(Reported::Nothing),
    )
    .expect("an unreported memory figure leaves the planned figure unreconciled");
    assert!(artifact.allocation().aggregate_peak_bytes > 0);
}
