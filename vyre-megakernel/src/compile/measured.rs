//! Measured live-device compilation workflow.

use std::time::Instant;

use crate::certificate::{PruneReason, SearchCertificate};
use crate::error::{failure, CompileError, CompilerFailureKind};
use crate::grammar::ScheduleProduction;
use crate::measure::{
    self, CandidateMeasurement, MeasurementEnvironment, MeasurementProtocol, MeasurementRecord,
    ReplacementVerdict, SampleEstimate,
};
use crate::request::ValidatedCompileRequest;
use crate::schema::{Artifact, PlanMeasurement};
use crate::target::{TargetCompileError, TargetCompiler};
use crate::{candidate, cost, select};

use super::{assemble, first_ranked, require_single_artifact};

/// Registers, spill and shared bytes one emitted entry point allocates.
///
/// A target compiler assigns physical registers and decides what spills; a
/// device reports what the loaded module holds. Both figures are measurements of
/// the entry point the compiler is about to time, and both outrank the estimate
/// candidate search derived from the IR. Zero means the backend reported nothing
/// for that term, and the estimate stands for it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EmittedResources {
    /// Registers the entry point allocates per invocation.
    pub registers_per_invocation: u32,
    /// Local-memory spill bytes per invocation.
    pub spill_bytes_per_invocation: u32,
    /// Statically declared workgroup-scoped bytes.
    pub shared_memory_bytes: u32,
}

/// What one counted launch reported about itself.
///
/// A launch is the only moment the artifact's storage is bound, so both figures
/// come from the instance that ran and neither can be read from anywhere else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchObservation {
    /// Device time of the launch, in nanoseconds, from the device clock.
    pub device_ns: u64,
    /// Device bytes the launched instance held resident, or `None` when the
    /// backend has no memory query.
    pub resident_device_bytes: Option<u64>,
}

/// Device access the compiler borrows to time its finalists.
///
/// The compiler owns which plans are finalists and how their times are compared.
/// The caller owns the device: it supplies the target compiler that turns one
/// artifact into loadable bytes, the resources each emitted entry point turned
/// out to need, and a launch that returns the device time of one execution.
/// Nothing here acquires a device, so a caller without one calls [`super::compile`]
/// instead.
pub trait FinalistEvaluator {
    /// Target compiler that turns one candidate artifact into target bytes.
    fn target_compiler(&self) -> &dyn TargetCompiler;

    /// What each entry point of `payload` allocates, in payload entry order.
    ///
    /// The compiler re-ranks its emitted finalists on these figures before it
    /// spends a measurement, so a plan whose real register allocation costs
    /// occupancy is measured after one whose does not. A backend whose API
    /// reports none of it returns one default record per entry, which leaves
    /// every candidate ranked on the analytic estimate.
    ///
    /// # Errors
    ///
    /// Returns an error when the payload cannot be inspected on this device.
    fn resources(
        &self,
        artifact: &Artifact,
        payload: &crate::TargetPayload,
    ) -> Result<Vec<EmittedResources>, TargetCompileError>;

    /// Launch `payload` once and report the device time of that launch and what
    /// the launched instance held resident. The time must come from the device,
    /// not the host clock.
    ///
    /// The launch must be complete before this returns. The protocol counts one
    /// sample per call and compares samples across candidates, so a call that
    /// returned while the device was still running would attribute one
    /// candidate's work to whichever candidate the round measured next.
    ///
    /// The resident figure is read from the instance this call launched, while
    /// that instance still holds its storage. It cannot be recovered afterwards:
    /// once the instance is dropped the bytes are released, and a fresh instance
    /// reports whatever the allocator holds for something else.
    ///
    /// # Errors
    ///
    /// Returns an error when the payload cannot be launched or timed on this
    /// device.
    fn measure(
        &self,
        artifact: &Artifact,
        payload: &crate::TargetPayload,
    ) -> Result<LaunchObservation, TargetCompileError>;

    /// Clock, thermal and power state the device reports as the session starts.
    ///
    /// Retained beside the samples so a reader can tell a slow candidate from a
    /// throttled device. A backend whose API reports none of it returns
    /// [`DeviceState::unreported`](crate::measure::DeviceState::unreported), and
    /// the drift the session observes across its own rounds still holds.
    fn device_state(&self) -> measure::DeviceState;
}

/// Compile with the ranked finalists emitted for the target and timed on the
/// device under the versioned measurement protocol.
pub fn compile_measured(
    request: &ValidatedCompileRequest,
    evaluator: &dyn FinalistEvaluator,
) -> Result<Artifact, CompileError> {
    require_single_artifact(request)?;
    let context = super::prepare(request)?;
    let budget = request.search_budget;
    if budget.max_measurements == 0 || budget.max_target_compilations == 0 {
        return assemble(
            request,
            &context,
            first_ranked(&context)?,
            &context.certificate,
            context.work,
            PlanMeasurement::Unbudgeted,
        );
    }
    if !request.device.supports_device_timestamps() {
        return assemble(
            request,
            &context,
            first_ranked(&context)?,
            &context.certificate,
            context.work,
            PlanMeasurement::UntimedDevice,
        );
    }

    let finalists = context
        .ranked
        .len()
        .min(budget.max_target_compilations as usize);
    let started = Instant::now();
    let mut work = context.work;
    let mut certificate = context.certificate.clone();
    let mut emitted: Vec<(usize, Artifact, crate::TargetPayload)> = Vec::new();
    let mut rejection = None;
    for index in 0..finalists {
        if spent(started) >= budget.max_elapsed_ns {
            break;
        }
        let provisional = assemble(
            request,
            &context,
            &context.ranked[index],
            &context.certificate,
            context.work,
            PlanMeasurement::Unbudgeted,
        )?;
        work.target_compilations = work.target_compilations.saturating_add(1);
        match evaluator.target_compiler().compile(&provisional) {
            Ok(payload) => emitted.push((index, provisional, payload)),
            Err(error) => {
                eliminate(&mut certificate, &context.ranked[index]);
                rejection = Some(finalist_failure(index, &error));
            }
        }
    }
    // Emission and load turn every register and shared-byte estimate into a
    // measurement. Re-rank on those before spending a device measurement, so the
    // plan measured first is the plan the reported allocation favours rather than
    // the one the IR estimate favoured.
    let mut reranked: Vec<(u64, usize, usize)> = Vec::with_capacity(emitted.len());
    let mut over_ceiling: Vec<usize> = Vec::new();
    for (position, (index, provisional, payload)) in emitted.iter().enumerate() {
        let reported = evaluator
            .resources(provisional, payload)
            .map_err(|error| finalist_failure(*index, &error))?;
        let candidate = &context.ranked[*index].candidate;
        let groups = reported_groups(&reported, payload, candidate);
        let ceiling = request.device.hardware_registers_per_invocation();
        if ceiling > 0
            && groups
                .iter()
                .any(|group| group.registers_per_invocation > ceiling)
        {
            over_ceiling.push(position);
            continue;
        }
        let cost = cost::evaluate_reported(
            candidate,
            &context.facts,
            &context.dependencies,
            request.device,
            &groups,
        );
        reranked.push((cost.total, *index, position));
    }
    for position in over_ceiling {
        let (index, ..) = &emitted[position];
        eliminate(&mut certificate, &context.ranked[*index]);
    }
    // Total first, then the analytic rank, so a tie on the reported cost keeps
    // the order a caller can reproduce from the certificate.
    reranked.sort_unstable();
    certificate.canonicalize();
    let mut session: Vec<Sampling> = reranked
        .into_iter()
        .map(|(predicted_ns, index, position)| Sampling {
            index,
            position,
            predicted_ns,
            samples: Vec::new(),
            reconciled: false,
        })
        .collect();

    let protocol = MeasurementProtocol::V1.fitted(budget.max_measurements);
    // Warmup is charged against the budget like any counted launch. Its samples
    // are discarded: a first launch measures module load and cold allocation,
    // which is not what distinguishes two schedules.
    for entry in &mut session {
        if spent(started) >= budget.max_elapsed_ns {
            break;
        }
        let (_, provisional, payload) = &emitted[entry.position];
        for _ in 0..protocol.warmup_launches {
            let observation = evaluator
                .measure(provisional, payload)
                .map_err(|error| finalist_failure(entry.index, &error))?;
            work.measurements = work.measurements.saturating_add(1);
            reconcile_launch(provisional, &observation, entry)?;
        }
    }

    let mut rounds = 0_u32;
    let mut first_round_ns = 0_u64;
    let mut last_round_ns = 0_u64;
    while !protocol.rounds_exhausted(rounds) && !session.is_empty() {
        if spent(started) >= budget.max_elapsed_ns {
            break;
        }
        let mut round = Vec::with_capacity(session.len());
        // Rotate the visit order every round. Measuring the same candidate first
        // every time charges it for whatever the device does at the start of a
        // round, which is how a ranking becomes an artefact of position.
        for offset in 0..session.len() {
            let slot = (rounds as usize + offset) % session.len();
            let entry = &mut session[slot];
            let (_, provisional, payload) = &emitted[entry.position];
            for _ in 0..protocol.repetitions_per_round {
                let observation = evaluator
                    .measure(provisional, payload)
                    .map_err(|error| finalist_failure(entry.index, &error))?;
                entry.samples.push(observation.device_ns);
                round.push(observation.device_ns);
                work.measurements = work.measurements.saturating_add(1);
                reconcile_launch(provisional, &observation, entry)?;
            }
        }
        rounds = rounds.saturating_add(1);
        round.sort_unstable();
        last_round_ns = round.get(round.len() / 2).copied().unwrap_or(0);
        if rounds == 1 {
            first_round_ns = last_round_ns;
        }
        if protocol.rounds_sufficient(rounds)
            && session.iter().all(|entry| settled(entry, &protocol))
        {
            break;
        }
    }
    work.elapsed_ns = work.elapsed_ns.saturating_add(spent(started));

    let mut ranked_indices = Vec::with_capacity(session.len());
    let mut candidates = Vec::with_capacity(session.len());
    for entry in &session {
        let Some(estimate) = SampleEstimate::from_samples(&entry.samples, &protocol) else {
            continue;
        };
        let (_, provisional, _) = &emitted[entry.position];
        ranked_indices.push(entry.index);
        candidates.push(CandidateMeasurement {
            identity: provisional.digest(),
            analytic_rank: u32::try_from(entry.index).unwrap_or(u32::MAX),
            predicted_ns: entry.predicted_ns,
            samples: entry.samples.clone(),
            estimate,
        });
    }
    if candidates.is_empty() {
        return match rejection {
            Some(error) => Err(error),
            None => assemble(
                request,
                &context,
                first_ranked(&context)?,
                &certificate,
                work,
                PlanMeasurement::Unbudgeted,
            ),
        };
    }

    // Candidates are in reported-cost order, so the first is the canonical
    // lower-risk finalist. A later one takes the selection only by clearing the
    // equivalence band, which is what makes two runs of the same search on the
    // same device select the same artifact.
    let mut winner = 0_usize;
    for (slot, candidate) in candidates.iter().enumerate().skip(1) {
        if measure::improves(&candidates[winner].estimate, &candidate.estimate, &protocol) {
            winner = slot;
        }
    }
    let mut record = MeasurementRecord {
        protocol,
        environment: MeasurementEnvironment {
            warmup_launches: protocol.warmup_launches,
            facts_calibration_version: request.device().calibration_version(),
            first_round_ns,
            last_round_ns,
            state: evaluator.device_state(),
        },
        rounds,
        candidates,
        winner: u32::try_from(winner).unwrap_or(u32::MAX),
    };
    if let Some(incumbent) = request.recorded_measurement() {
        if record.verdict_against(incumbent) == ReplacementVerdict::Equivalent {
            if let Some(authenticated) = incumbent.winner().and_then(|authenticated| {
                record
                    .candidates
                    .iter()
                    .position(|candidate| candidate.identity == authenticated.identity)
            }) {
                record.winner = u32::try_from(authenticated).unwrap_or(u32::MAX);
                winner = authenticated;
            }
        }
    }
    assemble(
        request,
        &context,
        &context.ranked[ranked_indices[winner]],
        &certificate,
        work,
        PlanMeasurement::Measured(record),
    )
}

/// One finalist and every counted sample taken against it.
struct Sampling {
    /// Position in the analytic ranking.
    index: usize,
    /// Position in the emitted finalist list.
    position: usize,
    /// Analytic cost the reported-resource ranking predicted, in nanoseconds.
    predicted_ns: u64,
    /// Counted device times in measurement order.
    samples: Vec<u64>,
    /// Whether the resident-byte figure was reconciled after this finalist ran.
    reconciled: bool,
}

/// Whether this candidate's samples are precise enough to stop sampling it.
fn settled(entry: &Sampling, protocol: &MeasurementProtocol) -> bool {
    SampleEstimate::from_samples(&entry.samples, protocol)
        .is_some_and(|estimate| estimate.is_settled(protocol))
}

/// Nanoseconds the measured path has spent since it started.
fn spent(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Project what each emitted entry reported onto the fusion groups the cost
/// model prices.
fn reported_groups(
    reported: &[EmittedResources],
    payload: &crate::TargetPayload,
    candidate: &candidate::CandidatePlan,
) -> Vec<cost::ReportedGroup> {
    let mut groups = vec![cost::ReportedGroup::default(); candidate.group_count()];
    for (entry, resources) in payload.entries().iter().zip(reported) {
        let node = usize::try_from(entry.node.0).unwrap_or(usize::MAX);
        let Some(group) = candidate.node_groups.get(node).copied() else {
            continue;
        };
        let Some(slot) = groups.get_mut(usize::try_from(group).unwrap_or(usize::MAX)) else {
            continue;
        };
        let invocations = entry
            .grid_size
            .iter()
            .chain(entry.workgroup_size.iter())
            .fold(1_u64, |total, extent| {
                total.saturating_mul(u64::from(*extent))
            });
        slot.registers_per_invocation = slot
            .registers_per_invocation
            .max(resources.registers_per_invocation);
        slot.shared_memory_bytes = slot.shared_memory_bytes.max(resources.shared_memory_bytes);
        slot.spill_traffic_bytes = slot.spill_traffic_bytes.saturating_add(
            u64::from(resources.spill_bytes_per_invocation).saturating_mul(invocations),
        );
    }
    groups
}

fn reconcile_resident_bytes(artifact: &Artifact, observed: u64) -> Result<(), CompileError> {
    let planned = artifact.allocation().artifact_peak_bytes()?;
    if observed >= planned {
        return Ok(());
    }
    Err(failure(
        CompilerFailureKind::UnreconciledResidentBytes,
        "measurement.resident_device_bytes",
        format!(
            "the launched instance held {observed} bytes while the artifact-owned storage of the selected allocation plan requires {planned}"
        ),
        "bind the allocation plan the artifact records before measuring it",
    ))
}

fn reconcile_launch(
    artifact: &Artifact,
    observation: &LaunchObservation,
    entry: &mut Sampling,
) -> Result<(), CompileError> {
    if entry.reconciled {
        return Ok(());
    }
    let Some(observed) = observation.resident_device_bytes else {
        return Ok(());
    };
    reconcile_resident_bytes(artifact, observed)?;
    entry.reconciled = true;
    Ok(())
}

fn eliminate(certificate: &mut SearchCertificate, selection: &select::Selection) {
    let production = selection
        .candidate
        .derivation
        .last()
        .map_or(ScheduleProduction::Fusion, |step| step.production);
    certificate.pruned(production, PruneReason::Emission);
}

fn finalist_failure(index: usize, error: &TargetCompileError) -> CompileError {
    failure(
        CompilerFailureKind::FinalistEvaluation,
        format!("search.finalists[{index}]"),
        error.to_string(),
        "supply a finalist evaluator whose target compiler and device accept every ranked plan",
    )
}
