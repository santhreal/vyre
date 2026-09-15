//! The versioned protocol budgeted device measurement runs under, and the
//! evidence one measured selection retains.
//!
//! A device time is a noisy quantity. The same entry point launched twice on the
//! same device returns two numbers, and the difference between them is not a
//! property of the program: it is queue occupancy, clock state, cache state and
//! whatever else shares the device. A comparison that ignores that selects a
//! different artifact from the same search on the same host, which makes the
//! artifact unreproducible. This module states, in one versioned record, how many
//! launches settle the device, how many are counted, in what order candidates are
//! interleaved, how samples become one estimate, how wide that estimate's
//! uncertainty is, when sampling stops, and how close two candidates have to be
//! before the difference is not evidence at all.
//!
//! Every figure carries its unit in its name: `_ns` nanoseconds, `_khz`
//! kilohertz, `_permille` parts per thousand, `_millicelsius` thousandths of a
//! degree Celsius, `_milliwatts` thousandths of a watt. Every figure carries its
//! provenance in its documentation: derived from the counted samples, reported by
//! the backend, or fixed by the protocol version.

use serde::{Deserialize, Serialize};

use crate::error::{failure, CompileError, CompilerFailureKind};
use crate::identity::Digest;

/// Protocol the current compiler measures under.
///
/// A record measured under a different version is not comparable with one
/// measured under this version, because the launches counted and the estimator
/// applied to them both changed. Advancing this constant is what allows a
/// recalibration to replace an authenticated winner.
pub const MEASUREMENT_PROTOCOL_VERSION: u16 = 1;

/// Scale from median absolute deviation to a standard-deviation-equivalent
/// spread, in permille. The constant is 1.4826 for a normal distribution.
const MAD_TO_SIGMA_PERMILLE: u64 = 1_483;

/// Fixed rules one measurement session runs under.
///
/// Provenance: every field is fixed by [`MEASUREMENT_PROTOCOL_VERSION`] and
/// recorded in the artifact, so a reader of the artifact knows what the numbers
/// beside it mean without knowing which compiler produced them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MeasurementProtocol {
    /// Protocol version these rules come from.
    pub version: u16,
    /// Launches performed per candidate before the first counted sample, to
    /// bring module load, allocation and clocks out of their cold state.
    pub warmup_launches: u32,
    /// Counted launches one candidate receives per round.
    pub repetitions_per_round: u32,
    /// Rounds performed before the stopping rule may end sampling.
    pub min_rounds: u32,
    /// Rounds after which sampling ends whether or not estimates settled.
    pub max_rounds: u32,
    /// Share of each candidate's samples dropped from the slow end before the
    /// estimate is taken, in permille of the sample count.
    pub trim_permille: u16,
    /// Relative uncertainty at or below which a candidate counts as settled, in
    /// permille of its own estimate.
    pub uncertainty_target_permille: u16,
    /// Width of the band inside which two estimates are not distinguishable, in
    /// permille of the incumbent estimate.
    pub equivalence_permille: u16,
}

impl MeasurementProtocol {
    /// Protocol version 1.
    ///
    /// Two warmup launches cover module load and first-touch allocation. The
    /// trim is one-sided at 200 permille because device noise is one-sided:
    /// interference makes a launch slower and nothing makes it faster than the
    /// device can run it. The 30-permille uncertainty target and the 20-permille
    /// equivalence band are set so the band is narrower than the noise a settled
    /// candidate still carries, which keeps a tie a tie.
    pub const V1: Self = Self {
        version: MEASUREMENT_PROTOCOL_VERSION,
        warmup_launches: 2,
        repetitions_per_round: 1,
        min_rounds: 3,
        max_rounds: 64,
        trim_permille: 200,
        uncertainty_target_permille: 30,
        equivalence_permille: 20,
    };

    /// The same rules narrowed to what a per-candidate launch budget affords.
    ///
    /// Warmup is charged against the budget like any other launch, so a budget
    /// of one launch counts that launch and warms up with nothing. Rounds shrink
    /// before repetitions do: more rounds spread a candidate's samples further
    /// across the session, which is what makes a clock or thermal change visible
    /// as drift rather than invisible as a slower candidate.
    #[must_use]
    pub fn fitted(&self, launch_budget: u32) -> Self {
        if launch_budget == 0 {
            return Self {
                warmup_launches: 0,
                repetitions_per_round: 0,
                min_rounds: 0,
                max_rounds: 0,
                ..*self
            };
        }
        let warmup = self.warmup_launches.min(launch_budget - 1);
        let countable = launch_budget - warmup;
        let repetitions = self.repetitions_per_round.max(1).min(countable);
        let max_rounds = self.max_rounds.min(countable / repetitions);
        Self {
            warmup_launches: warmup,
            repetitions_per_round: repetitions,
            min_rounds: self.min_rounds.min(max_rounds),
            max_rounds,
            ..*self
        }
    }

    /// Launches one candidate receives when sampling runs to `max_rounds`.
    #[must_use]
    pub const fn launches_per_candidate(&self) -> u32 {
        self.warmup_launches
            .saturating_add(self.max_rounds.saturating_mul(self.repetitions_per_round))
    }

    /// Whether sampling must end after `rounds` completed rounds.
    #[must_use]
    pub const fn rounds_exhausted(&self, rounds: u32) -> bool {
        rounds >= self.max_rounds
    }

    /// Whether `rounds` completed rounds satisfy the minimum the protocol counts
    /// before a settled estimate may end sampling.
    #[must_use]
    pub const fn rounds_sufficient(&self, rounds: u32) -> bool {
        rounds >= self.min_rounds
    }

    /// Validate the recorded rules.
    ///
    /// # Errors
    ///
    /// Returns a malformed-artifact diagnostic when a version is absent, a round
    /// or repetition count cannot produce a sample, the round bounds are
    /// inverted, or a share exceeds what a share can be.
    pub fn validate(&self) -> Result<(), CompileError> {
        let invalid = |path: &str, message: String, fix: &str| {
            failure(
                CompilerFailureKind::MalformedArtifact,
                format!("artifact.body.selected_plan.measurement.protocol.{path}"),
                message,
                fix,
            )
        };
        if self.version == 0 {
            return Err(invalid(
                "version",
                "measured evidence records no protocol version".to_string(),
                "record the protocol version the samples were measured under",
            ));
        }
        if self.repetitions_per_round == 0 || self.max_rounds == 0 {
            return Err(invalid(
                "repetitions_per_round",
                format!(
                    "{} round(s) of {} repetition(s) counts no sample",
                    self.max_rounds, self.repetitions_per_round
                ),
                "record the rounds and repetitions the session actually counted",
            ));
        }
        if self.min_rounds > self.max_rounds {
            return Err(invalid(
                "min_rounds",
                format!(
                    "sampling requires {} round(s) but ends after {}",
                    self.min_rounds, self.max_rounds
                ),
                "fit the protocol to the launch budget before measuring under it",
            ));
        }
        for (path, permille, ceiling) in [
            ("trim_permille", self.trim_permille, 500),
            (
                "uncertainty_target_permille",
                self.uncertainty_target_permille,
                1_000,
            ),
            ("equivalence_permille", self.equivalence_permille, 1_000),
        ] {
            if permille == 0 || permille > ceiling {
                return Err(invalid(
                    path,
                    format!("{path} is {permille}, outside 1..={ceiling}"),
                    "record a share the estimator and the equivalence band can apply",
                ));
            }
        }
        Ok(())
    }
}

/// One candidate's counted samples reduced to an estimate and its spread.
///
/// Provenance: derived from the counted samples of one candidate. The estimate is
/// a trimmed median, which a single stalled launch cannot move; the uncertainty
/// is the median absolute deviation of the kept samples scaled to a
/// standard-deviation equivalent, which states how far apart two estimates have
/// to be before the difference is the program rather than the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SampleEstimate {
    /// Trimmed median device time in nanoseconds.
    pub estimate_ns: u64,
    /// Robust spread of the kept samples in nanoseconds.
    pub uncertainty_ns: u64,
    /// Samples the estimate was taken over.
    pub kept: u32,
    /// Samples dropped from the slow end before the estimate was taken.
    pub trimmed: u32,
}

impl SampleEstimate {
    /// Reduce `samples` under `protocol`, or `None` when nothing was counted.
    #[must_use]
    pub fn from_samples(samples: &[u64], protocol: &MeasurementProtocol) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        let trimmed = trimmed_count(sorted.len(), protocol.trim_permille);
        let kept = &sorted[..sorted.len() - trimmed];
        let estimate_ns = median(kept);
        let mut deviations: Vec<u64> = kept
            .iter()
            .map(|sample| sample.abs_diff(estimate_ns))
            .collect();
        deviations.sort_unstable();
        let uncertainty_ns = median(&deviations).saturating_mul(MAD_TO_SIGMA_PERMILLE) / 1_000;
        Some(Self {
            estimate_ns,
            uncertainty_ns,
            kept: u32::try_from(kept.len()).unwrap_or(u32::MAX),
            trimmed: u32::try_from(trimmed).unwrap_or(u32::MAX),
        })
    }

    /// Uncertainty as a share of the estimate, in permille.
    #[must_use]
    pub const fn relative_uncertainty_permille(&self) -> u32 {
        if self.estimate_ns == 0 {
            return u32::MAX;
        }
        let permille = self.uncertainty_ns.saturating_mul(1_000) / self.estimate_ns;
        if permille > u32::MAX as u64 {
            u32::MAX
        } else {
            permille as u32
        }
    }

    /// Whether this estimate is precise enough for sampling to stop.
    #[must_use]
    pub const fn is_settled(&self, protocol: &MeasurementProtocol) -> bool {
        self.relative_uncertainty_permille() <= protocol.uncertainty_target_permille as u32
    }
}

/// Whether the backend reported a clock-limiting condition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThrottleState {
    /// The backend exposes no throttle reporting on this device.
    #[default]
    Unreported,
    /// The backend reports the device running unthrottled.
    Clear,
    /// The backend reports the device clock limited by power, thermal or
    /// reliability control.
    Throttled,
}

/// Clock, thermal and power state the backend reported for the device.
///
/// Provenance: reported by the backend at the start of the measurement session.
/// A zero figure means the backend reports nothing for that term, the same
/// convention emitted resource records use, so a reader never mistakes an
/// unreported clock for a stopped one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceState {
    /// Compute clock in kilohertz.
    pub graphics_clock_khz: u32,
    /// Memory clock in kilohertz.
    pub memory_clock_khz: u32,
    /// Device temperature in thousandths of a degree Celsius.
    pub temperature_millicelsius: i32,
    /// Instantaneous board power draw in thousandths of a watt.
    pub power_draw_milliwatts: u32,
    /// Whether the reported clock is limited.
    pub throttle: ThrottleState,
}

impl DeviceState {
    /// State a backend with no clock, thermal or power reporting supplies.
    #[must_use]
    pub const fn unreported() -> Self {
        Self {
            graphics_clock_khz: 0,
            memory_clock_khz: 0,
            temperature_millicelsius: 0,
            power_draw_milliwatts: 0,
            throttle: ThrottleState::Unreported,
        }
    }

    /// Whether the backend reported no term of this state.
    #[must_use]
    pub const fn is_unreported(&self) -> bool {
        self.graphics_clock_khz == 0
            && self.memory_clock_khz == 0
            && self.temperature_millicelsius == 0
            && self.power_draw_milliwatts == 0
            && matches!(self.throttle, ThrottleState::Unreported)
    }
}

/// What the device was doing while the session measured it.
///
/// Provenance: `warmup_launches` and the round estimates are counted by the
/// session; `state` is whatever the backend reported. Drift between the first
/// and last counted round is the device-neutral observation of a clock or
/// thermal change, and it is available on a backend that reports no state at
/// all.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MeasurementEnvironment {
    /// Launches spent settling each candidate before its first counted sample.
    pub warmup_launches: u32,
    /// Version of the calibrated device fact set the session ranked against,
    /// zero when uncalibrated.
    pub facts_calibration_version: u16,
    /// Median counted sample of the first round across every candidate, in
    /// nanoseconds.
    pub first_round_ns: u64,
    /// Median counted sample of the last round across every candidate, in
    /// nanoseconds.
    pub last_round_ns: u64,
    /// Clock, thermal and power state the backend reported.
    pub state: DeviceState,
}

impl MeasurementEnvironment {
    /// Signed change from the first counted round to the last, in permille of
    /// the first. A device whose clocks fall or whose queue fills across the
    /// session reports positive drift.
    #[must_use]
    pub fn drift_permille(&self) -> i32 {
        if self.first_round_ns == 0 {
            return 0;
        }
        let first = i128::from(self.first_round_ns);
        let last = i128::from(self.last_round_ns);
        permille_i32((last - first) * 1_000 / first)
    }
}

/// Every counted sample of one candidate, and what the cost model predicted for
/// it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CandidateMeasurement {
    /// Artifact digest of the candidate as it was emitted and measured, which is
    /// the identity a later session matches an authenticated winner against.
    pub identity: Digest,
    /// Position this candidate held in the analytic ranking, zero-based.
    pub analytic_rank: u32,
    /// Analytic cost the ranking predicted for it, in nanoseconds.
    pub predicted_ns: u64,
    /// Counted device times in the order they were measured, in nanoseconds.
    pub samples: Vec<u64>,
    /// The estimate those samples reduce to.
    pub estimate: SampleEstimate,
}

impl CandidateMeasurement {
    /// Signed prediction error of the analytic cost against the measured
    /// estimate, in permille of the estimate. A model that predicts a candidate
    /// slower than it measured reports positive error.
    ///
    /// Provenance: this is the figure a fact-set recalibration reads. It never
    /// changes a selection, which is decided by the measurement alone.
    #[must_use]
    pub fn prediction_error_permille(&self) -> i32 {
        if self.estimate.estimate_ns == 0 {
            return 0;
        }
        let measured = i128::from(self.estimate.estimate_ns);
        let predicted = i128::from(self.predicted_ns);
        permille_i32((predicted - measured) * 1_000 / measured)
    }
}

/// Whether `challenger` is faster than `incumbent` by enough to count as
/// faster.
///
/// The margin has to clear both the equivalence band, which is a share of the
/// incumbent estimate fixed by the protocol, and the combined uncertainty of the
/// two estimates. A margin inside either is the device, not the program, so the
/// incumbent stands and selection stays deterministic across re-runs.
#[must_use]
pub fn improves(
    incumbent: &SampleEstimate,
    challenger: &SampleEstimate,
    protocol: &MeasurementProtocol,
) -> bool {
    let band = incumbent
        .estimate_ns
        .saturating_mul(u64::from(protocol.equivalence_permille))
        / 1_000;
    let noise = incumbent
        .uncertainty_ns
        .saturating_add(challenger.uncertainty_ns);
    let margin = incumbent.estimate_ns.saturating_sub(challenger.estimate_ns);
    margin > band && margin > noise
}

/// Whether a newly measured winner may replace an authenticated one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplacementVerdict {
    /// The challenger is faster than the incumbent by more than the equivalence
    /// band and by more than the uncertainty of both estimates.
    Replaces,
    /// The two are the same candidate, or their estimates are inside the
    /// equivalence band, so the incumbent stands.
    Equivalent,
    /// Nothing the two records hold makes them comparable: they were measured
    /// under different protocol versions, or the incumbent's winner is not among
    /// the candidates this session measured.
    Incomparable,
}

/// Everything one measured selection retains about how it was decided.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MeasurementRecord {
    /// Rules the session ran under, already fitted to the launch budget.
    pub protocol: MeasurementProtocol,
    /// What the device was doing while it ran.
    pub environment: MeasurementEnvironment,
    /// Rounds the session completed before the stopping rule ended it.
    pub rounds: u32,
    /// Every measured candidate, in the order the session measured them first.
    pub candidates: Vec<CandidateMeasurement>,
    /// Index into `candidates` of the selected candidate.
    pub winner: u32,
}

impl MeasurementRecord {
    /// The selected candidate.
    #[must_use]
    pub fn winner(&self) -> Option<&CandidateMeasurement> {
        self.candidates.get(self.winner as usize)
    }

    /// Counted launches performed against the selected candidate.
    #[must_use]
    pub fn winning_launches(&self) -> u32 {
        self.winner().map_or(0, |candidate| candidate.estimate.kept)
    }

    /// Estimated device time of the selected candidate in nanoseconds.
    #[must_use]
    pub fn winning_estimate_ns(&self) -> u64 {
        self.winner()
            .map_or(0, |candidate| candidate.estimate.estimate_ns)
    }

    /// The candidate carrying `identity`, when this session measured it.
    #[must_use]
    pub fn candidate(&self, identity: Digest) -> Option<&CandidateMeasurement> {
        self.candidates
            .iter()
            .find(|candidate| candidate.identity == identity)
    }

    /// Whether this session's winner may replace the winner `incumbent`
    /// authenticated.
    ///
    /// Re-running an unchanged protocol over an unchanged candidate space
    /// produces estimates that differ by measurement noise. Replacing the
    /// recorded winner on that difference would make the artifact a function of
    /// when it was compiled, so a challenger has to clear the equivalence band
    /// and the uncertainty of both estimates before it counts as faster.
    ///
    /// A protocol version or a calibrated fact-set version that differs from the
    /// incumbent's makes the two incomparable: the rules or the priced figures
    /// changed, so the earlier winner carries no authority over the later
    /// session. That is the only way a recorded winner is set aside without
    /// being measured slower.
    #[must_use]
    pub fn verdict_against(&self, incumbent: &Self) -> ReplacementVerdict {
        if incumbent.protocol.version != self.protocol.version
            || incumbent.environment.facts_calibration_version
                != self.environment.facts_calibration_version
        {
            return ReplacementVerdict::Incomparable;
        }
        let Some(incumbent_winner) = incumbent.winner() else {
            return ReplacementVerdict::Incomparable;
        };
        let Some(challenger) = self.winner() else {
            return ReplacementVerdict::Incomparable;
        };
        if challenger.identity == incumbent_winner.identity {
            return ReplacementVerdict::Equivalent;
        }
        let Some(retained) = self.candidate(incumbent_winner.identity) else {
            return ReplacementVerdict::Incomparable;
        };
        if improves(&retained.estimate, &challenger.estimate, &self.protocol) {
            ReplacementVerdict::Replaces
        } else {
            ReplacementVerdict::Equivalent
        }
    }

    /// Validate the retained evidence.
    ///
    /// # Errors
    ///
    /// Returns a malformed-artifact diagnostic when the protocol is malformed,
    /// no candidate was measured, the winner index names no candidate, or a
    /// candidate's samples and estimate disagree.
    pub fn validate(&self) -> Result<(), CompileError> {
        let invalid = |path: &str, message: String, fix: &str| {
            failure(
                CompilerFailureKind::MalformedArtifact,
                format!("artifact.body.selected_plan.measurement.{path}"),
                message,
                fix,
            )
        };
        self.protocol.validate()?;
        if self.candidates.is_empty() {
            return Err(invalid(
                "candidates",
                "measured selection retains no candidate samples".to_string(),
                "retain every candidate the session measured",
            ));
        }
        if self.rounds == 0 || self.rounds > self.protocol.max_rounds {
            return Err(invalid(
                "rounds",
                format!(
                    "session records {} round(s) under a protocol ending after {}",
                    self.rounds, self.protocol.max_rounds
                ),
                "record the rounds the session completed",
            ));
        }
        if self.winner().is_none() {
            return Err(invalid(
                "winner",
                format!(
                    "winner index {} names none of the {} measured candidates",
                    self.winner,
                    self.candidates.len()
                ),
                "record the index of the candidate the session selected",
            ));
        }
        for (position, candidate) in self.candidates.iter().enumerate() {
            let counted =
                u64::from(candidate.estimate.kept) + u64::from(candidate.estimate.trimmed);
            if counted != candidate.samples.len() as u64 {
                return Err(invalid(
                    "candidates.estimate",
                    format!(
                        "candidate {position} retains {} sample(s) but its estimate covers {counted}",
                        candidate.samples.len()
                    ),
                    "reduce the estimate from the retained samples",
                ));
            }
            if candidate.estimate.kept == 0 || candidate.estimate.estimate_ns == 0 {
                return Err(invalid(
                    "candidates.estimate",
                    format!(
                        "candidate {position} records an estimate over no sample or a zero device time"
                    ),
                    "retain positive device times for every measured candidate",
                ));
            }
        }
        Ok(())
    }
}

/// Samples dropped from the slow end for a sorted run of `len` samples. At least
/// one sample always survives, so a two-sample run trims nothing.
fn trimmed_count(len: usize, trim_permille: u16) -> usize {
    let trimmed = len.saturating_mul(usize::from(trim_permille)) / 1_000;
    trimmed.min(len.saturating_sub(1))
}

/// Median of a sorted slice. The upper of the two middles for an even count, so
/// the estimate never reports a time no launch achieved.
fn median(sorted: &[u64]) -> u64 {
    sorted.get(sorted.len() / 2).copied().unwrap_or(0)
}

/// Narrow a computed permille figure to the recorded width.
fn permille_i32(permille: i128) -> i32 {
    i32::try_from(permille.clamp(i128::from(i32::MIN), i128::from(i32::MAX))).unwrap_or(0)
}
