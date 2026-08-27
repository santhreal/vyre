//! The selected whole-program plan and the consistency rule its records are
//! decoded under.

use serde::{Deserialize, Serialize};
use vyre_foundation::schedule::SelectedSchedule;

use crate::certificate::SearchCertificate;
use crate::cost;
use crate::error::{failure, CompileError, CompilerFailureKind};
use crate::grammar::{DerivationStep, ScheduleProduction, SCHEDULE_GRAMMAR_VERSION};
use crate::measure::{MeasurementRecord, MEASUREMENT_PROTOCOL_VERSION};
use crate::request::{SearchBudget, SearchWork};

use super::records::{BarrierRecord, FusionRecord, FusionRejection, MaterializationRecord};

/// How the runtime executes one compiled artifact.
///
/// The compiler decides this, not the dispatcher: the decision needs the launch
/// count the caller declared and the device launch costs, both of which are
/// compile-time facts recorded in the request. A consumer executes the mode the
/// artifact names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    /// One kernel launch per stage per submission.
    Static,
    /// One resident kernel that polls a device-side work queue for the whole
    /// launch batch, paying one setup cost instead of one launch per item.
    Persistent {
        /// Launch overhead this mode removes, less the setup it pays, in
        /// nanoseconds. Computed from the device launch costs and the declared
        /// launch batch, and always positive: a non-positive figure is recorded
        /// as [`Self::Static`].
        saved_ns: u64,
    },
}

/// Whether a device measurement selected the plan, and what it measured.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanMeasurement {
    /// The search budget allowed no measurement, so the plan is the analytic
    /// winner and carries no measured device time.
    Unbudgeted,
    /// The device reports no launch timestamp, so nothing measured on it would
    /// be a device time.
    UntimedDevice,
    /// Selected by the lowest measured device-time estimate across the
    /// finalists, under the versioned measurement protocol whose evidence this
    /// carries.
    Measured(MeasurementRecord),
}

/// Immutable compiler-selected whole-program plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectedPlan {
    /// Executable queue or resident-partition topology selected by search.
    pub topology: crate::candidate::ExecutionTopology,
    /// Versioned backend-neutral phase and transform schedule selected by search.
    pub schedule: SelectedSchedule,
    /// Grammar productions the search applied to the unfused baseline, in order.
    pub derivation: Vec<DerivationStep>,
    /// Reproducible record of the bounded search that selected this plan.
    pub certificate: SearchCertificate,
    /// Selected fusion groups.
    pub fusion: Vec<FusionRecord>,
    /// Required dependency-completion boundaries.
    pub barriers: Vec<BarrierRecord>,
    /// Required cross-stage value materializations.
    pub materializations: Vec<MaterializationRecord>,
    /// Number of legal candidates examined.
    pub candidates_explored: u32,
    /// Search bounds under which this plan was selected.
    pub search_budget: SearchBudget,
    /// Exact work charged against the bounded search.
    pub search_work: SearchWork,
    /// Open-model cost of the selected plan.
    pub selection_cost: cost::CostBreakdown,
    /// Illegal producer-consumer fusions pruned with stable reasons.
    pub pruned_fusions: Vec<FusionRejection>,
    /// How the runtime executes this artifact.
    pub execution: ExecutionMode,
    /// Whether a device measurement chose this plan over its finalists.
    pub measurement: PlanMeasurement,
}

impl SelectedPlan {
    /// Validate the immutable selected-schedule stage and its bounded search
    /// provenance.
    ///
    /// # Errors
    ///
    /// Returns a malformed-artifact diagnostic when topology cardinalities,
    /// search accounting, or measurement provenance are inconsistent.
    pub fn validate(&self) -> Result<(), CompileError> {
        let invalid = |path: &str, message: String, fix: &str| {
            failure(
                CompilerFailureKind::MalformedArtifact,
                format!("artifact.body.selected_plan.{path}"),
                message,
                fix,
            )
        };
        self.schedule.validate().map_err(|error| {
            invalid(
                "schedule",
                error.to_string(),
                "re-run bounded schedule search and persist only a validated neutral schedule",
            )
        })?;
        if self.certificate.grammar_version != SCHEDULE_GRAMMAR_VERSION {
            return Err(invalid(
                "certificate.grammar_version",
                format!(
                    "plan was derived by grammar {} but this compiler derives grammar {}",
                    self.certificate.grammar_version, SCHEDULE_GRAMMAR_VERSION
                ),
                "re-compile the graph so the plan carries a derivation of the current grammar",
            ));
        }
        for step in &self.derivation {
            for transform in &step.transforms {
                if ScheduleProduction::deriving(transform) != step.production {
                    return Err(invalid(
                        "derivation.production",
                        format!(
                            "step {} records a transform production {} derives",
                            step.production.code(),
                            ScheduleProduction::deriving(transform).code()
                        ),
                        "record each step under the production that derives its transform",
                    ));
                }
                if !self
                    .schedule
                    .transforms
                    .iter()
                    .any(|record| record.transform == *transform)
                {
                    return Err(invalid(
                        "derivation.transforms",
                        format!(
                            "step {} names a transform the selected schedule never applied",
                            step.production.code()
                        ),
                        "record only the derivation the persisted schedule replays",
                    ));
                }
            }
        }
        for family in &self.certificate.derived {
            if family.derived == 0 || family.admitted > family.derived {
                return Err(invalid(
                    "certificate.derived",
                    format!(
                        "family {} records {} admitted of {} derived candidates",
                        family.production.code(),
                        family.admitted,
                        family.derived
                    ),
                    "record one derived family per production that proposed a candidate",
                ));
            }
        }
        for family in &self.certificate.pruned {
            if family.count == 0 {
                return Err(invalid(
                    "certificate.pruned",
                    format!(
                        "family {} records reason {} without an eliminated candidate",
                        family.production.code(),
                        family.reason.code()
                    ),
                    "record an eliminated family only when a candidate was eliminated",
                ));
            }
        }
        match self.topology {
            crate::candidate::ExecutionTopology::Sequential => {}
            crate::candidate::ExecutionTopology::ConcurrentQueue { queues } if queues > 0 => {}
            crate::candidate::ExecutionTopology::ResidentPartition { partitions, .. }
                if partitions > 0 => {}
            crate::candidate::ExecutionTopology::ConcurrentQueue { .. } => {
                return Err(invalid(
                    "topology.queues",
                    "concurrent queue topology contains zero queues".to_string(),
                    "select at least one executable queue",
                ));
            }
            crate::candidate::ExecutionTopology::ResidentPartition { .. } => {
                return Err(invalid(
                    "topology.partitions",
                    "resident topology contains zero partitions".to_string(),
                    "select at least one resident partition",
                ));
            }
        }
        if self.candidates_explored == 0 {
            return Err(invalid(
                "candidates_explored",
                "selected schedule records no explored legal candidate".to_string(),
                "record the executable unfused baseline candidate",
            ));
        }
        if self.candidates_explored != self.search_work.candidates_explored {
            return Err(invalid(
                "search_work.candidates_explored",
                format!(
                    "selected plan records {} explored candidates but search work records {}",
                    self.candidates_explored, self.search_work.candidates_explored
                ),
                "derive both fields from the same bounded search result",
            ));
        }
        for (path, actual, limit) in [
            (
                "search_work.candidates_explored",
                u64::from(self.search_work.candidates_explored),
                u64::from(self.search_budget.max_candidates),
            ),
            (
                "search_work.cpu_work",
                self.search_work.cpu_work,
                self.search_budget.max_cpu_work,
            ),
            (
                "search_work.target_compilations",
                u64::from(self.search_work.target_compilations),
                u64::from(self.search_budget.max_target_compilations),
            ),
            (
                "search_work.measurements",
                u64::from(self.search_work.measurements),
                u64::from(self.search_budget.max_measurements)
                    .saturating_mul(u64::from(self.search_work.target_compilations)),
            ),
        ] {
            if actual > limit {
                return Err(invalid(
                    path,
                    format!("search charged {actual} units against a limit of {limit}"),
                    "record a schedule selected within its authenticated search budget",
                ));
            }
        }
        match &self.measurement {
            PlanMeasurement::Measured(evidence) => {
                evidence.validate()?;
                // `MeasurementRecord::validate` already refused a winner with no
                // kept sample or a zero estimate, so what is left here is the
                // cross-record agreement it cannot see: the plan's own sample
                // accounting and the protocol this compiler measures under.
                let launches = evidence.winning_launches();
                if launches > self.search_work.measurements {
                    return Err(invalid(
                        "measurement.launches",
                        format!(
                            "winning finalist records {launches} launches but the search records only {} measurements",
                            self.search_work.measurements
                        ),
                        "derive winning launch count from the recorded search samples",
                    ));
                }
                if evidence.protocol.version != MEASUREMENT_PROTOCOL_VERSION {
                    return Err(invalid(
                        "measurement.protocol.version",
                        format!(
                            "samples were measured under protocol {} but this compiler measures under {MEASUREMENT_PROTOCOL_VERSION}",
                            evidence.protocol.version
                        ),
                        "re-measure the finalists under the current protocol",
                    ));
                }
            }
            PlanMeasurement::Unbudgeted | PlanMeasurement::UntimedDevice
                if self.search_work.measurements != 0 =>
            {
                return Err(invalid(
                    "measurement",
                    format!(
                        "unmeasured selection records {} on-device measurements",
                        self.search_work.measurements
                    ),
                    "record measured evidence when samples selected the plan, or record zero samples",
                ));
            }
            PlanMeasurement::Unbudgeted | PlanMeasurement::UntimedDevice => {}
        }
        Ok(())
    }
}
