//! Per-cell verdict engine and statistical evaluation for BACKLOG row 47.
//!
//! BACKLOG row 47 requires:
//! "Report a per-cell verdict, not an aggregate: a regression on one target,
//! sequence layout, or retained-state mode is a finding even when the mean improves.
//! Statistically indistinguishable is its own verdict, never a win."
//! "The final route beats the prior Vyre artifact and is competitive with or faster
//! than the best available native baseline on every claimed target and workload
//! outside the declared measurement-equivalence band; evidence reports losses and
//! indistinguishable cases instead of averaging them away."

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::equality::{validate_equality_conditions, EqualityConditionRefusal};
use super::measurement::CaseMeasurementRecord;
use super::native_baseline::VersionPinnedNativeBaseline;
use super::provenance::validate_payload_provenance;

/// A single discrete benchmark execution cell.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MeasurementCell {
    /// Workload identifier.
    pub workload_id: String,
    /// Hardware target device architecture.
    pub target: String,
    /// Sequence layout mode (e.g. "uniform_contiguous", "ragged_power_law", "interleaved").
    pub sequence_layout: String,
    /// Retained state execution mode (e.g. "stateless", "retained_resident_pool", "streaming_pipeline").
    pub retained_state_mode: String,
}

impl MeasurementCell {
    /// Create a new measurement cell descriptor.
    #[must_use]
    pub fn new(
        workload_id: impl Into<String>,
        target: impl Into<String>,
        sequence_layout: impl Into<String>,
        retained_state_mode: impl Into<String>,
    ) -> Self {
        Self {
            workload_id: workload_id.into(),
            target: target.into(),
            sequence_layout: sequence_layout.into(),
            retained_state_mode: retained_state_mode.into(),
        }
    }
}

/// The discrete outcome verdict for an individual measurement cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CellVerdict {
    /// Vyre is strictly and statistically significantly faster than the native baseline.
    Win {
        /// Measured speedup ratio (native_time / vyre_time).
        speedup_x: f64,
        /// Latency delta in nanoseconds (native_time - vyre_time).
        delta_ns: f64,
        /// Statistical confidence level (e.g. 0.95).
        confidence_level: f64,
    },
    /// Vyre is statistically significantly slower than the native baseline (loss/regression).
    Loss {
        /// Measured speedup ratio (< 1.0).
        speedup_x: f64,
        /// Regression percentage (e.g. 15.0 for 15% slower).
        regression_pct: f64,
        /// Statistical confidence level.
        confidence_level: f64,
    },
    /// Explicit regression against declared performance floor or prior artifact.
    Regression {
        /// Cell identifier.
        cell_id: String,
        /// Measured speedup ratio.
        speedup_x: f64,
        /// Regression percentage.
        regression_pct: f64,
        /// Diagnostic reason.
        reason: String,
    },
    /// Measurement is within the declared equivalence band; statistically indistinguishable.
    ///
    /// Invariant: a statistically indistinguishable result is never classified as a Win.
    StatisticallyIndistinguishable {
        /// Measured speedup ratio.
        speedup_x: f64,
        /// 95% confidence interval for the speedup ratio.
        confidence_interval: (f64, f64),
        /// Equivalence band tolerance percentage (e.g. 2.5%).
        equivalence_band_pct: f64,
    },
    /// Refusal: Measured payload did not originate from generic IR and schedule search.
    RefusedNonIrPayload {
        /// Case identifier.
        case_id: String,
        /// The invalid payload provenance kind.
        provenance_kind: String,
        /// Diagnostic refusal reason.
        reason: String,
    },
    /// Refusal: A required comparison equality condition is unset.
    RefusedUnsetEqualityCondition {
        /// Case identifier.
        case_id: String,
        /// The missing equality dimension.
        missing_dimension: String,
        /// Diagnostic reason.
        reason: String,
    },
    /// Refusal: A comparison equality condition differed between Vyre and native baseline.
    RefusedMismatchedEqualityCondition {
        /// Case identifier.
        case_id: String,
        /// The mismatched dimension.
        dimension: String,
        /// Vyre condition value.
        vyre_value: String,
        /// Native baseline condition value.
        native_value: String,
        /// Diagnostic reason.
        reason: String,
    },
}

impl CellVerdict {
    /// Whether this verdict represents a clean win.
    #[must_use]
    pub const fn is_win(&self) -> bool {
        matches!(self, Self::Win { .. })
    }

    /// Whether this verdict is statistically indistinguishable.
    #[must_use]
    pub const fn is_indistinguishable(&self) -> bool {
        matches!(self, Self::StatisticallyIndistinguishable { .. })
    }

    /// Whether this verdict represents a regression or loss.
    #[must_use]
    pub const fn is_regression_or_loss(&self) -> bool {
        matches!(self, Self::Loss { .. } | Self::Regression { .. })
    }

    /// Whether this verdict was refused due to invalid provenance or condition violations.
    #[must_use]
    pub const fn is_refusal(&self) -> bool {
        matches!(
            self,
            Self::RefusedNonIrPayload { .. }
                | Self::RefusedUnsetEqualityCondition { .. }
                | Self::RefusedMismatchedEqualityCondition { .. }
        )
    }
}

/// Evaluation record for one benchmark measurement cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellEvaluation {
    /// The measurement cell coordinate.
    pub cell: MeasurementCell,
    /// Measured Vyre case record.
    pub vyre_measurement: Option<CaseMeasurementRecord>,
    /// Pinned native baseline.
    pub native_baseline: Option<VersionPinnedNativeBaseline>,
    /// Computed cell verdict.
    pub verdict: CellVerdict,
}

/// Overall aggregate verdict for a suite or campaign.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AggregateVerdict {
    /// All cells passed without regressions.
    Pass {
        /// Number of cells that won.
        wins: usize,
        /// Number of cells that are statistically indistinguishable.
        indistinguishable: usize,
        /// Mean speedup ratio across cells.
        mean_speedup_x: f64,
    },
    /// One or more cells regressed (even if the overall mean speedup improved).
    Regression {
        /// Cells that suffered regressions.
        regressed_cells: Vec<MeasurementCell>,
        /// Mean speedup ratio across all cells (which may be > 1.0!).
        mean_speedup_x: f64,
        /// Summary diagnostic.
        reason: String,
    },
    /// All cells are statistically indistinguishable within the equivalence band.
    Indistinguishable {
        /// Number of evaluated cells.
        cell_count: usize,
        /// Mean speedup ratio.
        mean_speedup_x: f64,
    },
    /// One or more cells were refused due to contract violations.
    Refusal {
        /// List of refusal diagnostics.
        refusal_reasons: Vec<String>,
    },
}

/// Report containing per-cell evaluations and the overall aggregate verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FloorComparisonReport {
    /// Evaluated cells.
    pub cells: BTreeMap<MeasurementCell, CellEvaluation>,
    /// Mean speedup ratio across all valid measured cells.
    pub mean_speedup_x: f64,
    /// Computed overall verdict adhering to the per-cell strictness rule.
    pub overall_verdict: AggregateVerdict,
}

/// Evaluate a single cell comparison between Vyre measurement and native baseline.
pub fn evaluate_cell_comparison(
    cell: &MeasurementCell,
    vyre: &CaseMeasurementRecord,
    native: &VersionPinnedNativeBaseline,
    equivalence_band_pct: f64,
) -> CellEvaluation {
    // 1. Provenance check: must be generic IR + schedule search
    if let Err(refusal) = validate_payload_provenance(&vyre.case_id, &vyre.provenance) {
        return CellEvaluation {
            cell: cell.clone(),
            vyre_measurement: Some(vyre.clone()),
            native_baseline: Some(native.clone()),
            verdict: CellVerdict::RefusedNonIrPayload {
                case_id: vyre.case_id.clone(),
                provenance_kind: refusal.provenance.kind_str().to_string(),
                reason: refusal.reason,
            },
        };
    }

    // 2. Equality conditions check: all 14 dimensions must match
    if let Err(refusal) = validate_equality_conditions(
        &vyre.case_id,
        &vyre.equality_conditions,
        &native.equality_conditions,
    ) {
        let verdict = match refusal {
            EqualityConditionRefusal::Unset {
                case_id,
                dimension,
                reason,
            } => CellVerdict::RefusedUnsetEqualityCondition {
                case_id,
                missing_dimension: dimension.as_str().to_string(),
                reason,
            },
            EqualityConditionRefusal::Mismatched {
                case_id,
                dimension,
                vyre_value,
                native_value,
                reason,
            } => CellVerdict::RefusedMismatchedEqualityCondition {
                case_id,
                dimension: dimension.as_str().to_string(),
                vyre_value,
                native_value,
                reason,
            },
        };
        return CellEvaluation {
            cell: cell.clone(),
            vyre_measurement: Some(vyre.clone()),
            native_baseline: Some(native.clone()),
            verdict,
        };
    }

    // 3. Compare empirical execution times
    let native_meas = match &native.measurement {
        Some(m) => m,
        None => {
            return CellEvaluation {
                cell: cell.clone(),
                vyre_measurement: Some(vyre.clone()),
                native_baseline: Some(native.clone()),
                verdict: CellVerdict::Regression {
                    cell_id: format!("{cell:?}"),
                    speedup_x: 0.0,
                    regression_pct: 100.0,
                    reason: "native baseline has no recorded measurement".to_string(),
                },
            };
        }
    };

    let vyre_time = vyre.estimator_and_uncertainty.estimator_value_ns;
    let native_time = native_meas.estimator_and_uncertainty.estimator_value_ns;

    if vyre_time <= 0.0 || native_time <= 0.0 {
        return CellEvaluation {
            cell: cell.clone(),
            vyre_measurement: Some(vyre.clone()),
            native_baseline: Some(native.clone()),
            verdict: CellVerdict::Regression {
                cell_id: format!("{cell:?}"),
                speedup_x: 0.0,
                regression_pct: 100.0,
                reason: "non-positive execution time recorded".to_string(),
            },
        };
    }

    let speedup_x = native_time / vyre_time;
    let delta_ns = native_time - vyre_time;

    // Statistical equivalence band test
    let tolerance = equivalence_band_pct / 100.0;
    let lower_band = 1.0 - tolerance;
    let upper_band = 1.0 + tolerance;

    // Check uncertainty bounds overlap
    let vyre_ci_lower = vyre.estimator_and_uncertainty.uncertainty_ci_lower_ns;
    let vyre_ci_upper = vyre.estimator_and_uncertainty.uncertainty_ci_upper_ns;
    let native_ci_lower = native_meas
        .estimator_and_uncertainty
        .uncertainty_ci_lower_ns;
    let native_ci_upper = native_meas
        .estimator_and_uncertainty
        .uncertainty_ci_upper_ns;

    let ci_overlap = (vyre_ci_lower <= native_ci_upper) && (native_ci_lower <= vyre_ci_upper);
    let within_band = speedup_x >= lower_band && speedup_x <= upper_band;

    let verdict = if within_band || ci_overlap {
        let ci_ratio_lower = if vyre_ci_upper > 0.0 {
            native_ci_lower / vyre_ci_upper
        } else {
            speedup_x
        };
        let ci_ratio_upper = if vyre_ci_lower > 0.0 {
            native_ci_upper / vyre_ci_lower
        } else {
            speedup_x
        };
        CellVerdict::StatisticallyIndistinguishable {
            speedup_x,
            confidence_interval: (ci_ratio_lower, ci_ratio_upper),
            equivalence_band_pct,
        }
    } else if speedup_x > 1.0 {
        CellVerdict::Win {
            speedup_x,
            delta_ns,
            confidence_level: vyre.estimator_and_uncertainty.confidence_level,
        }
    } else {
        let regression_pct = (1.0 - speedup_x) * 100.0;
        CellVerdict::Loss {
            speedup_x,
            regression_pct,
            confidence_level: vyre.estimator_and_uncertainty.confidence_level,
        }
    };

    CellEvaluation {
        cell: cell.clone(),
        vyre_measurement: Some(vyre.clone()),
        native_baseline: Some(native.clone()),
        verdict,
    }
}

/// Generate an aggregate comparison report enforcing that any per-cell regression
/// turns the overall verdict into a regression finding, regardless of mean speedup.
pub fn generate_floor_comparison_report(evaluations: Vec<CellEvaluation>) -> FloorComparisonReport {
    let mut cells = BTreeMap::new();
    let mut speedup_sum = 0.0;
    let mut measured_count = 0;
    let mut regressed_cells = Vec::new();
    let mut refusal_reasons = Vec::new();
    let mut win_count = 0;
    let mut indistinguishable_count = 0;

    for eval in evaluations {
        match &eval.verdict {
            CellVerdict::Win { speedup_x, .. } => {
                win_count += 1;
                speedup_sum += *speedup_x;
                measured_count += 1;
            }
            CellVerdict::StatisticallyIndistinguishable { speedup_x, .. } => {
                indistinguishable_count += 1;
                speedup_sum += *speedup_x;
                measured_count += 1;
            }
            CellVerdict::Loss { speedup_x, .. } => {
                regressed_cells.push(eval.cell.clone());
                speedup_sum += *speedup_x;
                measured_count += 1;
            }
            CellVerdict::Regression { speedup_x, .. } => {
                regressed_cells.push(eval.cell.clone());
                speedup_sum += *speedup_x;
                measured_count += 1;
            }
            CellVerdict::RefusedNonIrPayload { reason, .. } => {
                refusal_reasons.push(reason.clone());
            }
            CellVerdict::RefusedUnsetEqualityCondition { reason, .. } => {
                refusal_reasons.push(reason.clone());
            }
            CellVerdict::RefusedMismatchedEqualityCondition { reason, .. } => {
                refusal_reasons.push(reason.clone());
            }
        }
        cells.insert(eval.cell.clone(), eval);
    }

    let mean_speedup_x = if measured_count > 0 {
        speedup_sum / measured_count as f64
    } else {
        0.0
    };

    // Strict aggregate verdict logic:
    // 1. If any refusal exists -> Refusal
    // 2. If any single cell regressed -> Regression (EVEN IF mean_speedup_x > 1.0)
    // 3. If all cells are indistinguishable -> Indistinguishable (NEVER Win)
    // 4. Otherwise Pass
    let overall_verdict = if !refusal_reasons.is_empty() {
        AggregateVerdict::Refusal { refusal_reasons }
    } else if !regressed_cells.is_empty() {
        AggregateVerdict::Regression {
            regressed_cells: regressed_cells.clone(),
            mean_speedup_x,
            reason: format!(
                "{} cell(s) regressed against the native baseline floor despite mean speedup {:.2}x",
                regressed_cells.len(),
                mean_speedup_x
            ),
        }
    } else if win_count == 0 && indistinguishable_count > 0 {
        AggregateVerdict::Indistinguishable {
            cell_count: indistinguishable_count,
            mean_speedup_x,
        }
    } else {
        AggregateVerdict::Pass {
            wins: win_count,
            indistinguishable: indistinguishable_count,
            mean_speedup_x,
        }
    };

    FloorComparisonReport {
        cells,
        mean_speedup_x,
        overall_verdict,
    }
}
