//! Floor validation and unmeasured floor refusal.
//!
//! BACKLOG row 95 requires:
//! A known blocker: one registered case, the resident optimizer pipeline, carries
//! the floor `min_speedup_over_baseline = 0.10` in `docs/optimization/BENCH_TARGETS.toml`,
//! which was written before any device measured it. Make an unmeasured floor impossible to
//! record: a case whose floor has no recorded measurement behind it must be refused by
//! name rather than judged against an invented number.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::receipt::BenchmarkReceipt;
use super::store::{EvidenceStore, EvidenceStoreError};

/// Structured refusal when a benchmark target floor has no recorded measurement backing it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Error)]
#[error("refused: benchmark target floor for `{case_id}` ({declared_floor}x) has no recorded measurement behind it: {reason}")]
pub struct UnmeasuredFloorRefusal {
    /// Benchmark case identifier.
    pub case_id: String,
    /// The declared floor value that was attempted to be judged or recorded.
    pub declared_floor: f64,
    /// Detailed reason why the floor is unmeasured and refused.
    pub reason: String,
}

/// A verified proof of an empirical, recorded baseline measurement establishing a performance floor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordedFloorProof {
    /// Benchmark case identifier.
    pub case_id: String,
    /// Baseline class name (e.g. "cpu_sota", "self_unoptimized").
    pub baseline_class: String,
    /// Measured floor value derived from empirical runs.
    pub floor_value: f64,
    /// Content address of the backing benchmark receipt in the evidence store.
    pub backing_measurement_address: String,
    /// Number of verified sample executions backing the floor.
    pub sample_count: usize,
    /// Physical device on which the floor was calibrated.
    pub device_name: String,
    /// Whether exact reference parity was verified for the calibration measurement.
    pub verified_parity: bool,
}

/// Validate that a declared benchmark target floor is backed by a recorded measurement.
///
/// Refuses by name if:
/// 1. No backing proof is supplied (unmeasured constant).
/// 2. The backing proof does not match the case ID.
/// 3. The backing measurement receipt is not present in the content-addressed evidence store.
/// 4. The backing measurement receipt failed parity or has fewer than 1 sample.
pub fn validate_floor_has_recorded_measurement(
    case_id: &str,
    declared_floor: f64,
    backing_proof: Option<&RecordedFloorProof>,
    store: &EvidenceStore,
) -> Result<RecordedFloorProof, UnmeasuredFloorRefusal> {
    let Some(proof) = backing_proof else {
        return Err(UnmeasuredFloorRefusal {
            case_id: case_id.to_string(),
            declared_floor,
            reason: format!(
                "Fix: floor {declared_floor}x for `{case_id}` has no recorded measurement receipt; invented floors are refused"
            ),
        });
    };

    if proof.case_id != case_id {
        return Err(UnmeasuredFloorRefusal {
            case_id: case_id.to_string(),
            declared_floor,
            reason: format!(
                "Fix: backing floor proof is for case `{}` but target is `{case_id}`",
                proof.case_id
            ),
        });
    }

    if !proof.verified_parity {
        return Err(UnmeasuredFloorRefusal {
            case_id: case_id.to_string(),
            declared_floor,
            reason: format!(
                "Fix: backing measurement `{}` for case `{case_id}` failed parity verification",
                proof.backing_measurement_address
            ),
        });
    }

    if proof.sample_count == 0 {
        return Err(UnmeasuredFloorRefusal {
            case_id: case_id.to_string(),
            declared_floor,
            reason: format!(
                "Fix: backing measurement `{}` for case `{case_id}` contains zero samples",
                proof.backing_measurement_address
            ),
        });
    }

    // Verify receipt in evidence store
    let receipt = match store.get(&proof.backing_measurement_address) {
        Ok(r) => r,
        Err(EvidenceStoreError::NotFound(_)) => {
            return Err(UnmeasuredFloorRefusal {
                case_id: case_id.to_string(),
                declared_floor,
                reason: format!(
                    "Fix: backing measurement receipt `{}` not found in evidence store",
                    proof.backing_measurement_address
                ),
            });
        }
        Err(err) => {
            return Err(UnmeasuredFloorRefusal {
                case_id: case_id.to_string(),
                declared_floor,
                reason: format!("Fix: failed to read backing receipt: {err}"),
            });
        }
    };

    if !receipt.parity.passed {
        return Err(UnmeasuredFloorRefusal {
            case_id: case_id.to_string(),
            declared_floor,
            reason: format!(
                "Fix: evidence receipt `{}` failed parity check (max error={})",
                proof.backing_measurement_address, receipt.parity.max_absolute_error
            ),
        });
    }

    Ok(proof.clone())
}

/// Record a measured benchmark floor from an empirical receipt into the evidence store.
///
/// Computes the floor value from the empirical speedup ratio with an admissible margin,
/// binds it to the receipt's cryptographic content address, and returns a verified proof.
pub fn record_measured_floor(
    case_id: &str,
    baseline_class: &str,
    receipt: &BenchmarkReceipt,
    store: &EvidenceStore,
    tolerance_margin: f64,
) -> Result<RecordedFloorProof, UnmeasuredFloorRefusal> {
    if !receipt.parity.passed {
        return Err(UnmeasuredFloorRefusal {
            case_id: case_id.to_string(),
            declared_floor: receipt.native_baseline.speedup_ratio,
            reason: "receipt failed parity check against reference oracle".to_string(),
        });
    }

    if receipt.raw_samples.is_empty() {
        return Err(UnmeasuredFloorRefusal {
            case_id: case_id.to_string(),
            declared_floor: receipt.native_baseline.speedup_ratio,
            reason: "receipt contains zero sample timings".to_string(),
        });
    }

    let address = store.put(receipt).map_err(|err| UnmeasuredFloorRefusal {
        case_id: case_id.to_string(),
        declared_floor: receipt.native_baseline.speedup_ratio,
        reason: format!("failed to store measurement receipt: {err}"),
    })?;

    let base_speedup = receipt.native_baseline.speedup_ratio;
    let floor_value = (base_speedup * (1.0 - tolerance_margin.clamp(0.0, 0.99))).max(0.0001);

    Ok(RecordedFloorProof {
        case_id: case_id.to_string(),
        baseline_class: baseline_class.to_string(),
        floor_value,
        backing_measurement_address: address,
        sample_count: receipt.raw_samples.len(),
        device_name: receipt.target_facts.device_name.clone(),
        verified_parity: receipt.parity.passed,
    })
}
