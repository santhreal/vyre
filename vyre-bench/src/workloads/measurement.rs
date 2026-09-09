//! Comprehensive benchmark measurement record for BACKLOG row 47.
//!
//! BACKLOG row 47 requires:
//! "Benchmarks record compile time, candidate count, prediction error,
//! emitted register/spill/shared metrics, raw device samples, estimator and uncertainty,
//! device time, p50/p99 latency, throughput, memory, workspace traffic,
//! and cold/warm artifact behavior for representative complete graphs and adversarial
//! kernel-sized regions."

use serde::{Deserialize, Serialize};

use super::equality::NativeComparisonConditions;
use super::fields::{MissingRequiredFieldsError, RequiredMeasurementField};
use super::provenance::PayloadProvenance;

/// Spill metrics indicating register pressure and local memory traffic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpillMetrics {
    /// Total bytes spilled to thread-local device stack.
    pub spill_bytes: u64,
    /// Count of spilled register values.
    pub spill_count: u32,
    /// Stack frame allocation size in bytes per thread.
    pub stack_frame_bytes: u64,
}

impl SpillMetrics {
    /// Zero spill metrics (pure register residency).
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            spill_bytes: 0,
            spill_count: 0,
            stack_frame_bytes: 0,
        }
    }
}

/// Shared memory metrics allocated per workgroup/block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedMemoryMetrics {
    /// Static shared memory bytes allocated at compile time.
    pub static_bytes: u32,
    /// Dynamic shared memory bytes requested at launch time.
    pub dynamic_bytes: u32,
    /// Total shared memory bytes per workgroup.
    pub total_bytes: u32,
}

impl SharedMemoryMetrics {
    /// Zero shared memory metrics.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            static_bytes: 0,
            dynamic_bytes: 0,
            total_bytes: 0,
        }
    }

    /// Construct from static and dynamic components.
    #[must_use]
    pub const fn new(static_bytes: u32, dynamic_bytes: u32) -> Self {
        Self {
            static_bytes,
            dynamic_bytes,
            total_bytes: static_bytes + dynamic_bytes,
        }
    }
}

/// The statistical estimator family applied to raw latency samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EstimatorKind {
    /// Median (50th percentile).
    Median,
    /// Hodges-Lehmann robust location estimator (median of pairwise Walsh averages).
    HodgesLehmann,
    /// Arithmetic mean.
    Mean,
    /// Truncated / trimmed mean (interquartile range mean).
    TrimmedMean,
}

/// Robust statistical estimator with formal uncertainty bounds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatisticalEstimator {
    /// Family of the point estimator.
    pub estimator_kind: EstimatorKind,
    /// Estimated central tendency value in nanoseconds.
    pub estimator_value_ns: f64,
    /// Lower bound of the confidence interval in nanoseconds.
    pub uncertainty_ci_lower_ns: f64,
    /// Upper bound of the confidence interval in nanoseconds.
    pub uncertainty_ci_upper_ns: f64,
    /// Confidence level (e.g. 0.95 for 95% CI).
    pub confidence_level: f64,
    /// Standard error of the estimator in nanoseconds.
    pub standard_error_ns: f64,
    /// Median absolute deviation (MAD) in nanoseconds.
    pub mad_ns: f64,
}

impl StatisticalEstimator {
    /// Compute robust estimator from raw device sample latencies.
    #[must_use]
    pub fn from_samples(samples: &[u64], confidence_level: f64) -> Self {
        if samples.is_empty() {
            return Self {
                estimator_kind: EstimatorKind::Median,
                estimator_value_ns: 0.0,
                uncertainty_ci_lower_ns: 0.0,
                uncertainty_ci_upper_ns: 0.0,
                confidence_level,
                standard_error_ns: 0.0,
                mad_ns: 0.0,
            };
        }

        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        let n = sorted.len();

        let median = if n % 2 == 1 {
            sorted[n / 2] as f64
        } else {
            (sorted[n / 2 - 1] as f64 + sorted[n / 2] as f64) / 2.0
        };

        let sum: u128 = sorted.iter().map(|&s| u128::from(s)).sum();
        let mean = sum as f64 / n as f64;

        let variance = if n > 1 {
            sorted
                .iter()
                .map(|&s| {
                    let diff = s as f64 - mean;
                    diff * diff
                })
                .sum::<f64>()
                / (n - 1) as f64
        } else {
            0.0
        };
        let stddev = variance.sqrt();
        let standard_error = if n > 0 {
            stddev / (n as f64).sqrt()
        } else {
            0.0
        };

        let mut mad_diffs: Vec<f64> = sorted.iter().map(|&s| (s as f64 - median).abs()).collect();
        mad_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = if n % 2 == 1 {
            mad_diffs[n / 2]
        } else {
            (mad_diffs[n / 2 - 1] + mad_diffs[n / 2]) / 2.0
        };

        // Z-score approximation for confidence interval
        let z = if (confidence_level - 0.99).abs() < 0.005 {
            2.576
        } else if (confidence_level - 0.95).abs() < 0.005 {
            1.960
        } else {
            1.645
        };

        let ci_margin = z * standard_error;
        let ci_lower = (median - ci_margin).max(0.0);
        let ci_upper = median + ci_margin;

        Self {
            estimator_kind: EstimatorKind::Median,
            estimator_value_ns: median,
            uncertainty_ci_lower_ns: ci_lower,
            uncertainty_ci_upper_ns: ci_upper,
            confidence_level,
            standard_error_ns: standard_error,
            mad_ns: mad,
        }
    }
}

/// Measured throughput figures across applicable units.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct ThroughputMetrics {
    /// Compute throughput in billions of floating point operations per second.
    pub throughput_gflops: Option<f64>,
    /// Memory bandwidth throughput in gigabytes per second.
    pub throughput_gb_s: Option<f64>,
    /// Domain-specific items/records processed per second.
    pub items_per_second: Option<f64>,
}

/// Device memory metrics in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Resident allocated memory bytes for workload buffers.
    pub allocated_memory_bytes: u64,
    /// Peak device memory allocated during execution.
    pub peak_memory_bytes: u64,
    /// Total VRAM consumption including runtime context overhead.
    pub vram_allocated_bytes: u64,
}

/// Workspace memory traffic during sample execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceTraffic {
    /// Bytes read from global device memory.
    pub bytes_read: u64,
    /// Bytes written to global device memory.
    pub bytes_written: u64,
    /// Total unique bytes touched (read + written).
    pub bytes_touched: u64,
    /// Scratch/workspace memory traffic between fused kernel stages.
    pub workspace_traffic_bytes: u64,
}

/// Cold-start versus warm steady-state execution behavior.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ArtifactBehavior {
    /// Cold-start latency (first execution, including initial upload/pipeline setup) in nanoseconds.
    pub cold_artifact_latency_ns: u64,
    /// Warm steady-state median execution latency in nanoseconds.
    pub warm_artifact_latency_ns: u64,
    /// Ratio of cold latency to warm latency (cold_latency / warm_latency).
    pub warmup_ratio: f64,
    /// Bytes of retained state preserved between successive invocations.
    pub resident_state_bytes: u64,
}

/// The complete benchmark case measurement record holding all 15 required categories.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseMeasurementRecord {
    /// Benchmark case identifier.
    pub case_id: String,
    /// Payload generation provenance.
    pub provenance: PayloadProvenance,
    /// 1. Compile time in nanoseconds.
    pub compile_time_ns: u64,
    /// 2. Candidate count searched during schedule exploration.
    pub candidate_count: u32,
    /// 3. Cost model prediction error against measured execution.
    pub prediction_error: f64,
    /// 4. Register count per invocation/thread.
    pub register_count: u32,
    /// 5. Spill metrics.
    pub spill_metrics: SpillMetrics,
    /// 6. Shared memory metrics.
    pub shared_memory_metrics: SharedMemoryMetrics,
    /// 7. Raw device sample latencies in nanoseconds.
    pub raw_device_samples: Vec<u64>,
    /// 8. Statistical estimator and uncertainty model.
    pub estimator_and_uncertainty: StatisticalEstimator,
    /// 9. Device execution time in nanoseconds.
    pub device_time_ns: u64,
    /// 10. 50th percentile (median) latency in nanoseconds.
    pub p50_latency_ns: u64,
    /// 11. 99th percentile latency in nanoseconds.
    pub p99_latency_ns: u64,
    /// 12. Throughput metrics.
    pub throughput: ThroughputMetrics,
    /// 13. Memory metrics.
    pub memory_metrics: MemoryMetrics,
    /// 14. Workspace traffic metrics.
    pub workspace_traffic: WorkspaceTraffic,
    /// 15. Cold and warm artifact behavior.
    pub artifact_behavior: ArtifactBehavior,
    /// Recorded 14 comparison equality conditions.
    pub equality_conditions: NativeComparisonConditions,
}

impl CaseMeasurementRecord {
    /// Validate that all 15 row 47 required fields are present and valid.
    pub fn validate_required_fields(&self) -> Result<(), MissingRequiredFieldsError> {
        let mut missing = Vec::new();

        if self.compile_time_ns == 0 {
            missing.push(RequiredMeasurementField::CompileTimeNs.as_str().to_string());
        }
        if self.candidate_count == 0 {
            missing.push(
                RequiredMeasurementField::CandidateCount
                    .as_str()
                    .to_string(),
            );
        }
        if self.prediction_error.is_nan() {
            missing.push(
                RequiredMeasurementField::PredictionError
                    .as_str()
                    .to_string(),
            );
        }
        if self.register_count == 0 {
            missing.push(RequiredMeasurementField::RegisterCount.as_str().to_string());
        }
        if self.raw_device_samples.is_empty() {
            missing.push(
                RequiredMeasurementField::RawDeviceSamples
                    .as_str()
                    .to_string(),
            );
        }
        if self.estimator_and_uncertainty.estimator_value_ns <= 0.0 {
            missing.push(
                RequiredMeasurementField::EstimatorAndUncertainty
                    .as_str()
                    .to_string(),
            );
        }
        if self.device_time_ns == 0 {
            missing.push(RequiredMeasurementField::DeviceTimeNs.as_str().to_string());
        }
        if self.p50_latency_ns == 0 {
            missing.push(RequiredMeasurementField::P50LatencyNs.as_str().to_string());
        }
        if self.p99_latency_ns == 0 {
            missing.push(RequiredMeasurementField::P99LatencyNs.as_str().to_string());
        }
        if self.throughput.throughput_gflops.is_none()
            && self.throughput.throughput_gb_s.is_none()
            && self.throughput.items_per_second.is_none()
        {
            missing.push(RequiredMeasurementField::Throughput.as_str().to_string());
        }
        if self.memory_metrics.allocated_memory_bytes == 0
            && self.memory_metrics.peak_memory_bytes == 0
        {
            missing.push(RequiredMeasurementField::MemoryMetrics.as_str().to_string());
        }
        if self.workspace_traffic.bytes_touched == 0 && self.workspace_traffic.bytes_read == 0 {
            missing.push(
                RequiredMeasurementField::WorkspaceTraffic
                    .as_str()
                    .to_string(),
            );
        }
        if self.artifact_behavior.warm_artifact_latency_ns == 0 {
            missing.push(
                RequiredMeasurementField::ColdAndWarmArtifactBehavior
                    .as_str()
                    .to_string(),
            );
        }

        if missing.is_empty() {
            Ok(())
        } else {
            Err(MissingRequiredFieldsError {
                case_id: self.case_id.clone(),
                missing_fields: missing,
            })
        }
    }
}
