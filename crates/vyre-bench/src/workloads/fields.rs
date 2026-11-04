//! Required measurement fields.
//!
//! The contract requires benchmarks to record:
//! 1. compile time
//! 2. candidate count
//! 3. prediction error
//! 4. emitted register metrics
//! 5. spill metrics
//! 6. shared-memory metrics
//! 7. raw device samples
//! 8. estimator and uncertainty
//! 9. device time
//! 10. p50 latency
//! 11. p99 latency
//! 12. throughput
//! 13. memory metrics
//! 14. workspace traffic
//! 15. cold and warm artifact behavior

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// The canonical enum of required benchmark measurement fields specified by row 47.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RequiredMeasurementField {
    /// Wall-clock compile time in nanoseconds.
    CompileTimeNs,
    /// Number of candidate schedules evaluated during schedule search.
    CandidateCount,
    /// Absolute or relative prediction error of the cost model against device execution.
    PredictionError,
    /// Emitted register allocation per invocation/thread.
    RegisterCount,
    /// Spill metrics including spill bytes, spill instruction count, and stack frame bytes.
    SpillMetrics,
    /// Shared memory metrics including static, dynamic, and total allocated bytes.
    SharedMemoryMetrics,
    /// Raw sample execution timings in nanoseconds.
    RawDeviceSamples,
    /// Statistical estimator and its uncertainty intervals (confidence interval, standard error, MAD).
    EstimatorAndUncertainty,
    /// Active device execution time in nanoseconds.
    DeviceTimeNs,
    /// 50th percentile (median) execution latency in nanoseconds.
    P50LatencyNs,
    /// 99th percentile execution latency in nanoseconds.
    P99LatencyNs,
    /// Measured throughput (GFLOP/s, GB/s, or items/second).
    Throughput,
    /// Memory metrics including allocated memory, peak memory, and resident VRAM bytes.
    MemoryMetrics,
    /// Workspace traffic metrics including bytes read, bytes written, and total bytes touched.
    WorkspaceTraffic,
    /// Cold-start versus warm steady-state artifact behavior and warmup ratio.
    ColdAndWarmArtifactBehavior,
}

impl RequiredMeasurementField {
    /// String identifier for this required field in reports and diagnostics.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::CompileTimeNs => "compile_time_ns",
            Self::CandidateCount => "candidate_count",
            Self::PredictionError => "prediction_error",
            Self::RegisterCount => "register_count",
            Self::SpillMetrics => "spill_metrics",
            Self::SharedMemoryMetrics => "shared_memory_metrics",
            Self::RawDeviceSamples => "raw_device_samples",
            Self::EstimatorAndUncertainty => "estimator_and_uncertainty",
            Self::DeviceTimeNs => "device_time_ns",
            Self::P50LatencyNs => "p50_latency_ns",
            Self::P99LatencyNs => "p99_latency_ns",
            Self::Throughput => "throughput",
            Self::MemoryMetrics => "memory_metrics",
            Self::WorkspaceTraffic => "workspace_traffic",
            Self::ColdAndWarmArtifactBehavior => "cold_and_warm_artifact_behavior",
        }
    }

    /// Complete exhaustive array of all row 47 required fields.
    pub const ALL: &'static [RequiredMeasurementField] = &[
        Self::CompileTimeNs,
        Self::CandidateCount,
        Self::PredictionError,
        Self::RegisterCount,
        Self::SpillMetrics,
        Self::SharedMemoryMetrics,
        Self::RawDeviceSamples,
        Self::EstimatorAndUncertainty,
        Self::DeviceTimeNs,
        Self::P50LatencyNs,
        Self::P99LatencyNs,
        Self::Throughput,
        Self::MemoryMetrics,
        Self::WorkspaceTraffic,
        Self::ColdAndWarmArtifactBehavior,
    ];

    /// Derive the set of required field names dynamically at runtime.
    #[must_use]
    pub fn required_field_names() -> BTreeSet<&'static str> {
        Self::ALL.iter().map(|field| field.as_str()).collect()
    }
}

/// Validation error when a measurement record is missing one or more row 47 required fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissingRequiredFieldsError {
    /// Benchmark case identifier.
    pub case_id: String,
    /// Names of missing required fields.
    pub missing_fields: Vec<String>,
}

impl std::fmt::Display for MissingRequiredFieldsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "case `{}` is missing required row 47 fields: {:?}",
            self.case_id, self.missing_fields
        )
    }
}

impl std::error::Error for MissingRequiredFieldsError {}
