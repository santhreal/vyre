//! Whole-application workload definitions, typed measurement records, and evidence contracts.
//!
//! A whole-application record states what one complete representative
//! application did on one acquired device: parity against the reference
//! evaluator, compile and load time, p50 and p99 latency, throughput, peak and
//! resident bytes, cold and warm state, the selected schedule identity, and the
//! comparison against a version-pinned native baseline when one is measured on
//! the host. Three unrelated domains are covered: dense numerical work,
//! irregular stateful work, and latency-sensitive interactive work. A proxy or
//! an isolated kernel is refused instead of recorded.
//!
//! Every latency figure comes from a submission the device completed through
//! the production route. The reference evaluator produces the parity oracle and
//! nothing else, and a host with no dispatch device produces a refusal instead
//! of a record.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use vyre::compiler::{
    attach_target, compile, CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts,
    ObjectiveMetric, ResourceLifetime, SearchBudget, TargetCompiler,
};
use vyre::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, GraphValueId, Node, Program,
    ProgramGraph, ValueContract, ValueLifetime,
};
use vyre_driver::{
    acquire, acquire_preferred_dispatch_backend, backend_registration, ArtifactMaterializer,
    BackendRegistration, BoundResource, Completion,
};
use vyre_reference::value::Value;
use vyre_runtime::artifact_admission::{ArtifactSession, TypedResource, TypedResourceDataset};

use super::equality::{validate_equality_conditions, NativeComparisonConditions, WorkloadFacts};
use super::native_baseline::{whole_application_native_baselines, VersionPinnedNativeBaseline};
use super::provenance::MeasurementProvenance;
use crate::api::metric::elapsed_ns;

mod catalog;
mod evidence_suite;
mod record_fields;
mod workload;

pub use catalog::{
    all_whole_application_workloads, dense_numerical_pipeline, interactive_event_pipeline,
    irregular_stateful_traversal,
};
pub use evidence_suite::{
    generate_whole_application_evidence_suite, write_whole_application_evidence_artifacts,
    WholeApplicationDomainMatrixRecord,
};
pub use record_fields::{RecordFieldSource, WholeApplicationRecordField};
pub use workload::WholeApplicationWorkload;

/// Schema version for whole-application measurement records.
pub const WHOLE_APPLICATION_RECORD_SCHEMA_V2: &str = "vyre.whole-application-record.v2";

/// Superseded record schema, retained so a stale record is refused by name.
///
/// A v1 record stated a device latency taken from the host reference evaluator,
/// a native baseline derived from that latency, and a constant host string. None
/// of those survive the current shape, so a v1 record fails closed rather than
/// decoding into it.
pub const WHOLE_APPLICATION_RECORD_SCHEMA_V1: &str = "vyre.whole-application-record.v1";

/// Smallest measured sample count a recorded latency distribution may rest on.
pub const MIN_MEASURED_SAMPLES: usize = 30;

/// Half-width of the band inside which a comparison is indistinguishable.
///
/// A ratio inside `1.0 ± this` is reported as indistinguishable, never as a win.
pub const COMPARISON_EQUIVALENCE_BAND: f64 = 0.05;

/// Application domain classification for representative whole applications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ApplicationDomain {
    /// Dense numerical linear algebra, matrix contractions, activations, and normalization pipelines.
    DenseNumerical,
    /// Irregular graph, sparse matrix-vector multiplications, and stateful persistent dataflow.
    IrregularStateful,
    /// Latency-sensitive interactive streaming, spatial layout, and multi-layer rendering pipelines.
    LatencySensitiveInteractive,
}

impl ApplicationDomain {
    /// All canonical application domains. Derived programmatically at run time.
    pub const ALL: [Self; 3] = [
        Self::DenseNumerical,
        Self::IrregularStateful,
        Self::LatencySensitiveInteractive,
    ];

    /// Stable string identifier for this domain.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DenseNumerical => "dense_numerical",
            Self::IrregularStateful => "irregular_stateful",
            Self::LatencySensitiveInteractive => "latency_sensitive_interactive",
        }
    }

    /// Human-readable title for this domain.
    #[must_use]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::DenseNumerical => "Dense Numerical Contraction",
            Self::IrregularStateful => "Irregular Stateful Dataflow",
            Self::LatencySensitiveInteractive => "Latency-Sensitive Interactive Pipeline",
        }
    }

    /// Summary description of domain workload characteristics.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::DenseNumerical => {
                "End-to-end multi-stage tensor contraction, normalization, and activation pipeline"
            }
            Self::IrregularStateful => {
                "Irregular CSR sparse graph traversal and degree-weighted stateful accumulator"
            }
            Self::LatencySensitiveInteractive => {
                "Interactive dirty-region culling, spatial transform, and multi-layer blend raster"
            }
        }
    }
}

/// Refusal errors when a candidate case fails whole-application verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
pub enum WholeApplicationRefusal {
    /// Rejection: single-node graph or wrapped kernel cannot satisfy whole-application class.
    #[error(
        "Whole-application workload `{workload_id}` is refused: single-node graph with {node_count} node(s) is an isolated kernel, not a whole-application graph. Whole applications require at least 2 connected nodes."
    )]
    SingleNodeIsolatedKernel {
        /// Workload identifier.
        workload_id: String,
        /// Observed node count.
        node_count: usize,
    },
    /// Rejection: disconnected graph lacking internal dataflow edges.
    #[error(
        "Whole-application workload `{workload_id}` is refused: disconnected graph with {edge_count} internal dataflow edges. Whole applications require at least 1 connected edge between nodes."
    )]
    DisconnectedGraph {
        /// Workload identifier.
        workload_id: String,
        /// Observed internal edge count.
        edge_count: usize,
    },
    /// Rejection: invalid or unvalidated CompileRequest.
    #[error(
        "Whole-application workload `{workload_id}` is refused: did not compile through a validated CompileRequest: {reason}"
    )]
    InvalidCompileRequest {
        /// Workload identifier.
        workload_id: String,
        /// Rejection reason.
        reason: String,
    },
    /// Rejection: did not execute through the canonical production route.
    #[error(
        "Whole-application workload `{workload_id}` is refused: did not execute through production route (CompileRequest -> ArtifactEnvelope -> TargetPayload -> ArtifactInstance -> BindingSet -> Completion): {reason}"
    )]
    NonProductionExecutionRoute {
        /// Workload identifier.
        workload_id: String,
        /// Rejection reason.
        reason: String,
    },
    /// Rejection: missing or invalid native baseline comparator on identical inputs.
    #[error(
        "Whole-application workload `{workload_id}` is refused: missing or invalid native baseline comparator on identical inputs: {reason}"
    )]
    MissingNativeBaseline {
        /// Workload identifier.
        workload_id: String,
        /// Rejection reason.
        reason: String,
    },
    /// Rejection: stale or unsupported schema version.
    #[error(
        "Whole-application record `{workload_id}` is refused: stale or unsupported schema version `{version}`. Expected `{expected}`. Stale records fail closed."
    )]
    StaleSchemaVersion {
        /// Workload identifier.
        workload_id: String,
        /// Stale version string observed.
        version: String,
        /// Expected schema version.
        expected: String,
    },
    /// Rejection: missing required measurement field.
    #[error(
        "Whole-application record `{workload_id}` is refused: missing required field `{field}`: {reason}"
    )]
    MissingRequiredField {
        /// Workload identifier.
        workload_id: String,
        /// Missing field name.
        field: String,
        /// Rejection reason.
        reason: String,
    },
    /// Rejection: this host acquired no dispatch device to execute on.
    #[error(
        "Whole-application measurement is refused: no dispatch device is available on this host: {reason}. A whole-application record states device performance, so a host without a device records nothing. Fix: run the producer on a host with a linked dispatch backend and a working driver, or pass an explicit backend id that acquires."
    )]
    NoDispatchDevice {
        /// Acquisition diagnostic from the driver registry.
        reason: String,
    },
    /// Rejection: the acquired backend is a conformance oracle, not a device.
    #[error(
        "Whole-application measurement is refused: backend `{backend_id}` is a reference conformance oracle and executes on the host. Fix: select a dispatch backend that executes on a device."
    )]
    ReferenceOracleBackend {
        /// Backend identifier that was acquired.
        backend_id: String,
    },
    /// Rejection: device outputs and reference outputs did not agree.
    #[error(
        "Whole-application workload `{workload_id}` is refused: device outputs did not match the reference evaluator on identical inputs: {reason}"
    )]
    ParityMismatch {
        /// Workload identifier.
        workload_id: String,
        /// Comparison diagnostic.
        reason: String,
    },
}

/// Canonical required fields that must be present in every whole-application measurement record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RequiredWholeApplicationField {
    /// Current record schema version string.
    SchemaVersion,
    /// Workload domain classification.
    Domain,
    /// Multi-node graph topology (node_count >= 2, edge_count >= 1).
    MultiNodeGraphTopology,
    /// Execution through production route (CompileRequest -> ArtifactEnvelope -> TargetPayload -> ArtifactInstance -> BindingSet -> Completion).
    ProductionRouteExecution,
    /// Parity verification against vyre-reference on identical inputs.
    ParityAgainstReference,
    /// Compile time in nanoseconds.
    CompileTimeNs,
    /// Module load / materialization time in nanoseconds.
    LoadTimeNs,
    /// 50th percentile (median) latency in nanoseconds.
    P50LatencyNs,
    /// 99th percentile latency in nanoseconds.
    P99LatencyNs,
    /// Throughput metrics (GFLOP/s, GB/s, or items/s).
    Throughput,
    /// Peak device memory allocated in bytes.
    PeakBytes,
    /// Retained / resident memory in bytes.
    ResidentBytes,
    /// Cold-start versus warm steady-state latency and memory behavior.
    ColdAndWarmState,
    /// Selected schedule identifier from schedule search.
    SelectedScheduleIdentity,
    /// Native baseline comparison on identical input identity, or the recorded
    /// reason the pinned baseline was not measured on this host.
    NativeBaselineComparison,
    /// 14-point comparison equality conditions the measured comparison held.
    EqualityConditions,
}

impl RequiredWholeApplicationField {
    /// All canonical required fields. Derived programmatically at run time.
    pub const ALL: [Self; 16] = [
        Self::SchemaVersion,
        Self::Domain,
        Self::MultiNodeGraphTopology,
        Self::ProductionRouteExecution,
        Self::ParityAgainstReference,
        Self::CompileTimeNs,
        Self::LoadTimeNs,
        Self::P50LatencyNs,
        Self::P99LatencyNs,
        Self::Throughput,
        Self::PeakBytes,
        Self::ResidentBytes,
        Self::ColdAndWarmState,
        Self::SelectedScheduleIdentity,
        Self::NativeBaselineComparison,
        Self::EqualityConditions,
    ];

    /// String name of the required field.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SchemaVersion => "schema_version",
            Self::Domain => "domain",
            Self::MultiNodeGraphTopology => "multi_node_graph_topology",
            Self::ProductionRouteExecution => "production_route",
            Self::ParityAgainstReference => "parity_against_reference",
            Self::CompileTimeNs => "compile_time_ns",
            Self::LoadTimeNs => "load_time_ns",
            Self::P50LatencyNs => "p50_latency_ns",
            Self::P99LatencyNs => "p99_latency_ns",
            Self::Throughput => "throughput",
            Self::PeakBytes => "peak_bytes",
            Self::ResidentBytes => "resident_bytes",
            Self::ColdAndWarmState => "cold_and_warm_state",
            Self::SelectedScheduleIdentity => "selected_schedule_identity",
            Self::NativeBaselineComparison => "native_baseline_comparison",
            Self::EqualityConditions => "equality_conditions",
        }
    }
}

/// Parity comparison of device outputs against the reference evaluator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WholeAppParityRecord {
    /// Number of named output values compared on both sides.
    pub compared_values: usize,
    /// Whether every compared output matched byte for byte.
    pub is_exact_match: bool,
    /// Largest absolute difference observed between corresponding lanes.
    pub max_ulp_distance: u32,
    /// Digest of the reference evaluator outputs.
    pub reference_digest: String,
    /// Digest of the device outputs.
    pub candidate_digest: String,
    /// Comparison outcome derived from the compared bytes.
    pub parity_status: String,
}

impl WholeAppParityRecord {
    /// Compare named reference outputs against named device outputs.
    ///
    /// # Errors
    ///
    /// Returns a diagnostic when either side is empty or the two sides do not
    /// carry the same output names, because a comparison over a subset states a
    /// parity result the run did not establish.
    pub fn compare(
        reference: &BTreeMap<String, Vec<u8>>,
        candidate: &BTreeMap<String, Vec<u8>>,
    ) -> Result<Self, String> {
        if reference.is_empty() {
            return Err("reference evaluator produced no named outputs to compare".to_string());
        }
        let reference_names: BTreeSet<&String> = reference.keys().collect();
        let candidate_names: BTreeSet<&String> = candidate.keys().collect();
        if reference_names != candidate_names {
            return Err(format!(
                "output name sets differ: reference {reference_names:?}, device {candidate_names:?}"
            ));
        }

        let mut max_difference = 0u32;
        let mut is_exact_match = true;
        for (name, reference_bytes) in reference {
            let candidate_bytes = &candidate[name];
            if reference_bytes != candidate_bytes {
                is_exact_match = false;
            }
            if reference_bytes.len() != candidate_bytes.len() {
                return Err(format!(
                    "output `{name}` byte length differs: reference {}, device {}",
                    reference_bytes.len(),
                    candidate_bytes.len()
                ));
            }
            for (reference_lane, candidate_lane) in reference_bytes
                .chunks_exact(4)
                .zip(candidate_bytes.chunks_exact(4))
            {
                let reference_value = u32::from_le_bytes([
                    reference_lane[0],
                    reference_lane[1],
                    reference_lane[2],
                    reference_lane[3],
                ]);
                let candidate_value = u32::from_le_bytes([
                    candidate_lane[0],
                    candidate_lane[1],
                    candidate_lane[2],
                    candidate_lane[3],
                ]);
                max_difference = max_difference.max(reference_value.abs_diff(candidate_value));
            }
        }

        let parity_status = if is_exact_match {
            "matched_exact_bytes".to_string()
        } else {
            format!("mismatched_max_lane_difference_{max_difference}")
        };

        Ok(Self {
            compared_values: reference.len(),
            is_exact_match,
            max_ulp_distance: max_difference,
            reference_digest: digest_named_bytes(reference),
            candidate_digest: digest_named_bytes(candidate),
            parity_status,
        })
    }
}

/// Content hash of a named byte map, hashing each name then its bytes in key order.
fn hash_named_bytes(values: &BTreeMap<String, Vec<u8>>) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new();
    for (name, bytes) in values {
        hasher.update(name.as_bytes());
        hasher.update(bytes);
    }
    hasher.finalize()
}

/// Hex digest of a named byte map.
fn digest_named_bytes(values: &BTreeMap<String, Vec<u8>>) -> String {
    hash_named_bytes(values).to_hex().to_string()
}

/// Nearest-rank percentile of an ascending sample slice.
///
/// # Errors
///
/// Returns a diagnostic when the slice is empty, because a percentile of no
/// samples is not a measurement.
fn percentile(ascending_samples: &[u64], percent: u64) -> Result<u64, String> {
    if ascending_samples.is_empty() {
        return Err(format!(
            "no samples were recorded, so no p{percent} latency exists"
        ));
    }
    let last = ascending_samples.len() - 1;
    let rank = (percent as usize * last).div_ceil(100);
    Ok(ascending_samples[rank.min(last)])
}

/// Throughput figures with explicit units.
///
/// A unit that nothing in the run counts is `None`. A floating-point operation
/// count is not recorded by the compiled artifact, so `gflops` stays `None`
/// until an artifact reports one.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct WholeAppThroughputRecord {
    /// Compute throughput in GFLOP/s, when the artifact reports an operation count.
    pub gflops: Option<f64>,
    /// Bytes the artifact's resource envelope moves per submission, in GB/s.
    pub gb_per_sec: Option<f64>,
    /// Output elements produced per second.
    pub items_per_sec: Option<f64>,
}

/// State metrics for cold-start and warm steady-state execution.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WholeAppStateMetrics {
    /// Wall-clock latency of the submissions this state covers, in nanoseconds.
    pub latency_ns: u64,
    /// Device bytes held during this state, when a counter reports them.
    ///
    /// No per-state device allocator counter is read on this path, so this is
    /// `None` rather than the whole-run peak repeated per state.
    pub memory_bytes: Option<u64>,
    /// Whether this state covers the first submission after materialization.
    pub is_cold_start: bool,
    /// Cold latency divided by warm latency, when both states are recorded.
    pub warmup_ratio: Option<f64>,
}

/// Production-route identities one whole-application run passed through.
///
/// Each field is read back from the session that executed, so the record states
/// the route it took instead of asserting that it took one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WholeAppProductionRouteRecord {
    /// Backend identifier that materialized and executed the artifact.
    pub backend_id: String,
    /// Device identifier the executing instance reported.
    pub device_id: String,
    /// Device generation the executing instance belonged to.
    pub device_generation: u64,
    /// Neutral artifact digest the session admitted.
    pub artifact_digest: String,
    /// Authenticated target payload digest the session materialized.
    pub target_payload_digest: String,
    /// Number of canonical values bound for each submission.
    pub bound_resource_count: usize,
    /// Number of submissions the device completed.
    pub completed_submissions: usize,
}

impl WholeAppProductionRouteRecord {
    /// Names of the route identities this record leaves unstated.
    #[must_use]
    pub fn missing_identities(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if self.backend_id.is_empty() {
            missing.push("production_route.backend_id");
        }
        if self.device_id.is_empty() {
            missing.push("production_route.device_id");
        }
        if self.artifact_digest.is_empty() {
            missing.push("production_route.artifact_digest");
        }
        if self.target_payload_digest.is_empty() {
            missing.push("production_route.target_payload_digest");
        }
        if self.bound_resource_count == 0 {
            missing.push("production_route.bound_resource_count");
        }
        if self.completed_submissions == 0 {
            missing.push("production_route.completed_submissions");
        }
        missing
    }
}

/// Pinned native baseline comparison on identical input identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeAppNativeComparisonRecord {
    /// Pinned native baseline identifier.
    pub baseline_id: String,
    /// Human-readable baseline title.
    pub baseline_name: String,
    /// Released version the compared native kernel was pinned to.
    pub baseline_pinned_version: String,
    /// Digest of the identical input bytes both systems evaluated.
    pub identical_input_digest: String,
    /// Measured 50th percentile latency of the native baseline in nanoseconds.
    pub native_p50_latency_ns: u64,
    /// Native p50 divided by the Vyre p50 measured in the same record.
    pub speedup_ratio: f64,
    /// Comparison verdict derived from the ratio.
    pub verdict: String,
    /// The 14 comparison conditions both measurements held.
    pub equality_conditions: NativeComparisonConditions,
}

impl WholeAppNativeComparisonRecord {
    /// Build a comparison from a measured version-pinned baseline.
    ///
    /// The native latency is read from the baseline's own recorded measurement.
    /// No path derives it from the Vyre latency, so a baseline that was never
    /// measured produces an error rather than a ratio.
    ///
    /// # Errors
    ///
    /// Returns a diagnostic when the baseline carries no measurement, when its
    /// measured p50 is zero, when the Vyre p50 is zero, or when the 14
    /// comparison conditions do not agree on both sides.
    pub fn from_measured_baseline(
        workload_id: &str,
        baseline: &VersionPinnedNativeBaseline,
        declared_conditions: &NativeComparisonConditions,
        identical_input_digest: &str,
        vyre_p50_latency_ns: u64,
    ) -> Result<Self, String> {
        let Some(measurement) = baseline.measurement.as_ref() else {
            return Err(format!(
                "native baseline `{}` carries no recorded measurement",
                baseline.baseline_id
            ));
        };
        if measurement.p50_latency_ns == 0 {
            return Err(format!(
                "native baseline `{}` recorded a zero p50 latency",
                baseline.baseline_id
            ));
        }
        if vyre_p50_latency_ns == 0 {
            return Err(format!(
                "workload `{workload_id}` recorded a zero p50 latency, so no ratio is defined"
            ));
        }
        if identical_input_digest.is_empty() {
            return Err(format!(
                "workload `{workload_id}` recorded no input digest to compare on"
            ));
        }
        validate_equality_conditions(
            workload_id,
            declared_conditions,
            &baseline.equality_conditions,
        )
        .map_err(|err| err.to_string())?;

        let speedup_ratio = (measurement.p50_latency_ns as f64) / (vyre_p50_latency_ns as f64);
        let verdict = if speedup_ratio >= 1.0 + COMPARISON_EQUIVALENCE_BAND {
            "win"
        } else if speedup_ratio <= 1.0 - COMPARISON_EQUIVALENCE_BAND {
            "loss"
        } else {
            "statistically_indistinguishable"
        };

        Ok(Self {
            baseline_id: baseline.baseline_id.clone(),
            baseline_name: baseline.name.clone(),
            baseline_pinned_version: baseline.pinned_version.clone(),
            identical_input_digest: identical_input_digest.to_string(),
            native_p50_latency_ns: measurement.p50_latency_ns,
            speedup_ratio,
            verdict: verdict.to_string(),
            equality_conditions: baseline.equality_conditions.clone(),
        })
    }
}

/// Recorded reason a whole-application run published no native comparison.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WholeAppNativeBaselineUnmeasured {
    /// Identifier of the baseline the workload pins.
    pub baseline_id: String,
    /// Human-readable baseline title.
    pub baseline_name: String,
    /// Why no comparison was computed against it.
    pub reason: String,
}

/// Error returned when required fields are missing or invalid in a whole-application record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissingRequiredWholeAppFieldsError {
    /// Workload identifier.
    pub workload_id: String,
    /// List of missing or invalid field names.
    pub missing_fields: Vec<String>,
}

impl std::fmt::Display for MissingRequiredWholeAppFieldsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Whole-application record `{}` is missing or has invalid required fields: {:?}",
            self.workload_id, self.missing_fields
        )
    }
}

impl std::error::Error for MissingRequiredWholeAppFieldsError {}

/// Typed measurement record for one whole-application run on one device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeApplicationRecord {
    /// Schema version string.
    pub schema_version: String,
    /// Workload identifier.
    pub workload_id: String,
    /// Workload title.
    pub workload_name: String,
    /// Workload domain classification.
    pub domain: ApplicationDomain,
    /// Number of distinct connected nodes in the compiled application graph.
    pub graph_node_count: usize,
    /// Number of internal value dataflow edges connecting stages.
    pub graph_edge_count: usize,
    /// Whether the request compiled through a validated `CompileRequest`.
    pub compile_request_validated: bool,
    /// Production-route identities the run passed through.
    pub production_route: WholeAppProductionRouteRecord,
    /// Parity of device outputs against the reference evaluator.
    pub parity_result: WholeAppParityRecord,
    /// Compilation time in nanoseconds.
    pub compile_time_ns: u64,
    /// Module admission and materialization time in nanoseconds.
    pub load_time_ns: u64,
    /// Number of measured warm submissions the percentiles rest on.
    pub measured_samples: usize,
    /// 50th percentile submission wall-clock latency in nanoseconds.
    pub p50_latency_ns: u64,
    /// 99th percentile submission wall-clock latency in nanoseconds.
    pub p99_latency_ns: u64,
    /// 50th percentile backend-reported device duration, when the backend reports one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_p50_latency_ns: Option<u64>,
    /// Throughput derived from the measured p50 and the artifact's own counts.
    pub throughput: WholeAppThroughputRecord,
    /// Peak device bytes the artifact's own storage holds at once.
    pub peak_bytes: Option<u64>,
    /// Bytes the artifact's retained-lifetime resources hold across submissions.
    pub resident_bytes: Option<u64>,
    /// Cold-start state metrics.
    pub cold_state: WholeAppStateMetrics,
    /// Warm steady-state metrics.
    pub warm_state: WholeAppStateMetrics,
    /// Identity of the schedule the compiler selected.
    pub selected_schedule_id: String,
    /// Digest of the input bytes both the device and the reference evaluated.
    pub input_digest: String,
    /// Comparison against the pinned native baseline, when one is measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_baseline_comparison: Option<WholeAppNativeComparisonRecord>,
    /// Reason no comparison was computed, when the pinned baseline is unmeasured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_baseline_unmeasured: Option<WholeAppNativeBaselineUnmeasured>,
    /// Host and device the measurement ran on.
    pub host_environment: String,
    /// UTC timestamp the record was written at.
    pub recorded_at_utc: String,
}

impl WholeApplicationRecord {
    /// Whether every recorded latency came from a completed device submission.
    #[must_use]
    pub fn executed_via_production_route(&self) -> bool {
        self.production_route.missing_identities().is_empty()
            && self.production_route.completed_submissions >= self.measured_samples
    }

    /// Required fields this record states nothing for.
    ///
    /// A record is valid without these. It is not complete whole-application
    /// readiness evidence without them, and a consumer reads the gap here rather
    /// than inferring completeness from a status string.
    #[must_use]
    pub fn readiness_gaps(&self) -> Vec<RequiredWholeApplicationField> {
        let mut gaps = Vec::new();
        if self.native_baseline_comparison.is_none() {
            gaps.push(RequiredWholeApplicationField::NativeBaselineComparison);
            gaps.push(RequiredWholeApplicationField::EqualityConditions);
        }
        gaps
    }

    /// Validate that every field the record does state is present and consistent.
    ///
    /// # Errors
    ///
    /// Returns the list of fields that are absent, out of range, or mutually
    /// inconsistent.
    pub fn validate_required_fields(&self) -> Result<(), MissingRequiredWholeAppFieldsError> {
        let mut missing = Vec::new();

        if self.schema_version != WHOLE_APPLICATION_RECORD_SCHEMA_V2 {
            missing.push(format!(
                "schema_version (expected `{}`, got `{}`)",
                WHOLE_APPLICATION_RECORD_SCHEMA_V2, self.schema_version
            ));
        }

        if self.graph_node_count < 2 {
            missing.push(format!(
                "multi_node_graph_topology: node_count must be >= 2, got {}",
                self.graph_node_count
            ));
        }
        if self.graph_edge_count < 1 {
            missing.push(format!(
                "multi_node_graph_topology: edge_count must be >= 1, got {}",
                self.graph_edge_count
            ));
        }
        if !self.compile_request_validated {
            missing.push("compile_request_validated".to_string());
        }
        for identity in self.production_route.missing_identities() {
            missing.push(identity.to_string());
        }
        if self.production_route.completed_submissions < self.measured_samples {
            missing.push(format!(
                "production_route.completed_submissions ({}) is below measured_samples ({})",
                self.production_route.completed_submissions, self.measured_samples
            ));
        }
        if self.parity_result.compared_values == 0
            || !self.parity_result.is_exact_match
            || self.parity_result.reference_digest.is_empty()
            || self.parity_result.candidate_digest.is_empty()
        {
            missing.push("parity_against_reference".to_string());
        }
        if self.compile_time_ns == 0 {
            missing.push("compile_time_ns".to_string());
        }
        if self.load_time_ns == 0 {
            missing.push("load_time_ns".to_string());
        }
        if self.measured_samples < MIN_MEASURED_SAMPLES {
            missing.push(format!(
                "measured_samples ({}) is below the minimum of {MIN_MEASURED_SAMPLES}",
                self.measured_samples
            ));
        }
        if self.p50_latency_ns == 0 {
            missing.push("p50_latency_ns".to_string());
        }
        if self.p99_latency_ns == 0 || self.p99_latency_ns < self.p50_latency_ns {
            missing.push("p99_latency_ns".to_string());
        }
        if self.throughput.gflops.is_none()
            && self.throughput.gb_per_sec.is_none()
            && self.throughput.items_per_sec.is_none()
        {
            missing.push("throughput".to_string());
        }
        if self.peak_bytes.is_none_or(|bytes| bytes == 0) {
            missing.push("peak_bytes".to_string());
        }
        if self.resident_bytes.is_none() {
            missing.push("resident_bytes".to_string());
        }
        if self.cold_state.latency_ns == 0 || self.warm_state.latency_ns == 0 {
            missing.push("cold_and_warm_state".to_string());
        }
        if self.cold_state.is_cold_start == self.warm_state.is_cold_start {
            missing
                .push("cold_and_warm_state: exactly one state must be the cold start".to_string());
        }
        if self.selected_schedule_id.is_empty() {
            missing.push("selected_schedule_identity".to_string());
        }
        if self.input_digest.is_empty() {
            missing.push("input_digest".to_string());
        }
        match (
            self.native_baseline_comparison.as_ref(),
            self.native_baseline_unmeasured.as_ref(),
        ) {
            (Some(_), Some(_)) => missing.push(
                "native_baseline_comparison and native_baseline_unmeasured are both present"
                    .to_string(),
            ),
            (None, None) => missing
                .push("native_baseline_comparison is absent with no recorded reason".to_string()),
            (Some(comparison), None) => {
                if comparison.baseline_id.is_empty()
                    || comparison.baseline_pinned_version.is_empty()
                    || comparison.identical_input_digest.is_empty()
                    || comparison.native_p50_latency_ns == 0
                    || comparison.speedup_ratio <= 0.0
                {
                    missing.push("native_baseline_comparison".to_string());
                }
                if comparison.identical_input_digest != self.input_digest {
                    missing.push(
                        "native_baseline_comparison.identical_input_digest does not match input_digest"
                            .to_string(),
                    );
                }
                let unset = comparison.equality_conditions.unset_dimensions();
                if !unset.is_empty() {
                    missing.push(format!("equality_conditions (unset dimensions: {unset:?})"));
                }
            }
            (None, Some(unmeasured)) => {
                if unmeasured.baseline_id.is_empty() || unmeasured.reason.trim().is_empty() {
                    missing.push("native_baseline_unmeasured".to_string());
                }
            }
        }
        if self.host_environment.is_empty() {
            missing.push("host_environment".to_string());
        }
        if self.recorded_at_utc.is_empty() {
            missing.push("recorded_at_utc".to_string());
        }

        if missing.is_empty() {
            Ok(())
        } else {
            Err(MissingRequiredWholeAppFieldsError {
                workload_id: self.workload_id.clone(),
                missing_fields: missing,
            })
        }
    }

    /// Fail closed if this record is stale, partial, or invalid.
    ///
    /// # Errors
    ///
    /// Returns a refusal naming the stale schema version, or the fields that
    /// failed validation.
    pub fn fail_closed_if_stale_or_partial(&self) -> Result<(), WholeApplicationRefusal> {
        if self.schema_version != WHOLE_APPLICATION_RECORD_SCHEMA_V2 {
            return Err(WholeApplicationRefusal::StaleSchemaVersion {
                workload_id: self.workload_id.clone(),
                version: self.schema_version.clone(),
                expected: WHOLE_APPLICATION_RECORD_SCHEMA_V2.to_string(),
            });
        }
        self.validate_required_fields().map_err(|err| {
            WholeApplicationRefusal::MissingRequiredField {
                workload_id: self.workload_id.clone(),
                field: err.missing_fields.join(", "),
                reason: "Record failed validation".to_string(),
            }
        })
    }
}

/// One acquired dispatch device a whole-application measurement runs on.
///
/// Probing is separate from measuring so a device-less host is named once,
/// before any record exists, and every record that does exist names the device
/// it ran on.
pub struct WholeApplicationDevice {
    backend_id: String,
    registration: &'static BackendRegistration,
    materializer: Arc<dyn ArtifactMaterializer>,
    target_compiler: Box<dyn TargetCompiler>,
    device_facts: DeviceFacts,
}

impl WholeApplicationDevice {
    /// Acquire a dispatch device, its target compiler, and its materializer.
    ///
    /// `backend_id` names one backend. `None` takes the highest-precedence
    /// dispatch backend that acquires, which excludes every conformance oracle.
    ///
    /// # Errors
    ///
    /// Returns [`WholeApplicationRefusal::NoDispatchDevice`] when no backend
    /// acquires on this host, or when the acquired backend registers no target
    /// compiler or materializer, and
    /// [`WholeApplicationRefusal::ReferenceOracleBackend`] when the named
    /// backend executes on the host instead of a device.
    pub fn probe(backend_id: Option<&str>) -> Result<Self, WholeApplicationRefusal> {
        let backend = match backend_id {
            Some(id) => acquire(id),
            None => acquire_preferred_dispatch_backend(),
        }
        .map_err(|error| WholeApplicationRefusal::NoDispatchDevice {
            reason: error.to_string(),
        })?;
        let id = backend.id().to_string();
        let registration = backend_registration(&id).map_err(|error| {
            WholeApplicationRefusal::NoDispatchDevice {
                reason: error.to_string(),
            }
        })?;
        if registration.reference_oracle {
            return Err(WholeApplicationRefusal::ReferenceOracleBackend { backend_id: id });
        }
        let materializer = registration.materializer().map_err(|error| {
            WholeApplicationRefusal::NoDispatchDevice {
                reason: error.to_string(),
            }
        })?;
        let target_compiler = registration.target_compiler().map_err(|error| {
            WholeApplicationRefusal::NoDispatchDevice {
                reason: error.to_string(),
            }
        })?;
        Ok(Self {
            device_facts: backend.device_profile().compile_facts(),
            backend_id: id,
            registration,
            materializer: Arc::from(materializer),
            target_compiler,
        })
    }

    /// Identifier of the acquired backend.
    #[must_use]
    pub fn backend_id(&self) -> &str {
        &self.backend_id
    }
}
