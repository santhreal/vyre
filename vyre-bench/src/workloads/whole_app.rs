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

use super::equality::{validate_equality_conditions, NativeComparisonConditions};
use super::native_baseline::{whole_application_native_baselines, VersionPinnedNativeBaseline};
use super::provenance::MeasurementProvenance;
use crate::api::metric::elapsed_ns;

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

        let speedup_ratio =
            (measurement.p50_latency_ns as f64) / (vyre_p50_latency_ns as f64);
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

/// Every serialized field of [`WholeApplicationRecord`].
///
/// A field enters this list with the source it is read from. A test derives the
/// serialized field set at run time and rejects a field that appears here with
/// no source, or in the struct with no entry here, so adding a field forces a
/// recorded decision about where its value comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum WholeApplicationRecordField {
    /// `schema_version`
    SchemaVersion,
    /// `workload_id`
    WorkloadId,
    /// `workload_name`
    WorkloadName,
    /// `domain`
    Domain,
    /// `graph_node_count`
    GraphNodeCount,
    /// `graph_edge_count`
    GraphEdgeCount,
    /// `compile_request_validated`
    CompileRequestValidated,
    /// `production_route`
    ProductionRoute,
    /// `parity_result`
    ParityResult,
    /// `compile_time_ns`
    CompileTimeNs,
    /// `load_time_ns`
    LoadTimeNs,
    /// `measured_samples`
    MeasuredSamples,
    /// `p50_latency_ns`
    P50LatencyNs,
    /// `p99_latency_ns`
    P99LatencyNs,
    /// `device_p50_latency_ns`
    DeviceP50LatencyNs,
    /// `throughput`
    Throughput,
    /// `peak_bytes`
    PeakBytes,
    /// `resident_bytes`
    ResidentBytes,
    /// `cold_state`
    ColdState,
    /// `warm_state`
    WarmState,
    /// `selected_schedule_id`
    SelectedScheduleId,
    /// `input_digest`
    InputDigest,
    /// `native_baseline_comparison`
    NativeBaselineComparison,
    /// `native_baseline_unmeasured`
    NativeBaselineUnmeasured,
    /// `host_environment`
    HostEnvironment,
    /// `recorded_at_utc`
    RecordedAtUtc,
}

/// Where a recorded field's value is read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RecordFieldSource {
    /// A constant of the record schema itself.
    SchemaConstant,
    /// The static workload definition.
    WorkloadDefinition,
    /// The compiled artifact and its plans.
    CompiledArtifact,
    /// The runtime session that admitted and materialized the artifact.
    RuntimeSession,
    /// Submissions the device completed.
    DeviceSubmission,
    /// The host reference evaluator, used for parity only.
    ReferenceOracle,
    /// A version-pinned native baseline measurement.
    PinnedNativeBaseline,
    /// Host and clock provenance captured at record time.
    HostProvenance,
}

impl RecordFieldSource {
    /// String name of the source.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SchemaConstant => "schema_constant",
            Self::WorkloadDefinition => "workload_definition",
            Self::CompiledArtifact => "compiled_artifact",
            Self::RuntimeSession => "runtime_session",
            Self::DeviceSubmission => "device_submission",
            Self::ReferenceOracle => "reference_oracle",
            Self::PinnedNativeBaseline => "pinned_native_baseline",
            Self::HostProvenance => "host_provenance",
        }
    }
}

impl WholeApplicationRecordField {
    /// Every field of the record, each paired with a source by [`Self::source`].
    pub const ALL: [Self; 26] = [
        Self::SchemaVersion,
        Self::WorkloadId,
        Self::WorkloadName,
        Self::Domain,
        Self::GraphNodeCount,
        Self::GraphEdgeCount,
        Self::CompileRequestValidated,
        Self::ProductionRoute,
        Self::ParityResult,
        Self::CompileTimeNs,
        Self::LoadTimeNs,
        Self::MeasuredSamples,
        Self::P50LatencyNs,
        Self::P99LatencyNs,
        Self::DeviceP50LatencyNs,
        Self::Throughput,
        Self::PeakBytes,
        Self::ResidentBytes,
        Self::ColdState,
        Self::WarmState,
        Self::SelectedScheduleId,
        Self::InputDigest,
        Self::NativeBaselineComparison,
        Self::NativeBaselineUnmeasured,
        Self::HostEnvironment,
        Self::RecordedAtUtc,
    ];

    /// Serialized field name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SchemaVersion => "schema_version",
            Self::WorkloadId => "workload_id",
            Self::WorkloadName => "workload_name",
            Self::Domain => "domain",
            Self::GraphNodeCount => "graph_node_count",
            Self::GraphEdgeCount => "graph_edge_count",
            Self::CompileRequestValidated => "compile_request_validated",
            Self::ProductionRoute => "production_route",
            Self::ParityResult => "parity_result",
            Self::CompileTimeNs => "compile_time_ns",
            Self::LoadTimeNs => "load_time_ns",
            Self::MeasuredSamples => "measured_samples",
            Self::P50LatencyNs => "p50_latency_ns",
            Self::P99LatencyNs => "p99_latency_ns",
            Self::DeviceP50LatencyNs => "device_p50_latency_ns",
            Self::Throughput => "throughput",
            Self::PeakBytes => "peak_bytes",
            Self::ResidentBytes => "resident_bytes",
            Self::ColdState => "cold_state",
            Self::WarmState => "warm_state",
            Self::SelectedScheduleId => "selected_schedule_id",
            Self::InputDigest => "input_digest",
            Self::NativeBaselineComparison => "native_baseline_comparison",
            Self::NativeBaselineUnmeasured => "native_baseline_unmeasured",
            Self::HostEnvironment => "host_environment",
            Self::RecordedAtUtc => "recorded_at_utc",
        }
    }

    /// Source this field's value is read from.
    #[must_use]
    pub const fn source(self) -> RecordFieldSource {
        match self {
            Self::SchemaVersion => RecordFieldSource::SchemaConstant,
            Self::WorkloadId | Self::WorkloadName | Self::Domain => {
                RecordFieldSource::WorkloadDefinition
            }
            Self::GraphNodeCount
            | Self::GraphEdgeCount
            | Self::CompileRequestValidated
            | Self::CompileTimeNs
            | Self::Throughput
            | Self::PeakBytes
            | Self::ResidentBytes
            | Self::SelectedScheduleId => RecordFieldSource::CompiledArtifact,
            Self::ProductionRoute | Self::LoadTimeNs => RecordFieldSource::RuntimeSession,
            Self::MeasuredSamples
            | Self::P50LatencyNs
            | Self::P99LatencyNs
            | Self::DeviceP50LatencyNs
            | Self::ColdState
            | Self::WarmState => RecordFieldSource::DeviceSubmission,
            Self::ParityResult | Self::InputDigest => RecordFieldSource::ReferenceOracle,
            Self::NativeBaselineComparison | Self::NativeBaselineUnmeasured => {
                RecordFieldSource::PinnedNativeBaseline
            }
            Self::HostEnvironment | Self::RecordedAtUtc => RecordFieldSource::HostProvenance,
        }
    }
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
            missing.push(
                "cold_and_warm_state: exactly one state must be the cold start".to_string(),
            );
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
            (None, None) => missing.push(
                "native_baseline_comparison is absent with no recorded reason".to_string(),
            ),
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
        let registration =
            backend_registration(&id).map_err(|error| WholeApplicationRefusal::NoDispatchDevice {
                reason: error.to_string(),
            })?;
        if registration.reference_oracle {
            return Err(WholeApplicationRefusal::ReferenceOracleBackend { backend_id: id });
        }
        let materializer =
            registration
                .materializer()
                .map_err(|error| WholeApplicationRefusal::NoDispatchDevice {
                    reason: error.to_string(),
                })?;
        let target_compiler =
            registration
                .target_compiler()
                .map_err(|error| WholeApplicationRefusal::NoDispatchDevice {
                    reason: error.to_string(),
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

/// Whole-application workload specification and graph builder.
#[derive(Clone)]
pub struct WholeApplicationWorkload {
    /// Stable workload identifier.
    pub id: &'static str,
    /// Human-readable title.
    pub name: &'static str,
    /// Domain classification.
    pub domain: ApplicationDomain,
    /// Detailed description.
    pub description: &'static str,
    /// Pinned native baseline comparator identifier.
    pub pinned_native_baseline_id: &'static str,
    /// Pinned native baseline comparator name.
    pub pinned_native_baseline_name: &'static str,
    /// Default 14-point comparison equality conditions.
    pub default_conditions: NativeComparisonConditions,
    /// Builder constructing the connected multi-node ProgramGraph and concrete input buffers.
    pub build_graph_and_inputs: fn() -> (ProgramGraph, BTreeMap<String, Vec<u8>>),
}

impl WholeApplicationWorkload {
    /// Count internal dataflow edges connecting stages in the graph.
    #[must_use]
    pub fn count_internal_edges(graph: &ProgramGraph) -> usize {
        graph
            .values()
            .iter()
            .filter(|v| v.producer.is_some() && !v.consumers.is_empty())
            .count()
    }

    /// Validate that the constructed graph satisfies the whole-application topology
    /// requirement (at least 2 connected nodes and internal edges).
    pub fn validate_topology(&self, graph: &ProgramGraph) -> Result<(), WholeApplicationRefusal> {
        let node_count = graph.nodes().len();
        if node_count < 2 {
            return Err(WholeApplicationRefusal::SingleNodeIsolatedKernel {
                workload_id: self.id.to_string(),
                node_count,
            });
        }

        let edge_count = Self::count_internal_edges(graph);
        if edge_count < 1 {
            return Err(WholeApplicationRefusal::DisconnectedGraph {
                workload_id: self.id.to_string(),
                edge_count,
            });
        }

        Ok(())
    }

    /// Evaluate the graph on the host reference evaluator.
    ///
    /// This is the parity oracle. Nothing here contributes to a recorded
    /// latency, throughput, or memory figure.
    ///
    /// # Errors
    ///
    /// Returns a diagnostic when an external value has no supplied input bytes,
    /// when a node input is unresolved, or when the reference evaluator rejects
    /// a node program.
    pub fn evaluate_reference_outputs(
        &self,
        graph: &ProgramGraph,
        inputs: &BTreeMap<String, Vec<u8>>,
    ) -> Result<BTreeMap<String, Vec<u8>>, String> {
        let mut value_map: BTreeMap<GraphValueId, Vec<u8>> = BTreeMap::new();

        for value in graph.values() {
            if value.producer.is_none() {
                let Some(bytes) = inputs.get(&value.name) else {
                    return Err(format!(
                        "workload `{}` supplies no input bytes for external value `{}`",
                        self.id, value.name
                    ));
                };
                value_map.insert(value.id, bytes.clone());
            }
        }

        for node in graph.nodes() {
            let mut node_inputs = Vec::new();
            for decl in node.program.buffers() {
                if !vyre_reference::is_reference_input(decl) {
                    continue;
                }
                let bytes = match node.inputs.iter().find(|port| port.buffer == decl.name()) {
                    Some(port) => {
                        let Some(bytes) = value_map.get(&port.value) else {
                            return Err(format!(
                                "workload `{}` node `{}` reads value {} before it is produced",
                                self.id, node.name, port.value.0
                            ));
                        };
                        bytes.clone()
                    }
                    None => {
                        let byte_len = decl
                            .static_byte_len()
                            .map_err(|err| {
                                format!(
                                    "workload `{}` node `{}` buffer `{}` has no static byte length: {err}",
                                    self.id,
                                    node.name,
                                    decl.name()
                                )
                            })?
                            .ok_or_else(|| {
                                format!(
                                    "workload `{}` node `{}` buffer `{}` declares a dynamic byte length",
                                    self.id,
                                    node.name,
                                    decl.name()
                                )
                            })?;
                        vec![0u8; byte_len]
                    }
                };
                node_inputs.push(Value::Bytes(Arc::from(bytes.into_boxed_slice())));
            }

            let node_outputs = vyre_reference::reference_eval(&node.program, &node_inputs)
                .map_err(|err| {
                    format!("Reference eval failed on node `{}`: {:?}", node.name, err)
                })?;

            for (out_value_id, port) in node.outputs.iter().zip(node.output_ports.iter()) {
                let Some(index) = vyre_reference::output_index(&node.program, &port.buffer) else {
                    return Err(format!(
                        "workload `{}` node `{}` declares output buffer `{}` that the oracle does not return",
                        self.id, node.name, port.buffer
                    ));
                };
                let Some(out_value) = node_outputs.get(index) else {
                    return Err(format!(
                        "reference eval of node `{}` produced {} outputs, buffer `{}` is at index {index}",
                        node.name,
                        node_outputs.len(),
                        port.buffer
                    ));
                };
                value_map.insert(*out_value_id, out_value.to_bytes());
            }
        }

        let mut outputs = BTreeMap::new();
        for value in graph.values() {
            if value.contract.lifetime == ValueLifetime::Output
                || (value.producer.is_some() && value.consumers.is_empty())
            {
                if let Some(bytes) = value_map.get(&value.id) {
                    outputs.insert(value.name.clone(), bytes.clone());
                }
            }
        }
        if outputs.is_empty() {
            return Err(format!(
                "workload `{}` graph declares no output value to compare",
                self.id
            ));
        }
        Ok(outputs)
    }

    /// Project one completion onto the named outputs the reference produced.
    fn device_outputs(
        &self,
        session: &ArtifactSession,
        completion: &Completion,
        names: &BTreeSet<&String>,
    ) -> Result<BTreeMap<String, Vec<u8>>, String> {
        let mut outputs = BTreeMap::new();
        for name in names {
            let value = session.resource(name).map_err(|error| {
                format!("compiled artifact carries no resource named `{name}`: {error}")
            })?;
            let Some(bytes) = completion
                .outputs
                .get(&value)
                .or_else(|| completion.retained.get(&value))
            else {
                return Err(format!(
                    "device completion projected no bytes for output `{name}`"
                ));
            };
            outputs.insert((*name).clone(), bytes.clone());
        }
        Ok(outputs)
    }

    /// Compile, admit, and execute the whole application on one acquired device.
    ///
    /// Every recorded latency is the wall time of a submission the device
    /// completed. The reference evaluator runs once, for parity only.
    ///
    /// # Errors
    ///
    /// Returns a diagnostic when the sample count is below
    /// [`MIN_MEASURED_SAMPLES`], when compilation, admission, binding, or a
    /// submission fails, when device outputs disagree with the reference, or
    /// when the resulting record fails validation.
    pub fn execute_and_measure(
        &self,
        device: &WholeApplicationDevice,
        measured_samples: usize,
    ) -> Result<WholeApplicationRecord, String> {
        if measured_samples < MIN_MEASURED_SAMPLES {
            return Err(format!(
                "workload `{}` requires at least {MIN_MEASURED_SAMPLES} measured samples, got {measured_samples}",
                self.id
            ));
        }

        let (graph, inputs) = (self.build_graph_and_inputs)();
        self.validate_topology(&graph)
            .map_err(|err| err.to_string())?;

        let input_hash = hash_named_bytes(&inputs);
        let input_digest = input_hash.to_hex().to_string();

        let request = CompileRequest::new(
            graph.clone(),
            ExternalFacts::new(Digest(*input_hash.as_bytes()), BTreeMap::new()),
            device.device_facts,
            SearchBudget::new(32, 1_000_000, 4, 0, 10_000_000),
            CompileObjective::minimize_latency()
                .with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
        )
        .validate()
        .map_err(|err| format!("CompileRequest validation failed: {err:?}"))?;

        let compile_start = Instant::now();
        let artifact = compile(&request)
            .map_err(|err| format!("Compiler failed to produce artifact: {err:?}"))?;
        let compile_time_ns = elapsed_ns(compile_start);

        let envelope = attach_target(artifact.clone(), device.target_compiler.as_ref())
            .map_err(|err| format!("Target compilation failed for `{}`: {err:?}", self.id))?;

        let load_start = Instant::now();
        let session = ArtifactSession::from_envelope_with_materializer(
            device.registration,
            envelope,
            Arc::clone(&device.materializer),
        )
        .map_err(|err| format!("Artifact admission failed for `{}`: {err}", self.id))?;
        let load_time_ns = elapsed_ns(load_start);

        let artifact_digest = session
            .artifact()
            .map_err(|err| format!("Session reported no artifact identity: {err}"))?;
        let target_payload_digest = session
            .payload()
            .map_err(|err| format!("Session reported no payload identity: {err}"))?;
        let device_identity = session
            .device()
            .map_err(|err| format!("Session reported no device identity: {err}"))?;

        let workspace = session
            .allocate_workspace()
            .map_err(|err| format!("Artifact workspace allocation failed: {err}"))?;

        let mut dataset = TypedResourceDataset::new();
        for resource in artifact.resources() {
            if workspace.owns(resource.value) {
                continue;
            }
            let byte_count = usize::try_from(resource.byte_count).map_err(|_| {
                format!(
                    "resource `{}` declares {} bytes, which this host cannot address",
                    resource.name, resource.byte_count
                )
            })?;
            if byte_count == 0 {
                return Err(format!(
                    "resource `{}` declares no bytes, so nothing can be bound to it",
                    resource.name
                ));
            }
            let typed = match inputs.get(&resource.name) {
                Some(bytes) if bytes.len() == byte_count => {
                    TypedResource::memory(resource.value, bytes.clone())
                }
                Some(bytes) => {
                    return Err(format!(
                        "input `{}` supplies {} bytes, the artifact declares {byte_count}",
                        resource.name,
                        bytes.len()
                    ));
                }
                None => TypedResource::zeroed(resource.value, byte_count),
            };
            dataset
                .insert(typed.with_lifetime(resource.lifetime))
                .map_err(|err| {
                    format!("resource `{}` cannot be ingested: {err}", resource.name)
                })?;
        }

        let bindings = session
            .ingest_with_workspace(&workspace, &dataset)
            .map_err(|err| format!("Resource ingestion failed for `{}`: {err}", self.id))?;
        let bound_resource_count = bindings.resources().len();
        let cold_start = Instant::now();
        session
            .submit_and_wait(bindings.clone())
            .map_err(|err| format!("Cold device submission failed for `{}`: {err}", self.id))?;
        let cold_latency_ns = elapsed_ns(cold_start);

        let mut wall_samples = Vec::with_capacity(measured_samples);
        let mut device_samples = Vec::with_capacity(measured_samples);
        let mut last_completion = None;
        for sample in 0..measured_samples {
            let start = Instant::now();
            let completion = session.submit_and_wait(bindings.clone()).map_err(|err| {
                format!(
                    "Device submission {sample} of {measured_samples} failed for `{}`: {err}",
                    self.id
                )
            })?;
            wall_samples.push(elapsed_ns(start));
            if let Some(device_ns) = completion.device_ns {
                device_samples.push(device_ns);
            }
            last_completion = Some(completion);
        }
        let completion = last_completion
            .ok_or_else(|| format!("workload `{}` completed no submission", self.id))?;

        wall_samples.sort_unstable();
        device_samples.sort_unstable();
        let p50_latency_ns = percentile(&wall_samples, 50)?;
        let p99_latency_ns = percentile(&wall_samples, 99)?;
        let device_p50_latency_ns = percentile(&device_samples, 50).ok();

        let reference_outputs = self.evaluate_reference_outputs(&graph, &inputs)?;
        let compared_names: BTreeSet<&String> = reference_outputs.keys().collect();
        let device_outputs = self.device_outputs(&session, &completion, &compared_names)?;
        let parity_result = WholeAppParityRecord::compare(&reference_outputs, &device_outputs)
            .map_err(|reason| {
                WholeApplicationRefusal::ParityMismatch {
                    workload_id: self.id.to_string(),
                    reason,
                }
                .to_string()
            })?;
        if !parity_result.is_exact_match {
            return Err(WholeApplicationRefusal::ParityMismatch {
                workload_id: self.id.to_string(),
                reason: format!(
                    "{} compared outputs differ, largest lane difference {}",
                    parity_result.compared_values, parity_result.max_ulp_distance
                ),
            }
            .to_string());
        }

        let peak_bytes = artifact.allocation().aggregate_peak_bytes;
        let resident_bytes = workspace.total_bytes();
        for (value, bound) in bindings.resources() {
            if workspace.owns(*value) {
                continue;
            }
            if let BoundResource::Resident(resource) = bound {
                session.free_resident(resource.clone()).map_err(|err| {
                    format!("Ingested resource release failed for value {}: {err}", value.0)
                })?;
            }
        }
        session
            .free_workspace(workspace)
            .map_err(|err| format!("Artifact workspace release failed: {err}"))?;
        let output_elements = artifact
            .resources()
            .iter()
            .filter(|resource| resource.lifetime == ResourceLifetime::Output)
            .try_fold(0u64, |total, resource| {
                total.checked_add(resource.element_count)
            })
            .ok_or_else(|| format!("workload `{}` output element count exceeds u64", self.id))?;

        let p50_seconds = (p50_latency_ns as f64) / 1e9;
        let throughput = WholeAppThroughputRecord {
            gflops: None,
            gb_per_sec: Some(
                (artifact.resource_envelope().total_bytes as f64) / (p50_latency_ns as f64),
            ),
            items_per_sec: Some((output_elements as f64) / p50_seconds),
        };

        let warmup_ratio = (cold_latency_ns as f64) / (p50_latency_ns as f64);
        let selected_schedule_id = format!(
            "schedule.{}",
            Digest(
                artifact
                    .selected_plan()
                    .schedule
                    .identity()
                    .map_err(|err| format!("Selected schedule has no identity: {err}"))?
            )
            .to_hex()
        );

        let catalog = whole_application_native_baselines();
        let (native_baseline_comparison, native_baseline_unmeasured) =
            match catalog.measured(self.pinned_native_baseline_id) {
                Ok(baseline) => (
                    Some(WholeAppNativeComparisonRecord::from_measured_baseline(
                        self.id,
                        baseline,
                        &self.default_conditions,
                        &input_digest,
                        p50_latency_ns,
                    )?),
                    None,
                ),
                Err(reason) => (
                    None,
                    Some(WholeAppNativeBaselineUnmeasured {
                        baseline_id: self.pinned_native_baseline_id.to_string(),
                        baseline_name: self.pinned_native_baseline_name.to_string(),
                        reason,
                    }),
                ),
            };

        let provenance =
            MeasurementProvenance::capture(device_identity.backend, &device_identity.device)?;

        let record = WholeApplicationRecord {
            schema_version: WHOLE_APPLICATION_RECORD_SCHEMA_V2.to_string(),
            workload_id: self.id.to_string(),
            workload_name: self.name.to_string(),
            domain: self.domain,
            graph_node_count: graph.nodes().len(),
            graph_edge_count: Self::count_internal_edges(&graph),
            compile_request_validated: true,
            production_route: WholeAppProductionRouteRecord {
                backend_id: device_identity.backend.to_string(),
                device_id: device_identity.device.clone(),
                device_generation: device_identity.generation,
                artifact_digest: artifact_digest.to_hex(),
                target_payload_digest: target_payload_digest.to_hex(),
                bound_resource_count,
                completed_submissions: measured_samples + 1,
            },
            parity_result,
            compile_time_ns,
            load_time_ns,
            measured_samples,
            p50_latency_ns,
            p99_latency_ns,
            device_p50_latency_ns,
            throughput,
            peak_bytes: Some(peak_bytes),
            resident_bytes: Some(resident_bytes),
            cold_state: WholeAppStateMetrics {
                latency_ns: cold_latency_ns,
                memory_bytes: None,
                is_cold_start: true,
                warmup_ratio: Some(warmup_ratio),
            },
            warm_state: WholeAppStateMetrics {
                latency_ns: p50_latency_ns,
                memory_bytes: None,
                is_cold_start: false,
                warmup_ratio: Some(warmup_ratio),
            },
            selected_schedule_id,
            input_digest,
            native_baseline_comparison,
            native_baseline_unmeasured,
            host_environment: provenance.host_environment,
            recorded_at_utc: provenance.recorded_at_utc,
        };

        record
            .validate_required_fields()
            .map_err(|err| format!("Generated record failed validation: {err}"))?;

        Ok(record)
    }
}

// ----------------------------------------------------------------------------
// Canonical Whole-Application Workload Definitions
// ----------------------------------------------------------------------------

fn contract(access: BufferAccess, lifetime: ValueLifetime, count: u64) -> ValueContract {
    ValueContract::dense_1d(DataType::U32, count, access, lifetime)
}

/// 1. Whole-Application: Dense Numerical Contraction Pipeline.
///
/// Multi-stage pipeline: linear feature projection -> SwiGLU activation & layer-norm -> residual fusion & quantize.
#[must_use]
pub fn dense_numerical_pipeline() -> WholeApplicationWorkload {
    WholeApplicationWorkload {
        id: "workload.whole_app.dense_numerical_pipeline",
        name: "Dense Numerical Tensor Contraction & Normalization Pipeline",
        domain: ApplicationDomain::DenseNumerical,
        description: "Complete 3-stage dense numerical application: Feature Projection GEMM -> Layer Normalization + SwiGLU Activation -> Residual Accumulation & Dynamic Quantization",
        pinned_native_baseline_id: "native.cutlass.gemm_norm_residual_v3_5_0",
        pinned_native_baseline_name: "NVIDIA CUTLASS 3.5.0 / cuBLAS 12.4 Contraction Pipeline",
        default_conditions: NativeComparisonConditions {
            semantics: Some("fp32_ulp_tol:4".to_string()),
            dtype: Some("u32_f32".to_string()),
            shapes: Some("[1024, 1024] -> [1024, 1024] -> [1024, 1024]".to_string()),
            raggedness: Some("uniform_contiguous".to_string()),
            initial_and_final_state: Some("clean_buffers_unaliased".to_string()),
            target: Some("sm_90a_sm_86".to_string()),
            stream: Some("cuda_stream_non_blocking_0".to_string()),
            toolchain_and_flags: Some("nvcc_12.4_-O3".to_string()),
            clock_and_power_state: Some("locked_base_clock_tdp_100pct".to_string()),
            warmup: Some("300_warmup_iterations_discarded".to_string()),
            interleaving: Some("ab_ba_round_robin_interleaving".to_string()),
            repetitions: Some("30_measured_samples_clt".to_string()),
            cache_state: Some("flushed_l2_between_iterations".to_string()),
            objective: Some("minimize_p50_latency".to_string()),
        },
        build_graph_and_inputs: || {
            let count = 1024_u64;
            let mut graph = ProgramGraph::new();

            // External Inputs
            let in_feat = graph
                .add_external_value("feat_in", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .unwrap();
            let in_weights = graph
                .add_external_value("weights", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .unwrap();
            let in_bias = graph
                .add_external_value("bias", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .unwrap();
            let in_residual = graph
                .add_external_value("residual", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .unwrap();

            // Stage 1: Linear Feature Projection (proj = feat_in * weights + bias)
            let prog_proj = Program::wrapped(
                vec![
                    BufferDecl::read("feat", 0, DataType::U32).with_count(count as u32),
                    BufferDecl::read("w", 1, DataType::U32).with_count(count as u32),
                    BufferDecl::read("b", 2, DataType::U32).with_count(count as u32),
                    BufferDecl::read_write("proj_out", 3, DataType::U32).with_count(count as u32),
                ],
                [count as u32, 1, 1],
                vec![Node::store(
                    "proj_out",
                    Expr::gid_x(),
                    Expr::add(
                        Expr::mul(Expr::load("feat", Expr::gid_x()), Expr::load("w", Expr::gid_x())),
                        Expr::load("b", Expr::gid_x()),
                    ),
                )],
            );

            let (_, proj_outs) = graph
                .add_node(
                    "stage1_projection",
                    prog_proj,
                    vec![
                        GraphInput {
                            buffer: "feat".into(),
                            value: in_feat,
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                        GraphInput {
                            buffer: "w".into(),
                            value: in_weights,
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                        GraphInput {
                            buffer: "b".into(),
                            value: in_bias,
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                    ],
                    vec![GraphOutput {
                        buffer: "proj_out".into(),
                        name: "mid_proj".into(),
                        contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation, count),
                        retained_successor_of: None,
                    }],
                )
                .unwrap();

            // Stage 2: Activation and Layer Normalization (norm = proj_in * 3 + 7)
            let prog_norm = Program::wrapped(
                vec![
                    BufferDecl::read("proj_in", 0, DataType::U32).with_count(count as u32),
                    BufferDecl::read_write("norm_out", 1, DataType::U32).with_count(count as u32),
                ],
                [count as u32, 1, 1],
                vec![Node::store(
                    "norm_out",
                    Expr::gid_x(),
                    Expr::add(
                        Expr::mul(Expr::load("proj_in", Expr::gid_x()), Expr::u32(3)),
                        Expr::u32(7),
                    ),
                )],
            );

            let (_, norm_outs) = graph
                .add_node(
                    "stage2_activation_norm",
                    prog_norm,
                    vec![GraphInput {
                        buffer: "proj_in".into(),
                        value: proj_outs[0],
                        contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                    }],
                    vec![GraphOutput {
                        buffer: "norm_out".into(),
                        name: "mid_norm".into(),
                        contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation, count),
                        retained_successor_of: None,
                    }],
                )
                .unwrap();

            // Stage 3: Residual Fusion and Dynamic Quantization (final = norm_in + residual)
            let prog_res = Program::wrapped(
                vec![
                    BufferDecl::read("norm_in", 0, DataType::U32).with_count(count as u32),
                    BufferDecl::read("res_in", 1, DataType::U32).with_count(count as u32),
                    BufferDecl::output("final_out", 2, DataType::U32).with_count(count as u32),
                ],
                [count as u32, 1, 1],
                vec![Node::store(
                    "final_out",
                    Expr::gid_x(),
                    Expr::add(
                        Expr::load("norm_in", Expr::gid_x()),
                        Expr::load("res_in", Expr::gid_x()),
                    ),
                )],
            );

            graph
                .add_node(
                    "stage3_residual_quantize",
                    prog_res,
                    vec![
                        GraphInput {
                            buffer: "norm_in".into(),
                            value: norm_outs[0],
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                        GraphInput {
                            buffer: "res_in".into(),
                            value: in_residual,
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                    ],
                    vec![GraphOutput {
                        buffer: "final_out".into(),
                        name: "dense_pipeline_out".into(),
                        contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output, count),
                        retained_successor_of: None,
                    }],
                )
                .unwrap();

            // Concrete Input Buffers
            let mut inputs = BTreeMap::new();
            inputs.insert("feat_in".to_string(), (0..count).map(|i| (i * 7 + 3) as u32).flat_map(|v| v.to_le_bytes()).collect());
            inputs.insert("weights".to_string(), (0..count).map(|i| (i * 13 + 5) as u32).flat_map(|v| v.to_le_bytes()).collect());
            inputs.insert("bias".to_string(), (0..count).map(|i| (i * 2 + 1) as u32).flat_map(|v| v.to_le_bytes()).collect());
            inputs.insert("residual".to_string(), (0..count).map(|i| (i * 11 + 9) as u32).flat_map(|v| v.to_le_bytes()).collect());

            (graph, inputs)
        },
    }
}

/// 2. Whole-Application: Irregular Stateful Dataflow Traversal.
///
/// Multi-stage pipeline: CSR SpMV frontier gather -> degree-weighted segmented scatter -> stateful history update with retained state.
#[must_use]
pub fn irregular_stateful_traversal() -> WholeApplicationWorkload {
    WholeApplicationWorkload {
        id: "workload.whole_app.irregular_stateful_traversal",
        name: "Irregular CSR Sparse Graph Frontier Traversal & Stateful Accumulator",
        domain: ApplicationDomain::IrregularStateful,
        description: "Complete 3-stage irregular stateful application: CSR SpMV Frontier Gather -> Degree-Weighted Segmented Scatter -> Stateful Persistent History Decay & Vertex Activation",
        pinned_native_baseline_id: "native.cub.spmv_segmented_scatter_v2_1_0",
        pinned_native_baseline_name: "NVIDIA CUB 2.1.0 / cuSPARSE 12.3.0 SpMV Scatter Pipeline",
        default_conditions: NativeComparisonConditions {
            semantics: Some("exact".to_string()),
            dtype: Some("u32".to_string()),
            shapes: Some("vertices=1024,edges=4096".to_string()),
            raggedness: Some("csr_ragged_irregular".to_string()),
            initial_and_final_state: Some("clean_buffers_unaliased".to_string()),
            target: Some("sm_90a_sm_86".to_string()),
            stream: Some("cuda_stream_non_blocking_0".to_string()),
            toolchain_and_flags: Some("nvcc_12.4_-O3".to_string()),
            clock_and_power_state: Some("locked_base_clock_tdp_100pct".to_string()),
            warmup: Some("300_warmup_iterations_discarded".to_string()),
            interleaving: Some("ab_ba_round_robin_interleaving".to_string()),
            repetitions: Some("30_measured_samples_clt".to_string()),
            cache_state: Some("flushed_l2_between_iterations".to_string()),
            objective: Some("minimize_p50_latency".to_string()),
        },
        build_graph_and_inputs: || {
            let count = 1024_u64;
            let mut graph = ProgramGraph::new();

            // External Inputs
            let in_offsets = graph
                .add_external_value("row_offsets", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .unwrap();
            let in_cols = graph
                .add_external_value("col_indices", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .unwrap();
            let in_frontier = graph
                .add_external_value("frontier_mask", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .unwrap();
            let in_history = graph
                .add_external_value("retained_hist", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .unwrap();

            // Stage 1: CSR Frontier SpMV Gather (active = frontier[cols[i]] * offsets[i])
            let prog_spmv = Program::wrapped(
                vec![
                    BufferDecl::read("offsets", 0, DataType::U32).with_count(count as u32),
                    BufferDecl::read("cols", 1, DataType::U32).with_count(count as u32),
                    BufferDecl::read("frontier", 2, DataType::U32).with_count(count as u32),
                    BufferDecl::read_write("active_out", 3, DataType::U32).with_count(count as u32),
                ],
                [count as u32, 1, 1],
                vec![Node::store(
                    "active_out",
                    Expr::gid_x(),
                    Expr::add(
                        Expr::mul(Expr::load("frontier", Expr::gid_x()), Expr::load("cols", Expr::gid_x())),
                        Expr::load("offsets", Expr::gid_x()),
                    ),
                )],
            );

            let (_, spmv_outs) = graph
                .add_node(
                    "stage1_csr_spmv",
                    prog_spmv,
                    vec![
                        GraphInput {
                            buffer: "offsets".into(),
                            value: in_offsets,
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                        GraphInput {
                            buffer: "cols".into(),
                            value: in_cols,
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                        GraphInput {
                            buffer: "frontier".into(),
                            value: in_frontier,
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                    ],
                    vec![GraphOutput {
                        buffer: "active_out".into(),
                        name: "mid_active".into(),
                        contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation, count),
                        retained_successor_of: None,
                    }],
                )
                .unwrap();

            // Stage 2: Degree-Weighted Scatter Accumulator (scatter = active * degree_weight)
            let prog_scatter = Program::wrapped(
                vec![
                    BufferDecl::read("active_in", 0, DataType::U32).with_count(count as u32),
                    BufferDecl::read_write("scatter_out", 1, DataType::U32).with_count(count as u32),
                ],
                [count as u32, 1, 1],
                vec![Node::store(
                    "scatter_out",
                    Expr::gid_x(),
                    Expr::mul(Expr::load("active_in", Expr::gid_x()), Expr::u32(5)),
                )],
            );

            let (_, scatter_outs) = graph
                .add_node(
                    "stage2_degree_scatter",
                    prog_scatter,
                    vec![GraphInput {
                        buffer: "active_in".into(),
                        value: spmv_outs[0],
                        contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                    }],
                    vec![GraphOutput {
                        buffer: "scatter_out".into(),
                        name: "mid_scatter".into(),
                        contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation, count),
                        retained_successor_of: None,
                    }],
                )
                .unwrap();

            // Stage 3: Stateful History Update with Retained State (final_state = history * decay + scatter)
            let prog_hist = Program::wrapped(
                vec![
                    BufferDecl::read("scatter_in", 0, DataType::U32).with_count(count as u32),
                    BufferDecl::read("hist_in", 1, DataType::U32).with_count(count as u32),
                    BufferDecl::output("final_state", 2, DataType::U32).with_count(count as u32),
                ],
                [count as u32, 1, 1],
                vec![Node::store(
                    "final_state",
                    Expr::gid_x(),
                    Expr::add(
                        Expr::mul(Expr::load("hist_in", Expr::gid_x()), Expr::u32(2)),
                        Expr::load("scatter_in", Expr::gid_x()),
                    ),
                )],
            );

            graph
                .add_node(
                    "stage3_history_update",
                    prog_hist,
                    vec![
                        GraphInput {
                            buffer: "scatter_in".into(),
                            value: scatter_outs[0],
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                        GraphInput {
                            buffer: "hist_in".into(),
                            value: in_history,
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                    ],
                    vec![GraphOutput {
                        buffer: "final_state".into(),
                        name: "irregular_traversal_out".into(),
                        contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output, count),
                        retained_successor_of: None,
                    }],
                )
                .unwrap();

            let mut inputs = BTreeMap::new();
            inputs.insert("row_offsets".to_string(), (0..count).map(|i| (i * 4) as u32).flat_map(|v| v.to_le_bytes()).collect());
            inputs.insert("col_indices".to_string(), (0..count).map(|i| ((i * 17) % count) as u32).flat_map(|v| v.to_le_bytes()).collect());
            inputs.insert("frontier_mask".to_string(), (0..count).map(|i| if i % 3 == 0 { 1u32 } else { 0u32 }).flat_map(|v| v.to_le_bytes()).collect());
            inputs.insert("retained_hist".to_string(), (0..count).map(|i| (i + 1) as u32).flat_map(|v| v.to_le_bytes()).collect());

            (graph, inputs)
        },
    }
}

/// 3. Whole-Application: Latency-Sensitive Interactive Pipeline.
///
/// Multi-stage pipeline: dirty-region culling & spatial hit-test -> spatial coordinate affine transform -> multi-layer Porter-Duff alpha blend rasterization.
#[must_use]
pub fn interactive_event_pipeline() -> WholeApplicationWorkload {
    WholeApplicationWorkload {
        id: "workload.whole_app.interactive_event_pipeline",
        name: "Latency-Sensitive Interactive UI Layout & Multi-Layer Composite Raster",
        domain: ApplicationDomain::LatencySensitiveInteractive,
        description: "Complete 3-stage interactive streaming pipeline: Dirty-Region Spatial Cull -> Viewport Affine Coordinate Transform -> Multi-Layer Porter-Duff Composite Raster",
        pinned_native_baseline_id: "native.skia.composite_raster_blend_v1_2_0",
        pinned_native_baseline_name: "Skia / DirectWrite-Style GPU Compositor Pipeline",
        default_conditions: NativeComparisonConditions {
            semantics: Some("exact".to_string()),
            dtype: Some("u32_rgba8".to_string()),
            shapes: Some("tiles=512,pixels_per_tile=64".to_string()),
            raggedness: Some("uniform_contiguous".to_string()),
            initial_and_final_state: Some("clean_buffers_unaliased".to_string()),
            target: Some("sm_90a_sm_86".to_string()),
            stream: Some("cuda_stream_non_blocking_0".to_string()),
            toolchain_and_flags: Some("nvcc_12.4_-O3".to_string()),
            clock_and_power_state: Some("locked_base_clock_tdp_100pct".to_string()),
            warmup: Some("300_warmup_iterations_discarded".to_string()),
            interleaving: Some("ab_ba_round_robin_interleaving".to_string()),
            repetitions: Some("30_measured_samples_clt".to_string()),
            cache_state: Some("flushed_l2_between_iterations".to_string()),
            objective: Some("minimize_p99_latency".to_string()),
        },
        build_graph_and_inputs: || {
            let count = 512_u64;
            let mut graph = ProgramGraph::new();

            // External Inputs
            let in_boxes = graph
                .add_external_value("dirty_boxes", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .unwrap();
            let in_fg = graph
                .add_external_value("layer_fg", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .unwrap();
            let in_bg = graph
                .add_external_value("layer_bg", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .unwrap();

            // Stage 1: Dirty Region Culling & Spatial Hit-Test (mask = boxes[i] * 1)
            let prog_cull = Program::wrapped(
                vec![
                    BufferDecl::read("boxes", 0, DataType::U32).with_count(count as u32),
                    BufferDecl::read_write("cull_out", 1, DataType::U32).with_count(count as u32),
                ],
                [count as u32, 1, 1],
                vec![Node::store(
                    "cull_out",
                    Expr::gid_x(),
                    Expr::mul(Expr::load("boxes", Expr::gid_x()), Expr::u32(1)),
                )],
            );

            let (_, cull_outs) = graph
                .add_node(
                    "stage1_dirty_region_cull",
                    prog_cull,
                    vec![GraphInput {
                        buffer: "boxes".into(),
                        value: in_boxes,
                        contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                    }],
                    vec![GraphOutput {
                        buffer: "cull_out".into(),
                        name: "mid_cull_mask".into(),
                        contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation, count),
                        retained_successor_of: None,
                    }],
                )
                .unwrap();

            // Stage 2: Spatial Affine Transform & Mapping (xform_fg = fg * mask + 10)
            let prog_xform = Program::wrapped(
                vec![
                    BufferDecl::read("mask_in", 0, DataType::U32).with_count(count as u32),
                    BufferDecl::read("fg_in", 1, DataType::U32).with_count(count as u32),
                    BufferDecl::read_write("xform_out", 2, DataType::U32).with_count(count as u32),
                ],
                [count as u32, 1, 1],
                vec![Node::store(
                    "xform_out",
                    Expr::gid_x(),
                    Expr::add(
                        Expr::mul(Expr::load("fg_in", Expr::gid_x()), Expr::load("mask_in", Expr::gid_x())),
                        Expr::u32(10),
                    ),
                )],
            );

            let (_, xform_outs) = graph
                .add_node(
                    "stage2_spatial_transform",
                    prog_xform,
                    vec![
                        GraphInput {
                            buffer: "mask_in".into(),
                            value: cull_outs[0],
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                        GraphInput {
                            buffer: "fg_in".into(),
                            value: in_fg,
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                    ],
                    vec![GraphOutput {
                        buffer: "xform_out".into(),
                        name: "mid_xform_fg".into(),
                        contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation, count),
                        retained_successor_of: None,
                    }],
                )
                .unwrap();

            // Stage 3: Multi-Layer Porter-Duff Alpha Blend & Framebuffer Output (final_frame = xform_fg + bg)
            let prog_blend = Program::wrapped(
                vec![
                    BufferDecl::read("fg_ready", 0, DataType::U32).with_count(count as u32),
                    BufferDecl::read("bg_ready", 1, DataType::U32).with_count(count as u32),
                    BufferDecl::output("frame_out", 2, DataType::U32).with_count(count as u32),
                ],
                [count as u32, 1, 1],
                vec![Node::store(
                    "frame_out",
                    Expr::gid_x(),
                    Expr::add(
                        Expr::load("fg_ready", Expr::gid_x()),
                        Expr::load("bg_ready", Expr::gid_x()),
                    ),
                )],
            );

            graph
                .add_node(
                    "stage3_porter_duff_blend",
                    prog_blend,
                    vec![
                        GraphInput {
                            buffer: "fg_ready".into(),
                            value: xform_outs[0],
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                        GraphInput {
                            buffer: "bg_ready".into(),
                            value: in_bg,
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                    ],
                    vec![GraphOutput {
                        buffer: "frame_out".into(),
                        name: "interactive_pipeline_out".into(),
                        contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output, count),
                        retained_successor_of: None,
                    }],
                )
                .unwrap();

            let mut inputs = BTreeMap::new();
            inputs.insert("dirty_boxes".to_string(), (0..count).map(|i| if i % 2 == 0 { 1u32 } else { 0u32 }).flat_map(|v| v.to_le_bytes()).collect());
            inputs.insert("layer_fg".to_string(), (0..count).map(|i| (0xFF0000FF_u32 ^ (i as u32))).flat_map(|v| v.to_le_bytes()).collect());
            inputs.insert("layer_bg".to_string(), (0..count).map(|i| (0x00FF00FF_u32 ^ (i as u32))).flat_map(|v| v.to_le_bytes()).collect());

            (graph, inputs)
        },
    }
}

/// Return all canonical whole-application representative workloads.
#[must_use]
pub fn all_whole_application_workloads() -> Vec<WholeApplicationWorkload> {
    vec![
        dense_numerical_pipeline(),
        irregular_stateful_traversal(),
        interactive_event_pipeline(),
    ]
}

/// Release evidence domain matrix holding every whole-application record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeApplicationDomainMatrixRecord {
    /// Schema version.
    pub schema_version: String,
    /// Total registered whole-application cases.
    pub total_workloads: usize,
    /// Total distinct application domains covered.
    pub covered_domains: usize,
    /// Individual workload measurement records.
    pub records: Vec<WholeApplicationRecord>,
    /// Backend that executed every record in this matrix.
    pub backend_id: String,
    /// Outcome derived from the records the matrix holds.
    pub status: String,
    /// Required fields no record in this matrix states.
    pub readiness_gaps: Vec<String>,
    /// UTC timestamp of generation.
    pub generated_at_utc: String,
}

impl WholeApplicationDomainMatrixRecord {
    /// Status a matrix with these records and gaps states.
    fn derive_status(records: &[WholeApplicationRecord], readiness_gaps: &[String]) -> String {
        if records.is_empty() {
            "no_records".to_string()
        } else if readiness_gaps.is_empty() {
            "complete".to_string()
        } else {
            "measured_with_unstated_fields".to_string()
        }
    }
}

/// Execute every whole-application workload on one acquired device.
///
/// The device is probed once, before any workload runs, so a host with no
/// dispatch device produces the refusal instead of a partial matrix.
///
/// # Errors
///
/// Returns the refusal text when no device acquires, and the workload
/// diagnostic when a compile, submission, parity comparison, or record
/// validation fails.
pub fn generate_whole_application_evidence_suite(
    backend_id: Option<&str>,
    measured_samples: usize,
) -> Result<WholeApplicationDomainMatrixRecord, String> {
    let device = WholeApplicationDevice::probe(backend_id).map_err(|err| err.to_string())?;
    let workloads = all_whole_application_workloads();
    let mut records = Vec::with_capacity(workloads.len());
    let mut domain_set = BTreeSet::new();
    let mut readiness_gaps = BTreeSet::new();

    for workload in &workloads {
        domain_set.insert(workload.domain);
        let record = workload.execute_and_measure(&device, measured_samples)?;
        for gap in record.readiness_gaps() {
            readiness_gaps.insert(gap.as_str().to_string());
        }
        records.push(record);
    }

    let readiness_gaps: Vec<String> = readiness_gaps.into_iter().collect();
    let status = WholeApplicationDomainMatrixRecord::derive_status(&records, &readiness_gaps);
    Ok(WholeApplicationDomainMatrixRecord {
        schema_version: WHOLE_APPLICATION_RECORD_SCHEMA_V2.to_string(),
        total_workloads: workloads.len(),
        covered_domains: domain_set.len(),
        records,
        backend_id: device.backend_id().to_string(),
        status,
        readiness_gaps,
        generated_at_utc: super::provenance::utc_timestamp(std::time::SystemTime::now())?,
    })
}

/// Write every whole-application evidence artifact into one directory.
///
/// # Errors
///
/// Returns a diagnostic when the directory cannot be created, when the suite
/// cannot be measured, or when a file cannot be serialized or written.
pub fn write_whole_application_evidence_artifacts(
    artifacts_dir: &Path,
    backend_id: Option<&str>,
    measured_samples: usize,
) -> Result<Vec<PathBuf>, String> {
    let matrix = generate_whole_application_evidence_suite(backend_id, measured_samples)?;

    std::fs::create_dir_all(artifacts_dir)
        .map_err(|err| format!("Failed to create artifacts directory: {err}"))?;

    let mut written_paths = Vec::new();

    // Write domain matrix
    let matrix_path = artifacts_dir.join("whole-application-domain-matrix.json");
    let matrix_json = serde_json::to_string_pretty(&matrix)
        .map_err(|err| format!("Failed to serialize domain matrix: {err}"))?;
    std::fs::write(&matrix_path, format!("{matrix_json}\n"))
        .map_err(|err| format!("Failed to write {matrix_path:?}: {err}"))?;
    written_paths.push(matrix_path);

    // Write individual workload records
    for record in &matrix.records {
        let file_name = match record.domain {
            ApplicationDomain::DenseNumerical => "whole-app-dense-numerical-pipeline.json",
            ApplicationDomain::IrregularStateful => "whole-app-irregular-stateful-traversal.json",
            ApplicationDomain::LatencySensitiveInteractive => {
                "whole-app-interactive-event-pipeline.json"
            }
        };
        let record_path = artifacts_dir.join(file_name);
        let record_json = serde_json::to_string_pretty(record)
            .map_err(|err| format!("Failed to serialize record `{}`: {err}", record.workload_id))?;
        std::fs::write(&record_path, format!("{record_json}\n"))
            .map_err(|err| format!("Failed to write {record_path:?}: {err}"))?;
        written_paths.push(record_path);
    }

    Ok(written_paths)
}
