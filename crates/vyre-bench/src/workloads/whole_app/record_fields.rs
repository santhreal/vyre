//! Serialized field census of a whole-application record.
//!
//! Every field a record serializes appears here with the source its value is
//! read from, so adding a field forces a recorded decision about where the
//! value comes from rather than letting an unsourced field ship.

use super::*;

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
