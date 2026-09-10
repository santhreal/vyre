//! Benchmark workload specifications, native baseline comparisons, and per-cell verdicts.
//!
//! The modules here own one part of a recorded comparison each:
//! - Required measurement field introspection and verification (`fields`)
//! - Payload generation provenance and host and clock capture (`provenance`)
//! - Strict 14-condition comparison equality validation (`equality`)
//! - 15-category case measurement records (`measurement`)
//! - Version-pinned external native kernel baseline catalog (`native_baseline`)
//! - Per-cell verdict engine and statistical equivalence evaluation (`verdict`)
//! - Canonical complete graph and adversarial region workload specifications (`definitions`)
//! - Whole-application device measurement and release evidence (`whole_app`)

pub mod definitions;
pub mod equality;
pub mod fields;
pub mod measurement;
pub mod native_baseline;
pub mod provenance;
pub mod verdict;
pub mod whole_app;

pub use definitions::{WorkloadDomain, WorkloadSpecification};
pub use equality::{
    validate_equality_conditions, EqualityConditionRefusal, EqualityDimension,
    NativeComparisonConditions,
};
pub use fields::{MissingRequiredFieldsError, RequiredMeasurementField};
pub use measurement::{
    ArtifactBehavior, CaseMeasurementRecord, EstimatorKind, MemoryMetrics, SharedMemoryMetrics,
    SpillMetrics, StatisticalEstimator, ThroughputMetrics, WorkspaceTraffic,
};
pub use native_baseline::{
    whole_application_native_baselines, NativeBaselineCatalog, VersionPinnedNativeBaseline,
};
pub use provenance::{
    utc_timestamp, validate_payload_provenance, MeasurementProvenance, PayloadProvenance,
    ProvenanceRefusal,
};
pub use verdict::{
    evaluate_cell_comparison, generate_floor_comparison_report, AggregateVerdict, CellEvaluation,
    CellVerdict, FloorComparisonReport, MeasurementCell,
};
pub use whole_app::{
    all_whole_application_workloads, dense_numerical_pipeline,
    generate_whole_application_evidence_suite, interactive_event_pipeline,
    irregular_stateful_traversal, write_whole_application_evidence_artifacts, ApplicationDomain,
    MissingRequiredWholeAppFieldsError, RecordFieldSource, RequiredWholeApplicationField,
    WholeAppNativeBaselineUnmeasured, WholeAppNativeComparisonRecord, WholeAppParityRecord,
    WholeAppProductionRouteRecord, WholeAppStateMetrics, WholeAppThroughputRecord,
    WholeApplicationDevice, WholeApplicationDomainMatrixRecord, WholeApplicationRecord,
    WholeApplicationRecordField, WholeApplicationRefusal, WholeApplicationWorkload,
    COMPARISON_EQUIVALENCE_BAND, MIN_MEASURED_SAMPLES, WHOLE_APPLICATION_RECORD_SCHEMA_V1,
    WHOLE_APPLICATION_RECORD_SCHEMA_V2,
};
