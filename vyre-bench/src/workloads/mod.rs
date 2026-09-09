//! Benchmark workload specifications, native baseline comparisons, and per-cell verdicts.
//!
//! BACKLOG row 47 implementation module providing:
//! - Required measurement field introspection and verification (`fields`)
//! - Payload generation provenance validation (`provenance`)
//! - Strict 14-condition comparison equality validation (`equality`)
//! - Comprehensive 15-category case measurement records (`measurement`)
//! - Version-pinned external native kernel baseline catalog (`native_baseline`)
//! - Per-cell verdict engine and statistical equivalence evaluation (`verdict`)
//! - Canonical complete graph and adversarial region workload specifications (`definitions`)

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
pub use native_baseline::{NativeBaselineCatalog, VersionPinnedNativeBaseline};
pub use provenance::{validate_payload_provenance, PayloadProvenance, ProvenanceRefusal};
pub use verdict::{
    evaluate_cell_comparison, generate_floor_comparison_report, AggregateVerdict, CellEvaluation,
    CellVerdict, FloorComparisonReport, MeasurementCell,
};
pub use whole_app::{
    all_whole_application_workloads, dense_numerical_pipeline,
    generate_whole_application_evidence_suite, interactive_event_pipeline,
    irregular_stateful_traversal, write_whole_application_evidence_artifacts,
    ApplicationDomain, MissingRequiredWholeAppFieldsError, RequiredWholeApplicationField,
    WholeAppNativeComparisonRecord, WholeAppParityRecord, WholeAppStateMetrics,
    WholeAppThroughputRecord, WholeApplicationDomainMatrixRecord, WholeApplicationRecord,
    WholeApplicationRefusal, WholeApplicationWorkload, WHOLE_APPLICATION_RECORD_SCHEMA_V1,
};
