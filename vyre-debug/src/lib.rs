//! Inspection and diagnostic helpers for Vyre IR and lowered kernel descriptors.
/// Canonical compiler artifact and selected-plan diagnostics.
pub(crate) mod artifact_report;
pub(crate) mod body_path_map;
/// Loop-carrier diagnostics.
pub(crate) mod carriers;
/// Dangling descriptor-reference diagnostics.
pub(crate) mod dangling;
/// Structural descriptor comparison and rewrite bisection.
pub(crate) mod descriptor_diff;
/// Human-readable descriptor rendering.
pub(crate) mod descriptor_dump;
/// Reusable diagnostic fixtures.
pub mod fixtures;
/// Human-readable Naga module rendering.
pub(crate) mod naga_dump;
/// Naga validation and binding failure traces.
pub(crate) mod naga_trace;
/// Source-level assignment traversal.
pub mod source_assignments;
/// WGSL emission and source-line mapping.
pub(crate) mod wgsl;
/// Sanitizer correctness failures and PMU performance expectations.
pub mod sanitizer;

pub use artifact_report::{ArtifactReport, TargetPayloadReport};
pub use carriers::{carrier_summary, find_uncarriered_assigns, CarrierSummary, UncarrieredAssign};
pub use dangling::{find_dangling_refs, DanglingRef};
pub use descriptor_diff::{diff_descriptors, DescriptorDiff};
pub use descriptor_dump::{dump_descriptor, DescriptorDump, DescriptorDumpOptions};
pub use naga_dump::{dump_naga_module, NagaDump};
pub use naga_trace::{
    failure_trace, failure_trace_wgsl, load_bind_result_log, BindResultLogError, FailureTrace,
};
pub use sanitizer::{
    PmuExpectation, PmuMeasurement, PmuWarning, PmuWorkloadClass, SanitizerFailure, SanitizerKind,
};
pub use wgsl::{dump_wgsl, dump_wgsl_with_lines, WgslDump};
