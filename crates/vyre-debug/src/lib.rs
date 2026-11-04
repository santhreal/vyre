//! Inspection and diagnostic helpers for Vyre IR and lowered kernel descriptors.
/// Canonical compiler artifact and selected-plan diagnostics.
/// Capability classification for neutral vs target-specific debug features.
pub(crate) mod capability;

/// Allocation and memory map inspection.
pub(crate) mod allocation_report;
/// Compiled Artifact and SelectedPlan structural diffs.
pub(crate) mod artifact_diff;
pub(crate) mod artifact_report;
pub(crate) mod body_path_map;
/// Candidate funnel and prune reason reports.
pub(crate) mod candidate_report;
/// Loop-carrier diagnostics.
pub(crate) mod carriers;
/// Causal receipt and critical-path inspection report.
pub(crate) mod causal_report;
/// Source-level assignment traversal.
/// Five-level compiler view and structural diffs.
pub(crate) mod compiler_level;
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
/// Frontend Program and ProgramGraph structural diffs.
pub(crate) mod program_diff;
/// Sanitizer correctness failures and PMU performance expectations.
pub(crate) mod sanitizer;
pub mod source_assignments;
/// WGSL emission and source-line mapping.
pub(crate) mod wgsl;

pub use allocation_report::{
    diff_allocations, AllocationDiff, AllocationReport, ResourceAllocationInfo,
};
pub use artifact_diff::{diff_artifacts, diff_selected_plans, ArtifactDiff, PlanDiff};
pub use artifact_report::{ArtifactReport, TargetPayloadReport};
pub use candidate_report::{
    diff_search_certificates, CandidateReport, EliminatedFamilyReport, SearchCertificateDiff,
};
pub use capability::{
    neutral_debug_capabilities, DebugCapabilityInfo, DebugCapabilityKind, DEBUG_CAPABILITIES,
};
pub use carriers::{carrier_summary, find_uncarriered_assigns, CarrierSummary, UncarrieredAssign};
pub use causal_report::CausalReceiptReport;
pub use compiler_level::{diff_compiler_levels, CompilerLevelDiff, CompilerLevelView};
pub use dangling::{find_dangling_refs, DanglingRef};
pub use descriptor_diff::{diff_descriptors, DescriptorDiff};
pub use descriptor_dump::{dump_descriptor, DescriptorDump, DescriptorDumpOptions};
pub use naga_dump::{dump_naga_module, NagaDump};
pub use naga_trace::{
    failure_trace, failure_trace_wgsl, load_bind_result_log, BindResultLogError, FailureTrace,
};
pub use program_diff::{diff_program_graphs, diff_programs, GraphDiff, ProgramDiff};
pub use sanitizer::{
    PmuExpectation, PmuMeasurement, PmuWarning, PmuWorkloadClass, SanitizerFailure, SanitizerKind,
};
pub use wgsl::{dump_wgsl, dump_wgsl_with_lines, WgslDump};
