#![allow(
    clippy::doc_lazy_continuation,
    clippy::double_must_use,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::collapsible_if,
    clippy::match_like_matches_macro,
    clippy::redundant_closure,
    clippy::too_many_arguments,
    clippy::nonminimal_bool,
    clippy::derivable_impls,
    clippy::single_char_add_str,
    clippy::type_complexity,
    clippy::map_entry,
    clippy::only_used_in_recursion,
    clippy::manual_flatten,
    clippy::explicit_counter_loop
)]
//! Inspection and diagnostic helpers for Vyre IR and lowered kernel descriptors.
///
/// Canonical compiler artifact and selected-plan diagnostics.
pub mod artifact_report;
/// Loop-carrier diagnostics.
pub mod carriers;
/// Dangling descriptor-reference diagnostics.
pub mod dangling;
/// Structural descriptor comparison and rewrite bisection.
pub mod descriptor_diff;
/// Human-readable descriptor rendering.
pub mod descriptor_dump;
/// Reusable diagnostic fixtures.
pub mod fixtures;
/// Human-readable Naga module rendering.
pub mod naga_dump;
/// Naga validation and binding failure traces.
pub mod naga_trace;
pub(crate) mod path_map_serde;
pub mod scan_explain;
/// Source-level assignment traversal.
pub mod source_walker;
/// WGSL emission and source-line mapping.
pub mod wgsl;

pub use artifact_report::{ArtifactReport, TargetPayloadReport};
pub use carriers::{carrier_summary, find_uncarriered_assigns, CarrierSummary, UncarrieredAssign};
pub use dangling::{find_dangling_refs, DanglingRef};
pub use descriptor_diff::{bisect_rewrites, diff_descriptors, DescriptorDiff, RewriteBisectResult};
pub use descriptor_dump::{dump_descriptor, DescriptorDump, DescriptorDumpOptions};
pub use naga_dump::{dump_naga_module, NagaDump};
pub use naga_trace::{
    failure_trace, failure_trace_wgsl, load_bind_result_log, BindResultLogError, FailureTrace,
};
pub use scan_explain::{
    scan_explain_report, ScanExplainEngine, ScanExplainError, ScanExplainExactnessClass,
    ScanExplainFactor, ScanExplainRejectedEngine, ScanExplainReport, ScanExplainRouteEvidence,
    ScanExplainVerifierFragment, ScanFactorRole, SCAN_EXPLAIN_REPORT_SCHEMA_VERSION,
};
pub use wgsl::{dump_wgsl, dump_wgsl_with_lines, WgslDump};
