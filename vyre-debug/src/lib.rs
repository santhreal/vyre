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

pub use artifact_report::{ArtifactReport, TargetPayloadReport};
pub use carriers::{carrier_summary, find_uncarriered_assigns, CarrierSummary, UncarrieredAssign};
pub use dangling::{find_dangling_refs, DanglingRef};
pub use descriptor_diff::{diff_descriptors, DescriptorDiff};
pub use descriptor_dump::{dump_descriptor, DescriptorDump, DescriptorDumpOptions};
pub use naga_dump::{dump_naga_module, NagaDump};
pub use naga_trace::{
    failure_trace, failure_trace_wgsl, load_bind_result_log, BindResultLogError, FailureTrace,
};
pub use wgsl::{dump_wgsl, dump_wgsl_with_lines, WgslDump};
