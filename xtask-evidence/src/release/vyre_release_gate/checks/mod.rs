//! Shared release-gate check helpers.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use super::paths::{read_text_bounded, resolve_artifact_path, resolve_manifest_path};
use super::gate_inputs::Requirement;

fn check_duplicate_object_rows(
    report: &serde_json::Value,
    array_field: &str,
    value_field: &str,
    context: &str,
    failures: &mut Vec<String>,
) {
    let duplicates =
        crate::bench::benchmark_evidence_semantics::duplicate_nonblank_object_array_field_values(
            report,
            array_field,
            value_field,
        );
    if !duplicates.is_empty() {
        failures.push(format!(
            "{context}: {}",
            duplicates.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
}

mod backend_suite;
mod benchmark_backend;
mod benchmark_provenance;
mod markdown_evidence;
mod optimization_evidence;
mod parser_conformance;
mod release_evidence;
mod workload_evidence;

// The check helpers were one module before the split; re-export keeps
// `checks::<helper>` paths intact for callers and siblings.
pub(crate) use backend_suite::*;
pub(crate) use benchmark_backend::*;
pub(crate) use benchmark_provenance::*;
pub(crate) use markdown_evidence::*;
pub(crate) use optimization_evidence::*;
pub(crate) use parser_conformance::*;
pub(crate) use release_evidence::*;
pub(crate) use workload_evidence::*;
