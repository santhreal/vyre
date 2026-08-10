//! Shared release-gate check helpers.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use super::paths::{read_text_bounded, resolve_artifact_path, resolve_manifest_path};
use super::types::Requirement;

fn check_duplicate_object_rows(
    report: &serde_json::Value,
    array_field: &str,
    value_field: &str,
    context: &str,
    failures: &mut Vec<String>,
) {
    let duplicates =
        crate::benchmark_evidence_semantics::duplicate_nonblank_object_array_field_values(
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

include!("release_evidence.rs");
include!("workload_evidence.rs");
include!("benchmark_backend.rs");
include!("optimization_evidence.rs");
include!("parser_conformance.rs");
include!("benchmark_provenance.rs");
include!("backend_suite.rs");
include!("markdown_evidence.rs");
