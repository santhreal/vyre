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

include!("part1.rs");
include!("part2.rs");
include!("part3.rs");
include!("part4.rs");
include!("part5.rs");
include!("part6.rs");
include!("part7.rs");
include!("part8.rs");
