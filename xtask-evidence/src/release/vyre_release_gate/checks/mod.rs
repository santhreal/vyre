//! Shared release-gate check helpers.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use super::gate_inputs::Requirement;
use super::paths::{read_text_bounded, resolve_artifact_path, resolve_manifest_path};

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

/// Reject an inspection record that reports a non-positive aggregate wall-clock
/// percentile, naming `subject` as the thing that reported it.
///
/// Every caller reads the same fields with the same missing-is-zero rule and
/// ends its message the same way, so the fields, the rule and the sentence are
/// stated once. `subject` is the only part a caller decides.
pub(crate) fn check_aggregate_wall_percentiles_positive(
    record: &serde_json::Value,
    subject: &str,
    failures: &mut Vec<String>,
) {
    for field in crate::bench::benchmark_evidence_semantics::AGGREGATE_WALL_PERCENTILE_FIELDS {
        if record
            .get(field)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            failures.push(format!("{subject} has non-positive `{field}`"));
        }
    }
}

/// A numeric field of an evidence record, or `default` when it is absent or not
/// a number.
///
/// Callers pass the value that fails their own comparison as `default`, which
/// is how a missing field is rejected rather than read as zero.
pub(crate) fn u64_field(value: &serde_json::Value, field: &str, default: u64) -> u64 {
    value
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(default)
}

/// The length of an array field, or `usize::MAX` when it is absent or not an
/// array.
///
/// Every caller demands a count of zero, so an absent array has to read as a
/// number no caller accepts. Fifty-odd readers spelled this out inline and each
/// one was free to write `unwrap_or(0)` instead, which turns a missing blocker
/// list into proof that there are no blockers.
pub(crate) fn array_len(value: &serde_json::Value, field: &str) -> usize {
    value
        .get(field)
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len)
}

/// The first JSON evidence file matching `suffix`, refused unless it declares
/// schema 2 or later. `kind` names the evidence class in the failure.
///
/// The schema floor is one rule, and every reader that enforced it wrote its
/// own copy with its own sentence. One of those copies never said which file it
/// had read, so a requirement citing several artifacts reported a schema
/// failure nobody could locate. The sentence is built here and always names the
/// file.
pub(crate) fn schema_2_json_evidence(
    requirement: &Requirement,
    base_dir: &Path,
    suffix: &str,
    kind: &str,
    failures: &mut Vec<String>,
) -> Option<serde_json::Value> {
    let report = first_json_evidence(requirement, base_dir, suffix, failures)?;
    let schema_version = u64_field(&report, "schema_version", 0);
    if schema_version < 2 {
        failures.push(format!(
            "requirement `{}` {kind} `{suffix}` schema_version={schema_version}; expected schema>=2",
            requirement.id
        ));
    }
    Some(report)
}

/// The release workload matrix cited by `requirement`.
///
/// Three semantic checks name the same artifact, so the name is stated here.
pub(crate) fn release_workload_matrix(
    requirement: &Requirement,
    base_dir: &Path,
    failures: &mut Vec<String>,
) -> Option<serde_json::Value> {
    first_json_evidence(
        requirement,
        base_dir,
        "release-workload-matrix.json",
        failures,
    )
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
