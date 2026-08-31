use std::path::Path;

use super::super::checks::*;
use super::super::gate_inputs::Requirement;

pub(super) fn check(requirement: &Requirement, base_dir: &Path, failures: &mut Vec<String>) {
    let Some(matrix) = first_json_evidence(requirement, base_dir, "hygiene-matrix.json", failures)
    else {
        return;
    };
    let scanned = u64_field(&matrix, "scanned_files", 0);
    let blockers = array_len(&matrix, "blockers");
    if scanned == 0 {
        failures.push("requirement `release-hygiene` scanned zero source files".to_string());
    }
    let finding_count = array_len(&matrix, "findings");
    let summary_count = matrix
        .get("finding_summary")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("count").and_then(serde_json::Value::as_u64))
                .sum::<u64>() as usize
        })
        .unwrap_or(usize::MAX);
    if finding_count != summary_count {
        failures.push(format!(
            "requirement `release-hygiene` finding_summary count {summary_count} does not match findings count {finding_count}"
        ));
    }
    check_hygiene_release_surface_coverage("release-hygiene", &matrix, failures);
    for required_root in ["."] {
        if !matrix
            .get("scanned_roots")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|roots| {
                roots
                    .iter()
                    .any(|root| root.as_str() == Some(required_root))
            })
        {
            failures.push(format!(
                "requirement `release-hygiene` scanned_roots is missing `{required_root}`"
            ));
        }
    }
    if blockers != 0 {
        failures.push(format!(
            "requirement `release-hygiene` matrix still reports {blockers} blocker(s)"
        ));
    }
    for suffix in [
        "no-stubs-scan.json",
        "no-hidden-fallback-scan.json",
        "resource-bound-scan.json",
        "error-surface-scan.json",
        "cargo-wrapper-scan.json",
    ] {
        check_json_evidence_has_no_blockers(requirement, base_dir, suffix, failures);
    }
}
