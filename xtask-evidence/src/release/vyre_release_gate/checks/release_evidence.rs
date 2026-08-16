use super::*;

pub(crate) fn check_hygiene_release_surface_coverage(
    requirement_id: &str,
    matrix: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    crate::bench::benchmark_evidence_semantics::inspect_hygiene_release_surface_coverage(
        &format!("requirement `{requirement_id}` hygiene"),
        matrix,
        failures,
    );
}

/// Judge the committed release-evidence census.
///
/// The rules and the field names live with the generator that writes the
/// artifact, so this reads nothing itself: a list of required generators here
/// went stale the moment the sweep stopped spawning them, and a reader that
/// defaults a missing counter reports a clean run as broken and a broken one as
/// clean.
pub(crate) fn check_release_evidence_run(
    requirement: &Requirement,
    run: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    crate::release::release_evidence::judge_committed_run(
        run,
        &format!("requirement `{}`", requirement.id),
        failures,
    );
}
