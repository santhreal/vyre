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

/// Artifacts every release-evidence generator command must produce.
const REQUIRED_GENERATORS: &[(&str, &[&str])] = &[
    (
        "version-matrix",
        &["version-matrix.json", "release-tag-plan.json"],
    ),
    ("backend-matrix", &["backend-matrix.json"]),
    ("conformance-matrix", &["conformance-matrix.json"]),
    ("release-workload-matrix", &["release-workload-matrix.json"]),
    (
        "hygiene-matrix",
        &[
            "hygiene-matrix.json",
            "no-stubs-scan.json",
            "no-hidden-fallback-scan.json",
            "resource-bound-scan.json",
            "error-surface-scan.json",
            "cargo-wrapper-scan.json",
        ],
    ),
    ("metadata-matrix", &["metadata-matrix.json"]),
    ("feature-matrix", &["feature-matrix.json"]),
    ("package-readiness", &["publish-readiness.json"]),
    (
        "optimization-corpus",
        &[
            "optimization-corpus.json",
            "optimization-corpus-contracts.json",
            "optimization-family-manifest.json",
            "optimization-case-manifest.json",
            "optimizer-pass-manifest.json",
        ],
    ),
    (
        "optimization-matrix",
        &["optimization-integration-matrix.json"],
    ),
];

pub(crate) fn check_release_evidence_run(
    requirement: &Requirement,
    run: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    let total = run
        .get("total_commands")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let schema_version = run
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let successful = run
        .get("successful_commands")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let required = run
        .get("required_command_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let command_failures = run
        .get("command_failures")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    let artifact_failures = run
        .get("artifact_failures")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    let blockers = run
        .get("blockers")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if schema_version < 2
        || total < 13
        || required < 13
        || successful != total
        || command_failures != 0
        || artifact_failures != 0
        || blockers != 0
    {
        failures.push(format!(
            "requirement `{}` release-evidence-run must be schema>=2 and clean: schema_version={schema_version}, total={total}, required={required}, successful={successful}, command_failures={command_failures}, artifact_failures={artifact_failures}, blockers={blockers}",
            requirement.id
        ));
    }

    let commands = run
        .get("commands")
        .and_then(serde_json::Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    for (generator, expected_artifacts) in REQUIRED_GENERATORS {
        let Some(command) = commands.iter().find(|command| {
            command
                .get("args")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|args| {
                    args.iter()
                        .any(|arg| arg.as_str().is_some_and(|arg| arg == *generator))
                })
        }) else {
            failures.push(format!(
                "requirement `{}` release-evidence-run is missing generator `{generator}`",
                requirement.id
            ));
            continue;
        };

        let status = command
            .get("status")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if status != "success" {
            failures.push(format!(
                "requirement `{}` release-evidence-run generator `{generator}` status is `{status}`, expected `success`",
                requirement.id
            ));
        }

        let artifacts = command
            .get("expected_artifacts")
            .and_then(serde_json::Value::as_array)
            .map_or(&[][..], Vec::as_slice);
        let artifact_statuses = command
            .get("artifact_statuses")
            .and_then(serde_json::Value::as_array)
            .map_or(&[][..], Vec::as_slice);
        for expected in *expected_artifacts {
            if !artifacts.iter().any(|artifact| {
                artifact
                    .as_str()
                    .is_some_and(|artifact| artifact.ends_with(expected))
            }) {
                failures.push(format!(
                    "requirement `{}` release-evidence-run generator `{generator}` does not declare expected artifact `{expected}`",
                    requirement.id
                ));
            }
            let Some(status) = artifact_statuses.iter().find(|status| {
                status
                    .get("path")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|path| path.ends_with(expected))
            }) else {
                failures.push(format!(
                    "requirement `{}` release-evidence-run generator `{generator}` has no artifact status for `{expected}`",
                    requirement.id
                ));
                continue;
            };
            let exists = status
                .get("exists")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            let bytes = status
                .get("bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let read_error = status.get("read_error");
            let read_error_is_clean = read_error.is_some_and(serde_json::Value::is_null);
            if !exists || bytes == 0 || !read_error_is_clean {
                failures.push(format!(
                    "requirement `{}` release-evidence-run generator `{generator}` artifact `{expected}` exists={exists} bytes={bytes} read_error={}",
                    requirement.id,
                    read_error
                        .map(serde_json::Value::to_string)
                        .unwrap_or_else(|| "<missing>".to_string())
                ));
            }
        }
    }
}
