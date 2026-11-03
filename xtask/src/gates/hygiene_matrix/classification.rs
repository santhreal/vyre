use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use structure_gate::cfg_test::TestSources;

use super::records::{
    HygieneClassificationSummary, HygieneFinding, HygieneFindingClass, HygieneFindingSummary,
    HygieneIntakeSummary, StructuralGateArtifact,
};
use super::rules::is_hidden_fallback_pattern;
use super::structural_gates::is_declared_structural_gate;
use super::threshold_policy::relative_to_vyre;

pub(crate) fn finding_summary(findings: &[HygieneFinding]) -> Vec<HygieneFindingSummary> {
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    for finding in findings {
        *counts.entry(finding.pattern.to_string()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(|(pattern, count)| HygieneFindingSummary { pattern, count })
        .collect()
}

pub(crate) fn classify_findings(
    vyre_root: &Path,
    findings: &[HygieneFinding],
    hot_paths: &std::collections::BTreeSet<String>,
    structural_gates: &StructuralGateArtifact,
    tests: &TestSources,
) -> Vec<HygieneFindingClass> {
    findings
        .iter()
        .map(|finding| {
            let member = hygiene_member_for_path(vyre_root, &finding.path);
            let surface = hygiene_surface_for_path(vyre_root, &finding.path, tests);
            let hot_path = hygiene_finding_is_hot_path(vyre_root, &finding.path, hot_paths);
            let declared = is_declared_structural_gate(vyre_root, finding, structural_gates);
            let risk = hygiene_risk(finding.pattern, surface, hot_path, declared);
            HygieneFindingClass {
                path: finding.path.clone(),
                line: finding.line,
                pattern: finding.pattern,
                member,
                surface,
                risk,
                hot_path,
                release_blocker: risk == "release_blocker",
            }
        })
        .collect()
}

pub(crate) fn classification_summary(
    classes: &[HygieneFindingClass],
) -> Vec<HygieneClassificationSummary> {
    let mut counts = BTreeMap::<(&str, &'static str, &'static str, bool, bool), usize>::new();
    for class in classes {
        *counts
            .entry((
                class.member.as_str(),
                class.surface,
                class.risk,
                class.hot_path,
                class.release_blocker,
            ))
            .or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(
            |((member, surface, risk, hot_path, release_blocker), count)| {
                HygieneClassificationSummary {
                    member: member.to_string(),
                    surface,
                    risk,
                    hot_path,
                    release_blocker,
                    count,
                }
            },
        )
        .collect()
}

pub(crate) fn hygiene_intake_summary(classes: &[HygieneFindingClass]) -> Vec<HygieneIntakeSummary> {
    let mut counts =
        BTreeMap::<(&str, &'static str, &'static str, bool, &'static str, bool), usize>::new();
    for class in classes {
        *counts
            .entry((
                class.member.as_str(),
                class.surface,
                class.risk,
                class.hot_path,
                class.pattern,
                class.release_blocker,
            ))
            .or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(
            |((member, surface, risk, hot_path, pattern, release_blocker), count)| {
                HygieneIntakeSummary {
                    member: member.to_string(),
                    surface,
                    risk,
                    hot_path,
                    pattern,
                    release_blocker,
                    count,
                }
            },
        )
        .collect()
}

/// The first repository-relative segment of `path`: the workspace member
/// directory a finding sits in, or the top-level directory of a tree file.
pub(crate) fn hygiene_member_for_path(vyre_root: &Path, path: &str) -> String {
    let relative = relative_to_vyre(vyre_root, Path::new(path));
    let relative = relative.trim_start_matches('/');
    relative
        .split_once('/')
        .map_or(relative, |(member, _)| member)
        .to_string()
}

pub(crate) fn hygiene_surface_for_path(
    vyre_root: &Path,
    path: &str,
    tests: &TestSources,
) -> &'static str {
    let normalized = path.replace('\\', "/");
    if normalized.contains("/target/")
        || normalized.contains("/target-codex/")
        || normalized.contains("/release/evidence/")
        || normalized.contains("/contract_cases/")
        || normalized.contains("/generated/")
    {
        return "generated";
    }
    let relative = relative_to_vyre(vyre_root, Path::new(path));
    if is_example_source(&relative) {
        return "example";
    }
    if relative.starts_with("vyre-test-support/") || tests.contains(&relative) {
        return "test";
    }
    if is_xtask_source_path(&normalized)
        || normalized.contains("/scripts/")
        || normalized.contains("/.github/")
    {
        return "release_tooling";
    }
    if normalized.ends_with(".md") || normalized.contains("/docs/") {
        return "docs";
    }
    "production"
}

/// Whether a path is example source: under an `examples/` directory, and not
/// under a test, bench or fuzz directory below it.
fn is_example_source(relative: &str) -> bool {
    let mut directories = relative.split('/');
    directories.next_back();
    let mut example = false;
    for directory in directories {
        match directory {
            "examples" => example = true,
            "tests" | "benches" | "fuzz" => return false,
            _ => {}
        }
    }
    example
}

/// Whether a path is xtask source rather than an xtask manifest or README.
pub(crate) fn is_xtask_source_path(normalized: &str) -> bool {
    normalized.contains("/xtask/src/") || xtask_crate_source_segment(normalized)
}

/// Whether `normalized` runs through `xtask-<name>/src/`.
pub(crate) fn xtask_crate_source_segment(normalized: &str) -> bool {
    normalized.split("/xtask-").skip(1).any(|tail| {
        tail.split_once('/')
            .is_some_and(|(_crate_name, rest)| rest == "src" || rest.starts_with("src/"))
    })
}

/// The release risk of one finding.
///
/// `declared` is true only for a source-inspecting test that
/// `docs/testing/STRUCTURAL_GATES.toml` records as asserting a property with no
/// run-time witness. Everything else about a source-inspecting test is
/// unchanged: it is a release blocker, because a test that reads source when it
/// could have run the code is a test that proves nothing about behaviour.
pub(crate) fn hygiene_risk(
    pattern: &str,
    surface: &str,
    hot_path: bool,
    declared: bool,
) -> &'static str {
    if surface == "generated" || surface == "example" {
        return "informational";
    }
    if pattern == "source_inspection_test" {
        return if declared {
            "informational"
        } else {
            "release_blocker"
        };
    }
    if surface == "test" || pattern.starts_with("test_") {
        return "test_hygiene";
    }
    if hot_path {
        return "release_blocker";
    }
    if matches!(
        pattern,
        "todo_macro"
            | "unimplemented_macro"
            | "not_implemented_text"
            | "unbounded_read"
            | "truncating_duration_cast"
            | "unreadable_source_file"
            | "unreadable_tooling_file"
            | "missing_cargo_wrapper"
    ) || is_hidden_fallback_pattern(pattern)
    {
        return "release_blocker";
    }
    if surface == "release_tooling"
        && matches!(
            pattern,
            "raw_workspace_cargo" | "invalid_cargo_full_xtask" | "heredoc"
        )
    {
        return "release_blocker";
    }
    if matches!(pattern, "TODO" | "FIXME" | "placeholder_text" | "stub_text") {
        return "release_debt";
    }
    "informational"
}

pub(crate) fn load_hot_path_files(vyre_root: &Path) -> std::collections::BTreeSet<String> {
    let path = vyre_root.join("docs/optimization/HOT_PATHS.toml");
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(_) => return std::collections::BTreeSet::new(),
    };
    let value = match toml::from_str::<toml::Value>(&text) {
        Ok(value) => value,
        Err(_) => return std::collections::BTreeSet::new(),
    };
    value
        .get("hot_path")
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.get("file").and_then(toml::Value::as_str))
        .map(ToString::to_string)
        .collect()
}

pub(crate) fn hygiene_finding_is_hot_path(
    vyre_root: &Path,
    path: &str,
    hot_paths: &std::collections::BTreeSet<String>,
) -> bool {
    let normalized = path.replace('\\', "/");
    let relative = Path::new(path)
        .strip_prefix(vyre_root)
        .ok()
        .map(|path| path.to_string_lossy().replace('\\', "/"))
        .unwrap_or(normalized);
    hot_paths.contains(&relative)
}
