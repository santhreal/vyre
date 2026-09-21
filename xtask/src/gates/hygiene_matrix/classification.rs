use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

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
    test_gated: &BTreeSet<String>,
) -> Vec<HygieneFindingClass> {
    findings
        .iter()
        .map(|finding| {
            let owner_lane = hygiene_owner_lane_for_path(&finding.path);
            let surface = hygiene_surface_for_path(vyre_root, &finding.path, test_gated);
            let hot_path = hygiene_finding_is_hot_path(vyre_root, &finding.path, hot_paths);
            let declared = is_declared_structural_gate(vyre_root, finding, structural_gates);
            let risk = hygiene_risk(finding.pattern, surface, hot_path, declared);
            HygieneFindingClass {
                path: finding.path.clone(),
                line: finding.line,
                pattern: finding.pattern,
                owner_lane,
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
    let mut counts =
        BTreeMap::<(&'static str, &'static str, &'static str, bool, bool), usize>::new();
    for class in classes {
        *counts
            .entry((
                class.owner_lane,
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
            |((owner_lane, surface, risk, hot_path, release_blocker), count)| {
                HygieneClassificationSummary {
                    owner_lane,
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
    let mut counts = BTreeMap::<
        (
            &'static str,
            &'static str,
            &'static str,
            bool,
            &'static str,
            bool,
        ),
        usize,
    >::new();
    for class in classes {
        *counts
            .entry((
                class.owner_lane,
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
            |((owner_lane, surface, risk, hot_path, pattern, release_blocker), count)| {
                HygieneIntakeSummary {
                    owner_lane,
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

pub(crate) fn hygiene_owner_lane_for_path(path: &str) -> &'static str {
    let normalized = path.replace('\\', "/");
    if normalized.contains("/vyre-libs/src/parsing/")
        || normalized.contains("/vyre-primitives/src/parsing/")
    {
        return "frontend_parsing";
    }
    if normalized.contains("/vyre-foundation/src/optimizer/")
        || normalized.contains("/vyre-foundation/src/transform/")
    {
        return "foundation_optimizer";
    }
    if normalized.contains("/vyre-foundation/src/serial/")
        || normalized.contains("/vyre-foundation/src/ir_inner/")
        || normalized.contains("/vyre-foundation/src/vast/mod.rs")
        || normalized.contains("/vyre-foundation/fuzz/")
        || normalized.contains("/vyre-spec/")
        || normalized.contains("/vyre-libs/src/lib.rs")
        || normalized.contains("/vyre-libs/src/intern/")
        || normalized.contains("/vyre-primitives/src/hash/")
        || normalized.contains("/vyre-primitives/src/wire.rs")
    {
        return "foundation_wire";
    }
    if normalized.contains("/vyre-driver-cuda/") {
        return "driver_cuda";
    }
    if normalized.contains("/vyre-driver-wgpu/") {
        return "driver_wgpu";
    }
    if normalized.contains("/vyre-driver-spirv/") {
        return "driver_spirv";
    }
    if normalized.contains("/vyre-driver-metal/") || normalized.contains("/vyre-emit-metal/") {
        return "driver_metal";
    }
    if normalized.contains("/vyre-driver/") {
        return "driver_shared";
    }
    if normalized.contains("/vyre-foundation/src/runtime/")
        || normalized.contains("/vyre-reference/")
        || normalized.contains("/vyre-primitives/src/hardware/")
    {
        return "driver_shared";
    }
    if normalized.contains("/vyre-lower/")
        || normalized.contains("/vyre-emit-naga/")
        || normalized.contains("/vyre-emit-ptx/")
        || normalized.contains("/vyre-emit-spirv/")
    {
        return "lower_emit";
    }
    if normalized.contains("/vyre-runtime/src/resident_work_queue/") {
        return "runtime_resident_work_queue";
    }
    if normalized.contains("/vyre-libs/src/scheduling/")
        || normalized.contains("/vyre-libs/src/device/")
        || normalized.contains("/vyre-runtime/src/")
    {
        return "runtime_resident_work_queue";
    }
    if normalized.contains("/vyre-bench/") {
        return "bench_harness";
    }
    if normalized.contains("/vyre-libs/src/scan/")
        || normalized.contains("/vyre-libs/src/decode/")
        || normalized.contains("/vyre-libs/src/rule/")
        || normalized.contains("/vyre-libs/src/encoding/")
        || normalized.contains("/vyre-primitives/src/matching/")
        || normalized.contains("/vyre-primitives/src/decode/")
        || normalized.contains("/vyre-primitives/src/nfa/")
    {
        return "scan_static";
    }
    if normalized.contains("/vyre-libs/src/security/")
        || normalized.contains("/vyre-libs/src/dataflow/")
        || normalized.contains("/vyre-libs/src/borrowck/")
        || normalized.contains("/vyre-libs/src/analysis/")
        || normalized.contains("/vyre-libs/src/graph/")
        || normalized.contains("/vyre-primitives/src/graph/")
        || normalized.contains("/vyre-primitives/src/fixpoint/")
        || normalized.contains("/vyre-primitives/src/predicate/")
        || normalized.contains("/vyre-primitives/src/bitset/")
    {
        return "security_dataflow";
    }
    if normalized.contains("/vyre-libs/src/nn/")
        || normalized.contains("/vyre-libs/src/math/")
        || normalized.contains("/vyre-primitives/src/math/")
    {
        return "nn_math";
    }
    if is_xtask_tree_path(&normalized)
        || normalized.contains("/vyre-lints/")
        || normalized.contains("/vyre-libs/src/test_support/")
        || normalized.contains("/conform/")
        || normalized.contains("/release/evidence/")
        || normalized.contains("/docs/")
        || normalized.contains("/.github/")
        || normalized.contains("/scripts/")
    {
        return "testing_evidence";
    }
    "coordination"
}

pub(crate) fn hygiene_surface_for_path(
    vyre_root: &Path,
    path: &str,
    test_gated: &BTreeSet<String>,
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
    if normalized.contains("/vyre-test-support/") || normalized.starts_with("vyre-test-support/") {
        return "test";
    }
    if normalized.contains("/tests/")
        || normalized.contains("/fuzz/")
        || normalized.contains("/test_harness/")
        || normalized.ends_with("/tests.rs")
        || normalized.ends_with("_test.rs")
        || normalized.ends_with("_tests.rs")
        || normalized.contains("_tests_")
        || normalized.contains("_test_")
        || test_gated.contains(&relative_to_vyre(vyre_root, Path::new(path)))
    {
        return "test";
    }
    if normalized.contains("/examples/") {
        return "example";
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

/// Whether a path is inside one of the xtask tooling crates.
///
/// The tooling is split across `xtask` and the `xtask-*` crates that link vyre,
/// and which crate a module ended up in is a dependency-weight decision the
/// hygiene rules have no stake in. Match the family, not one member of it.
pub(crate) fn is_xtask_tree_path(normalized: &str) -> bool {
    normalized.contains("/xtask/") || normalized.contains("/xtask-")
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
