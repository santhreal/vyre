use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::tree_walk::{self, BUILD_OUTPUT_AND_VCS};

use super::records::{HygieneFinding, BLOCKED_PATTERNS};
use super::rules::{
    has_documented_panic_contract, is_hidden_fallback_guard_source, is_non_release_cfg_attr,
    line_contains_blocked_pattern, line_contains_heredoc, line_contains_invalid_cargo_full_xtask,
    line_contains_raw_workspace_cargo, line_contains_read_call, line_contains_unbounded_read,
    read_text_bounded, truncating_duration_cast_lines, BraceDepthState,
};
use super::syntax::scan_source_inspection_tests;
use super::{CARGO_WRAPPER_PATTERNS, HIDDEN_FALLBACK_PATTERNS, RESOURCE_BOUND_PATTERNS};

pub(crate) const HYGIENE_SCANS: &[(&str, &str, &[&str])] = &[
    (
        "no-stubs-scan.json",
        "no-stubs",
        &[
            "TODO",
            "FIXME",
            "placeholder_text",
            "stub_text",
            "not_implemented_text",
            "todo_macro",
            "unimplemented_macro",
        ],
    ),
    (
        "no-hidden-fallback-scan.json",
        "no-hidden-fallback",
        HIDDEN_FALLBACK_PATTERNS,
    ),
    (
        "resource-bound-scan.json",
        "resource-bound",
        RESOURCE_BOUND_PATTERNS,
    ),
    (
        "error-surface-scan.json",
        "error-surface",
        &[
            "panic_macro",
            "unwrap_call",
            "expect_call",
            "documented_panic_contract",
        ],
    ),
    (
        "cargo-wrapper-scan.json",
        "cargo-wrapper",
        CARGO_WRAPPER_PATTERNS,
    ),
];

/// Whether a path holds test source rather than the production surface.
///
/// One owner for the question: the root walk and the xtask walk both ask it, and
/// a scan that answered it differently would hold one tree to a rule it did not
/// hold the other to.
pub(crate) fn is_test_source_path(path: &Path) -> bool {
    let path = path.display().to_string();
    path.contains("/tests/")
        || path.contains("/benches/")
        || path.contains("/examples/")
        || path.ends_with("/tests.rs")
        || path.ends_with("_test.rs")
        || path.ends_with("_tests.rs")
        || path.contains("_tests_")
        || path.contains("_test_")
}

pub(crate) fn scan_root(
    root: &Path,
    scanned_files: &mut usize,
    findings: &mut Vec<HygieneFinding>,
) {
    for entry in tree_walk::pruned_by(root, |name| {
        !BUILD_OUTPUT_AND_VCS.contains(&name) && name != "release" && !is_xtask_tree_directory(name)
    }) {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                push_walk_error(root, &error, findings);
                continue;
            }
        };
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        if path.file_name().and_then(|name| name.to_str()) == Some("hygiene_matrix.rs") {
            continue;
        }
        if is_test_source_path(path) {
            continue;
        }
        scan_file(path, scanned_files, findings);
    }
}
pub(crate) fn scan_source_inspection_test_files(
    root: &Path,
    scanned_files: &mut usize,
    findings: &mut Vec<HygieneFinding>,
) {
    for entry in tree_walk::pruned_by(root, |name| {
        !BUILD_OUTPUT_AND_VCS.contains(&name) && name != "release"
    }) {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                push_walk_error(root, &error, findings);
                continue;
            }
        };
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let path_string = path.display().to_string();
        let is_test_file = path_string.contains("/tests/")
            || path_string.ends_with("/tests.rs")
            || path_string.ends_with("_test.rs")
            || path_string.ends_with("_tests.rs")
            || path_string.contains("_tests_")
            || path_string.contains("_test_");
        if !is_test_file {
            continue;
        }
        let text = match read_text_bounded(path) {
            Ok(text) => text,
            Err(error) => {
                push_read_error(path, "unreadable_source_file", error, findings);
                continue;
            }
        };
        *scanned_files += 1;
        scan_source_inspection_tests(path, &text, findings);
    }
}

/// Whether a directory name is one of the xtask tooling crates.
pub(crate) fn is_xtask_tree_directory(name: &str) -> bool {
    name == "xtask" || name.starts_with("xtask-")
}

/// The `src` directory of every xtask crate, `xtask` first.
pub(crate) fn xtask_source_roots(root: &Path) -> Vec<PathBuf> {
    let mut roots = vec![root.join("xtask/src")];
    let mut siblings: Vec<PathBuf> = fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("xtask-"))
        })
        .map(|path| path.join("src"))
        .filter(|path| path.is_dir())
        .collect();
    siblings.sort();
    roots.extend(siblings);
    roots
}

/// Hold every xtask source file to the same command hygiene as the tree it gates.
///
/// This read thirteen hand-typed command modules, so a release command added
/// beside them was never scanned, and a renamed module could keep its row here
/// and read as coverage while resolving to nothing. The set is the tree: every
/// xtask crate's non-test source, which cannot fall out of date.
pub(crate) fn scan_release_xtask(
    root: &Path,
    scanned_files: &mut usize,
    findings: &mut Vec<HygieneFinding>,
) {
    for source_root in xtask_source_roots(root) {
        for entry in tree_walk::pruned(&source_root, BUILD_OUTPUT_AND_VCS) {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    push_walk_error(&source_root, &error, findings);
                    continue;
                }
            };
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
                continue;
            }
            if is_test_source_path(path) {
                continue;
            }
            if path.file_name().and_then(|name| name.to_str()) == Some("hygiene_matrix.rs") {
                continue;
            }
            scan_file(path, scanned_files, findings);
        }
    }
}

pub(crate) fn scan_release_tooling(
    root: &Path,
    scanned_files: &mut usize,
    findings: &mut Vec<HygieneFinding>,
) {
    for relative_root in ["scripts", ".github/workflows", ".github/actions"] {
        let tooling_root = root.join(relative_root);
        if !tooling_root.exists() {
            continue;
        }
        for entry in tree_walk::pruned(&tooling_root, BUILD_OUTPUT_AND_VCS) {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    push_walk_error(&tooling_root, &error, findings);
                    continue;
                }
            };
            let path = entry.path();
            let Some(extension) = path.extension().and_then(|ext| ext.to_str()) else {
                continue;
            };
            // Python belongs here as much as shell does. A gate written as a
            // `.py` under `scripts/` is release tooling that runs in CI, and
            // leaving the extension out meant a rule could be evaded by moving
            // the body from a shell heredoc into a file beside it.
            if matches!(extension, "sh" | "yml" | "yaml" | "py") {
                scan_tooling_file(path, scanned_files, findings);
            }
        }
    }
}

/// Hold the release-facing documents to the same command hygiene as the scripts.
///
/// This list named one release runbook three times and a checklist beside it,
/// all deleted with the book, and skipped each one because it is not a file.
/// The gate therefore reported clean while scanning none of the documents its
/// name claims. A listed document that is absent is now a finding: the list is
/// the contract, so a deletion has to be answered here rather than absorbed.
///
/// The list holds authored documents only. `CHANGELOG.md` and the release notes
/// beside it are generated from `release/changes`, and a released entry states
/// what a version did rather than telling a reader what to run, so a bare
/// `cargo` inside one is a record and not an instruction. Scanning them also
/// recorded a line number that every new fragment moved, which turned the
/// evidence artifact red for a document nobody had edited.
pub(crate) fn scan_release_docs(
    vyre_root: &Path,
    scanned_files: &mut usize,
    findings: &mut Vec<HygieneFinding>,
) {
    for doc in [
        "README.md",
        "CONTRIBUTING.md",
        "docs/testing/TESTING.toml",
        "conform/README.md",
        "vyre-bench/README.md",
    ] {
        let path = vyre_root.join(doc);
        if path.is_file() {
            scan_doc_file(&path, scanned_files, findings);
        } else {
            findings.push(HygieneFinding {
                path: doc.to_string(),
                line: 0,
                pattern: "missing_release_doc",
                text: format!(
                    "release document `{doc}` is listed for hygiene scanning and does not exist"
                ),
                test: None,
            });
        }
    }
}

pub(crate) fn scan_release_workflows(
    vyre_root: &Path,
    scanned_files: &mut usize,
    findings: &mut Vec<HygieneFinding>,
) {
    let workflows = vyre_root.join(".github/workflows");
    if !workflows.exists() {
        return;
    }
    for entry in tree_walk::pruned(&workflows, BUILD_OUTPUT_AND_VCS) {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                push_walk_error(&workflows, &error, findings);
                continue;
            }
        };
        let path = entry.path();
        let Some(extension) = path.extension().and_then(|ext| ext.to_str()) else {
            continue;
        };
        if matches!(extension, "yml" | "yaml") {
            scan_tooling_file(path, scanned_files, findings);
        }
    }
}

pub(crate) fn check_required_cargo_wrappers(vyre_root: &Path, findings: &mut Vec<HygieneFinding>) {
    for path in [vyre_root.join("cargo_full")] {
        if !path.is_file() {
            findings.push(HygieneFinding {
                path: path.display().to_string(),
                line: 1,
                pattern: "missing_cargo_wrapper",
                text: "required bounded cargo_full wrapper is missing".to_string(),
                test: None,
            });
        }
    }
}

pub(crate) fn scan_release_controls(
    vyre_root: &Path,
    scanned_files: &mut usize,
    findings: &mut Vec<HygieneFinding>,
) {
    let required_status_doc = vyre_root.join(".github/CI_REQUIRED.md");
    if required_status_doc.is_file() {
        scan_doc_file(&required_status_doc, scanned_files, findings);
    }
    for control in [
        "scripts/apply-branch-protection.sh",
        "xtask/src/gates/layering.rs",
    ] {
        let path = vyre_root.join(control);
        if path.is_file() {
            scan_tooling_file(&path, scanned_files, findings);
        }
    }
}

pub(crate) fn scan_file(
    path: &Path,
    scanned_files: &mut usize,
    findings: &mut Vec<HygieneFinding>,
) {
    let text = match read_text_bounded(path) {
        Ok(text) => text,
        Err(error) => {
            push_read_error(path, "unreadable_source_file", error, findings);
            return;
        }
    };
    *scanned_files += 1;
    for line in truncating_duration_cast_lines(path, &text) {
        findings.push(HygieneFinding {
            path: path.display().to_string(),
            line,
            pattern: "truncating_duration_cast",
            text: text
                .lines()
                .nth(line - 1)
                .unwrap_or_default()
                .trim()
                .to_string(),
            test: None,
        });
    }
    let mut pending_cfg_test = false;
    let mut pending_test_attr = false;
    let mut test_module_braces = BraceDepthState::default();
    let mut skipping_cfg_test_item = false;
    let mut cfg_test_item_braces = BraceDepthState::default();
    let mut pending_bounded_read_chain = false;
    for (line_index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        let bounded_read_chain = pending_bounded_read_chain || trimmed.contains(".take(");
        if trimmed.contains(".take(") && !line_contains_read_call(line) {
            pending_bounded_read_chain = true;
        }
        if skipping_cfg_test_item {
            if cfg_test_item_braces.depth == 0 {
                if trimmed.contains('{') {
                    cfg_test_item_braces = BraceDepthState::default();
                    cfg_test_item_braces.update(line);
                    if cfg_test_item_braces.depth == 0 {
                        skipping_cfg_test_item = false;
                    }
                } else if trimmed.ends_with(';') {
                    skipping_cfg_test_item = false;
                }
            } else {
                cfg_test_item_braces.update(line);
                if cfg_test_item_braces.depth == 0 {
                    skipping_cfg_test_item = false;
                }
            }
            continue;
        }
        if test_module_braces.depth > 0 {
            test_module_braces.update(line);
            continue;
        }
        if pending_cfg_test {
            if trimmed.contains('{') {
                test_module_braces = BraceDepthState::default();
                test_module_braces.update(line);
            } else {
                skipping_cfg_test_item = true;
                cfg_test_item_braces = BraceDepthState::default();
            }
            pending_cfg_test = false;
            continue;
        }
        if pending_test_attr && trimmed.starts_with("fn ") && trimmed.contains('{') {
            test_module_braces = BraceDepthState::default();
            test_module_braces.update(line);
            pending_test_attr = false;
            continue;
        }
        if pending_test_attr && trimmed.starts_with("#[") {
            continue;
        }
        pending_cfg_test = is_non_release_cfg_attr(trimmed);
        pending_test_attr = trimmed == "#[test]"
            || trimmed.starts_with("#[tokio::test")
            || trimmed.starts_with("#[should_panic");
        let lower = line.to_ascii_lowercase();
        if line_contains_raw_workspace_cargo(line) {
            findings.push(HygieneFinding {
                path: path.display().to_string(),
                line: line_index + 1,
                pattern: "raw_workspace_cargo",
                text: line.trim().to_string(),
                test: None,
            });
        }
        if line_contains_invalid_cargo_full_xtask(line) {
            findings.push(HygieneFinding {
                path: path.display().to_string(),
                line: line_index + 1,
                pattern: "invalid_cargo_full_xtask",
                text: line.trim().to_string(),
                test: None,
            });
        }
        for &(name, pattern) in BLOCKED_PATTERNS {
            if line_contains_blocked_pattern(path, name, pattern, line, &lower) {
                let name = if matches!(name, "panic_macro" | "unwrap_call" | "expect_call")
                    && has_documented_panic_contract(&text, line_index)
                {
                    "documented_panic_contract"
                } else {
                    name
                };
                findings.push(HygieneFinding {
                    path: path.display().to_string(),
                    line: line_index + 1,
                    pattern: name,
                    text: line.trim().to_string(),
                    test: None,
                });
            }
        }
        if line_contains_unbounded_read(path, line) && !bounded_read_chain {
            findings.push(HygieneFinding {
                path: path.display().to_string(),
                line: line_index + 1,
                pattern: "unbounded_read",
                text: line.trim().to_string(),
                test: None,
            });
        }
        if bounded_read_chain && line_contains_read_call(line) {
            pending_bounded_read_chain = false;
        } else if pending_bounded_read_chain && trimmed.ends_with(';') {
            pending_bounded_read_chain = false;
        }
        if (line.contains("GpuUnavailable")
            || lower.contains("gpu unavailable")
            || lower.contains("gpu not available")
            || lower.contains("no gpu available"))
            && (lower.contains("skip") || lower.contains("fallback") || lower.contains("fall back"))
            && !is_hidden_fallback_guard_source(path)
        {
            findings.push(HygieneFinding {
                path: path.display().to_string(),
                line: line_index + 1,
                pattern: "gpu_unavailable_skip",
                text: line.trim().to_string(),
                test: None,
            });
        }
    }
    scan_source_inspection_tests(path, &text, findings);
}

pub(crate) fn scan_tooling_file(
    path: &Path,
    scanned_files: &mut usize,
    findings: &mut Vec<HygieneFinding>,
) {
    scan_command_file(path, "unreadable_tooling_file", scanned_files, findings);
}

pub(crate) fn scan_doc_file(
    path: &Path,
    scanned_files: &mut usize,
    findings: &mut Vec<HygieneFinding>,
) {
    scan_command_file(path, "unreadable_doc_file", scanned_files, findings);
}

pub(crate) fn scan_command_file(
    path: &Path,
    read_error_pattern: &'static str,
    scanned_files: &mut usize,
    findings: &mut Vec<HygieneFinding>,
) {
    let text = match read_text_bounded(path) {
        Ok(text) => text,
        Err(error) => {
            push_read_error(path, read_error_pattern, error, findings);
            return;
        }
    };
    *scanned_files += 1;
    for (line_index, line) in text.lines().enumerate() {
        for (matches, pattern) in [
            (
                line_contains_raw_workspace_cargo(line),
                "raw_workspace_cargo",
            ),
            (
                line_contains_invalid_cargo_full_xtask(line),
                "invalid_cargo_full_xtask",
            ),
            (line_contains_heredoc(line), "heredoc"),
        ] {
            if matches {
                findings.push(HygieneFinding {
                    path: path.display().to_string(),
                    line: line_index + 1,
                    pattern,
                    text: line.trim().to_string(),
                    test: None,
                });
            }
        }
    }
}

pub(crate) fn push_walk_error(
    root: &Path,
    error: &walkdir::Error,
    findings: &mut Vec<HygieneFinding>,
) {
    findings.push(HygieneFinding {
        path: error
            .path()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| root.display().to_string()),
        line: 1,
        pattern: "unreadable_scan_entry",
        text: format!("failed to walk release hygiene root: {error}"),
        test: None,
    });
}

pub(crate) fn push_read_error(
    path: &Path,
    pattern: &'static str,
    error: io::Error,
    findings: &mut Vec<HygieneFinding>,
) {
    findings.push(HygieneFinding {
        path: path.display().to_string(),
        line: 1,
        pattern,
        text: format!("failed to read release hygiene input: {error}"),
        test: None,
    });
}
