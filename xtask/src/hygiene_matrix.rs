//! Source hygiene release evidence for Vyre and Weir.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use serde::Serialize;
use walkdir::WalkDir;

#[derive(Debug, Serialize)]
struct HygieneMatrix {
    schema_version: u32,
    scanned_roots: Vec<String>,
    scanned_files: usize,
    release_surface_coverage: ReleaseSurfaceCoverage,
    finding_summary: Vec<HygieneFindingSummary>,
    findings: Vec<HygieneFinding>,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ReleaseSurfaceCoverage {
    vyre_workspace: bool,
    cuda_driver_crate: bool,
    wgpu_driver_crate: bool,
    weir_crate: bool,
    vyrec_tool: bool,
    surgec_tool: bool,
    surgec_grammar_gen: bool,
    release_scripts: bool,
    github_workflows: bool,
    branch_protection_controls: bool,
    resource_bound_patterns: Vec<&'static str>,
    hidden_fallback_patterns: Vec<&'static str>,
    release_tooling_patterns: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
struct HygieneFinding {
    path: String,
    line: usize,
    pattern: &'static str,
    text: String,
}

#[derive(Debug, Clone, Serialize)]
struct HygieneFindingSummary {
    pattern: String,
    count: usize,
}

#[derive(Debug, Serialize)]
struct HygieneScan {
    schema_version: u32,
    scan: String,
    findings: Vec<HygieneFinding>,
    blockers: Vec<String>,
}

const BLOCKED_PATTERNS: &[(&str, &str)] = &[
    ("TODO", "TODO"),
    ("FIXME", "FIXME"),
    ("placeholder_text", "placeholder"),
    ("stub_text", "stub"),
    ("not_implemented_text", "not implemented"),
    ("todo_macro", "todo!("),
    ("unimplemented_macro", "unimplemented!("),
    ("panic_macro", "panic!("),
    ("unwrap_call", ".unwrap("),
    ("expect_call", concat!(".", "expect", "(")),
    ("std_thread_sleep", "std::thread::sleep"),
    ("thread_sleep", "thread::sleep"),
    ("tokio_sleep", "tokio::time::sleep"),
    ("silent_gpu_skip", "skip: no gpu"),
    ("silent_gpu_skipped", "skipped: no gpu"),
    ("cfg_not_gpu", "cfg(not(feature = \"gpu\"))"),
    ("cpu_fallback", "cpu fallback"),
    ("software_fallback", "software fallback"),
    ("fallback_dispatch", "fallback dispatch"),
    ("falling_back_to_cpu", "falling back to cpu"),
    ("fallback_to_cpu", "fallback to cpu"),
    ("synthetic_gpu_timing", "synthetic gpu timing"),
    ("fake_gpu_timing_formula", "cpu_ms * 0.01"),
];

const MAX_HYGIENE_SCAN_FILE_BYTES: u64 = 4_194_304;

pub(crate) fn run(args: &[String]) {
    let output = match parse_output(args) {
        Ok(output) => output,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let vyre_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let santh_root = vyre_root
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| vyre_root.clone());
    let roots = [vyre_root, santh_root.join("libs/dataflow/weir")];
    let optional_roots = [
        santh_root.join("tools/vyrec"),
        santh_root.join("libs/tools/surgec"),
        santh_root.join("libs/shared/surgec-grammar-gen"),
    ];
    let mut scanned_roots = roots
        .iter()
        .map(|root| root.display().to_string())
        .collect::<Vec<_>>();
    scanned_roots.extend(
        optional_roots
            .iter()
            .filter(|root| root.exists())
            .map(|root| root.display().to_string()),
    );
    let mut scanned_files = 0usize;
    let mut findings = Vec::new();
    for root in &roots {
        scan_root(root, &mut scanned_files, &mut findings);
        scan_test_root(root, &mut scanned_files, &mut findings);
    }
    for root in &optional_roots {
        scan_optional_root(&root, &mut scanned_files, &mut findings);
        scan_optional_test_root(&root, &mut scanned_files, &mut findings);
    }
    scan_release_xtask(&roots[0], &mut scanned_files, &mut findings);
    scan_release_tooling(&roots[0], &mut scanned_files, &mut findings);
    scan_release_docs(&roots[0], &santh_root, &mut scanned_files, &mut findings);
    scan_santh_workflows(&santh_root, &mut scanned_files, &mut findings);
    scan_santh_release_controls(&santh_root, &mut scanned_files, &mut findings);
    for root in [
        roots[0].clone(),
        santh_root.join("libs/dataflow/weir"),
        santh_root.join("tools/vyrec"),
        santh_root.join("libs/tools/surgec"),
        santh_root.join("libs/shared/surgec-grammar-gen"),
    ] {
        scan_audit_report_locations(&root, &mut scanned_files, &mut findings);
    }
    check_required_cargo_wrappers(&roots[0], &santh_root, &mut findings);
    let release_surface_coverage = release_surface_coverage(&roots[0], &santh_root);
    let blockers = if findings.is_empty() {
        Vec::new()
    } else {
        vec![format!(
            "{} source hygiene finding(s) remain",
            findings.len()
        )]
    };
    let finding_summary = finding_summary(&findings);
    let matrix = HygieneMatrix {
        schema_version: 1,
        scanned_roots,
        scanned_files,
        release_surface_coverage,
        finding_summary,
        findings,
        blockers,
    };
    let json = match serde_json::to_string_pretty(&matrix) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("Fix: failed to serialize hygiene matrix: {error}");
            std::process::exit(1);
        }
    };
    if let Some(parent) = output.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            eprintln!("Fix: failed to create `{}`: {error}", parent.display());
            std::process::exit(1);
        }
    }
    if let Err(error) = fs::write(&output, format!("{json}\n")) {
        eprintln!("Fix: failed to write `{}`: {error}", output.display());
        std::process::exit(1);
    }
    write_sibling_artifacts(&output, &matrix);
    println!("hygiene-matrix: wrote {}", output.display());
    if !matrix.blockers.is_empty() {
        std::process::exit(1);
    }
}

fn finding_summary(findings: &[HygieneFinding]) -> Vec<HygieneFindingSummary> {
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    for finding in findings {
        *counts.entry(finding.pattern.to_string()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(|(pattern, count)| HygieneFindingSummary { pattern, count })
        .collect()
}

fn release_surface_coverage(vyre_root: &Path, santh_root: &Path) -> ReleaseSurfaceCoverage {
    ReleaseSurfaceCoverage {
        vyre_workspace: vyre_root.join("vyre-core").is_dir(),
        cuda_driver_crate: vyre_root.join("vyre-driver-cuda/src/lib.rs").is_file(),
        wgpu_driver_crate: vyre_root.join("vyre-driver-wgpu/src/lib.rs").is_file(),
        weir_crate: santh_root.join("libs/dataflow/weir/src/lib.rs").is_file(),
        vyrec_tool: santh_root.join("tools/vyrec/src").is_dir(),
        surgec_tool: santh_root.join("libs/tools/surgec/src").is_dir(),
        surgec_grammar_gen: santh_root
            .join("libs/shared/surgec-grammar-gen/src")
            .is_dir(),
        release_scripts: santh_root
            .join("scripts/apply-branch-protection.sh")
            .is_file()
            && santh_root
                .join("scripts/architectural_invariants.sh")
                .is_file(),
        github_workflows: santh_root.join(".github/workflows").is_dir(),
        branch_protection_controls: santh_root.join(".github/CI_REQUIRED.md").is_file()
            && santh_root
                .join("scripts/apply-branch-protection.sh")
                .is_file(),
        resource_bound_patterns: vec![
            "std_thread_sleep",
            "thread_sleep",
            "tokio_sleep",
            "unbounded_read",
        ],
        hidden_fallback_patterns: vec![
            "silent_gpu_skip",
            "silent_gpu_skipped",
            "gpu_unavailable_skip",
            "cfg_not_gpu",
            "cpu_fallback",
            "software_fallback",
            "fallback_dispatch",
            "falling_back_to_cpu",
            "fallback_to_cpu",
            "synthetic_gpu_timing",
            "fake_gpu_timing_formula",
        ],
        release_tooling_patterns: vec![
            "raw_workspace_cargo",
            "invalid_cargo_full_xtask",
            "heredoc",
            "missing_cargo_wrapper",
        ],
    }
}

fn write_sibling_artifacts(output: &Path, matrix: &HygieneMatrix) {
    let Some(parent) = output.parent() else {
        eprintln!(
            "Fix: hygiene matrix output `{}` has no parent directory.",
            output.display()
        );
        std::process::exit(1);
    };
    for &(artifact, scan, patterns) in HYGIENE_SCANS {
        let findings = matrix
            .findings
            .iter()
            .filter(|finding| patterns.iter().any(|pattern| pattern == &finding.pattern))
            .cloned()
            .collect::<Vec<_>>();
        let blockers = if findings.is_empty() {
            Vec::new()
        } else {
            vec![format!("{} `{scan}` finding(s) remain", findings.len())]
        };
        write_json(
            &parent.join(artifact),
            &HygieneScan {
                schema_version: 1,
                scan: scan.to_string(),
                findings,
                blockers,
            },
        );
    }
}

const HYGIENE_SCANS: &[(&str, &str, &[&str])] = &[
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
        &[
            "silent_gpu_skip",
            "silent_gpu_skipped",
            "gpu_unavailable_skip",
            "cfg_not_gpu",
            "cpu_fallback",
            "software_fallback",
            "fallback_dispatch",
            "falling_back_to_cpu",
            "fallback_to_cpu",
            "synthetic_gpu_timing",
            "fake_gpu_timing_formula",
        ],
    ),
    (
        "resource-bound-scan.json",
        "resource-bound",
        &[
            "std_thread_sleep",
            "thread_sleep",
            "tokio_sleep",
            "unbounded_read",
        ],
    ),
    (
        "error-surface-scan.json",
        "error-surface",
        &["panic_macro", "unwrap_call", "expect_call"],
    ),
    (
        "cargo-wrapper-scan.json",
        "cargo-wrapper",
        &[
            "raw_workspace_cargo",
            "invalid_cargo_full_xtask",
            "heredoc",
            "missing_cargo_wrapper",
        ],
    ),
    (
        "audit-location-scan.json",
        "audit-location",
        &["stray_audit_report"],
    ),
    (
        "public-doc-scan.json",
        "public-docs",
        &["undocumented_public_api"],
    ),
    (
        "test-hygiene-scan.json",
        "test-hygiene",
        &[
            "test_TODO",
            "test_FIXME",
            "test_todo_macro",
            "test_unimplemented_macro",
            "test_ignored",
            "test_let_underscore_result",
            "test_assert_true",
        ],
    ),
];

fn write_json(path: &Path, value: &impl Serialize) {
    let json = match serde_json::to_string_pretty(value) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("Fix: failed to serialize `{}`: {error}", path.display());
            std::process::exit(1);
        }
    };
    if let Err(error) = fs::write(path, format!("{json}\n")) {
        eprintln!("Fix: failed to write `{}`: {error}", path.display());
        std::process::exit(1);
    }
}

fn scan_root(root: &Path, scanned_files: &mut usize, findings: &mut Vec<HygieneFinding>) {
    for entry in WalkDir::new(root).into_iter().filter_entry(|entry| {
        let name = entry.file_name().to_string_lossy();
        !matches!(
            name.as_ref(),
            "target"
                | "target-codex"
                | "target_tests"
                | ".git"
                | ".cargo-target"
                | "release"
                | "xtask"
        )
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
        let path_string = path.display().to_string();
        if path_string.contains("/tests/")
            || path_string.contains("/benches/")
            || path_string.contains("/examples/")
            || path_string.ends_with("/tests.rs")
            || path_string.ends_with("_test.rs")
            || path_string.ends_with("_tests.rs")
            || path_string.contains("_tests_")
            || path_string.contains("_test_")
        {
            continue;
        }
        scan_file(path, scanned_files, findings);
    }
}

fn scan_optional_root(root: &Path, scanned_files: &mut usize, findings: &mut Vec<HygieneFinding>) {
    if root.exists() {
        scan_root(root, scanned_files, findings);
    }
}

fn scan_test_root(root: &Path, scanned_files: &mut usize, findings: &mut Vec<HygieneFinding>) {
    for entry in WalkDir::new(root).into_iter().filter_entry(|entry| {
        let name = entry.file_name().to_string_lossy();
        !matches!(
            name.as_ref(),
            "target"
                | "target-codex"
                | "target_tests"
                | ".git"
                | ".cargo-target"
                | "release"
                | "xtask"
        )
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
        if is_test_file {
            scan_test_file(path, scanned_files, findings);
        }
    }
}

fn scan_optional_test_root(
    root: &Path,
    scanned_files: &mut usize,
    findings: &mut Vec<HygieneFinding>,
) {
    if root.exists() {
        scan_test_root(root, scanned_files, findings);
    }
}

fn scan_release_xtask(root: &Path, scanned_files: &mut usize, findings: &mut Vec<HygieneFinding>) {
    let xtask = root.join("xtask/src");
    for relative in [
        "backend_matrix.rs",
        "c_parser_corpus.rs",
        "conformance_matrix.rs",
        "docs_matrix.rs",
        "feature_matrix.rs",
        "hygiene_matrix.rs",
        "metadata_matrix.rs",
        "optimization_corpus.rs",
        "optimization_matrix.rs",
        "parser_coherence.rs",
        "release_benchmarks.rs",
        "release_completion_audit.rs",
        "release_conformance.rs",
        "release_evidence.rs",
        "release_gate.rs",
        "test_matrix.rs",
        "version_matrix.rs",
        "vyre_weir_release_gate.rs",
        "weir_matrix.rs",
    ] {
        scan_file(&xtask.join(relative), scanned_files, findings);
    }
}

fn scan_release_tooling(
    root: &Path,
    scanned_files: &mut usize,
    findings: &mut Vec<HygieneFinding>,
) {
    for relative_root in ["scripts", ".github/workflows", ".github/actions"] {
        let tooling_root = root.join(relative_root);
        if !tooling_root.exists() {
            continue;
        }
        for entry in WalkDir::new(&tooling_root)
            .into_iter()
            .filter_entry(|entry| {
                let name = entry.file_name().to_string_lossy();
                !matches!(name.as_ref(), "target" | ".git")
            })
        {
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
            if matches!(extension, "sh" | "yml" | "yaml") {
                scan_tooling_file(path, scanned_files, findings);
            }
        }
    }
}

fn scan_release_docs(
    vyre_root: &Path,
    santh_root: &Path,
    scanned_files: &mut usize,
    findings: &mut Vec<HygieneFinding>,
) {
    for doc in [
        santh_root.join("docs/vyre-weir-release-plan.md"),
        vyre_root.join("README.md"),
        vyre_root.join("docs/RELEASE.md"),
        vyre_root.join("docs/RELEASE_ENGINEERING.md"),
        vyre_root.join("docs/RELEASE_CHECKLIST.md"),
        vyre_root.join("docs/PUBLISH_GATE.md"),
        vyre_root.join("docs/TESTING_PROGRAM.md"),
        vyre_root.join("docs/optimization/AGENT_CONTRACT.md"),
        vyre_root.join("conform/README.md"),
        vyre_root.join("vyre-bench/README.md"),
        vyre_root.join("vyre-frontend-c/README.md"),
        santh_root.join("tools/vyrec/README.md"),
        santh_root.join("libs/dataflow/weir/README.md"),
        santh_root.join("libs/dataflow/weir/VISION.md"),
    ] {
        if doc.is_file() {
            scan_doc_file(&doc, scanned_files, findings);
        }
    }
}

fn scan_santh_workflows(
    santh_root: &Path,
    scanned_files: &mut usize,
    findings: &mut Vec<HygieneFinding>,
) {
    let workflows = santh_root.join(".github/workflows");
    if !workflows.exists() {
        return;
    }
    for entry in WalkDir::new(&workflows).into_iter().filter_entry(|entry| {
        let name = entry.file_name().to_string_lossy();
        !matches!(name.as_ref(), "target" | ".git")
    }) {
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

fn scan_audit_report_locations(
    root: &Path,
    scanned_files: &mut usize,
    findings: &mut Vec<HygieneFinding>,
) {
    if !root.exists() {
        return;
    }
    for entry in WalkDir::new(root).into_iter().filter_entry(|entry| {
        let name = entry.file_name().to_string_lossy();
        !matches!(
            name.as_ref(),
            "target" | "target-codex" | "target_tests" | ".git" | ".cargo-target" | "release"
        )
    }) {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                push_walk_error(root, &error, findings);
                continue;
            }
        };
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !(file_name.starts_with("AUDIT")
            || file_name.starts_with("FINDINGS")
            || file_name.starts_with("PLAN"))
        {
            continue;
        }
        *scanned_files += 1;
        let normalized = path.to_string_lossy();
        if !normalized.contains("/.audits/") && !normalized.contains("/audits/") {
            findings.push(HygieneFinding {
                path: path.display().to_string(),
                line: 1,
                pattern: "stray_audit_report",
                text: "audit, findings, and plan reports must live under .audits/".to_string(),
            });
        }
    }
}

fn check_required_cargo_wrappers(
    vyre_root: &Path,
    santh_root: &Path,
    findings: &mut Vec<HygieneFinding>,
) {
    for path in [santh_root.join("cargo_full"), vyre_root.join("cargo_full")] {
        if !path.is_file() {
            findings.push(HygieneFinding {
                path: path.display().to_string(),
                line: 1,
                pattern: "missing_cargo_wrapper",
                text: "required bounded cargo_full wrapper is missing".to_string(),
            });
        }
    }
}

fn scan_santh_release_controls(
    santh_root: &Path,
    scanned_files: &mut usize,
    findings: &mut Vec<HygieneFinding>,
) {
    let required_status_doc = santh_root.join(".github/CI_REQUIRED.md");
    if required_status_doc.is_file() {
        scan_doc_file(&required_status_doc, scanned_files, findings);
    }
    for script in [
        "scripts/apply-branch-protection.sh",
        "scripts/architectural_invariants.sh",
    ] {
        let path = santh_root.join(script);
        if path.is_file() {
            scan_tooling_file(&path, scanned_files, findings);
        }
    }
}

fn scan_file(path: &Path, scanned_files: &mut usize, findings: &mut Vec<HygieneFinding>) {
    let text = match read_text_bounded(path) {
        Ok(text) => text,
        Err(error) => {
            push_read_error(path, "unreadable_source_file", error, findings);
            return;
        }
    };
    *scanned_files += 1;
    let mut pending_cfg_test = false;
    let mut pending_test_attr = false;
    let mut test_module_depth = 0usize;
    let mut pending_bounded_read_chain = false;
    for (line_index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        let bounded_read_chain = pending_bounded_read_chain || trimmed.contains(".take(");
        if trimmed.contains(".take(") && !line_contains_read_call(line) {
            pending_bounded_read_chain = true;
        }
        if test_module_depth > 0 {
            test_module_depth = update_brace_depth(test_module_depth, line);
            continue;
        }
        if pending_cfg_test && trimmed.starts_with("mod ") && trimmed.contains('{') {
            test_module_depth = update_brace_depth(0, line);
            pending_cfg_test = false;
            continue;
        }
        if pending_test_attr && trimmed.starts_with("fn ") && trimmed.contains('{') {
            test_module_depth = update_brace_depth(0, line);
            pending_test_attr = false;
            continue;
        }
        pending_cfg_test = trimmed == "#[cfg(test)]";
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
            });
        }
        if line_contains_invalid_cargo_full_xtask(line) {
            findings.push(HygieneFinding {
                path: path.display().to_string(),
                line: line_index + 1,
                pattern: "invalid_cargo_full_xtask",
                text: line.trim().to_string(),
            });
        }
        for &(name, pattern) in BLOCKED_PATTERNS {
            if line_contains_blocked_pattern(path, name, pattern, line, &lower) {
                findings.push(HygieneFinding {
                    path: path.display().to_string(),
                    line: line_index + 1,
                    pattern: name,
                    text: line.trim().to_string(),
                });
            }
        }
        if line_contains_unbounded_read(path, line) && !bounded_read_chain {
            findings.push(HygieneFinding {
                path: path.display().to_string(),
                line: line_index + 1,
                pattern: "unbounded_read",
                text: line.trim().to_string(),
            });
        }
        if bounded_read_chain && line_contains_read_call(line) {
            pending_bounded_read_chain = false;
        } else if pending_bounded_read_chain && trimmed.ends_with(';') {
            pending_bounded_read_chain = false;
        }
        if is_undocumented_public_api(&text, line_index) {
            findings.push(HygieneFinding {
                path: path.display().to_string(),
                line: line_index + 1,
                pattern: "undocumented_public_api",
                text: line.trim().to_string(),
            });
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
            });
        }
    }
}

fn line_contains_read_call(line: &str) -> bool {
    line.contains("fs::read_to_string(")
        || line.contains("std::fs::read_to_string(")
        || line.contains("fs::read(")
        || line.contains("std::fs::read(")
        || line.contains(".read_to_end(")
        || line.contains(".read_to_string(")
}

fn line_contains_unbounded_read(path: &Path, line: &str) -> bool {
    let normalized = path.to_string_lossy();
    if normalized.contains("/xtask/src/") {
        return false;
    }
    let trimmed = line.trim_start();
    if trimmed.starts_with("//") || is_release_rule_text(trimmed) {
        return false;
    }
    if trimmed.contains(".take(") {
        return false;
    }
    line_contains_read_call(trimmed)
}

fn is_undocumented_public_api(text: &str, line_index: usize) -> bool {
    let lines = text.lines().collect::<Vec<_>>();
    let Some(line) = lines.get(line_index) else {
        return false;
    };
    let trimmed = line.trim_start();
    if !(trimmed.starts_with("pub struct ")
        || trimmed.starts_with("pub enum ")
        || trimmed.starts_with("pub trait ")
        || trimmed.starts_with("pub type "))
    {
        return false;
    }
    let mut cursor = line_index;
    while cursor > 0 {
        cursor -= 1;
        let previous = lines[cursor].trim();
        if previous.is_empty() || previous.starts_with("#[") {
            continue;
        }
        return !(previous.starts_with("///") || previous.starts_with("//!"));
    }
    true
}

fn scan_tooling_file(path: &Path, scanned_files: &mut usize, findings: &mut Vec<HygieneFinding>) {
    let text = match read_text_bounded(path) {
        Ok(text) => text,
        Err(error) => {
            push_read_error(path, "unreadable_tooling_file", error, findings);
            return;
        }
    };
    *scanned_files += 1;
    for (line_index, line) in text.lines().enumerate() {
        if line_contains_raw_workspace_cargo(line) {
            findings.push(HygieneFinding {
                path: path.display().to_string(),
                line: line_index + 1,
                pattern: "raw_workspace_cargo",
                text: line.trim().to_string(),
            });
        }
        if line_contains_invalid_cargo_full_xtask(line) {
            findings.push(HygieneFinding {
                path: path.display().to_string(),
                line: line_index + 1,
                pattern: "invalid_cargo_full_xtask",
                text: line.trim().to_string(),
            });
        }
        if line_contains_heredoc(line) {
            findings.push(HygieneFinding {
                path: path.display().to_string(),
                line: line_index + 1,
                pattern: "heredoc",
                text: line.trim().to_string(),
            });
        }
    }
}

fn scan_doc_file(path: &Path, scanned_files: &mut usize, findings: &mut Vec<HygieneFinding>) {
    let text = match read_text_bounded(path) {
        Ok(text) => text,
        Err(error) => {
            push_read_error(path, "unreadable_doc_file", error, findings);
            return;
        }
    };
    *scanned_files += 1;
    for (line_index, line) in text.lines().enumerate() {
        if line_contains_raw_workspace_cargo(line) {
            findings.push(HygieneFinding {
                path: path.display().to_string(),
                line: line_index + 1,
                pattern: "raw_workspace_cargo",
                text: line.trim().to_string(),
            });
        }
        if line_contains_invalid_cargo_full_xtask(line) {
            findings.push(HygieneFinding {
                path: path.display().to_string(),
                line: line_index + 1,
                pattern: "invalid_cargo_full_xtask",
                text: line.trim().to_string(),
            });
        }
        if line_contains_heredoc(line) {
            findings.push(HygieneFinding {
                path: path.display().to_string(),
                line: line_index + 1,
                pattern: "heredoc",
                text: line.trim().to_string(),
            });
        }
    }
}

fn scan_test_file(path: &Path, scanned_files: &mut usize, findings: &mut Vec<HygieneFinding>) {
    let text = match read_text_bounded(path) {
        Ok(text) => text,
        Err(error) => {
            push_read_error(path, "unreadable_test_file", error, findings);
            return;
        }
    };
    *scanned_files += 1;
    for (line_index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if is_release_rule_text(trimmed) {
            continue;
        }
        if line.contains("TODO") {
            push_test_finding(path, line_index, "test_TODO", trimmed, findings);
        }
        if line.contains("FIXME") {
            push_test_finding(path, line_index, "test_FIXME", trimmed, findings);
        }
        if line.contains("todo!(") {
            push_test_finding(path, line_index, "test_todo_macro", trimmed, findings);
        }
        if line.contains("unimplemented!(") {
            push_test_finding(
                path,
                line_index,
                "test_unimplemented_macro",
                trimmed,
                findings,
            );
        }
        if trimmed == "#[ignore]"
            || trimmed.starts_with("#[ignore(")
            || trimmed.starts_with("#[ignore =")
        {
            push_test_finding(path, line_index, "test_ignored", trimmed, findings);
        }
        if trimmed.starts_with("let _ =") {
            push_test_finding(
                path,
                line_index,
                "test_let_underscore_result",
                trimmed,
                findings,
            );
        }
        if matches!(
            trimmed,
            "assert!(true);"
                | "assert_eq!(true, true);"
                | "assert_eq!(1, 1);"
                | "assert_ne!(1, 2);"
        ) {
            push_test_finding(path, line_index, "test_assert_true", trimmed, findings);
        }
    }
}

fn push_test_finding(
    path: &Path,
    line_index: usize,
    pattern: &'static str,
    text: &str,
    findings: &mut Vec<HygieneFinding>,
) {
    findings.push(HygieneFinding {
        path: path.display().to_string(),
        line: line_index + 1,
        pattern,
        text: text.to_string(),
    });
}

fn push_walk_error(root: &Path, error: &walkdir::Error, findings: &mut Vec<HygieneFinding>) {
    findings.push(HygieneFinding {
        path: error
            .path()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| root.display().to_string()),
        line: 1,
        pattern: "unreadable_scan_entry",
        text: format!("failed to walk release hygiene root: {error}"),
    });
}

fn push_read_error(
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
    });
}

fn line_contains_blocked_pattern(
    path: &Path,
    name: &str,
    pattern: &str,
    line: &str,
    lower: &str,
) -> bool {
    let trimmed = line.trim();
    if is_hygiene_rule_source(path) {
        return false;
    }
    if is_hidden_fallback_pattern(name) && is_hidden_fallback_guard_source(path) {
        return false;
    }
    if is_hidden_fallback_pattern(name) && is_negated_hidden_fallback_statement(lower) {
        return false;
    }
    if name == "cfg_not_gpu" && !line_cfg_not_gpu_hides_work(lower) {
        return false;
    }
    if is_release_rule_text(trimmed) {
        return false;
    }
    match name {
        "placeholder_text" => contains_word(lower, pattern),
        "stub_text" => contains_word(lower, pattern),
        "not_implemented_text" => lower.contains(pattern),
        "TODO" | "FIXME" => line.contains(pattern),
        _ => line.contains(pattern) || lower.contains(pattern),
    }
}

fn is_hidden_fallback_pattern(name: &str) -> bool {
    matches!(
        name,
        "silent_gpu_skip"
            | "silent_gpu_skipped"
            | "gpu_unavailable_skip"
            | "cfg_not_gpu"
            | "cpu_fallback"
            | "software_fallback"
            | "fallback_dispatch"
            | "falling_back_to_cpu"
            | "fallback_to_cpu"
            | "synthetic_gpu_timing"
            | "fake_gpu_timing_formula"
    )
}

fn is_negated_hidden_fallback_statement(lower: &str) -> bool {
    lower.contains("no cpu fallback")
        || lower.contains("no hidden fallback")
        || lower.contains("no software fallback")
        || lower.contains("never hides")
        || lower.contains("must not hide")
}

fn line_cfg_not_gpu_hides_work(lower: &str) -> bool {
    lower.contains("fallback")
        || lower.contains("skip")
        || lower.contains("return ok")
        || lower.contains("success")
}

fn line_contains_raw_workspace_cargo(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty()
        || trimmed.starts_with('#')
        || trimmed.starts_with("name:")
        || is_release_rule_text(trimmed)
        || trimmed.starts_with("echo ")
        || trimmed.contains("cargo install")
        || trimmed.contains("cargo_full")
        || trimmed.contains("CARGO_RUNNER")
        || trimmed.contains("./cargo_full")
        || trimmed.contains("VYRE_CARGO_RUNNER")
    {
        return false;
    }
    [
        "cargo build",
        "cargo check",
        "cargo test",
        "cargo clippy",
        "cargo doc",
        "cargo fmt",
        "cargo run",
        "cargo xtask",
        "cargo bench",
        "cargo publish",
        "cargo machete",
        "cargo udeps",
        "cargo fuzz",
        "cargo public-api",
    ]
    .iter()
    .any(|needle| trimmed.contains(needle))
        || trimmed.starts_with("cargo +")
}

fn line_contains_invalid_cargo_full_xtask(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() || is_release_rule_text(trimmed) {
        return false;
    }
    let plain = ["cargo_full", " xtask"].concat();
    let dotted = ["./cargo_full", " xtask"].concat();
    trimmed.contains(&plain) || trimmed.contains(&dotted)
}

fn line_contains_heredoc(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return false;
    }
    trimmed.contains("<<") && !trimmed.contains("<<<")
}

fn is_release_rule_text(trimmed: &str) -> bool {
    trimmed.starts_with('"')
        || trimmed.starts_with("(\"")
        || trimmed.starts_with("&[")
        || trimmed.contains("no-stubs")
        || trimmed.contains("unresolved marker")
        || trimmed.contains("No shipped stubs")
}

fn is_hygiene_rule_source(path: &Path) -> bool {
    let normalized = path.to_string_lossy().replace('\\', "/");
    [
        "scripts/check_repo_split_readiness.sh",
        "scripts/check_dialect_coverage.sh",
        "scripts/check_unsafe_justifications.sh",
        "scripts/check_no_deferred_work.sh",
        "scripts/check_tests_can_fail.sh",
        "scripts/check_primitive_contract.sh",
        "jules_tickets/_generate.py",
        "jules_tickets/test_dump.py",
        "xtask/src/backend_matrix.rs",
        "xtask/src/docs_matrix.rs",
        "xtask/src/feature_matrix.rs",
        "xtask/src/hygiene_matrix.rs",
        "xtask/src/optimization_matrix.rs",
        "xtask/src/release_completion_audit.rs",
        "xtask/src/vyre_weir_release_gate.rs",
        "xtask/src/weir_matrix.rs",
        "xtask/src/whats_similar.rs",
        "xtask/src/parser_coherence.rs",
    ]
    .iter()
    .any(|suffix| normalized.ends_with(suffix))
}

fn is_hidden_fallback_guard_source(path: &Path) -> bool {
    let normalized = path.to_string_lossy().replace('\\', "/");
    [
        "vyre-lints/src/production_cpu_fallbacks.rs",
        "vyre-lints/src/gpu_skip_guards.rs",
        "vyre-lints/src/lib.rs",
        "vyre-lints/src/main.rs",
        "vyre-lints/tests/production_cpu_fallbacks.rs",
        "vyre-lints/tests/gpu_skip_guards.rs",
    ]
    .iter()
    .any(|suffix| normalized.ends_with(suffix))
}

fn contains_word(haystack: &str, needle: &str) -> bool {
    haystack.match_indices(needle).any(|(index, _)| {
        is_word_start(haystack, index) && is_word_end(haystack, index + needle.len())
    })
}

fn is_word_start(text: &str, index: usize) -> bool {
    text.get(..index)
        .and_then(|prefix| prefix.chars().next_back())
        .is_none_or(|ch| !ch.is_ascii_alphanumeric() && ch != '_')
}

fn is_word_end(text: &str, index: usize) -> bool {
    text.get(index..)
        .and_then(|suffix| suffix.chars().next())
        .is_none_or(|ch| !ch.is_ascii_alphanumeric() && ch != '_')
}

fn update_brace_depth(current: usize, line: &str) -> usize {
    let mut depth = current;
    let code = line.split("//").next().unwrap_or(line);
    for ch in code.chars() {
        match ch {
            '{' => depth = depth.saturating_add(1),
            '}' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    depth
}

fn parse_output(args: &[String]) -> Result<PathBuf, String> {
    let mut output = None;
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--output" => {
                let Some(path) = args.get(index + 1) else {
                    return Err("Fix: --output requires a path.".to_string());
                };
                output = Some(PathBuf::from(path));
                index += 2;
            }
            "--help" | "-h" => {
                println!(
                    "USAGE:\n  cargo_full run --bin xtask -- hygiene-matrix [--output PATH]\n\n\
                     Scans Vyre/Weir shipped Rust source for release hygiene blockers."
                );
                std::process::exit(0);
            }
            other => return Err(format!("Fix: unknown hygiene-matrix option `{other}`.")),
        }
    }
    Ok(output.unwrap_or_else(default_output))
}

fn default_output() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|path| path.join("release/evidence/hygiene/hygiene-matrix.json"))
        .unwrap_or_else(|| PathBuf::from("release/evidence/hygiene/hygiene-matrix.json"))
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    let mut reader = fs::File::open(path)?.take(MAX_HYGIENE_SCAN_FILE_BYTES.saturating_add(1));
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    if text.len() as u64 > MAX_HYGIENE_SCAN_FILE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} exceeds {MAX_HYGIENE_SCAN_FILE_BYTES} byte hygiene scan read cap",
                path.display()
            ),
        ));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_fallback_scan_ignores_guard_implementation_text() {
        let guard = Path::new("vyre-lints/src/production_cpu_fallbacks.rs");

        assert!(
            !line_contains_blocked_pattern(
                guard,
                "cpu_fallback",
                "cpu fallback",
                "//! Production CPU fallback guard.",
                "//! production cpu fallback guard.",
            ),
            "Fix: hygiene evidence must not count the guard's own forbidden-token description as shipped fallback behavior."
        );
    }

    #[test]
    fn hidden_fallback_scan_ignores_negated_product_status() {
        let source = Path::new("tools/vyrec/src/lib.rs");

        assert!(
            !line_contains_blocked_pattern(
                source,
                "cpu_fallback",
                "cpu fallback",
                "status: beta compile-evidence driver; no CPU fallback",
                "status: beta compile-evidence driver; no cpu fallback",
            ),
            "Fix: explicit no-fallback product status text must not be reported as hidden fallback behavior."
        );
    }

    #[test]
    fn hidden_fallback_scan_still_flags_positive_product_fallback() {
        let source = Path::new("libs/tools/surgec/src/scan/pipeline/parse_driver.rs");

        assert!(
            line_contains_blocked_pattern(
                source,
                "cpu_fallback",
                "cpu fallback",
                "CpuRayonParseDriver is a temporary CPU fallback.",
                "cpurayonparsedriver is a temporary cpu fallback.",
            ),
            "Fix: real positive fallback claims must remain visible in release hygiene evidence."
        );
    }

    #[test]
    fn cfg_not_gpu_attr_is_not_a_hidden_fallback_by_itself() {
        let source = Path::new("libs/tools/surgec/src/cmd_scan.rs");

        assert!(
            !line_contains_blocked_pattern(
                source,
                "cfg_not_gpu",
                "cfg(not(feature = \"gpu\"))",
                "#[cfg(not(feature = \"gpu\"))]",
                "#[cfg(not(feature = \"gpu\"))]",
            ),
            "Fix: a fail-closed compile-time GPU feature guard must not be treated as a runtime hidden fallback without fallback behavior."
        );
    }

    #[test]
    fn hidden_fallback_guard_source_is_identified_for_gpu_skip_phrases() {
        assert!(is_hidden_fallback_guard_source(Path::new(
            "vyre-lints/src/gpu_skip_guards.rs"
        )));
    }
}
