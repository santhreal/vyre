//! CI workflow wiring for the conformance and release gates.
//!
//! Every check here reads workflow and script text, so it links no vyre
//! crate: whether a gate is wired into CI is a fact about the checked-in
//! workflow files, not about the linked backends.

use std::collections::BTreeSet;
use std::path::Path;

use serde::Serialize;

use crate::release::conformance_evidence_semantics::read_conformance_text;
use crate::tree_walk::{self, BUILD_OUTPUT_AND_VCS};

/// One conformance gate as a workflow file declares it.
#[derive(Debug, Clone, Serialize)]
pub struct CiConformanceGate {
    /// Workflow file the gate is declared in.
    pub workflow: String,
    /// Why the workflow could not be read, when it could not.
    pub read_error: Option<String>,
    /// Job name the gate runs under.
    pub gate: String,
    /// Whether the job exists at all.
    pub present: bool,
    /// Whether the job runs the gate command.
    pub command_present: bool,
    /// Whether the job checks the artifact the command writes.
    pub artifact_check_present: bool,
}

/// Read every conformance gate out of the workflows that must declare one.
pub fn inspect_ci_conformance_gates(vyre_root: &Path) -> Vec<CiConformanceGate> {
    vec![
        inspect_ci_gate(
            vyre_root,
            ".github/workflows/gpu-parity.yml",
            "GPU release gate",
            "cargo_full run --release --bin xtask -- release-conformance --backend all",
            "vyre-release-benchmark-evidence",
        ),
        inspect_ci_gate(
            vyre_root,
            ".github/workflows/conform.yml",
            "Conform release gate",
            "cargo_full run --bin xtask -- conformance-matrix",
            "conformance-matrix.json",
        ),
        inspect_ci_gate(
            vyre_root,
            ".github/workflows/ci.yml",
            "CI release gate",
            "cargo_full run --bin xtask -- release-evidence",
            "release/evidence/**/*.json",
        ),
        inspect_ci_gate(
            vyre_root,
            ".github/workflows/architectural-invariants.yml",
            "Architecture release gate",
            "cargo_full run -p xtask -- op-matrix --check",
            "scripts/architecture_docs.py . --check",
        ),
        inspect_ci_gate(
            vyre_root,
            "scripts/apply-branch-protection.sh",
            "required_status_checks",
            ".github/CI_REQUIRED.md",
            "gh \"${args[@]}\"",
        ),
    ]
}

fn inspect_ci_gate(
    vyre_root: &Path,
    workflow: &str,
    gate: &str,
    command: &str,
    artifact_marker: &str,
) -> CiConformanceGate {
    let workflow_path = vyre_root.join(workflow);
    let (text, read_error) = match read_conformance_text(&workflow_path) {
        Ok(text) => (text, None),
        Err(error) => (String::new(), Some(error.to_string())),
    };
    CiConformanceGate {
        workflow: workflow_path.display().to_string(),
        read_error,
        gate: gate.to_string(),
        present: text.contains(gate),
        command_present: text.contains(command),
        artifact_check_present: text.contains(artifact_marker),
    }
}

/// The required CI status names, and the errors hit reading them.
pub fn parse_required_ci_statuses(vyre_root: &Path) -> (Vec<String>, Vec<String>) {
    let path = vyre_root.join(".github/CI_REQUIRED.md");
    let text = match read_conformance_text(&path) {
        Ok(text) => text,
        Err(error) => {
            return (
                Vec::new(),
                vec![format!(
                    "could not read required CI status manifest `{}`: {error}",
                    path.display()
                )],
            );
        }
    };
    let mut statuses = BTreeSet::new();
    let mut skip_rest = false;
    for line in text.lines() {
        if line.starts_with("## Scheduled or Manual Deep Gates") {
            skip_rest = true;
        }
        if skip_rest {
            continue;
        }
        let Some(stripped) = line.strip_prefix("- `") else {
            continue;
        };
        let Some((status, _)) = stripped.split_once('`') else {
            continue;
        };
        statuses.insert(status.to_string());
    }
    (statuses.into_iter().collect(), Vec::new())
}

/// Whether any workflow defines a job producing the named status.
pub fn ci_status_defined(vyre_root: &Path, status: &str, scan_errors: &mut Vec<String>) -> bool {
    let workflow_root = vyre_root.join(".github/workflows");
    if !workflow_root.is_dir() {
        scan_errors.push(format!(
            "workflow root `{}` is not a directory while searching status `{status}`",
            workflow_root.display()
        ));
        return false;
    }
    for entry in tree_walk::pruned(&workflow_root, BUILD_OUTPUT_AND_VCS) {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                scan_errors.push(format!(
                    "could not walk workflow tree `{}` while searching status `{status}`: {error}",
                    workflow_root.display()
                ));
                continue;
            }
        };
        let path = entry.path();
        let Some(extension) = path.extension().and_then(|ext| ext.to_str()) else {
            continue;
        };
        if !matches!(extension, "yml" | "yaml") {
            continue;
        }
        let text = match read_conformance_text(path) {
            Ok(text) => text,
            Err(error) => {
                scan_errors.push(format!(
                    "could not read workflow `{}` while searching status `{status}`: {error}",
                    path.display()
                ));
                continue;
            }
        };
        if text.contains(&format!("name: {status}"))
            || text.contains(&format!("  {status}:"))
            || text.contains(&format!("    name: {status}"))
        {
            return true;
        }
    }
    false
}

/// Required workflows that a path filter can skip, which makes them optional.
pub fn inspect_path_filtered_required_workflows(vyre_root: &Path) -> Vec<String> {
    let mut findings = Vec::new();
    for workflow in REQUIRED_WORKFLOWS {
        let path = vyre_root.join(workflow);
        let Ok(text) = read_conformance_text(&path) else {
            continue;
        };
        let trigger_prefix = text
            .split_once("\njobs:")
            .map_or(text.as_str(), |(prefix, _)| prefix);
        if trigger_prefix.lines().any(|line| {
            let trimmed = line.trim_start();
            trimmed.starts_with("paths:") || trimmed.starts_with("paths-ignore:")
        }) {
            findings.push(path.display().to_string());
        }
    }
    findings
}

/// Required workflows missing a trigger, or unreadable.
pub fn inspect_required_workflow_triggers(vyre_root: &Path) -> Vec<String> {
    let mut missing = Vec::new();
    for workflow in REQUIRED_WORKFLOWS {
        let path = vyre_root.join(workflow);
        let Ok(text) = read_conformance_text(&path) else {
            missing.push(format!("{}:unreadable", path.display()));
            continue;
        };
        let trigger_prefix = text
            .split_once("\njobs:")
            .map_or(text.as_str(), |(prefix, _)| prefix);
        let has_pull_request = trigger_prefix.lines().any(|line| {
            let trimmed = line.trim();
            trimmed == "pull_request:" || trimmed.starts_with("pull_request:")
        });
        let has_push = trigger_prefix.lines().any(|line| {
            let trimmed = line.trim();
            trimmed == "push:" || trimmed.starts_with("push:")
        });
        let has_main_branch = trigger_prefix.lines().any(|line| {
            let trimmed = line.trim();
            trimmed.starts_with("branches:")
                && (trimmed.contains("[main]")
                    || trimmed.contains("[\"main\"]")
                    || trimmed.contains("[ 'main' ]")
                    || trimmed.contains("[ \"main\" ]")
                    || trimmed == "branches: main"
                    || trimmed == "branches: [ main ]")
        });
        if !(has_pull_request && has_push && has_main_branch) {
            missing.push(format!(
                "{}:pull_request={has_pull_request},push={has_push},main_branch={has_main_branch}",
                path.display()
            ));
        }
    }
    missing
}

/// Release gates whose fan-in job does not fail closed on a skipped dependency.
pub fn inspect_fail_closed_fanins(vyre_root: &Path) -> Vec<String> {
    let mut missing = Vec::new();
    for (workflow, job_name) in [
        (".github/workflows/ci.yml", "CI release gate"),
        (".github/workflows/conform.yml", "Conform release gate"),
        (".github/workflows/gpu-parity.yml", "GPU release gate"),
    ] {
        let path = vyre_root.join(workflow);
        let Ok(text) = read_conformance_text(&path) else {
            missing.push(format!("{}:{job_name}", path.display()));
            continue;
        };
        let Some(section) = workflow_job_section(&text, job_name) else {
            missing.push(format!("{}:{job_name}", path.display()));
            continue;
        };
        if !(section.contains("if: ${{ always() }}")
            && section.contains(".result")
            && section.contains("exit 1"))
        {
            missing.push(format!("{}:{job_name}", path.display()));
        }
    }
    missing
}

const REQUIRED_WORKFLOWS: &[&str] = &[
    ".github/workflows/ci.yml",
    ".github/workflows/bench.yml",
    ".github/workflows/architectural-invariants.yml",
    ".github/workflows/conform.yml",
    ".github/workflows/gpu-parity.yml",
];

fn workflow_job_section<'a>(workflow: &'a str, job_name: &str) -> Option<&'a str> {
    let marker = format!("name: {job_name}");
    let name_index = workflow.find(&marker)?;
    let job_start = workflow[..name_index]
        .rfind("\n  ")
        .map_or(0, |index| index + 1);
    let rest = &workflow[job_start..];
    let mut section_end = rest.len();
    for (offset, _) in rest.match_indices("\n  ") {
        if offset == 0 {
            continue;
        }
        let candidate = &rest[offset + 3..];
        let Some(first) = candidate.chars().next() else {
            continue;
        };
        if first.is_whitespace() {
            continue;
        }
        let first_line = candidate.lines().next().unwrap_or_default();
        if first_line.contains(':') {
            section_end = offset;
            break;
        }
    }
    Some(&rest[..section_end])
}

