//! Source hygiene release evidence for Vyre.

use std::fs;
use std::path::Path;

use crate::gate::{Finding, GateBehavior, GateCtx, GateError, Report};

mod classification;
mod panic_budget;
mod records;
mod rules;
mod scanner;
mod structural_gates;
mod syntax;
mod threshold_policy;

#[cfg(test)]
mod tests;

use classification::*;
use panic_budget::*;
use records::*;
#[cfg(test)]
use rules::*;
use scanner::*;
use structural_gates::*;
#[cfg(test)]
use syntax::*;
use threshold_policy::*;

/// Hidden-fallback pattern names the hygiene scan emits.
///
/// This list and the two below are the scan's output vocabulary, which the
/// release gate then requires the recorded scan to have covered. Both sides
/// used to spell the names out, so adding a pattern here left the gate
/// accepting evidence that never looked for it.
pub const HIDDEN_FALLBACK_PATTERNS: &[&str] = &[
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
];

/// Resource-bound pattern names the hygiene scan emits.
///
/// `truncating_duration_cast` sits here because the bound it enforces is on a
/// reported number rather than on memory: a count that exceeds what a `u64`
/// holds must clamp at the maximum, not wrap into a small value that reads as a
/// fast sample.
pub const RESOURCE_BOUND_PATTERNS: &[&str] = &[
    "std_thread_sleep",
    "thread_sleep",
    "tokio_sleep",
    "unbounded_read",
    "truncating_duration_cast",
];

/// Cargo-wrapper pattern names the hygiene scan emits.
pub const CARGO_WRAPPER_PATTERNS: &[&str] = &[
    "raw_workspace_cargo",
    "invalid_cargo_full_xtask",
    "heredoc",
    "missing_cargo_wrapper",
];

/// The hidden-fallback family as `(pattern name, matched text)`, lowercase.
///
/// A scan outside this gate reads the same excuses over a narrower surface, and
/// the backend matrix carried five of these phrases as its own list: half the
/// family, so a phrase added here reached the tree-wide scan and never reached
/// the production driver scan. The vocabulary has one owner and one accessor.
/// `gpu_unavailable_skip` is in the name list and not here because it is
/// detected structurally rather than by a phrase.
#[must_use]
pub fn hidden_fallback_pattern_texts() -> Vec<(&'static str, &'static str)> {
    records::BLOCKED_PATTERNS
        .iter()
        .copied()
        .filter(|(name, _)| rules::is_hidden_fallback_pattern(name))
        .collect()
}

/// The unresolved-work family as `(pattern name, matched text)`.
///
/// A scan outside this gate reads the same markers over a narrower surface,
/// and the backend matrix carried eight of its own. Neither list was a subset
/// of the other: `not implemented` prose reached the tree-wide scan alone, and
/// `tbd` reached the production driver surface alone, so each gate certified a
/// vocabulary the other had already extended. The vocabulary has one owner and
/// one accessor. `tbd` is absent from it because this scan reads every crate
/// and `tbd` is a schema token in the conformance certificates, where it
/// records a field state rather than unfinished work. The texts carry the case
/// they are declared with, so a caller matching lowercased source lowercases
/// them.
#[must_use]
pub fn unresolved_marker_pattern_texts() -> Vec<(&'static str, &'static str)> {
    records::BLOCKED_PATTERNS
        .iter()
        .copied()
        .filter(|(name, _)| rules::is_unresolved_marker_pattern(name))
        .collect()
}

/// The release-surface coverage flags the hygiene scan records.
///
/// A consumer outside this gate required six flag names it spelled out itself,
/// so a surface added to the record was never required of the recorded
/// evidence. The names have one owner here, and a test in this gate holds the
/// list to the record's boolean fields, so a surface added to the record fails
/// until it is listed.
pub const RELEASE_SURFACE_COVERAGE_FLAGS: &[&str] = &[
    "vyre_workspace",
    "cuda_driver_crate",
    "wgpu_driver_crate",
    "release_scripts",
    "github_workflows",
    "branch_protection_controls",
];

/// Scans the release surface for hidden fallbacks, unbounded reads, missing
/// panic contracts and undeclared thresholds, and owns the evidence artifacts.
pub struct HygieneMatrix;

impl GateBehavior for HygieneMatrix {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let root = ctx.root.clone();
        // The gate owns the artifact directory, so there is no flag that moves
        // it. A caller that could redirect the output could also make the gate
        // check a file nothing reads.
        let mut inspection = crate::artifact_gate::Inspection::new();
        let scanned_roots = vec![root.display().to_string()];
        let mut scanned_files = 0usize;
        let mut findings = Vec::new();
        scan_root(&root, &mut scanned_files, &mut findings);
        scan_source_inspection_test_files(&root, &mut scanned_files, &mut findings);
        scan_release_xtask(&root, &mut scanned_files, &mut findings);
        scan_release_tooling(&root, &mut scanned_files, &mut findings);
        scan_release_docs(&root, &mut scanned_files, &mut findings);
        scan_release_workflows(&root, &mut scanned_files, &mut findings);
        scan_release_controls(&root, &mut scanned_files, &mut findings);
        check_required_cargo_wrappers(&root, &mut findings);
        let threshold_policy = collect_threshold_policy(&root);
        let release_surface_coverage = release_surface_coverage(&root);
        let hot_paths = load_hot_path_files(&root);
        let mut structural_gates = load_structural_gates(&root);
        let test_gated = structure_gate::cfg_test::test_gated_module_files(&root);
        let finding_classes =
            classify_findings(&root, &findings, &hot_paths, &structural_gates, &test_gated);
        let panic_budget = collect_panic_budget(&root, &finding_classes);
        structural_gates.blockers.extend(stale_declaration_blockers(
            &root,
            &structural_gates.declarations,
            &findings,
        ));
        let release_blocker_count = finding_classes
            .iter()
            .filter(|finding| finding.release_blocker)
            .count();
        // Every release-blocking finding is one finding, so the pin counts what
        // the tree owes rather than one sentence about how much it owes.
        for finding in finding_classes.iter().filter(|row| row.release_blocker) {
            inspection.find(Finding::at(
                &finding.path,
                u32::try_from(finding.line).unwrap_or(u32::MAX),
                format!(
                    "release-blocking `{}` on the {} surface, owner lane {}, risk {}",
                    finding.pattern, finding.surface, finding.owner_lane, finding.risk
                ),
                "remove the pattern from the release surface, or move the code off it",
            ));
        }
        let mut blockers = if release_blocker_count == 0 {
            Vec::new()
        } else {
            vec![format!(
                "{release_blocker_count} release-blocking source hygiene finding(s) remain; {} total finding(s) preserved in classification output",
                findings.len()
            )]
        };
        blockers.extend(threshold_policy.blockers.iter().cloned());
        blockers.extend(structural_gates.blockers.iter().cloned());
        blockers.extend(panic_budget.blockers.iter().cloned());
        for blocker in threshold_policy
            .blockers
            .iter()
            .chain(structural_gates.blockers.iter())
            .chain(panic_budget.blockers.iter())
        {
            inspection.find(Finding::new(
                blocker.clone(),
                "declare the threshold, the structural gate or the panic ceiling the blocker names, or delete the stale declaration",
            ));
        }
        let scan_note = format!(
            "scanned {scanned_files} file(s) | {} hygiene finding(s) | {release_blocker_count} release-blocking",
            findings.len()
        );
        let finding_summary = finding_summary(&findings);
        let classification_summary = classification_summary(&finding_classes);
        let intake_summary = hygiene_intake_summary(&finding_classes);
        let matrix = HygieneMatrixArtifact {
            schema_version: 6,
            scanned_roots,
            scanned_files,
            release_surface_coverage,
            finding_summary,
            classification_summary,
            intake_summary,
            threshold_policy,
            structural_gates,
            panic_budget,
            finding_classes,
            release_blocker_count,
            findings,
            blockers,
        };

        inspection.generates(&format!("{ARTIFACT_DIR}/hygiene-matrix.json"), &matrix);
        declare_sibling_artifacts(&mut inspection, &matrix);
        let mut report = crate::artifact_gate::settle_inspection(ctx, ctx.gate_name()?, inspection);
        report.cover_complete("scanned release files", scanned_files);
        report.note(scan_note);
        for note in &matrix.panic_budget.notes {
            report.note(note.clone());
        }
        Ok(report)
    }
}

/// Directory every artifact this gate owns lives in.
const ARTIFACT_DIR: &str = "release/evidence/hygiene";

/// Whether `.github/workflows` holds a workflow file.
///
/// The directory itself survives the deletion of every workflow in it, so its
/// presence is not coverage.
fn holds_workflow(vyre_root: &Path) -> bool {
    let Ok(entries) = fs::read_dir(vyre_root.join(".github/workflows")) else {
        return false;
    };
    entries.flatten().any(|entry| {
        entry
            .path()
            .extension()
            .is_some_and(|extension| extension == "yml" || extension == "yaml")
    })
}

fn release_surface_coverage(vyre_root: &Path) -> ReleaseSurfaceCoverage {
    ReleaseSurfaceCoverage {
        vyre_workspace: vyre_root.join("vyre/src/lib.rs").is_file(),
        cuda_driver_crate: vyre_root.join("vyre-driver-cuda/src/lib.rs").is_file(),
        wgpu_driver_crate: vyre_root.join("vyre-driver-wgpu/src/lib.rs").is_file(),
        release_scripts: vyre_root
            .join("scripts/apply-branch-protection.sh")
            .is_file()
            && vyre_root.join("xtask/src/gates/layering.rs").is_file(),
        github_workflows: holds_workflow(vyre_root),
        branch_protection_controls: vyre_root.join(".github/CI_REQUIRED.md").is_file()
            && vyre_root
                .join("scripts/apply-branch-protection.sh")
                .is_file(),
        // The three vocabularies above are the scan's output, and the recorded
        // coverage is a claim about that output. Restating them here let a
        // pattern be added to the scan while the evidence kept naming the older
        // set.
        resource_bound_patterns: RESOURCE_BOUND_PATTERNS.to_vec(),
        hidden_fallback_patterns: HIDDEN_FALLBACK_PATTERNS.to_vec(),
        release_tooling_patterns: CARGO_WRAPPER_PATTERNS.to_vec(),
    }
}

/// Declare the eight artifacts that travel with the matrix.
///
/// Every one is derived from the matrix, so they are declared where the matrix
/// is built rather than written by a second pass that could disagree with it.
fn declare_sibling_artifacts(
    inspection: &mut crate::artifact_gate::Inspection,
    matrix: &HygieneMatrixArtifact,
) {
    let intake_blockers = if matrix.release_blocker_count == 0 {
        Vec::new()
    } else {
        vec![format!(
            "{} release-blocking hygiene finding(s) remain; implementation-intake.json groups them by owner lane, surface, risk, hot-path flag, and pattern",
            matrix.release_blocker_count
        )]
    };
    inspection.generates(
        &format!("{ARTIFACT_DIR}/implementation-intake.json"),
        &HygieneIntakeArtifact {
            schema_version: 1,
            release_blocker_count: matrix.release_blocker_count,
            intake_summary: matrix.intake_summary.clone(),
            blockers: intake_blockers,
        },
    );
    inspection.generates(
        &format!("{ARTIFACT_DIR}/threshold-policy.json"),
        &matrix.threshold_policy,
    );
    for &(artifact, scan, patterns) in HYGIENE_SCANS {
        let findings = matrix
            .findings
            .iter()
            .filter(|finding| patterns.iter().any(|pattern| pattern == &finding.pattern))
            .cloned()
            .collect::<Vec<_>>();
        let release_blocking_findings = matrix
            .finding_classes
            .iter()
            .filter(|finding| {
                finding.release_blocker
                    && patterns.iter().any(|pattern| *pattern == finding.pattern)
            })
            .cloned()
            .collect::<Vec<_>>();
        let blockers = if release_blocking_findings.is_empty() {
            Vec::new()
        } else {
            vec![format!(
                "{} release-blocking `{scan}` finding(s) remain",
                release_blocking_findings.len()
            )]
        };
        inspection.generates(
            &format!("{ARTIFACT_DIR}/{artifact}"),
            &HygieneScan {
                schema_version: 1,
                scan: scan.to_string(),
                findings,
                release_blocking_findings,
                blockers,
            },
        );
    }
}
