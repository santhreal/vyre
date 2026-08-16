//! Source hygiene release evidence for Vyre.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use quote::ToTokens;
use serde::{Deserialize, Serialize};
use syn::visit::Visit;

use crate::tree_walk::{self, BUILD_OUTPUT_AND_VCS};

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};

#[derive(Debug, Serialize)]
struct HygieneMatrixArtifact {
    schema_version: u32,
    scanned_roots: Vec<String>,
    scanned_files: usize,
    release_surface_coverage: ReleaseSurfaceCoverage,
    finding_summary: Vec<HygieneFindingSummary>,
    classification_summary: Vec<HygieneClassificationSummary>,
    intake_summary: Vec<HygieneIntakeSummary>,
    threshold_policy: ThresholdPolicyArtifact,
    structural_gates: StructuralGateArtifact,
    panic_budget: PanicBudgetArtifact,
    finding_classes: Vec<HygieneFindingClass>,
    release_blocker_count: usize,
    findings: Vec<HygieneFinding>,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ReleaseSurfaceCoverage {
    vyre_workspace: bool,
    cuda_driver_crate: bool,
    wgpu_driver_crate: bool,
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
    /// The test this finding belongs to, for the patterns that judge a test
    /// rather than a line. The structural-gate registry is keyed on it, so the
    /// name has to reach classification rather than being formatted into
    /// `text` and parsed back out.
    #[serde(skip_serializing_if = "Option::is_none")]
    test: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct HygieneFindingSummary {
    pattern: String,
    count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct HygieneFindingClass {
    path: String,
    line: usize,
    pattern: &'static str,
    owner_lane: &'static str,
    surface: &'static str,
    risk: &'static str,
    hot_path: bool,
    release_blocker: bool,
}

#[derive(Debug, Clone, Serialize)]
struct HygieneClassificationSummary {
    owner_lane: &'static str,
    surface: &'static str,
    risk: &'static str,
    hot_path: bool,
    release_blocker: bool,
    count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct HygieneIntakeSummary {
    owner_lane: &'static str,
    surface: &'static str,
    risk: &'static str,
    hot_path: bool,
    pattern: &'static str,
    release_blocker: bool,
    count: usize,
}

#[derive(Debug, Serialize)]
struct HygieneIntakeArtifact {
    schema_version: u32,
    release_blocker_count: usize,
    intake_summary: Vec<HygieneIntakeSummary>,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct HygieneScan {
    schema_version: u32,
    scan: String,
    findings: Vec<HygieneFinding>,
    release_blocking_findings: Vec<HygieneFindingClass>,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ThresholdPolicyArtifact {
    schema_version: u32,
    source_manifest: &'static str,
    evidence_artifact: String,
    owner_lane: String,
    threshold_const_count: usize,
    registered_policy_count: usize,
    rows: Vec<ThresholdPolicyEvidenceRow>,
    findings: Vec<ThresholdPolicyFinding>,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ThresholdPolicyEvidenceRow {
    id: String,
    path: String,
    line: usize,
    name: String,
    observed_value: String,
    unit: String,
    provenance: String,
    config_tier: String,
    override_path: String,
    evidence_link: String,
    release_rule: String,
}

#[derive(Debug, Clone, Serialize)]
struct ThresholdPolicyFinding {
    path: String,
    line: usize,
    name: String,
    finding: String,
    fix: String,
}

#[derive(Debug, Deserialize)]
struct ThresholdPolicyDocument {
    schema_version: u32,
    owner_lane: String,
    evidence_artifact: String,
    #[serde(default)]
    threshold: Vec<ThresholdPolicyTomlRow>,
}

#[derive(Debug, Clone, Deserialize)]
struct ThresholdPolicyTomlRow {
    id: String,
    path: String,
    name: String,
    unit: String,
    provenance: String,
    config_tier: String,
    override_path: String,
    evidence_link: String,
    release_rule: String,
}

#[derive(Debug)]
struct ObservedThresholdConst {
    path: String,
    line: usize,
    name: String,
    value: String,
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
const THRESHOLD_POLICY_SCHEMA_VERSION: u32 = 1;
const THRESHOLD_POLICY_SOURCE: &str = "docs/optimization/THRESHOLD_POLICY.toml";
const THRESHOLD_POLICY_ARTIFACT: &str = "release/evidence/hygiene/threshold-policy.json";
const THRESHOLD_POLICY_OWNER_LANE: &str = "testing_evidence";
const STRUCTURAL_GATE_SCHEMA_VERSION: u32 = 1;
const STRUCTURAL_GATE_SOURCE: &str = "docs/testing/STRUCTURAL_GATES.toml";
const PANIC_BUDGET_SCHEMA_VERSION: u32 = 1;
const PANIC_BUDGET_SOURCE: &str = "docs/testing/PANIC_BUDGET.toml";
const THRESHOLD_SUFFIXES: &[&str] = &[
    "_THRESHOLD",
    "_LIMIT",
    "_MAX",
    "_MIN",
    "_CAP",
    "_BUDGET",
    "_FLOOR",
    "_CEILING",
    "_TIMEOUT",
    "_DEADLINE",
    "_RETRY",
    "_BACKOFF",
];

/// Structural gates whose property has no run-time witness, and their status.
///
/// A source-inspecting test is a release blocker by default, because the usual
/// reason a test reads source is that nobody worked out how to assert the
/// behaviour. That default is wrong for a property no execution can observe:
/// that no other file calls a function, that a registration is visible from the
/// crate root, that a table covers every variant. Rust offers no reflection, so
/// the source is the only witness those have.
///
/// The declaration is what makes the exemption reviewable. Both halves are
/// derived from the tree, so a row naming a test that no longer exists is a
/// blocker of its own: a stale registry is worth what no registry is worth.
#[derive(Debug, Clone, Serialize)]
struct StructuralGateArtifact {
    schema_version: u32,
    source: &'static str,
    declarations: Vec<StructuralGateDeclaration>,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct StructuralGateDeclaration {
    file: String,
    test: String,
    reason: String,
}

#[derive(Debug, Deserialize)]
struct StructuralGateDocument {
    schema: u32,
    #[serde(default)]
    gate: Vec<StructuralGateTomlRow>,
}

#[derive(Debug, Deserialize)]
struct StructuralGateTomlRow {
    file: String,
    test: String,
    reason: String,
}

/// The recorded ceiling on panics that fail closed without saying so.
///
/// A panicking call in production code is acceptable when failing closed IS the
/// contract and the contract is written down, which is what
/// [`has_documented_panic_contract`] reads. A panicking call on a hot path is a
/// release blocker whatever its documentation says. Between those two sits the
/// population this ratchet bounds: a panic that is neither documented nor on the
/// release surface, which for most of this repository's history was bounded by
/// nothing. The deleted `check_no_raw_unwrap` script tried to bound it at zero
/// and could never be turned on, because zero declares the documented-panic
/// convention a violation.
///
/// A ceiling per crate rather than one number for the tree, because the crate is
/// who fixes it. Over the ceiling is a blocker: that is new debt. Under it with
/// the count still above zero is a note carrying the number to write, because a
/// gate that fails on the improvement it exists to encourage is a gate somebody
/// switches off, which is how the deleted `check_proptest_coverage` floor died.
/// A crate that reaches zero while its row still permits panics IS a blocker,
/// since that row is the only thing standing between a closed class and the next
/// panic added to that crate. So a ceiling only ever moves down.
#[derive(Debug, Clone, Serialize)]
struct PanicBudgetArtifact {
    schema_version: u32,
    source: &'static str,
    rows: Vec<PanicBudgetRow>,
    unrecorded: Vec<String>,
    notes: Vec<String>,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct PanicBudgetRow {
    crate_name: String,
    ceiling: usize,
    measured: usize,
}

#[derive(Debug, Deserialize)]
struct PanicBudgetDocument {
    schema: u32,
    #[serde(default)]
    crate_budget: Vec<PanicBudgetTomlRow>,
}

#[derive(Debug, Deserialize)]
struct PanicBudgetTomlRow {
    name: String,
    ceiling: usize,
}

/// Whether a classified finding is a panic that nothing else answers for.
///
/// Read off the classification rather than re-scanning, so the population is the
/// one the artifact records. A documented contract is a different pattern by the
/// time it reaches here, and a hot-path panic is already a release blocker, so
/// neither is counted twice.
fn is_unbounded_panic(class: &HygieneFindingClass) -> bool {
    matches!(class.pattern, "panic_macro" | "unwrap_call" | "expect_call")
        && !class.release_blocker
        && matches!(class.surface, "production" | "release_tooling")
}

/// The crate a scanned path belongs to.
///
/// The first path component, which is the crate directory for every workspace
/// member and the containing directory for the two nested ones. Derived from the
/// path so a new crate needs no edit here to be counted.
fn crate_of_path(path: &str) -> String {
    path.replace('\\', "/")
        .split('/')
        .next()
        .unwrap_or_default()
        .to_string()
}

/// Hold the undocumented panic population to the recorded per-crate ceiling.
///
/// Every failure path returns a blocker rather than an empty budget, because a
/// budget that could not be read must not read as a tree that owes nothing.
fn collect_panic_budget(vyre_root: &Path, classes: &[HygieneFindingClass]) -> PanicBudgetArtifact {
    let mut measured = BTreeMap::<String, usize>::new();
    for class in classes.iter().filter(|class| is_unbounded_panic(class)) {
        let relative = relative_to_vyre(vyre_root, Path::new(&class.path));
        *measured.entry(crate_of_path(&relative)).or_insert(0) += 1;
    }

    let mut artifact = PanicBudgetArtifact {
        schema_version: PANIC_BUDGET_SCHEMA_VERSION,
        source: PANIC_BUDGET_SOURCE,
        rows: Vec::new(),
        unrecorded: Vec::new(),
        notes: Vec::new(),
        blockers: Vec::new(),
    };

    let path = vyre_root.join(PANIC_BUDGET_SOURCE);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) => {
            artifact.blockers.push(format!(
                "{PANIC_BUDGET_SOURCE} could not be read ({error}), so {} undocumented panic(s) outside the release surface are bounded by nothing",
                measured.values().sum::<usize>()
            ));
            return artifact;
        }
    };
    let document = match toml::from_str::<PanicBudgetDocument>(&text) {
        Ok(document) => document,
        Err(error) => {
            artifact.blockers.push(format!(
                "{PANIC_BUDGET_SOURCE} is not readable as a panic budget: {error}"
            ));
            return artifact;
        }
    };
    if document.schema != PANIC_BUDGET_SCHEMA_VERSION {
        artifact.blockers.push(format!(
            "{PANIC_BUDGET_SOURCE} declares schema {} against {PANIC_BUDGET_SCHEMA_VERSION}",
            document.schema
        ));
        return artifact;
    }

    let mut ceilings = BTreeMap::<String, usize>::new();
    for row in document.crate_budget {
        if let Some(previous) = ceilings.insert(row.name.clone(), row.ceiling) {
            artifact.blockers.push(format!(
                "{PANIC_BUDGET_SOURCE} records {} twice, at {previous} and {}, so one ceiling is unread",
                row.name, row.ceiling
            ));
        }
    }

    for (crate_name, count) in &measured {
        match ceilings.get(crate_name) {
            Some(ceiling) if count > ceiling => artifact.blockers.push(format!(
                "{crate_name} carries {count} undocumented panic(s) outside the release surface against a ceiling of {ceiling}: document the contract in a `# Panics` section, return an error instead, or delete the panic"
            )),
            Some(ceiling) if count < ceiling => artifact.notes.push(format!(
                "{crate_name} carries {count} undocumented panic(s) against a ceiling of {ceiling}: lower the ceiling in {PANIC_BUDGET_SOURCE} to {count}, because a ceiling above the tree covers the next panic added to it"
            )),
            Some(_) => {}
            None => {
                artifact.unrecorded.push(crate_name.clone());
                artifact.blockers.push(format!(
                    "{crate_name} carries {count} undocumented panic(s) outside the release surface and {PANIC_BUDGET_SOURCE} records no ceiling for it"
                ));
            }
        }
    }
    for (crate_name, ceiling) in &ceilings {
        if !measured.contains_key(crate_name) && *ceiling > 0 {
            artifact.blockers.push(format!(
                "{PANIC_BUDGET_SOURCE} records a ceiling of {ceiling} for {crate_name}, which now carries none: lower the row to 0, because the ceiling is what stands between the crate and the next panic added to it"
            ));
        }
    }

    artifact.rows = ceilings
        .into_iter()
        .map(|(crate_name, ceiling)| PanicBudgetRow {
            measured: measured.get(&crate_name).copied().unwrap_or_default(),
            crate_name,
            ceiling,
        })
        .collect();
    artifact
}

/// Read the structural-gate registry, or report why it could not be trusted.
///
/// Every failure path returns an empty declaration set plus a blocker, so an
/// unreadable or malformed registry exempts nothing rather than everything.
fn load_structural_gates(vyre_root: &Path) -> StructuralGateArtifact {
    let mut artifact = StructuralGateArtifact {
        schema_version: STRUCTURAL_GATE_SCHEMA_VERSION,
        source: STRUCTURAL_GATE_SOURCE,
        declarations: Vec::new(),
        blockers: Vec::new(),
    };
    let path = vyre_root.join(STRUCTURAL_GATE_SOURCE);
    let text = match read_text_bounded(&path) {
        Ok(text) => text,
        Err(error) => {
            artifact.blockers.push(format!(
                "{STRUCTURAL_GATE_SOURCE} is unreadable: {error}. Fix: restore the structural-gate registry; without it every source-inspecting gate is a release blocker."
            ));
            return artifact;
        }
    };
    let document = match toml::from_str::<StructuralGateDocument>(&text) {
        Ok(document) => document,
        Err(error) => {
            artifact.blockers.push(format!(
                "{STRUCTURAL_GATE_SOURCE} is not valid structural-gate TOML: {error}. Fix: repair the schema before release."
            ));
            return artifact;
        }
    };
    if document.schema != STRUCTURAL_GATE_SCHEMA_VERSION {
        artifact.blockers.push(format!(
            "{STRUCTURAL_GATE_SOURCE} declares schema {} but this gate reads schema {STRUCTURAL_GATE_SCHEMA_VERSION}. Fix: migrate the registry before release.",
            document.schema
        ));
        return artifact;
    }
    let mut seen = BTreeSet::new();
    for row in document.gate {
        if row.file.trim().is_empty() || row.test.trim().is_empty() {
            artifact.blockers.push(format!(
                "{STRUCTURAL_GATE_SOURCE} has a row with an empty file or test name. Fix: name both."
            ));
            continue;
        }
        if row.reason.trim().is_empty() {
            artifact.blockers.push(format!(
                "{STRUCTURAL_GATE_SOURCE}: `{}` in `{}` declares no reason. Fix: state why the property has no run-time witness.",
                row.test, row.file
            ));
            continue;
        }
        if !seen.insert((row.file.clone(), row.test.clone())) {
            artifact.blockers.push(format!(
                "{STRUCTURAL_GATE_SOURCE}: `{}` in `{}` is declared twice. Fix: keep one row.",
                row.test, row.file
            ));
            continue;
        }
        artifact.declarations.push(StructuralGateDeclaration {
            file: row.file,
            test: row.test,
            reason: row.reason.trim().to_string(),
        });
    }
    artifact
}

/// Blockers for registry rows the tree no longer backs.
///
/// Derived from the same scan that produced the findings, so the registry
/// cannot outlive the gates it exempts.
fn stale_declaration_blockers(
    vyre_root: &Path,
    declarations: &[StructuralGateDeclaration],
    findings: &[HygieneFinding],
) -> Vec<String> {
    let mut inspecting = BTreeMap::<String, BTreeSet<&str>>::new();
    for finding in findings {
        if finding.pattern != "source_inspection_test" {
            continue;
        }
        let Some(test) = finding.test.as_deref() else {
            continue;
        };
        inspecting
            .entry(relative_to_vyre(vyre_root, Path::new(&finding.path)))
            .or_default()
            .insert(test);
    }
    let mut blockers = Vec::new();
    for declaration in declarations {
        match inspecting.get(&declaration.file) {
            None => blockers.push(format!(
                "{STRUCTURAL_GATE_SOURCE}: `{}` names `{}`, which contains no source-inspecting test. Fix: delete the row; a registry that outlives its gate exempts nothing and hides the next one.",
                declaration.test, declaration.file
            )),
            Some(tests) if !tests.contains(declaration.test.as_str()) => blockers.push(format!(
                "{STRUCTURAL_GATE_SOURCE}: `{}` is declared for `{}`, which no longer has a test by that name that inspects source. Fix: delete the row or correct the name.",
                declaration.test, declaration.file
            )),
            Some(_) => {}
        }
    }
    blockers
}

/// Whether a source-inspecting `finding` is covered by a reviewed declaration.
fn is_declared_structural_gate(
    vyre_root: &Path,
    finding: &HygieneFinding,
    structural_gates: &StructuralGateArtifact,
) -> bool {
    let Some(test) = finding.test.as_deref() else {
        return false;
    };
    let file = relative_to_vyre(vyre_root, Path::new(&finding.path));
    structural_gates
        .declarations
        .iter()
        .any(|declaration| declaration.file == file && declaration.test == test)
}

/// Scans the release surface for hidden fallbacks, unbounded reads, missing
/// panic contracts and undeclared thresholds, and owns the evidence artifacts.
pub struct HygieneMatrix;

impl Gate for HygieneMatrix {
    fn name(&self) -> &'static str {
        "hygiene-matrix"
    }

    fn help(&self) -> &'static str {
        "Scan the release surface for hidden fallbacks, unbounded reads, undocumented panics and undeclared thresholds, and hold release/evidence/hygiene to what it found; --write regenerates it"
    }

    fn generates(&self) -> bool {
        true
    }

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
        let finding_classes = classify_findings(&root, &findings, &hot_paths, &structural_gates);
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
        let mut report = crate::artifact_gate::settle_inspection(ctx, self.name(), inspection);
        report.note(scan_note);
        for note in &matrix.panic_budget.notes {
            report.note(note.clone());
        }
        Ok(report)
    }
}

/// Directory every artifact this gate owns lives in.
const ARTIFACT_DIR: &str = "release/evidence/hygiene";

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

fn classify_findings(
    vyre_root: &Path,
    findings: &[HygieneFinding],
    hot_paths: &std::collections::BTreeSet<String>,
    structural_gates: &StructuralGateArtifact,
) -> Vec<HygieneFindingClass> {
    findings
        .iter()
        .map(|finding| {
            let owner_lane = hygiene_owner_lane_for_path(&finding.path);
            let surface = hygiene_surface_for_path(&finding.path);
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

fn classification_summary(classes: &[HygieneFindingClass]) -> Vec<HygieneClassificationSummary> {
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

fn hygiene_intake_summary(classes: &[HygieneFindingClass]) -> Vec<HygieneIntakeSummary> {
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

fn hygiene_owner_lane_for_path(path: &str) -> &'static str {
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

fn hygiene_surface_for_path(path: &str) -> &'static str {
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
        || is_cpu_parity_oracle_source(&normalized)
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
fn is_xtask_tree_path(normalized: &str) -> bool {
    normalized.contains("/xtask/") || normalized.contains("/xtask-")
}

/// Whether a path is xtask source rather than an xtask manifest or README.
fn is_xtask_source_path(normalized: &str) -> bool {
    normalized.contains("/xtask/src/") || xtask_crate_source_segment(normalized)
}

/// Whether `normalized` runs through `xtask-<name>/src/`.
fn xtask_crate_source_segment(normalized: &str) -> bool {
    normalized.split("/xtask-").skip(1).any(|tail| {
        tail.split_once('/')
            .is_some_and(|(_crate_name, rest)| rest == "src" || rest.starts_with("src/"))
    })
}

fn is_cpu_parity_oracle_source(normalized_path: &str) -> bool {
    normalized_path.ends_with("/cpu_oracle.rs")
        || normalized_path.ends_with("_cpu_oracle.rs")
        || normalized_path.ends_with("/bitset_closure_oracle.rs")
        || normalized_path.ends_with("/reaching/oracle.rs")
}

/// The release risk of one finding.
///
/// `declared` is true only for a source-inspecting test that
/// `docs/testing/STRUCTURAL_GATES.toml` records as asserting a property with no
/// run-time witness. Everything else about a source-inspecting test is
/// unchanged: it is a release blocker, because a test that reads source when it
/// could have run the code is a test that proves nothing about behaviour.
fn hygiene_risk(pattern: &str, surface: &str, hot_path: bool, declared: bool) -> &'static str {
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

fn load_hot_path_files(vyre_root: &Path) -> std::collections::BTreeSet<String> {
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

fn hygiene_finding_is_hot_path(
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

fn collect_threshold_policy(vyre_root: &Path) -> ThresholdPolicyArtifact {
    let observed = scan_threshold_constants(vyre_root);
    let mut findings = Vec::new();
    let mut blockers = Vec::new();
    let policy_path = vyre_root.join(THRESHOLD_POLICY_SOURCE);
    let document = match fs::read_to_string(&policy_path) {
        Ok(text) => match toml::from_str::<ThresholdPolicyDocument>(&text) {
            Ok(document) => Some(document),
            Err(error) => {
                blockers.push(format!(
                    "{} is not valid threshold policy TOML: {error}. Fix: repair the TOML schema before release.",
                    THRESHOLD_POLICY_SOURCE
                ));
                None
            }
        },
        Err(error) => {
            blockers.push(format!(
                "missing {}: {error}. Fix: add unit, provenance, config tier, override path, evidence link, and release rule for every threshold-shaped const.",
                THRESHOLD_POLICY_SOURCE
            ));
            None
        }
    };
    let Some(document) = document else {
        return ThresholdPolicyArtifact {
            schema_version: THRESHOLD_POLICY_SCHEMA_VERSION,
            source_manifest: THRESHOLD_POLICY_SOURCE,
            evidence_artifact: THRESHOLD_POLICY_ARTIFACT.to_string(),
            owner_lane: THRESHOLD_POLICY_OWNER_LANE.to_string(),
            threshold_const_count: observed.len(),
            registered_policy_count: 0,
            rows: Vec::new(),
            findings,
            blockers,
        };
    };
    if document.schema_version != THRESHOLD_POLICY_SCHEMA_VERSION {
        blockers.push(format!(
            "{} schema_version={} must be {THRESHOLD_POLICY_SCHEMA_VERSION}. Fix: update the threshold policy reader and manifest together.",
            THRESHOLD_POLICY_SOURCE, document.schema_version
        ));
    }
    if document.owner_lane != THRESHOLD_POLICY_OWNER_LANE {
        blockers.push(format!(
            "{} owner_lane `{}` must be `{THRESHOLD_POLICY_OWNER_LANE}`. Fix: keep threshold evidence under the hygiene/testing lane.",
            THRESHOLD_POLICY_SOURCE, document.owner_lane
        ));
    }
    if document.evidence_artifact != THRESHOLD_POLICY_ARTIFACT {
        blockers.push(format!(
            "{} evidence_artifact `{}` must be `{THRESHOLD_POLICY_ARTIFACT}`. Fix: point the policy at the generated hygiene sibling artifact.",
            THRESHOLD_POLICY_SOURCE, document.evidence_artifact
        ));
    }
    let mut observed_by_key = BTreeMap::new();
    for threshold in observed {
        observed_by_key.insert(threshold_key(&threshold.path, &threshold.name), threshold);
    }
    let mut policy_by_key = BTreeMap::new();
    for row in &document.threshold {
        let row_key = threshold_key(&row.path, &row.name);
        if let Some(previous) = policy_by_key.insert(row_key.clone(), row.clone()) {
            blockers.push(format!(
                "{} duplicates threshold policy key `{}` for ids `{}` and `{}`. Fix: keep exactly one row per path/name threshold.",
                THRESHOLD_POLICY_SOURCE, row_key, previous.id, row.id
            ));
        }
        validate_threshold_policy_row(row, &mut blockers);
    }
    let mut rows = Vec::new();
    for (key, threshold) in &observed_by_key {
        let Some(policy) = policy_by_key.get(key) else {
            findings.push(ThresholdPolicyFinding {
                path: threshold.path.clone(),
                line: threshold.line,
                name: threshold.name.clone(),
                finding: "unregistered-threshold-const".to_string(),
                fix: format!(
                    "Fix: add `{}`/`{}` to {} with unit, provenance, config_tier, override_path, evidence_link, and release_rule.",
                    threshold.path, threshold.name, THRESHOLD_POLICY_SOURCE
                ),
            });
            blockers.push(format!(
                "{}:{} threshold const `{}` is missing from {}. Fix: register its unit, provenance, config tier, override path, evidence link, and VX release rule.",
                threshold.path, threshold.line, threshold.name, THRESHOLD_POLICY_SOURCE
            ));
            continue;
        };
        rows.push(ThresholdPolicyEvidenceRow {
            id: policy.id.clone(),
            path: threshold.path.clone(),
            line: threshold.line,
            name: threshold.name.clone(),
            observed_value: threshold.value.clone(),
            unit: policy.unit.clone(),
            provenance: policy.provenance.clone(),
            config_tier: policy.config_tier.clone(),
            override_path: policy.override_path.clone(),
            evidence_link: policy.evidence_link.clone(),
            release_rule: policy.release_rule.clone(),
        });
    }
    for (key, policy) in &policy_by_key {
        if !observed_by_key.contains_key(key) {
            findings.push(ThresholdPolicyFinding {
                path: policy.path.clone(),
                line: 1,
                name: policy.name.clone(),
                finding: "stale-threshold-policy-row".to_string(),
                fix: format!(
                    "Fix: remove or update stale threshold policy row `{}` after moving the source constant.",
                    policy.id
                ),
            });
            blockers.push(format!(
                "{} row `{}` points at `{}`/`{}` but no matching threshold const was observed. Fix: update or remove the stale policy row.",
                THRESHOLD_POLICY_SOURCE, policy.id, policy.path, policy.name
            ));
        }
    }
    rows.sort_by(|left, right| {
        left.path
            .cmp(&right.path)
            .then(left.line.cmp(&right.line))
            .then(left.name.cmp(&right.name))
    });
    ThresholdPolicyArtifact {
        schema_version: THRESHOLD_POLICY_SCHEMA_VERSION,
        source_manifest: THRESHOLD_POLICY_SOURCE,
        evidence_artifact: THRESHOLD_POLICY_ARTIFACT.to_string(),
        owner_lane: document.owner_lane,
        threshold_const_count: observed_by_key.len(),
        registered_policy_count: policy_by_key.len(),
        rows,
        findings,
        blockers,
    }
}

fn validate_threshold_policy_row(row: &ThresholdPolicyTomlRow, blockers: &mut Vec<String>) {
    for (field, value) in [
        ("id", row.id.as_str()),
        ("path", row.path.as_str()),
        ("name", row.name.as_str()),
        ("unit", row.unit.as_str()),
        ("provenance", row.provenance.as_str()),
        ("config_tier", row.config_tier.as_str()),
        ("override_path", row.override_path.as_str()),
        ("evidence_link", row.evidence_link.as_str()),
        ("release_rule", row.release_rule.as_str()),
    ] {
        if value.trim().is_empty() {
            blockers.push(format!(
                "{} row `{}` has blank {field}. Fix: every threshold policy row must carry unit, provenance, tier, override, evidence, and VX ownership.",
                THRESHOLD_POLICY_SOURCE, row.id
            ));
        }
    }
    if !matches!(row.config_tier.as_str(), "tier_a" | "tier_b" | "structural") {
        blockers.push(format!(
            "{} row `{}` uses config_tier `{}`. Fix: use `tier_a`, `tier_b`, or `structural`.",
            THRESHOLD_POLICY_SOURCE, row.id, row.config_tier
        ));
    }
    if row.config_tier == "tier_a"
        && !(row.override_path.contains("tool.toml") && row.override_path.contains("CLI"))
    {
        blockers.push(format!(
            "{} row `{}` is Tier A but override_path does not name tool.toml and CLI override behavior. Fix: record compiled default -> tool.toml -> CLI precedence.",
            THRESHOLD_POLICY_SOURCE, row.id
        ));
    }
    if row.config_tier == "tier_b" && !row.override_path.contains("TOML data") {
        blockers.push(format!(
            "{} row `{}` is Tier B but override_path does not name TOML data ownership. Fix: keep community/data thresholds out of CLI flags.",
            THRESHOLD_POLICY_SOURCE, row.id
        ));
    }
    if row.config_tier == "structural" && !row.override_path.contains("not operator configurable") {
        blockers.push(format!(
            "{} row `{}` is structural but override_path does not say `not operator configurable`. Fix: separate wire/ABI bounds from runtime knobs.",
            THRESHOLD_POLICY_SOURCE, row.id
        ));
    }
    if row.evidence_link != THRESHOLD_POLICY_ARTIFACT {
        blockers.push(format!(
            "{} row `{}` evidence_link `{}` must be `{THRESHOLD_POLICY_ARTIFACT}`.",
            THRESHOLD_POLICY_SOURCE, row.id, row.evidence_link
        ));
    }
    if row.release_rule != "VX-475" {
        blockers.push(format!(
            "{} row `{}` release_rule `{}` must be `VX-475`.",
            THRESHOLD_POLICY_SOURCE, row.id, row.release_rule
        ));
    }
}

fn scan_threshold_constants(vyre_root: &Path) -> Vec<ObservedThresholdConst> {
    let mut thresholds = Vec::new();
    for root in threshold_scan_roots(vyre_root) {
        if !root.exists() {
            continue;
        }
        for entry in tree_walk::pruned_by(&root, |name| {
            !BUILD_OUTPUT_AND_VCS.contains(&name) && name != "tests"
        }) {
            let Ok(entry) = entry else {
                continue;
            };
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
                continue;
            }
            let Ok(text) = read_text_bounded(path) else {
                thresholds.push(ObservedThresholdConst {
                    path: relative_to_vyre(vyre_root, path),
                    line: 1,
                    name: "unreadable-threshold-source".to_string(),
                    value: "unreadable".to_string(),
                });
                continue;
            };
            for (line_index, line) in text.lines().enumerate() {
                let Some((name, value)) = parse_threshold_const(line) else {
                    continue;
                };
                thresholds.push(ObservedThresholdConst {
                    path: relative_to_vyre(vyre_root, path),
                    line: line_index + 1,
                    name,
                    value,
                });
            }
        }
    }
    thresholds
}

fn threshold_scan_roots(vyre_root: &Path) -> Vec<PathBuf> {
    [
        "vyre-foundation/src/optimizer",
        "vyre-runtime/src/resident_work_queue",
        "vyre-driver-wgpu/src/runtime",
        "vyre-driver-wgpu/src/buffer",
    ]
    .iter()
    .map(|relative| vyre_root.join(relative))
    .collect()
}

fn parse_threshold_const(line: &str) -> Option<(String, String)> {
    let code = line.split("//").next().unwrap_or(line).trim();
    let const_index = code.find("const ")?;
    let rest = &code[const_index + "const ".len()..];
    let colon_index = rest.find(':')?;
    let name = rest[..colon_index].trim();
    if name.is_empty()
        || !name
            .chars()
            .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_')
        || !THRESHOLD_SUFFIXES
            .iter()
            .any(|suffix| name.ends_with(suffix))
    {
        return None;
    }
    let equals_index = rest.find('=')?;
    let value = rest[equals_index + 1..].split(';').next()?.trim();
    if !value.chars().next().is_some_and(|ch| ch.is_ascii_digit()) {
        return None;
    }
    Some((name.to_string(), value.to_string()))
}

fn threshold_key(path: &str, name: &str) -> String {
    format!("{path}::{name}")
}

fn relative_to_vyre(vyre_root: &Path, path: &Path) -> String {
    path.strip_prefix(vyre_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

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
pub const RESOURCE_BOUND_PATTERNS: &[&str] = &[
    "std_thread_sleep",
    "thread_sleep",
    "tokio_sleep",
    "unbounded_read",
];

/// Cargo-wrapper pattern names the hygiene scan emits.
pub const CARGO_WRAPPER_PATTERNS: &[&str] = &[
    "raw_workspace_cargo",
    "invalid_cargo_full_xtask",
    "heredoc",
    "missing_cargo_wrapper",
];

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
fn is_test_source_path(path: &Path) -> bool {
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

fn scan_root(root: &Path, scanned_files: &mut usize, findings: &mut Vec<HygieneFinding>) {
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
fn scan_source_inspection_test_files(
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
fn is_xtask_tree_directory(name: &str) -> bool {
    name == "xtask" || name.starts_with("xtask-")
}

/// The `src` directory of every xtask crate, `xtask` first.
fn xtask_source_roots(root: &Path) -> Vec<PathBuf> {
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
fn scan_release_xtask(root: &Path, scanned_files: &mut usize, findings: &mut Vec<HygieneFinding>) {
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
fn scan_release_docs(
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

fn scan_release_workflows(
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

fn check_required_cargo_wrappers(vyre_root: &Path, findings: &mut Vec<HygieneFinding>) {
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

fn scan_release_controls(
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

#[derive(Default)]
struct RustSourceFactsVisitor {
    reads_rust_source: bool,
    calls_read_to_string: bool,
    mentions_rust_path: bool,
    inspects_text: bool,
    callees: BTreeSet<String>,
    aliases: BTreeMap<String, String>,
}

impl RustSourceFactsVisitor {
    fn callee_name(expression: &syn::Expr) -> Option<String> {
        match expression {
            syn::Expr::Path(path) => path
                .path
                .segments
                .last()
                .map(|segment| segment.ident.to_string()),
            syn::Expr::Paren(paren) => Self::callee_name(&paren.expr),
            _ => None,
        }
    }

    fn resolved_callee(&self, name: String) -> String {
        self.aliases.get(&name).cloned().unwrap_or(name)
    }
}

/// Collect the identifiers a macro body names, recursing into its groups.
///
/// A macro body is opaque to `syn`'s typed visitors, so a call inside
/// `assert!(...)` is only reachable through its raw tokens. Rendering those
/// tokens to a string and splitting on non-identifier characters also splits
/// the CONTENTS of every string literal: `assert!(text.contains("vyre-scan"))`
/// then claims a call to `scan`, and the transitive walk enters whatever
/// unrelated local function carries that name. Walking the token trees keeps
/// the real call names and drops literals, punctuation, and lifetimes.
fn collect_macro_identifiers(tokens: proc_macro2::TokenStream, callees: &mut BTreeSet<String>) {
    for tree in tokens {
        match tree {
            proc_macro2::TokenTree::Ident(ident) => {
                callees.insert(ident.to_string());
            }
            proc_macro2::TokenTree::Group(group) => {
                collect_macro_identifiers(group.stream(), callees);
            }
            proc_macro2::TokenTree::Literal(_) | proc_macro2::TokenTree::Punct(_) => {}
        }
    }
}

impl<'ast> Visit<'ast> for RustSourceFactsVisitor {
    fn visit_macro(&mut self, expression: &'ast syn::Macro) {
        if expression.path.is_ident("include_str")
            && syn::parse2::<syn::LitStr>(expression.tokens.clone())
                .is_ok_and(|path| path.value().ends_with(".rs"))
        {
            self.reads_rust_source = true;
        }
        collect_macro_identifiers(expression.tokens.clone(), &mut self.callees);
        syn::visit::visit_macro(self, expression);
    }

    fn visit_expr_call(&mut self, expression: &'ast syn::ExprCall) {
        if let Some(name) = Self::callee_name(&expression.func) {
            let name = self.resolved_callee(name);
            if name == "read_to_string" {
                self.calls_read_to_string = true;
                let arguments = expression.args.to_token_stream().to_string();
                self.reads_rust_source |= arguments.contains(".rs");
            }
            self.callees.insert(name);
        }
        syn::visit::visit_expr_call(self, expression);
    }

    fn visit_expr_path(&mut self, expression: &'ast syn::ExprPath) {
        if let Some(segment) = expression.path.segments.last() {
            self.callees.insert(segment.ident.to_string());
            if segment.ident == "read_to_string" {
                self.calls_read_to_string = true;
            }
        }
        syn::visit::visit_expr_path(self, expression);
    }

    fn visit_expr_method_call(&mut self, expression: &'ast syn::ExprMethodCall) {
        let method = expression.method.to_string();
        if method == "read_to_string" {
            self.calls_read_to_string = true;
        }
        if matches!(
            method.as_str(),
            "contains" | "split" | "matches" | "starts_with" | "ends_with"
        ) {
            self.inspects_text = true;
        }
        self.callees.insert(method);
        if let syn::Expr::Path(receiver) = expression.receiver.as_ref() {
            if let Some(segment) = receiver.path.segments.last() {
                self.callees.insert(segment.ident.to_string());
            }
        }
        syn::visit::visit_expr_method_call(self, expression);
    }

    fn visit_expr_lit(&mut self, expression: &'ast syn::ExprLit) {
        if let syn::Lit::Str(value) = &expression.lit {
            let value = value.value();
            if value == "rs" || value.ends_with(".rs") {
                self.mentions_rust_path = true;
            }
        }
        syn::visit::visit_expr_lit(self, expression);
    }

    fn visit_local(&mut self, local: &'ast syn::Local) {
        if let (syn::Pat::Ident(alias), Some(initializer)) = (&local.pat, &local.init) {
            if let Some(target) = Self::callee_name(&initializer.expr) {
                self.aliases.insert(alias.ident.to_string(), target);
            }
        }
        syn::visit::visit_local(self, local);
    }
}

fn attrs_are_test(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attribute| {
        attribute
            .path()
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "test")
    })
}

struct SourceInspectionFunction {
    line: usize,
    name: String,
    is_test: bool,
    facts: RustSourceFactsVisitor,
}

#[derive(Default)]
struct SourceInspectionFunctionCollector {
    cfg_test_depth: usize,
    functions: Vec<SourceInspectionFunction>,
}

impl SourceInspectionFunctionCollector {
    fn push_function(&mut self, name: String, line: usize, is_test: bool, block: &syn::Block) {
        let mut facts = RustSourceFactsVisitor::default();
        facts.visit_block(block);
        let mut tokens = block.to_token_stream().to_string();
        tokens.retain(|character| !character.is_whitespace());
        if !is_test {
            facts.calls_read_to_string |= tokens.contains("read_to_string");
            facts.mentions_rust_path |= tokens.contains("\"rs\"")
                || tokens.contains(".rs\"")
                || (tokens.contains("extension()") && tokens.contains("==\"rs\""));
            if tokens.contains("read_to_string") {
                facts.calls_read_to_string = true;
                facts.callees.insert("read_to_string".to_string());
            }
            facts.reads_rust_source |= facts.calls_read_to_string && facts.mentions_rust_path;
        }
        facts.inspects_text |= [
            ".contains(",
            ".split(",
            ".matches(",
            ".starts_with(",
            ".ends_with(",
        ]
        .iter()
        .any(|needle| tokens.contains(needle));
        if !is_test && facts.calls_read_to_string && facts.mentions_rust_path {
            facts.reads_rust_source = true;
        }
        self.functions.push(SourceInspectionFunction {
            line,
            name,
            is_test,
            facts,
        });
    }
}

impl<'ast> Visit<'ast> for SourceInspectionFunctionCollector {
    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        self.push_function(
            item.sig.ident.to_string(),
            item.sig.ident.span().start().line,
            self.cfg_test_depth != 0 || attrs_are_test(&item.attrs),
            &item.block,
        );
        for statement in &item.block.stmts {
            syn::visit::visit_stmt(self, statement);
        }
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        self.push_function(
            item.sig.ident.to_string(),
            item.sig.ident.span().start().line,
            self.cfg_test_depth != 0 || attrs_are_test(&item.attrs),
            &item.block,
        );
        for statement in &item.block.stmts {
            syn::visit::visit_stmt(self, statement);
        }
    }
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        let is_test_module = item.attrs.iter().any(|attribute| {
            attribute.path().is_ident("cfg")
                && attribute
                    .meta
                    .to_token_stream()
                    .to_string()
                    .contains("test")
        });
        if is_test_module {
            self.cfg_test_depth += 1;
        }
        if let Some((_, items)) = &item.content {
            for nested in items {
                self.visit_item(nested);
            }
        }
        if is_test_module {
            self.cfg_test_depth -= 1;
        }
    }

    fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
        for implementation_item in &item.items {
            if let syn::ImplItem::Fn(function) = implementation_item {
                self.visit_impl_item_fn(function);
            }
        }
    }
}

fn source_inspection_test_findings(file: &syn::File) -> Vec<(usize, String)> {
    let mut collector = SourceInspectionFunctionCollector::default();
    collector.visit_file(file);
    let functions = &collector.functions;
    let mut by_name = BTreeMap::<&str, Vec<usize>>::new();
    for (index, function) in functions.iter().enumerate() {
        by_name.entry(&function.name).or_default().push(index);
    }
    let mut findings = Vec::new();

    for (test_index, test) in functions
        .iter()
        .enumerate()
        .filter(|(_, function)| function.is_test)
    {
        let mut stack = vec![test_index];
        let mut visited = BTreeSet::new();
        let mut reads_rust_source = false;
        let mut calls_read_to_string = false;
        let mut mentions_rust_path = false;
        let mut inspects_text = false;
        while let Some(index) = stack.pop() {
            if !visited.insert(index) {
                continue;
            }
            let facts = &functions[index].facts;
            reads_rust_source |= facts.reads_rust_source;
            calls_read_to_string |= facts.calls_read_to_string;
            mentions_rust_path |= facts.mentions_rust_path;
            inspects_text |= facts.inspects_text;
            for callee in &facts.callees {
                if let Some(indices) = by_name.get(callee.as_str()) {
                    stack.extend(indices);
                }
            }
        }
        if reads_rust_source && inspects_text {
            findings.push((test.line, test.name.clone()));
        }
    }
    findings.sort();
    findings.dedup();
    findings
}

fn scan_source_inspection_tests(path: &Path, text: &str, findings: &mut Vec<HygieneFinding>) {
    if path.file_name().and_then(|name| name.to_str()) == Some("hygiene_matrix.rs") {
        return;
    }
    let Ok(file) = syn::parse_file(text) else {
        return;
    };
    for (line, test) in source_inspection_test_findings(&file) {
        findings.push(HygieneFinding {
            path: path.display().to_string(),
            line,
            pattern: "source_inspection_test",
            text: format!(
                "test `{test}` inspects Rust source text. Fix: assert behavior, lifecycle ownership, generated registry ownership, or emitted artifacts instead, or declare the property as unobservable in {STRUCTURAL_GATE_SOURCE}."
            ),
            test: Some(test),
        });
    }
}

/// True when a `#[cfg(...)]` attribute gates the item to test builds only.
///
/// Any predicate mentioning the `test` cfg compiles only in a test build, so the item
/// behind it is test code no matter how the predicate is spelled. The previous version
/// listed four exact spellings and missed `#[cfg(all(test, feature = "..."))]`, which is
/// how the regex scan suites gate themselves: four `mod tests` blocks were scanned as
/// production source and their test helpers reported as release blockers. `not(test)` is
/// the opposite gate and stays in scope.
fn is_non_release_cfg_attr(trimmed: &str) -> bool {
    if !trimmed.starts_with("#[cfg(") || trimmed.contains("not(test)") {
        return false;
    }
    let predicate = trimmed
        .trim_start_matches("#[cfg(")
        .trim_end_matches(")]")
        .trim_end_matches(')');
    predicate
        .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        .any(|token| token == "test")
}

/// True when `line` calls a filesystem read or reads a stream to the end.
///
/// The `fs::` forms are matched as a path segment, not as a substring. A plain
/// `line.contains("fs::read(")` also matched `BufferRefs::read(count_buffer)`,
/// whose type name happens to end in `fs`; that call reads a GPU buffer
/// reference and has no file, no length, and nothing to bound. A false positive
/// here is not harmless: it is a permanent release blocker on correct code, and
/// the only way to clear it would have been to rename the type.
fn line_contains_read_call(line: &str) -> bool {
    calls_path_function(line, "fs::read_to_string")
        || calls_path_function(line, "fs::read")
        || line.contains(".read_to_end(")
        || line.contains(".read_to_string(")
}

/// True when `line` calls `name` as a whole path segment rather than as a suffix.
fn calls_path_function(line: &str, name: &str) -> bool {
    line.match_indices(name)
        .any(|(index, _)| is_word_start(line, index) && line[index + name.len()..].starts_with('('))
}

fn line_contains_unbounded_read(path: &Path, line: &str) -> bool {
    let normalized = path.to_string_lossy();
    if is_xtask_source_path(&normalized.replace('\\', "/")) {
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

/// True when the panicking call at `line_index` sits in a function whose docs declare a
/// `# Panics` section.
///
/// A panic in production code is acceptable only when failing closed IS the contract and
/// the contract is written down. Vyre's panicking functions are infallible wrappers over
/// `try_*` twins, because the quiet alternative (return an empty match set, an empty
/// table, no offsets) reports a dirty input as clean and is a total recall-loss silent
/// fallback (Law 10). Rust already has one place to record that: the `# Panics` doc
/// section, which `clippy::missing_panics_doc` enforces the same way. The gate reads the
/// docs instead of keeping a second allowlist file that would drift out of date, and an
/// undocumented panic stays a release blocker.
fn has_documented_panic_contract(text: &str, line_index: usize) -> bool {
    let lines = text.lines().collect::<Vec<_>>();
    let Some(site) = lines.get(line_index) else {
        return false;
    };
    let site_indent = site.len() - site.trim_start().len();
    let mut cursor = line_index;
    while cursor > 0 {
        cursor -= 1;
        let line = lines[cursor];
        let trimmed = line.trim_start();
        // Only an enclosing item counts: a signature at or past the call's own indent
        // belongs to a sibling, not to the function the call sits in.
        if line.len() - trimmed.len() >= site_indent || !is_fn_signature_line(trimmed) {
            continue;
        }
        let mut doc = cursor;
        while doc > 0 {
            doc -= 1;
            let previous = lines[doc].trim();
            if previous.starts_with("///") || previous.starts_with("//!") {
                if previous.contains("# Panics") {
                    return true;
                }
                continue;
            }
            // Attributes and plain `//` notes sit between a doc block and its signature
            // (`// INTENTIONAL: ...` above `#[allow(clippy::expect_used)]` is the house
            // style for a deliberate panic), so walking up must step over them or the doc
            // block is never reached.
            if previous.is_empty() || previous.starts_with("#[") || previous.starts_with("//") {
                continue;
            }
            break;
        }
        return false;
    }
    false
}

/// True when `trimmed` opens a function signature, whatever the leading keywords.
fn is_fn_signature_line(trimmed: &str) -> bool {
    let mut rest = trimmed;
    loop {
        if rest.starts_with("fn ") {
            return true;
        }
        let Some((head, tail)) = rest.split_once(' ') else {
            return false;
        };
        let is_signature_keyword = head.starts_with("pub")
            || head.starts_with("extern")
            || head.starts_with('"')
            || matches!(head, "const" | "async" | "unsafe" | "default");
        if !is_signature_keyword {
            return false;
        }
        rest = tail.trim_start();
    }
}

fn scan_tooling_file(path: &Path, scanned_files: &mut usize, findings: &mut Vec<HygieneFinding>) {
    scan_command_file(path, "unreadable_tooling_file", scanned_files, findings);
}

fn scan_doc_file(path: &Path, scanned_files: &mut usize, findings: &mut Vec<HygieneFinding>) {
    scan_command_file(path, "unreadable_doc_file", scanned_files, findings);
}

fn scan_command_file(
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

fn push_walk_error(root: &Path, error: &walkdir::Error, findings: &mut Vec<HygieneFinding>) {
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
        test: None,
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
    if is_code_call_blocker(name)
        && (is_rust_doc_comment_line(trimmed) || pattern_only_inside_literal(pattern, line))
    {
        return false;
    }
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

fn is_rust_doc_comment_line(trimmed: &str) -> bool {
    trimmed.starts_with("///") || trimmed.starts_with("//!")
}

fn is_code_call_blocker(name: &str) -> bool {
    matches!(
        name,
        "panic_macro"
            | "unwrap_call"
            | "expect_call"
            | "todo_macro"
            | "unimplemented_macro"
            | "not_implemented_text"
    )
}

/// Whether a code-call pattern appears only inside string literals on `line`.
///
/// A gate that detects `todo!(` has to spell `todo!(` to detect it, and a
/// pattern table row reading `text: "todo!(",` is that spelling, not a stub.
/// The rule already exempted a doc comment for the same reason. This is the
/// other half: a string literal names a call, it does not make one. It applies
/// to the code-call family only, because the hidden-fallback family is meant to
/// read prose and printed excuses, where the literal IS the evidence.
fn pattern_only_inside_literal(pattern: &str, line: &str) -> bool {
    let masked = crate::gates::scan::mask_literals(line);
    line.contains(pattern) && !masked.contains(pattern)
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

/// The workspace commands a reader is told to run through the wrapper.
const RAW_CARGO_COMMANDS: [&str; 14] = [
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
];

/// Whether a comment tells a reader to run the command it names.
///
/// A comment that says what cargo does with a member, or which build a rule is
/// about, is a description: the sentence is true and there is nothing to fix in
/// it. A comment that tells a maintainer to run something is an instruction,
/// and an instruction in this workspace names the wrapper.
///
/// Two signals have to agree. The verb comes before the command, because the
/// command itself contains the word run and matching the whole line read every
/// sentence that mentioned `cargo run` as an order to run it. And the command
/// is delimited as code, because prose says a full cargo build while an
/// instruction quotes what to type: a first attempt on the verb alone read
/// `the gates that run a full cargo build` as an order.
fn comment_instructs_a_run(before_command: &str) -> bool {
    let quoted_as_code = before_command.ends_with('`')
        || before_command.ends_with('"')
        || before_command.ends_with("`./")
        || before_command.ends_with("\"./");
    if !quoted_as_code {
        return false;
    }
    let lower = before_command.to_ascii_lowercase();
    [
        "run ",
        "runs ",
        "running ",
        "invoke",
        "rebuild",
        "regenerate",
        "reproduce",
        "re-run",
        "rerun",
        "reverify",
        "re-verify",
        "via ",
        "with ",
    ]
    .iter()
    .any(|verb| lower.contains(verb))
}

/// Whether a line is a comment rather than code or an emitted string.
fn is_comment_line(trimmed: &str) -> bool {
    trimmed.starts_with("//") || trimmed.starts_with("* ") || trimmed == "*"
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
    let Some(offset) = RAW_CARGO_COMMANDS
        .iter()
        .filter_map(|needle| trimmed.find(needle))
        .min()
    else {
        return trimmed.starts_with("cargo +");
    };
    !is_comment_line(trimmed) || comment_instructs_a_run(&trimmed[..offset])
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

/// The files that own a hygiene rule and therefore spell what it forbids.
///
/// Two rows named generator scripts that were deleted with the ticket tree they
/// belonged to, so the list exempted files that do not exist. The test below
/// reads this array and requires every row to resolve, because an exemption that
/// names nothing reads as a decision while doing nothing.
const HYGIENE_RULE_SOURCES: [&str; 6] = [
    "xtask/src/gates/lint_hygiene.rs",
    "xtask/src/release/feature_matrix.rs",
    "xtask/src/gates/hygiene_matrix.rs",
    "xtask-evidence/src/release/backend_matrix.rs",
    "xtask-evidence/src/release/vyre_release_gate/mod.rs",
    "xtask-registry/src/release/optimization_matrix.rs",
];

/// The files that own the hidden-fallback rule and spell the prose it catches.
const HIDDEN_FALLBACK_GUARD_SOURCES: [&str; 7] = [
    "xtask/src/gates/gpu_loudness.rs",
    "vyre-lints/src/production_cpu_fallbacks.rs",
    "vyre-lints/src/gpu_skip_guards.rs",
    "vyre-lints/src/lib.rs",
    "vyre-lints/src/main.rs",
    "vyre-lints/tests/production_cpu_fallbacks.rs",
    "vyre-lints/tests/gpu_skip_guards.rs",
];

fn is_hygiene_rule_source(path: &Path) -> bool {
    let normalized = path.to_string_lossy().replace('\\', "/");
    HYGIENE_RULE_SOURCES
        .iter()
        .any(|suffix| normalized.ends_with(suffix))
}

fn is_hidden_fallback_guard_source(path: &Path) -> bool {
    let normalized = path.to_string_lossy().replace('\\', "/");
    HIDDEN_FALLBACK_GUARD_SOURCES
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

#[derive(Default)]
struct BraceDepthState {
    depth: usize,
    block_comment_depth: usize,
    raw_string_hashes: Option<usize>,
}

impl BraceDepthState {
    fn with_depth(depth: usize) -> Self {
        Self {
            depth,
            ..Self::default()
        }
    }

    fn update(&mut self, line: &str) {
        let bytes = line.as_bytes();
        let mut index = 0usize;
        let mut in_string = false;
        let mut in_char = false;
        let mut escaped = false;

        while index < bytes.len() {
            if let Some(hashes) = self.raw_string_hashes {
                if raw_string_end_at(bytes, index, hashes) {
                    self.raw_string_hashes = None;
                    index += hashes + 1;
                } else {
                    index += 1;
                }
                continue;
            }
            if self.block_comment_depth > 0 {
                if bytes[index..].starts_with(b"/*") {
                    self.block_comment_depth = self.block_comment_depth.saturating_add(1);
                    index += 2;
                } else if bytes[index..].starts_with(b"*/") {
                    self.block_comment_depth = self.block_comment_depth.saturating_sub(1);
                    index += 2;
                } else {
                    index += 1;
                }
                continue;
            }
            if in_string {
                match bytes[index] {
                    _ if escaped => escaped = false,
                    b'\\' => escaped = true,
                    b'"' => in_string = false,
                    _ => {}
                }
                index += 1;
                continue;
            }
            if in_char {
                match bytes[index] {
                    _ if escaped => escaped = false,
                    b'\\' => escaped = true,
                    b'\'' => in_char = false,
                    _ => {}
                }
                index += 1;
                continue;
            }

            if bytes[index..].starts_with(b"//") {
                break;
            }
            if bytes[index..].starts_with(b"/*") {
                self.block_comment_depth = 1;
                index += 2;
                continue;
            }
            if let Some((hashes, consumed)) = raw_string_start(bytes, index) {
                self.raw_string_hashes = Some(hashes);
                index += consumed;
                continue;
            }

            match bytes[index] {
                b'"' => in_string = true,
                b'\'' if bytes[index + 1..].contains(&b'\'') => in_char = true,
                b'{' => self.depth = self.depth.saturating_add(1),
                b'}' => self.depth = self.depth.saturating_sub(1),
                _ => {}
            }
            index += 1;
        }
    }
}

fn raw_string_start(bytes: &[u8], index: usize) -> Option<(usize, usize)> {
    let mut cursor = index;
    if bytes.get(cursor) == Some(&b'b') {
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b'r') {
        return None;
    }
    cursor += 1;
    let hash_start = cursor;
    while bytes.get(cursor) == Some(&b'#') {
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b'"') {
        return None;
    }
    let hashes = cursor - hash_start;
    Some((hashes, cursor - index + 1))
}

fn raw_string_end_at(bytes: &[u8], index: usize, hashes: usize) -> bool {
    bytes.get(index) == Some(&b'"')
        && bytes
            .get(index + 1..index + 1 + hashes)
            .is_some_and(|suffix| suffix.iter().all(|byte| *byte == b'#'))
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    crate::output_arg::read_text_bounded(path, MAX_HYGIENE_SCAN_FILE_BYTES, "hygiene scan")
}

fn update_brace_depth(current: usize, line: &str) -> usize {
    let mut state = BraceDepthState::with_depth(current);
    state.update(line);
    state.depth
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: `unbounded_read` matched `fs::read(` as a bare substring, so any
    /// type whose name ends in `fs` produced a permanent release blocker on
    /// correct code. `BufferRefs::read(count_buffer)` reads a GPU buffer
    /// reference: there is no file, no length, and nothing to bound, and the
    /// only way to clear the finding would have been to rename the type.
    #[test]
    fn a_method_on_a_type_ending_in_fs_is_not_a_filesystem_read() {
        assert!(!line_contains_read_call(
            "Node::IndirectDispatch { count_buffer, .. } => BufferRefs::read(count_buffer),"
        ));
        assert!(!line_contains_read_call("let refs = Refs::read(buffer);"));
    }

    /// The narrowing above must not stop the rule catching a real read.
    #[test]
    fn every_filesystem_read_spelling_is_still_a_read_call() {
        for line in [
            "let text = fs::read_to_string(path)?;",
            "let text = std::fs::read_to_string(path)?;",
            "let bytes = fs::read(path)?;",
            "let bytes = std::fs::read(path)?;",
            "file.read_to_end(&mut bytes)?;",
            "handle.read_to_string(&mut text)?;",
        ] {
            assert!(line_contains_read_call(line), "missed `{line}`");
        }
    }

    /// WHY: the wrapper rule read every line that spelled a cargo command, so a
    /// sentence describing what a build does was a finding a reader could only
    /// clear by describing the build less precisely. An instruction is what the
    /// rule is about, and the verb that makes it one comes before the command.
    #[test]
    fn a_cargo_command_is_a_finding_when_a_comment_tells_a_reader_to_run_it() {
        for instruction in [
            "//! Run it with `cargo run -p structure-gate`.",
            "/// Regenerate the table with `cargo test -p vyre-driver`.",
            "// rebuild it with `cargo build -p xtask`",
            "let usage = \"cargo xtask gate1\";",
            "println!(\"  cargo run -p {package} -- <subcommand>\");",
        ] {
            assert!(
                line_contains_raw_workspace_cargo(instruction),
                "missed the instruction `{instruction}`"
            );
        }
        for description in [
            "//! The gates that run a full cargo build of the workspace.",
            "//! `cargo check -p <member>` is what the plain default build gets.",
            "// A cargo test target that does not exist fails before it runs.",
            "//! `cargo check -p <member>` is what the plain default build gets.",
        ] {
            assert!(
                !line_contains_raw_workspace_cargo(description),
                "read the description `{description}` as an instruction"
            );
        }
    }

    /// WHY: widening the release scan to every xtask source made each gate that
    /// detects a stub report itself: the pattern table row `text: "todo!(",` is
    /// how the hot-path scan spells the thing it looks for. A string literal
    /// names a call and does not make one, which is the same reason a doc
    /// comment was already exempt. The call itself must still block.
    #[test]
    fn a_code_call_named_in_a_literal_is_not_a_call() {
        let rule_row = "        text: \"todo!(\",";
        assert!(
            !line_contains_blocked_pattern(
                Path::new("/w/xtask/src/gates/hot_path_scan.rs"),
                "todo_macro",
                "todo!(",
                rule_row,
                &rule_row.to_ascii_lowercase()
            ),
            "Fix: a pattern table row is a rule definition, not a stub."
        );
        let call = "        todo!(\"finish the lowering\");";
        assert!(
            line_contains_blocked_pattern(
                Path::new("/w/vyre-driver/src/backend/dispatch.rs"),
                "todo_macro",
                "todo!(",
                call,
                &call.to_ascii_lowercase()
            ),
            "Fix: a real todo call must still block the release."
        );
    }

    /// WHY: two path lists exempt the files that own a rule from the rule. A row
    /// naming a file that no longer exists exempts nothing while reading as a
    /// decision, which is how an exemption list rots into a lie.
    #[test]
    fn every_exempted_rule_source_exists() {
        let root = crate::checkout::checkout_root();
        for candidate in HYGIENE_RULE_SOURCES
            .iter()
            .chain(HIDDEN_FALLBACK_GUARD_SOURCES.iter())
        {
            let path = root.join(candidate);
            assert!(
                path.is_file(),
                "Fix: exempted rule source `{candidate}` does not exist; delete the row."
            );
        }
    }

    /// WHY: `CHANGELOG.md` is generated from `release/changes`, and a released
    /// entry records what a version did instead of telling a reader what to
    /// run. Scanning it recorded eleven line numbers that every added fragment
    /// moved, so the evidence artifact went red for a document nobody edited,
    /// and the only place the text could be edited is a fragment that no longer
    /// exists. An authored document is still scanned.
    #[test]
    fn a_generated_release_history_is_recorded_not_instructed() {
        let tree = tempfile::TempDir::new().expect("Fix: create a fixture tree.");
        let bare = "Run `cargo test --workspace` to reproduce it.\n";
        for name in ["README.md", "CHANGELOG.md"] {
            fs::write(tree.path().join(name), bare).expect("Fix: write the fixture document.");
        }
        for (relative, body) in [
            ("CONTRIBUTING.md", "See the README.\n"),
            ("docs/testing/TESTING.toml", "suite = \"none\"\n"),
            ("conform/README.md", "See the README.\n"),
            ("vyre-bench/README.md", "See the README.\n"),
        ] {
            let path = tree.path().join(relative);
            fs::create_dir_all(path.parent().expect("Fix: a fixture path has a parent."))
                .expect("Fix: create the fixture directory.");
            fs::write(path, body).expect("Fix: write the fixture document.");
        }

        let mut scanned = 0usize;
        let mut findings = Vec::new();
        scan_release_docs(tree.path(), &mut scanned, &mut findings);

        let flagged = findings
            .iter()
            .map(|finding| finding.path.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            scanned, 5,
            "Fix: the five authored documents are the scanned set, got {scanned}."
        );
        assert!(
            flagged.iter().any(|path| path.ends_with("README.md")),
            "Fix: an authored document that spells a bare cargo command is still a finding, got {flagged:?}."
        );
        assert!(
            !flagged.iter().any(|path| path.ends_with("CHANGELOG.md")),
            "Fix: the generated release history is not scanned, got {flagged:?}."
        );
    }

    /// WHY: the release-tooling scan read `.sh`, `.yml` and `.yaml` only, so a
    /// rule that blocks a shell heredoc could be satisfied by moving the body
    /// into a `.py` beside it, where nothing looked. Seven gate scripts were
    /// rewritten that way, which would have moved 1100 lines of release tooling
    /// out of scan range in the same change that cleared the findings.
    #[test]
    fn python_release_tooling_is_scanned_like_shell_release_tooling() {
        let tree = tempfile::TempDir::new().expect("Fix: create a fixture tree.");
        let scripts = tree.path().join("scripts/lib");
        fs::create_dir_all(&scripts).expect("Fix: create the fixture scripts directory.");
        for (name, body) in [
            ("gate.sh", "#!/usr/bin/env bash\ncargo build --workspace\n"),
            (
                "gate.py",
                "import sys\nrun([\"x\"])  # cargo build --workspace\n",
            ),
        ] {
            fs::write(scripts.join(name), body).expect("Fix: write the fixture script.");
        }

        let mut scanned = 0usize;
        let mut findings = Vec::new();
        scan_release_tooling(tree.path(), &mut scanned, &mut findings);

        let scanned_extensions = findings
            .iter()
            .filter(|finding| finding.pattern == "raw_workspace_cargo")
            .filter_map(|finding| {
                Path::new(&finding.path)
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .map(ToString::to_string)
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(
            scanned_extensions,
            BTreeSet::from(["py".to_string(), "sh".to_string()]),
            "Fix: release tooling written in Python must be scanned like release tooling written in shell; findings={findings:?}"
        );
    }

    /// WHY: surface classification matched `/docs/` anywhere in the path, so an
    /// xtask module grouped under `xtask/src/docs/` was filed as documentation
    /// and lost its release-tooling thresholds. What decides the surface is
    /// which tree the file lives in, not which subdirectory of that tree.
    #[test]
    fn xtask_sources_are_release_tooling_whatever_group_holds_them() {
        for path in [
            "/w/xtask/src/main.rs",
            "/w/xtask/src/docs/catalog.rs",
            "/w/xtask/src/release/version_matrix.rs",
            "/w/xtask/src/bench/bench_release.rs",
            "/w/xtask/src/gates/gate1.rs",
        ] {
            assert_eq!(
                hygiene_surface_for_path(path),
                "release_tooling",
                "Fix: {path} is xtask source and must carry release-tooling thresholds."
            );
        }
        assert_eq!(
            hygiene_surface_for_path("/w/docs/optimization/PASSES.md"),
            "docs",
            "Fix: real documentation must still classify as docs."
        );
        assert_eq!(
            hygiene_surface_for_path("/w/vyre-libs/src/docs/loader.rs"),
            "docs",
            "Fix: only the xtask tree is reclassified; other trees keep the docs rule."
        );
    }

    /// WHY: the release hygiene scan named thirteen xtask command modules by
    /// hand and resolved each to a file. A command added beside them was never
    /// scanned, and a renamed module kept its row while resolving to nothing,
    /// which reads as coverage. The scan walks every xtask crate instead, so
    /// the contract is that a command module the tree holds is scanned, and a
    /// test source beside it is not.
    #[test]
    fn every_xtask_command_module_is_scanned_and_no_test_source_is() {
        let root = crate::checkout::checkout_root();
        let mut expected = 0usize;
        for source_root in xtask_source_roots(&root) {
            for entry in tree_walk::pruned(&source_root, BUILD_OUTPUT_AND_VCS).flatten() {
                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
                    continue;
                }
                if is_test_source_path(path)
                    || path.file_name().and_then(|name| name.to_str()) == Some("hygiene_matrix.rs")
                {
                    continue;
                }
                expected += 1;
            }
        }
        let mut scanned = 0usize;
        let mut findings = Vec::new();
        scan_release_xtask(&root, &mut scanned, &mut findings);
        assert!(
            expected > 0,
            "Fix: the xtask crates hold production source; the enumeration found none."
        );
        assert_eq!(
            scanned, expected,
            "Fix: the walk scanned {scanned} of the {expected} xtask production source file(s)."
        );
        for finding in &findings {
            assert!(
                !is_test_source_path(Path::new(&finding.path)),
                "Fix: `{}` is test source and must stay out of the production scan.",
                finding.path
            );
        }
        assert!(
            findings
                .iter()
                .all(|finding| finding.pattern != "unreadable_source_file"),
            "Fix: every file the walk reached must be readable: {:?}",
            findings
                .iter()
                .filter(|finding| finding.pattern == "unreadable_source_file")
                .map(|finding| finding.path.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn source_inspection_test_scanner_is_syntax_aware_and_fail_closed() {
        let forbidden = r#"
            #[cfg(test)]
            mod tests {
                #[test]
                fn freezes_helper_spelling() {
                    let source = include_str!("owner.rs");
                    assert!(source.contains("fn helper"));
                }
            }
        "#;
        let allowed = r###"
            #[cfg(test)]
            mod tests {
                #[test]
                fn verifies_product_text() {
                    let template = include_str!("launcher.rs.tmpl");
                    assert!(template.contains("pub fn launch"));
                }

                #[test]
                fn verifies_behavior() {
                    let summary = ResultSummary { source: "derived_pair_envelope" };
                    assert!(summary.source.contains("derived_pair_envelope"));
                }

                #[test]
                fn scanner_fixture_is_data() {
                    let forbidden = r##"include_str!("owner.rs").contains("fn helper")"##;
                    assert!(forbidden.contains("owner.rs"));
                }
            }
        "###;
        let mut findings = Vec::new();
        scan_source_inspection_tests(Path::new("driver/src/lib.rs"), forbidden, &mut findings);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].pattern, "source_inspection_test");
        assert!(findings[0].text.contains("freezes_helper_spelling"));

        findings.clear();
        scan_source_inspection_tests(Path::new("driver/src/lib.rs"), allowed, &mut findings);
        assert!(findings.is_empty());
    }

    /// A macro body reaches the callee graph through raw tokens, and the
    /// scanner used to render those tokens to a string and split on every
    /// non-identifier character. That split the CONTENTS of string literals,
    /// so `assert!(failures.iter().any(|f| f.contains("vyre-scan")))` claimed
    /// a call to a local `fn scan`, whose real body reads Rust source, and the
    /// pure test that owns that assertion was reported as a release blocker.
    /// Punctuation inside a literal is not a call.
    #[test]
    fn a_string_literal_inside_a_macro_is_not_a_call() {
        let source = r#"
            fn scan(root: &str) -> Vec<String> {
                let text = std::fs::read_to_string("owner.rs").unwrap();
                text.split('\n').map(ToOwned::to_owned).collect()
            }

            fn roster_failures(members: &[String]) -> Vec<String> {
                members.iter().filter(|m| m.starts_with("vyre")).cloned().collect()
            }

            #[cfg(test)]
            mod tests {
                #[test]
                fn a_product_crate_on_the_roster_is_rejected() {
                    let failures = roster_failures(&["vyre-scan".to_string()]);
                    assert!(failures.iter().any(|f| f.contains("vyre-scan")));
                }

                #[test]
                fn a_real_call_inside_a_macro_is_still_seen() {
                    assert!(scan("root").iter().any(|f| f.contains("owner")));
                }
            }
        "#;
        let mut findings = Vec::new();
        scan_source_inspection_tests(Path::new("gate/src/lib.rs"), source, &mut findings);
        let names = findings
            .iter()
            .map(|finding| finding.text.clone())
            .collect::<Vec<_>>();
        assert!(
            !names
                .iter()
                .any(|text| text.contains("a_product_crate_on_the_roster_is_rejected")),
            "a literal naming `vyre-scan` must not resolve to `fn scan`: {names:?}"
        );
        assert!(
            names
                .iter()
                .any(|text| text.contains("a_real_call_inside_a_macro_is_still_seen")),
            "a genuine call written inside a macro must still be followed: {names:?}"
        );
    }

    #[test]
    fn source_inspection_test_scanner_covers_integration_files_and_inline_test_modules() {
        let root = tempfile::tempdir().expect("Fix: scanner fixture root must be creatable.");
        let inline = root.path().join("driver/src/lib.rs");
        let integration = root.path().join("driver/tests/source_contract.rs");
        std::fs::create_dir_all(
            inline
                .parent()
                .expect("Fix: inline scanner fixture must have a parent."),
        )
        .expect("Fix: inline scanner fixture directory must be creatable.");
        std::fs::create_dir_all(
            integration
                .parent()
                .expect("Fix: integration scanner fixture must have a parent."),
        )
        .expect("Fix: integration scanner fixture directory must be creatable.");
        std::fs::write(
            &inline,
            r#"
                #[cfg(test)]
                mod tests {
                    #[test]
                    fn inline_contract() {
                        let source = include_str!("owner.rs");
                        assert!(source.contains("fn helper"));
                    }
                }
            "#,
        )
        .expect("Fix: inline scanner fixture must be writable.");
        std::fs::write(
            &integration,
            r#"
                #[test]
                fn integration_contract() {
                    let source = include_str!("../src/lib.rs");
                    assert!(source.contains("fn helper"));
                }
            "#,
        )
        .expect("Fix: integration scanner fixture must be writable.");

        let mut findings = Vec::new();
        let mut scanned_files = 0;
        scan_root(root.path(), &mut scanned_files, &mut findings);
        scan_source_inspection_test_files(root.path(), &mut scanned_files, &mut findings);

        let source_findings = findings
            .iter()
            .filter(|finding| finding.pattern == "source_inspection_test")
            .collect::<Vec<_>>();
        assert_eq!(
            source_findings.len(),
            2,
            "Fix: the repository scanner must reject source-shape tests in both inline modules and integration-test files."
        );
        assert!(source_findings
            .iter()
            .any(|finding| finding.path == inline.display().to_string()));
        assert!(source_findings
            .iter()
            .any(|finding| finding.path == integration.display().to_string()));
    }

    #[test]
    fn source_inspection_test_scanner_rejects_transitive_nested_and_aliased_walks() {
        let forbidden = r#"
            use std::path::{Path, PathBuf};

            #[test]
            fn freezes_architecture_spelling() {
                assert!(collect_sources(Path::new("src")).is_empty());
            }

            struct Helpers;

            impl Helpers {
                fn rust_files(root: &Path) -> Vec<PathBuf> {
                    collect_sources(root)
                }
            }

            fn collect_sources(root: &Path) -> Vec<PathBuf> {
                let mut files = Vec::new();
                for entry in std::fs::read_dir(root).unwrap() {
                    let path = entry.unwrap().path();
                    if path.extension().is_some_and(|extension| extension == "rs") {
                        let source = std::fs::read_to_string(&path).unwrap();
                        if source.contains("fn helper") {
                            files.push(path);
                        }
                    }
                }
                files
            }

            #[test]
            fn unrelated_behavior_remains_allowed() {
                assert_eq!(2 + 2, 4);
            }
        "#;
        let mut findings = Vec::new();
        scan_source_inspection_tests(
            Path::new("driver/tests/source_contract.rs"),
            forbidden,
            &mut findings,
        );

        assert_eq!(findings.len(), 1);
        assert!(findings[0].text.contains("freezes_architecture_spelling"));
    }

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
        let source = Path::new("tools/example-consumer/src/lib.rs");

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
        let source = Path::new("surge/surgec/src/scan/pipeline/parse_driver.rs");

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
        let source = Path::new("surge/surgec/src/cmd_scan.rs");

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

    /// A registry with the given `(file, test)` rows, each with a stated reason.
    fn structural_gates(rows: &[(&str, &str)]) -> StructuralGateArtifact {
        StructuralGateArtifact {
            schema_version: STRUCTURAL_GATE_SCHEMA_VERSION,
            source: STRUCTURAL_GATE_SOURCE,
            declarations: rows
                .iter()
                .map(|(file, test)| StructuralGateDeclaration {
                    file: (*file).to_string(),
                    test: (*test).to_string(),
                    reason: "no run-time witness".to_string(),
                })
                .collect(),
            blockers: Vec::new(),
        }
    }

    #[test]
    fn hygiene_classifier_separates_test_from_release_blocker() {
        let hot_paths = std::collections::BTreeSet::new();
        let findings = vec![
            HygieneFinding {
                path: "vyre-driver/src/pipeline/mod.rs".to_string(),
                line: 10,
                pattern: "unbounded_read",
                text: "std::fs::read(path)?".to_string(),
                test: None,
            },
            HygieneFinding {
                path: "vyre-driver/tests/pipeline_contracts.rs".to_string(),
                line: 20,
                pattern: "test_ignored",
                text: "#[ignore]".to_string(),
                test: None,
            },
        ];

        let classes = classify_findings(
            Path::new("."),
            &findings,
            &hot_paths,
            &structural_gates(&[]),
        );

        assert_eq!(classes[0].surface, "production");
        assert_eq!(classes[0].risk, "release_blocker");
        assert!(classes[0].release_blocker);
        assert_eq!(classes[1].surface, "test");
        assert_eq!(classes[1].risk, "test_hygiene");
        assert!(!classes[1].release_blocker);
    }

    #[test]
    fn undeclared_source_inspection_tests_are_release_blockers() {
        let findings = vec![HygieneFinding {
            path: "driver/tests/source_contracts.rs".to_string(),
            line: 7,
            pattern: "source_inspection_test",
            text: "test inspects Rust source text".to_string(),
            test: Some("every_module_is_reachable".to_string()),
        }];

        let classes = classify_findings(
            Path::new("."),
            &findings,
            &std::collections::BTreeSet::new(),
            &structural_gates(&[]),
        );

        assert_eq!(classes[0].surface, "test");
        assert_eq!(classes[0].risk, "release_blocker");
        assert!(classes[0].release_blocker);
    }

    /// A declared gate is informational; its neighbour in the same file is not.
    ///
    /// Keying on the file alone would let one reviewed declaration exempt every
    /// later source-inspecting test added beside it, which is the cost the
    /// declaration exists to charge.
    #[test]
    fn only_the_declared_source_inspection_test_is_informational() {
        let findings = vec![
            HygieneFinding {
                path: "/repo/driver/tests/source_contracts.rs".to_string(),
                line: 7,
                pattern: "source_inspection_test",
                text: "declared".to_string(),
                test: Some("no_other_file_calls_the_owner".to_string()),
            },
            HygieneFinding {
                path: "/repo/driver/tests/source_contracts.rs".to_string(),
                line: 40,
                pattern: "source_inspection_test",
                text: "undeclared".to_string(),
                test: Some("added_later_without_a_row".to_string()),
            },
        ];

        let classes = classify_findings(
            Path::new("/repo"),
            &findings,
            &std::collections::BTreeSet::new(),
            &structural_gates(&[(
                "driver/tests/source_contracts.rs",
                "no_other_file_calls_the_owner",
            )]),
        );

        assert_eq!(
            classes[0].risk, "informational",
            "Fix: a reviewed row in {STRUCTURAL_GATE_SOURCE} must exempt the test it names"
        );
        assert!(!classes[0].release_blocker);
        assert_eq!(
            classes[1].risk, "release_blocker",
            "Fix: a source-inspecting test with no reviewed row must block the release"
        );
        assert!(classes[1].release_blocker);
    }

    /// A row the tree no longer backs is a blocker, not a silent no-op.
    #[test]
    fn stale_structural_gate_rows_block_the_release() {
        let findings = vec![HygieneFinding {
            path: "/repo/driver/tests/source_contracts.rs".to_string(),
            line: 7,
            pattern: "source_inspection_test",
            text: "declared".to_string(),
            test: Some("still_here".to_string()),
        }];
        let declarations = structural_gates(&[
            ("driver/tests/source_contracts.rs", "still_here"),
            ("driver/tests/source_contracts.rs", "renamed_away"),
            ("driver/tests/deleted_contracts.rs", "gone_with_the_file"),
        ])
        .declarations;

        let blockers = stale_declaration_blockers(Path::new("/repo"), &declarations, &findings);

        assert_eq!(
            blockers.len(),
            2,
            "Fix: a row whose test or file the tree no longer has must block the release; blockers={blockers:?}"
        );
        assert!(
            blockers[0].contains("renamed_away")
                && blockers[0].contains("no longer has a test by that name"),
            "{blockers:?}"
        );
        assert!(
            blockers[1].contains("deleted_contracts.rs")
                && blockers[1].contains("contains no source-inspecting test"),
            "{blockers:?}"
        );
    }

    #[test]
    fn cpu_parity_oracle_sources_are_test_hygiene_not_release_blockers() {
        let hot_paths = std::collections::BTreeSet::new();
        let findings = vec![HygieneFinding {
            path: "/repo/vyre-reference/src/ifds_cpu_oracle.rs".to_string(),
            line: 37,
            pattern: "panic_macro",
            text: "panic!(\"IFDS CPU oracle\")".to_string(),
            test: None,
        }];

        let classes = classify_findings(
            Path::new("."),
            &findings,
            &hot_paths,
            &structural_gates(&[]),
        );

        assert_eq!(classes[0].surface, "test");
        assert_eq!(classes[0].risk, "test_hygiene");
        assert!(!classes[0].release_blocker);
    }

    /// The dedicated test-support crate is test infrastructure even though its code lives under `src`.
    #[test]
    fn test_support_crate_findings_are_test_hygiene() {
        let hot_paths = std::collections::BTreeSet::new();
        let findings = vec![
            HygieneFinding {
                path: "vyre-test-support/src/consumer_boundary.rs".to_string(),
                line: 161,
                pattern: "panic_macro",
                text: "panic!(\"fixture contract failed\")".to_string(),
                test: None,
            },
            HygieneFinding {
                path: "/repo/vyre-test-support/src/monorepo.rs".to_string(),
                line: 66,
                pattern: "expect_call",
                text: ".expect(\"workspace root\")".to_string(),
                test: None,
            },
        ];

        let classes = classify_findings(
            Path::new("/repo"),
            &findings,
            &hot_paths,
            &structural_gates(&[]),
        );

        assert!(classes.iter().all(|class| class.surface == "test"));
        assert!(classes.iter().all(|class| class.risk == "test_hygiene"));
        assert!(classes.iter().all(|class| !class.release_blocker));
    }

    #[test]
    fn rust_doc_comment_call_examples_do_not_count_as_production_blockers() {
        assert!(!line_contains_blocked_pattern(
            Path::new("vyre-libs/src/lib.rs"),
            "unwrap_call",
            ".unwrap()",
            "//! let value = fallible().unwrap();",
            "//! let value = fallible().unwrap();",
        ));
    }

    /// Feature-gated test harness modules remain test infrastructure even when Cargo places them under `src`.
    #[test]
    fn feature_gated_test_harness_sources_are_test_hygiene() {
        assert_eq!(
            hygiene_surface_for_path("/repo/vyre-driver-cuda/src/test_harness/fake_backend.rs"),
            "test"
        );
    }

    #[test]
    fn fuzz_targets_are_test_surface_not_release_production() {
        assert_eq!(
            hygiene_surface_for_path("vyre-foundation/fuzz/fuzz_targets/reachability.rs"),
            "test"
        );
    }

    #[test]
    fn cfg_cpu_parity_attr_is_classified_as_non_release_item() {
        assert!(is_non_release_cfg_attr(
            "#[cfg(any(test, feature = \"cpu-parity\"))]"
        ));
        assert!(is_non_release_cfg_attr(
            "#[cfg(any(test, feature = \"legacy-infallible\"))]"
        ));
        assert!(!is_non_release_cfg_attr("#[cfg(feature = \"serde\")]"));
    }

    #[test]
    fn stacked_cfg_after_test_attr_still_counts_as_test_body() {
        let mut findings = Vec::new();
        let mut scanned_files = 0;
        let dir =
            std::env::temp_dir().join(format!("vyre-hygiene-stacked-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("test temp dir");
        let path = dir.join("stacked_test.rs");
        std::fs::write(
            &path,
            "#[test]\n#[cfg(feature = \"gpu\")]\nfn generated_e2e() {\n    fallible().expect(\"test-only assertion\");\n}\n",
        )
        .expect("write stacked test fixture");
        scan_file(&path, &mut scanned_files, &mut findings);
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
        assert_eq!(scanned_files, 1);
        assert!(
            findings.is_empty(),
            "stacked #[test] + #[cfg] function body must not be release hygiene"
        );
    }

    #[test]
    fn hygiene_classifier_marks_hot_path_debt_as_release_blocker() {
        let hot_paths = std::collections::BTreeSet::from([
            "vyre-runtime/src/resident_work_queue/ring.rs".to_string(),
        ]);
        let findings = vec![HygieneFinding {
            path: "vyre-runtime/src/resident_work_queue/ring.rs".to_string(),
            line: 12,
            pattern: "TODO",
            text: "// TODO: remove allocation".to_string(),
            test: None,
        }];

        let classes = classify_findings(
            Path::new("."),
            &findings,
            &hot_paths,
            &structural_gates(&[]),
        );

        assert!(classes[0].hot_path);
        assert_eq!(classes[0].risk, "release_blocker");
    }

    #[test]
    fn hidden_fallback_guard_source_is_identified_for_gpu_skip_phrases() {
        assert!(is_hidden_fallback_guard_source(Path::new(
            "vyre-lints/src/gpu_skip_guards.rs"
        )));
    }

    #[test]
    fn required_cargo_wrapper_is_tool_owned() {
        let workspace = tempfile::TempDir::new()
            .expect("Fix: create temp workspace for cargo wrapper hygiene test.");
        let vyre_root = workspace.path().join("vyre");
        fs::create_dir_all(&vyre_root)
            .expect("Fix: create temp vyre root for cargo wrapper hygiene test.");
        fs::write(vyre_root.join("cargo_full"), b"#!/usr/bin/env bash\n")
            .expect("Fix: write temp cargo_full wrapper for hygiene test.");

        let mut findings = Vec::new();
        check_required_cargo_wrappers(&vyre_root, &mut findings);

        assert!(
            findings.is_empty(),
            "Fix: Vyre release hygiene must require the tool-owned bounded cargo wrapper; findings={findings:?}"
        );
    }

    /// A production `panic!` whose enclosing function documents `# Panics` is a declared
    /// contract, not release debt.
    ///
    /// Vyre ships infallible wrappers over `try_*` twins because the quiet alternative
    /// (an empty match set, an empty table) reports a dirty input as clean (Law 10). The
    /// gate reads Rust's own `# Panics` section so there is no second allowlist to rot.
    #[test]
    fn documented_panic_contract_is_recognized() {
        let source = "\
/// Pack a haystack.
///
/// # Panics
/// Panics when the haystack exceeds the u32 ABI.
pub fn pack(haystack: &[u8]) -> Vec<u8> {
    panic!(\"nope\")
}
";
        let panic_line = source
            .lines()
            .position(|line| line.contains("panic!("))
            .expect("Fix: keep the panic site in the documented-contract fixture.");

        assert!(
            has_documented_panic_contract(source, panic_line),
            "Fix: a panic inside a function documenting `# Panics` must not be a release blocker."
        );
    }

    /// An undocumented production panic stays a release blocker.
    ///
    /// This is the whole point of reading the docs: if the contract is not written down,
    /// a caller cannot know the call can abort, and the panic is debt.
    #[test]
    fn undocumented_panic_is_not_a_contract() {
        let source = "\
/// Pack a haystack.
pub fn pack(haystack: &[u8]) -> Vec<u8> {
    panic!(\"nope\")
}
";
        let panic_line = source
            .lines()
            .position(|line| line.contains("panic!("))
            .expect("Fix: keep the panic site in the undocumented fixture.");

        assert!(
            !has_documented_panic_contract(source, panic_line),
            "Fix: an undocumented panic must remain a release blocker."
        );
    }

    /// Attributes and plain `//` notes between the doc block and the signature must not
    /// hide the contract.
    ///
    /// `// INTENTIONAL: ...` above `#[allow(clippy::expect_used)]` is the house style for
    /// a deliberate panic; a walk that stopped at the first non-doc line reported both
    /// `vyre-grammar-gen` DFA builders as blockers even though each documents `# Panics`.
    #[test]
    fn documented_contract_survives_attributes_and_plain_comments() {
        let source = "\
/// Build the lexer DFA.
///
/// # Panics
/// Panics when a compile-time pattern is invalid.
// INTENTIONAL: the pattern table is a constant; a failure is a broken build.
#[must_use]
#[allow(clippy::expect_used)]
pub fn build() -> Dfa {
    inner().expect(\"constant patterns must compile\")
}
";
        let site = source
            .lines()
            .position(|line| line.contains(".expect("))
            .expect("Fix: keep the expect site in the attribute fixture.");

        assert!(
            has_documented_panic_contract(source, site),
            "Fix: attributes and plain comments between docs and signature must not hide a `# Panics` contract."
        );
    }

    /// A `# Panics` section on a neighbouring function must not exempt an undocumented one.
    ///
    /// The walk back looks for the ENCLOSING signature. If it drifted past the function
    /// it started in, one documented panic anywhere in a file would silence the rest.
    #[test]
    fn documented_contract_does_not_leak_to_the_next_function() {
        let source = "\
/// Documented.
///
/// # Panics
/// Panics on bad input.
pub fn documented() {
    unreachable!()
}

pub fn undocumented() {
    panic!(\"nope\")
}
";
        let site = source
            .lines()
            .position(|line| line.contains("panic!("))
            .expect("Fix: keep the panic site in the leak fixture.");

        assert!(
            !has_documented_panic_contract(source, site),
            "Fix: a `# Panics` section on an earlier function must not exempt a later one."
        );
    }

    /// Braces inside string and character literals must not terminate a cfg(test) module early.
    ///
    /// The hygiene scan previously treated `split("}\n}")` in an inline test as two closing
    /// module braces, then reported the remaining test assertions as production panic blockers.
    #[test]
    fn brace_depth_ignores_literal_and_comment_delimiters() {
        assert_eq!(
            update_brace_depth(1, r#"let _ = source.split("}\n}").next();"#),
            1
        );
        assert_eq!(update_brace_depth(1, "let brace = '}';"), 1);
        assert_eq!(update_brace_depth(1, "call(); // }"), 1);
        assert_eq!(update_brace_depth(1, "if ready {"), 2);
        let mut raw = BraceDepthState::with_depth(1);
        raw.update("let artifact = br#\"{");
        raw.update("  \"nested\": {");
        raw.update("}\"#;");
        assert_eq!(raw.depth, 1);
    }

    /// Every spelling of a test cfg gates the item out of the production scan.
    ///
    /// The scan used to list four exact predicate spellings, so
    /// `#[cfg(all(test, feature = \"...\"))]` (how the regex scan suites gate themselves)
    /// was treated as production source and four `mod tests` blocks had their helpers
    /// reported as release blockers.
    #[test]
    fn every_test_cfg_spelling_is_non_release() {
        for attribute in [
            "#[cfg(test)]",
            "#[cfg(any(test, feature = \"cpu-parity\"))]",
            "#[cfg(all(test, feature = \"matching-regex\", feature = \"matching-dfa\"))]",
            "#[cfg(all(feature = \"matching-regex\", test))]",
        ] {
            assert!(
                is_non_release_cfg_attr(attribute),
                "Fix: `{attribute}` gates the item to test builds and must be excluded from the production hygiene scan."
            );
        }
    }

    /// `not(test)` and feature-only gates stay in the production scan.
    ///
    /// `#[cfg(not(test))]` is the OPPOSITE gate: that code ships. Treating it as test-only
    /// would blind the scan to exactly the production paths it exists to check.
    #[test]
    fn production_cfg_attributes_stay_in_scope() {
        for attribute in [
            "#[cfg(not(test))]",
            "#[cfg(feature = \"cuda\")]",
            "#[cfg(target_os = \"linux\")]",
            "#[derive(Debug)]",
        ] {
            assert!(
                !is_non_release_cfg_attr(attribute),
                "Fix: `{attribute}` does not gate the item to test builds and must stay in the production hygiene scan."
            );
        }
    }

    mod threshold_policy_contracts {
        use super::*;

        fn valid_row() -> ThresholdPolicyTomlRow {
            ThresholdPolicyTomlRow {
                id: "fixture".to_string(),
                path: "src/fixture.rs".to_string(),
                name: "FIXTURE_THRESHOLD".to_string(),
                unit: "items".to_string(),
                provenance: "measured fixture".to_string(),
                config_tier: "tier_a".to_string(),
                override_path: "compiled default -> tool.toml -> CLI override".to_string(),
                evidence_link: THRESHOLD_POLICY_ARTIFACT.to_string(),
                release_rule: "VX-475".to_string(),
            }
        }

        /// A blank required field must remain a release blocker so malformed policy data cannot pass through the rules-as-data gate.
        #[test]
        fn malformed_threshold_policy_rows_are_rejected() {
            let mut row = valid_row();
            row.unit.clear();
            let mut blockers = Vec::new();

            validate_threshold_policy_row(&row, &mut blockers);

            assert_eq!(
                blockers,
                vec![
                    "docs/optimization/THRESHOLD_POLICY.toml row `fixture` has blank unit. Fix: every threshold policy row must carry unit, provenance, tier, override, evidence, and VX ownership."
                ]
            );
        }

        /// A valid Tier A row must stay accepted so the malformed-fixture proof does not reject correctly governed operator thresholds.
        #[test]
        fn valid_threshold_policy_rows_are_accepted() {
            let mut blockers = Vec::new();

            validate_threshold_policy_row(&valid_row(), &mut blockers);

            assert_eq!(blockers, Vec::<String>::new());
        }

        /// An unknown tier must fail even when every descriptive field is present, because an unclassified threshold has no override contract.
        #[test]
        fn unknown_threshold_policy_tiers_are_rejected() {
            let mut row = valid_row();
            row.config_tier = "runtime".to_string();
            let mut blockers = Vec::new();

            validate_threshold_policy_row(&row, &mut blockers);

            assert_eq!(
                blockers,
                vec![
                    "docs/optimization/THRESHOLD_POLICY.toml row `fixture` uses config_tier `runtime`. Fix: use `tier_a`, `tier_b`, or `structural`."
                ]
            );
        }

        /// A structural threshold must reject operator overrides because changing a wire or ABI bound requires compatibility review.
        #[test]
        fn structural_threshold_policy_rejects_operator_overrides() {
            let mut row = valid_row();
            row.config_tier = "structural".to_string();
            let mut blockers = Vec::new();

            validate_threshold_policy_row(&row, &mut blockers);

            assert_eq!(
                blockers,
                vec![
                    "docs/optimization/THRESHOLD_POLICY.toml row `fixture` is structural but override_path does not say `not operator configurable`. Fix: separate wire/ABI bounds from runtime knobs."
                ]
            );
        }
    }

    /// WHY: the xtask tooling is split across `xtask` and the `xtask-*` crates
    /// that link vyre, and three separate rules key off that: the surface a file
    /// is classified under, the owner lane it is attributed to, and whether the
    /// generic source walk skips it. Each rule used to match the literal string
    /// `xtask`, so moving a module into a sibling crate reclassified it as
    /// production source under production thresholds. The crate list is read out
    /// of the workspace manifest at run time, so a fourth xtask crate turns this
    /// red instead of quietly inheriting the wrong rules.
    #[test]
    fn every_xtask_crate_carries_the_release_tooling_rules() {
        let manifest = fs::read_to_string(crate::checkout::checkout_root().join("Cargo.toml"))
            .expect("Fix: the workspace manifest must be readable");
        let crates: Vec<String> = manifest
            .lines()
            .filter_map(|line| {
                line.trim()
                    .strip_prefix('"')?
                    .strip_suffix("\",")
                    .map(str::to_string)
            })
            .filter(|member| member == "xtask" || member.starts_with("xtask-"))
            .collect();
        assert!(
            crates.len() >= 3,
            "expected the xtask family in the workspace roster, found {crates:?}"
        );
        for member in &crates {
            let source = format!("/w/{member}/src/gates/some_gate.rs");
            assert_eq!(
                hygiene_surface_for_path(&source),
                "release_tooling",
                "Fix: {source} is xtask source and must carry release-tooling thresholds."
            );
            assert_eq!(
                hygiene_owner_lane_for_path(&source),
                "testing_evidence",
                "Fix: {source} is xtask source and must be owned by testing_evidence."
            );
            assert!(
                is_xtask_tree_directory(member),
                "Fix: the generic source walk must skip `{member}`, which the \
                 release xtask scan already reads."
            );
        }
    }

    /// WHY: `is_xtask_source_path` gates the unbounded-read exemption, so a match
    /// that is too loose exempts production source from the read cap. A crate
    /// merely named `xtask-...` outside its `src` tree, and an unrelated crate
    /// whose path happens to contain the word, must both stay unexempt.
    #[test]
    fn the_xtask_source_match_does_not_leak_past_the_src_tree() {
        for exempt in [
            "/w/xtask/src/gates/a.rs",
            "/w/xtask-registry/src/gates/a.rs",
            "/w/xtask-evidence/src/release/a.rs",
        ] {
            assert!(is_xtask_source_path(exempt), "`{exempt}` must be exempt");
        }
        for not_exempt in [
            "/w/xtask-registry/tests/a.rs",
            "/w/xtask-registry/build.rs",
            "/w/vyre-libs/src/xtask-notes/a.rs",
            "/w/vyre-libs/src/a.rs",
        ] {
            assert!(
                !is_xtask_source_path(not_exempt),
                "`{not_exempt}` is not xtask source and must keep the read cap"
            );
        }
    }

    /// WHY: a panic that is neither documented nor on a hot path was bounded by
    /// nothing, and the answer has to fail in three directions or it is an
    /// allowlist. Over the ceiling is new debt, a crate with no row at all is a
    /// crate nobody decided about, and a ceiling left above a crate that reached
    /// zero is what covers the next panic added there. Improvement short of zero
    /// is a note, because a gate that fails on the improvement it asks for is a
    /// gate somebody switches off.
    #[test]
    fn the_panic_ceiling_fails_over_unrecorded_and_stale_and_only_notes_slack() {
        let (_directory, root) = crate::gates::fixture_checkout::checkout(&[(
            "docs/testing/PANIC_BUDGET.toml",
            "schema = 1\n\n[[crate_budget]]\nname = \"over\"\nceiling = 1\n\n[[crate_budget]]\nname = \"slack\"\nceiling = 3\n\n[[crate_budget]]\nname = \"stale\"\nceiling = 2\n",
        )]);
        let class = |path: &str, pattern: &'static str, surface: &'static str, blocker: bool| {
            HygieneFindingClass {
                path: root.join(path).display().to_string(),
                line: 1,
                pattern,
                owner_lane: "testing_evidence",
                surface,
                risk: if blocker {
                    "release_blocker"
                } else {
                    "informational"
                },
                hot_path: blocker,
                release_blocker: blocker,
            }
        };
        let classes = vec![
            class("over/src/a.rs", "panic_macro", "production", false),
            class("over/src/b.rs", "unwrap_call", "production", false),
            class("slack/src/a.rs", "expect_call", "release_tooling", false),
            class("unrecorded/src/a.rs", "expect_call", "production", false),
            // Neither of these is this ratchet's population: one is documented,
            // the other is already a release blocker and counted as one.
            class(
                "over/src/c.rs",
                "documented_panic_contract",
                "production",
                false,
            ),
            class("over/src/d.rs", "panic_macro", "production", true),
        ];

        let budget = collect_panic_budget(&root, &classes);
        let blockers = budget.blockers.join("\n");
        assert!(
            blockers.contains("over carries 2 undocumented panic(s)")
                && blockers.contains("ceiling of 1"),
            "over the ceiling has to block: {blockers}"
        );
        assert!(
            blockers.contains("unrecorded carries 1")
                && blockers.contains("records no ceiling for it"),
            "a crate with no row has to block: {blockers}"
        );
        assert!(
            blockers.contains("ceiling of 2 for stale, which now carries none"),
            "a ceiling above a crate that reached zero has to block: {blockers}"
        );
        assert_eq!(
            budget.notes.len(),
            1,
            "slack is one note, not a blocker: {:?}",
            budget.notes
        );
        assert!(
            budget.notes[0].contains("slack carries 1") && budget.notes[0].contains("to 1"),
            "the note carries the number to write: {:?}",
            budget.notes
        );
        assert_eq!(
            budget
                .rows
                .iter()
                .map(|row| (row.crate_name.as_str(), row.ceiling, row.measured))
                .collect::<Vec<_>>(),
            [("over", 1, 2), ("slack", 3, 1), ("stale", 2, 0)],
            "every recorded row carries what the tree measured against it"
        );
    }
}
