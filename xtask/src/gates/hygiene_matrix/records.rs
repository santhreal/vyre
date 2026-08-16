use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub(crate) struct HygieneMatrixArtifact {
    pub(crate) schema_version: u32,
    pub(crate) scanned_roots: Vec<String>,
    pub(crate) scanned_files: usize,
    pub(crate) release_surface_coverage: ReleaseSurfaceCoverage,
    pub(crate) finding_summary: Vec<HygieneFindingSummary>,
    pub(crate) classification_summary: Vec<HygieneClassificationSummary>,
    pub(crate) intake_summary: Vec<HygieneIntakeSummary>,
    pub(crate) threshold_policy: ThresholdPolicyArtifact,
    pub(crate) structural_gates: StructuralGateArtifact,
    pub(crate) panic_budget: PanicBudgetArtifact,
    pub(crate) finding_classes: Vec<HygieneFindingClass>,
    pub(crate) release_blocker_count: usize,
    pub(crate) findings: Vec<HygieneFinding>,
    pub(crate) blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ReleaseSurfaceCoverage {
    pub(crate) vyre_workspace: bool,
    pub(crate) cuda_driver_crate: bool,
    pub(crate) wgpu_driver_crate: bool,
    pub(crate) release_scripts: bool,
    pub(crate) github_workflows: bool,
    pub(crate) branch_protection_controls: bool,
    pub(crate) resource_bound_patterns: Vec<&'static str>,
    pub(crate) hidden_fallback_patterns: Vec<&'static str>,
    pub(crate) release_tooling_patterns: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct HygieneFinding {
    pub(crate) path: String,
    pub(crate) line: usize,
    pub(crate) pattern: &'static str,
    pub(crate) text: String,
    /// The test this finding belongs to, for the patterns that judge a test
    /// rather than a line. The structural-gate registry is keyed on it, so the
    /// name has to reach classification rather than being formatted into
    /// `text` and parsed back out.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) test: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct HygieneFindingSummary {
    pub(crate) pattern: String,
    pub(crate) count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct HygieneFindingClass {
    pub(crate) path: String,
    pub(crate) line: usize,
    pub(crate) pattern: &'static str,
    pub(crate) owner_lane: &'static str,
    pub(crate) surface: &'static str,
    pub(crate) risk: &'static str,
    pub(crate) hot_path: bool,
    pub(crate) release_blocker: bool,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct HygieneClassificationSummary {
    pub(crate) owner_lane: &'static str,
    pub(crate) surface: &'static str,
    pub(crate) risk: &'static str,
    pub(crate) hot_path: bool,
    pub(crate) release_blocker: bool,
    pub(crate) count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct HygieneIntakeSummary {
    pub(crate) owner_lane: &'static str,
    pub(crate) surface: &'static str,
    pub(crate) risk: &'static str,
    pub(crate) hot_path: bool,
    pub(crate) pattern: &'static str,
    pub(crate) release_blocker: bool,
    pub(crate) count: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct HygieneIntakeArtifact {
    pub(crate) schema_version: u32,
    pub(crate) release_blocker_count: usize,
    pub(crate) intake_summary: Vec<HygieneIntakeSummary>,
    pub(crate) blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct HygieneScan {
    pub(crate) schema_version: u32,
    pub(crate) scan: String,
    pub(crate) findings: Vec<HygieneFinding>,
    pub(crate) release_blocking_findings: Vec<HygieneFindingClass>,
    pub(crate) blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ThresholdPolicyArtifact {
    pub(crate) schema_version: u32,
    pub(crate) source_manifest: &'static str,
    pub(crate) evidence_artifact: String,
    pub(crate) owner_lane: String,
    pub(crate) threshold_const_count: usize,
    pub(crate) registered_policy_count: usize,
    pub(crate) rows: Vec<ThresholdPolicyEvidenceRow>,
    pub(crate) findings: Vec<ThresholdPolicyFinding>,
    pub(crate) blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ThresholdPolicyEvidenceRow {
    pub(crate) id: String,
    pub(crate) path: String,
    pub(crate) line: usize,
    pub(crate) name: String,
    pub(crate) observed_value: String,
    pub(crate) unit: String,
    pub(crate) provenance: String,
    pub(crate) config_tier: String,
    pub(crate) override_path: String,
    pub(crate) evidence_link: String,
    pub(crate) release_rule: String,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ThresholdPolicyFinding {
    pub(crate) path: String,
    pub(crate) line: usize,
    pub(crate) name: String,
    pub(crate) finding: String,
    pub(crate) fix: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ThresholdPolicyDocument {
    pub(crate) schema_version: u32,
    pub(crate) owner_lane: String,
    pub(crate) evidence_artifact: String,
    #[serde(default)]
    pub(crate) threshold: Vec<ThresholdPolicyTomlRow>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ThresholdPolicyTomlRow {
    pub(crate) id: String,
    pub(crate) path: String,
    pub(crate) name: String,
    pub(crate) unit: String,
    pub(crate) provenance: String,
    pub(crate) config_tier: String,
    pub(crate) override_path: String,
    pub(crate) evidence_link: String,
    pub(crate) release_rule: String,
}

#[derive(Debug)]
pub(crate) struct ObservedThresholdConst {
    pub(crate) path: String,
    pub(crate) line: usize,
    pub(crate) name: String,
    pub(crate) value: String,
}

pub(crate) const BLOCKED_PATTERNS: &[(&str, &str)] = &[
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

pub(crate) const MAX_HYGIENE_SCAN_FILE_BYTES: u64 = 4_194_304;
pub(crate) const THRESHOLD_POLICY_SCHEMA_VERSION: u32 = 1;
pub(crate) const THRESHOLD_POLICY_SOURCE: &str = "docs/optimization/THRESHOLD_POLICY.toml";
pub(crate) const THRESHOLD_POLICY_ARTIFACT: &str = "release/evidence/hygiene/threshold-policy.json";
pub(crate) const THRESHOLD_POLICY_OWNER_LANE: &str = "testing_evidence";
pub(crate) const STRUCTURAL_GATE_SCHEMA_VERSION: u32 = 1;
pub(crate) const STRUCTURAL_GATE_SOURCE: &str = "docs/testing/STRUCTURAL_GATES.toml";
pub(crate) const PANIC_BUDGET_SCHEMA_VERSION: u32 = 1;
pub(crate) const PANIC_BUDGET_SOURCE: &str = "docs/testing/PANIC_BUDGET.toml";
pub(crate) const THRESHOLD_SUFFIXES: &[&str] = &[
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
pub(crate) struct StructuralGateArtifact {
    pub(crate) schema_version: u32,
    pub(crate) source: &'static str,
    pub(crate) declarations: Vec<StructuralGateDeclaration>,
    pub(crate) blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct StructuralGateDeclaration {
    pub(crate) file: String,
    pub(crate) test: String,
    pub(crate) reason: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StructuralGateDocument {
    pub(crate) schema: u32,
    #[serde(default)]
    pub(crate) gate: Vec<StructuralGateTomlRow>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StructuralGateTomlRow {
    pub(crate) file: String,
    pub(crate) test: String,
    pub(crate) reason: String,
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
pub(crate) struct PanicBudgetArtifact {
    pub(crate) schema_version: u32,
    pub(crate) source: &'static str,
    pub(crate) rows: Vec<PanicBudgetRow>,
    pub(crate) unrecorded: Vec<String>,
    pub(crate) notes: Vec<String>,
    pub(crate) blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct PanicBudgetRow {
    pub(crate) crate_name: String,
    pub(crate) ceiling: usize,
    pub(crate) measured: usize,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PanicBudgetDocument {
    pub(crate) schema: u32,
    #[serde(default)]
    pub(crate) crate_budget: Vec<PanicBudgetTomlRow>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PanicBudgetTomlRow {
    pub(crate) name: String,
    pub(crate) ceiling: usize,
}
