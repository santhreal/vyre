//! The gate contract every registered check is written against.
//!
//! There used to be five kinds of registered subcommand, and the kind decided
//! whether a check could fail a build, whether it had a baseline, whether the
//! sweep saw it, and whether anyone ran it at all. Three gates stayed red for a
//! fortnight behind a per-row exemption. There is one kind now: a gate reads the
//! tree, returns what it found, and the runner decides what that means.
//!
//! A gate never prints, never exits, and never writes unless it owns a
//! generated artifact and the caller passed `--write`. Rendering happens once,
//! in the runner, so every gate reports identically.

use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// One thing a gate found wrong. `fix` states the corrective action, because a
/// finding a reader cannot act on is a complaint.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Finding {
    /// File the finding is about, relative to the checkout root.
    pub file: Option<PathBuf>,
    /// Line within `file`, when the gate knows one.
    pub line: Option<u32>,
    /// What is wrong, in one sentence.
    pub message: String,
    /// The corrective action, in one sentence.
    pub fix: String,
}

impl Finding {
    /// A finding with no source location, for a fact about the tree as a whole.
    pub fn new(message: impl Into<String>, fix: impl Into<String>) -> Self {
        Self {
            file: None,
            line: None,
            message: message.into(),
            fix: fix.into(),
        }
    }

    /// A finding about one file.
    pub fn in_file(
        file: impl Into<PathBuf>,
        message: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self {
            file: Some(file.into()),
            line: None,
            message: message.into(),
            fix: fix.into(),
        }
    }

    /// A finding about one line of one file.
    pub fn at(
        file: impl Into<PathBuf>,
        line: u32,
        message: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self {
            file: Some(file.into()),
            line: Some(line),
            message: message.into(),
            fix: fix.into(),
        }
    }

    /// The file this finding names, displayable, empty when it names none.
    ///
    /// A finding with no file is about the tree rather than a place in it, and
    /// every reader of the field wanted one string covering both cases.
    #[must_use]
    pub fn named_file(&self) -> String {
        self.file
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default()
    }

    /// Rewrite an absolute path as a path relative to the checkout root, so two
    /// checkouts of the same tree report the same finding text.
    #[must_use]
    pub fn relative_to(mut self, root: &Path) -> Self {
        if let Some(path) = self.file.take() {
            self.file = Some(
                path.strip_prefix(root)
                    .map_or_else(|_| path.clone(), Path::to_path_buf),
            );
        }
        self
    }

    /// Format multiple findings as newline-separated message strings.
    #[must_use]
    pub fn messages(findings: &[Self]) -> String {
        findings
            .iter()
            .map(|finding| finding.message.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Machine-checkable semantic distinction that justifies an exemption (Section 182.4.5).
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ExemptionDistinction {
    /// Explicit domain-owned pure-IR leaf with exhaustive backend and reference facets.
    DeclaredPureIrLeaf,
    /// Multi-phase decomposition internal marker.
    InternalPhaseMarker,
    /// Hardware intrinsic or substrate-level primitive.
    HardwareIntrinsic,
    /// Dedicated shared kernel substrate plumbing directory.
    KernelSubstratePlumbing,
}

/// A machine-checkable exemption of a single live subject (Section 182.4.5).
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SubjectExemption {
    /// Exact identity of the live subject being exempted.
    pub subject_identity: String,
    /// Closed-enum semantic distinction category.
    pub distinction: ExemptionDistinction,
    /// Machine-verifiable evidence or authority path/symbol.
    pub evidence_identity: String,
}

impl SubjectExemption {
    /// Create a new typed, machine-checkable subject exemption.
    #[must_use]
    pub fn new(
        subject_identity: impl Into<String>,
        distinction: ExemptionDistinction,
        evidence_identity: impl Into<String>,
    ) -> Self {
        Self {
            subject_identity: subject_identity.into(),
            distinction,
            evidence_identity: evidence_identity.into(),
        }
    }
}

/// The complete subject universe one gate judged.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Coverage {
    /// Stable name for the subject class.
    pub subject: String,
    /// Subjects derived from the authoritative registry or source tree.
    pub discovered: usize,
    /// Subjects actually judged.
    pub judged: usize,
    /// Authoritative subject identities discovered at runtime, against which exemptions must validate.
    #[serde(default)]
    pub discovered_identities: Vec<String>,
    /// Subjects excluded by a typed, live exemption (Section 182.4.5).
    #[serde(default)]
    pub exemptions: Vec<SubjectExemption>,
}

impl Coverage {
    /// Construct one complete coverage row with zero exemptions.
    #[must_use]
    pub fn complete(subject: impl Into<String>, discovered: usize) -> Self {
        Self {
            subject: subject.into(),
            discovered,
            judged: discovered,
            discovered_identities: Vec::new(),
            exemptions: Vec::new(),
        }
    }

    /// Construct one complete coverage row with explicit discovered identities and zero exemptions.
    #[must_use]
    pub fn complete_identities(
        subject: impl Into<String>,
        discovered_identities: Vec<String>,
    ) -> Self {
        let count = discovered_identities.len();
        Self {
            subject: subject.into(),
            discovered: count,
            judged: count,
            discovered_identities,
            exemptions: Vec::new(),
        }
    }

    /// Construct a coverage row with explicit discovered identities and typed exemptions (Section 182.4.5).
    #[must_use]
    pub fn with_discovered_and_exemptions(
        subject: impl Into<String>,
        discovered_identities: Vec<String>,
        judged: usize,
        exemptions: Vec<SubjectExemption>,
    ) -> Self {
        let discovered = discovered_identities.len();
        Self {
            subject: subject.into(),
            discovered,
            judged,
            discovered_identities,
            exemptions,
        }
    }

    /// Number of typed exemptions.
    #[must_use]
    pub fn exempted(&self) -> usize {
        self.exemptions.len()
    }

    /// Whether this row accounts for its entire discovered universe.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.discovered > 0 && self.judged + self.exemptions.len() == self.discovered
    }
}

/// What a gate produces. `findings` is the pinned number; `notes` is context
/// that must never be counted.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Report {
    /// Everything wrong the gate found. This count is what the baseline pins.
    pub findings: Vec<Finding>,
    /// Context the gate wants a reader to have, never counted and never pinned.
    pub notes: Vec<String>,
    /// Complete subject universes judged by this gate.
    #[serde(default)]
    pub coverage: Vec<Coverage>,
    /// Exact workspace-relative artifacts this execution owns.
    #[serde(default)]
    pub artifacts: Vec<PathBuf>,
}

impl Report {
    /// A report with nothing wrong and nothing to say.
    #[must_use]
    pub fn clean() -> Self {
        Self::default()
    }

    /// Format all findings in this report as newline-separated message strings.
    #[must_use]
    pub fn finding_messages(&self) -> String {
        Finding::messages(&self.findings)
    }

    /// A report of findings with no notes.
    #[must_use]
    pub fn with_findings(findings: Vec<Finding>) -> Self {
        Self {
            findings,
            notes: Vec::new(),
            coverage: Vec::new(),
            artifacts: Vec::new(),
        }
    }

    /// One finding per message, all sharing one corrective action.
    ///
    /// A gate whose analysis already yields a list of prose problems states the
    /// fix once for the class, which is how these gates printed it: one trailing
    /// `Fix:` sentence under a list. Attaching it to each finding is what makes a
    /// single finding actionable when the sweep prints it on its own.
    #[must_use]
    pub fn from_messages(messages: Vec<String>, fix: &str) -> Self {
        Self {
            findings: messages
                .into_iter()
                .map(|message| Finding::new(message, fix))
                .collect(),
            notes: Vec::new(),
            coverage: Vec::new(),
            artifacts: Vec::new(),
        }
    }

    /// Record context that must not count against the pin.
    pub fn note(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }

    /// Record one thing the gate found wrong.
    pub fn find(&mut self, finding: Finding) {
        self.findings.push(finding);
    }
    /// Record one complete subject universe.
    pub fn cover(&mut self, coverage: Coverage) {
        self.coverage.push(coverage);
    }

    /// Record one complete subject universe by subject name and discovered count.
    pub fn cover_complete(&mut self, subject: impl Into<String>, discovered: usize) {
        self.coverage.push(Coverage::complete(subject, discovered));
    }

    /// Record one artifact this execution produced.
    pub fn produced(&mut self, path: impl Into<PathBuf>) {
        self.artifacts.push(path.into());
    }

    /// Every execution-contract violation against the authoritative descriptor.
    #[must_use]
    pub fn contract_failures(&self, descriptor: &GateDescriptor) -> Vec<String> {
        let mut failures = self.coverage_failures();
        let mut declared: Vec<&str> = descriptor.artifacts.to_vec();
        declared.sort_unstable();
        let mut produced: Vec<&str> = self
            .artifacts
            .iter()
            .map(|path| path.to_str().unwrap_or_default())
            .collect();
        produced.sort_unstable();
        if declared != produced {
            failures.push(format!(
                "declared artifacts {declared:?} but produced {produced:?}"
            ));
        }
        failures
    }
    /// Every reason this report does not account for its complete subject set.
    #[must_use]
    pub fn coverage_failures(&self) -> Vec<String> {
        let mut failures = Vec::new();
        if self.coverage.is_empty() {
            failures.push("reported no subject coverage".to_string());
            return failures;
        }
        let mut subjects = std::collections::BTreeSet::new();
        for row in &self.coverage {
            if !subjects.insert(row.subject.as_str()) {
                failures.push(format!(
                    "reported duplicate coverage rows for `{}`",
                    row.subject
                ));
            }
            if row.discovered == 0 {
                failures.push(format!(
                    "discovered zero `{}` subjects; an empty universe cannot prove the rule",
                    row.subject
                ));
            } else if row.judged + row.exemptions.len() != row.discovered {
                failures.push(format!(
                    "accounted for {} judged and {} exempted `{}` subjects after discovering {}",
                    row.judged,
                    row.exemptions.len(),
                    row.subject,
                    row.discovered
                ));
            }
            if !row.discovered_identities.is_empty()
                && row.discovered != row.discovered_identities.len()
            {
                failures.push(format!(
                    "discovered count {} does not match discovered_identities count {} in `{}` (Section 182.4.5)",
                    row.discovered, row.discovered_identities.len(), row.subject
                ));
            }
            let mut seen_discovered = std::collections::BTreeSet::new();
            for id in &row.discovered_identities {
                if !seen_discovered.insert(id.as_str()) {
                    failures.push(format!(
                        "duplicate discovered identity `{id}` in `{}` (Section 182.4.5)",
                        row.subject
                    ));
                }
            }
            if !row.exemptions.is_empty() && row.discovered_identities.is_empty() {
                failures.push(format!(
                    "exemptions claimed in `{}` require a non-empty discovered identity universe (Section 182.4.5)",
                    row.subject
                ));
            }
            let mut seen_exemptions = std::collections::BTreeSet::new();
            for ex in &row.exemptions {
                if ex.subject_identity.trim().is_empty() {
                    failures.push(format!(
                        "exemption in `{}` has empty subject identity (Section 182.4.5)",
                        row.subject
                    ));
                } else if !seen_exemptions.insert(ex.subject_identity.as_str()) {
                    failures.push(format!(
                        "duplicate exemption `{}` in `{}` (Section 182.4.4)",
                        ex.subject_identity, row.subject
                    ));
                }
                // Live subject membership validation (Section 182.4.5)
                if !row
                    .discovered_identities
                    .iter()
                    .any(|id| id == &ex.subject_identity)
                {
                    failures.push(format!(
                        "exemption `{}` in `{}` does not name a live discovered subject in the universe (Section 182.4.5)",
                        ex.subject_identity, row.subject
                    ));
                }
                if ex.evidence_identity.trim().is_empty() {
                    failures.push(format!(
                        "exemption `{}` in `{}` has empty evidence identity; prose alone is not an exemption (Section 182.4.5)",
                        ex.subject_identity, row.subject
                    ));
                }
            }
        }
        failures
    }

    /// The pinned number.
    #[must_use]
    pub fn count(&self) -> usize {
        self.findings.len()
    }

    /// The file each finding names, in the order the gate found them.
    #[must_use]
    pub fn named_files(&self) -> Vec<String> {
        self.findings.iter().map(Finding::named_file).collect()
    }
}

/// A gate that could not run at all, which is distinct from a gate that ran and
/// found things. An unreadable manifest is not a clean tree.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GateError {
    /// Why the gate could not judge the tree.
    pub message: String,
    /// What to do so it can.
    pub fix: String,
}

impl GateError {
    /// An error stating what stopped the gate and how to let it run.
    pub fn new(message: impl Into<String>, fix: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            fix: fix.into(),
        }
    }
}

impl fmt::Display for GateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}. Fix: {}", self.message, self.fix)
    }
}

impl std::error::Error for GateError {}

/// Everything a gate is allowed to know about how it was invoked.
pub struct GateCtx {
    /// Workspace root, resolved once by the runner.
    pub root: PathBuf,
    /// Caller flags after the subcommand name.
    pub args: Vec<String>,
    /// True when the caller passed --write and the gate owns a generated artifact.
    pub write: bool,
    /// Descriptor-owned identity of the gate currently executing.
    gate_name: Option<&'static str>,
}

impl GateCtx {
    /// Build a context from a resolved root and the caller's flags.
    #[must_use]
    pub fn new(root: PathBuf, args: Vec<String>) -> Self {
        let write = args.iter().any(|argument| argument == "--write");
        Self {
            root,
            args,
            write,
            gate_name: None,
        }
    }

    /// Bind this invocation to one authoritative gate descriptor.
    fn for_gate(&self, gate_name: &'static str, allows_write: bool) -> Self {
        Self {
            root: self.root.clone(),
            args: self.args.clone(),
            write: self.write && allows_write,
            gate_name: Some(gate_name),
        }
    }

    /// Stable identity of the gate currently executing.
    pub fn gate_name(&self) -> Result<&'static str, GateError> {
        self.gate_name.ok_or_else(|| {
            GateError::new(
                "gate behavior ran without a registered descriptor",
                "execute it through RegisteredGate so artifact ownership has one identity",
            )
        })
    }

    /// The value of `--flag VALUE`, or `None` when the caller did not pass it.
    #[must_use]
    pub fn flag(&self, flag: &str) -> Option<&str> {
        let at = self.args.iter().position(|argument| argument == flag)?;
        self.args.get(at + 1).map(String::as_str)
    }

    /// Whether the caller passed a bare flag.
    #[must_use]
    pub fn has(&self, flag: &str) -> bool {
        self.args.iter().any(|argument| argument == flag)
    }
}

/// Static contract that makes a gate discoverable and auditable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GateDescriptor {
    /// Stable gate name as typed on the command line.
    pub name: &'static str,
    /// One line describing what the gate judges, shown in help.
    pub help: &'static str,
    /// Owning package ("xtask", "xtask-registry", or "xtask-evidence").
    pub package: &'static str,
    /// Stable areas from which named subsets are derived.
    pub areas: &'static [&'static str],
    /// Authoritative class of subjects the gate judges.
    pub subject: &'static str,
    /// Exact workspace-relative artifacts this gate may rewrite.
    pub artifacts: &'static [&'static str],
    /// Prerequisites for executing the gate.
    pub prerequisites: &'static [&'static str],
    /// Test symbol that mutation-proves the invariant.
    pub proof: &'static str,
}
impl GateDescriptor {
    /// Whether this gate owns any generated artifact.
    #[must_use]
    pub fn generates(&self) -> bool {
        !self.artifacts.is_empty()
    }

    /// Every defect in a metadata row.
    #[must_use]
    pub fn failures(self) -> Vec<String> {
        let mut failures = Vec::new();
        if self.name.trim().is_empty() {
            failures.push("declares an empty gate name".to_string());
        }
        if self.help.trim().is_empty() {
            failures.push(format!("gate `{}` declares an empty help line", self.name));
        }
        if !["xtask", "xtask-registry", "xtask-evidence"].contains(&self.package) {
            failures.push(format!(
                "gate `{}` declares unknown owner package `{}`",
                self.name, self.package
            ));
        }
        if self.areas.is_empty() {
            failures.push(format!("gate `{}` belongs to no area", self.name));
        }
        if self.subject.trim().is_empty() || self.subject == "workspace contract subjects" {
            failures.push(format!(
                "gate `{}` declares no authoritative subject class",
                self.name
            ));
        }
        if self.proof.trim().is_empty() || self.proof == "definition-site mutation tests" {
            failures.push(format!(
                "gate `{}` declares no mutation-proof test",
                self.name
            ));
        }
        if self.proof.ends_with("::enforces_invariants") {
            failures.push(format!(
                "gate `{}` declares generic invented proof placeholder `{}`",
                self.name, self.proof
            ));
        }
        let expected_prefix = match self.package {
            "xtask" => Some("crate::"),
            "xtask-registry" => Some("xtask_registry::"),
            "xtask-evidence" => Some("xtask_evidence::"),
            _ => None,
        };
        if let Some(prefix) = expected_prefix {
            if !self.proof.starts_with(prefix) {
                failures.push(format!(
                    "gate `{}` owned by package `{}` declares proof `{}` missing required package prefix `{prefix}`",
                    self.name, self.package, self.proof
                ));
            }
        }
        failures
    }
}

/// Execution behavior for a gate implementation.
pub trait GateBehavior: Sync {
    /// The option lines this gate answers `--help` with.
    fn usage(&self) -> &'static [&'static str] {
        &[]
    }
    /// Judge the tree and report what is wrong with it.
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError>;
}

/// One registered gate pairing its authoritative descriptor with its execution behavior.
#[derive(Clone, Copy)]
pub struct RegisteredGate {
    /// Authoritative metadata descriptor.
    descriptor: &'static GateDescriptor,
    /// In-process execution behavior. Delegated gates execute through their descriptor package.
    behavior: Option<&'static dyn GateBehavior>,
}

impl RegisteredGate {
    /// Create an in-process registered gate pairing descriptor and behavior.
    #[must_use]
    pub const fn new(
        descriptor: &'static GateDescriptor,
        behavior: &'static dyn GateBehavior,
    ) -> Self {
        Self {
            descriptor,
            behavior: Some(behavior),
        }
    }

    /// Create a gate whose implementation executes in its owning package.
    #[must_use]
    pub const fn delegated(descriptor: &'static GateDescriptor) -> Self {
        Self {
            descriptor,
            behavior: None,
        }
    }

    /// Stable gate name as typed on the command line, from the authoritative descriptor.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.descriptor.name
    }

    /// One line describing what the gate judges, from the authoritative descriptor.
    #[must_use]
    pub fn help(&self) -> &'static str {
        self.descriptor.help
    }

    /// Owning package, from the authoritative descriptor.
    #[must_use]
    pub fn package(&self) -> &'static str {
        self.descriptor.package
    }

    /// Whether this gate owns any generated artifact, from the authoritative descriptor.
    #[must_use]
    pub fn generates(&self) -> bool {
        self.descriptor.generates()
    }

    /// Whether execution belongs to the descriptor's external owner package.
    #[must_use]
    pub const fn is_delegated(&self) -> bool {
        self.behavior.is_none()
    }

    /// The option lines this gate answers `--help` with.
    #[must_use]
    pub fn usage(&self) -> &'static [&'static str] {
        self.behavior.map_or(&[], |behavior| behavior.usage())
    }

    /// Judge the tree and report what is wrong with it.
    pub fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        match self.behavior {
            Some(behavior) => {
                let gate_ctx = ctx.for_gate(self.name(), self.generates());
                behavior.run(&gate_ctx)
            }
            None => crate::delegate::run_child_gate(self.package(), self.name(), ctx),
        }
    }
}

/// Whether the leading argument asks the gate for its usage.
///
/// Only the leading one does, so `--only --help` names a family called `--help`
/// and is the gate's to refuse.
#[must_use]
pub fn help_requested(args: &[String]) -> bool {
    matches!(args.first().map(String::as_str), Some("--help" | "-h"))
}

/// The report a gate answers `--help` with.
///
/// Usage is an answer, not an exit. A gate that printed its options on stdout
/// broke the report protocol its parent reads, and `bench-crossback --help`
/// read 35 measurements and reported a clean gate, which is the check running
/// on the caller who asked what the check takes.
#[must_use]
pub fn usage_report(gate: &RegisteredGate) -> Report {
    let mut report = Report::clean();
    let write = if gate.generates() { " [--write]" } else { "" };
    report.note(format!(
        "usage: ./cargo_full run -p xtask --bin xtask -- {}{write}",
        gate.name()
    ));
    report.note(gate.help());
    if gate.generates() {
        report.note(
            "--write regenerates the artifact this gate owns; without it the gate only judges",
        );
    }
    for line in gate.usage() {
        report.note(*line);
    }
    report
}

/// Every option a gate names in its help line and does not answer `--help` with.
///
/// The roster is the caller's gate table, so a gate added to it is judged
/// without anyone listing it here. A gate whose implementation lives in another
/// package is skipped: its usage lives with the implementation and is judged in
/// that package's own table.
#[must_use]
pub fn usage_gaps(gates: &[RegisteredGate]) -> Vec<String> {
    let mut gaps = Vec::new();
    for gate in gates {
        if gate.package() != "xtask" {
            continue;
        }
        let answered = gate.usage().join(" ");
        for flag in named_flags(gate.help()) {
            if flag == "--write" || answered.contains(&flag) {
                continue;
            }
            gaps.push(format!(
                "gate `{}` names `{flag}` in its help line and does not answer `--help` with it",
                gate.name()
            ));
        }
    }
    gaps
}
/// Every option a delegated gate names in its help line and does not answer `--help` with.
#[must_use]
pub fn usage_gaps_delegated(gates: &[(&'static str, &'static dyn GateBehavior)]) -> Vec<String> {
    let mut gaps = Vec::new();
    for (name, behavior) in gates {
        let Some(desc) = crate::gate_metadata::descriptor(name) else {
            continue;
        };
        let answered = behavior.usage().join(" ");
        for flag in named_flags(desc.help) {
            if flag == "--write" || answered.contains(&flag) {
                continue;
            }
            gaps.push(format!(
                "gate `{name}` names `{flag}` in its help line and does not answer `--help` with it"
            ));
        }
    }
    gaps
}
/// Every `--flag` token a line of prose names as this gate's own.
///
/// A backticked span is a command, and the flags in it belong to whatever the
/// span names: `launch-state` points the reader at
/// `vyre-release-gate --launch-complete`, which is another gate's option and
/// not one this gate reads.
fn named_flags(text: &str) -> Vec<String> {
    text.split('`')
        .skip(1)
        .step_by(2)
        .map(str::trim)
        .filter(|token| {
            token.starts_with("--") && token.len() > 2 && !token.contains(char::is_whitespace)
        })
        .map(|token| {
            token
                .trim_end_matches(|character: char| !character.is_ascii_alphanumeric())
                .to_string()
        })
        .collect()
}

/// Human rendering of one gate's report, in one place, so every gate reports
/// identically whatever crate implements it.
#[must_use]
pub fn render(name: &str, report: &Report) -> String {
    let mut text = String::new();
    for note in &report.notes {
        text.push_str(&format!("{name}: note: {note}\n"));
    }
    for finding in &report.findings {
        let location = match (&finding.file, finding.line) {
            (Some(file), Some(line)) => format!("{}:{line}", file.display()),
            (Some(file), None) => file.display().to_string(),
            (None, _) => name.to_string(),
        };
        text.push_str(&format!("{location}: {}\n", finding.message));
        text.push_str(&format!("  Fix: {}\n", finding.fix));
    }
    text.push_str(&format!("{name}: {} finding(s)\n", report.findings.len()));
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PanicsOnRun;

    impl GateBehavior for PanicsOnRun {
        fn usage(&self) -> &'static [&'static str] {
            &[]
        }

        fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
            panic!("Fix: a gate asked for its usage must not read the tree.");
        }
    }

    struct ReportsWrite;

    impl GateBehavior for ReportsWrite {
        fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
            let mut report = Report::clean();
            report.note(ctx.write.to_string());
            Ok(report)
        }
    }

    /// WHY: usage is an answer, not an exit and not a run. A gate asked for its
    /// options printed a table and reported clean, so the caller who asked what
    /// the check takes got the check instead. The answer names the command, what
    /// the gate judges, and every option it reads, and it counts nothing.
    #[test]
    fn a_gate_answers_help_without_reading_the_tree() {
        let desc = crate::gate_metadata::descriptor_by_name("crate-readmes");
        let gate = RegisteredGate::new(desc, &PanicsOnRun);
        let report = usage_report(&gate);
        assert!(report.findings.is_empty());
        assert_eq!(
            report.notes[0],
            "usage: ./cargo_full run -p xtask --bin xtask -- crate-readmes [--write]"
        );
        assert!(report
            .notes
            .iter()
            .any(|note| note.contains("--write regenerates")));
        assert!(usage_gaps(&[gate]).is_empty(), "{:?}", usage_gaps(&[gate]));
    }

    /// WHY: `--write` grants mutation authority only to a descriptor with exact
    /// artifact paths. A comparison-only gate must never receive write authority.
    #[test]
    fn only_artifact_owners_receive_write_authority() {
        let ctx = GateCtx::new(PathBuf::from("."), vec!["--write".to_string()]);
        let comparison = RegisteredGate::new(
            crate::gate_metadata::descriptor_by_name("cli-docs"),
            &ReportsWrite,
        );
        let generator = RegisteredGate::new(
            crate::gate_metadata::descriptor_by_name("crate-readmes"),
            &ReportsWrite,
        );

        assert_eq!(comparison.run(&ctx).unwrap().notes, ["false"]);
        assert_eq!(generator.run(&ctx).unwrap().notes, ["true"]);
    }
    /// WHY: only the leading flag is a usage request. `--only --help` names a
    /// case called `--help`, which is the gate's to refuse, and a gate that read
    /// the flag anywhere would answer usage instead of reporting the bad value.
    #[test]
    fn only_a_leading_flag_asks_for_usage() {
        let leading: Vec<String> = vec!["--help".to_string(), "--only".to_string()];
        let trailing: Vec<String> = vec!["--only".to_string(), "--help".to_string()];
        assert!(help_requested(&leading));
        assert!(help_requested(&["-h".to_string()]));
        assert!(!help_requested(&trailing));
        assert!(!help_requested(&[]));
    }

    /// WHY: an option a gate reads and never names is an option nobody can find.
    /// The rule reads the gate's own help line, so a gate that grows a flag and
    /// no usage entry goes red without anyone maintaining a list of gates.
    #[test]
    fn an_option_named_in_help_is_extracted_as_a_flag() {
        let flags = named_flags("Judge the tree; `--only` narrows it, `--write` records it");
        assert_eq!(flags, vec!["--only", "--write"]);
    }

    /// WHY: notes exist so a gate can report what it walked without inflating
    /// the pinned number. A composition audit enumerates thousands of registered
    /// operations as context; if a note ever counted, the pin would ratchet on
    /// the size of the tree instead of on its defects.
    #[test]
    fn notes_are_not_counted() {
        let mut report = Report::clean();
        report.note("walked 4000 ops");
        report.note("walked 12 crates");
        assert_eq!(report.count(), 0);
        report.find(Finding::new("a real defect", "fix it"));
        assert_eq!(report.count(), 1);
    }

    /// WHY: a finding with no corrective action is a complaint, and the render
    /// is the only place a reader sees one. Location, message and fix all have
    /// to survive rendering or a finding becomes unactionable in the sweep.
    #[test]
    fn every_finding_renders_its_location_message_and_fix() {
        let report = Report::with_findings(vec![
            Finding::at("src/a.rs", 12, "bad thing", "do the good thing"),
            Finding::in_file("src/b.rs", "file thing", "fix the file"),
            Finding::new("tree thing", "fix the tree"),
        ]);
        let text = render("demo", &report);
        assert!(text.contains("src/a.rs:12: bad thing\n  Fix: do the good thing\n"));
        assert!(text.contains("src/b.rs: file thing\n  Fix: fix the file\n"));
        assert!(text.contains("demo: tree thing\n  Fix: fix the tree\n"));
        assert!(text.ends_with("demo: 3 finding(s)\n"));
    }

    /// WHY: a clean finding count is meaningless when the gate discovered no
    /// subjects or silently skipped part of its universe. Every invalid shape
    /// must fail, while one complete non-empty row remains valid.
    #[test]
    fn coverage_requires_one_nonempty_closed_row_per_subject_class() {
        let mut report = Report::clean();
        assert_eq!(
            report.coverage_failures(),
            vec!["reported no subject coverage"]
        );

        // Zero discovered subjects fails closed
        report.cover(Coverage::complete("operations", 0));
        assert!(report.coverage_failures()[0].contains("zero `operations`"));

        // Partial accounting (judged + exempted < discovered) fails
        report.coverage.clear();
        report.cover(Coverage {
            subject: "operations".to_string(),
            discovered: 4,
            judged: 2,
            discovered_identities: vec!["ex1".into(), "op2".into(), "op3".into(), "op4".into()],
            exemptions: vec![SubjectExemption::new(
                "ex1",
                ExemptionDistinction::DeclaredPureIrLeaf,
                "proof1",
            )],
        });
        assert!(report.coverage_failures()[0].contains("accounted for 2 judged and 1 exempted"));

        // Over-accounting (judged + exempted > discovered) fails
        report.coverage.clear();
        report.cover(Coverage {
            subject: "operations".to_string(),
            discovered: 3,
            judged: 3,
            discovered_identities: vec!["ex1".into(), "op2".into(), "op3".into()],
            exemptions: vec![SubjectExemption::new(
                "ex1",
                ExemptionDistinction::DeclaredPureIrLeaf,
                "proof1",
            )],
        });
        assert!(report.coverage_failures()[0].contains("accounted for 3 judged and 1 exempted"));
        // Duplicate coverage row fails
        report.coverage.clear();
        report.cover(Coverage::complete("operations", 3));
        report.cover(Coverage::complete("operations", 3));
        assert!(report.coverage_failures()[0].contains("duplicate coverage"));

        // Newly added unjudged subject makes discovered > judged, failing gate
        report.coverage.clear();
        let discovered_ops = 5; // e.g. 5 operations discovered
        let judged_ops = 4; // but only 4 judged
        report.cover(Coverage {
            subject: "operations".to_string(),
            discovered: discovered_ops,
            judged: judged_ops,
            discovered_identities: Vec::new(),
            exemptions: Vec::new(),
        });
        assert!(!report.coverage_failures().is_empty());
        assert!(report.coverage_failures()[0].contains("accounted for 4"));

        // Exactly closed non-empty coverage succeeds
        report.coverage.clear();
        report.cover(Coverage::complete("operations", 5));
        assert!(report.coverage_failures().is_empty());
    }

    /// WHY: Section 182.4.5 requires exemptions to name a live subject and a machine-checkable semantic distinction.
    #[test]
    fn prose_only_exemption_without_evidence_is_rejected() {
        let mut report = Report::clean();
        let ex = SubjectExemption::new(
            "vyre-libs::math::foo",
            ExemptionDistinction::DeclaredPureIrLeaf,
            "",
        );
        report.cover(Coverage::with_discovered_and_exemptions(
            "operations",
            vec!["vyre-libs::math::foo".into()],
            0,
            vec![ex],
        ));
        let failures = report.coverage_failures();
        assert!(!failures.is_empty());
        assert!(failures
            .iter()
            .any(|f| f.contains("empty evidence identity")));
        assert!(failures.iter().any(|f| f.contains("Section 182.4.5")));
    }

    /// WHY: Section 182.4.5 requires exemptions to have non-empty subject identity.
    #[test]
    fn empty_subject_identity_exemption_is_rejected() {
        let mut report = Report::clean();
        let ex = SubjectExemption::new(
            "",
            ExemptionDistinction::DeclaredPureIrLeaf,
            "crates/math/src/leaf.rs",
        );
        report.cover(Coverage::with_discovered_and_exemptions(
            "operations",
            vec!["".into()],
            0,
            vec![ex],
        ));
        let failures = report.coverage_failures();
        assert!(!failures.is_empty());
        assert!(failures
            .iter()
            .any(|f| f.contains("empty subject identity")));
    }

    /// WHY: Section 182.4.4 requires rejecting duplicate exemption identities.
    #[test]
    fn duplicate_exemption_identity_is_rejected() {
        let mut report = Report::clean();
        let ex1 = SubjectExemption::new(
            "vyre-libs::math::foo",
            ExemptionDistinction::DeclaredPureIrLeaf,
            "proof::math",
        );
        let ex2 = SubjectExemption::new(
            "vyre-libs::math::foo",
            ExemptionDistinction::DeclaredPureIrLeaf,
            "proof::math",
        );
        report.cover(Coverage::with_discovered_and_exemptions(
            "operations",
            vec!["vyre-libs::math::foo".into(), "vyre-libs::math::bar".into()],
            0,
            vec![ex1, ex2],
        ));
        let failures = report.coverage_failures();
        assert!(!failures.is_empty());
        assert!(failures
            .iter()
            .any(|f| f.contains("duplicate exemption `vyre-libs::math::foo`")));
    }

    /// WHY: Section 182.4.5 allows typed exemptions with machine-checkable distinctions to close coverage.
    #[test]
    fn typed_exemption_with_closed_distinction_and_evidence_succeeds() {
        let mut report = Report::clean();
        let ex = SubjectExemption::new(
            "vyre-libs::math::leaf",
            ExemptionDistinction::DeclaredPureIrLeaf,
            "proof::leaf",
        );
        report.cover(Coverage::with_discovered_and_exemptions(
            "operations",
            vec![
                "vyre-libs::math::leaf".into(),
                "vyre-libs::math::other".into(),
            ],
            1,
            vec![ex],
        ));
        assert!(report.coverage_failures().is_empty());
    }
    /// WHY: Section 182.4.5 requires exemptions to name a live subject from the discovered universe.
    #[test]
    fn nonexistent_subject_identity_exemption_fails_live_validation() {
        let mut report = Report::clean();
        let ex = SubjectExemption::new(
            "nonexistent::op",
            ExemptionDistinction::DeclaredPureIrLeaf,
            "proof::leaf",
        );
        report.cover(Coverage::with_discovered_and_exemptions(
            "operations",
            vec!["op1".into(), "op2".into()],
            1,
            vec![ex],
        ));
        let failures = report.coverage_failures();
        assert!(!failures.is_empty());
        assert!(failures
            .iter()
            .any(|f| f.contains("does not name a live discovered subject")));
    }

    /// WHY: Section 182.4.5 allows valid live subject exemptions to close coverage.
    #[test]
    fn live_subject_exemption_matching_discovered_universe_succeeds() {
        let mut report = Report::clean();
        let ex = SubjectExemption::new(
            "op2",
            ExemptionDistinction::DeclaredPureIrLeaf,
            "proof::leaf",
        );
        report.cover(Coverage::with_discovered_and_exemptions(
            "operations",
            vec!["op1".into(), "op2".into()],
            1,
            vec![ex],
        ));
        assert!(report.coverage_failures().is_empty());
    }
    /// WHY: Section 182.4.5 requires exemptions to have a non-empty discovered identity universe.
    #[test]
    fn exemptions_with_empty_discovered_universe_fails() {
        let mut report = Report::clean();
        let ex = SubjectExemption::new(
            "op1",
            ExemptionDistinction::DeclaredPureIrLeaf,
            "proof::leaf",
        );
        report.cover(Coverage {
            subject: "operations".to_string(),
            discovered: 1,
            judged: 0,
            discovered_identities: Vec::new(),
            exemptions: vec![ex],
        });
        let failures = report.coverage_failures();
        assert!(!failures.is_empty());
        assert!(failures
            .iter()
            .any(|f| f.contains("require a non-empty discovered identity universe")));
    }

    /// WHY: Section 182.4.5 requires discovered identity universe to have unique identities.
    #[test]
    fn duplicate_discovered_identities_fails() {
        let mut report = Report::clean();
        report.cover(Coverage::complete_identities(
            "operations",
            vec!["op1".into(), "op1".into()],
        ));
        let failures = report.coverage_failures();
        assert!(!failures.is_empty());
        assert!(failures
            .iter()
            .any(|f| f.contains("duplicate discovered identity `op1`")));
    }

    /// WHY: Section 182.4.5 requires discovered count to match discovered_identities.len().
    #[test]
    fn discovered_count_mismatch_fails() {
        let mut report = Report::clean();
        report.cover(Coverage {
            subject: "operations".to_string(),
            discovered: 5,
            judged: 2,
            discovered_identities: vec!["op1".into(), "op2".into()],
            exemptions: Vec::new(),
        });
        let failures = report.coverage_failures();
        assert!(!failures.is_empty());
        assert!(failures
            .iter()
            .any(|f| f.contains("does not match discovered_identities count")));
    }
    /// WHY: artifact ownership is an execution contract, not permission alone.
    /// A declared generator that emits nothing and an undeclared output from a
    /// non-generator must both fail, while exact descriptor-owned outputs remain valid.
    #[test]
    fn report_artifacts_must_exactly_match_the_descriptor() {
        // Case 1: Declared generator gate ("catalog" -> ["docs/generated/catalog.toml"])
        let generator_desc = crate::gate_metadata::descriptor_by_name("catalog");
        let mut report = Report::clean();
        report.cover_complete("registered operations", 1);

        // Missing declared output from generator
        assert!(report
            .contract_failures(generator_desc)
            .iter()
            .any(|failure| failure.contains("produced []")));

        // Wrong output from generator
        report.produced("wrong/output.toml");
        assert!(report
            .contract_failures(generator_desc)
            .iter()
            .any(|failure| failure.contains("wrong/output.toml")));

        // Exact match for generator
        report.artifacts.clear();
        report.produced("docs/generated/catalog.toml");
        assert!(report.contract_failures(generator_desc).is_empty());

        // Case 2: Non-generator gate with no declared artifacts ("ci-matrix" -> [])
        let non_generator_desc = crate::gate_metadata::descriptor_by_name("ci-matrix");
        let mut non_gen_report = Report::clean();
        non_gen_report.cover_complete("ci workflows", 1);

        // Clean non-generator produces nothing -> exact match
        assert!(non_gen_report
            .contract_failures(non_generator_desc)
            .is_empty());

        // Undeclared output from non-generator -> fails
        non_gen_report.produced("unexpected/produced.toml");
        assert!(non_gen_report
            .contract_failures(non_generator_desc)
            .iter()
            .any(|failure| failure.contains("unexpected/produced.toml")));
    }

    /// WHY: the child of a delegated gate serialises its report and the parent
    /// renders it, so the two processes must agree on the shape. A field added
    /// on one side and not the other would silently drop findings.
    #[test]
    fn a_report_survives_the_process_boundary() {
        let mut report = Report::with_findings(vec![Finding::at("src/a.rs", 3, "m", "f")]);
        report.note("context");
        report.cover(Coverage::complete("fixture files", 1));
        let json = serde_json::to_string(&report).expect("a report serialises");
        assert_eq!(
            serde_json::from_str::<Report>(&json).expect("a report deserialises"),
            report
        );
    }

    /// WHY: `--write` is the only way a gate may touch the tree, and the runner
    /// derives it once so no gate has to parse it and none can disagree.
    #[test]
    fn write_is_derived_from_the_caller_flags() {
        let root = PathBuf::from("/tmp");
        assert!(!GateCtx::new(root.clone(), vec![]).write);
        assert!(!GateCtx::new(root.clone(), vec!["--check".to_string()]).write);
        assert!(GateCtx::new(root.clone(), vec!["--write".to_string()]).write);
        let ctx = GateCtx::new(root, vec!["--op-id".to_string(), "add.f32".to_string()]);
        assert_eq!(ctx.flag("--op-id"), Some("add.f32"));
        assert_eq!(ctx.flag("--missing"), None);
        assert!(ctx.has("--op-id"));
    }

    /// WHY: a gate reports paths so an operator can open them, and two checkouts
    /// of the same tree have to produce the same text or the pinned count is the
    /// only comparable part of the report.
    #[test]
    fn findings_report_paths_relative_to_the_checkout() {
        let root = PathBuf::from("/w/tree");
        let finding = Finding::at("/w/tree/src/a.rs", 1, "m", "f").relative_to(&root);
        assert_eq!(finding.file, Some(PathBuf::from("src/a.rs")));
        let outside = Finding::in_file("/other/a.rs", "m", "f").relative_to(&root);
        assert_eq!(outside.file, Some(PathBuf::from("/other/a.rs")));
    }
}
