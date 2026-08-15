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
}

/// What a gate produces. `findings` is the pinned number; `notes` is context
/// that must never be counted.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Report {
    /// Everything wrong the gate found. This count is what the baseline pins.
    pub findings: Vec<Finding>,
    /// Context the gate wants a reader to have, never counted and never pinned.
    pub notes: Vec<String>,
}

impl Report {
    /// A report with nothing wrong and nothing to say.
    #[must_use]
    pub fn clean() -> Self {
        Self::default()
    }

    /// A report of findings with no notes.
    #[must_use]
    pub fn with_findings(findings: Vec<Finding>) -> Self {
        Self {
            findings,
            notes: Vec::new(),
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

    /// The pinned number.
    #[must_use]
    pub fn count(&self) -> usize {
        self.findings.len()
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
}

impl GateCtx {
    /// Build a context from a resolved root and the caller's flags.
    #[must_use]
    pub fn new(root: PathBuf, args: Vec<String>) -> Self {
        let write = args.iter().any(|argument| argument == "--write");
        Self { root, args, write }
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

/// One registered check. Everything the runner knows how to run is one of these.
pub trait Gate: Sync {
    /// Name as typed on the command line, and the key of its baseline row.
    fn name(&self) -> &'static str;
    /// One line describing what the gate judges, shown in help.
    fn help(&self) -> &'static str;
    /// A gate that owns a generated artifact returns true and must honour ctx.write.
    fn generates(&self) -> bool {
        false
    }
    /// Judge the tree and report what is wrong with it.
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError>;
    /// Package whose binary runs this gate, or `None` when `xtask` runs it in
    /// process. The runner needs the answer to check that the package it
    /// delegates to implements every gate assigned to it.
    fn package(&self) -> Option<&'static str> {
        None
    }
}

/// A gate implemented in a crate that links vyre, run as a child process.
///
/// Delegation is a property of one gate and not a category. `xtask` links no
/// vyre crate so it cannot call these in process; it builds the owning package
/// on demand, runs it, and reads the `Report` the child serialises on stdout.
/// Everything else about such a gate is identical to a local one.
pub struct Delegated {
    /// Name as typed on the command line.
    pub name: &'static str,
    /// One line describing what the gate judges.
    pub help: &'static str,
    /// Package whose binary implements it.
    pub package: &'static str,
    /// Whether the gate owns a generated artifact.
    pub generates: bool,
}

impl Gate for Delegated {
    fn name(&self) -> &'static str {
        self.name
    }

    fn help(&self) -> &'static str {
        self.help
    }

    fn generates(&self) -> bool {
        self.generates
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        crate::delegate::run_child_gate(self.package, self.name, ctx)
    }

    fn package(&self) -> Option<&'static str> {
        Some(self.package)
    }
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
    text.push_str(&format!(
        "{name}: {} finding(s)\n",
        report.findings.len()
    ));
    text
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// WHY: the child of a delegated gate serialises its report and the parent
    /// renders it, so the two processes must agree on the shape. A field added
    /// on one side and not the other would silently drop findings.
    #[test]
    fn a_report_survives_the_process_boundary() {
        let mut report = Report::with_findings(vec![Finding::at("src/a.rs", 3, "m", "f")]);
        report.note("context");
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
        let ctx = GateCtx::new(
            root,
            vec!["--op-id".to_string(), "add.f32".to_string()],
        );
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
        let finding =
            Finding::at("/w/tree/src/a.rs", 1, "m", "f").relative_to(&root);
        assert_eq!(finding.file, Some(PathBuf::from("src/a.rs")));
        let outside = Finding::in_file("/other/a.rs", "m", "f").relative_to(&root);
        assert_eq!(outside.file, Some(PathBuf::from("/other/a.rs")));
    }
}
