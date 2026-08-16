//! `xtask/script-assertion-ledger.md` held to the tree and to the registry.
//!
//! The ledger is the record of a port: for every script this repository ever
//! ran, what it asserted, who called it, which gate carries the assertion now,
//! and the injection that proved that gate red. A hand-maintained record of that
//! kind decays in one direction only. A script is deleted and its row still says
//! `present`, so the totals overstate the surface; a row claims a gate that was
//! renamed, so the port looks finished and the assertion is gone; a script is
//! added and no row mentions it, which is how an assertion re-enters the tree
//! outside the registry.
//!
//! So the derived half of the document is generated and the prose half is
//! checked. The totals and the two lists are rendered from the parse under
//! `--write`, which is why the prose above them states no number: a count with
//! two owners is a count that disagrees with itself. Every row is then held to
//! two facts the tree already knows. Whether the script is tracked decides
//! whether the row may claim `present`, and a row whose script has left the tree
//! must name a registered gate and the injection that proved it red, because
//! that is the whole of what a finished port is.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::Tree;
use crate::subcommands;

/// The document this gate owns.
const LEDGER: &str = "xtask/script-assertion-ledger.md";
/// The directory every row describes.
const DIRECTORY: &str = "scripts";
/// What a caller does about a stale generated section.
const FIX: &str = "run `xtask script-ledger --write`";
/// The heading that opens the generated section.
const TOTALS: &str = "## Totals";
/// The heading that closes the generated section.
const ROWS: &str = "## Rows";
/// The phrase a row about a script that is still tracked must carry, because the
/// only reason to keep a script is that it does something no rule can.
const OPERATOR_ACTION: &str = "operator action";
/// The phrase an injection record must carry. A gate nobody has seen fail is a
/// gate that has not been shown to judge anything.
const PROVED_RED: &str = "proved red";

/// One `### ` row of the ledger.
#[derive(Debug)]
struct Row {
    /// The script path the heading names, with any backticks removed.
    path: String,
    /// Whether the heading spelled the path in backticks.
    quoted: bool,
    /// The heading's line number, for a finding a reader can jump to.
    line: u32,
    /// The `Subject:` field, which claims the script is present or gone.
    subject: Option<String>,
    /// The `Invoked by:` field.
    invoked: Option<String>,
    /// The `Gate:` field, which names what carries the assertions now.
    gate: Option<String>,
    /// The `Injection:` field, which records the proof that gate can fail.
    injection: Option<String>,
    /// How many assertions the row lists.
    assertions: usize,
    /// How many findings the row lists.
    findings: usize,
}

impl Row {
    /// Whether the row claims its script has left the tree.
    fn departed(&self) -> bool {
        self.subject
            .as_deref()
            .is_some_and(|subject| subject.starts_with("gone"))
    }

    /// Whether the row claims its script is still in the tree.
    fn present(&self) -> bool {
        self.subject
            .as_deref()
            .is_some_and(|subject| subject.starts_with("present"))
    }

    /// Whether the row says nothing in the tree runs the script.
    fn uninvoked(&self) -> bool {
        self.invoked
            .as_deref()
            .is_some_and(|invoked| invoked.starts_with("nothing"))
    }
}

/// The ledger split into the part this gate generates and the parts it checks.
struct Ledger {
    /// Every line of the document.
    lines: Vec<String>,
    /// The rows, in document order.
    rows: Vec<Row>,
    /// Line index of the `## Totals` heading.
    totals: usize,
    /// Line index of the `## Rows` heading.
    rows_heading: usize,
    /// The paths the `### Left the tree` list names.
    left: Vec<String>,
    /// The paths the `### Nothing invokes it` list names.
    nothing: Vec<String>,
}

/// The ledger, or an error naming the structural heading it lacks.
fn parse(text: &str) -> Result<Ledger, GateError> {
    let lines: Vec<String> = text.lines().map(str::to_string).collect();
    let heading = |wanted: &str| {
        lines
            .iter()
            .position(|line| line.trim_end() == wanted)
            .ok_or_else(|| {
                GateError::new(
                    format!("`{LEDGER}` has no `{wanted}` heading"),
                    format!("restore the `{wanted}` heading; {FIX} owns what follows it"),
                )
            })
    };
    let totals = heading(TOTALS)?;
    let rows_heading = heading(ROWS)?;
    if rows_heading < totals {
        return Err(GateError::new(
            format!("`{LEDGER}` puts `{ROWS}` before `{TOTALS}`"),
            format!("keep the generated totals between the prose and `{ROWS}`"),
        ));
    }

    let mut rows: Vec<Row> = Vec::new();
    let mut left = Vec::new();
    let mut nothing = Vec::new();
    let mut list: Option<&'static str> = None;
    let mut bullets: Option<&'static str> = None;
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim_end();
        if let Some(rest) = trimmed.strip_prefix("### ") {
            bullets = None;
            match rest.trim() {
                "Left the tree" => list = Some("left"),
                "Nothing invokes it" => list = Some("nothing"),
                path => {
                    list = None;
                    let quoted = path.starts_with('`') && path.ends_with('`');
                    rows.push(Row {
                        path: path.trim_matches('`').to_string(),
                        quoted,
                        line: u32::try_from(index + 1).unwrap_or(u32::MAX),
                        subject: None,
                        invoked: None,
                        gate: None,
                        injection: None,
                        assertions: 0,
                        findings: 0,
                    });
                }
            }
            continue;
        }
        if trimmed.starts_with("## ") {
            list = None;
            bullets = None;
            continue;
        }
        if let Some(entry) = trimmed.strip_prefix("- ") {
            match list {
                Some("left") => left.push(entry.trim_matches('`').to_string()),
                Some("nothing") => nothing.push(entry.trim_matches('`').to_string()),
                _ => {
                    if let Some(row) = rows.last_mut() {
                        match bullets {
                            Some("assertions") => row.assertions += 1,
                            Some("findings") => row.findings += 1,
                            _ => {}
                        }
                    }
                }
            }
            continue;
        }
        match trimmed {
            "Assertions:" => bullets = Some("assertions"),
            "Findings:" => bullets = Some("findings"),
            "Exits nonzero on:" | "Reads:" => bullets = Some("other"),
            _ => {}
        }
        let Some(row) = rows.last_mut() else { continue };
        for (label, field) in [
            ("Subject: ", &mut row.subject),
            ("Invoked by: ", &mut row.invoked),
            ("Gate: ", &mut row.gate),
            ("Injection: ", &mut row.injection),
        ] {
            if let Some(rest) = trimmed.strip_prefix(label) {
                *field = Some(rest.trim().to_string());
            }
        }
    }
    Ok(Ledger {
        lines,
        rows,
        totals,
        rows_heading,
        left,
        nothing,
    })
}

/// Every gate name the registry accepts, spelled the way a row must spell it.
fn registered() -> BTreeSet<String> {
    subcommands::registry()
        .into_iter()
        .map(|gate| format!("`{}`", gate.name()))
        .collect()
}

/// Whether a field names at least one registered gate in backticks.
fn names_a_gate(field: &str, gates: &BTreeSet<String>) -> bool {
    gates.iter().any(|gate| field.contains(gate.as_str()))
}

/// The generated section: the totals and the two derived lists.
fn render(ledger: &Ledger, tracked: &BTreeSet<String>) -> String {
    let assertions: usize = ledger.rows.iter().map(|row| row.assertions).sum();
    let findings: usize = ledger.rows.iter().map(|row| row.findings).sum();
    let shell = tracked.iter().filter(|path| path.ends_with(".sh")).count();
    let python = tracked.iter().filter(|path| path.ends_with(".py")).count();
    let departed: Vec<&str> = ledger
        .rows
        .iter()
        .filter(|row| row.departed())
        .map(|row| row.path.as_str())
        .collect();
    let uninvoked: Vec<&str> = ledger
        .rows
        .iter()
        .filter(|row| row.uninvoked() && tracked.contains(&row.path))
        .map(|row| row.path.as_str())
        .collect();

    let mut text = String::new();
    text.push_str(TOTALS);
    text.push_str("\n\n");
    text.push_str(&format!(
        "- Rows: {}. Assertions: {assertions}. Findings: {findings}.\n",
        ledger.rows.len()
    ));
    text.push_str(&format!(
        "- Tracked files: {}: {shell} shell and {python} Python.\n",
        tracked.len()
    ));
    text.push_str(&format!(
        "- Rows whose script has left the tree: {}.\n",
        departed.len()
    ));
    text.push_str(&format!(
        "- Tracked files nothing invokes: {}.\n",
        uninvoked.len()
    ));
    text.push_str("\n### Left the tree\n\n");
    for path in &departed {
        text.push_str(&format!("- {path}\n"));
    }
    text.push_str("\n### Nothing invokes it\n\n");
    for path in &uninvoked {
        text.push_str(&format!("- `{path}`\n"));
    }
    text.push('\n');
    text
}

/// The ledger, and the two lists and totals it derives from the tree.
pub struct ScriptLedger;

impl Gate for ScriptLedger {
    fn name(&self) -> &'static str {
        "script-ledger"
    }

    fn help(&self) -> &'static str {
        "Hold the script assertion ledger to the tracked scripts and the gate registry; --write regenerates its totals and lists"
    }

    fn generates(&self) -> bool {
        true
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let text = tree.read(LEDGER)?;
        let ledger = parse(&text)?;
        let gates = registered();
        let tracked: BTreeSet<String> = tree
            .paths()
            .iter()
            .filter_map(|path| path.to_str())
            .filter(|path| path.starts_with(&format!("{DIRECTORY}/")))
            .map(str::to_string)
            .collect();
        let mut report = Report::clean();

        let mut seen: BTreeMap<&str, u32> = BTreeMap::new();
        for row in &ledger.rows {
            if let Some(first) = seen.insert(row.path.as_str(), row.line) {
                report.find(Finding::at(
                    LEDGER,
                    row.line,
                    format!("`{}` already has a row at line {first}", row.path),
                    "keep one row per script and merge what the second one records",
                ));
            }
        }
        let mut ordered: Vec<&str> = seen.keys().copied().collect();
        ordered.sort_unstable();
        let written: Vec<&str> = ledger.rows.iter().map(|row| row.path.as_str()).collect();
        if written.len() == ordered.len() && written != ordered {
            report.find(Finding::in_file(
                LEDGER,
                "the rows are not in path order",
                "sort the rows by script path so a reader can find one",
            ));
        }

        for path in &tracked {
            if !seen.contains_key(path.as_str()) {
                report.find(Finding::in_file(
                    LEDGER,
                    format!("`{path}` is tracked and the ledger has no row for it"),
                    "record the script's assertions, callers and owning gate, or delete the script",
                ));
            }
        }

        for row in &ledger.rows {
            let present = tracked.contains(&row.path);
            let Some(subject) = row.subject.as_deref() else {
                report.find(Finding::at(
                    LEDGER,
                    row.line,
                    format!("the row for `{}` states no subject", row.path),
                    "open the row with `Subject: present.` or `Subject: gone: ...`",
                ));
                continue;
            };
            if !row.present() && !row.departed() {
                report.find(Finding::at(
                    LEDGER,
                    row.line,
                    format!(
                        "the subject of `{}` claims neither present nor gone: {subject}",
                        row.path
                    ),
                    "state whether the script is in the tree as the first word of the subject",
                ));
                continue;
            }
            if present != row.present() {
                let (claim, truth) = if present {
                    ("gone", "is tracked")
                } else {
                    ("present", "is not tracked")
                };
                report.find(Finding::at(
                    LEDGER,
                    row.line,
                    format!("the row for `{}` says {claim} and the script {truth}", row.path),
                    "state what the tree holds, and name the gate that took the assertions",
                ));
                continue;
            }
            if row.quoted != present {
                report.find(Finding::at(
                    LEDGER,
                    row.line,
                    format!(
                        "the heading for `{}` is {} and the script {}",
                        row.path,
                        if row.quoted { "quoted" } else { "unquoted" },
                        if present { "is tracked" } else { "is not tracked" }
                    ),
                    "quote the heading path of a tracked script and leave a departed one unquoted",
                ));
            }
            if row.invoked.is_none() {
                report.find(Finding::at(
                    LEDGER,
                    row.line,
                    format!("the row for `{}` names no caller", row.path),
                    "record `Invoked by:`, `nothing` included",
                ));
            }
            let Some(gate) = row.gate.as_deref() else {
                report.find(Finding::at(
                    LEDGER,
                    row.line,
                    format!("the row for `{}` names no gate", row.path),
                    "record `Gate:` with the registered gate that owns the assertions",
                ));
                continue;
            };
            if present {
                if !gate.contains(OPERATOR_ACTION) {
                    report.find(Finding::at(
                        LEDGER,
                        row.line,
                        format!(
                            "`{}` is still tracked and its row does not call it an {OPERATOR_ACTION}",
                            row.path
                        ),
                        "port the assertions into a gate and delete the script, or record why it is an operator action",
                    ));
                }
                continue;
            }
            if !names_a_gate(gate, &gates) {
                report.find(Finding::at(
                    LEDGER,
                    row.line,
                    format!(
                        "the row for `{}` names no registered gate in backticks",
                        row.path
                    ),
                    "name the gate that carries the assertions, spelled as the registry spells it",
                ));
            }
            match row.injection.as_deref() {
                None => report.find(Finding::at(
                    LEDGER,
                    row.line,
                    format!("the row for `{}` records no injection", row.path),
                    format!("record `Injection:` naming the change and the gate it {PROVED_RED}"),
                )),
                Some(injection) => {
                    if !injection.contains(PROVED_RED) || !names_a_gate(injection, &gates) {
                        report.find(Finding::at(
                            LEDGER,
                            row.line,
                            format!(
                                "the injection for `{}` does not say which gate it {PROVED_RED}",
                                row.path
                            ),
                            "record the change, the gate in backticks, and that the gate was proved red",
                        ));
                    }
                }
            }
        }

        // The counts live in the generated section alone. A number in the prose
        // above it is a second owner, and the second owner is the stale one.
        for (index, line) in ledger.lines[..ledger.totals].iter().enumerate() {
            if line.chars().any(char::is_numeric) {
                report.find(Finding::at(
                    LEDGER,
                    u32::try_from(index + 1).unwrap_or(u32::MAX),
                    "the prose above the totals states a number",
                    "keep every count in the generated totals",
                ));
            }
        }

        let expected = render(&ledger, &tracked);
        let current = format!(
            "{}\n",
            ledger.lines[ledger.totals..ledger.rows_heading].join("\n")
        );
        if ctx.write {
            if current != expected {
                let mut rebuilt = String::new();
                for line in &ledger.lines[..ledger.totals] {
                    rebuilt.push_str(line);
                    rebuilt.push('\n');
                }
                rebuilt.push_str(&expected);
                for line in &ledger.lines[ledger.rows_heading..] {
                    rebuilt.push_str(line);
                    rebuilt.push('\n');
                }
                fs::write(ctx.root.join(LEDGER), rebuilt).map_err(|error| {
                    GateError::new(
                        format!("cannot write `{LEDGER}`: {error}"),
                        "make the ledger writable",
                    )
                })?;
            }
            report.note("wrote the ledger totals and lists");
        } else {
            if current != expected {
                report.find(Finding::at(
                    LEDGER,
                    u32::try_from(ledger.totals + 1).unwrap_or(u32::MAX),
                    "the totals and lists do not match the rows and the tracked scripts",
                    FIX,
                ));
            }
            if ledger.left.len() != ledger.rows.iter().filter(|row| row.departed()).count() {
                report.find(Finding::in_file(
                    LEDGER,
                    "the departed list and the departed rows disagree",
                    FIX,
                ));
            }
            if ledger.nothing.iter().any(|path| !tracked.contains(path)) {
                report.find(Finding::in_file(
                    LEDGER,
                    "the uninvoked list names a script the tree no longer holds",
                    FIX,
                ));
            }
        }

        report.note(format!(
            "{} row(s), {} tracked script(s), {} departed",
            ledger.rows.len(),
            tracked.len(),
            ledger.rows.iter().filter(|row| row.departed()).count()
        ));
        Ok(report)
    }
}

/// WHY: the parser and the renderer are crate-private, and an integration test
/// can only reach them through the live ledger, which is one tree and therefore
/// one case. These cover the shapes the document is not currently in: a row
/// whose subject contradicts the tree, a departed row with no injection, and a
/// generated section that has drifted from the rows.
#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "# Script assertion ledger\n\nProse with no count.\n\n## Totals\n\n- Rows: 2. Assertions: 1. Findings: 0.\n- Tracked files: 1: 1 shell and 0 Python.\n- Rows whose script has left the tree: 1.\n- Tracked files nothing invokes: 1.\n\n### Left the tree\n\n- scripts/gone.sh\n\n### Nothing invokes it\n\n- `scripts/here.sh`\n\n## Rows\n\n### scripts/gone.sh\n\nSubject: gone: the script is not in the tree.\n\nInvoked by: nothing.\n\nGate: `file-size` owns it.\n\nInjection: raised a file over the cap; `file-size` proved red.\n\nAssertions:\n\n- one thing\n\n### `scripts/here.sh`\n\nSubject: present.\n\nInvoked by: nothing.\n\nGate: an operator action against a remote host.\n";

    #[test]
    fn a_row_reports_its_fields_its_bullets_and_its_heading_spelling() {
        let ledger = parse(SAMPLE).expect("the sample parses");
        assert_eq!(ledger.rows.len(), 2);
        assert!(ledger.rows[0].departed());
        assert!(!ledger.rows[0].quoted);
        assert_eq!(ledger.rows[0].assertions, 1);
        assert_eq!(ledger.rows[0].findings, 0);
        assert!(ledger.rows[1].present());
        assert!(ledger.rows[1].quoted);
        assert!(ledger.rows[1].uninvoked());
        assert_eq!(ledger.left, vec!["scripts/gone.sh".to_string()]);
        assert_eq!(ledger.nothing, vec!["scripts/here.sh".to_string()]);
    }

    #[test]
    fn the_rendered_totals_match_a_ledger_that_agrees_with_the_tree() {
        let ledger = parse(SAMPLE).expect("the sample parses");
        let tracked = BTreeSet::from(["scripts/here.sh".to_string()]);
        let generated = render(&ledger, &tracked);
        let current = format!(
            "{}\n",
            ledger.lines[ledger.totals..ledger.rows_heading].join("\n")
        );
        assert_eq!(generated, current);
    }

    #[test]
    fn a_row_count_that_moves_moves_the_rendered_totals() {
        let extended = format!(
            "{SAMPLE}\n### scripts/second.sh\n\nSubject: gone: the script is not in the tree.\n\nInvoked by: nothing.\n\nGate: `file-size` owns it.\n\nInjection: raised a file over the cap; `file-size` proved red.\n"
        );
        let ledger = parse(&extended).expect("the extended sample parses");
        let tracked = BTreeSet::from(["scripts/here.sh".to_string()]);
        let generated = render(&ledger, &tracked);
        assert!(generated.contains("- Rows: 3."), "got {generated}");
        assert!(
            generated.contains("- Rows whose script has left the tree: 2."),
            "got {generated}"
        );
        assert!(generated.contains("- scripts/second.sh"), "got {generated}");
    }

    #[test]
    fn only_a_registered_gate_name_in_backticks_counts_as_a_named_gate() {
        let gates = BTreeSet::from(["`file-size`".to_string()]);
        assert!(names_a_gate("Gate: `file-size` owns it.", &gates));
        assert!(!names_a_gate("Gate: file-size owns it.", &gates));
        assert!(!names_a_gate("Gate: `bench-retired` owns it.", &gates));
    }

    #[test]
    fn a_ledger_missing_a_structural_heading_is_an_error_not_a_clean_run() {
        let Err(error) = parse("# Script assertion ledger\n\n## Rows\n\n### scripts/a.sh\n") else {
            panic!("a ledger with no totals cannot be judged");
        };
        assert!(error.to_string().contains("## Totals"), "got {error}");
    }
}
