//! Ownership, membership and existence are read from content, never from a path.
//!
//! Git tracks files, not directories. Deleting a domain leaves its directory in
//! every checkout that pulled the deletion rather than cloning fresh, so an
//! empty shell survives with the name of code that lives somewhere else now.
//! Two attempts at the op matrix owner column got this wrong in a row: the first
//! read the frozen operation id, the second read `is_dir()` and named
//! `vyre-primitives/src/matching`, an empty shell containing an empty `ops/`,
//! the owner of eleven operations defined in `vyre-libs`.
//!
//! The rule is mechanical. A statement in the judging crates that builds a path
//! out of a name and then asks only whether that directory is there is a
//! finding. It has to read something: the directory's files, the declaration in
//! a `lib.rs`, the manifest roster, or the tracked-path listing. A statement
//! that reads alongside the test is a precondition on that read, and a path an
//! enumeration already returned is a fact about the walk, so neither is
//! reported.

use std::path::{Path, PathBuf};

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::{mask_literals, Tree};

/// Crates whose job is judging the tree.
///
/// A production crate may ask whether a file is there for its own reasons; these
/// are the ones whose output is a verdict about ownership or membership, so a
/// wrong answer here is a wrong answer about the repository.
const JUDGING_ROOTS: &[&str] = &["xtask", "xtask-registry", "structure-gate"];

/// The one path question a directory cannot answer.
///
/// A file's absence is a fact git records, so `is_file` and `exists` on a file
/// are honest reads. A directory is not tracked at all: every deletion that
/// arrives as a pull leaves it behind, so its presence says nothing about what
/// it holds.
const EXISTENCE_CALL: &str = ".is_dir()";

/// Reads that answer the same question from content.
///
/// Each one opens the thing named, enumerates it, or consults a roster the
/// repository declares. A statement performing one alongside the existence test
/// is reading, so the test guards the read instead of standing in for it.
const CONTENT_READS: &[&str] = &[
    "read_dir",
    "read_to_string",
    "read_source",
    "read_text",
    "read_toml",
    "carries_rust_source",
    "rust_sources",
    "source_files",
    "members(",
    "member_manifests(",
    "paths(",
    "hits(",
    ".read(",
    "walkdir",
    "WalkDir",
    "ls-files",
];

/// Holds every ownership answer in the judging crates to a content read.
pub struct PlacementPredicates;

impl Gate for PlacementPredicates {
    fn name(&self) -> &'static str {
        "placement-predicates"
    }

    fn help(&self) -> &'static str {
        "Hold every crate, module and domain placement answer to a content read instead of a directory existence test"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let sources = tree.rust(JUDGING_ROOTS)?;
        let mut scanned = 0usize;
        for path in &sources {
            let text = tree.read(path)?;
            scanned += 1;
            for finding in findings_in(path, &text) {
                report.find(finding);
            }
        }
        report.note(format!("{scanned} judging source file(s)"));
        Ok(report)
    }
}

/// Every placement answer in one file that reads a path instead of content.
fn findings_in(path: &Path, text: &str) -> Vec<Finding> {
    let mut findings = Vec::new();
    for function in functions(text) {
        for (number, line) in &function.body {
            if line.trim_start().starts_with("//") {
                continue;
            }
            let statement = mask_literals(line);
            if !statement.contains(EXISTENCE_CALL) {
                continue;
            }
            if CONTENT_READS.iter().any(|read| line.contains(*read)) {
                continue;
            }
            if !name_derived(&statement, &function.body) {
                continue;
            }
            findings.push(Finding::at(
                PathBuf::from(path),
                *number,
                format!(
                    "`{}` answers for a named path with `{EXISTENCE_CALL}` and reads nothing",
                    function.name
                ),
                "read what the directory holds, the declaration that names the module, or the roster the manifest declares; a directory survives the deletion of every file in it, so its presence answers nothing",
            ));
        }
    }
    findings
}

/// Whether the tested path was built by naming a crate, module or domain.
///
/// `root.join(member).is_dir()` asks whether a name has a directory, which is
/// the question this rule rejects. `entry.path().is_dir()` asks about a path an
/// enumeration already returned, which is a fact about that walk. A path bound
/// to a local one line earlier is the same question spelled over two lines, so
/// the binding is followed.
fn name_derived(statement: &str, body: &[(u32, String)]) -> bool {
    if statement.contains(".join(") {
        return true;
    }
    let Some(receiver) = receiver_of(statement) else {
        return false;
    };
    body.iter().any(|(_, line)| binds_join(line, &receiver))
}

/// The trailing identifier the existence call is made on, when there is one.
fn receiver_of(statement: &str) -> Option<String> {
    let before = statement.split(EXISTENCE_CALL).next()?;
    let name: String = before
        .chars()
        .rev()
        .take_while(|character| character.is_alphanumeric() || *character == '_')
        .collect();
    (!name.is_empty()).then(|| name.chars().rev().collect())
}

/// Whether one line binds `name` to a path built with `join`.
fn binds_join(line: &str, name: &str) -> bool {
    if !line.contains(".join(") {
        return false;
    }
    let Some(after) = line.trim_start().strip_prefix("let ") else {
        return false;
    };
    let Some(tail) = after.strip_prefix(name) else {
        return false;
    };
    !tail
        .chars()
        .next()
        .is_some_and(|character| character.is_alphanumeric() || character == '_')
}

/// One function body, numbered by the line each statement sits on.
struct Function {
    name: String,
    body: Vec<(u32, String)>,
}

/// Every function in one source file, with its body lines and their numbers.
///
/// The split is on `fn <name>` at the start of a line, up to the next one, which
/// is enough to attribute a statement to the function that makes it without
/// carrying a parser. A nested function is read as part of its parent, which
/// only widens the search for the binding, never narrows it.
fn functions(text: &str) -> Vec<Function> {
    let mut found: Vec<Function> = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let trimmed = line.trim_start();
        let Some(name) = signature_name(trimmed) else {
            if let Some(current) = found.last_mut() {
                let number = u32::try_from(index + 1).unwrap_or(u32::MAX);
                current.body.push((number, line.to_string()));
            }
            continue;
        };
        found.push(Function {
            name,
            body: Vec::new(),
        });
    }
    found
}

/// Name of the function a line declares, if it declares one.
fn signature_name(trimmed: &str) -> Option<String> {
    let after = trimmed
        .strip_prefix("fn ")
        .or_else(|| trimmed.strip_prefix("pub fn "))
        .or_else(|| trimmed.strip_prefix("pub(crate) fn "))
        .or_else(|| trimmed.strip_prefix("pub(super) fn "))
        .or_else(|| trimmed.strip_prefix("const fn "))
        .or_else(|| trimmed.strip_prefix("pub const fn "))?;
    let name: String = after
        .chars()
        .take_while(|character| character.is_alphanumeric() || *character == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: this is the exact shape that named an empty shell the owner of
    /// eleven operations. The function formats a crate path and asks only
    /// whether the directory is there.
    #[test]
    fn a_directory_existence_test_on_a_crate_path_is_reported() {
        let source = "fn owner_dir(crate_name: &str, domain: &str) -> String {\n    let path = format!(\"{crate_name}/src/{domain}\");\n    if Path::new(&path).is_dir() {\n        return path;\n    }\n    String::new()\n}\n";
        let findings = findings_in(Path::new("xtask/src/gates/example.rs"), source);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(
            findings[0].message.contains("owner_dir"),
            "{:?}",
            findings[0].message
        );
    }

    /// WHY: the fixed shape asks what the directory holds. Reporting it would
    /// make the gate fire on the correction, which is how a gate gets switched
    /// off.
    #[test]
    fn the_same_answer_read_from_content_is_accepted() {
        let source = "fn owner_dir(crate_name: &str, domain: &str) -> String {\n    let path = format!(\"{crate_name}/src/{domain}\");\n    if carries_rust_source(&path) {\n        return path;\n    }\n    String::new()\n}\n";
        assert!(findings_in(Path::new("xtask/src/gates/example.rs"), source).is_empty());
    }

    /// WHY: git tracks files, so a missing file is a fact the repository
    /// records and `exists` on one is a read of that fact. Reporting it would
    /// make the rule fire on every artifact check and get it switched off.
    #[test]
    fn a_file_existence_test_is_a_tracked_fact() {
        let source = "fn wrote_report(root: &Path) -> bool {\n    root.join(\"target/report.json\").exists()\n}\n";
        assert!(findings_in(Path::new("xtask/src/gates/example.rs"), source).is_empty());
    }

    /// WHY: a departed member leaves its directory behind, and that is the
    /// shape the rule exists to catch. The roster read is the fix, so the
    /// planted shell must be reported and the roster read must not.
    #[test]
    fn a_shell_left_by_a_departed_member_is_reported() {
        let planted = "fn member_lives(root: &Path, member: &str) -> bool {\n    root.join(member).is_dir()\n}\n";
        assert_eq!(
            findings_in(Path::new("structure-gate/src/example.rs"), planted).len(),
            1
        );
        let read = "fn member_lives(root: &Path, member: &str) -> bool {\n    carries_rust_source(&root.join(member))\n}\n";
        assert!(findings_in(Path::new("structure-gate/src/example.rs"), read).is_empty());
    }

    #[test]
    fn signature_name_reads_every_visibility() {
        assert_eq!(signature_name("fn plain()"), Some("plain".to_string()));
        assert_eq!(signature_name("pub fn open()"), Some("open".to_string()));
        assert_eq!(
            signature_name("pub(crate) fn inner()"),
            Some("inner".to_string())
        );
        assert_eq!(signature_name("let fn_like = 1;"), None);
    }
}
