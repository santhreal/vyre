//! What the workspace lint floor cannot say by itself.
//!
//! `[workspace.lints.rust]` denies `unsafe_code` and `missing_docs`, and every
//! member inherits it, so the set of files carrying an `allow` override is the
//! complete exception surface and rustc is the thing enforcing it. These gates
//! pin that surface, require a justification beside every unsafe block, and
//! require corrective guidance in every panic message a caller can hit.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::{self, Tree};

/// The reviewed list of files permitted to carry `allow(unsafe_code)`.
const BUDGET: &str = "xtask/unsafe-budget.txt";

/// Every production `.expect("...")` states the corrective action.
///
/// A panic message a reader cannot act on is a crash with extra words. The
/// reader in question is whoever hit the panic in a shipped run, so the scan
/// covers production code: `tests/` and `benches/` trees are out of scope and so
/// is an inline `#[cfg(test)]` item, which is the same code in a different file.
pub struct ExpectHasFix;

impl Gate for ExpectHasFix {
    fn name(&self) -> &'static str {
        "lint-expect-fix"
    }

    fn help(&self) -> &'static str {
        "expect() sites with no corrective guidance in their message"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        if let Some(note) = tree.absence_note() {
            report.note(note);
        }
        let files: Vec<PathBuf> = tree
            .all_rust()
            .into_iter()
            .filter(|path| !is_outside_production(path))
            .collect();
        report.note(format!("scanned {} production source file(s)", files.len()));
        for file in &files {
            let text = tree.read(file)?;
            let lines: Vec<&str> = text.lines().collect();
            // A `#[cfg(test)]` item is the same code as a `tests/` tree, which
            // this scan already leaves alone: its panic text is read by whoever
            // broke the test, and the corrective action is in the change, not in
            // the fixture. A production panic is the subject of the rule.
            let in_test_item = scan::cfg_test_lines(&lines);
            for (index, line) in lines.iter().enumerate() {
                if !line.contains(".expect(\"") {
                    continue;
                }
                if in_test_item.get(index).copied().unwrap_or(false) {
                    continue;
                }
                // A gate that scans for the string `.expect("` writes that
                // string, in code and in the prose beside it, and neither is a
                // panic site.
                if scan::is_comment(line)
                    || line.contains("contains(\".expect(\"")
                    || line.contains("concat!")
                {
                    continue;
                }
                let end = (index + 4).min(lines.len());
                let window = lines[index..end].join("\n");
                if window.contains("Fix:") {
                    continue;
                }
                report.find(Finding::at(
                    file.clone(),
                    u32::try_from(index + 1).unwrap_or(u32::MAX),
                    format!("expect() with no corrective guidance: {}", line.trim()),
                    "state the corrective action in the message, as `Fix: ...`",
                ));
            }
        }
        Ok(report)
    }
}

/// No crate turns the documentation floor off for itself.
pub struct MissingDocsOverride;

impl Gate for MissingDocsOverride {
    fn name(&self) -> &'static str {
        "lint-missing-docs-override"
    }

    fn help(&self) -> &'static str {
        "crate-root allow(missing_docs) overrides of the workspace deny floor"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let roots: Vec<PathBuf> = tree
            .paths()
            .iter()
            .filter(|path| path.ends_with("src/lib.rs"))
            .cloned()
            .collect();
        report.note(format!("scanned {} crate root(s)", roots.len()));
        for file in &roots {
            let text = tree.read(file)?;
            for (number, line) in scan::numbered(&text) {
                if !is_inner_allow_of(line, "missing_docs") {
                    continue;
                }
                report.find(Finding::at(
                    file.clone(),
                    number,
                    "crate root disables the workspace missing_docs floor",
                    "delete the inner attribute and document the public items the lint names; \
                     a module-scoped allow on a generated module is the narrow form",
                ));
            }
        }
        Ok(report)
    }
}

/// The unsafe surface matches the reviewed list exactly.
///
/// An addition fails because new unsafe needs a review. A removal fails too: a
/// list naming a file that no longer carries the override overstates the audited
/// surface. Three of the nine entries in the version before this one named a
/// crate that no longer existed, so the budget reserved review for nothing.
pub struct UnsafeBudget;

impl Gate for UnsafeBudget {
    fn name(&self) -> &'static str {
        "lint-unsafe-budget"
    }

    fn help(&self) -> &'static str {
        "files carrying allow(unsafe_code) against the reviewed budget"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let budget_text = tree.read(BUDGET)?;
        let reviewed: BTreeSet<&str> = budget_text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .collect();
        let mut actual: BTreeSet<String> = BTreeSet::new();
        for file in tree.all_rust() {
            // The override is an attribute, so the scan reads code: literals are
            // masked and comment lines are skipped. This gate spells
            // allow(unsafe_code) in both places to look for it, and a rule that
            // counts its own source has one exception it can never lose.
            let text = scan::mask_literals(&tree.read(&file)?);
            let carries = text
                .lines()
                .any(|line| !scan::is_comment(line) && line.contains("allow(unsafe_code)"));
            if carries {
                actual.insert(file.to_string_lossy().into_owned());
            }
        }
        report.note(format!(
            "{} file(s) reviewed, {} file(s) carrying the override",
            reviewed.len(),
            actual.len()
        ));
        for file in &actual {
            if !reviewed.contains(file.as_str()) {
                report.find(Finding::in_file(
                    file,
                    "unsafe surface not on the reviewed budget",
                    format!(
                        "remove the unsafe, wrap it inside a file already on the list, or add \
                         the path to {BUDGET} after a security review; every site owes a SAFETY \
                         comment naming the invariant its caller relies on"
                    ),
                ));
            }
        }
        for file in &reviewed {
            if !actual.contains(*file) {
                report.find(Finding::in_file(
                    *file,
                    "reviewed budget names a file that no longer carries allow(unsafe_code)",
                    format!("delete the line from {BUDGET}; a stale entry reserves audited budget for a file that does not use it"),
                ));
            }
        }
        Ok(report)
    }
}

/// Every unsafe block carries a justification a reader can check.
pub struct UnsafeJustification;

impl Gate for UnsafeJustification {
    fn name(&self) -> &'static str {
        "lint-unsafe-justification"
    }

    fn help(&self) -> &'static str {
        "unsafe blocks with no SAFETY justification, or a placeholder one"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        const COP_OUTS: &[&str] = &[
            "todo", "fixme", "unclear", "investigate", "unknown", "tbd", "???",
        ];
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let files: Vec<PathBuf> = tree
            .all_rust()
            .into_iter()
            .filter(|path| !is_outside_production(path))
            .collect();
        report.note(format!("scanned {} production source file(s)", files.len()));
        for file in &files {
            // A quoted block is fixture text, including this gate's own examples,
            // so the scan reads code with literals masked.
            let text = scan::mask_literals(&tree.read(file)?);
            let lines: Vec<&str> = text.lines().collect();
            for (index, line) in lines.iter().enumerate() {
                if !opens_unsafe_block(line) {
                    continue;
                }
                let comment = preceding_comment_block(&lines, index);
                let number = u32::try_from(index + 1).unwrap_or(u32::MAX);
                match safety_justification(&comment) {
                    None => report.find(Finding::at(
                        file.clone(),
                        number,
                        "unsafe block with no SAFETY comment",
                        "write a SAFETY comment in the immediately preceding comment block, \
                         naming the invariants that make the block sound",
                    )),
                    Some(justification) => {
                        let lowered = justification.to_ascii_lowercase();
                        if COP_OUTS.iter().any(|marker| lowered.starts_with(marker)) {
                            report.find(Finding::at(
                                file.clone(),
                                number,
                                format!("unsafe block with a placeholder SAFETY comment: {justification}"),
                                "write the real justification; a comment promising one that does \
                                 not exist is worse than none",
                            ));
                        }
                    }
                }
            }
        }
        Ok(report)
    }
}

/// Whether a path sits outside production sources.
///
/// Test and benchmark trees are not production, and neither is the fragment
/// directory a historical split left behind. The rule is written out rather than
/// shared with the hot-path scanner, because that one also excludes fuzz targets
/// and excluding them here would narrow the scan.
fn is_outside_production(path: &Path) -> bool {
    let path = path.to_string_lossy();
    path.contains("/tests/")
        || path.starts_with("tests/")
        || path.contains("/benches/")
        || path.starts_with("benches/")
        || path.contains("/__law7_split/")
}

/// Whether a line opens an unsafe block.
fn opens_unsafe_block(line: &str) -> bool {
    let Some(at) = line.find("unsafe") else {
        return false;
    };
    if scan::is_comment(line) {
        return false;
    }
    let rest = line[at + "unsafe".len()..].trim_start();
    rest.starts_with('{')
}

/// The comment lines directly above a line, up to the first line that is not one.
///
/// A blank line ends the block: a justification belongs against the block it
/// justifies, and walking past a gap would let a doc comment several lines up
/// answer for an unsafe block it never mentions. The block has no line bound,
/// because a marker followed by a long list of invariants is the shape the rule
/// is asking for and a bound would drop the marker out of the window.
fn preceding_comment_block(lines: &[&str], index: usize) -> String {
    let mut collected: Vec<&str> = Vec::new();
    let mut cursor = index;
    while cursor > 0 {
        cursor -= 1;
        let line = lines[cursor];
        if !line.trim_start().starts_with("//") {
            break;
        }
        collected.push(line);
    }
    collected.reverse();
    collected.join("\n")
}

/// The text after a `// SAFETY:` marker, when the block carries one.
///
/// The marker line is often bare, with the invariants listed as bullets on the
/// comment lines under it. Those lines are the justification, so they are joined
/// into it: a reader checking the block reads the whole list, and a placeholder
/// hiding one line below the marker is still caught.
fn safety_justification(comment: &str) -> Option<String> {
    let mut lines = comment.lines();
    while let Some(line) = lines.next() {
        let Some(rest) = comment_body(line) else {
            continue;
        };
        let Some(text) = rest.strip_prefix("SAFETY:") else {
            continue;
        };
        let mut justification = text.trim().to_string();
        for line in lines.by_ref() {
            let Some(rest) = comment_body(line) else {
                break;
            };
            let rest = rest.trim().trim_start_matches('*').trim();
            if rest.is_empty() {
                continue;
            }
            if !justification.is_empty() {
                justification.push(' ');
            }
            justification.push_str(rest);
        }
        if !justification.is_empty() {
            return Some(justification);
        }
    }
    None
}

/// The text of a line comment, when the line is one.
fn comment_body(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    let rest = trimmed.strip_prefix("//")?;
    Some(
        rest.trim_start_matches('/')
            .trim_start_matches('!')
            .trim_start(),
    )
}

/// Whether a line is a crate-root `#![allow(...)]` naming a lint.
fn is_inner_allow_of(line: &str, lint: &str) -> bool {
    let trimmed = line.trim_start();
    let Some(rest) = trimmed.strip_prefix("#![allow(") else {
        return false;
    };
    let end = rest.find(')').unwrap_or(rest.len());
    rest[..end].contains(lint)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::process::Command;

    use tempfile::TempDir;

    use super::*;

    /// WHY: the floor is crate-wide, and the narrow module-scoped form on a
    /// generated module is deliberately allowed. A check that could not tell the
    /// two apart would either miss the override or forbid the generated module.
    #[test]
    fn only_the_crate_root_form_disables_the_floor() {
        assert!(is_inner_allow_of("#![allow(missing_docs)]", "missing_docs"));
        assert!(is_inner_allow_of(
            "  #![allow(dead_code, missing_docs)]",
            "missing_docs"
        ));
        assert!(!is_inner_allow_of("#[allow(missing_docs)]", "missing_docs"));
        assert!(!is_inner_allow_of(
            "#![allow(dead_code)] // missing_docs stays denied",
            "missing_docs"
        ));
    }

    /// WHY: a SAFETY comment that says TODO promises a justification that does
    /// not exist, and the shell original matched the cop-out list case
    /// insensitively anywhere in the block, so a comment mentioning "unknown
    /// alignment" in prose read as a cop-out.
    #[test]
    fn a_placeholder_justification_is_told_apart_from_a_real_one() {
        assert_eq!(
            safety_justification("// SAFETY: the pointer is valid for len bytes"),
            Some("the pointer is valid for len bytes".to_string())
        );
        assert_eq!(
            safety_justification("// SAFETY: TODO"),
            Some("TODO".to_string())
        );
        assert_eq!(safety_justification("// no marker here"), None);
        assert_eq!(safety_justification("// SAFETY:"), None);
    }

    /// WHY: `unsafe` also appears in `unsafe fn`, in `unsafe impl` and in prose.
    /// Only a block is the thing that needs a justification above it.
    #[test]
    fn only_an_unsafe_block_needs_a_justification() {
        assert!(opens_unsafe_block("        unsafe {"));
        assert!(opens_unsafe_block("let value = unsafe { read(ptr) };"));
        assert!(!opens_unsafe_block("unsafe fn caller() {"));
        assert!(!opens_unsafe_block("unsafe impl Send for Handle {}"));
        assert!(!opens_unsafe_block("// unsafe { } appears in prose"));
    }

    /// WHY: the comment block above a block is where the justification lives,
    /// and it must stop at the first line of code so a justification cannot be
    /// borrowed from an unrelated function above.
    #[test]
    fn a_comment_block_stops_at_the_first_line_of_code() {
        let lines = vec![
            "// SAFETY: belongs to the function above",
            "fn other() {}",
            "",
            "// a plain note",
            "unsafe {",
        ];
        let block = preceding_comment_block(&lines, 4);
        assert!(block.contains("a plain note"));
        assert!(!block.contains("belongs to the function above"));
    }

    /// WHY: the marker line is usually bare, with the invariants listed under
    /// it, and the previous reader looked only at the marker line and at eight
    /// lines of block. A long justification lost its own marker out of that
    /// window, so the soundest block in the workspace read as unjustified while
    /// a one-line "SAFETY: TODO" one line lower read as fine.
    #[test]
    fn a_justification_under_the_marker_is_read_and_a_placeholder_there_is_caught() {
        let wrapped = "// SAFETY:\n// * the pointer is valid for len bytes\n// * no other reference aliases it";
        assert_eq!(
            safety_justification(wrapped),
            Some("the pointer is valid for len bytes no other reference aliases it".to_string())
        );
        assert_eq!(
            safety_justification("// SAFETY:\n// TODO work out the aliasing"),
            Some("TODO work out the aliasing".to_string())
        );
    }

    /// WHY: the rule is about a panic a shipped run can hit. It already skips
    /// `tests/` and `benches/` trees, and an inline `#[cfg(test)]` item is the
    /// same code in another place, so 412 of the 466 findings were fixture text
    /// whose corrective action lives in the change that broke the test. The
    /// production site next to it must still be reported, or the rule cannot fail.
    #[test]
    fn a_production_expect_owes_a_fix_and_a_test_item_does_not() {
        let (_directory, root) = fixture_tree(&[(
            "site.rs",
            "fn load(path: &str) -> String {\n    std::fs::read_to_string(path).expect(\"the config file\")\n}\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn it_loads() {\n        let value = super::load(\"x\").expect(\"a loaded config\");\n        assert!(!value.is_empty());\n    }\n}\n",
        )]);

        let report = ExpectHasFix
            .run(&GateCtx::new(root, Vec::new()))
            .expect("Fix: the gate must read the fixture tree; check the fixture git step");
        let lines: Vec<u32> = report
            .findings
            .iter()
            .filter_map(|finding| finding.line)
            .collect();
        assert_eq!(
            lines,
            [2],
            "only the production site owes a corrective action: {:?}",
            report
                .findings
                .iter()
                .map(|finding| finding.message.clone())
                .collect::<Vec<_>>()
        );
    }

    /// WHY: both rules read every Rust file in the tree, so their own text is in
    /// scope: this gate spells `unsafe {` in the fixtures above and spells the
    /// override in the scan that looks for it. A rule that reports itself spends
    /// a pin on the size of its own test module. Masked literals and skipped
    /// comment lines keep the examples readable, and this proves both directions.
    #[test]
    fn a_quoted_unsafe_block_is_data_and_a_real_one_still_needs_its_justification() {
        let (_directory, root) = fixture_tree(&[
            (
                "quoted.rs",
                "fn fixture() {\n    let needles = [\"unsafe {\", \"allow(unsafe_code)\"];\n}\n",
            ),
            (
                "justified.rs",
                "fn read(ptr: *const u8, len: usize) {\n    // SAFETY:\n    // * the caller owns len readable bytes at ptr\n    // * nothing else writes them while this borrow lives\n    unsafe {\n        let _ = core::slice::from_raw_parts(ptr, len);\n    }\n}\n",
            ),
            (
                "bare.rs",
                "fn read(ptr: *const u8, len: usize) {\n    unsafe {\n        let _ = core::slice::from_raw_parts(ptr, len);\n    }\n}\n",
            ),
        ]);

        let report = UnsafeJustification
            .run(&GateCtx::new(root, Vec::new()))
            .expect("the gate reads the fixture tree");
        assert_eq!(
            reported_files(&report),
            ["bare.rs"],
            "a quoted block is data, a wrapped justification is a justification: {:?}",
            reported_files(&report)
        );
    }

    /// WHY: the override is an attribute. The scan spelled it in a literal and in
    /// the comment beside that literal, so its own source counted as an unsafe
    /// surface and the pin could only be met by deleting the explanation.
    #[test]
    fn only_a_real_override_counts_against_the_budget() {
        let (_directory, root) = fixture_tree(&[
            (
                "xtask/unsafe-budget.txt",
                "# reviewed surfaces\nreal.rs\n",
            ),
            (
                "quoted.rs",
                "// allow(unsafe_code) in a comment is prose\nfn fixture() {\n    let needle = \"allow(unsafe_code)\";\n}\n",
            ),
            (
                "real.rs",
                "#[allow(unsafe_code)]\nfn wrapper() {}\n",
            ),
        ]);

        let report = UnsafeBudget
            .run(&GateCtx::new(root, Vec::new()))
            .expect("the gate reads the fixture tree");
        assert!(
            report.findings.is_empty(),
            "the reviewed file carries the override and no other file does: {:?}",
            reported_files(&report)
        );
        assert!(
            report
                .notes
                .iter()
                .any(|note| note.contains("1 file(s) reviewed, 1 file(s) carrying the override")),
            "the note counts the surface: {:?}",
            report.notes
        );
    }

    /// A git checkout holding the given files, which is what `Tree::open` needs.
    ///
    /// The directory is returned with it: dropping it deletes the tree, so the
    /// caller holds it for as long as the gate reads it.
    fn fixture_tree(files: &[(&str, &str)]) -> (TempDir, PathBuf) {
        let temporary = TempDir::new().expect("a temporary directory");
        let root = temporary.path().to_path_buf();
        for (path, text) in files {
            let target = root.join(path);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent).expect("a fixture directory");
            }
            fs::write(target, text).expect("a fixture file");
        }
        let status = Command::new("git")
            .args(["init", "-q", "."])
            .current_dir(&root)
            .status()
            .expect("git is available");
        assert!(status.success(), "the fixture git step failed");
        (temporary, root)
    }

    /// The files a report names, in the order it named them.
    fn reported_files(report: &Report) -> Vec<String> {
        report
            .findings
            .iter()
            .map(|finding| {
                finding
                    .file
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_default()
            })
            .collect()
    }
}
