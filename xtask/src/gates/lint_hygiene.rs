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

/// Every `.expect("...")` states the corrective action.
///
/// A panic message a reader cannot act on is a crash with extra words.
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
            for (index, line) in lines.iter().enumerate() {
                if !line.contains(".expect(\"") {
                    continue;
                }
                // A gate that scans for the string `.expect("` writes that
                // string, and its own source is not a panic site.
                if line.contains("contains(\".expect(\"") || line.contains("concat!") {
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
            if tree.read(&file)?.contains("allow(unsafe_code)") {
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
            let text = tree.read(file)?;
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

/// The contiguous comment block immediately above a line, bounded to eight lines.
fn preceding_comment_block(lines: &[&str], index: usize) -> String {
    let mut collected: Vec<&str> = Vec::new();
    let mut cursor = index;
    while cursor > 0 {
        cursor -= 1;
        let line = lines[cursor];
        let trimmed = line.trim();
        if !(trimmed.is_empty() || trimmed.starts_with("//")) {
            break;
        }
        collected.push(line);
        if index - cursor >= 8 {
            break;
        }
    }
    collected.reverse();
    collected.join("\n")
}

/// The text after a `// SAFETY:` marker, when the block carries one.
fn safety_justification(comment: &str) -> Option<String> {
    for line in comment.lines() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix("//") else {
            continue;
        };
        let rest = rest.trim_start_matches('/').trim_start_matches('!').trim_start();
        let Some(text) = rest.strip_prefix("SAFETY:") else {
            continue;
        };
        let text = text.trim();
        if !text.is_empty() {
            return Some(text.to_string());
        }
    }
    None
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
}
