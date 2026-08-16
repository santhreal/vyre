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

/// The lint policy is declared once, in the workspace manifest.
///
/// Two things make a member diverge, and this reports both. A manifest that
/// declares its own `[lints.*]` table replaces the inherited policy wholesale
/// for that tool, which is how `vyre-driver-metal` allowed `unsafe_code`
/// crate-wide outside the reviewed budget and `vyre-grammar-gen` held
/// `missing_docs` at `warn` while the workspace denied it. A crate-root
/// `#![allow(...)]`, `#![deny(...)]` or `#![forbid(...)]` does the same thing one
/// lint at a time, and it wins over the manifest, so a suppression there is
/// invisible in the table a reader consults to learn the policy.
///
/// The member set is read from the workspace manifest at run time, so a crate
/// added to the workspace is held to this from its first commit. A hardcoded
/// roster is what let 41 of 42 members ignore the table.
///
/// One exception: `#![allow(unsafe_code)]`, alone in its attribute. FFI crates
/// need it, `unsafe_code = "deny"` in the workspace table is what makes the
/// override visible, and `lint-unsafe-budget` already holds the resulting file
/// set to a reviewed list. A module-scoped `#[allow]` on a generated module is
/// also untouched: it names the item it covers, which is the narrow form this
/// rule asks for. This gate subsumes the crate-root `allow(missing_docs)` check
/// that used to stand beside it, which read one lint out of that population.
pub struct OneLintPolicy;

impl Gate for OneLintPolicy {
    fn name(&self) -> &'static str {
        "lint-one-policy"
    }

    fn help(&self) -> &'static str {
        "members that do not inherit the workspace lint policy, or override it at their crate root"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let members = tree.members()?;
        report.note(format!("{} workspace member(s)", members.len()));
        for member in &members {
            let manifest_path = format!("{member}/Cargo.toml");
            let manifest = tree.read_toml(&manifest_path)?;
            match manifest.get("lints").and_then(toml::Value::as_table) {
                None => report.find(Finding::in_file(
                    &manifest_path,
                    "member declares no lint policy at all",
                    "add `[lints]` with `workspace = true`; the workspace table is the policy \
                     and a member outside it is judged by nothing",
                )),
                Some(table) => {
                    if table.get("workspace").and_then(toml::Value::as_bool) != Some(true) {
                        report.find(Finding::in_file(
                            &manifest_path,
                            "member does not inherit the workspace lint policy",
                            "set `workspace = true` under `[lints]`",
                        ));
                    }
                    for key in table.keys().filter(|key| key.as_str() != "workspace") {
                        report.find(Finding::in_file(
                            &manifest_path,
                            format!("member declares its own `[lints.{key}]` table"),
                            "delete the table and inherit; promote an entry the whole tree needs \
                             into `[workspace.lints]` with the justification comment that table \
                             uses, or narrow it to the item that needs it",
                        ));
                    }
                }
            }
            for root in [
                format!("{member}/src/lib.rs"),
                format!("{member}/src/main.rs"),
            ] {
                if !tree.exists(&root) {
                    continue;
                }
                let text = tree.read(&root)?;
                for (number, attribute) in inner_lint_attributes(&text) {
                    report.find(Finding::at(
                        root.clone(),
                        number,
                        format!("crate root sets a lint level: {attribute}"),
                        "delete the attribute; the workspace table owns every level, and the \
                         only crate-root exception is `#![allow(unsafe_code)]` alone, reviewed \
                         through xtask/unsafe-budget.txt",
                    ));
                }
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

/// Every crate-root inner attribute that sets a lint level, with its line.
///
/// The attribute may span lines, and it is the level word that matters rather
/// than the lint names after it, so the scan reads the leading path of each
/// `#![...]` and keeps the ones that are a level. `cfg_attr` is read too: the
/// levels it carries apply on the configurations it names, and a `deny` behind
/// `not(test)` is still policy declared outside the table.
///
/// `#![allow(unsafe_code)]` alone is the one accepted form, so an attribute that
/// bundles it with other lints is reported: the reviewed budget names files, and
/// a bundle makes the file's exception ambiguous.
fn inner_lint_attributes(text: &str) -> Vec<(u32, String)> {
    const LEVELS: [&str; 5] = ["allow", "warn", "deny", "forbid", "expect"];
    let lines: Vec<&str> = text.lines().collect();
    let mut found = Vec::new();
    let mut index = 0usize;
    while index < lines.len() {
        let trimmed = lines[index].trim_start();
        if !trimmed.starts_with("#![") {
            index += 1;
            continue;
        }
        let start = index;
        let mut attribute = String::new();
        let mut depth = 0i32;
        loop {
            let line = lines[index].trim();
            if !attribute.is_empty() {
                attribute.push(' ');
            }
            attribute.push_str(line);
            depth += i32::try_from(line.matches('(').count()).unwrap_or(0)
                - i32::try_from(line.matches(')').count()).unwrap_or(0);
            index += 1;
            if depth <= 0 || index >= lines.len() {
                break;
            }
        }
        let body = attribute.trim_start_matches("#![");
        let path = body
            .split(|character: char| !is_attribute_path_byte(character))
            .next()
            .unwrap_or_default();
        let is_level = LEVELS.contains(&path);
        let carries_level = path == "cfg_attr"
            && LEVELS
                .iter()
                .any(|level| body.contains(&format!("{level}(")));
        if (is_level || carries_level) && attribute != "#![allow(unsafe_code)]" {
            found.push((u32::try_from(start + 1).unwrap_or(u32::MAX), attribute));
        }
    }
    found
}

/// Whether `character` can appear in an attribute's leading path.
fn is_attribute_path_byte(character: char) -> bool {
    character.is_ascii_alphanumeric() || character == '_'
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::fixture_checkout::checkout;

    /// WHY: the policy is crate-wide, and three neighbours of the defect must
    /// stay unreported: the module-scoped `#[allow]` on a generated module, the
    /// reviewed `#![allow(unsafe_code)]`, and an inner attribute that is not a
    /// lint level at all. A scanner that could not tell them apart would either
    /// miss the multi-line blankets, which is the shape every crate used, or
    /// forbid `#![no_std]`.
    #[test]
    fn a_crate_root_level_is_told_apart_from_its_neighbours() {
        let found = inner_lint_attributes(
            "#![no_std]\n\
             #![allow(unsafe_code)]\n\
             #![warn(missing_docs)]\n\
             #[allow(missing_docs)]\n\
             #![allow(\n    clippy::type_complexity,\n    clippy::let_and_return\n)]\n\
             #![cfg_attr(not(test), deny(clippy::panic))]\n",
        );
        assert_eq!(
            found,
            vec![
                (3, "#![warn(missing_docs)]".to_string()),
                (
                    5,
                    "#![allow( clippy::type_complexity, clippy::let_and_return )]".to_string()
                ),
                (
                    9,
                    "#![cfg_attr(not(test), deny(clippy::panic))]".to_string()
                ),
            ]
        );
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
        let (_directory, root) = checkout(&[(
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
        let (_directory, root) = checkout(&[
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
        let (_directory, root) = checkout(&[
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

    /// WHY: this is the rule that has to fail on the state the workspace was in,
    /// where one member of 42 inherited the table and the rest declared their own
    /// policy or overrode it at the crate root. Both defects are injected here
    /// against a member that inherits correctly, and the member roster comes from
    /// the fixture's own workspace manifest, so a crate added to the workspace is
    /// judged without an edit to this gate.
    #[test]
    fn a_member_outside_the_workspace_policy_is_reported_and_an_inheriting_one_is_not() {
        let (_directory, root) = checkout(&[
            (
                "Cargo.toml",
                "[workspace]\nmembers = [\"inheriting\", \"own-table\", \"root-override\"]\n\n[workspace.lints.rust]\nmissing_docs = \"deny\"\n",
            ),
            (
                "inheriting/Cargo.toml",
                "[package]\nname = \"inheriting\"\n\n[lints]\nworkspace = true\n",
            ),
            (
                "inheriting/src/lib.rs",
                "//! An inheriting member.\n\n#![allow(unsafe_code)]\n",
            ),
            (
                "own-table/Cargo.toml",
                "[package]\nname = \"own-table\"\n\n[lints.rust]\nmissing_docs = \"warn\"\n",
            ),
            ("own-table/src/lib.rs", "//! A member with its own table.\n"),
            (
                "root-override/Cargo.toml",
                "[package]\nname = \"root-override\"\n\n[lints]\nworkspace = true\n",
            ),
            (
                "root-override/src/lib.rs",
                "//! A member that overrides at its root.\n\n#![allow(missing_docs)]\n",
            ),
        ]);

        let report = OneLintPolicy
            .run(&GateCtx::new(root, Vec::new()))
            .expect("the gate reads the fixture tree");
        assert_eq!(
            reported_files(&report),
            [
                "own-table/Cargo.toml",
                "own-table/Cargo.toml",
                "root-override/src/lib.rs"
            ],
            "the divergent member is reported twice, once for the missing inheritance and once \
             for the table it declared instead, and the crate-root override is reported once: \
             {:?}",
            report
                .findings
                .iter()
                .map(|finding| finding.message.clone())
                .collect::<Vec<_>>()
        );
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
