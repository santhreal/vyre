//! What the workspace lint floor cannot say by itself.
//!
//! `[workspace.lints.rust]` denies `unsafe_code` and `missing_docs`, and every
//! member inherits it, so the set of files carrying an `allow` override is the
//! complete exception surface and rustc is the thing enforcing it. These gates
//! pin that surface, require a justification beside every unsafe block, and
//! require corrective guidance in every panic message a caller can hit.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::{self, Tree};

/// The reviewed list of files permitted to carry `allow(unsafe_code)`.
const BUDGET: &str = "xtask/unsafe-budget.txt";

/// The denied lints another gate answers for, so this one does not judge them.
///
/// `unsafe_code` is reviewed file by file through `xtask/unsafe-budget.txt`, and
/// `missing_docs` is read out of crate roots by `OneLintPolicy`. Every other
/// denied lint is derived from the table, so promoting a lint to `deny` puts it
/// under both the mutation suite and the override scan without an edit here.
pub const LINTS_OWNED_ELSEWHERE: &[&str] = &["missing_docs", "unsafe_code"];

/// A lint the workspace denies is not re-allowed in production.
///
/// The table is the policy, and an attribute beats the table. `dead_code` was
/// denied workspace-wide while `vyre-runtime/src/uring/raw_platform.rs` carried
/// a module-scoped `#![allow(dead_code)]`, and no gate saw it: `OneLintPolicy`
/// reads `src/lib.rs` and `src/main.rs`, so a suppression one directory down was
/// outside every scan. That allowance turned out to suppress nothing at all,
/// which is the other half of the defect: an override that has expired reads as
/// a standing exemption and hides the next real one.
///
/// `#[expect(..., reason = "...")]` is the accepted production form. It states
/// why in the source, and `unfulfilled_lint_expectations = "deny"` deletes it
/// for whoever wires the item up, so an exception cannot outlive its cause.
/// `allow` has neither property and is reported wherever it names a denied lint.
///
/// The lint set is read from `[workspace.lints.rust]` on each run rather than
/// written here, so a lint promoted to `deny` is covered from that commit.
pub struct DeniedLintOverride;

impl crate::gate::GateBehavior for DeniedLintOverride {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let all_rust = tree.all_rust();
        report.cover_complete("source files", all_rust.len());
        if let Some(note) = tree.absence_note() {
            report.note(note);
        }

        let denied = denied_rust_lints(&tree)?;
        if denied.is_empty() {
            return Err(GateError::new(
                "[workspace.lints.rust] denies no lint this gate judges",
                "restore the denied levels in the root manifest; a scan over an empty lint set \
                 reports success forever",
            ));
        }
        report.note(format!(
            "{} denied lint(s) read from [workspace.lints.rust]: {}",
            denied.len(),
            denied
                .iter()
                .map(String::as_str)
                .collect::<Vec<&str>>()
                .join(", ")
        ));

        let test_modules = scan::test_module_files(&tree, &all_rust)?;
        let files: Vec<PathBuf> = all_rust
            .into_iter()
            .filter(|path| !is_outside_production(path) && !test_modules.contains(path))
            .collect();
        report.note(format!("scanned {} production source file(s)", files.len()));

        let groups = groups_covering(&denied)?;
        report.note(format!(
            "{} lint group(s) reach a denied lint: {}",
            groups.len(),
            groups
                .iter()
                .map(String::as_str)
                .collect::<Vec<&str>>()
                .join(", ")
        ));

        for file in &files {
            let text = tree.read(file)?;
            for (number, attribute) in level_attributes(&text, false) {
                let named: Vec<&str> = identifiers(&attribute)
                    .into_iter()
                    .filter(|token| denied.contains(*token) || groups.contains(*token))
                    .collect();
                if named.is_empty() {
                    continue;
                }
                if attribute.contains("expect(") && attribute.contains("reason") {
                    continue;
                }
                report.find(Finding::at(
                    file.clone(),
                    number,
                    format!(
                        "production attribute sets the level of denied lint(s) {}: {attribute}",
                        named.join(", ")
                    ),
                    "delete the attribute and give the item a caller or delete the item; where \
                     the suppression is real, write `#[expect(lint, reason = \"...\")]` at the \
                     smallest item, which expires on its own once the cause is gone",
                ));
            }
        }
        Ok(report)
    }
}

/// The lints `[workspace.lints.rust]` denies, less the ones another gate owns.
fn denied_rust_lints(tree: &Tree) -> Result<BTreeSet<String>, GateError> {
    let manifest = tree.read_toml("Cargo.toml")?;
    let rust = manifest
        .get("workspace")
        .and_then(|workspace| workspace.get("lints"))
        .and_then(|lints| lints.get("rust"))
        .and_then(toml::Value::as_table)
        .ok_or_else(|| {
            GateError::new(
                "the root manifest declares no [workspace.lints.rust]",
                "restore the table; it is the one place this workspace states a lint level",
            )
        })?;
    Ok(rust
        .iter()
        .filter(|(lint, _)| !LINTS_OWNED_ELSEWHERE.contains(&lint.as_str()))
        .filter(|(_, value)| {
            let level = match value {
                toml::Value::String(level) => level.as_str(),
                toml::Value::Table(entry) => entry
                    .get("level")
                    .and_then(toml::Value::as_str)
                    .unwrap_or_default(),
                _ => "",
            };
            level == "deny" || level == "forbid"
        })
        .map(|(lint, _)| lint.clone())
        .collect())
}

/// The rustc lint groups that reach at least one lint in `denied`.
///
/// A group name in an attribute sets the level of every lint under it, so
/// `allow(unused)` silences `dead_code` as completely as naming it. The
/// membership is rustc's, not this workspace's, so it is read from the
/// compiler that will build the tree rather than written down here: a lint
/// moved into a group between toolchains is followed without an edit.
///
/// A group whose sub-lint column is prose rather than lint names, which is how
/// `warnings` states that it covers every lint currently at warn level, reaches
/// every denied lint and is returned unconditionally.
fn groups_covering(denied: &BTreeSet<String>) -> Result<BTreeSet<String>, GateError> {
    let output = Command::new("rustc")
        .arg("-W")
        .arg("help")
        .output()
        .map_err(|err| {
            GateError::new(
            format!("cannot run `rustc -W help`: {err}"),
            "install the toolchain this workspace builds with; the lint group table is read from \
             the compiler rather than written into the gate",
        )
        })?;
    let text = String::from_utf8_lossy(&output.stdout);
    let table = text
        .split_once("Lint groups provided by rustc:")
        .map(|(_, rest)| rest)
        .ok_or_else(|| {
            GateError::new(
                "`rustc -W help` printed no lint group table",
                "check the toolchain: without the table a group name in an attribute would \
                 silence a denied lint unreported",
            )
        })?;

    let mut covering = BTreeSet::new();
    for line in table.lines() {
        let Some((name, members)) = line.trim().split_once("  ") else {
            continue;
        };
        let name = name.trim();
        if name.is_empty() || name == "name" || name.starts_with('-') {
            continue;
        }
        let members: Vec<&str> = members.split(',').map(str::trim).collect();
        let prose = members.iter().any(|member| member.contains(' '));
        if prose
            || members
                .iter()
                .any(|member| denied.contains(&member.replace('-', "_")))
        {
            covering.insert(name.replace('-', "_"));
        }
    }
    if covering.is_empty() {
        return Err(GateError::new(
            "no rustc lint group reaches a denied lint, which the `warnings` group alone rules out",
            "check the parse of `rustc -W help`; an empty group set lets `allow(unused)` pass",
        ));
    }
    Ok(covering)
}

/// Every production `.expect("...")` states the corrective action.
///
/// A panic message a reader cannot act on is a crash with extra words. The
/// reader in question is whoever hit the panic in a shipped run, so the scan
/// covers production code: `tests/` and `benches/` trees are out of scope, so is
/// an inline `#[cfg(test)]` item, and so is a file a `#[cfg(test)] mod`
/// declaration reaches, which is the same code with the attribute one file up.
pub struct ExpectHasFix;

impl crate::gate::GateBehavior for ExpectHasFix {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        report.cover_complete("source files", tree.all_rust().len());
        if let Some(note) = tree.absence_note() {
            report.note(note);
        }
        let all_rust = tree.all_rust();
        let test_modules = scan::test_module_files(&tree, &all_rust)?;
        let files: Vec<PathBuf> = all_rust
            .into_iter()
            .filter(|path| !is_outside_production(path) && !test_modules.contains(path))
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

impl crate::gate::GateBehavior for OneLintPolicy {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        report.cover_complete("source files", tree.all_rust().len());
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

impl crate::gate::GateBehavior for UnsafeBudget {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        report.cover_complete("source files", tree.all_rust().len());
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

impl crate::gate::GateBehavior for UnsafeJustification {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        const COP_OUTS: &[&str] = &[
            "todo",
            "fixme",
            "unclear",
            "investigate",
            "unknown",
            "tbd",
            "???",
        ];
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let files: Vec<PathBuf> = tree
            .all_rust()
            .into_iter()
            .filter(|path| !is_outside_production(path))
            .collect();
        report.cover_complete("production source files", files.len());
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

/// A liveness lint cannot be satisfied by pretending.
///
/// `dead_code = "deny"` in the workspace table is the cheapest signal this tree
/// has for code with no owner, and two writes silence it without giving the code
/// one. A discarded read counts as a use, so `let _ = MARKER;` keeps a constant
/// nothing consults. A `#[cfg(test)]` caller counts as a use too, so an item
/// whose only caller is a test ships with no production owner and nothing is
/// red. In both the lint reports what it was told rather than what the tree
/// does, which is worse than the lint being off: the tree now certifies
/// liveness it never checked.
///
/// # What is reported
///
/// A production statement whose whole effect is discarding a path or field read.
/// `let _ = fallible();` discards a real result, and `let _ = &guard;` borrows
/// something whose drop is the point, so a read carrying a call, a macro, a
/// borrow or `?` is left alone.
///
/// A private, `pub(crate)` or `pub(super)` production item whose every reference
/// in the tree is a test one. Bare `pub` is out of scope: its callers are in
/// other checkouts, so this one cannot answer whether it has any.
///
/// # What it does not catch
///
/// References are matched by identifier, so only a name declared exactly once in
/// the workspace is judged. A second declaration makes an identifier match
/// ambiguous, and a gate that guesses is worse than one that declines. The
/// declared-once set is derived from the tree on each run, so a name that
/// becomes unique becomes judged. A self-recursive item references itself from
/// production and is left to the compiler.
pub struct LivenessEvasion;

impl crate::gate::GateBehavior for LivenessEvasion {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let all_rust = tree.all_rust();
        report.cover_complete("source files", all_rust.len());
        if let Some(note) = tree.absence_note() {
            report.note(note);
        }
        let test_modules = scan::test_module_files(&tree, &all_rust)?;
        // One read per file, kept for both halves: the discarded-read scan wants
        // production lines, and the reference count wants every line of every
        // file, including the test trees that are what make an item look live.
        let mut sources = Vec::with_capacity(all_rust.len());
        for path in all_rust {
            let text = tree.read(&path)?;
            let production = !is_outside_production(&path) && !test_modules.contains(&path);
            let lines = text.lines().collect::<Vec<&str>>();
            let test_only = scan::cfg_test_lines(&lines);
            let trait_impl = scan::trait_impl_lines(&lines);
            sources.push(Source {
                path,
                text,
                production,
                test_only,
                trait_impl,
            });
        }

        let mut declarations: BTreeMap<String, Vec<Declaration>> = BTreeMap::new();
        for source in &sources {
            for (index, line) in source.text.lines().enumerate() {
                if scan::is_comment(line) {
                    continue;
                }
                let test_side = source.test_side(index);
                // A trait-impl method is dispatched through the trait, so a
                // name scan finds no call site for one however live it is.
                if let Some(name) = declared_item_name(line) {
                    if !source.trait_impl.get(index).copied().unwrap_or(false) {
                        declarations.entry(name).or_default().push(Declaration {
                            file: source.path.clone(),
                            line: index,
                            test_side,
                        });
                    }
                }
                if test_side {
                    continue;
                }
                if let Some(read) = discarded_read(line) {
                    report.find(Finding::at(
                        source.path.clone(),
                        u32::try_from(index + 1).unwrap_or(u32::MAX),
                        format!("statement discards `{read}` and does nothing with it"),
                        "delete the statement, then give what it kept alive a real caller or \
                         delete that too; a discarded read satisfies dead_code without giving \
                         the item an owner",
                    ));
                }
            }
        }

        let candidates: BTreeMap<String, Declaration> = declarations
            .into_iter()
            .filter_map(|(name, mut sites)| {
                if sites.len() != 1 {
                    return None;
                }
                let site = sites.pop()?;
                (!site.test_side).then_some((name, site))
            })
            .collect();
        if candidates.is_empty() {
            return Err(GateError::new(
                "no uniquely declared restricted production item found",
                "run this gate inside the workspace checkout; a reference scan over an empty \
                 candidate set reports success forever",
            ));
        }
        report.note(format!(
            "{} uniquely declared restricted production item(s) reference-counted",
            candidates.len()
        ));

        let mut counts: BTreeMap<&str, (u32, u32)> = candidates
            .keys()
            .map(|name| (name.as_str(), (0, 0)))
            .collect();
        for source in &sources {
            for (index, line) in source.text.lines().enumerate() {
                if scan::is_comment(line) {
                    continue;
                }
                let test_side = source.test_side(index);
                for word in identifiers(line) {
                    let Some(entry) = counts.get_mut(word) else {
                        continue;
                    };
                    let site = &candidates[word];
                    if site.file == source.path && site.line == index {
                        continue;
                    }
                    if test_side {
                        entry.1 += 1;
                    } else {
                        entry.0 += 1;
                    }
                }
            }
        }

        for (name, (production_references, test_references)) in &counts {
            if *production_references > 0 || *test_references == 0 {
                continue;
            }
            let site = &candidates[*name];
            report.find(Finding::at(
                site.file.clone(),
                u32::try_from(site.line + 1).unwrap_or(u32::MAX),
                format!(
                    "`{name}` is referenced {test_references} time(s), every one of them from \
                     test code"
                ),
                "give the item a production caller or delete it; dead_code counts a \
                 #[cfg(test)] call as a use, so a test-only caller keeps an unowned item quiet",
            ));
        }
        Ok(report)
    }
}

/// One tracked Rust file, read once and classified once.
struct Source {
    /// Path relative to the checkout root.
    path: PathBuf,
    /// File contents.
    text: String,
    /// Whether the file is production source at all.
    production: bool,
    /// Which of its lines belong to a test-only item, by 0-based index.
    test_only: Vec<bool>,
    /// Which of its lines sit inside a trait impl, by 0-based index.
    trait_impl: Vec<bool>,
}

impl Source {
    /// Whether a line is test code, whichever of the two ways makes it so.
    fn test_side(&self, index: usize) -> bool {
        !self.production || self.test_only.get(index).copied().unwrap_or(false)
    }
}

/// Where a candidate item is declared.
struct Declaration {
    /// File holding the declaration.
    file: PathBuf,
    /// 0-based line of the declaration.
    line: usize,
    /// Whether the declaration itself is test code.
    test_side: bool,
}

/// The item a line declares, when it declares a restricted one.
///
/// Restricted visibility closes over this tree, so `pub(crate)` and `pub(super)`
/// are judged like a private item. Bare `pub` returns nothing: whether it has a
/// caller is a question about other checkouts.
fn declared_item_name(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with("//") {
        return None;
    }
    // Attempt syn Item parsing
    let item_candidate = if trimmed.ends_with(';') || trimmed.ends_with('{') {
        trimmed.to_string()
    } else {
        format!("{trimmed};")
    };
    if let Ok(item) = syn::parse_str::<syn::Item>(&item_candidate) {
        let is_bare_pub = matches!(&item, syn::Item::Fn(f) if matches!(f.vis, syn::Visibility::Public(_)))
            || matches!(&item, syn::Item::Struct(s) if matches!(s.vis, syn::Visibility::Public(_)))
            || matches!(&item, syn::Item::Enum(e) if matches!(e.vis, syn::Visibility::Public(_)))
            || matches!(&item, syn::Item::Trait(t) if matches!(t.vis, syn::Visibility::Public(_)))
            || matches!(&item, syn::Item::Type(t) if matches!(t.vis, syn::Visibility::Public(_)))
            || matches!(&item, syn::Item::Static(s) if matches!(s.vis, syn::Visibility::Public(_)))
            || matches!(&item, syn::Item::Const(c) if matches!(c.vis, syn::Visibility::Public(_)));
        if is_bare_pub {
            return None;
        }
        return match item {
            syn::Item::Fn(f) => Some(f.sig.ident.to_string()),
            syn::Item::Struct(s) => Some(s.ident.to_string()),
            syn::Item::Enum(e) => Some(e.ident.to_string()),
            syn::Item::Trait(t) => Some(t.ident.to_string()),
            syn::Item::Type(t) => Some(t.ident.to_string()),
            syn::Item::Static(s) => Some(s.ident.to_string()),
            syn::Item::Const(c) => Some(c.ident.to_string()),
            syn::Item::Union(u) => Some(u.ident.to_string()),
            _ => None,
        };
    }
    let mut rest = line.trim();
    match strip_restricted_visibility(rest) {
        Some(tail) => rest = tail,
        None if rest.starts_with("pub ") || rest.starts_with("pub(") => return None,
        None => {}
    }
    loop {
        let stripped = ["default ", "async ", "unsafe "]
            .iter()
            .find_map(|prefix| rest.strip_prefix(prefix));
        match stripped {
            Some(tail) => rest = tail.trim_start(),
            None => break,
        }
    }
    if let Some(tail) = rest.strip_prefix("const ") {
        let tail = tail.trim_start();
        return item_name(tail.strip_prefix("fn ").unwrap_or(tail));
    }
    for keyword in [
        "fn ", "static ", "struct ", "enum ", "trait ", "type ", "union ",
    ] {
        if let Some(tail) = rest.strip_prefix(keyword) {
            return item_name(tail);
        }
    }
    None
}

/// The remainder of a line after a restricted visibility qualifier.
fn strip_restricted_visibility(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("pub(")?;
    let close = rest.find(')')?;
    Some(rest[close + 1..].trim_start())
}

/// The leading identifier of a declaration's remainder.
fn item_name(rest: &str) -> Option<String> {
    let rest = rest.trim_start();
    let rest = rest.strip_prefix("mut ").map_or(rest, str::trim_start);
    let name: String = rest
        .chars()
        .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
        .collect();
    if name.is_empty() || name.starts_with(|character: char| character.is_ascii_digit()) {
        return None;
    }
    Some(name)
}

/// The path a statement discards, when discarding it is the whole statement.
fn discarded_read(line: &str) -> Option<String> {
    let rest = line.trim().strip_prefix("let _")?;
    let rest = match rest.strip_prefix(':') {
        Some(typed) => typed.split_once('=')?.1,
        None => rest.trim_start().strip_prefix('=')?,
    };
    let value = rest.trim().strip_suffix(';')?.trim();
    if value.is_empty() || value.contains(['(', ')', '!', '?', '&', '*', '[', '{', ' ']) {
        return None;
    }
    Some(value.to_string())
}

/// The identifiers of a line's code, with comments, keywords and string prose
/// removed, plus the inline format captures its string literals name.
///
/// A name inside prose or inside a message is not a reference to the item, and a
/// gate that counted one would report an item live because its own doc comment
/// mentions it. A keyword is not a name either: no item can be called `let`, so
/// a token that only the grammar can produce is not a reference to anything. An
/// inline format capture is the opposite case: `format!("{NAME}")` resolves
/// `NAME` in the surrounding scope and deleting the item breaks the build, so
/// the capture is a reference and is counted as one.
fn identifiers(line: &str) -> Vec<&str> {
    let mut found = Vec::new();
    let bytes = line.as_bytes();
    let mut index = 0usize;
    let mut in_string = false;
    while index < bytes.len() {
        let byte = bytes[index];
        if in_string {
            if byte == b'\\' {
                index += 2;
                continue;
            }
            if byte == b'"' {
                in_string = false;
                index += 1;
                continue;
            }
            if byte == b'{' {
                if bytes.get(index + 1) == Some(&b'{') {
                    index += 2;
                    continue;
                }
                let start = index + 1;
                let mut end = start;
                while end < bytes.len()
                    && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_')
                {
                    end += 1;
                }
                let token = &line[start..end];
                let capture = matches!(bytes.get(end), Some(b'}' | b':'))
                    && !token.is_empty()
                    && !token.as_bytes()[0].is_ascii_digit();
                if capture && !is_keyword(token) {
                    found.push(token);
                }
                index = end;
                continue;
            }
            index += 1;
            continue;
        }
        if byte == b'"' {
            in_string = true;
            index += 1;
            continue;
        }
        if byte == b'/' && bytes.get(index + 1) == Some(&b'/') {
            break;
        }
        if byte.is_ascii_alphabetic() || byte == b'_' {
            let start = index;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
            {
                index += 1;
            }
            let token = &line[start..index];
            if !is_keyword(token) {
                found.push(token);
            }
            continue;
        }
        index += 1;
    }
    found
}

/// Every strict and reserved Rust keyword, sorted.
///
/// A keyword cannot name an item, so a token the grammar owns is never a
/// reference to one. The contextual keywords are absent on purpose: `union`
/// and `macro_rules` are legal item names, and dropping a token that names a
/// real item would report that item dead.
const KEYWORDS: &[&str] = &[
    "Self", "abstract", "as", "async", "await", "become", "box", "break", "const", "continue",
    "crate", "do", "dyn", "else", "enum", "extern", "false", "final", "fn", "for", "if", "impl",
    "in", "let", "loop", "macro", "match", "mod", "move", "mut", "override", "priv", "pub", "ref",
    "return", "self", "static", "struct", "super", "trait", "true", "try", "type", "typeof",
    "unsafe", "unsized", "use", "virtual", "where", "while", "yield",
];

/// Whether `token` is a Rust keyword rather than a name.
fn is_keyword(token: &str) -> bool {
    KEYWORDS.binary_search(&token).is_ok()
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
/// `#![allow(unsafe_code)]` alone is the one accepted form, so an attribute that
/// bundles it with other lints is reported: the reviewed budget names files, and
/// a bundle makes the file's exception ambiguous.
fn inner_lint_attributes(text: &str) -> Vec<(u32, String)> {
    level_attributes(text, true)
        .into_iter()
        .filter(|(_, attribute)| attribute != "#![allow(unsafe_code)]")
        .collect()
}

/// Every attribute that sets a lint level, with its 1-based line.
///
/// `inner_only` keeps the crate-root and module-scoped `#![...]` form alone;
/// otherwise an outer `#[...]` on an item counts too, which is the form a
/// narrow suppression takes.
///
/// The attribute may span lines, and it is the level word that matters rather
/// than the lint names after it, so the scan reads the leading path of each
/// attribute and keeps the ones that are a level. `cfg_attr` is read too: the
/// levels it carries apply on the configurations it names, and a `deny` behind
/// `not(test)` is still policy declared outside the table.
fn level_attributes(text: &str, inner_only: bool) -> Vec<(u32, String)> {
    const LEVELS: [&str; 5] = ["allow", "warn", "deny", "forbid", "expect"];
    let lines: Vec<&str> = text.lines().collect();
    let mut found = Vec::new();
    let mut index = 0usize;
    while index < lines.len() {
        let trimmed = lines[index].trim_start();
        let inner = trimmed.starts_with("#![");
        if !inner && (inner_only || !trimmed.starts_with("#[")) {
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
        let body = attribute.trim_start_matches("#![").trim_start_matches("#[");
        let path = body
            .split(|character: char| !is_attribute_path_byte(character))
            .next()
            .unwrap_or_default();
        let is_level = LEVELS.contains(&path);
        let carries_level = path == "cfg_attr"
            && LEVELS
                .iter()
                .any(|level| body.contains(&format!("{level}(")));
        if is_level || carries_level {
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
    use crate::gate::GateBehavior;
    use crate::gates::fixture_checkout::{checkout, files, messages, sites};

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
            messages(&report)
        );
    }

    /// WHY: a whole file can be the test module, declared `#[cfg(test)] mod
    /// plain_tests;` or, with the file named outright, `#[cfg(test)] #[path =
    /// "renamed_tests.rs"] mod tests;`. The attribute is in the parent, so a scan
    /// that reads only the file sees shipped source, and 14 of the 16 findings
    /// this gate reported were fixture panics in such files. The production site
    /// in the same parent must still be reported, or the rule cannot fail.
    #[test]
    fn a_file_the_test_module_declaration_reaches_is_not_production() {
        let (_directory, root) = checkout(&[
            (
                "lib.rs",
                "pub fn load(path: &str) -> String {\n    std::fs::read_to_string(path).expect(\"the config file\")\n}\n\n#[cfg(test)]\nmod plain_tests;\n\n#[cfg(test)]\n#[path = \"renamed_tests.rs\"]\nmod tests;\n",
            ),
            (
                "plain_tests.rs",
                "fn fixture() -> String {\n    std::fs::read_to_string(\"fixture\").expect(\"a readable fixture\")\n}\n\n#[test]\nfn it_loads() {\n    assert!(!fixture().is_empty());\n}\n",
            ),
            (
                "renamed_tests.rs",
                "fn other_fixture() -> String {\n    std::fs::read_to_string(\"other\").expect(\"a second readable fixture\")\n}\n\n#[test]\nfn it_loads_again() {\n    assert!(!other_fixture().is_empty());\n}\n",
            ),
        ]);

        let report = ExpectHasFix
            .run(&GateCtx::new(root, Vec::new()))
            .expect("Fix: the gate must read the fixture tree; check the fixture git step");
        let reported = sites(&report);
        assert_eq!(
            reported,
            ["lib.rs:2"],
            "only the production site owes a corrective action: {:?}",
            messages(&report)
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
            report.named_files(),
            ["bare.rs"],
            "a quoted block is data, a wrapped justification is a justification: {:?}",
            report.named_files()
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
            report.named_files()
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

    /// WHY: the defect this closes is a module-scoped `#![allow(dead_code)]` in
    /// production, one directory below a crate root, which every gate in this
    /// file walked past because `OneLintPolicy` reads `src/lib.rs` and
    /// `src/main.rs` alone. The fixture puts one there and one on an item, and
    /// beside them the three shapes that must stay unreported: the accepted
    /// `expect` with a reason, an `allow` of a lint the table does not deny, and
    /// the same `allow` in a `tests/` tree, which is another lane's population.
    /// A scanner that could not tell those apart would either report nothing or
    /// report every harness in the workspace.
    #[test]
    fn a_production_allow_of_a_denied_lint_is_reported_and_an_expect_with_a_reason_is_not() {
        let (_directory, root) = checkout(&[
            (
                "Cargo.toml",
                "[workspace]\nmembers = [\"member\"]\n\n[workspace.lints.rust]\ndead_code = \"deny\"\nunused_variables = \"deny\"\nmissing_docs = \"deny\"\ndropping_copy_types = \"allow\"\n",
            ),
            (
                "member/Cargo.toml",
                "[package]\nname = \"member\"\n\n[lints]\nworkspace = true\n",
            ),
            ("member/src/lib.rs", "//! A member.\n"),
            (
                "member/src/buried.rs",
                "//! A module a crate-root scan never reads.\n#![allow(dead_code)]\n",
            ),
            (
                "member/src/item.rs",
                "//! An item-scoped override.\n\n#[allow(unused_variables)]\nfn one() {}\n",
            ),
            (
                "member/src/accepted.rs",
                "//! The accepted form.\n\n#[expect(dead_code, reason = \"the emitter below owns it\")]\nfn two() {}\n",
            ),
            (
                "member/src/undenied.rs",
                "//! A lint the table does not deny.\n\n#[allow(unused_mut)]\nfn three() {}\n",
            ),
            (
                "member/src/owned_elsewhere.rs",
                "//! A lint the unsafe budget and the crate-root scan answer for.\n#![allow(missing_docs)]\n",
            ),
            (
                "member/src/group.rs",
                "//! A group name reaching a denied lint.\n#![allow(unused)]\n",
            ),
            (
                "member/src/blanket.rs",
                "//! Every lint at once.\n#![allow(warnings)]\n",
            ),
            (
                "member/tests/harness.rs",
                "//! Another lane's population.\n#![allow(dead_code)]\n",
            ),
        ]);

        let report = DeniedLintOverride
            .run(&GateCtx::new(root, Vec::new()))
            .expect("the gate reads the fixture tree");
        assert_eq!(
            sites(&report),
            [
                "member/src/blanket.rs:2",
                "member/src/buried.rs:2",
                "member/src/group.rs:2",
                "member/src/item.rs:3",
            ],
            "naming a denied lint, naming a group that contains one, and naming every lint are \
             the same override, and nothing else in the fixture is one: {:?}",
            messages(&report)
        );
    }

    /// WHY: the lint set is the whole strength of the gate. Read as a written
    /// list it goes stale the first time the table changes, and this asserts the
    /// two directions that keep it derived: a level raised to `deny` joins the
    /// set, and a lint the table merely warns about does not.
    #[test]
    fn the_judged_lint_set_follows_the_table_rather_than_a_written_list() {
        let (_directory, root) = checkout(&[
            (
                "Cargo.toml",
                "[workspace]\nmembers = []\n\n[workspace.lints.rust]\ndead_code = \"deny\"\nunreachable_pub = { level = \"deny\", priority = -1 }\nunused_variables = \"warn\"\nunsafe_code = \"deny\"\n",
            ),
        ]);
        let tree = Tree::open(&root).expect("the fixture tree opens");
        let denied = denied_rust_lints(&tree).expect("the fixture manifest parses");
        assert_eq!(
            denied.iter().map(String::as_str).collect::<Vec<&str>>(),
            ["dead_code", "unreachable_pub"],
            "a denied lint in either spelling joins the set, a warned one stays out, and a lint \
             another gate owns is not judged twice"
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
            files(&report),
            [
                "own-table/Cargo.toml",
                "own-table/Cargo.toml",
                "root-override/src/lib.rs"
            ],
            "the divergent member is reported twice, once for the missing inheritance and once \
             for the table it declared instead, and the crate-root override is reported once: \
             {:?}",
            messages(&report)
        );
    }

    /// WHY: a discarded read is how `dead_code = "deny"` gets satisfied without
    /// giving anything an owner, and the neighbouring writes that look like one
    /// are not: a discarded `Result` is a real discard, and a discarded borrow
    /// exists for the drop. A scanner that could not tell them apart would
    /// either miss the marker reads, which is the shape the defect takes, or
    /// forbid discarding a fallible call.
    #[test]
    fn a_discarded_path_read_is_told_apart_from_a_discarded_call() {
        assert_eq!(
            discarded_read("        let _ = CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;"),
            Some("CU_STREAM_CAPTURE_MODE_THREAD_LOCAL".to_string())
        );
        assert_eq!(
            discarded_read("    let _: u32 = PROBE_MARKER;"),
            Some("PROBE_MARKER".to_string())
        );
        assert_eq!(
            discarded_read("    let _ = self.retained_field;"),
            Some("self.retained_field".to_string())
        );
        assert_eq!(discarded_read("    let _ = write_all(&mut sink);"), None);
        assert_eq!(discarded_read("    let _ = &guard;"), None);
        assert_eq!(discarded_read("    let _ = try_read()?;"), None);
        assert_eq!(discarded_read("    let _unused = MARKER;"), None);
        assert_eq!(discarded_read("    let value = MARKER;"), None);
    }

    /// WHY: the reference count is what makes a test-only caller visible, and it
    /// must read code rather than text. A name in a doc comment, in a trailing
    /// comment, or inside a panic message is not a call, and counting one would
    /// report an item live because its own documentation mentions it.
    #[test]
    fn identifiers_are_read_from_code_and_not_from_prose_or_messages() {
        assert_eq!(identifiers("    marker_probe();"), vec!["marker_probe"]);
        assert_eq!(
            identifiers("    let value = holder.marker_probe; // marker_probe again"),
            vec!["value", "holder", "marker_probe"]
        );
        assert_eq!(
            identifiers("    panic!(\"marker_probe is missing\");"),
            vec!["panic"]
        );
        assert!(identifiers("/// marker_probe is documented here").is_empty());
    }

    /// WHY: an inline format capture is a real reference. `format!("{MARKER}")`
    /// resolves `MARKER` in scope, so deleting the item breaks the build, yet
    /// the capture sits inside a string literal where the scanner drops prose.
    /// Missing it reported a production item as referenced only by tests, and
    /// the advice on that finding is to delete the item.
    #[test]
    fn an_inline_format_capture_counts_as_a_reference_but_an_escaped_brace_does_not() {
        assert_eq!(
            identifiers("    panic!(\"read it through `{MARKER}` instead\");"),
            vec!["panic", "MARKER"]
        );
        assert_eq!(
            identifiers("    format!(\"{marker_probe:?} and {0} and {}\");"),
            vec!["format", "marker_probe"]
        );
        assert_eq!(
            identifiers("    format!(\"{{MARKER}} stays literal\");"),
            vec!["format"]
        );
        assert_eq!(
            identifiers("    format!(\"{ MARKER } is not a capture\");"),
            vec!["format"]
        );
    }

    /// WHY: `is_keyword` binary-searches the table, so an unsorted entry is not
    /// found and the keyword it names is counted as a reference to an item. The
    /// miss is silent: the gate keeps passing and reports a dead item live.
    #[test]
    fn every_keyword_is_reachable_by_the_search_that_reads_them() {
        assert!(
            KEYWORDS.windows(2).all(|pair| pair[0] < pair[1]),
            "Fix: KEYWORDS must be sorted and free of duplicates for `binary_search`."
        );
        for keyword in KEYWORDS {
            assert!(
                is_keyword(keyword),
                "Fix: `{keyword}` is in the table and the search cannot find it."
            );
        }
        assert!(
            !is_keyword("union"),
            "Fix: a contextual keyword is a legal item name and must stay countable."
        );
    }

    /// WHY: bare `pub` cannot be judged from one checkout, and restricted
    /// visibility can. The keyword forms also have to be told apart, because
    /// `const fn` and `const` both start with `const` and only one of them
    /// declares a function.
    #[test]
    fn a_restricted_declaration_is_named_and_a_public_one_is_not() {
        assert_eq!(
            declared_item_name("fn helper(value: u32) -> u32 {"),
            Some("helper".to_string())
        );
        assert_eq!(
            declared_item_name("    pub(crate) const PROBE: u32 = 1;"),
            Some("PROBE".to_string())
        );
        assert_eq!(
            declared_item_name("pub(super) const fn folded() -> u32 {"),
            Some("folded".to_string())
        );
        assert_eq!(
            declared_item_name("    static mut COUNTER: u32 = 0;"),
            Some("COUNTER".to_string())
        );
        assert_eq!(declared_item_name("pub fn exported() {"), None);
        assert_eq!(declared_item_name("mod inner;"), None);
        assert_eq!(declared_item_name("    self.helper();"), None);
    }

    /// WHY: this is the mutation the gate exists for. `dead_code` counts the
    /// `#[cfg(test)]` call as a use, so the compiler is green on a production
    /// item no production caller reaches. The item beside it with a production
    /// caller must stay unreported, or the gate rejects every private helper the
    /// tree has.
    #[test]
    fn an_item_only_test_code_calls_is_reported_and_an_owned_one_is_not() {
        let (_directory, root) = checkout(&[(
            "lib.rs",
            "fn owned_helper() -> u32 {\n    7\n}\n\n\
             fn test_only_helper() -> u32 {\n    9\n}\n\n\
             pub fn entry() -> u32 {\n    owned_helper()\n}\n\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn it_runs() {\n        \
             assert_eq!(super::test_only_helper(), 9);\n        \
             assert_eq!(super::entry(), 7);\n    }\n}\n",
        )]);

        let report = LivenessEvasion
            .run(&GateCtx::new(root, Vec::new()))
            .expect("Fix: the gate must read the fixture tree; check the fixture git step");
        let reported = messages(&report);
        assert_eq!(
            reported.len(),
            1,
            "exactly one item is unowned: {reported:?}"
        );
        assert!(
            reported[0].contains("test_only_helper"),
            "the unowned item is the one only the test calls: {reported:?}"
        );
    }

    /// WHY: a marker read in production is the other half of the class, and the
    /// same write inside a test item is a fixture rather than a shipped
    /// pretence, so only the production one is reported.
    #[test]
    fn a_production_marker_read_is_reported_and_a_test_one_is_not() {
        let (_directory, root) = checkout(&[(
            "lib.rs",
            "const PROBE: u32 = 1;\n\n\
             pub fn entry() {\n    let _ = PROBE;\n}\n\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn it_runs() {\n        \
             let _ = super::PROBE;\n    }\n}\n",
        )]);

        let report = LivenessEvasion
            .run(&GateCtx::new(root, Vec::new()))
            .expect("Fix: the gate must read the fixture tree; check the fixture git step");
        let lines: Vec<u32> = report
            .findings
            .iter()
            .filter_map(|finding| finding.line)
            .collect();
        assert_eq!(
            lines,
            [4],
            "only the production discard is reported: {:?}",
            messages(&report)
        );
    }

    /// WHY: a trait method is reached through the trait, never by its own name,
    /// so counting identifier occurrences finds zero production callers for a
    /// live one. `BindingSlotSet::from_iter` in `vyre-driver` was reported that
    /// way: its production caller is a `.collect()` two lines above it, and the
    /// only literal `from_iter` in the tree was in a test.
    ///
    /// The fixture covers the whole dispatch family, not the reported member:
    /// `from_iter` behind `collect`, `from` behind `into`, `fmt` behind
    /// `to_string`, and `next` behind a `for` loop. An inherent method with the
    /// same shape stays reportable, because a name scan does see its callers.
    #[test]
    fn a_trait_method_is_not_judged_by_name_and_an_inherent_one_still_is() {
        let (_directory, root) = checkout(&[(
            "lib.rs",
            "use std::fmt;\n\n\
             struct Bag {\n    items: Vec<u32>,\n}\n\n\
             impl FromIterator<u32> for Bag {\n    \
             fn from_iter<T: IntoIterator<Item = u32>>(iter: T) -> Self {\n        \
             Self {\n            items: iter.into_iter().collect(),\n        }\n    }\n}\n\n\
             impl From<u32> for Bag {\n    fn from(value: u32) -> Self {\n        \
             Self {\n            items: vec![value],\n        }\n    }\n}\n\n\
             impl fmt::Display for Bag {\n    \
             fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {\n        \
             write!(f, \"{}\", self.items.len())\n    }\n}\n\n\
             impl Iterator for Bag {\n    type Item = u32;\n    \
             fn next(&mut self) -> Option<u32> {\n        self.items.pop()\n    }\n}\n\n\
             impl Bag {\n    fn inherent_helper(&self) -> usize {\n        \
             self.items.len()\n    }\n}\n\n\
             pub fn entry(value: u32) -> String {\n    \
             let bag: Bag = Bag::from(value);\n    \
             let collected: Bag = (0..3).collect();\n    \
             let mut total = 0usize;\n    for _ in collected {\n        total += 1;\n    }\n    \
             format!(\"{bag}{total}\")\n}\n\n\
             #[cfg(test)]\nmod tests {\n    use super::Bag;\n    #[test]\n    \
             fn it_runs() {\n        \
             let bag = Bag::from_iter([1u32]);\n        \
             assert_eq!(bag.inherent_helper(), 1);\n    }\n}\n",
        )]);

        let report = LivenessEvasion
            .run(&GateCtx::new(root, Vec::new()))
            .expect("Fix: the gate must read the fixture tree; check the fixture git step");
        let reported = messages(&report);
        for method in ["from_iter", "from", "fmt", "next"] {
            assert!(
                !reported.iter().any(|line| line.contains(&format!("`{method}`"))),
                "`{method}` is dispatched through its trait, so a name count cannot judge it: {reported:?}"
            );
        }
        assert_eq!(
            reported.len(),
            1,
            "only the inherent method is judged by name: {reported:?}"
        );
        assert!(
            reported[0].contains("inherent_helper"),
            "the inherent method's only caller is the test: {reported:?}"
        );
    }
}
