//! The gate that makes softening a gate impossible without saying so.
//!
//! Every other rule in this registry judges the product. This one judges the
//! registry. Every gate holds zero findings, so the number a gate tolerates
//! lives in its source: a ratchet constant that a one-character change relaxes.
//! The constant was reviewable and not checkable, so the cheapest way to make a
//! red gate green was to move the number rather than the code.
//!
//! Four clauses. One reads the registry as it stands: a registered gate no
//! subset contains. Three read the diff, because softening is a direction and a
//! direction needs a before: a limit that was raised, a target that was
//! lowered, and a floor that was lowered without recording what was measured.
//! The comparison is `--base REF` or `GITHUB_BASE_REF` against the merge base,
//! and the worktree against `HEAD` when neither is set, which is what a local
//! caller has before committing.
//!
//! A raised limit and a lowered floor are legal when the change states what it
//! measured: the constant's own doc comment changes in the same revision and
//! carries a number. A target has no such exception. A bound on what a gate
//! reads can grow, and recording the measurement is the cost; a target is what
//! the tree is aiming at, so lowering it is a claim about the product rather
//! than about the gate.
//!
//! The constant scan derives its files from the registry: every crate that
//! implements a registered gate, `xtask` included, contributes its tracked Rust
//! sources. A gate moved into a new crate is covered by being registered.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

use crate::gate::{Finding, GateCtx, GateError, RegisteredGate, Report};
use crate::gates::scan::Tree;
use crate::subcommands;

/// Every registered gate no subset contains.
///
/// A gate no subset contains is reachable only by running the whole registry,
/// so the domain that owns it never sees its verdict on its own.
#[must_use]
fn subset_failures(gate_names: &[&str]) -> Vec<String> {
    let subsets = subcommands::subsets();
    gate_names
        .iter()
        .filter(|name| !subsets.iter().any(|subset| subset.gates.contains(name)))
        .map(|name| format!(
            "gate `{name}` is registered and belongs to no subset; add it to the subset whose domain owns it, so its verdict reaches that domain instead of only the whole-registry run"
        ))
        .collect()
}

/// Which way a ratchet constant is allowed to move.
#[derive(Clone, Copy, Eq, PartialEq)]
enum Ratchet {
    /// A bound on what the gate tolerates. Raising it is softening.
    Limit,
    /// A count the tree must reach. Lowering it is softening.
    Floor,
    /// A count the tree is aiming at. Lowering it is softening and has no
    /// documented escape, because the number is the ambition rather than a
    /// measurement of what the gate can read.
    Target,
}

/// One ratchet constant as one revision declares it.
struct Constant {
    /// Repository-relative source file.
    file: String,
    /// One-based line of the `const` item.
    line: u32,
    /// Constant name, which is also its key across revisions.
    name: String,
    /// Declared value.
    value: u128,
    /// Contiguous `///` lines immediately above the item, joined with newlines.
    doc: String,
    /// Which way it may move.
    ratchet: Ratchet,
}

/// The gate that holds the registry to its own ratchets.
pub struct GateCanon;

impl crate::gate::GateBehavior for GateCanon {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let registry = subcommands::registry();
        let gate_names: Vec<&str> = registry.iter().map(|gate| gate.name()).collect();
        report.cover_complete("registered gates", gate_names.len());

        for message in subset_failures(&gate_names) {
            report.find(Finding::new(
                message,
                "declare an area on the gate's descriptor that a subset holds",
            ));
        }

        let sources = gate_sources(&ctx.root, &registry)?;
        let current = constants(&ctx.root, &sources, &mut report);
        report.note(format!(
            "{} gate source file(s), {} ratchet constant(s)",
            sources.len(),
            current.len()
        ));

        let named = crate::checkout::requested_base_ref(
            ctx.flag("--base"),
            std::env::var(crate::checkout::BASE_REF_VARIABLE).ok(),
        );
        let Some(base) = base_revision(&ctx.root, named.as_deref()) else {
            let reference = named.unwrap_or_default();
            report.find(Finding::new(
                format!("`{reference}` is not in this checkout, so no direction can be judged against it"),
                "fetch the base ref before running this gate: `actions/checkout` with `fetch-depth: 0`, or pass `--base REF` naming a ref this checkout holds",
            ));
            return Ok(report);
        };
        report.note(format!("compared against `{base}`"));

        report
            .findings
            .extend(constant_findings(&ctx.root, &base, &sources, &current)?);
        Ok(report)
    }
}

/// The revision every before-state is read from.
///
/// With a base ref it is the merge base with `HEAD`, which is what a pull
/// request proposes to change. Without one it is `HEAD`, so the comparison is
/// the uncommitted worktree, which is what a local caller is about to commit.
/// The name arrives as an argument and is resolved by
/// [`crate::checkout::resolvable_base_ref`], so this reader answers from its
/// arguments alone.
fn base_revision(root: &Path, named: Option<&str>) -> Option<String> {
    let Some(named) = named else {
        return revision_exists(root, "HEAD").then(|| "HEAD".to_string());
    };
    let reference = crate::checkout::resolvable_base_ref(root, named)?;
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["merge-base", &reference, "HEAD"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|found| !found.is_empty())
}

/// Whether this checkout resolves a revision to a commit.
fn revision_exists(root: &Path, reference: &str) -> bool {
    crate::checkout::git_ref_exists(root, reference)
}

/// One file as one revision holds it, or `None` when that revision has no such
/// file, which is what a newly added gate source looks like.
fn at_revision(root: &Path, revision: &str, relative: &str) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["show", &format!("{revision}:{relative}")])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Every tracked Rust source of every crate that implements a registered gate.
///
/// The crate set is the registry's own answer: a gate names the package that
/// runs it, and a gate `xtask` runs in process names none. Deriving it means a
/// gate moved into a new crate is scanned because it is registered, not because
/// somebody remembered to widen a list.
fn gate_sources(root: &Path, registry: &[RegisteredGate]) -> Result<Vec<String>, GateError> {
    let mut crates: BTreeSet<&str> = BTreeSet::new();
    crates.insert("xtask");
    for gate in registry {
        if gate.package() != "xtask" {
            crates.insert(gate.package());
        }
    }
    let tree = Tree::open(root)?;
    let mut directories = Vec::with_capacity(crates.len());
    for package in &crates {
        directories.push(format!("{}/src", tree.member_directory(package)?));
    }
    let roots: Vec<&str> = directories.iter().map(String::as_str).collect();
    let mut files: Vec<String> = tree
        .rust(&roots)?
        .into_iter()
        .map(|path| path.to_string_lossy().replace('\\', "/"))
        .collect();
    if files.is_empty() {
        return Err(GateError::new(
            format!(
                "no tracked Rust source under the {} gate crate(s) the registry names",
                crates.len()
            ),
            "run this gate inside a checkout that carries the gate crates",
        ));
    }
    files.sort();
    Ok(files)
}

/// Every ratchet constant declared by the working tree copy of each source.
///
/// A tracked file the working tree does not carry is a routine state during a
/// migration, and aborting on it made this gate unrunnable exactly when the
/// registry was moving. It is a finding instead: a constant in a file this run
/// cannot read is a constant no direction was judged for, so silence about it
/// would let a bound move behind a deletion.
fn constants(root: &Path, files: &[String], report: &mut Report) -> Vec<Constant> {
    let mut found = Vec::new();
    for file in files {
        match structure_gate::read_source_bounded(&root.join(file)) {
            Ok(text) => found.extend(constants_in(file, &text)),
            Err(error) => report.find(Finding::in_file(
                file,
                format!("git tracks this gate source and this run cannot read it: {error}"),
                "restore the file as UTF-8 text, or commit its deletion so the registry moves with it",
            )),
        }
    }
    found
}

/// Integer types a ratchet constant may be declared with.
const RATCHET_TYPES: &[&str] = &["usize", "u8", "u16", "u32", "u64", "u128", "i32", "i64"];

/// Every ratchet constant one source text declares.
///
/// Only an integer constant whose name marks a direction is read. `SHINGLE`,
/// `BATCH` and `CALL_DEPTH` are tuning of how a gate looks rather than of how
/// much it tolerates, and a string or tuple constant carries no direction at
/// all, so neither is a ratchet and neither is judged here.
fn constants_in(file: &str, text: &str) -> Vec<Constant> {
    let lines: Vec<&str> = text.lines().collect();
    let mut found = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let Some((name, value)) = integer_constant(line) else {
            continue;
        };
        let Some(ratchet) = classify(&name) else {
            continue;
        };
        found.push(Constant {
            file: file.to_string(),
            line: u32::try_from(index + 1).unwrap_or(u32::MAX),
            name,
            value,
            doc: doc_above(&lines, index),
            ratchet,
        });
    }
    found
}

/// One declaration with its visibility removed.
///
/// `pub`, `pub(crate)`, `pub(super)` and `pub(in path)` all read as the
/// declaration that follows them.
fn strip_visibility(code: &str) -> &str {
    let Some(rest) = code.strip_prefix("pub") else {
        return code;
    };
    let rest = match rest.strip_prefix('(') {
        Some(scoped) => match scoped.split_once(')') {
            Some((_, after)) => after,
            None => return code,
        },
        None => rest,
    };
    match rest.strip_prefix(' ') {
        Some(declaration) => declaration.trim_start(),
        None => code,
    }
}

/// The name and value of an integer `const` declaration on one line.
///
/// Visibility is stripped rather than listed. A ratchet declared
/// `pub(crate) const` or `pub(super) const` is the same bound as one declared
/// `pub const`, and reading only the two shortest spellings left every
/// module-visible bound free to move with nothing recorded.
fn integer_constant(line: &str) -> Option<(String, u128)> {
    let code = line.trim();
    let rest = strip_visibility(code).strip_prefix("const ")?;
    let (name, rest) = rest.split_once(':')?;
    let name = name.trim();
    if name.is_empty()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte == b'_' || byte.is_ascii_digit())
    {
        return None;
    }
    let (declared, literal) = rest.split_once('=')?;
    if !RATCHET_TYPES.contains(&declared.trim()) {
        return None;
    }
    let literal = literal.trim().trim_end_matches(';').trim();
    let digits: String = literal
        .chars()
        .filter(|character| *character != '_')
        .collect();
    digits
        .parse::<u128>()
        .ok()
        .map(|value| (name.to_string(), value))
}

/// Which way a constant of this name may move, or `None` when the name marks no
/// direction.
fn classify(name: &str) -> Option<Ratchet> {
    if name.contains("TARGET") {
        return Some(Ratchet::Target);
    }
    if name.contains("MAX_")
        || name.ends_with("_CAP")
        || name.ends_with("_BUDGET")
        || name.ends_with("_LIMIT")
    {
        return Some(Ratchet::Limit);
    }
    if name.contains("MIN_") || name.contains("FLOOR") {
        return Some(Ratchet::Floor);
    }
    None
}

/// The contiguous `///` block immediately above `index`.
fn doc_above(lines: &[&str], index: usize) -> String {
    let mut collected = Vec::new();
    let mut at = index;
    while at > 0 {
        at -= 1;
        let line = lines[at].trim();
        if line.starts_with("///") {
            collected.push(line.to_string());
            continue;
        }
        break;
    }
    collected.reverse();
    collected.join("\n")
}

/// Whether a doc comment records what was measured.
///
/// Two halves, both required. The text has to differ from what the base
/// revision carried, so a change that only moved the number is not excused by a
/// sentence written for the previous value; and it has to carry a decimal
/// number, which is the measurement the campaign rule asks for.
fn records_measurement(before: &str, after: &str) -> bool {
    after != before && after.chars().any(|character| character.is_ascii_digit())
}

/// Every ratchet constant that moved the lax way.
fn constant_findings(
    root: &Path,
    base: &str,
    sources: &[String],
    current: &[Constant],
) -> Result<Vec<Finding>, GateError> {
    let mut before: BTreeMap<(String, String), Constant> = BTreeMap::new();
    for file in sources {
        let Some(text) = at_revision(root, base, file) else {
            continue;
        };
        for constant in constants_in(file, &text) {
            before.insert((constant.file.clone(), constant.name.clone()), constant);
        }
    }
    let mut findings = Vec::new();
    for constant in current {
        let Some(was) = before.get(&(constant.file.clone(), constant.name.clone())) else {
            continue;
        };
        let name = &constant.name;
        let file = &constant.file;
        match constant.ratchet {
            Ratchet::Target if constant.value < was.value => findings.push(Finding::at(
                file,
                constant.line,
                format!(
                    "target `{name}` is {} against {} in `{base}`",
                    constant.value, was.value
                ),
                "raise the tree to the target instead of lowering the target; a target has no documented exception, because the number is what the tree is aiming at",
            )),
            Ratchet::Limit
                if constant.value > was.value
                    && !records_measurement(&was.doc, &constant.doc) =>
            {
                findings.push(Finding::at(
                    file,
                    constant.line,
                    format!(
                        "limit `{name}` is {} against {} in `{base}` and its doc comment is unchanged",
                        constant.value, was.value
                    ),
                    "record the measured value and why the bound moved in the constant's own doc comment, or leave the bound where it is",
                ));
            }
            Ratchet::Floor
                if constant.value < was.value
                    && !records_measurement(&was.doc, &constant.doc) =>
            {
                findings.push(Finding::at(
                    file,
                    constant.line,
                    format!(
                        "floor `{name}` is {} against {} in `{base}` and its doc comment carries no new measurement",
                        constant.value, was.value
                    ),
                    "record the measured count and the code that left the tree in the constant's own doc comment, or leave the floor where it is",
                ));
            }
            _ => {}
        }
    }
    Ok(findings)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: this reader resolved the event variable itself, so on a pull request
    /// it answered about the merge base to a caller who asked about the
    /// worktree. The name now arrives as an argument, and `None` means `HEAD`
    /// even on a runner whose event names a base branch.
    #[test]
    fn the_base_revision_is_head_when_no_name_is_passed_and_a_merge_base_when_one_is() {
        let root = structure_gate::workspace_root();

        assert_eq!(
            base_revision(&root, None),
            Some("HEAD".to_string()),
            "Fix: without a name the comparison is the uncommitted worktree"
        );
        let merge_base = base_revision(&root, Some("HEAD"))
            .expect("Fix: `HEAD` resolves here, so it has a merge base with itself");
        assert_eq!(
            merge_base.len(),
            40,
            "Fix: a base revision is the resolved commit, got `{merge_base}`"
        );
        assert_eq!(
            base_revision(&root, Some("a-branch-no-checkout-has-6f1c2a")),
            None,
            "Fix: a name that resolves neither locally nor on origin has no base"
        );
    }

    /// WHY: the direction is decided by the constant's name, and a name that
    /// carries no direction is not a ratchet. Reading a tuning constant as a
    /// limit would fail every change to how a gate looks at the tree.
    #[test]
    fn classifies_only_directional_names() {
        assert!(matches!(classify("MAX_LINES"), Some(Ratchet::Limit)));
        assert!(matches!(classify("CORE_MAX_LINES"), Some(Ratchet::Limit)));
        assert!(matches!(classify("OCCUPANT_CAP"), Some(Ratchet::Limit)));
        assert!(matches!(classify("UNSAFE_BUDGET"), Some(Ratchet::Limit)));
        assert!(matches!(classify("FLOOR"), Some(Ratchet::Floor)));
        assert!(matches!(classify("MIN_CASES"), Some(Ratchet::Floor)));
        assert!(matches!(classify("TARGET"), Some(Ratchet::Target)));
        assert!(classify("SHINGLE").is_none());
        assert!(classify("CALL_DEPTH").is_none());
        assert!(classify("BATCH").is_none());
    }

    /// WHY: the scan reads a `const` line and nothing else. A tuple constant
    /// whose name contains TARGET is not an integer ratchet, and reading it as
    /// one would report a move that cannot be expressed.
    #[test]
    fn reads_integer_constants_only() {
        assert_eq!(
            integer_constant("const FLOOR: usize = 175;"),
            Some(("FLOOR".to_string(), 175))
        );
        assert_eq!(
            integer_constant("pub const MAX_FILE_BYTES: u64 = 16_777_216;"),
            Some(("MAX_FILE_BYTES".to_string(), 16_777_216))
        );
        assert_eq!(
            integer_constant(r#"const SMOKE_TARGET: (&str, &str) = ("a", "b");"#),
            None
        );
        assert_eq!(integer_constant("const FIX: &str = \"x\";"), None);
        assert_eq!(integer_constant("let MAX_LINES: usize = 3;"), None);
    }

    /// WHY: a bound is a bound at every visibility. Reading only `const` and
    /// `pub const` left `pub(crate)` and `pub(super)` ratchets free to move
    /// with no measurement recorded, which is the one thing this gate exists to
    /// prevent, and the gate reported nothing because it never saw them.
    #[test]
    fn reads_a_bound_at_every_visibility() {
        for declaration in [
            "pub(crate) const MAX_LINES: usize = 400;",
            "pub(super) const MAX_LINES: usize = 400;",
            "pub(in crate::gates) const MAX_LINES: usize = 400;",
            "pub const MAX_LINES: usize = 400;",
            "const MAX_LINES: usize = 400;",
        ] {
            assert_eq!(
                integer_constant(declaration),
                Some(("MAX_LINES".to_string(), 400)),
                "{declaration}"
            );
        }
        assert_eq!(
            integer_constant("published const MAX_LINES: usize = 4;"),
            None
        );
    }

    /// WHY: the doc comment is the record of what was measured, so it is the
    /// block directly above the item and nothing further up. Taking a blank
    /// line or an attribute as part of it would let an unrelated comment excuse
    /// a move.
    #[test]
    fn reads_the_doc_block_directly_above() {
        let lines = vec![
            "/// unrelated",
            "",
            "/// measured at 12 files",
            "/// after the module left the tree",
            "const FLOOR: usize = 12;",
        ];
        assert_eq!(
            doc_above(&lines, 4),
            "/// measured at 12 files\n/// after the module left the tree"
        );
    }

    /// WHY: a doc comment that did not change cannot be a record of this
    /// change, and one with no number records no measurement. Both halves have
    /// to hold or the escape is a sentence anybody can leave in place forever.
    #[test]
    fn measurement_needs_a_new_number() {
        assert!(records_measurement("/// old", "/// measured 12 files"));
        assert!(!records_measurement("/// measured 12", "/// measured 12"));
        assert!(!records_measurement("/// old", "/// fewer files now"));
    }

    /// WHY: a gate no subset holds runs only in the whole-registry sweep, so
    /// its domain never sees its verdict. The agreeing case is asserted first:
    /// a rule that reports on a registry that already agrees says nothing when
    /// it disagrees.
    #[test]
    fn a_gate_no_subset_holds_fails() {
        let subset = subcommands::subsets()
            .into_iter()
            .next()
            .expect("the descriptor registry has at least one nonempty area");
        assert_eq!(subset_failures(&subset.gates), Vec::<String>::new());
        let failures = subset_failures(&[subset.gates[0], "unsubsetted-gate"]);
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(
            failures[0].contains("`unsubsetted-gate` is registered and belongs to no subset"),
            "{failures:?}"
        );
    }
}
