//! WHY: closes the class where a new variant of a closed IR enum reaches the
//! reference oracle through a `_ =>` arm and produces a run-time refusal (or,
//! worse, a plausible wrong answer) instead of a compile error. A catch-all is
//! invisible: `cargo check` stays green when `BinOp` grows an operator, and the
//! defect surfaces as a differential failure against a backend that did
//! implement it.
//!
//! Two mechanisms, because the language only offers one of them here.
//!
//! Inside the crate, a match over a type this crate can see the whole of is
//! made exhaustive and the compiler owns it. Every `DataType` match in
//! `value.rs` and `oob.rs`, and `RuleConditionWitness`'s `PartialEq`, are now
//! written that way.
//!
//! Across the crate boundary the compiler cannot help. `BinOp`, `UnOp`,
//! `AtomicOp`, `Node` and `Expr` are `#[non_exhaustive]`, so rustc REQUIRES a
//! wildcard arm in every downstream match and rejects an exhaustive one. Those
//! wildcards are held by [`declared_variants_are_named_or_recorded`] instead:
//! it reads each declaration at run time and requires every variant to be
//! either named in `vyre-reference/src` or recorded in
//! [`UNIMPLEMENTED_IR_VARIANTS`] with a reason.
//!
//! [`catch_all_arms_match_the_allowance`] pins the remaining wildcards to an
//! exact per-file count, so a new one turns this suite RED until someone
//! records why it cannot be exhaustive.
//!
//! What this does NOT catch: whether the semantics behind a named arm are
//! right. Naming `BinOp::Add` proves a decision was recorded for it, not that
//! the addition is correct; the arithmetic contracts own that. It also does not
//! see a catch-all built by a macro that this crate expands from another crate,
//! because the scan reads this crate's source text.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use vyre_test_support::monorepo::vyre_workspace_root;
use vyre_test_support::{braced_body, read_source_file_bounded, top_level_variant_names};

/// Every catch-all arm left in `vyre-reference/src`, with the count each file
/// carries and why that scrutinee cannot be matched exhaustively.
///
/// A file absent from this table is required to carry zero catch-all arms.
const CATCH_ALL_ALLOWANCE: &[(&str, usize, &str)] = &[
    (
        "atomics.rs",
        1,
        "AtomicOp is #[non_exhaustive] in vyre-spec",
    ),
    (
        "composition_witness/encoding.rs",
        1,
        "scrutinee is a u8 hex digit",
    ),
    (
        "composition_witness/graph_dataflow.rs",
        2,
        "scrutinee is a numeric d-DNNF node tag",
    ),
    (
        "composition_witness/hash.rs",
        1,
        "scrutinee is a slice whose length the compiler does not bound",
    ),
    (
        "composition_witness/math_quant.rs",
        1,
        "scrutinee is a numeric sum-product node tag",
    ),
    (
        "composition_witness/parsing.rs",
        1,
        "scrutinee is a u32 the compiler does not narrow to two bits",
    ),
    (
        "composition_witness/pattern.rs",
        1,
        "scrutinee is a numeric bracket tag",
    ),
    (
        "composition_witness/reduction.rs",
        1,
        "scrutinee is a u32 bit count",
    ),
    (
        "composition_witness/text.rs",
        2,
        "scrutinee is a u8 UTF-8 lead byte",
    ),
    ("execution/call.rs", 1, "scrutinee is a &str type spelling"),
    (
        "execution/hashmap/mod.rs",
        1,
        "Expr is #[non_exhaustive] in vyre-foundation",
    ),
    (
        "execution/hashmap/step/node_step.rs",
        1,
        "Node is #[non_exhaustive] in vyre-foundation",
    ),
    (
        "execution/step_budget.rs",
        1,
        "Expr is #[non_exhaustive] in vyre-foundation",
    ),
    (
        "execution/typed_ops/float_ops.rs",
        2,
        "BinOp and UnOp are #[non_exhaustive] in vyre-spec",
    ),
    (
        "execution/typed_ops/mod.rs",
        9,
        "BinOp and UnOp are #[non_exhaustive] in vyre-spec",
    ),
    (
        "float16.rs",
        1,
        "scrutinee is a (u32, u32) exponent/fraction pair",
    ),
    (
        "interleaving.rs",
        1,
        "Node is #[non_exhaustive] in vyre-foundation",
    ),
];

/// Fewest source files a working scan of `vyre-reference/src` finds.
///
/// A scan that walked the wrong directory would find nothing, report zero
/// catch-all arms everywhere and agree with an allowance of zero. The floor
/// sits below the current file count so it catches a broken walk without
/// needing an edit whenever a module is added.
const SCANNED_FILE_FLOOR: usize = 40;

/// Variants of a `#[non_exhaustive]` IR enum the reference oracle does not
/// implement, with the reason each reaches a structured refusal instead.
///
/// A variant here is a recorded decision, not an omission. A variant in neither
/// this table nor the oracle source turns this suite RED.
const UNIMPLEMENTED_IR_VARIANTS: &[(&str, &str)] = &[
    (
        "BinOp::Shuffle",
        "a lane-collective operator, evaluated through Expr::SubgroupShuffle rather than binary dispatch",
    ),
    (
        "BinOp::Ballot",
        "a lane-collective operator, evaluated through Expr::SubgroupBallot rather than binary dispatch",
    ),
    (
        "BinOp::WaveReduce",
        "a lane-collective operator, evaluated through Expr::SubgroupReduce rather than binary dispatch",
    ),
    (
        "BinOp::WaveBroadcast",
        "a lane-collective operator, evaluated through Expr::SubgroupShuffle rather than binary dispatch",
    ),
    (
        "BinOp::Opaque",
        "an extension operator declares its own semantics, which the oracle refuses rather than guesses",
    ),
    (
        "UnOp::Opaque",
        "an extension operator declares its own semantics, which the oracle refuses rather than guesses",
    ),
    (
        "AtomicOp::CompareExchangeWeak",
        "spurious failure is a device property with no single sequential answer to certify",
    ),
    (
        "AtomicOp::FetchNand",
        "no backend lowers it, so the oracle has nothing to be an oracle for",
    ),
    (
        "AtomicOp::Opaque",
        "an extension atomic declares its own semantics, which the oracle refuses rather than guesses",
    ),
    (
        "Expr::BufferRef",
        "a buffer name in operand position, resolved by the node that holds it rather than evaluated to a value",
    ),
];

/// Fewest variants a working declaration scan finds for each enum.
///
/// A scan that located no declaration reports an empty variant set, and an
/// empty set satisfies every coverage assertion vacuously.
const VARIANT_FLOORS: &[(&str, usize)] = &[
    ("BinOp", 30),
    ("UnOp", 34),
    ("AtomicOp", 10),
    ("Node", 24),
    ("Expr", 24),
];

/// The reference oracle's source directory.
fn oracle_src() -> PathBuf {
    vyre_workspace_root().join("vyre-reference/src")
}

/// Every `.rs` file under `vyre-reference/src`, relative to that directory.
fn oracle_sources() -> Vec<String> {
    let root = oracle_src();
    let mut found = Vec::new();
    collect_rust_files(&root, &root, &mut found);
    found.sort();
    found
}

fn collect_rust_files(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("Fix: cannot read the oracle source tree {dir:?}: {err}"));
    for entry in entries {
        let path = entry.expect("Fix: unreadable directory entry").path();
        if path.is_dir() {
            collect_rust_files(root, &path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            let relative = path
                .strip_prefix(root)
                .expect("Fix: walked path is not under the oracle source root");
            out.push(relative.to_string_lossy().replace('\\', "/"));
        }
    }
}

/// `source` with every comment, string literal, raw string literal and
/// character literal replaced by spaces of the same length.
///
/// The scan counts syntax, so a `"_ =>"` inside a diagnostic message or a doc
/// comment describing a catch-all must not read as one. Line structure is
/// preserved so a reported line number points at the real arm.
fn strip_comments_and_literals(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = String::with_capacity(source.len());
    let mut index = 0usize;
    while index < bytes.len() {
        let rest = &source[index..];
        if rest.starts_with("//") {
            let end = rest.find('\n').map_or(source.len(), |offset| index + offset);
            blank_out(&source[index..end], &mut out);
            index = end;
            continue;
        }
        if rest.starts_with("/*") {
            let end = rest
                .get(2..)
                .and_then(|tail| tail.find("*/"))
                .map_or(source.len(), |offset| index + offset + 4);
            blank_out(&source[index..end], &mut out);
            index = end;
            continue;
        }
        if let Some(end) = raw_string_end(source, index) {
            blank_out(&source[index..end], &mut out);
            index = end;
            continue;
        }
        if rest.starts_with('"') {
            let end = quoted_end(source, index, '"');
            blank_out(&source[index..end], &mut out);
            index = end;
            continue;
        }
        if rest.starts_with('\'') {
            if let Some(end) = char_literal_end(source, index) {
                blank_out(&source[index..end], &mut out);
                index = end;
                continue;
            }
        }
        let ch = rest.chars().next().expect("non-empty remainder");
        out.push(ch);
        index += ch.len_utf8();
    }
    out
}

fn blank_out(span: &str, out: &mut String) {
    for ch in span.chars() {
        out.push(if ch == '\n' { '\n' } else { ' ' });
    }
}

/// End offset of the raw string literal starting at `index`, if there is one.
fn raw_string_end(source: &str, index: usize) -> Option<usize> {
    let rest = &source[index..];
    let after_r = rest.strip_prefix('r')?;
    let hashes = after_r.len() - after_r.trim_start_matches('#').len();
    let body = after_r[hashes..].strip_prefix('"')?;
    let terminator = format!("\"{}", "#".repeat(hashes));
    let end = body
        .find(&terminator)
        .map_or(source.len(), |offset| {
            index + 1 + hashes + 1 + offset + terminator.len()
        });
    Some(end)
}

/// End offset of the quoted literal starting at `index`.
fn quoted_end(source: &str, index: usize, quote: char) -> usize {
    let mut offset = index + quote.len_utf8();
    let bytes = source.as_bytes();
    while offset < bytes.len() {
        match bytes[offset] {
            b'\\' => offset += 2,
            byte if byte == quote as u8 => return offset + 1,
            _ => offset += 1,
        }
    }
    source.len()
}

/// End offset of the character literal starting at `index`, if this apostrophe
/// opens one rather than a lifetime.
fn char_literal_end(source: &str, index: usize) -> Option<usize> {
    let rest = &source[index..];
    let mut chars = rest.char_indices();
    chars.next()?;
    let (_, first) = chars.next()?;
    if first == '\\' {
        let (offset, _) = chars.find(|&(_, ch)| ch == '\'')?;
        return Some(index + offset + 1);
    }
    let (offset, closing) = chars.next()?;
    if closing == '\'' {
        return Some(index + offset + 1);
    }
    None
}

/// Catch-all match arms in `source`, as `(line, arm text)`.
///
/// Both spellings count: a bare `_ =>` and a guarded `_ if cond =>`. A `_` that
/// is a binding elsewhere (`let _ = ...`, `Some(_) => ...`) is not an arm and is
/// excluded by requiring the underscore to open the pattern.
fn catch_all_arms(source: &str) -> Vec<(usize, String)> {
    let stripped = strip_comments_and_literals(source);
    let mut found = Vec::new();
    for (index, line) in stripped.lines().enumerate() {
        let mut offset = 0usize;
        let bytes = line.as_bytes();
        while let Some(position) = line[offset..].find('_') {
            let start = offset + position;
            offset = start + 1;
            let preceding = line[..start].chars().next_back();
            if preceding.is_some_and(|ch| ch.is_alphanumeric() || ch == '_' || ch == '.') {
                continue;
            }
            let tail = line[start + 1..].trim_start();
            if bytes.get(start + 1).is_some_and(|byte| {
                byte.is_ascii_alphanumeric() || *byte == b'_' || *byte == b'\''
            }) {
                continue;
            }
            let is_arm = tail.starts_with("=>")
                || (tail.starts_with("if ") && tail.contains("=>") && !tail.contains("let "));
            if is_arm {
                found.push((index + 1, line.trim().to_string()));
                break;
            }
        }
    }
    found
}

/// Every catch-all arm left in the oracle is one the allowance records.
///
/// A new wildcard, in a listed file or an unlisted one, fails here with the
/// file and the line, because the allowance states an exact count rather than
/// a ceiling.
#[test]
fn catch_all_arms_match_the_allowance() {
    let sources = oracle_sources();
    assert!(
        sources.len() >= SCANNED_FILE_FLOOR,
        "Fix: the oracle source scan found only {} files under {:?}, below the floor of \
         {SCANNED_FILE_FLOOR}. A scan that finds nothing certifies nothing.",
        sources.len(),
        oracle_src()
    );

    let allowed: BTreeMap<&str, usize> = CATCH_ALL_ALLOWANCE
        .iter()
        .map(|&(file, count, _)| (file, count))
        .collect();
    assert_eq!(
        allowed.len(),
        CATCH_ALL_ALLOWANCE.len(),
        "Fix: CATCH_ALL_ALLOWANCE names the same file twice; one row per file."
    );

    let root = oracle_src();
    let mut observed: BTreeMap<String, Vec<(usize, String)>> = BTreeMap::new();
    for relative in &sources {
        let text = read_source_file_bounded(&root.join(relative)).unwrap_or_else(|err| {
            panic!("Fix: cannot read the oracle source file {relative}: {err}")
        });
        let arms = catch_all_arms(&text);
        if !arms.is_empty() {
            observed.insert(relative.clone(), arms);
        }
    }

    let mut failures = Vec::new();
    for (file, arms) in &observed {
        let expected = allowed.get(file.as_str()).copied().unwrap_or(0);
        if arms.len() != expected {
            let sites = arms
                .iter()
                .map(|(line, text)| format!("{file}:{line}: {text}"))
                .collect::<Vec<_>>()
                .join("\n    ");
            failures.push(format!(
                "{file} carries {} catch-all arm(s), the allowance states {expected}:\n    {sites}",
                arms.len()
            ));
        }
    }
    for &(file, count, _) in CATCH_ALL_ALLOWANCE {
        if !observed.contains_key(file) {
            failures.push(format!(
                "{file} carries no catch-all arm, the allowance states {count}. Fix: drop the row."
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "Fix: every catch-all arm in the reference oracle must be exhaustive or recorded in \
         CATCH_ALL_ALLOWANCE with the reason its scrutinee is open.\n{}",
        failures.join("\n")
    );
}

/// The variant names one enum declares, read from source at run time.
fn declared_variants(relative_path: &str, declaration: &str) -> BTreeSet<String> {
    let path = vyre_workspace_root().join(relative_path);
    let source = read_source_file_bounded(&path)
        .unwrap_or_else(|err| panic!("Fix: cannot read {relative_path}: {err}"));
    let body = braced_body(&source, declaration).unwrap_or_else(|| {
        panic!("Fix: no `{declaration}` declaration in {relative_path}; update this enumeration")
    });
    top_level_variant_names(body)
}

/// The `Node` and `Expr` variant names the AST registry declares.
///
/// These come from the generated constants rather than a source scan, because
/// the macro emits them from the same declaration it builds the enum out of.
fn registry_variants(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| (*name).to_string()).collect()
}

/// Concatenated source text of the whole reference oracle.
fn oracle_source_text() -> String {
    let root = oracle_src();
    let mut text = String::new();
    for relative in oracle_sources() {
        let file = read_source_file_bounded(&root.join(&relative))
            .unwrap_or_else(|err| panic!("Fix: cannot read {relative}: {err}"));
        text.push_str(&strip_comments_and_literals(&file));
        text.push('\n');
    }
    text
}

/// `true` when the oracle source names `Enum::Variant` in code.
fn names_variant(source: &str, path: &str) -> bool {
    let mut from = 0usize;
    while let Some(offset) = source[from..].find(path) {
        let start = from + offset;
        let end = start + path.len();
        let before = source[..start].chars().next_back();
        let after = source[end..].chars().next();
        let boundary_before = before.is_none_or(|ch| !ch.is_alphanumeric() && ch != '_');
        let boundary_after = after.is_none_or(|ch| !ch.is_alphanumeric() && ch != '_');
        if boundary_before && boundary_after {
            return true;
        }
        from = end;
    }
    false
}

/// Every variant of every `#[non_exhaustive]` IR enum reaching the oracle is
/// either handled by name or recorded as deliberately unimplemented.
///
/// This is the assertion the compiler cannot make. `#[non_exhaustive]` forces a
/// wildcard on every match in this crate, so rustc accepts a new operator
/// silently; the variant set is read from the declaration here instead, and a
/// variant nobody decided about fails.
#[test]
fn declared_variants_are_named_or_recorded() {
    let mut sets: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
    sets.insert(
        "BinOp",
        declared_variants("vyre-spec/src/bin_op.rs", "pub enum BinOp {"),
    );
    sets.insert(
        "UnOp",
        declared_variants("vyre-spec/src/un_op.rs", "pub enum UnOp {"),
    );
    sets.insert(
        "AtomicOp",
        declared_variants("vyre-spec/src/atomic_op.rs", "pub enum AtomicOp {"),
    );
    sets.insert(
        "Node",
        registry_variants(vyre_foundation::ir::NODE_VARIANT_NAMES),
    );
    sets.insert(
        "Expr",
        registry_variants(vyre_foundation::ir::EXPR_VARIANT_NAMES),
    );

    for &(name, floor) in VARIANT_FLOORS {
        let found = sets
            .get(name)
            .unwrap_or_else(|| panic!("Fix: VARIANT_FLOORS names {name}, which is not derived"))
            .len();
        assert!(
            found >= floor,
            "Fix: the {name} declaration scan found {found} variants, below the floor of {floor}. \
             A short variant set satisfies every coverage assertion vacuously."
        );
    }

    let recorded: BTreeMap<&str, &str> = UNIMPLEMENTED_IR_VARIANTS.iter().copied().collect();
    let source = oracle_source_text();

    let mut undecided = Vec::new();
    let mut all_paths = BTreeSet::new();
    for (enum_name, variants) in &sets {
        for variant in variants {
            let path = format!("{enum_name}::{variant}");
            all_paths.insert(path.clone());
            if recorded.contains_key(path.as_str()) {
                continue;
            }
            if !names_variant(&source, &path) {
                undecided.push(path);
            }
        }
    }
    assert!(
        undecided.is_empty(),
        "Fix: the reference oracle neither names nor records a decision for {undecided:?}. \
         Give each one explicit reference semantics in vyre-reference/src, or add a row to \
         UNIMPLEMENTED_IR_VARIANTS stating why the oracle refuses it."
    );

    let stale: Vec<&str> = UNIMPLEMENTED_IR_VARIANTS
        .iter()
        .map(|&(path, _)| path)
        .filter(|path| !all_paths.contains(*path))
        .collect();
    assert!(
        stale.is_empty(),
        "Fix: UNIMPLEMENTED_IR_VARIANTS records {stale:?}, which the declarations no longer \
         contain. Drop the row so the table stays a statement about the current IR."
    );

    let empty_reason: Vec<&str> = UNIMPLEMENTED_IR_VARIANTS
        .iter()
        .filter(|(_, reason)| reason.trim().is_empty())
        .map(|&(path, _)| path)
        .collect();
    assert!(
        empty_reason.is_empty(),
        "Fix: {empty_reason:?} is recorded as unimplemented with no reason."
    );
}

/// The scan reads syntax rather than text.
///
/// Without this, a diagnostic string containing `_ =>` would inflate a file's
/// count and the allowance would be adjusted to match a phantom arm.
#[test]
fn the_scan_ignores_catch_alls_inside_comments_and_strings() {
    let source = r##"
fn f(x: u8) -> u8 {
    // _ => 1,
    let message = "_ => 2,";
    let raw = r#"_ => 3,"#;
    let quote = '_';
    match x {
        0 => 0,
        _ => 4,
    }
}
"##;
    let arms = catch_all_arms(source);
    assert_eq!(
        arms.len(),
        1,
        "Fix: the catch-all scan counted commented or quoted text as an arm: {arms:?}"
    );
    assert!(
        arms[0].1.contains("=> 4"),
        "Fix: the scan located the wrong arm: {arms:?}"
    );
}

/// A guarded wildcard is a catch-all too.
///
/// `_ if cond => ...` absorbs every variant the guard admits, which is the same
/// defect with an extra condition on it.
#[test]
fn the_scan_counts_guarded_wildcards() {
    let source = "
fn f(x: u8) -> u8 {
    match x {
        0 => 0,
        _ if x > 3 => 1,
        _ => 2,
    }
}
";
    assert_eq!(
        catch_all_arms(source).len(),
        2,
        "Fix: the catch-all scan missed a guarded wildcard arm."
    );
}
