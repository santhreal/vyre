//! The oracle runs the program as submitted, proved against the transform source.
//!
//! A parity oracle that shares a rewrite with the compiler cannot detect a
//! defect in that rewrite: both sides compute the same wrong answer and the
//! comparison is green. The oracle therefore calls no optimizer pass, no
//! lowering, no inliner, and no schedule selection.
//!
//! The forbidden set is derived from `vyre-foundation` source at run time
//! rather than listed here. A pass added to `optimizer`, `transform`, `lower`,
//! `schedule`, `execution_plan`, or `pass_math` joins the set on the next run,
//! so a new transform reaching the oracle turns this suite red without anyone
//! remembering to extend a list.
//!
//! Two foundation decisions stay shared and are pinned in
//! `oracle_foundation_dependencies`. Neither rewrites a program, and both are
//! outside the transform trees this scans.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Foundation module trees that hold a production program transform.
///
/// Each is a directory under `vyre-foundation/src`. A tree that disappears is
/// a failure rather than a silent reduction in coverage.
const TRANSFORM_TREES: &[&str] = &[
    "optimizer",
    "transform",
    "lower",
    "schedule",
    "execution_plan",
    "pass_math",
];

/// Lower bound on the derived forbidden-symbol count.
///
/// A derivation that silently matches nothing would pass every assertion
/// below. The floor is far under the count the trees carry today, so it fails
/// on a broken derivation and not on ordinary churn.
const MIN_DERIVED_SYMBOLS: usize = 100;

#[test]
fn no_reference_path_calls_a_production_transform() {
    let root = vyre_test_support::monorepo::vyre_workspace_root();
    let forbidden = derive_forbidden_symbols(&root);
    assert!(
        forbidden.len() >= MIN_DERIVED_SYMBOLS,
        "Fix: the transform scan derived only {} symbol(s) from {TRANSFORM_TREES:?}. The trees \
         moved and this contract is covering nothing.",
        forbidden.len()
    );

    let oracle = oracle_sources(&root);
    let defined_here = oracle_defined_names(&oracle);

    let mut violations = Vec::new();
    for (path, source) in &oracle {
        let code = strip_comments(source);
        for symbol in &forbidden {
            if defined_here.contains(symbol) {
                continue;
            }
            if !calls(&code, symbol) {
                continue;
            }
            violations.push(format!(
                "{}: calls the production transform `{symbol}`",
                display(&root, path)
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "Fix: the oracle must evaluate the program as submitted. Inline the semantics it needs \
         instead of calling the transform the oracle exists to check:\n{}",
        violations.join("\n")
    );
}

#[test]
fn no_reference_path_names_a_transform_module() {
    let root = vyre_test_support::monorepo::vyre_workspace_root();
    let oracle = oracle_sources(&root);

    let mut violations = Vec::new();
    for (path, source) in &oracle {
        let code = strip_comments(source);
        for tree in TRANSFORM_TREES {
            for prefix in ["vyre_foundation::", "foundation::"] {
                let qualified = format!("{prefix}{tree}");
                if names_path(&code, &qualified) {
                    violations.push(format!(
                        "{}: names the production transform module `{qualified}`",
                        display(&root, path)
                    ));
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "Fix: remove the transform module path from the oracle:\n{}",
        violations.join("\n")
    );
}

/// Every free function the transform trees export, keyed by name.
///
/// Only a module-level `pub fn` counts. An inherent method is reached through
/// a receiver whose type the oracle would also have to name, and its bare name
/// (`new`, `len`, `cost`) collides with ordinary vocabulary.
fn derive_forbidden_symbols(root: &Path) -> BTreeSet<String> {
    let mut symbols = BTreeSet::new();
    for tree in TRANSFORM_TREES {
        let dir = root.join("vyre-foundation/src").join(tree);
        assert!(
            dir.is_dir(),
            "Fix: `{}` is not a directory. Update TRANSFORM_TREES to the tree it became.",
            dir.display()
        );
        for (_, source) in rust_sources(&dir) {
            for line in source.lines() {
                if let Some(name) = module_level_pub_fn(line) {
                    symbols.insert(name.to_owned());
                }
            }
        }
    }
    symbols
}

/// The function name in a module-level `pub fn` line, if the line is one.
fn module_level_pub_fn(line: &str) -> Option<&str> {
    let rest = line
        .strip_prefix("pub fn ")
        .or_else(|| line.strip_prefix("pub const fn "))
        .or_else(|| line.strip_prefix("pub async fn "))?;
    let end = rest.find(|c: char| !c.is_alphanumeric() && c != '_')?;
    let name = &rest[..end];
    (!name.is_empty()).then_some(name)
}

/// Every function name the oracle defines itself.
///
/// A collision between an oracle function and a transform function is a name
/// in common, not a call across the boundary.
fn oracle_defined_names(sources: &[(PathBuf, String)]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for (_, source) in sources {
        for line in source.lines() {
            let trimmed = line.trim_start();
            for prefix in [
                "pub fn ",
                "fn ",
                "pub const fn ",
                "const fn ",
                "pub async fn ",
            ] {
                let Some(rest) = trimmed.strip_prefix(prefix) else {
                    continue;
                };
                if let Some(end) = rest.find(|c: char| !c.is_alphanumeric() && c != '_') {
                    names.insert(rest[..end].to_owned());
                }
                break;
            }
        }
    }
    names
}

fn oracle_sources(root: &Path) -> Vec<(PathBuf, String)> {
    let src = root.join("vyre-reference/src");
    let sources = rust_sources(&src);
    assert!(
        !sources.is_empty(),
        "Fix: no oracle source under `{}`.",
        src.display()
    );
    sources
}

fn rust_sources(dir: &Path) -> Vec<(PathBuf, String)> {
    let mut found = BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let entries = fs::read_dir(&current)
            .unwrap_or_else(|error| panic!("Fix: cannot read `{}`: {error}", current.display()));
        for entry in entries {
            let entry = entry.expect("Fix: a directory entry must be readable");
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let text = fs::read_to_string(&path).unwrap_or_else(|error| {
                    panic!("Fix: cannot read `{}`: {error}", path.display())
                });
                found.insert(path, text);
            }
        }
    }
    found.into_iter().collect()
}

/// The source with every `//` comment body removed.
///
/// A doc comment naming a transform is documentation, not a call. Line
/// structure is preserved so nothing else shifts.
fn strip_comments(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    for line in source.lines() {
        let keep = match line.find("//") {
            Some(at) if !in_string_literal(&line[..at]) => &line[..at],
            _ => line,
        };
        out.push_str(keep);
        out.push('\n');
    }
    out
}

/// Whether a `//` at the end of `prefix` sits inside a string literal.
fn in_string_literal(prefix: &str) -> bool {
    let mut open = false;
    let mut escaped = false;
    for c in prefix.chars() {
        if escaped {
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else if c == '"' {
            open = !open;
        }
    }
    open
}

/// Whether `code` calls `symbol`, as a whole word followed by an open paren.
fn calls(code: &str, symbol: &str) -> bool {
    occurrences(code, symbol).any(|end| code[end..].trim_start().starts_with('('))
}

/// Whether `code` names the module path `qualified` as a whole word.
fn names_path(code: &str, qualified: &str) -> bool {
    occurrences(code, qualified).any(|end| {
        code[end..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_alphanumeric() && c != '_')
    })
}

/// Byte offsets just past each whole-word occurrence of `needle` in `code`.
fn occurrences<'a>(code: &'a str, needle: &'a str) -> impl Iterator<Item = usize> + 'a {
    code.match_indices(needle).filter_map(move |(at, _)| {
        let before_ok = code[..at]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_alphanumeric() && c != '_' && c != ':');
        before_ok.then_some(at + needle.len())
    })
}

fn display(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}
