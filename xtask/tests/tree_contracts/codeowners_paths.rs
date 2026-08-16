//! Every path `CODEOWNERS` claims to protect exists in this tree.
//!
//! WHY: GitHub matches a CODEOWNERS pattern against changed files and silently
//! ignores a pattern that matches nothing. A row for a path that was renamed or
//! deleted reads as protection and grants none, and nothing goes red. That
//! shipped: nine of fourteen rows named an older layout, including
//! `/spec/src/lib.rs`, `/conform/src/reference/` and two scripts described as
//! future files, so the review requirement on the algebraic laws and the CPU
//! oracle had been off for as long as those directories had their current
//! names.
//!
//! The rows are read from the file at run time, so a row added tomorrow is
//! judged tomorrow.

use std::fs;
use std::path::{Path, PathBuf};

use super::workspace_sources::workspace_root;

/// One ownership row: the pattern and the line it sits on.
struct Row {
    pattern: String,
    line: usize,
}

/// Every pattern in `CODEOWNERS`, in file order.
///
/// A row is a pattern followed by owners. Comments and blank lines carry no
/// pattern.
fn rows(text: &str) -> Vec<Row> {
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some(pattern) = trimmed.split_whitespace().next() else {
            continue;
        };
        rows.push(Row {
            pattern: pattern.to_string(),
            line: index + 1,
        });
    }
    rows
}

/// Whether `pattern` matches at least one path in the checkout.
///
/// A pattern without a wildcard names one path and is resolved directly. A
/// pattern with one is matched against every path under the segments before it,
/// which is what GitHub does for the `dir/*.rs` shape this file uses.
fn matches_a_path(root: &Path, pattern: &str) -> bool {
    let relative = pattern.trim_start_matches('/');
    let relative = relative.trim_end_matches('/');
    if relative.is_empty() {
        return true;
    }
    let Some((prefix, suffix)) = relative.split_once('*') else {
        return root.join(relative).exists();
    };
    let base = root.join(prefix.trim_end_matches('/'));
    walk(&base).iter().any(|path| {
        path.to_string_lossy().ends_with(suffix.trim_start_matches('*'))
    })
}

/// Every file under `base`, or nothing when `base` is not a directory.
fn walk(base: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let Ok(entries) = fs::read_dir(base) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(walk(&path));
        } else {
            found.push(path);
        }
    }
    found
}

#[test]
fn every_codeowners_pattern_names_a_path_this_tree_carries() {
    let root = workspace_root();
    let path = root.join(".github/CODEOWNERS");
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("Fix: {path:?} must be readable: {error}"));

    let dangling: Vec<String> = rows(&text)
        .into_iter()
        .filter(|row| !matches_a_path(&root, &row.pattern))
        .map(|row| format!("line {}: {}", row.line, row.pattern))
        .collect();

    assert!(
        dangling.is_empty(),
        "Fix: a CODEOWNERS row that matches nothing requires review of nothing. Point each row \
         at the path that now owns the concern, or delete the row:\n{}",
        dangling.join("\n")
    );
}

#[test]
fn codeowners_protects_itself_and_the_workflows_that_gate_the_tree() {
    let root = workspace_root();
    let path = root.join(".github/CODEOWNERS");
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("Fix: {path:?} must be readable: {error}"));
    let patterns: Vec<String> = rows(&text).into_iter().map(|row| row.pattern).collect();

    for required in ["/.github/CODEOWNERS", "/.github/workflows/"] {
        assert!(
            patterns.iter().any(|pattern| pattern == required),
            "Fix: `{required}` must carry an ownership row. Without it a contributor can remove \
             the review requirement in the same change that needs it."
        );
    }
}
