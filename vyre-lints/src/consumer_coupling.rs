//! Consumer-name coupling guard.
//!
//! Platform crates and current public docs must describe capabilities, not
//! downstream products. Historical archives, tests, examples, and fixtures are
//! intentionally exempt because they may preserve migration context or consumer
//! integration examples.

use crate::{paths::workspace_relative, Violation, ViolationKind};
use anyhow::{Context, Result};
use std::path::Path;

const CONSUMER_NAMES: &[&str] = &["weir", "surgec", "gossan", "keyhog", "flare-native"];

const EXEMPT_PATH_FRAGMENTS: &[&str] = &[
    "/docs/archive/",
    "/docs/legacy/",
    "/.internals/",
    "/tests/",
    "/benches/",
    "/examples/",
    "/fixtures/",
    "/target/",
];

pub fn scan_tree(root: &Path) -> Result<Vec<Violation>> {
    let mut all = Vec::new();
    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|entry| entry.ok())
    {
        let path = entry.path();
        if !path.is_file() || !is_scanned_extension(path) {
            continue;
        }
        let workspace_rel = workspace_relative(path);
        if is_exempt_path(&workspace_rel) {
            continue;
        }
        all.extend(scan_file(path, &workspace_rel)?);
    }
    Ok(all)
}

fn is_scanned_extension(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("rs" | "md")
    )
}

fn is_exempt_path(workspace_rel: &str) -> bool {
    let wrapped = format!("/{workspace_rel}");
    EXEMPT_PATH_FRAGMENTS
        .iter()
        .any(|fragment| wrapped.contains(fragment))
}

fn scan_file(path: &Path, workspace_rel: &str) -> Result<Vec<Violation>> {
    let source =
        std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let is_markdown = path.extension().and_then(|ext| ext.to_str()) == Some("md");
    let mut violations = Vec::new();

    for (line_idx, line) in source.lines().enumerate() {
        if !is_markdown && !is_comment_line(line) {
            continue;
        }
        if let Some((column, name)) = find_consumer_name(line) {
            violations.push(Violation {
                file: workspace_rel.to_string(),
                line: (line_idx + 1) as u32,
                column: column as u32,
                kind: ViolationKind::ConsumerCoupling,
                message: format!(
                    "platform docs/comments mention downstream consumer `{name}`. Fix: use a capability name such as dataflow, static analysis, scan, or consumer integration."
                ),
            });
        }
    }
    Ok(violations)
}

fn is_comment_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("//")
        || trimmed.starts_with("/*")
        || trimmed.starts_with('*')
        || trimmed.starts_with("*/")
}

fn find_consumer_name(line: &str) -> Option<(usize, &'static str)> {
    let lower = line.to_ascii_lowercase();
    for name in CONSUMER_NAMES {
        let mut search_from = 0usize;
        while let Some(rel_idx) = lower[search_from..].find(name) {
            let idx = search_from + rel_idx;
            let end = idx + name.len();
            if is_start_boundary(lower.as_bytes(), idx)
                && is_end_boundary(lower.as_bytes(), end)
            {
                return Some((idx, *name));
            }
            search_from = end;
        }
    }
    None
}

fn is_start_boundary(bytes: &[u8], idx: usize) -> bool {
    if idx == 0 {
        return true;
    }
    !bytes[idx - 1].is_ascii_alphanumeric() && bytes[idx - 1] != b'_'
}

fn is_end_boundary(bytes: &[u8], idx: usize) -> bool {
    if idx >= bytes.len() {
        return true;
    }
    !bytes[idx].is_ascii_alphanumeric() && bytes[idx] != b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_boundary_rejects_substrings() {
        assert!(find_consumer_name("// weir dataflow").is_some());
        assert!(find_consumer_name("// weird control flow").is_none());
        assert!(find_consumer_name("// keyhog-style coupling").is_some());
    }
}
