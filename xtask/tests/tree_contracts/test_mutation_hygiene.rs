//! Invariant: tests must never contain write-environment triggers.
//!
//! WHY: Ordinary tests are read-only verification gates. A test that mutates tracked files
//! during execution corrupts clean checkouts, introduces non-deterministic diffs, and
//! circumvents explicit descriptor-owned artifact generation (`--write`).

use std::fs;
use std::path::{Path, PathBuf};

use super::workspace_sources::workspace_root;

fn forbidden_write_prefix() -> String {
    let part_a = "VYRE";
    let part_b = "WRITE_";
    format!("{part_a}_{part_b}")
}

/// Collect all Rust source files under test directories across the monorepo.
fn all_test_sources(root: &Path) -> Vec<PathBuf> {
    let mut sources = Vec::new();
    for entry in walkdir::WalkDir::new(root) {
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        if !entry.file_type().is_file() {
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let rel = path.strip_prefix(root).unwrap_or(path);
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str.starts_with("target/") || rel_str.starts_with(".git/") {
            continue;
        }
        if rel_str.contains("/tests/")
            || rel_str.ends_with("/tests.rs")
            || rel_str.ends_with("/tests/mod.rs")
            || rel_str.starts_with("tests/")
        {
            sources.push(path.to_path_buf());
        }
    }
    sources.sort();
    sources
}

#[test]
fn tests_contain_no_write_environment_triggers() {
    let root = workspace_root();
    let test_sources = all_test_sources(&root);
    assert!(
        test_sources.len() >= 10,
        "Fix: test source scan returned too few files ({}); directory walk failed",
        test_sources.len()
    );

    let prefix = forbidden_write_prefix();
    let mut violations = Vec::new();
    for path in test_sources {
        let text = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("Fix: cannot read {}: {err}", path.display()));
        if text.contains(&prefix) {
            let rel = path.strip_prefix(&root).unwrap_or(&path);
            violations.push(format!(
                "{}: contains forbidden write trigger prefix `{prefix}`. Fix: normal tests must be read-only; use descriptor-owned xtask gates with `--write` for regeneration",
                rel.display()
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "Fix: tests must never contain write-environment triggers:\n{}",
        violations.join("\n")
    );
}
