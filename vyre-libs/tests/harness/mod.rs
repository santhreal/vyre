//! Source-reading and IR-fingerprint helpers the `vyre-libs` contract tests
//! share.
//!
//! This module is compiled once per including test binary, and no binary uses
//! every helper: `scan_cpu_api_boundary` wants only
//! `assert_no_cpu_named_api_exports`,
//! `blake3_compress_optimizer_idempotence_contract` only
//! `optimizer::assert_optimizer_is_idempotent`, and
//! `dedup_conv_ast_walk_family_guard` only the source readers. Each
//! unused-in-this-binary helper is live in a sibling binary, so `dead_code`
//! here reports the inclusion shape rather than an item with no caller. An
//! `expect` cannot state that: the lint fires in some binaries and not others,
//! and the fulfilled half would fail `unfulfilled_lint_expectations`.
#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

pub(crate) mod ir_fingerprint;
pub(crate) mod optimizer;

/// This crate's directory, resolved from the working directory at run time.
pub(crate) fn crate_dir() -> PathBuf {
    vyre_test_support::monorepo::vyre_workspace_root().join("vyre-libs")
}

/// Resolve a source path across the workspace domain crates.
pub(crate) fn resolve_source_path(path: &str) -> PathBuf {
    let workspace = vyre_test_support::monorepo::vyre_workspace_root();
    if workspace.join(path).exists() {
        return workspace.join(path);
    }
    if let Some(rest) = path.strip_prefix("src/") {
        if let Some((domain, _)) = rest.split_once('/') {
            let candidate = workspace.join(format!("vyre-libs-{domain}")).join(path);
            if candidate.exists() {
                return candidate;
            }
        } else {
            let candidate = workspace.join(format!("vyre-libs-{rest}")).join(path);
            if candidate.exists() {
                return candidate;
            }
        }
    }
    crate_dir().join(path)
}

pub(crate) fn crate_file(path: &str) -> String {
    let resolved = resolve_source_path(path);
    fs::read_to_string(&resolved).unwrap_or_else(|error| {
        panic!(
            "failed to read {path} (resolved at {}): {error}",
            resolved.display()
        );
    })
}

pub(crate) fn assert_no_cpu_named_api_exports(
    relative_root: &str,
    read_context: &str,
    extra_trait_markers: &[&str],
    failure_message: &str,
) {
    let root = resolve_source_path(relative_root);
    let mut files = Vec::new();
    collect_rs_files(&root, read_context, &mut files);

    let mut offenders = Vec::new();
    for path in files {
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {read_context} source file: {error}"));
        for (line_idx, line) in source.lines().enumerate() {
            if is_cpu_named_api_export(line, extra_trait_markers) {
                offenders.push(format!("{}:{}: {line}", path.display(), line_idx + 1));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "{failure_message}:\n{}",
        offenders.join("\n")
    );
}

fn collect_rs_files(dir: &Path, read_context: &str, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("read {read_context} source directory: {error}"))
    {
        let entry =
            entry.unwrap_or_else(|error| panic!("read {read_context} source entry: {error}"));
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, read_context, files);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            files.push(path);
        }
    }
}

fn is_cpu_named_api_export(line: &str, extra_trait_markers: &[&str]) -> bool {
    let has_cpu_name = line.contains("_cpu") || line.contains("cpu_");
    let public_cpu_fn = line.contains("pub fn ") && has_cpu_name;
    let public_cpu_reexport = line.contains("pub use ") && has_cpu_name;
    let trait_marker = line.trim_start().starts_with("fn ")
        && extra_trait_markers
            .iter()
            .any(|marker| line.contains(marker));

    public_cpu_fn || public_cpu_reexport || trait_marker
}
