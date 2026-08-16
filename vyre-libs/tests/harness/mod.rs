use std::fs;
use std::path::{Path, PathBuf};

pub(crate) mod ir_fingerprint;
pub(crate) mod optimizer;

/// This crate's directory, resolved from the working directory at run time.
pub(crate) fn crate_dir() -> PathBuf {
    vyre_test_support::monorepo::vyre_workspace_root().join("vyre-libs")
}

pub(crate) fn crate_file(path: &str) -> String {
    fs::read_to_string(crate_dir().join(path)).unwrap_or_else(|error| {
        panic!("failed to read {path}: {error}");
    })
}

pub(crate) fn assert_contains_all(source: &str, needles: &[&str], message: &str) {
    let missing = needles
        .iter()
        .copied()
        .filter(|needle| !source.contains(needle))
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "{message} Missing required source fragment(s): {}",
        missing.join(" | ")
    );
}

pub(crate) fn assert_contains_none(source: &str, needles: &[&str], message: &str) {
    let present = needles
        .iter()
        .copied()
        .filter(|needle| source.contains(needle))
        .collect::<Vec<_>>();
    assert!(
        present.is_empty(),
        "{message} Forbidden source fragment(s): {}",
        present.join(" | ")
    );
}

pub(crate) fn assert_no_cpu_named_api_exports(
    relative_root: &str,
    read_context: &str,
    extra_trait_markers: &[&str],
    failure_message: &str,
) {
    let root = crate_dir().join(relative_root);
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
