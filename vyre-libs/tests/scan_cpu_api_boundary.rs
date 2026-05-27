//! API-boundary regression tests for production scan paths.

use std::fs;
use std::path::{Path, PathBuf};

fn collect_rs_files(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read scan source directory") {
        let entry = entry.expect("read scan source entry");
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, files);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            files.push(path);
        }
    }
}

#[test]
fn scan_layer_does_not_export_cpu_named_execution_paths() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/scan");
    let mut files = Vec::new();
    collect_rs_files(&root, &mut files);

    let mut offenders = Vec::new();
    for path in files {
        let source = fs::read_to_string(&path).expect("read scan source file");
        for (line_idx, line) in source.lines().enumerate() {
            let public_cpu_fn =
                line.contains("pub fn ") && (line.contains("_cpu") || line.contains("cpu_"));
            let public_cpu_trait_fn =
                line.trim_start().starts_with("fn ") && line.contains("scan_cpu");
            let public_cpu_reexport =
                line.contains("pub use ") && (line.contains("_cpu") || line.contains("cpu_"));
            if public_cpu_fn || public_cpu_trait_fn || public_cpu_reexport {
                offenders.push(format!("{}:{}: {line}", path.display(), line_idx + 1));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "scan-layer CPU-named APIs must be explicit reference/parity internals:\n{}",
        offenders.join("\n")
    );
}
