//! API-boundary regression tests for production preprocess paths.

use std::fs;
use std::path::{Path, PathBuf};

fn collect_rs_files(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read preprocess source directory") {
        let entry = entry.expect("read preprocess source entry");
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, files);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            files.push(path);
        }
    }
}

#[test]
fn preprocess_does_not_export_cpu_named_execution_paths() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/parsing/c/preprocess");
    let mut files = Vec::new();
    collect_rs_files(&root, &mut files);

    let mut offenders = Vec::new();
    for path in files {
        let source = fs::read_to_string(&path).expect("read preprocess source file");
        for (line_idx, line) in source.lines().enumerate() {
            let has_public_cpu_fn =
                line.contains("pub fn ") && (line.contains("_cpu") || line.contains("cpu_"));
            let has_public_cpu_reexport =
                line.contains("pub use ") && (line.contains("_cpu") || line.contains("cpu_"));
            if has_public_cpu_fn || has_public_cpu_reexport {
                offenders.push(format!("{}:{}: {line}", path.display(), line_idx + 1));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "C preprocessor CPU-named APIs must stay private to explicit reference/parity tests:\n{}",
        offenders.join("\n")
    );
}
