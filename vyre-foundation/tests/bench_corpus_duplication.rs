//! Bench corpus duplication contract.
//!
//! Files under `benches/competition/corpora/` may duplicate bytes only through
//! the manifest-declared cumulative-prefix policy. They must not duplicate
//! normal test fixtures by content hash.

use std::collections::HashMap;
use std::path::PathBuf;
use vyre_test_support::monorepo::vyre_workspace_root;

/// WHY: this rule ran as a Python script the test spawned, and the test asserted
/// the script file was present before running it. The script returned success
/// whenever the corpus directory was absent, which it is, so the contract read
/// as covered while nothing hashed a byte. The rule is the same one the fixture
/// half already implements in this file, over the same hashes.
#[test]
fn checked_in_corpus_carries_each_program_once() {
    let workspace_root = vyre_workspace_root();
    let corpus = workspace_root.join("benches/competition/corpora");
    let mut hashes: HashMap<String, Vec<PathBuf>> = HashMap::new();
    if corpus.is_dir() {
        collect_hashes(&corpus, &mut hashes);
    }

    let duplicates = hashes
        .values()
        .filter(|paths| paths.len() > 1)
        .map(|paths| {
            paths
                .iter()
                .map(|path| {
                    path.strip_prefix(&workspace_root)
                        .unwrap_or(path)
                        .display()
                        .to_string()
                })
                .collect::<Vec<_>>()
                .join(" == ")
        })
        .collect::<Vec<_>>();

    assert!(
        duplicates.is_empty(),
        "bench corpus entries must be distinct by content. Duplicates:\n{}",
        duplicates.join("\n")
    );
}

#[test]
fn bench_corpus_does_not_duplicate_test_fixtures() {
    let workspace_root = vyre_workspace_root();

    let bench_corpus = workspace_root.join("benches/competition/corpora");
    let fixture_dirs = [
        workspace_root.join("vyre-libs/fixtures"),
        workspace_root.join("vyre-libs/tests/fixtures"),
        workspace_root.join("vyre-foundation/tests/fixtures"),
        workspace_root.join("vyre-driver/tests/fixtures"),
        workspace_root.join("vyre-primitives/tests/fixtures"),
    ];

    let mut bench_hashes: HashMap<String, Vec<PathBuf>> = HashMap::new();
    if bench_corpus.is_dir() {
        collect_hashes(&bench_corpus, &mut bench_hashes);
    }

    let mut fixture_hashes: HashMap<String, Vec<PathBuf>> = HashMap::new();
    for dir in &fixture_dirs {
        if dir.is_dir() {
            collect_hashes(dir, &mut fixture_hashes);
        }
    }

    let mut duplicates = Vec::new();
    for (hash, bench_paths) in &bench_hashes {
        if let Some(fix_paths) = fixture_hashes.get(hash) {
            for bp in bench_paths {
                for fp in fix_paths {
                    duplicates.push(format!(
                        "{} duplicates {}",
                        bp.strip_prefix(&workspace_root).unwrap().display(),
                        fp.strip_prefix(&workspace_root).unwrap().display()
                    ));
                }
            }
        }
    }

    assert!(
        duplicates.is_empty(),
        "bench corpus must not duplicate test fixtures by content. Duplicates:\n{}",
        duplicates.join("\n")
    );
}

fn collect_hashes(dir: &std::path::Path, map: &mut HashMap<String, Vec<PathBuf>>) {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        for entry in std::fs::read_dir(&current).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if let Ok(content) = std::fs::read(&path) {
                let hash = blake3::hash(&content).to_hex().to_string();
                map.entry(hash).or_default().push(path);
            }
        }
    }
}
