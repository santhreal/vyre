//! Every committed proptest regression corpus is one a test replays.
//!
//! WHY: proptest replays a corpus only when failure persistence is on and the
//! corpus sits at the path proptest derives from the source file of the
//! `proptest!` block. Two defects broke that silently. The workspace cargo
//! config set `PROPTEST_DISABLE_FAILURE_PERSISTENCE`, so no corpus was read and
//! no new failure was written. And when tests moved between files and crates,
//! eighteen of twenty-five corpora stayed at their old paths, where no test
//! looks. Either way a counterexample that once failed stops being re-run, and
//! nothing goes red.
//!
//! The corpora and the `proptest!` sources are both read from the index at run
//! time, so a corpus left behind by tomorrow's move is judged tomorrow. What
//! this does not catch: a corpus whose source still holds a `proptest!` block
//! but no longer the test the seeds were recorded against. Seeds replay against
//! every block in the file, so such a corpus runs, against a different
//! strategy.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::workspace_sources::workspace_root;

/// Every path in the index, relative to `root`.
fn tracked_files(root: &Path) -> Vec<PathBuf> {
    let output = Command::new("git")
        .args(["ls-files", "-z"])
        .current_dir(root)
        .output()
        .expect("Fix: git must launch to list the tracked files");
    assert!(
        output.status.success(),
        "Fix: git ls-files failed in {root:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| PathBuf::from(String::from_utf8_lossy(path).into_owned()))
        .collect()
}

/// Whether `path` is a regression corpus in either layout proptest writes.
fn is_corpus(path: &Path) -> bool {
    let in_parallel_tree = path
        .components()
        .any(|component| component.as_os_str() == "proptest-regressions")
        && path.extension().is_some_and(|extension| extension == "txt");
    let beside_source = path
        .extension()
        .is_some_and(|extension| extension == "proptest-regressions");
    in_parallel_tree || beside_source
}

/// The corpus proptest reads and writes for a `proptest!` block in `source`.
///
/// Mirrors `FileFailurePersistence::SourceParallel("proptest-regressions")`,
/// the default: walk up from the source to the nearest directory holding
/// `lib.rs` or `main.rs`, and place the corpus under a `proptest-regressions`
/// sibling of that directory at the same relative path, with a `.txt`
/// extension. A source with no such ancestor inside the checkout, which is every
/// integration test, gets the corpus beside it with the extension
/// `proptest-regressions`.
fn corpus_for(root: &Path, source: &Path) -> PathBuf {
    let mut dir = source.to_path_buf();
    while dir.pop() && dir.starts_with(root) {
        if dir.join("lib.rs").is_file() || dir.join("main.rs").is_file() {
            let relative = source
                .strip_prefix(&dir)
                .expect("an ancestor of a path is a prefix of it");
            let mut corpus = dir.parent().unwrap_or(&dir).join("proptest-regressions");
            corpus.push(relative);
            corpus.set_extension("txt");
            return corpus;
        }
    }
    source.with_extension("proptest-regressions")
}

#[test]
fn every_committed_regression_corpus_sits_where_a_proptest_block_reads_it() {
    let root = workspace_root();
    let tracked = tracked_files(&root);

    let replayed: BTreeSet<PathBuf> = tracked
        .iter()
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .filter(|path| {
            fs::read_to_string(root.join(path)).is_ok_and(|text| text.contains("proptest!"))
        })
        .map(|source| corpus_for(&root, &root.join(source)))
        .collect();

    // A path deleted from disk but still in the index is a corpus nobody reads
    // either way; the commit that deletes it settles the index.
    let corpora: Vec<&PathBuf> = tracked
        .iter()
        .filter(|path| is_corpus(path) && root.join(path).is_file())
        .collect();
    assert!(
        !corpora.is_empty(),
        "Fix: no regression corpus was found in the index; the corpus layouts in `is_corpus` no \
         longer match the tree"
    );

    let unread: Vec<String> = corpora
        .into_iter()
        .filter(|corpus| !replayed.contains(&root.join(corpus)))
        .map(|corpus| corpus.display().to_string())
        .collect();
    assert!(
        unread.is_empty(),
        "Fix: no `proptest!` block reads these corpora, so their seeds never replay. Move each \
         one to the path proptest derives from the source file that now holds its test, or \
         delete it when that test no longer exists:\n{}",
        unread.join("\n")
    );
}

#[test]
fn failure_persistence_is_not_disabled_for_the_workspace() {
    assert!(
        std::env::var_os("PROPTEST_DISABLE_FAILURE_PERSISTENCE").is_none(),
        "Fix: PROPTEST_DISABLE_FAILURE_PERSISTENCE is set for this test run, so every \
         `proptest!` block skips its committed corpus and records no new failure. Delete it from \
         the `[env]` table of `.cargo/config.toml`, the workflow, or the shell that sets it."
    );
}
