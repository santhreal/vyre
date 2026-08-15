//! Paths and text of the checked-in public-API snapshots.
//!
//! # ONE definitional home for "where is the frozen surface of crate X"
//!
//! `scripts/check_public_api_snapshot.sh` extracts every publishable crate's
//! rustdoc surface into `docs/public-api/<package>.txt`, and a byte-stability
//! gate holds each file equal to that crate's real surface. Several contract
//! tests derive an expected set from those files at run time, which is how a new
//! IR variant, capability field or evaluator impl reaches them through the
//! snapshot refresh instead of through a hand-typed list that goes stale in
//! silence.
//!
//! Each of those tests used to locate the file itself, by walking `ancestors()`
//! until some `docs/public-api/<name>.txt` existed. That walk was wrong in both
//! directions it was spelled. Started from a compiled-in `CARGO_MANIFEST_DIR` it
//! names whichever checkout built the test binary, because every checkout here
//! shares one cargo target directory; started from the workspace root it can
//! leave the checkout entirely and read a neighbouring clone's snapshot, then
//! report that clone's surface as this tree's frozen set. The root is resolved
//! once from the working directory and the file sits directly under it, so there
//! is nothing to search for.

use std::path::PathBuf;

use crate::monorepo::vyre_workspace_root;
use crate::read_source_file_bounded;

/// Directory holding one snapshot per publishable crate.
#[must_use]
pub fn snapshot_directory() -> PathBuf {
    vyre_workspace_root().join("docs").join("public-api")
}

/// Path of `package`'s snapshot.
///
/// # Panics
///
/// Panics when the snapshot is absent, naming the command that writes it. A
/// caller enumerating a frozen surface has no answer without the file, and a
/// missing file must not read as an empty surface.
#[must_use]
pub fn snapshot_path(package: &str) -> PathBuf {
    let path = snapshot_directory().join(format!("{package}.txt"));
    assert!(
        path.is_file(),
        "Fix: no public-API snapshot at {}. Write it with \
         `scripts/check_public_api_snapshot.sh --refresh {package}`.",
        path.display()
    );
    path
}

/// Text of `package`'s snapshot, read through the workspace's bounded reader.
///
/// # Panics
///
/// Panics when the snapshot is absent, unreadable, or larger than the shared
/// source-read cap.
#[must_use]
pub fn snapshot_text(package: &str) -> String {
    let path = snapshot_path(package);
    read_source_file_bounded(&path).unwrap_or_else(|error| {
        panic!(
            "Fix: the public-API snapshot at {} must be readable: {error}",
            path.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::{snapshot_directory, snapshot_path, snapshot_text};

    /// The snapshot directory is inside the checkout this run was invoked in.
    #[test]
    fn the_snapshot_directory_sits_under_the_resolved_workspace_root() {
        let directory = snapshot_directory();
        let root = crate::monorepo::vyre_workspace_root();
        assert!(
            directory.starts_with(&root),
            "Fix: the snapshot directory {} must sit under the resolved checkout {}.",
            directory.display(),
            root.display()
        );
        assert!(
            directory.is_dir(),
            "Fix: {} must exist; every publishable crate has a snapshot there.",
            directory.display()
        );
    }

    /// A package with no snapshot names the command that writes one.
    ///
    /// The failure a caller actually hits is a package renamed without its
    /// snapshot following, and a raw missing-path panic does not say what to run.
    #[test]
    #[should_panic(expected = "--refresh vyre-not-a-package")]
    fn a_package_with_no_snapshot_names_the_refresh_command() {
        let _ = snapshot_path("vyre-not-a-package");
    }

    /// Every snapshot on disk is readable through this owner and non-empty.
    ///
    /// The package list is read from the directory on each run, so a snapshot
    /// added tomorrow is judged tomorrow rather than inheriting a listed set.
    #[test]
    fn every_snapshot_on_disk_reads_back_non_empty() {
        let directory = snapshot_directory();
        let mut read = 0_usize;
        for entry in std::fs::read_dir(&directory).expect("Fix: the snapshot directory must be readable")
        {
            let path = entry.expect("Fix: a snapshot directory entry must be readable").path();
            if path.extension().is_none_or(|extension| extension != "txt") {
                continue;
            }
            let package = path
                .file_stem()
                .expect("Fix: a snapshot file must have a stem")
                .to_string_lossy()
                .into_owned();
            assert!(
                !snapshot_text(&package).trim().is_empty(),
                "Fix: the snapshot for {package} is empty, so any surface derived from it is \
                 empty too. Refresh it with `scripts/check_public_api_snapshot.sh --refresh \
                 {package}`."
            );
            read += 1;
        }
        assert!(
            read > 0,
            "Fix: {} holds no .txt snapshot, so this test proves nothing.",
            directory.display()
        );
    }
}
