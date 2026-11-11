//! Where a launch rule and its truth data live, and the guard that keeps both
//! inside this repository.
//!
//! Both rule binaries resolved their trees as
//! `Path::new("../../../../../rules/launch")`, five levels up from whatever
//! directory the process happened to start in. In the monorepo that reached a
//! shared `rules/` tree; in this standalone workspace it reaches
//! `/media/.../Santh`, so `audit_rule_contracts` reported on a directory no
//! clone has and `scaffold_rule` created directories outside the repository.
//! Both now resolve from the repository root, which is the parent of the xtask
//! manifest directory, and every write is checked against it.
//!
//! Layout, owned here because the auditor and the scaffolder must agree on it:
//!
//! ```text
//! rules/launch/<slug>/CONTRACT.md
//! rules/launch/<slug>/truth/{positives,negatives,evasions,cross_file}/
//! rules/launch/<slug>/truth/{cve_replay,property,differential,e2e_cli}.toml
//! ```
//!
//! The truth data is TOML, not Rust. Nothing compiles a repository-root
//! `tests/` tree in this workspace, so the `property.rs` and `e2e_cli.rs`
//! placeholders the scaffolder used to write were tracked files that no cargo
//! target could compile, which is exactly what the `source-reachability` gate
//! now rejects.

use std::path::{Path, PathBuf};

/// The four case classes every launch rule owes truth data for.
pub const TRUTH_DIRS: [&str; 4] = ["positives", "negatives", "evasions", "cross_file"];

/// The four truth manifests a scaffolded rule starts with.
pub const TRUTH_FILES: [&str; 4] = [
    "cve_replay.toml",
    "property.toml",
    "differential.toml",
    "e2e_cli.toml",
];

/// Repository root, resolved from the directory this ran in.
///
/// Not from `CARGO_MANIFEST_DIR`: several checkouts share one target directory
/// and cargo reuses a binary across them, so a compiled-in path names whichever
/// tree built last. See `xtask::checkout`.
pub fn repo_root() -> PathBuf {
    crate::checkout::checkout_root()
}

/// Directory holding every launch rule.
pub fn launch_dir() -> PathBuf {
    repo_root().join("rules/launch")
}

/// Truth-data directory for one rule.
pub fn truth_dir(slug: &str) -> PathBuf {
    launch_dir().join(slug).join("truth")
}

/// Whether a path resolves outside the repository.
///
/// The paths this guards are built from the repository root, so unlike release
/// evidence, which is written relative to a manifest one level down, they have
/// earned no `..` at all: the first one leaves the tree. An absolute path is
/// rejected outright, and a path already rooted at `repo_root()` is accepted by
/// its prefix.
pub fn escapes_repository(path: &Path) -> bool {
    let root = repo_root();
    let relative = match path.strip_prefix(&root) {
        Ok(relative) => relative,
        Err(_) if path.is_absolute() => return true,
        Err(_) => path,
    };
    crate::output_arg::escapes_root(relative, 0)
}

/// Exit rather than touch a path outside the repository.
pub fn require_inside_repository(path: &Path) {
    if escapes_repository(path) {
        eprintln!(
            "Fix: `{}` resolves outside the repository at `{}`. A rule tree is written inside \
             this checkout or not at all.",
            path.display(),
            repo_root().display()
        );
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the class this closes is the historical five-level climb. Both
    /// binaries built `../../../../../rules/launch`, which in this standalone
    /// workspace is outside the repository, so the scaffolder created
    /// directories in whatever tree the checkout sits in and the auditor
    /// reported on them. It does not catch an in-repository path that names the
    /// wrong directory.
    #[test]
    fn paths_that_leave_the_repository_are_rejected() {
        assert!(escapes_repository(Path::new("../../../../../rules/launch")));
        assert!(escapes_repository(Path::new("../rules/launch")));
        assert!(escapes_repository(Path::new("rules/../../launch")));
        assert!(escapes_repository(Path::new("/tmp/rules/launch")));
        assert!(escapes_repository(&repo_root().join("../rules/launch")));

        assert!(!escapes_repository(Path::new("rules/launch")));
        assert!(!escapes_repository(Path::new("rules/launch/../launch")));
        assert!(!escapes_repository(&launch_dir()));
        assert!(!escapes_repository(&truth_dir("some-rule")));
    }

    /// WHY: the guard is only worth anything if the paths the binaries actually
    /// build pass it and stay under the root, slug included.
    #[test]
    fn the_scaffolded_layout_stays_under_the_repository_root() {
        let root = repo_root();
        assert!(launch_dir().starts_with(&root));
        for dir in TRUTH_DIRS {
            let path = truth_dir("example-rule").join(dir);
            assert!(path.starts_with(&root), "{} escaped", path.display());
            assert!(!escapes_repository(&path));
        }
        for file in TRUTH_FILES {
            assert!(!escapes_repository(&truth_dir("example-rule").join(file)));
        }
    }
}
