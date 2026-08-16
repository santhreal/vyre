//! The public-API stability gate and the roster it is taken over.
//!
//! This replaces a shell script that shelled out to a Python inventory. Two
//! things about it were load-bearing and are preserved: the extraction is sorted
//! in byte order so the snapshot is a function of the tree rather than of the
//! caller's locale, and a refresh prints the diff it is about to install so an
//! unintended bless is visible.
//!
//! A refresh reads the tree as it exists at that instant, so an unscoped refresh
//! installs every crate's current surface, including surface from in-flight work
//! nobody has reviewed. `--crate <name>` scopes it to one package.
//!
//! The extraction is taken with every feature enabled. A snapshot taken over the
//! default feature set promises stability for the default feature set only,
//! which is not what a file called a public-API snapshot claims: the whole
//! `graph` surface of `vyre-primitives`, including an exported macro, sat
//! outside it, and a lane deleted three public items there while the gate
//! stayed green.

use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// Directory holding one snapshot per publishable package.
pub const SNAPSHOT_DIR: &str = "docs/public-api";

/// One publishable package and where its sources live.
pub struct Snapshotted {
    /// Repository-relative member directory.
    pub directory: String,
    /// Package name, which is also the snapshot file stem.
    pub package: String,
}

/// Every publishable workspace package, ordered by package name.
///
/// Publishable is Cargo's own answer: `publish = false` and `publish = []` are
/// both out. The roster and the snapshot directory are one set, so a snapshot
/// naming a package that no longer publishes is a finding rather than a file
/// nobody notices.
pub fn roster(tree: &Tree) -> Result<Vec<Snapshotted>, GateError> {
    let mut rows: Vec<Snapshotted> = tree
        .member_manifests()?
        .into_iter()
        .filter(crate::gates::scan::Member::publishable)
        .map(|member| Snapshotted {
            directory: member.path,
            package: member.name,
        })
        .collect();
    rows.sort_by(|left, right| left.package.cmp(&right.package));
    if rows.is_empty() {
        return Err(GateError::new(
            "no publishable workspace package exists, so the stability gate would cover nothing",
            "restore a publishable member, or delete the gate rather than let it pass vacuously",
        ));
    }
    Ok(rows)
}

/// Every publishable crate's externally reachable API matches its snapshot.
pub struct PublicApiSnapshot;

impl crate::gate::GateBehavior for PublicApiSnapshot {
    fn usage(&self) -> &'static [&'static str] {
        &["--crate NAME judges one publishable crate instead of every one"]
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let rows = roster(&tree)?;
        let mut report = Report::clean();
        report.cover_complete("public api exports", rows.len());
        for row in &rows {
            report.produced(PathBuf::from(SNAPSHOT_DIR).join(format!("{}.txt", row.package)));
        }
        let scoped = ctx.flag("--crate").map(str::to_string);
        if let Some(name) = &scoped {
            if !rows
                .iter()
                .any(|row| &row.package == name || &row.directory == name)
            {
                return Err(GateError::new(
                    format!("`{name}` is not a snapshotted package"),
                    "name a publishable workspace package, or drop --crate to cover every one",
                ));
            }
        }
        let owned: BTreeSet<&str> = rows.iter().map(|row| row.package.as_str()).collect();
        for stale in unowned_snapshots(&ctx.root, &owned)? {
            report.find(Finding::in_file(
                stale.clone(),
                format!(
                    "`{}` promises a stable surface for a package that is not publishable",
                    stale.display()
                ),
                "delete the snapshot, or restore a publishable package with that name",
            ));
        }
        for row in &rows {
            if let Some(name) = &scoped {
                if &row.package != name && &row.directory != name {
                    continue;
                }
            }
            let source_root = format!("{}/src", row.directory);
            if !structure_gate::source_scan::carries_rust_source(&ctx.root.join(&source_root)) {
                report.find(Finding::in_file(
                    source_root.clone(),
                    format!(
                        "publishable package `{}` has no Rust source under `{source_root}`",
                        row.package
                    ),
                    "restore the source root, or stop publishing the package so the gate stops promising a surface for it",
                ));
                continue;
            }
            let current = extract(&ctx.root, &row.package)?;
            let snapshot = PathBuf::from(SNAPSHOT_DIR).join(format!("{}.txt", row.package));
            let absolute = ctx.root.join(&snapshot);
            let committed = match fs::read_to_string(&absolute) {
                Ok(text) => Some(text),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => {
                    return Err(GateError::new(
                        format!("cannot read `{}`: {error}", snapshot.display()),
                        "restore the snapshot as UTF-8 text",
                    ))
                }
            };
            if ctx.write {
                install(
                    &absolute,
                    &snapshot,
                    committed.as_deref(),
                    &current,
                    &mut report,
                )?;
                continue;
            }
            let Some(committed) = committed else {
                report.find(Finding::in_file(
                    snapshot,
                    format!("`{}` has no committed public-API snapshot", row.package),
                    "run the gate with --write and bump the crate version in the same change",
                ));
                continue;
            };
            if committed != current {
                report.find(Finding::in_file(
                    snapshot,
                    format!(
                        "the public API of `{}` no longer matches its snapshot: {}",
                        row.package,
                        summarize(&committed, &current)
                    ),
                    "refresh the snapshot with --write and record the change in the changelog in the same commit",
                ));
            }
        }
        Ok(report)
    }
}

/// Write one snapshot, recording what the write changed.
fn install(
    absolute: &Path,
    relative: &Path,
    committed: Option<&str>,
    current: &str,
    report: &mut Report,
) -> Result<(), GateError> {
    if committed == Some(current) {
        return Ok(());
    }
    if let Some(parent) = absolute.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            GateError::new(
                format!("cannot create `{}`: {error}", parent.display()),
                "make the snapshot directory writable",
            )
        })?;
    }
    fs::write(absolute, current).map_err(|error| {
        GateError::new(
            format!("cannot write `{}`: {error}", relative.display()),
            "make the snapshot writable",
        )
    })?;
    // The note is the whole guard against an unintended bless: a crate the
    // author did not touch showing a diff means somebody else's surface is
    // being blessed, and that is only visible if the write says so.
    report.note(match committed {
        Some(committed) => format!(
            "refreshed `{}`: {}",
            relative.display(),
            summarize(committed, current)
        ),
        None => format!(
            "wrote `{}`: new snapshot, {} items",
            relative.display(),
            current.lines().count()
        ),
    });
    Ok(())
}

/// How two snapshots differ, as added and removed item counts with a sample.
///
/// A line-set difference rather than an edit script: a public-API snapshot is a
/// sorted set of items, so what a reader needs is which items appeared and which
/// vanished, not where the file's lines moved.
fn summarize(committed: &str, current: &str) -> String {
    let before: BTreeSet<&str> = committed.lines().collect();
    let after: BTreeSet<&str> = current.lines().collect();
    let added: Vec<&str> = after.difference(&before).copied().collect();
    let removed: Vec<&str> = before.difference(&after).copied().collect();
    let mut sample = Vec::new();
    for item in removed.iter().take(3) {
        sample.push(format!("-{item}"));
    }
    for item in added.iter().take(3) {
        sample.push(format!("+{item}"));
    }
    format!(
        "{} added, {} removed ({})",
        added.len(),
        removed.len(),
        sample.join(", ")
    )
}

/// Snapshot files naming no publishable package.
fn unowned_snapshots(root: &Path, owned: &BTreeSet<&str>) -> Result<Vec<PathBuf>, GateError> {
    let directory = root.join(SNAPSHOT_DIR);
    let mut stale = Vec::new();
    let entries = match fs::read_dir(&directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(GateError::new(
                format!("cannot list `{SNAPSHOT_DIR}`: {error}"),
                "restore the snapshot directory",
            ))
        }
    };
    for entry in entries {
        let entry = entry.map_err(|error| {
            GateError::new(
                format!("cannot list `{SNAPSHOT_DIR}`: {error}"),
                "restore the snapshot directory",
            )
        })?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("txt") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
            continue;
        };
        if !owned.contains(stem) {
            stale.push(PathBuf::from(SNAPSHOT_DIR).join(format!("{stem}.txt")));
        }
    }
    stale.sort();
    Ok(stale)
}

/// The externally reachable API of one package, with every feature enabled.
///
/// `-p` is load-bearing: the workspace root is a virtual manifest and
/// `cargo public-api` refuses to list one, so without the package every crate
/// fails with the same unrelated message. Sorting here rather than through the
/// caller's `sort` is what makes the snapshot a function of the tree: byte order
/// does not move with a locale.
fn extract(root: &Path, package: &str) -> Result<String, GateError> {
    let cargo = crate::cargo_runner::binary(root);
    let output = Command::new(&cargo)
        .args([
            "public-api",
            "-sss",
            "--all-features",
            "-p",
            package,
        ])
        .current_dir(root)
        .output()
        .map_err(|error| {
            GateError::new(
                format!("cannot run `{} public-api -p {package}`: {error}", cargo.display()),
                "install cargo-public-api, and restore the cargo_full wrapper at the workspace root",
            )
        })?;
    if !output.status.success() {
        return Err(GateError::new(
            format!(
                "`cargo public-api -p {package}` exited {}: {}",
                output.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            format!("build `{package}` alone and rerun; a parallel build against the shared target directory can leave a zero-byte .rmeta, which reads as a crate with no items"),
        ));
    }
    let listed = String::from_utf8_lossy(&output.stdout);
    let items: BTreeSet<&str> = listed
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.is_empty())
        .collect();
    if items.is_empty() {
        // No committed snapshot is empty, so "no public items" is never the
        // answer for a publishable package: it is a truncated rustdoc or a
        // zero-byte .rmeta, and skipping it silently is how a crate drops out
        // of the gate unnoticed.
        return Err(GateError::new(
            format!(
                "`{package}` extracted no public item, and a publishable crate exports something"
            ),
            format!("build `{package}` alone and rerun"),
        ));
    }
    let mut text = String::new();
    for item in items {
        text.push_str(item);
        text.push('\n');
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::GateBehavior;
    use crate::gates::fixture_checkout;

    /// WHY: the roster and the snapshot directory are one set, and the two ways
    /// Cargo says a package does not publish are `false` and `[]`. A package
    /// excluded by one spelling and included by the other is a stability
    /// promise nobody makes.
    #[test]
    fn the_roster_is_every_package_cargo_would_publish() {
        let (_temporary, root) = fixture_checkout::checkout(&[
            (
                "Cargo.toml",
                "[workspace]\nresolver = \"2\"\nmembers = [\"zed\", \"alpha\", \"off\", \"empty\", \"registry\"]\n",
            ),
            (
                "zed/Cargo.toml",
                "[package]\nname = \"zed\"\nversion = \"0.0.0\"\n",
            ),
            (
                "alpha/Cargo.toml",
                "[package]\nname = \"alpha\"\nversion = \"0.0.0\"\n",
            ),
            (
                "off/Cargo.toml",
                "[package]\nname = \"off\"\nversion = \"0.0.0\"\npublish = false\n",
            ),
            (
                "empty/Cargo.toml",
                "[package]\nname = \"empty\"\nversion = \"0.0.0\"\npublish = []\n",
            ),
            (
                "registry/Cargo.toml",
                "[package]\nname = \"registry\"\nversion = \"0.0.0\"\npublish = [\"crates-io\"]\n",
            ),
        ]);
        let tree = Tree::open(&root).unwrap();
        let names: Vec<String> = roster(&tree)
            .unwrap()
            .into_iter()
            .map(|row| row.package)
            .collect();
        assert_eq!(names, vec!["alpha", "registry", "zed"]);
    }

    /// WHY: a gate over an empty roster passes forever, and the roster is
    /// derived, so an empty one is a broken derivation rather than a clean tree.
    #[test]
    fn an_empty_roster_stops_the_gate_instead_of_passing_it() {
        let (_temporary, root) = fixture_checkout::checkout(&[
            (
                "Cargo.toml",
                "[workspace]\nresolver = \"2\"\nmembers = [\"off\"]\n",
            ),
            (
                "off/Cargo.toml",
                "[package]\nname = \"off\"\nversion = \"0.0.0\"\npublish = false\n",
            ),
        ]);
        let tree = Tree::open(&root).unwrap();
        assert!(roster(&tree).is_err());
    }

    /// WHY: a snapshot for a package that stopped publishing is worse than a
    /// missing one, because the file reads as a promise nobody is holding.
    #[test]
    fn a_snapshot_for_an_unpublished_package_is_a_finding() {
        let (_temporary, root) = fixture_checkout::checkout(&[
            ("docs/public-api/gone.txt", "pub fn gone()\n"),
            ("docs/public-api/alpha.txt", "pub fn alpha()\n"),
            ("docs/public-api/notes.md", "prose\n"),
        ]);
        let owned: BTreeSet<&str> = ["alpha"].into_iter().collect();
        assert_eq!(
            unowned_snapshots(&root, &owned).unwrap(),
            vec![PathBuf::from("docs/public-api/gone.txt")]
        );
    }

    /// WHY: the difference a refresh prints is the only thing that makes an
    /// unintended bless visible, so it must name items rather than line offsets,
    /// and it must count both directions. A summary that reported only additions
    /// would hide exactly the removal a stability gate exists to catch.
    #[test]
    fn a_difference_names_the_items_that_appeared_and_vanished() {
        let summary = summarize("pub fn a()\npub fn b()\n", "pub fn b()\npub fn c()\n");
        assert!(summary.starts_with("1 added, 1 removed"), "{summary}");
        assert!(summary.contains("-pub fn a()"), "{summary}");
        assert!(summary.contains("+pub fn c()"), "{summary}");

        let unchanged = summarize("pub fn a()\n", "pub fn a()\n");
        assert!(unchanged.starts_with("0 added, 0 removed"), "{unchanged}");
    }

    /// WHY: a write that says nothing is how somebody else's in-flight surface
    /// gets blessed, and a write that rewrites an identical file churns the
    /// tree. Both directions are the contract of the refresh path.
    #[test]
    fn a_refresh_reports_what_it_installed_and_leaves_an_identical_file_alone() {
        let (_temporary, root) =
            fixture_checkout::checkout(&[("docs/public-api/alpha.txt", "pub fn a()\n")]);
        let absolute = root.join("docs/public-api/alpha.txt");
        let relative = Path::new("docs/public-api/alpha.txt");

        let mut report = Report::default();
        install(
            &absolute,
            relative,
            Some("pub fn a()\n"),
            "pub fn a()\n",
            &mut report,
        )
        .unwrap();
        assert!(report.notes.is_empty(), "{:?}", report.notes);

        let mut report = Report::default();
        install(
            &absolute,
            relative,
            Some("pub fn a()\n"),
            "pub fn a()\npub fn b()\n",
            &mut report,
        )
        .unwrap();
        assert_eq!(report.notes.len(), 1);
        assert!(
            report.notes[0].contains("1 added, 0 removed"),
            "{:?}",
            report.notes
        );
        assert_eq!(
            fs::read_to_string(&absolute).unwrap(),
            "pub fn a()\npub fn b()\n"
        );

        let fresh = root.join("docs/public-api/beta.txt");
        let mut report = Report::default();
        install(
            &fresh,
            Path::new("docs/public-api/beta.txt"),
            None,
            "pub fn b()\n",
            &mut report,
        )
        .unwrap();
        assert!(
            report.notes[0].contains("new snapshot, 1 items"),
            "{:?}",
            report.notes
        );
    }

    /// WHY: a missing source root under a publishable package means the gate is
    /// promising a surface for a crate whose code is gone, and the roster alone
    /// cannot see that.
    #[test]
    fn a_publishable_package_without_a_source_root_is_a_finding() {
        let (_temporary, root) = fixture_checkout::checkout(&[
            (
                "Cargo.toml",
                "[workspace]\nresolver = \"2\"\nmembers = [\"gone\"]\n",
            ),
            (
                "gone/Cargo.toml",
                "[package]\nname = \"gone\"\nversion = \"0.0.0\"\n",
            ),
        ]);
        let report = PublicApiSnapshot
            .run(&GateCtx::new(root, Vec::new()))
            .unwrap();
        assert_eq!(report.count(), 1, "{:?}", report.findings);
        assert!(
            report.findings[0]
                .message
                .contains("publishable package `gone` has no Rust source under `gone/src`"),
            "{:?}",
            report.findings[0].message
        );
    }

    /// WHY: `--crate` is the guard against an unscoped refresh, so a name that
    /// belongs to no package must stop the gate rather than silently cover
    /// everything or nothing.
    #[test]
    fn an_unknown_scope_stops_the_gate() {
        let (_temporary, root) = fixture_checkout::checkout(&[
            (
                "Cargo.toml",
                "[workspace]\nresolver = \"2\"\nmembers = [\"alpha\"]\n",
            ),
            (
                "alpha/Cargo.toml",
                "[package]\nname = \"alpha\"\nversion = \"0.0.0\"\n",
            ),
            ("alpha/src/lib.rs", ""),
        ]);
        let error = PublicApiSnapshot
            .run(&GateCtx::new(
                root,
                vec!["--crate".to_string(), "beta".to_string()],
            ))
            .unwrap_err();
        assert!(error.message.contains("not a snapshotted package"));
    }
}
