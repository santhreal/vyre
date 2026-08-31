//! The oracle-matrix sweeps, and the partition that decides where each one runs.
//!
//! A `sweep_*` integration test whose `[[test]] required-features` name a
//! non-default feature is skipped by a default `cargo test --workspace`, and an
//! `--all-targets` build compiles it without running it. These sweeps are the
//! oracle-parity matrices, so a skipped one is unproven parity that reports as
//! a green suite.
//!
//! The roster is derived from tracked sources and each crate's own manifest, so
//! a sweep added later runs by being a tracked `<crate>/tests/sweep_*.rs` file.
//! A written-down list of test binaries stops running the newest sweep in
//! silence, which is the same failure as running nothing.
//!
//! The roster splits in two. A target whose name carries `volume` is a wave of
//! 16k cases and belongs to the sharded runner; every other target is a matrix
//! sweep. Both partitions come from this one derivation, so each tracked sweep
//! is claimed by exactly one runner and neither can drop it.
//!
//! Two modes:
//!
//!   - Default: the roster derives, both partitions are non-empty, every
//!     `[[test]]` entry names a tracked source, and every `required-features`
//!     entry names a feature the crate defines. No cargo.
//!   - `--run`: executes one partition, one cargo invocation per crate with the
//!     union of the required-features its selected targets declare, because
//!     cargo refuses a `--test` whose required-features are unmet.
//!     `--partition volume --shard I --shards N` runs one wave shard; a shard
//!     index outside the count, or a count larger than the roster, is an error
//!     rather than a run that selects nothing and exits clean.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// The manifest that declares the workspace members.
const ROOT_MANIFEST: &str = "Cargo.toml";

/// Name prefix every oracle-matrix sweep source carries.
const SWEEP_PREFIX: &str = "sweep_";

/// Target-name fragment that marks a 16k-case volume wave.
const VOLUME: &str = "volume";

/// One tracked sweep target and the features its crate reserves for it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SweepTarget {
    /// Member directory the target lives in, relative to the checkout root.
    pub crate_dir: String,
    /// Package name, as `cargo -p` takes it.
    pub package: String,
    /// Test target name, as `cargo --test` takes it.
    pub target: String,
    /// Features the crate's `[[test]]` entry requires for the target.
    pub features: Vec<String>,
}

impl SweepTarget {
    /// Whether this target is a volume wave rather than a matrix sweep.
    #[must_use]
    pub fn is_volume(&self) -> bool {
        self.target.contains(VOLUME)
    }
}

/// Runs and holds the derived oracle-matrix sweep roster.
pub struct OracleSweeps;

impl crate::gate::GateBehavior for OracleSweeps {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let members = workspace_members(&tree)?;
        let mut report = Report::clean();
        let roster = derive(&tree, &members, &mut report)?;
        report.cover_complete("oracle sweep targets", roster.len());

        if roster.is_empty() {
            report.find(Finding::new(
                "no tracked <crate>/tests/sweep_*.rs source exists, so the sweep runners would report success without executing anything",
                "restore the oracle-matrix sweeps, or delete this gate in the commit that removes the last one",
            ));
            return Ok(report);
        }
        let volume = roster.iter().filter(|target| target.is_volume()).count();
        let matrix = roster.len() - volume;
        for (partition, count) in [("matrix", matrix), ("volume", volume)] {
            if count == 0 {
                report.find(Finding::new(
                    format!(
                        "the {partition} partition of the {} tracked sweep target(s) is empty, so its runner would execute nothing",
                        roster.len()
                    ),
                    "restore the partition's targets, or fold its runner into the other partition in the same commit",
                ));
            }
        }

        if !ctx.has("--run") {
            report.note(format!(
                "{} tracked sweep target(s): {matrix} matrix, {volume} volume",
                roster.len()
            ));
            return Ok(report);
        }
        if !report.findings.is_empty() {
            return Ok(report);
        }
        run_partition(ctx, &roster, &mut report)?;
        Ok(report)
    }
}

/// Every workspace member the root manifest declares.
fn workspace_members(tree: &Tree) -> Result<BTreeSet<String>, GateError> {
    let manifest = tree.read_toml(ROOT_MANIFEST)?;
    let members = manifest
        .get("workspace")
        .and_then(|workspace| workspace.get("members"))
        .and_then(toml::Value::as_array)
        .ok_or_else(|| {
            GateError::new(
                format!("{ROOT_MANIFEST} declares no [workspace.members]"),
                "declare the members; a roster derived from an empty workspace covers nothing",
            )
        })?;
    Ok(members
        .iter()
        .filter_map(|value| value.as_str().map(str::to_string))
        .collect())
}

/// The roster, recording every disagreement between sources and manifests.
fn derive(
    tree: &Tree,
    members: &BTreeSet<String>,
    report: &mut Report,
) -> Result<Vec<SweepTarget>, GateError> {
    let mut sources: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for path in tree.paths() {
        let Some((crate_dir, target)) = sweep_source(path) else {
            continue;
        };
        if !members.contains(&crate_dir) {
            report.find(Finding::in_file(
                path.clone(),
                format!(
                    "`{target}` is a sweep in `{crate_dir}`, which is not a [workspace.members] entry, so no cargo invocation reaches it"
                ),
                "add the crate to the workspace, or move the sweep into a member",
            ));
            continue;
        }
        sources.entry(crate_dir).or_default().insert(target);
    }

    let mut roster = Vec::new();
    for (crate_dir, targets) in &sources {
        let manifest_path = format!("{crate_dir}/Cargo.toml");
        let manifest = tree.read_toml(&manifest_path)?;
        let defined: BTreeSet<String> = manifest
            .get("features")
            .and_then(toml::Value::as_table)
            .map(|table| table.keys().cloned().collect())
            .unwrap_or_default();
        let declared = declared_tests(&manifest);
        for (name, features) in &declared {
            if !targets.contains(name) {
                report.find(Finding::in_file(
                    manifest_path.clone(),
                    format!(
                        "[[test]] `{name}` reserves features for a target with no tracked `{crate_dir}/tests/{name}.rs`"
                    ),
                    "restore the source, or delete the entry that reserves features for a target cargo cannot build",
                ));
            }
            let unknown: Vec<&String> = features
                .iter()
                .filter(|feature| !defined.contains(*feature))
                .collect();
            if !unknown.is_empty() {
                report.find(Finding::in_file(
                    manifest_path.clone(),
                    format!(
                        "[[test]] `{name}` requires {unknown:?}, which `{crate_dir}` does not define in [features], so cargo refuses the target"
                    ),
                    "declare the feature, or require the one the crate defines",
                ));
            }
        }
        let package = package_name(&manifest, &manifest_path)?;
        for target in targets {
            roster.push(SweepTarget {
                crate_dir: crate_dir.clone(),
                package: package.clone(),
                target: target.clone(),
                features: declared.get(target).cloned().unwrap_or_default(),
            });
        }
    }
    Ok(roster)
}

/// The package name a member manifest declares.
///
/// A directory name is not a package name, and `cargo -p` takes the package.
/// Reading the directory worked for every member whose two names agree and
/// would silently address another crate for one whose names differ, so the
/// manifest answers and a manifest that declares no name fails closed.
fn package_name(manifest: &toml::Table, manifest_path: &str) -> Result<String, GateError> {
    manifest
        .get("package")
        .and_then(|package| package.get("name"))
        .and_then(toml::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| {
            GateError::new(
                format!("{manifest_path} declares no [package] name"),
                "name the package; a sweep is run by package name and a directory name is not one",
            )
        })
}

/// The member directory and target a tracked path names, when it is a sweep
/// source.
///
/// The directory is everything above `tests/`, not the first path segment, so a
/// member declared at a nested path such as `conform/vyre-conform` contributes
/// its sweeps. Splitting at the first separator read that member as `conform`,
/// which is no member at all, so every sweep it carries was invisible.
fn sweep_source(path: &Path) -> Option<(String, String)> {
    let text = path.to_str()?;
    let (crate_dir, rest) = text.rsplit_once("/tests/")?;
    let target = rest.strip_suffix(".rs")?;
    if !target.starts_with(SWEEP_PREFIX) || target.contains('/') {
        return None;
    }
    Some((crate_dir.to_string(), target.to_string()))
}

/// Every `[[test]]` entry naming a sweep, with the features it requires.
fn declared_tests(manifest: &toml::Table) -> BTreeMap<String, Vec<String>> {
    let mut declared = BTreeMap::new();
    let Some(entries) = manifest.get("test").and_then(toml::Value::as_array) else {
        return declared;
    };
    for entry in entries {
        let Some(name) = entry.get("name").and_then(toml::Value::as_str) else {
            continue;
        };
        if !name.starts_with(SWEEP_PREFIX) {
            continue;
        }
        let features = entry
            .get("required-features")
            .and_then(toml::Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        declared.insert(name.to_string(), features);
    }
    declared
}

/// Execute the selected partition, one cargo invocation per crate.
fn run_partition(
    ctx: &GateCtx,
    roster: &[SweepTarget],
    report: &mut Report,
) -> Result<(), GateError> {
    let partition = ctx.flag("--partition").unwrap_or("matrix");
    let volume = match partition {
        "matrix" => false,
        "volume" => true,
        other => {
            return Err(GateError::new(
                format!("unknown partition `{other}`"),
                "use --partition matrix or --partition volume",
            ))
        }
    };
    let selected: Vec<&SweepTarget> = roster
        .iter()
        .filter(|target| target.is_volume() == volume)
        .collect();
    let shards = numeric(ctx, "--shards", 1)?;
    let shard = numeric(ctx, "--shard", 0)?;
    if shards == 0 {
        return Err(GateError::new(
            "shard count 0 selects nothing",
            "pass --shards with at least 1",
        ));
    }
    if shard >= shards {
        return Err(GateError::new(
            format!("shard index {shard} is outside shard count {shards}"),
            format!(
                "use 0 through {}; a shard that selects no target proves nothing",
                shards - 1
            ),
        ));
    }
    if shards > selected.len() {
        return Err(GateError::new(
            format!(
                "shard count {shards} exceeds the {} {partition} target(s), so the highest shards would run nothing",
                selected.len()
            ),
            "lower the shard count to at most the target count",
        ));
    }

    let mut by_crate: BTreeMap<&str, (&str, Vec<&str>, BTreeSet<&str>)> = BTreeMap::new();
    for (index, target) in selected.iter().enumerate() {
        if index % shards != shard {
            continue;
        }
        let entry = by_crate
            .entry(target.crate_dir.as_str())
            .or_insert_with(|| (target.package.as_str(), Vec::new(), BTreeSet::new()));
        entry.1.push(target.target.as_str());
        entry.2.extend(target.features.iter().map(String::as_str));
    }
    if by_crate.is_empty() {
        return Err(GateError::new(
            format!(
                "shard {shard} of {shards} selected none of the {} {partition} target(s)",
                selected.len()
            ),
            "run a shard index the roster reaches",
        ));
    }

    let mut executed = 0usize;
    for (crate_dir, (package, targets, features)) in &by_crate {
        let mut command = crate::cargo_runner::command(&ctx.root);
        command.args(["test", "-p", package]);
        if !features.is_empty() {
            command.arg("--features");
            command.arg(features.iter().copied().collect::<Vec<_>>().join(","));
        }
        for target in targets {
            command.args(["--test", target]);
        }
        let (status, diagnostics) =
            crate::cargo_runner::run_streaming(&mut command).map_err(|error| {
                GateError::new(
                    format!("cannot run cargo test for `{crate_dir}`: {error}"),
                    "install a cargo the runner can start, or set CARGO to one",
                )
            })?;
        executed += targets.len();
        if !status.success() {
            if let Some(missing) = crate::cargo_runner::unmeasured(&diagnostics) {
                report.find(Finding::new(
                    format!(
                        "`{crate_dir}` measured nothing: the build named `{missing}`, which the build directory does not carry"
                    ),
                    "run the sweep again against an intact build directory; a compile whose own inputs were deleted under it reports the state of the disk, and the sweep it was pointed at never ran",
                ));
                continue;
            }
            report.find(Finding::new(
                format!(
                    "`{crate_dir}` failed {} {partition} sweep target(s): {}",
                    targets.len(),
                    targets.join(", ")
                ),
                "fix the parity failure the sweep reported; a skipped or failing oracle matrix is unproven parity",
            ));
        }
    }
    report.note(format!(
        "shard {shard} of {shards}: ran {executed} {partition} target(s) across {} crate(s)",
        by_crate.len()
    ));
    Ok(())
}

/// One numeric flag, defaulting when it is not passed.
fn numeric(ctx: &GateCtx, flag: &str, default: usize) -> Result<usize, GateError> {
    let Some(value) = ctx.flag(flag) else {
        return Ok(default);
    };
    value.parse().map_err(|_| {
        GateError::new(
            format!("{flag} takes a non-negative integer, not `{value}`"),
            format!("pass {flag} <number>"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the partition is what decides which runner owns a target, and a
    /// target claimed by neither runs nowhere while both runners report clean.
    /// The predicate is crate-private, so no integration test reaches it.
    #[test]
    fn every_target_is_claimed_by_exactly_one_partition() {
        let matrix = SweepTarget {
            crate_dir: "vyre-libs".to_string(),
            package: "vyre-libs".to_string(),
            target: "sweep_matching_oracle".to_string(),
            features: Vec::new(),
        };
        let wave = SweepTarget {
            crate_dir: "vyre-libs".to_string(),
            package: "vyre-libs".to_string(),
            target: "sweep_matching_volume_wave".to_string(),
            features: Vec::new(),
        };
        assert!(!matrix.is_volume());
        assert!(wave.is_volume());
    }

    /// WHY: a nested support module under `tests/` is not a target, and a
    /// non-sweep test is another runner's business. Reading either as a sweep
    /// makes the runner pass cargo a `--test` it will refuse. A member declared
    /// at a nested path carries sweeps like any other, and splitting the path
    /// at its first separator read that member as its parent directory, which
    /// is no member, so its sweeps were invisible.
    #[test]
    fn a_sweep_source_names_the_member_directory_that_holds_it() {
        assert_eq!(
            sweep_source(Path::new("vyre-libs/tests/sweep_matching_oracle.rs")),
            Some(("vyre-libs".to_string(), "sweep_matching_oracle".to_string()))
        );
        assert_eq!(
            sweep_source(Path::new(
                "conform/vyre-conform/tests/sweep_backend_oracle.rs"
            )),
            Some((
                "conform/vyre-conform".to_string(),
                "sweep_backend_oracle".to_string()
            ))
        );
        assert_eq!(
            sweep_source(Path::new("vyre-libs/tests/sweep_support/mod.rs")),
            None
        );
        assert_eq!(sweep_source(Path::new("vyre-libs/tests/wire.rs")), None);
        assert_eq!(
            sweep_source(Path::new("vyre-libs/src/sweep_matching.rs")),
            None
        );
    }

    /// WHY: `cargo -p` takes a package name and the roster walks directories.
    /// A member whose directory and package names differ was addressed by its
    /// directory, which selects another crate or no crate at all, and a
    /// manifest that names no package must not resolve to a guess.
    #[test]
    fn a_package_is_named_by_its_manifest_and_never_by_its_directory() {
        let manifest: toml::Table =
            toml::from_str("[package]\nname = \"vyre-conform\"\nversion = \"0.1.0\"\n")
                .expect("table");
        assert_eq!(
            package_name(&manifest, "conform/vyre-conform/Cargo.toml").expect("a named package"),
            "vyre-conform"
        );
        let anonymous: toml::Table = toml::from_str("[workspace]\n").expect("table");
        let error = package_name(&anonymous, "conform/vyre-conform/Cargo.toml")
            .expect_err("a manifest with no package name fails closed");
        assert!(
            error.message.contains("conform/vyre-conform/Cargo.toml"),
            "{}",
            error.message
        );
    }

    /// WHY: the features a target needs come from the crate's own `[[test]]`
    /// entry. Reading them from anywhere else is how a runner passes cargo a
    /// selection that cannot build.
    #[test]
    fn the_manifest_entry_supplies_the_required_features() {
        let manifest: toml::Table = toml::from_str(
            "[[test]]\nname = \"sweep_a\"\nrequired-features = [\"math-kernels\"]\n\n[[test]]\nname = \"wire\"\nrequired-features = [\"other\"]\n",
        )
        .expect("table");
        let declared = declared_tests(&manifest);
        assert_eq!(
            declared.get("sweep_a"),
            Some(&vec!["math-kernels".to_string()])
        );
        assert!(!declared.contains_key("wire"));
    }
}
