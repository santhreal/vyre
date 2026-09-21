//! The oracle-matrix sweeps, and the partition that decides where each one runs.
//!
//! A `sweep_*` integration test is a module of the harness its package
//! declares, and a harness whose `required-features` name a non-default feature
//! is skipped by a default `cargo test --workspace`, while an `--all-targets`
//! build compiles it without running it. These sweeps are the oracle-parity
//! matrices, so a skipped one is unproven parity that reports as a green suite.
//!
//! The roster is derived from tracked sources, each package's own manifest, and
//! the module graph that decides which harness compiles which file, so a sweep
//! added later runs by being a tracked `<crate>/tests/sweep_*.rs` file a harness
//! declares. A written-down list of test binaries stops running the newest sweep
//! in silence, which is the same failure as running nothing.
//!
//! The roster splits in two. A target whose name carries `volume` is a wave of
//! 16k cases and belongs to the sharded runner; every other target is a matrix
//! sweep. Both partitions come from this one derivation, so each tracked sweep
//! is claimed by exactly one runner and neither can drop it.
//!
//! Two modes:
//!
//!   - Default: the roster derives, both partitions are non-empty, exactly one
//!     declared harness compiles each tracked sweep source, and every
//!     `required-features` entry of that harness, and every feature the sweep's
//!     own `cfg` names, is a feature the crate defines. No cargo.
//!   - `--run`: executes one partition, one cargo invocation per sweep, which
//!     selects the sweep's own module inside its harness by name filter and
//!     carries every feature the run needs: the harness `required-features`
//!     cargo demands before it builds the target, and the features the sweep's
//!     own crate-level `cfg` demands before the module holds a case.
//!     `--partition volume --shard I --shards N` runs one wave shard; a shard
//!     index outside the count, or a count larger than the roster, is an error
//!     rather than a run that selects nothing and exits clean.
//!
//! A run that selects no case is a finding rather than a pass. A sweep whose
//! body is behind a `cfg` the run does not satisfy compiles to an empty module,
//! its harness exits zero, and the parity it states is unproven.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::{Member, Tree};
use crate::gates::test_target_membership::ownership;

/// Name prefix every oracle-matrix sweep source carries.
const SWEEP_PREFIX: &str = "sweep_";

/// Target-name fragment that marks a 16k-case volume wave.
const VOLUME: &str = "volume";

/// One tracked sweep, and the harness cargo runs it through.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SweepTarget {
    /// Member directory the sweep lives in, relative to the checkout root.
    pub crate_dir: String,
    /// Package name, as `cargo -p` takes it.
    pub package: String,
    /// Sweep source stem, which is the module name inside the harness.
    pub sweep: String,
    /// Harness target compiling the sweep, as `cargo --test` takes it.
    pub harness: String,
    /// Features the crate's `[[test]]` harness entry requires.
    pub features: Vec<String>,
}

impl SweepTarget {
    /// Whether this target is a volume wave rather than a matrix sweep.
    #[must_use]
    pub fn is_volume(&self) -> bool {
        self.sweep.contains(VOLUME)
    }

    /// The libtest filter selecting this sweep's cases and no others.
    #[must_use]
    pub fn filter(&self) -> String {
        format!("{}::", self.sweep)
    }
}

/// Runs and holds the derived oracle-matrix sweep roster.
pub struct OracleSweeps;

impl crate::gate::GateBehavior for OracleSweeps {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let members = tree.member_manifests()?;
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

/// The roster, recording every disagreement between sources, manifests and the
/// module graph.
fn derive(
    tree: &Tree,
    members: &[Member],
    report: &mut Report,
) -> Result<Vec<SweepTarget>, GateError> {
    let directories: BTreeSet<&str> = members.iter().map(|member| member.path.as_str()).collect();
    let mut sources: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for path in tree.paths() {
        let Some((crate_dir, sweep)) = sweep_source(path) else {
            continue;
        };
        if !directories.contains(crate_dir.as_str()) {
            report.find(Finding::in_file(
                path.clone(),
                format!(
                    "`{sweep}` is a sweep in `{crate_dir}`, which is not a [workspace.members] entry, so no cargo invocation reaches it"
                ),
                "add the crate to the workspace, or move the sweep into a member",
            ));
            continue;
        }
        sources.entry(crate_dir).or_default().insert(sweep);
    }

    let mut roster = Vec::new();
    for member in members {
        let Some(sweeps) = sources.get(&member.path) else {
            continue;
        };
        let owned = ownership(tree, member);
        let defined: BTreeSet<String> = member.features().into_iter().collect();
        let manifest_path = format!("{}/Cargo.toml", member.path);
        for sweep in sweeps {
            let source = format!("{}/tests/{sweep}.rs", member.path);
            let harnesses = owned.owners.get(&source).map_or(&[][..], Vec::as_slice);
            let [harness] = harnesses else {
                report.find(Finding::in_file(
                    Path::new(&source).to_path_buf(),
                    if harnesses.is_empty() {
                        format!(
                            "no [[test]] target of `{}` compiles `{sweep}`, so cargo has no selector that runs it",
                            member.path
                        )
                    } else {
                        format!(
                            "{} [[test]] targets compile `{sweep}` ({}), so its cases run once per target and the roster cannot name one runner",
                            harnesses.len(),
                            harnesses.join(", ")
                        )
                    },
                    "declare the sweep as a module of exactly one of the package's test harnesses",
                ));
                continue;
            };
            let mut required = owned
                .targets
                .iter()
                .find(|target| &target.name == harness)
                .map(|target| target.required_features.clone())
                .unwrap_or_default();
            let unknown: Vec<&String> = required
                .iter()
                .filter(|feature| !defined.contains(*feature))
                .collect();
            if !unknown.is_empty() {
                report.find(Finding::in_file(
                    Path::new(&manifest_path).to_path_buf(),
                    format!(
                        "[[test]] `{harness}`, which compiles `{sweep}`, requires {unknown:?}, which `{}` does not define in [features], so cargo refuses the target",
                        member.path
                    ),
                    "declare the feature, or require the one the crate defines",
                ));
            }
            let gating = crate::gates::scan::crate_cfg_features(&tree.read(&source)?);
            let undefined: Vec<&String> = gating
                .iter()
                .filter(|feature| !defined.contains(*feature))
                .collect();
            if !undefined.is_empty() {
                report.find(Finding::in_file(
                    Path::new(&source).to_path_buf(),
                    format!(
                        "`{sweep}` compiles only under {undefined:?}, which `{}` does not define in [features], so the module is empty in every build",
                        member.path
                    ),
                    "gate the sweep on a feature the crate defines, or define the one it names",
                ));
            }
            required.extend(gating);
            roster.push(SweepTarget {
                crate_dir: member.path.clone(),
                package: member.name.clone(),
                sweep: sweep.clone(),
                harness: harness.clone(),
                features: required.into_iter().collect(),
            });
        }
    }
    Ok(roster)
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

/// Execute the selected partition, one cargo invocation per sweep.
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

    let mut shard_targets: Vec<&SweepTarget> = Vec::new();
    for (index, target) in selected.iter().enumerate() {
        if index % shards == shard {
            shard_targets.push(target);
        }
    }
    if shard_targets.is_empty() {
        return Err(GateError::new(
            format!(
                "shard {shard} of {shards} selected none of the {} {partition} target(s)",
                selected.len()
            ),
            "run a shard index the roster reaches",
        ));
    }

    let mut crates: BTreeSet<&str> = BTreeSet::new();
    for target in &shard_targets {
        crates.insert(target.crate_dir.as_str());
        let mut command = crate::cargo_runner::command(&ctx.root);
        command.args(["test", "-p", target.package.as_str()]);
        if !target.features.is_empty() {
            command.arg("--features");
            command.arg(target.features.join(","));
        }
        command.args(["--test", target.harness.as_str()]);
        command.args(["--", target.filter().as_str()]);
        let (status, output, diagnostics) = crate::cargo_runner::run_captured(&mut command)
            .map_err(|error| {
                GateError::new(
                    format!("cannot run cargo test for `{}`: {error}", target.crate_dir),
                    "install a cargo the runner can start, or set CARGO to one",
                )
            })?;
        if !status.success() {
            if let Some(missing) = crate::cargo_runner::unmeasured(&diagnostics) {
                report.find(Finding::new(
                    format!(
                        "`{}` measured nothing: the build named `{missing}`, which the build directory does not carry",
                        target.crate_dir
                    ),
                    "run the sweep again against an intact build directory; a compile whose own inputs were deleted under it reports the state of the disk, and the sweep it was pointed at never ran",
                ));
                continue;
            }
            report.find(Finding::new(
                format!(
                    "`{}` failed the {partition} sweep `{}`, run as `--test {} -- {}`",
                    target.crate_dir,
                    target.sweep,
                    target.harness,
                    target.filter()
                ),
                "fix the parity failure the sweep reported; a skipped or failing oracle matrix is unproven parity",
            ));
            continue;
        }
        if cases_run(&output) == 0 {
            report.find(Finding::new(
                format!(
                    "`{}` ran no case of the {partition} sweep `{}`: `--test {} -- {}` selected nothing and the harness exited zero",
                    target.crate_dir,
                    target.sweep,
                    target.harness,
                    target.filter()
                ),
                "compile the sweep under the features this partition runs with, or require them on the harness that declares it; an empty module proves no parity",
            ));
        }
    }
    report.note(format!(
        "shard {shard} of {shards}: ran {} {partition} target(s) across {} crate(s)",
        shard_targets.len(),
        crates.len()
    ));
    Ok(())
}

/// How many cases libtest reported running, across every harness in the output.
///
/// A filter that matches nothing is the failure this counts for, and libtest
/// reports it the same way as a suite that passed: `running 0 tests`, then a
/// zero exit. The count is taken from the harness's own line rather than from
/// the result line, because a run with every case filtered out still prints a
/// result line reading `ok`.
fn cases_run(output: &str) -> usize {
    output
        .lines()
        .filter_map(|line| line.trim().strip_prefix("running "))
        .filter_map(|rest| rest.split_whitespace().next())
        .filter_map(|count| count.parse::<usize>().ok())
        .sum()
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
        let matrix = target("sweep_matching_oracle");
        let wave = target("sweep_matching_volume_wave");
        assert!(!matrix.is_volume());
        assert!(wave.is_volume());
    }

    /// WHY: a sweep is a module of a shared harness, so the only thing that
    /// runs its cases and nothing else is a name filter scoped to the module.
    /// A filter without the separator also selects a sibling whose name starts
    /// with the same text, and running a neighbour's cases under this sweep's
    /// name is how an empty sweep reports as covered.
    #[test]
    fn a_sweep_is_selected_by_its_own_module_path() {
        assert_eq!(
            target("sweep_bitset_oracle").filter(),
            "sweep_bitset_oracle::"
        );
    }

    /// WHY: a filter that matches no case exits zero, which is the same exit a
    /// passing suite gives. The harness line is the only place the difference
    /// is stated, and reading the result line instead reports `ok` for a run
    /// that proved nothing.
    #[test]
    fn a_run_that_selected_no_case_is_counted_as_none() {
        assert_eq!(
            cases_run("running 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 412 filtered out\n"),
            0
        );
        assert_eq!(
            cases_run("running 12 tests\ntest sweep_a::one ... ok\n\ntest result: ok. 12 passed\n"),
            12
        );
        assert_eq!(cases_run("running 3 tests\nrunning 4 tests\n"), 7);
        assert_eq!(cases_run("Compiling vyre-libs v0.1.0\n"), 0);
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

    /// WHY: every field the runner passes to cargo comes from one place, and a
    /// fixture that spells them inline drifts from the struct the derivation
    /// fills.
    fn target(sweep: &str) -> SweepTarget {
        SweepTarget {
            crate_dir: "vyre-libs".to_string(),
            package: "vyre-libs".to_string(),
            sweep: sweep.to_string(),
            harness: "all_tests".to_string(),
            features: Vec::new(),
        }
    }
}
