//! What verification costs this workspace, per tier, against a declared budget.
//!
//! Test volume is not verification. A tree reaches a large number of assertions
//! two ways: one composable contract enumerated over its variant space, or the
//! same contract body copied into every package that wanted it. The second
//! reads as more coverage and is less, because the copies drift, and it is paid
//! for in compile units, link work and disk that grow with the copying rather
//! than with what is proven.
//!
//! Four facts separate the two, and none of them is countable by reading a test
//! count. A tier's **compile units** are the test targets cargo builds for it. Its
//! **link work** is the binaries those targets link, which is where a package that
//! declares one file per target spends its time. Its **duplicate source bytes** are
//! test sources byte-identical to a test source in another package, which is the
//! copying, measured. Its **disk use** is what the tier's test sources occupy.
//!
//! The budget is per tier because the tier is who fixes it. A layer of twenty
//! library crates and a single compiler-boundary crate do not carry comparable
//! verification, and one workspace-wide number would be met by whichever tier
//! grew slowest.
//!
//! # How each measure is checked
//!
//! The three counts are exact in both directions. Over the row fails, because
//! the targets arrived without anyone recording why. Under the row also fails,
//! with the number to write, because a row above the tree covers the next
//! target added to that tier instead of reporting it. Additional targets are
//! legitimate whenever they buy real process, feature, package or backend
//! isolation; what the row requires is that somebody wrote down that they do.
//!
//! Disk use is a ceiling, not an exact figure. Editing one assertion moves it,
//! so an exact pin would be red on most commits and would be raised until it
//! meant nothing. Over the ceiling fails; under it is recorded and reported.
//!
//! # One semantic owner
//!
//! A shared contract is owned by exactly one package. `vyre-test-support` is
//! where a backend-neutral contract body lives, and a test source elsewhere that
//! is byte-identical to one of its modules is a second copy of a contract that
//! already has an owner. A `#[path]` include resolving outside its own package
//! is the same defect reached textually: the body compiles into every consuming
//! target, is absent from the consuming package archive, and can behave
//! differently under each package's feature unification.
//!
//! # What it does not catch
//!
//! Two contract bodies that differ by a rename are two owners to this gate and
//! one to a reader. Duplication is measured on normalized bytes, so a copy
//! somebody edited is invisible here; `dup-scan` measures that shape across the
//! whole tree and pins it per crate.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;

use serde::Serialize;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::crate_registry;
use crate::gates::scan::Tree;

/// Gate name, as registered.
const NAME: &str = "verification-budget";

/// The recorded measurement.
const ARTIFACT: &str = "release/evidence/tests/verification-budget.json";

/// The hand-authored per-tier budget.
const BUDGET: &str = "docs/testing/VERIFICATION_BUDGET.toml";

/// The package that owns backend-neutral shared contract bodies.
const SHARED_CONTRACT_OWNER: &str = "vyre-test-support";

/// Schema of the recorded measurement.
const SCHEMA_VERSION: u32 = 1;

/// What one tier costs to verify.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
struct TierCost {
    /// Test targets cargo builds for the tier.
    compile_units: usize,
    /// Binaries those targets link, including each package's unit-test binary.
    link_units: usize,
    /// Bytes of test source byte-identical to a test source in another package.
    duplicate_source_bytes: u64,
    /// Bytes the tier's test sources occupy.
    disk_use_bytes: u64,
}

/// One tier's recorded row.
#[derive(Clone, Debug, Serialize)]
struct TierRow {
    /// Layer name from the ownership record.
    tier: String,
    /// Packages the tier holds.
    packages: Vec<String>,
    /// What the tier costs.
    #[serde(flatten)]
    cost: TierCost,
}

/// One shared contract body and the package that owns it.
#[derive(Clone, Debug, Serialize)]
struct SharedContract {
    /// Path of the owning module, relative to the checkout root.
    module: String,
    /// Package that owns it.
    owner: String,
}

/// The whole measurement, as recorded.
#[derive(Clone, Debug, Serialize)]
struct Measurement {
    /// Schema of this artifact.
    schema_version: u32,
    /// One row per tier, in tier order.
    tiers: Vec<TierRow>,
    /// Every shared contract body and its single owner.
    shared_contracts: Vec<SharedContract>,
    /// Test sources duplicating a shared contract body, by path.
    duplicated_shared_contracts: Vec<String>,
    /// Test sources whose `#[path]` include resolves outside their package.
    escaping_path_includes: Vec<String>,
}

/// One test source file, with what the duplication measure needs.
struct TestSource {
    /// Path relative to the checkout root.
    path: String,
    /// Package the file belongs to.
    package: String,
    /// Bytes on disk.
    bytes: u64,
    /// Content with blank lines and line comments removed.
    normalized: String,
}

/// Read every tracked test source, grouped by package.
fn test_sources(tree: &Tree, members: &BTreeMap<String, String>) -> Vec<TestSource> {
    let mut sources = Vec::new();
    for path in tree.paths() {
        let relative = path.to_string_lossy().replace('\\', "/");
        if !relative.ends_with(".rs") {
            continue;
        }
        let Some((package, directory)) = owning_member(&relative, members) else {
            continue;
        };
        if !relative.starts_with(&format!("{directory}/tests/")) {
            continue;
        }
        let Ok(text) = tree.read(&relative) else {
            continue;
        };
        sources.push(TestSource {
            path: relative,
            package,
            bytes: text.len() as u64,
            normalized: normalize(&text),
        });
    }
    sources.sort_by(|left, right| left.path.cmp(&right.path));
    sources
}

/// The member directory holding `relative`, longest match first.
///
/// Members nest: `conform/vyre-conform` sits under no member but shares a
/// prefix with none, while a shorter directory can prefix a longer one. The
/// longest match is the owning package.
fn owning_member(
    relative: &str,
    members: &BTreeMap<String, String>,
) -> Option<(String, String)> {
    members
        .iter()
        .filter(|(directory, _)| relative.starts_with(&format!("{directory}/")))
        .max_by_key(|(directory, _)| directory.len())
        .map(|(directory, package)| (package.clone(), directory.clone()))
}

/// Content with blank lines, line comments and indentation removed.
fn normalize(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Count the test targets a manifest declares.
///
/// A package with `autotests = false` declares each harness in a `[[test]]`
/// table and cargo discovers nothing. A package that leaves autodiscovery on
/// gets one target per file directly under `tests/`, plus any it declares.
fn compile_units(tree: &Tree, directory: &str, manifest: &toml::Table) -> usize {
    let declared = manifest
        .get("test")
        .and_then(toml::Value::as_array)
        .map_or(0, Vec::len);
    let autotests = manifest
        .get("package")
        .and_then(toml::Value::as_table)
        .and_then(|package| package.get("autotests"))
        .and_then(toml::Value::as_bool)
        .unwrap_or(true);
    if !autotests {
        return declared;
    }
    let discovered = tree
        .paths()
        .iter()
        .filter(|path| {
            let relative = path.to_string_lossy().replace('\\', "/");
            let Some(rest) = relative.strip_prefix(&format!("{directory}/tests/")) else {
                return false;
            };
            rest.ends_with(".rs") && !rest.contains('/')
        })
        .count();
    declared + discovered
}

/// Count the binaries a package's own targets link under `cargo test -p`.
///
/// Every integration-test target links one. The library and the binary each
/// link their own unit-test binary, and a package with neither links none.
fn unit_test_binaries(tree: &Tree, directory: &str) -> usize {
    usize::from(tree.has(&format!("{directory}/src/lib.rs")))
        + usize::from(tree.has(&format!("{directory}/src/main.rs")))
}

/// Test sources whose `#[path]` include resolves outside their own package.
fn escaping_path_includes(sources: &[TestSource], members: &BTreeMap<String, String>) -> Vec<String> {
    let mut escapes = Vec::new();
    for source in sources {
        let Some(directory) = members
            .iter()
            .find(|(_, package)| *package == &source.package)
            .map(|(directory, _)| directory.clone())
        else {
            continue;
        };
        for line in source.normalized.lines() {
            let Some(rest) = line.strip_prefix("#[path") else {
                continue;
            };
            let Some(open) = rest.find('"') else { continue };
            let Some(close) = rest[open + 1..].find('"') else {
                continue;
            };
            let include = &rest[open + 1..open + 1 + close];
            if resolves_outside(&source.path, include, &directory) {
                escapes.push(format!("{}: {include}", source.path));
            }
        }
    }
    escapes
}

/// Whether `include`, resolved beside `source`, leaves `directory`.
fn resolves_outside(source: &str, include: &str, directory: &str) -> bool {
    let mut segments: Vec<&str> = source.split('/').collect();
    segments.pop();
    for segment in include.split('/') {
        match segment {
            "." | "" => {}
            ".." => {
                segments.pop();
            }
            other => segments.push(other),
        }
    }
    !segments.join("/").starts_with(&format!("{directory}/"))
}

/// Bytes each package contributes as a copy of a body another package already has.
///
/// A normalized body compiled by more than one package is a copy. Everything past
/// the first occurrence in path order is the duplication, so the count does not
/// depend on which package is read first, and two copies inside one package are
/// compiled once and are not this defect.
fn duplicate_bytes_by_package(sources: &[TestSource]) -> BTreeMap<String, u64> {
    let mut by_content: BTreeMap<&str, Vec<&TestSource>> = BTreeMap::new();
    for source in sources {
        by_content
            .entry(source.normalized.as_str())
            .or_default()
            .push(source);
    }
    let mut duplicate: BTreeMap<String, u64> = BTreeMap::new();
    for copies in by_content.values() {
        let packages: BTreeSet<&str> = copies.iter().map(|copy| copy.package.as_str()).collect();
        if packages.len() < 2 {
            continue;
        }
        for copy in copies.iter().skip(1) {
            *duplicate.entry(copy.package.clone()).or_default() += copy.bytes;
        }
    }
    duplicate
}

/// Whether `measured` sits above `pin` in any column.
fn raises_any_column(measured: &TierCost, pin: &TierCost) -> bool {
    measured.compile_units > pin.compile_units
        || measured.link_units > pin.link_units
        || measured.duplicate_source_bytes > pin.duplicate_source_bytes
        || measured.disk_use_bytes > pin.disk_use_bytes
}

/// Measure the tree.
fn measure(tree: &Tree) -> Result<Measurement, GateError> {
    let declared = crate_registry::declared_crates(tree)?;
    let members: BTreeMap<String, String> = declared
        .iter()
        .map(|entry| (entry.path.trim_end_matches('/').to_string(), entry.package.clone()))
        .collect();
    let tier_of: BTreeMap<String, String> = declared
        .iter()
        .map(|entry| (entry.package.clone(), entry.layer.clone()))
        .collect();

    let sources = test_sources(tree, &members);
    let duplicate_bytes = duplicate_bytes_by_package(&sources);

    let mut costs: BTreeMap<String, TierCost> = BTreeMap::new();
    let mut packages_of: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for member in tree.member_manifests()? {
        let directory = member.path.trim_end_matches('/').to_string();
        let Some(tier) = tier_of.get(&member.name) else {
            continue;
        };
        let cost = costs.entry(tier.clone()).or_default();
        let targets = compile_units(tree, &directory, &member.manifest);
        cost.compile_units += targets;
        cost.link_units += targets + unit_test_binaries(tree, &directory);
        cost.duplicate_source_bytes += duplicate_bytes
            .get(member.name.as_str())
            .copied()
            .unwrap_or_default();
        cost.disk_use_bytes += sources
            .iter()
            .filter(|source| source.package == member.name)
            .map(|source| source.bytes)
            .sum::<u64>();
        packages_of.entry(tier.clone()).or_default().push(member.name);
    }

    // Every module the shared-contract owner declares, and every test source
    // elsewhere that repeats one of those bodies verbatim.
    let owner_directory = members
        .iter()
        .find(|(_, package)| *package == SHARED_CONTRACT_OWNER)
        .map(|(directory, _)| directory.clone())
        .unwrap_or_else(|| SHARED_CONTRACT_OWNER.to_string());
    let mut shared_contracts = Vec::new();
    let mut owner_bodies: BTreeMap<String, String> = BTreeMap::new();
    for path in tree.paths() {
        let relative = path.to_string_lossy().replace('\\', "/");
        if !relative.starts_with(&format!("{owner_directory}/src/")) || !relative.ends_with(".rs") {
            continue;
        }
        let Ok(text) = tree.read(&relative) else {
            continue;
        };
        owner_bodies.insert(normalize(&text), relative.clone());
        shared_contracts.push(SharedContract {
            module: relative,
            owner: SHARED_CONTRACT_OWNER.to_string(),
        });
    }
    let duplicated_shared_contracts = sources
        .iter()
        .filter(|source| source.package != SHARED_CONTRACT_OWNER)
        .filter(|source| owner_bodies.contains_key(&source.normalized))
        .map(|source| {
            let owner = &owner_bodies[&source.normalized];
            format!("{} repeats {owner}", source.path)
        })
        .collect();

    let tiers = costs
        .into_iter()
        .map(|(tier, cost)| TierRow {
            packages: packages_of.remove(&tier).unwrap_or_default(),
            tier,
            cost,
        })
        .collect();

    Ok(Measurement {
        schema_version: SCHEMA_VERSION,
        tiers,
        shared_contracts,
        duplicated_shared_contracts,
        escaping_path_includes: escaping_path_includes(&sources, &members),
    })
}

/// The declared budget, by tier.
fn declared_budget(tree: &Tree) -> Result<BTreeMap<String, TierCost>, GateError> {
    let table = tree.read_toml(BUDGET)?;
    let Some(tiers) = table.get("tier").and_then(toml::Value::as_table) else {
        return Ok(BTreeMap::new());
    };
    let mut budget = BTreeMap::new();
    for (tier, row) in tiers {
        let Some(row) = row.as_table() else { continue };
        let read = |key: &str| {
            row.get(key)
                .and_then(toml::Value::as_integer)
                .unwrap_or_default()
        };
        budget.insert(
            tier.clone(),
            TierCost {
                compile_units: usize::try_from(read("compile_units")).unwrap_or_default(),
                link_units: usize::try_from(read("link_units")).unwrap_or_default(),
                duplicate_source_bytes: u64::try_from(read("duplicate_source_bytes"))
                    .unwrap_or_default(),
                disk_use_bytes: u64::try_from(read("disk_use_bytes")).unwrap_or_default(),
            },
        );
    }
    Ok(budget)
}

/// Render the budget file from `measurement`, keeping every row already declared.
fn render_budget(measurement: &Measurement, declared: &BTreeMap<String, TierCost>) -> String {
    let mut text = String::from(HEADER);
    text.push_str("\nschema = 1\n");
    for row in &measurement.tiers {
        let cost = declared.get(&row.tier).unwrap_or(&row.cost);
        text.push_str(&format!(
            "\n[tier.{}]\ncompile_units = {}\nlink_units = {}\nduplicate_source_bytes = {}\ndisk_use_bytes = {}\n",
            row.tier,
            cost.compile_units,
            cost.link_units,
            cost.duplicate_source_bytes,
            cost.disk_use_bytes
        ));
    }
    text
}

/// Preamble of the budget file, restated on every write.
const HEADER: &str = "\
# What verification costs this workspace, per tier.
#
# `./cargo_full run --bin xtask -- verification-budget` derives every number here from
# the tree: the test targets built for it, the binaries they link, the test sources
# byte-identical to a test source in another package, and the bytes the tier's
# test sources occupy. A tier with no row fails, and a row for a tier the
# ownership record no longer declares fails as stale.
#
# The three counts are exact in both directions. Over the row means targets
# arrived with no decision recorded for them. Under the row means the row now
# covers the next target added to the tier instead of reporting it, so it is
# lowered to what the tree measures. Additional targets are legitimate whenever
# they buy real process, feature, package or backend isolation; the row is where
# that decision is written down.
#
# Disk use is a ceiling. One edited assertion moves it, so an exact pin would be
# red on most commits and would be raised until it meant nothing.
#
# `--write-budget` records a row for a tier that has none and touches nothing
# else, so it cannot erase an intentional red. `--lower-budget TIER` sets one
# row to what the tree measures and refuses to raise any of its numbers.
";

/// Write rows for tiers that have none.
fn write_budget(
    tree: &Tree,
    measurement: &Measurement,
    report: &mut Report,
) -> Result<(), GateError> {
    let declared = declared_budget(tree).unwrap_or_default();
    let added = measurement
        .tiers
        .iter()
        .filter(|row| !declared.contains_key(&row.tier))
        .map(|row| row.tier.clone())
        .collect::<Vec<_>>();
    let text = render_budget(measurement, &declared);
    write_text(tree, BUDGET, &text)?;
    report.note(if added.is_empty() {
        "every tier already carries a budget row".to_string()
    } else {
        format!("recorded a budget row for {}", added.join(", "))
    });
    Ok(())
}

/// Lower one tier's row to what the tree measures.
fn lower_budget(
    tree: &Tree,
    measurement: &Measurement,
    tier: &str,
    report: &mut Report,
) -> Result<(), GateError> {
    let mut declared = declared_budget(tree)?;
    let Some(row) = measurement.tiers.iter().find(|row| row.tier == tier) else {
        return Err(GateError::new(
            format!("no tier named `{tier}` is declared in the ownership record"),
            "name a tier the ownership record declares",
        ));
    };
    let Some(pin) = declared.get(tier) else {
        return Err(GateError::new(
            format!("tier `{tier}` has no budget row to lower"),
            "record the row with `--write-budget` first",
        ));
    };
    if raises_any_column(&row.cost, pin) {
        return Err(GateError::new(
            format!("tier `{tier}` measures above its row in at least one column"),
            "a budget row is never raised here; edit the row by hand in the change that earned the increase",
        ));
    }
    declared.insert(tier.to_string(), row.cost.clone());
    let text = render_budget(measurement, &declared);
    write_text(tree, BUDGET, &text)?;
    report.note(format!("lowered the budget row for `{tier}`"));
    Ok(())
}

/// Write `text` to `relative`, naming the path when the write fails.
fn write_text(tree: &Tree, relative: &str, text: &str) -> Result<(), GateError> {
    let path = tree.absolute(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            GateError::new(
                format!("cannot create {}: {error}", parent.display()),
                "make the budget directory writable",
            )
        })?;
    }
    fs::write(&path, text).map_err(|error| {
        GateError::new(
            format!("cannot write {relative}: {error}"),
            "make the budget file writable",
        )
    })
}

/// Every way `measurement` disagrees with the declared budget.
fn judge(measurement: &Measurement, declared: &BTreeMap<String, TierCost>) -> Vec<Finding> {
    let mut findings = Vec::new();

    let measured: BTreeSet<&str> = measurement
        .tiers
        .iter()
        .map(|row| row.tier.as_str())
        .collect();
    for tier in declared.keys() {
        if !measured.contains(tier.as_str()) {
            findings.push(Finding::in_file(
                BUDGET,
                format!("`{tier}` has a budget row and the ownership record declares no such tier"),
                "delete the stale row, or restore the tier in docs/CRATE_OWNERSHIP.toml",
            ));
        }
    }

    for row in &measurement.tiers {
        let Some(pin) = declared.get(&row.tier) else {
            findings.push(Finding::in_file(
                BUDGET,
                format!(
                    "tier `{}` verifies {} compile unit(s) and {} link unit(s) with no budget row",
                    row.tier, row.cost.compile_units, row.cost.link_units
                ),
                format!("record the row with `./cargo_full run --bin xtask -- {NAME} --write-budget`"),
            ));
            continue;
        };
        findings.extend(exact(
            &row.tier,
            "compile_units",
            row.cost.compile_units as u64,
            pin.compile_units as u64,
        ));
        findings.extend(exact(
            &row.tier,
            "link_units",
            row.cost.link_units as u64,
            pin.link_units as u64,
        ));
        findings.extend(exact(
            &row.tier,
            "duplicate_source_bytes",
            row.cost.duplicate_source_bytes,
            pin.duplicate_source_bytes,
        ));
        if row.cost.disk_use_bytes > pin.disk_use_bytes {
            findings.push(Finding::in_file(
                BUDGET,
                format!(
                    "tier `{}` test sources occupy {} byte(s) against a ceiling of {}",
                    row.tier, row.cost.disk_use_bytes, pin.disk_use_bytes
                ),
                "move the shared body into its owning package, or raise the ceiling in the change that earned it",
            ));
        }
    }

    for duplicate in &measurement.duplicated_shared_contracts {
        findings.push(Finding::new(
            format!("{duplicate} verbatim, so that contract has two owners"),
            format!("delete the copy and call the {SHARED_CONTRACT_OWNER} module"),
        ));
    }
    for escape in &measurement.escaping_path_includes {
        findings.push(Finding::new(
            format!("{escape} resolves outside its own Cargo package"),
            format!("move the body into {SHARED_CONTRACT_OWNER} and depend on it"),
        ));
    }
    findings
}

/// Judge `measurement` and record it.
fn inspect(tree: &Tree, measurement: &Measurement) -> crate::artifact_gate::Inspection {
    let mut inspection = crate::artifact_gate::Inspection::new();
    let declared = match declared_budget(tree) {
        Ok(declared) => declared,
        Err(error) => {
            inspection.find(Finding::in_file(
                BUDGET,
                error.to_string(),
                format!("record the per-tier budget with `./cargo_full run --bin xtask -- {NAME} --write-budget`"),
            ));
            BTreeMap::new()
        }
    };
    for finding in judge(measurement, &declared) {
        inspection.find(finding);
    }
    inspection.generates_host_evidence(ARTIFACT, measurement);
    inspection
}

/// Report a count that disagrees with its row, in either direction.
fn exact(tier: &str, column: &str, measured: u64, pinned: u64) -> Option<Finding> {
    if measured == pinned {
        return None;
    }
    Some(Finding::in_file(
        BUDGET,
        format!("tier `{tier}` measures {measured} {column} against a row of {pinned}"),
        if measured > pinned {
            format!(
                "record why the tier needs them by writing {measured} into `[tier.{tier}] {column}`, or remove the targets"
            )
        } else {
            format!(
                "lower the row with `./cargo_full run --bin xtask -- {NAME} --lower-budget {tier}`, which writes {measured}"
            )
        },
    ))
}

/// Holds each tier's verification cost to the budget its row declares.
pub struct VerificationBudget;

impl crate::gate::GateBehavior for VerificationBudget {
    fn usage(&self) -> &'static [&'static str] {
        &[
            "--write-budget records a row for a tier that has none and touches nothing else",
            "--lower-budget TIER sets one row to what the tree measures, refusing any raise",
        ]
    }

    fn write_arguments(&self) -> &'static [&'static str] {
        &["--write-budget", "--lower-budget"]
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let measurement = measure(&tree)?;
        let tiers = measurement.tiers.len();

        let mut report = if ctx.args.iter().any(|argument| argument == "--write-budget") {
            let mut report = Report::clean();
            report.produced(ARTIFACT);
            write_budget(&tree, &measurement, &mut report)?;
            report
        } else if let Some(position) = ctx
            .args
            .iter()
            .position(|argument| argument == "--lower-budget")
        {
            let Some(tier) = ctx
                .args
                .get(position + 1)
                .filter(|value| !value.starts_with("--"))
            else {
                return Err(GateError::new(
                    "`--lower-budget` was passed without a tier",
                    "name the tier whose row it lowers",
                ));
            };
            let mut report = Report::clean();
            report.produced(ARTIFACT);
            lower_budget(&tree, &measurement, tier, &mut report)?;
            report
        } else {
            crate::artifact_gate::settle_inspection(ctx, NAME, inspect(&tree, &measurement))
        };

        report.produced(BUDGET);
        report.cover_complete("workspace tiers", tiers);
        let compile_units: usize = measurement
            .tiers
            .iter()
            .map(|row| row.cost.compile_units)
            .sum();
        let link_units: usize = measurement.tiers.iter().map(|row| row.cost.link_units).sum();
        report.note(format!(
            "{compile_units} compile unit(s) and {link_units} link unit(s) across {tiers} tier(s)"
        ));
        report.note(format!(
            "{} shared contract module(s) in {SHARED_CONTRACT_OWNER}",
            measurement.shared_contracts.len()
        ));
        Ok(report)
    }
}

#[cfg(test)]
mod tests;
