//! The wire suites report the same result in the same order on every run.
//!
//! This was a local shell script an operator symlinked as a pre-push hook. It
//! ran the workspace checks the sweep already owns, and then the one assertion
//! nothing else made: run the wire contract suite twice with a fixed thread
//! count and compare the two logs line for line. The script mirrored a
//! `wire-ci.yml` workflow that no longer exists, so the assertion was reachable
//! only by an operator who remembered to install a hook, and its roster of
//! suites was a list in bash.
//!
//! The roster is read from the tree and the features each suite is built with
//! are read from the manifest entry that reserves them, because the script named
//! a feature `vyre-primitives` does not declare and cargo refuses such a target
//! before any suite runs. Comparison keeps the emitted order rather than
//! sorting, because a run-to-run reordering under a fixed thread count is
//! exactly the nondeterminism being looked for and sorting hides it.

use std::collections::{BTreeMap, BTreeSet};

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// Crate that owns the wire format.
const WIRE_CRATE: &str = "vyre-primitives";

/// Directory the wire suites live in.
const SUITE_DIR: &str = "vyre-primitives/tests";

/// Prefix a wire suite source carries.
const SUITE_PREFIX: &str = "wire_";

/// Runs the wire suites twice and holds them to one answer.
pub struct WireDeterminism;

impl Gate for WireDeterminism {
    fn name(&self) -> &'static str {
        "wire-determinism"
    }

    fn help(&self) -> &'static str {
        "Hold the wire suites to a tracked roster; --run executes each twice and compares the emitted order"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();

        let suites: Vec<String> = tree
            .paths()
            .iter()
            .filter_map(|path| {
                let path = path.to_str()?;
                let name = path
                    .strip_prefix(SUITE_DIR)?
                    .strip_prefix('/')?
                    .strip_suffix(".rs")?;
                (name.starts_with(SUITE_PREFIX) && !name.contains('/')).then(|| name.to_string())
            })
            .collect();
        if suites.is_empty() {
            report.find(Finding::in_file(
                SUITE_DIR,
                format!("no `{SUITE_PREFIX}*` suite is tracked, so nothing holds the wire format to a stable answer"),
                "restore the wire contract suite; the format is the boundary every backend serialises across",
            ));
        }

        let manifest_path = format!("{WIRE_CRATE}/Cargo.toml");
        let manifest = tree.read_toml(&manifest_path)?;
        let declared: BTreeSet<String> = manifest
            .get("features")
            .and_then(toml::Value::as_table)
            .map(|table| table.keys().cloned().collect())
            .unwrap_or_default();
        let mut reserved: BTreeMap<&String, Vec<String>> = BTreeMap::new();
        for suite in &suites {
            let features = required_features(&manifest, suite);
            let unknown: Vec<&String> = features
                .iter()
                .filter(|feature| !declared.contains(*feature))
                .collect();
            if !unknown.is_empty() {
                report.find(Finding::in_file(
                    &manifest_path,
                    format!(
                        "[[test]] `{suite}` requires {unknown:?}, which `{WIRE_CRATE}` does not declare, so cargo refuses the target before it runs"
                    ),
                    "reserve the features the crate declares; a suite behind a name that does not exist is never run twice, or once",
                ));
            }
            reserved.insert(suite, features);
        }

        if !ctx.has("--run") {
            report.note(format!("{} wire suite(s) tracked", suites.len()));
            return Ok(report);
        }

        for suite in &suites {
            let features = reserved.get(suite).map(Vec::as_slice).unwrap_or_default();
            let first = run_suite(ctx, suite, features)?;
            let second = run_suite(ctx, suite, features)?;
            if first != second {
                let divergence = first
                    .iter()
                    .zip(second.iter())
                    .position(|(left, right)| left != right)
                    .map_or_else(
                        || format!("{} lines against {}", first.len(), second.len()),
                        |at| {
                            format!(
                                "line {}: `{}` against `{}`",
                                at + 1,
                                first.get(at).map_or("", String::as_str),
                                second.get(at).map_or("", String::as_str)
                            )
                        },
                    );
                report.find(Finding::new(
                    format!("`{suite}` reported a different run twice over: {divergence}"),
                    "make the suite order-stable under a fixed thread count; a suite whose order moves cannot witness a regression",
                ));
            }
        }
        report.note(format!("{} wire suite(s) run twice", suites.len()));
        Ok(report)
    }
}

/// The features one manifest entry reserves for a suite, empty when cargo
/// discovers the target and builds it under the default set.
fn required_features(manifest: &toml::Table, suite: &str) -> Vec<String> {
    manifest
        .get("test")
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| entry.get("name").and_then(toml::Value::as_str) == Some(suite))
        .filter_map(|entry| entry.get("required-features"))
        .filter_map(toml::Value::as_array)
        .flatten()
        .filter_map(toml::Value::as_str)
        .map(str::to_string)
        .collect()
}

/// Emitted `test ...` lines of one suite, in the order the run reported them.
fn run_suite(ctx: &GateCtx, suite: &str, features: &[String]) -> Result<Vec<String>, GateError> {
    let mut command = crate::cargo_runner::command(&ctx.root);
    command.args(["test", "-p", WIRE_CRATE, "--test", suite]);
    if !features.is_empty() {
        command.args(["--features", &features.join(",")]);
    }
    let output = command
        .args(["--", "--nocapture", "--test-threads=1"])
        .output()
        .map_err(|error| {
            GateError::new(
                format!("`cargo test --test {suite}` could not be started: {error}"),
                "run the suite by hand; determinism cannot be judged without two runs",
            )
        })?;
    if !output.status.success() {
        return Err(GateError::new(
            format!(
                "`{suite}` failed, so determinism cannot be judged: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            "fix the failing suite first; two identical failures compare equal and would report determinism",
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| line.starts_with("test "))
        .map(str::to_string)
        .collect())
}

/// WHY: the feature set a suite is built with used to be a constant copied from
/// a shell script, and the constant named a feature the crate does not declare,
/// so every run cargo would have refused looked identical to a run that passed.
/// The reader is crate-private and no integration test can reach it, and the one
/// tree it runs against reserves nothing, which is the case that proves least.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_suite_reserves_only_what_its_own_manifest_entry_names() {
        let manifest: toml::Table = toml::from_str(
            "[[test]]\nname = \"wire_pack_into_contracts\"\nrequired-features = [\"matching\", \"hash\"]\n\n[[test]]\nname = \"wire_differential_std_io\"\n",
        )
        .expect("the fixture manifest parses");
        assert_eq!(
            required_features(&manifest, "wire_pack_into_contracts"),
            vec!["matching".to_string(), "hash".to_string()]
        );
        assert!(required_features(&manifest, "wire_differential_std_io").is_empty());
        assert!(required_features(&manifest, "wire_absent").is_empty());
    }
}
