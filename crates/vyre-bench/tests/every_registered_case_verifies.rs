//! Every registered benchmark case runs through the shipped `vyre-bench run`
//! command and its device output verifies against its baseline.
//!
//! WHY: a case's `verify` runs only when someone measures that case, and the
//! release lanes measure a named handful. A case whose device output stops
//! matching its baseline stays green until the next full release measurement.
//! This drives the CLI once over the whole registry, with one cold measured
//! sample per case, and requires a passing correctness verdict from each.
//!
//! The case set is the registry, read at run time, so a new case joins this
//! test with no edit here. A performance contract cannot be judged from one
//! cold sample, so a case may report `failed` only when its correctness passed
//! and its performance contract is what failed. Not covered: whether a case
//! meets its performance contract, which the release lanes judge with the
//! sample count the contract was calibrated for.
//!
//! Gated on `device-tests`: every case dispatches on the device it selects.
#![cfg(feature = "device-tests")]

use std::collections::BTreeSet;
use std::process::Command;

#[test]
fn every_registered_case_verifies_through_the_cli() {
    let registered: BTreeSet<String> = vyre_bench::registry::collect_all()
        .iter()
        .map(|case| case.metadata().id.0)
        .collect();
    assert!(
        !registered.is_empty(),
        "Fix: the benchmark registry collected no cases"
    );
    let mut command = Command::new(env!("CARGO_BIN_EXE_vyre-bench"));
    command.args([
        "run",
        "--suite",
        "smoke",
        "--warmup-samples",
        "0",
        "--measured-samples",
        "1",
        "--format",
        "json",
    ]);
    for id in &registered {
        command.args(["--case", id]);
    }
    let output = command
        .env("VYRE_ALLOW_FEW_SAMPLES", "1")
        .output()
        .expect("Fix: the vyre-bench binary must launch");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
            panic!("Fix: `run --format json` must print one report: {error}\n{stderr}")
        });
    let cases = report["cases"]
        .as_array()
        .unwrap_or_else(|| panic!("Fix: the report must list its cases:\n{stderr}"));

    let reported: BTreeSet<String> = cases
        .iter()
        .map(|case| {
            case["id"]
                .as_str()
                .expect("Fix: a case report carries its id")
                .to_string()
        })
        .collect();
    assert_eq!(
        reported, registered,
        "Fix: every registered case must report once"
    );
    assert_eq!(
        cases.len(),
        registered.len(),
        "Fix: no case may report twice"
    );

    let unverified: Vec<String> = cases
        .iter()
        .filter(|case| {
            let verified =
                !case["correctness"].is_null() && case["correctness"].get("Invalid").is_none();
            let performance_only = case["status"] != "failed"
                || case["performance"]["contract_passed"] == serde_json::Value::Bool(false);
            !(verified && performance_only)
        })
        .map(|case| {
            format!(
                "{} status={} correctness={}",
                case["id"], case["status"], case["correctness"]
            )
        })
        .collect();
    assert!(
        unverified.is_empty(),
        "Fix: these cases did not verify their device output against the baseline:\n{}\n{stderr}",
        unverified.join("\n")
    );
}
