//! The shipped binary exposes case listing and snapshot diff as auditable JSON.

use std::process::{Command, Output};

fn run_benchmark_cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_vyre-bench"))
        .args(args)
        .output()
        .expect("vyre-bench binary must launch")
}

#[test]
fn list_emits_registered_case_metadata_as_json() {
    let output = run_benchmark_cli(&["list", "--format", "json"]);
    assert!(
        output.status.success(),
        "list failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let cases: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("list output must be JSON");
    let cases = cases
        .as_array()
        .expect("list output must be a JSON array of benchmark cases");
    assert!(
        cases.iter().all(|case| {
            case.get("id").and_then(serde_json::Value::as_str).is_some()
                && case
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .is_some()
        }),
        "every listed case must carry its registry identity"
    );
}

#[test]
fn snapshot_diff_reports_a_missing_baseline() {
    let output = run_benchmark_cli(&[
        "snapshot-diff",
        "--base",
        "0000000000000000000000000000000000000000",
    ]);
    assert!(!output.status.success(), "missing baseline must fail");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(
            "snapshot for commit `0000000000000000000000000000000000000000` not found in snapshots/"
        ),
        "failure must identify the missing baseline and expected directory: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
