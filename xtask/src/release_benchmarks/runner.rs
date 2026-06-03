use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

use super::optimization::suite_case_has_cpu_sota_contract;
use super::suite_inspect::read_text_bounded;
use super::types::MAX_RELEASE_BENCHMARK_TEXT_BYTES;

pub(super) fn run_named_benchmark(
    workspace_root: &Path,
    case_id: &str,
    backend: &str,
    output: &str,
    measured_samples: Option<usize>,
    sample_timeout_secs: u64,
) {
    let mut owned_args = vec![
        "run".to_string(),
        "-p".to_string(),
        "vyre-bench".to_string(),
        "--quiet".to_string(),
        "--".to_string(),
        "run".to_string(),
        "--suite".to_string(),
        "release".to_string(),
        "--case".to_string(),
        case_id.to_string(),
        "--backend".to_string(),
        backend.to_string(),
        "--enforce-budgets".to_string(),
        "--output".to_string(),
        output.to_string(),
        "--sample-timeout-secs".to_string(),
        sample_timeout_secs.to_string(),
    ];
    if let Some(samples) = measured_samples {
        owned_args.push("--measured-samples".to_string());
        owned_args.push(samples.to_string());
    }
    let borrowed = owned_args.iter().map(String::as_str).collect::<Vec<_>>();
    run_command(workspace_root, &borrowed);
}

pub(super) fn run_named_benchmark_if_needed(
    workspace_root: &Path,
    case_id: &str,
    backend: &str,
    output: &str,
    measured_samples: Option<usize>,
    sample_timeout_secs: u64,
    reuse_existing: bool,
) {
    if reuse_existing
        && benchmark_artifact_is_reusable(workspace_root, backend, case_id, case_id, output, false)
    {
        return;
    }
    run_named_benchmark(
        workspace_root,
        case_id,
        backend,
        output,
        measured_samples,
        sample_timeout_secs,
    );
}

pub(super) fn benchmark_artifact_is_reusable(
    workspace_root: &Path,
    backend: &str,
    family_id: &str,
    case_id: &str,
    output: &str,
    cpu_sota_100x_required: bool,
) -> bool {
    let path = workspace_root.join(output);
    let text = match read_text_bounded(&path, MAX_RELEASE_BENCHMARK_TEXT_BYTES) {
        Ok(text) => text,
        Err(_) => return false,
    };
    let Ok(report) = serde_json::from_str::<Value>(&text) else {
        return false;
    };
    let Some(report_source_tree_fingerprint) = report
        .get("source_tree_fingerprint")
        .and_then(Value::as_str)
        .filter(|fingerprint| !fingerprint.trim().is_empty())
    else {
        let Some(report_source_fingerprint) = report
            .get("source_fingerprint")
            .and_then(Value::as_str)
            .filter(|fingerprint| !fingerprint.trim().is_empty())
        else {
            return false;
        };
        let current_git = vyre_bench::probes::capture_git_info_at(workspace_root);
        let current_source_fingerprint = vyre_bench::probes::source_fingerprint(&current_git);
        if report_source_fingerprint != current_source_fingerprint {
            return false;
        }
        return benchmark_artifact_report_shape_is_reusable(
            &report,
            backend,
            family_id,
            case_id,
            cpu_sota_100x_required,
        );
    };
    let current_source_tree_fingerprint =
        vyre_bench::probes::source_tree_fingerprint_at(workspace_root);
    if report_source_tree_fingerprint != current_source_tree_fingerprint {
        return false;
    }
    benchmark_artifact_report_shape_is_reusable(
        &report,
        backend,
        family_id,
        case_id,
        cpu_sota_100x_required,
    )
}

fn benchmark_artifact_report_shape_is_reusable(
    report: &Value,
    backend: &str,
    family_id: &str,
    case_id: &str,
    cpu_sota_100x_required: bool,
) -> bool {
    if report.get("selected_backend").and_then(Value::as_str) != Some(backend) {
        return false;
    }
    if report
        .get("summary")
        .and_then(|summary| summary.get("failed"))
        .and_then(Value::as_u64)
        != Some(0)
    {
        return false;
    }
    let Some(case) = report
        .get("cases")
        .and_then(Value::as_array)
        .and_then(|cases| {
            cases
                .iter()
                .find(|case| case.get("id").and_then(Value::as_str) == Some(case_id))
        })
    else {
        return false;
    };
    if case.get("backend_id").and_then(Value::as_str) != Some(backend) {
        return false;
    }
    if case.get("status").and_then(Value::as_str) != Some("pass") {
        return false;
    }
    if cpu_sota_100x_required && !suite_case_has_cpu_sota_contract(case, backend, 100.0) {
        return false;
    }
    if cpu_sota_100x_required {
        let contract_passed = case
            .get("performance")
            .and_then(|performance| performance.get("contract_passed"))
            .and_then(Value::as_bool)
            == Some(true);
        let speedup_passed = case
            .get("performance")
            .and_then(|performance| performance.get("speedup_x"))
            .and_then(Value::as_f64)
            .is_some_and(|speedup| speedup >= 100.0);
        if !contract_passed || !speedup_passed {
            return false;
        }
    }
    let _ = family_id;
    true
}

pub(super) fn copy_artifact(workspace_root: &Path, source: &str, target: &str) {
    let source = workspace_root.join(source);
    let target = workspace_root.join(target);
    if let Some(parent) = target.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            eprintln!("Fix: failed to create `{}`: {error}", parent.display());
            std::process::exit(1);
        }
    }
    if let Err(error) = fs::copy(&source, &target) {
        eprintln!(
            "Fix: failed to copy `{}` to `{}`: {error}",
            source.display(),
            target.display()
        );
        std::process::exit(1);
    }
}

pub(super) fn run_command(workspace_root: &Path, args: &[&str]) {
    if let Err(message) = run_command_status(workspace_root, args) {
        eprintln!("{message}");
        std::process::exit(1);
    }
}

pub(super) fn run_command_status(workspace_root: &Path, args: &[&str]) -> Result<(), String> {
    let runner = cargo_runner(workspace_root);
    let status = Command::new(&runner)
        .args(args)
        .current_dir(workspace_root)
        .status();
    let display = format!("{} {}", runner.display(), args.join(" "));
    match status {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(format!("Fix: `{display}` failed with {status}")),
        Err(error) => Err(format!(
            "Fix: failed to run `{display}`: {error}. Set VYRE_CARGO_RUNNER to the bounded workspace cargo wrapper if it is not named `cargo_full`."
        )),
    }
}

pub(super) fn cargo_runner(workspace_root: &Path) -> PathBuf {
    if let Some(runner) = std::env::var_os("VYRE_CARGO_RUNNER") {
        return PathBuf::from(runner);
    }
    let local = workspace_root.join("cargo_full");
    if local.is_file() {
        return local;
    }
    PathBuf::from("cargo_full")
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::TempDir;

    #[test]
    fn wgpu_reuse_accepts_matching_passed_artifact() {
        let dir = TempDir::new().expect("Fix: create temp workspace for WGPU reuse test.");
        write_benchmark_artifact(
            dir.path(),
            "release/evidence/benchmarks/wgpu-condition.json",
            serde_json::json!({
                "selected_backend": "wgpu",
                "source_fingerprint": current_test_source_fingerprint(dir.path()),
                "summary": {"failed": 0},
                "cases": [
                    {"id": "release.condition_eval.1m", "backend_id": "wgpu", "status": "pass"}
                ]
            }),
        );

        assert!(
            benchmark_artifact_is_reusable(
                dir.path(),
                "wgpu",
                "condition-eval",
                "release.condition_eval.1m",
                "release/evidence/benchmarks/wgpu-condition.json",
                false,
            ),
            "Fix: --reuse-existing should skip valid WGPU fallback artifacts instead of rerunning parity benchmarks."
        );
    }

    #[test]
    fn reuse_prefers_matching_source_tree_fingerprint() {
        let dir = TempDir::new().expect("Fix: create temp workspace for source-tree reuse test.");
        write_benchmark_artifact(
            dir.path(),
            "release/evidence/benchmarks/wgpu-source-tree.json",
            serde_json::json!({
                "selected_backend": "wgpu",
                "source_fingerprint": "git:stale:dirty=false",
                "source_tree_fingerprint": current_test_source_tree_fingerprint(dir.path()),
                "summary": {"failed": 0},
                "cases": [
                    {"id": "release.condition_eval.1m", "backend_id": "wgpu", "status": "pass"}
                ]
            }),
        );

        assert!(
            benchmark_artifact_is_reusable(
                dir.path(),
                "wgpu",
                "condition-eval",
                "release.condition_eval.1m",
                "release/evidence/benchmarks/wgpu-source-tree.json",
                false,
            ),
            "Fix: reusable benchmark evidence should survive evidence-only commit changes via source_tree_fingerprint."
        );
    }

    #[test]
    fn wgpu_reuse_rejects_backend_or_case_drift() {
        let dir = TempDir::new().expect("Fix: create temp workspace for WGPU reuse drift test.");
        write_benchmark_artifact(
            dir.path(),
            "release/evidence/benchmarks/wgpu-with-cuda-backend.json",
            serde_json::json!({
                "selected_backend": "cuda",
                "source_fingerprint": current_test_source_fingerprint(dir.path()),
                "summary": {"failed": 0},
                "cases": [
                    {"id": "release.condition_eval.1m", "backend_id": "cuda", "status": "pass"}
                ]
            }),
        );
        write_benchmark_artifact(
            dir.path(),
            "release/evidence/benchmarks/wgpu-wrong-case.json",
            serde_json::json!({
                "selected_backend": "wgpu",
                "source_fingerprint": current_test_source_fingerprint(dir.path()),
                "summary": {"failed": 0},
                "cases": [
                    {"id": "release.other.1m", "backend_id": "wgpu", "status": "pass"}
                ]
            }),
        );

        assert!(
            !benchmark_artifact_is_reusable(
                dir.path(),
                "wgpu",
                "condition-eval",
                "release.condition_eval.1m",
                "release/evidence/benchmarks/wgpu-with-cuda-backend.json",
                false,
            ),
            "Fix: WGPU reuse must reject artifacts whose selected backend drifted to CUDA."
        );
        assert!(
            !benchmark_artifact_is_reusable(
                dir.path(),
                "wgpu",
                "condition-eval",
                "release.condition_eval.1m",
                "release/evidence/benchmarks/wgpu-wrong-case.json",
                false,
            ),
            "Fix: WGPU reuse must reject artifacts that do not contain the requested release case."
        );
    }

    #[test]
    fn reuse_rejects_stale_source_fingerprint() {
        let dir = TempDir::new().expect("Fix: create temp workspace for stale source test.");
        write_benchmark_artifact(
            dir.path(),
            "release/evidence/benchmarks/wgpu-stale-source.json",
            serde_json::json!({
                "selected_backend": "wgpu",
                "source_fingerprint": "git:stale:dirty=false",
                "summary": {"failed": 0},
                "cases": [
                    {"id": "release.condition_eval.1m", "backend_id": "wgpu", "status": "pass"}
                ]
            }),
        );

        assert!(
            !benchmark_artifact_is_reusable(
                dir.path(),
                "wgpu",
                "condition-eval",
                "release.condition_eval.1m",
                "release/evidence/benchmarks/wgpu-stale-source.json",
                false,
            ),
            "Fix: --reuse-existing must rerun benchmark artifacts captured from a different source fingerprint."
        );
    }

    #[test]
    fn reuse_rejects_stale_source_tree_fingerprint() {
        let dir = TempDir::new().expect("Fix: create temp workspace for stale source-tree test.");
        write_benchmark_artifact(
            dir.path(),
            "release/evidence/benchmarks/wgpu-stale-source-tree.json",
            serde_json::json!({
                "selected_backend": "wgpu",
                "source_fingerprint": current_test_source_fingerprint(dir.path()),
                "source_tree_fingerprint": "source-tree-v1:stale",
                "summary": {"failed": 0},
                "cases": [
                    {"id": "release.condition_eval.1m", "backend_id": "wgpu", "status": "pass"}
                ]
            }),
        );

        assert!(
            !benchmark_artifact_is_reusable(
                dir.path(),
                "wgpu",
                "condition-eval",
                "release.condition_eval.1m",
                "release/evidence/benchmarks/wgpu-stale-source-tree.json",
                false,
            ),
            "Fix: source_tree_fingerprint must remain a real freshness gate, not only an optional annotation."
        );
    }

    fn current_test_source_fingerprint(workspace_root: &Path) -> String {
        let git = vyre_bench::probes::capture_git_info_at(workspace_root);
        vyre_bench::probes::source_fingerprint(&git)
    }

    fn current_test_source_tree_fingerprint(workspace_root: &Path) -> String {
        vyre_bench::probes::source_tree_fingerprint_at(workspace_root)
    }

    fn write_benchmark_artifact(workspace_root: &Path, relative: &str, value: Value) {
        let path = workspace_root.join(relative);
        fs::create_dir_all(
            path.parent()
                .expect("Fix: benchmark artifact test path must have a parent directory."),
        )
        .expect("Fix: create benchmark artifact test directory.");
        fs::write(&path, format!("{value}\n")).expect("Fix: write benchmark artifact test JSON.");
    }
}

pub(super) struct Config {
    backend: String,
    only: Option<String>,
    measured_samples: Option<usize>,
    sample_timeout_secs: u64,
    include_wgpu_comparison: bool,
    reuse_existing: bool,
}
