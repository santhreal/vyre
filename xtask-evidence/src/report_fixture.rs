//! Benchmark report fragments shared by the evidence tests.
//!
//! Every reader of a benchmark report is fed a schema-2 case carrying a
//! CPU-SOTA baseline contract. That contract is twelve lines of JSON saying the
//! same thing every time, the run summary another six, and the measured case
//! around it another fifteen, so the fixtures were the largest block of copied
//! source in this crate: adding a field to any of them meant editing every
//! copy, and a copy that was missed read as a deliberate variation. Build them
//! here and let each test spell out only the values it is actually about.
//!
//! Two contract shapes exist because two layers parse them. The suite
//! inspection readers require the named, crate-attributed baseline; the
//! semantic readers and the release gate checks parse only the class, the
//! backends and the demanded speedup.

use serde_json::{json, Value};

/// A single-case run summary in the long form the artifact readers require.
pub(crate) fn case_summary(passed: u64, failed: u64) -> Value {
    json!({
        "total_cases": 1,
        "passed": passed,
        "failed": failed,
        "total_time_ns": 0,
        "cache_hit_rate": null
    })
}

/// The CPU-SOTA baseline contract a release benchmark case carries.
pub(crate) fn cpu_sota_contract(primitive: &str, backend_ids: &[&str]) -> Value {
    json!({
        "primitive": primitive,
        "baselines": [
            {
                "name": "CPU-SOTA",
                "crate_name": "vyre-runtime",
                "class": "CpuSota",
                "min_speedup_x": 100.0,
                "backend_ids": backend_ids
            }
        ]
    })
}

/// The CPU-SOTA baseline contract in the short form the semantic readers and the
/// release gate checks parse.
pub(crate) fn cpu_sota_baseline(backend_ids: &[&str], min_speedup_x: f64) -> Value {
    json!({
        "baselines": [
            {
                "class": "CpuSota",
                "backend_ids": backend_ids,
                "min_speedup_x": min_speedup_x
            }
        ]
    })
}

/// One measured CPU-SOTA case: which backend ran it, the status the runner
/// recorded, and the wall timings a reader derives the achieved speedup from.
///
/// The claimed `performance.speedup_x` is fixed at a generous 200x on purpose.
/// Every reader under test must decide from the measured
/// `baseline_wall_ns / wall_ns` ratio, so a case that passes while the claim
/// alone would carry it proves the reader ignored the claim.
pub(crate) fn cpu_sota_case(
    id: &str,
    backend_id: &str,
    status: &str,
    contract_backend_ids: &[&str],
    wall_p50: u64,
    baseline_wall_p50: u64,
) -> Value {
    json!({
        "id": id,
        "backend_id": backend_id,
        "status": status,
        "contract": cpu_sota_baseline(contract_backend_ids, 100.0),
        "metrics": {
            "wall_ns": {"p50": wall_p50},
            "baseline_wall_ns": {"p50": baseline_wall_p50}
        },
        "performance": {"contract_passed": true, "speedup_x": 200.0}
    })
}

/// Wall and baseline wall timings as `[p50, p95, p99]`, each declaring the thirty
/// samples the release gate demands before it will read a percentile.
pub(crate) fn percentile_metrics(wall_ns: [u64; 3], baseline_wall_ns: [u64; 3]) -> Value {
    json!({
        "wall_ns": {
            "samples": 30,
            "p50": wall_ns[0],
            "p95": wall_ns[1],
            "p99": wall_ns[2]
        },
        "baseline_wall_ns": {
            "samples": 30,
            "p50": baseline_wall_ns[0],
            "p95": baseline_wall_ns[1],
            "p99": baseline_wall_ns[2]
        }
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{case_summary, cpu_sota_baseline, cpu_sota_case, cpu_sota_contract};

    /// WHY: the fragments stand in for hand-written JSON in a dozen fixtures,
    /// so a drift in either shape would silently change what those fixtures
    /// mean. Pin the exact keys and values the readers parse.
    #[test]
    fn case_summary_carries_every_field_a_reader_parses() {
        let summary = case_summary(1, 0);
        assert_eq!(summary["total_cases"], 1);
        assert_eq!(summary["passed"], 1);
        assert_eq!(summary["failed"], 0);
        assert_eq!(summary["total_time_ns"], 0);
        assert!(
            summary["cache_hit_rate"].is_null(),
            "Fix: cache_hit_rate must stay present and null; readers distinguish absent from null."
        );
        assert_eq!(
            summary.as_object().map(serde_json::Map::len),
            Some(5),
            "Fix: an extra summary field changes every fixture built on this helper."
        );
    }

    /// WHY: `backend_ids` is what decides whether a baseline applies to the
    /// case under inspection, and the class/name pair is what marks it as the
    /// CPU-SOTA baseline. A fixture that got either wrong would exercise the
    /// wrong branch while still looking like a CPU-SOTA case.
    #[test]
    fn cpu_sota_contract_names_the_baseline_and_its_backends() {
        let contract = cpu_sota_contract("release condition eval", &["cuda", "wgpu"]);
        assert_eq!(contract["primitive"], "release condition eval");
        let baselines = contract["baselines"]
            .as_array()
            .expect("Fix: contract must carry a baselines array.");
        assert_eq!(baselines.len(), 1);
        assert_eq!(baselines[0]["name"], "CPU-SOTA");
        assert_eq!(baselines[0]["class"], "CpuSota");
        assert_eq!(baselines[0]["crate_name"], "vyre-runtime");
        assert_eq!(baselines[0]["min_speedup_x"], 100.0);
        assert_eq!(
            baselines[0]["backend_ids"],
            json!(["cuda", "wgpu"]),
            "Fix: backend_ids decides which case the baseline applies to."
        );
    }

    /// WHY: the short contract is what the semantic readers and the release gate
    /// checks match on. `backend_ids` decides whether a baseline applies to the
    /// case at all, and `min_speedup_x` is the threshold under test, so a
    /// fixture that got either wrong would exercise the wrong branch.
    #[test]
    fn the_short_baseline_carries_only_what_the_readers_match_on() {
        let contract = cpu_sota_baseline(&["cuda"], 1.01);
        let baselines = contract["baselines"]
            .as_array()
            .expect("Fix: contract must carry a baselines array.");
        assert_eq!(baselines.len(), 1);
        assert_eq!(baselines[0]["class"], "CpuSota");
        assert_eq!(baselines[0]["backend_ids"], json!(["cuda"]));
        assert_eq!(baselines[0]["min_speedup_x"], 1.01);
        assert_eq!(
            baselines[0].as_object().map(serde_json::Map::len),
            Some(3),
            "Fix: the short baseline must stay short; the named form is cpu_sota_contract."
        );
    }

    /// WHY: every reader under test must derive the achieved speedup from the
    /// measured wall timings. The generous claimed speedup_x is what makes a
    /// claim-trusting reader visible, so it has to be present and unchanged, and
    /// the timings have to be exactly what the caller asked for.
    #[test]
    fn a_case_reports_measured_timings_alongside_a_generous_claim() {
        let case = cpu_sota_case("release.condition_eval.1m", "cuda", "pass", &["cuda"], 10, 2000);
        assert_eq!(case["id"], "release.condition_eval.1m");
        assert_eq!(case["backend_id"], "cuda");
        assert_eq!(case["status"], "pass");
        assert_eq!(case["metrics"]["wall_ns"]["p50"], 10);
        assert_eq!(case["metrics"]["baseline_wall_ns"]["p50"], 2000);
        assert_eq!(case["performance"]["contract_passed"], true);
        assert_eq!(case["performance"]["speedup_x"], 200.0);
        assert_eq!(
            case["contract"],
            cpu_sota_baseline(&["cuda"], 100.0),
            "Fix: a case carries the short baseline the gate checks parse."
        );
    }
}
