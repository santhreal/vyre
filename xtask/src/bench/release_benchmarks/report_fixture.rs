//! Benchmark report fragments shared by the suite inspection tests.
//!
//! Every inspection test feeds the readers a schema-2 report whose case
//! carries a CPU-SOTA baseline contract. That contract is twelve lines of JSON
//! saying the same thing every time, and the run summary another six, so the
//! fixtures were the largest block of copied source in this crate: adding a
//! field to either meant editing every copy, and a copy that was missed read
//! as a deliberate variation. Build both here and let each test spell out only
//! the values it is actually about.

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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{case_summary, cpu_sota_contract};

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
}
