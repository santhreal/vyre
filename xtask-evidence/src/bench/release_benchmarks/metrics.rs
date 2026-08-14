use std::path::Path;

use serde::Serialize;
use serde_json::Value;

use crate::bench::benchmark_evidence_semantics::{
    COLD_PIPELINE_BUILD_METRICS, SCAN_THROUGHPUT_METRICS,
};

pub(super) fn write_json(path: &Path, value: &impl Serialize) {
    if let Err(error) = xtask::json_output::write_pretty_json(path, value) {
        eprintln!("Fix: {error}");
        std::process::exit(1);
    }
}

pub(super) fn release_axis_blockers(reports: &[Value]) -> Vec<String> {
    let mut blockers = Vec::new();
    if reports.is_empty() {
        blockers.push("no benchmark case reports available for release axes".to_string());
    }
    if reports.len() < 12 {
        blockers.push(format!(
            "only {} benchmark report(s) available for release axes; release needs at least 12 workload reports",
            reports.len()
        ));
    }
    if min_metric_p50(reports, "wall_ns").is_none() {
        blockers.push("missing wall_ns metric for warm_us_per_file".to_string());
    }
    if min_first_available_metric_p50(reports, COLD_PIPELINE_BUILD_METRICS).is_none() {
        blockers.push("missing cold/compile metric for cold_pipeline_build_ms".to_string());
    }
    if max_first_available_metric_p50(reports, SCAN_THROUGHPUT_METRICS).is_none() {
        blockers.push("missing throughput metric for gbs_scan_throughput".to_string());
    }
    if max_vram_mib(reports).is_none() {
        blockers.push("missing GPU memory evidence for max_vram_mib".to_string());
    }
    blockers
}

pub(super) fn min_first_available_metric_p50(reports: &[Value], keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| min_metric_p50(reports, key))
}

/// The maximum p50 of the first metric in `keys` any report carries.
pub(super) fn max_first_available_metric_p50(reports: &[Value], keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| max_metric_p50(reports, key))
}

pub(super) fn min_metric_p50(reports: &[Value], key: &str) -> Option<u64> {
    metric_p50_values(reports, key).into_iter().min()
}

pub(super) fn max_metric_p50(reports: &[Value], key: &str) -> Option<u64> {
    metric_p50_values(reports, key).into_iter().max()
}

pub(super) fn metric_p50_values(reports: &[Value], key: &str) -> Vec<u64> {
    let mut values = Vec::new();
    for report in reports {
        let Some(cases) = report.get("cases").and_then(Value::as_array) else {
            continue;
        };
        for case in cases {
            let Some(metrics) = case.get("metrics").and_then(Value::as_object) else {
                continue;
            };
            let Some(value) = metrics
                .get(key)
                .and_then(|metric| metric.get("p50"))
                .and_then(Value::as_u64)
            else {
                continue;
            };
            values.push(value);
        }
    }
    values
}

pub(super) fn max_observed_ulp(reports: &[Value]) -> Option<u32> {
    let mut max_ulp = None::<u32>;
    for report in reports {
        let Some(cases) = report.get("cases").and_then(Value::as_array) else {
            continue;
        };
        for case in cases {
            if let Some(ulp) = case
                .get("correctness")
                .and_then(|correctness| correctness.get("Toleranced"))
                .and_then(|toleranced| toleranced.get("max_observed_ulp"))
                .and_then(Value::as_u64)
            {
                let ulp = ulp.min(u64::from(u32::MAX)) as u32;
                max_ulp = Some(max_ulp.map_or(ulp, |current| current.max(ulp)));
            }
        }
    }
    max_ulp
}

pub(super) fn max_vram_mib(reports: &[Value]) -> Option<u64> {
    let mut max_mib = None::<u64>;
    for report in reports {
        if let Some(devices) = report
            .get("environment")
            .and_then(|environment| environment.get("gpu_devices"))
            .and_then(Value::as_array)
        {
            for device in devices {
                if let Some(mib) = device.get("memory_total_mib").and_then(Value::as_u64) {
                    max_mib = Some(max_mib.map_or(mib, |current| current.max(mib)));
                }
            }
        }
        let Some(cases) = report.get("cases").and_then(Value::as_array) else {
            continue;
        };
        for case in cases {
            if let Some(mib) = case
                .get("metrics")
                .and_then(|metrics| metrics.get("memory_total_mib"))
                .and_then(|metric| metric.get("p50"))
                .and_then(Value::as_u64)
            {
                max_mib = Some(max_mib.map_or(mib, |current| current.max(mib)));
            }
        }
    }
    max_mib
}
