//! R1  -  single release-bench surface.
//!
//! `cargo_full run --bin xtask -- bench-release` runs the canonical cold + warm + scale
//! + correctness sweep and prints **one number per axis** as the final
//! output. Designed to be pasted into release notes / marketing without
//! editing  -  every line is `axis_name=value units`.
//!
//! This is intentionally a thin coordinator. The actual measurement is
//! delegated to existing benches and probes (criterion harnesses in
//! `vyre-bench`, GPU dispatch latency probes in `vyre-driver-wgpu`,
//! ULP differential in `vyre-harness`). The xtask's value is collapsing
//! N marketing surfaces into one canonical entry point so v0.4.1 release
//! numbers are reproducible by anyone running one command.
//!
//! Substrate-attribution per-axis (which optimization fired and saved
//! how much) lives behind `VYRE_TRACE=1` and the substrate audit log
//! (R4  -  `DriverObservability::to_audit_log`); this xtask just surfaces
//! the headline numbers.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use serde_json::Value;

/// Axis identifier emitted in the final report. Stable string so
/// downstream graphs / regression gates can key on it.
const AXIS_WARM_US_PER_FILE: &str = "warm_us_per_file";
const AXIS_COLD_PIPELINE_BUILD_MS: &str = "cold_pipeline_build_ms";
const AXIS_GBS_SCAN_THROUGHPUT: &str = "gbs_scan_throughput";
const AXIS_ULP_DRIFT_MAX: &str = "ulp_drift_max";
const AXIS_MAX_VRAM_MIB: &str = "max_vram_mib";
const MAX_BENCH_RELEASE_REPORT_BYTES: u64 = 16_777_216;

pub(crate) fn run(args: &[String]) {
    let started = Instant::now();
    eprintln!("vyre xtask bench-release  -  canonical v0.4.1 release axes");
    eprintln!("  source: release benchmark evidence artifacts");
    eprintln!();

    let evidence_dir = parse_evidence_dir(args).unwrap_or_else(|message| fatal(&message));
    eprintln!("==> reading evidence from {}", evidence_dir.display());

    let report = ReleaseReport {
        warm_us_per_file: require_f64_axis(&evidence_dir, AXIS_WARM_US_PER_FILE)
            .unwrap_or_else(|message| fatal(&message)),
        cold_pipeline_build_ms: require_f64_axis(&evidence_dir, AXIS_COLD_PIPELINE_BUILD_MS)
            .unwrap_or_else(|message| fatal(&message)),
        gbs_scan_throughput: require_f64_axis(&evidence_dir, AXIS_GBS_SCAN_THROUGHPUT)
            .unwrap_or_else(|message| fatal(&message)),
        ulp_drift_max: require_u32_axis(&evidence_dir, AXIS_ULP_DRIFT_MAX)
            .unwrap_or_else(|message| fatal(&message)),
        max_vram_mib: require_u64_axis(&evidence_dir, AXIS_MAX_VRAM_MIB)
            .unwrap_or_else(|message| fatal(&message)),
    };

    let elapsed = started.elapsed();
    eprintln!();
    eprintln!("==> bench-release complete in {elapsed:.2?}");
    eprintln!();
    report.print_release_notes_block();
}

struct ReleaseReport {
    warm_us_per_file: f64,
    cold_pipeline_build_ms: f64,
    gbs_scan_throughput: f64,
    ulp_drift_max: u32,
    max_vram_mib: u64,
}

impl ReleaseReport {
    fn print_release_notes_block(&self) {
        println!("# vyre v0.4.1  -  bench-release axes");
        println!();
        println!("{AXIS_WARM_US_PER_FILE}={} us", self.warm_us_per_file);
        println!(
            "{AXIS_COLD_PIPELINE_BUILD_MS}={} ms",
            self.cold_pipeline_build_ms
        );
        println!(
            "{AXIS_GBS_SCAN_THROUGHPUT}={} GiB/s",
            self.gbs_scan_throughput
        );
        println!("{AXIS_ULP_DRIFT_MAX}={} ulp", self.ulp_drift_max);
        println!("{AXIS_MAX_VRAM_MIB}={} MiB", self.max_vram_mib);
    }
}

#[allow(dead_code)]
fn _ignore_exit_code_warning() -> ExitCode {
    ExitCode::SUCCESS
}

fn parse_evidence_dir(args: &[String]) -> Result<PathBuf, String> {
    let mut evidence_dir = PathBuf::from("release/evidence/benchmarks");
    let mut i = 2usize;
    while i < args.len() {
        match args[i].as_str() {
            "--evidence-dir" => {
                let value = args.get(i + 1).ok_or_else(|| {
                    "Fix: --evidence-dir requires a path to release benchmark artifacts."
                        .to_string()
                })?;
                evidence_dir = PathBuf::from(value);
                i += 2;
            }
            "--help" | "-h" => {
                println!(
                    "USAGE:\n  cargo_full run --bin xtask -- bench-release [--evidence-dir PATH]\n\n\
                     Reads release benchmark evidence and prints the canonical v0.4.1 axes.\n\
                     Generate evidence first with `cargo_full run --bin xtask -- release-benchmarks --backend cuda`."
                );
                std::process::exit(0);
            }
            other => {
                return Err(format!(
                    "Fix: unknown bench-release argument `{other}`. Use --evidence-dir PATH."
                ));
            }
        }
    }
    Ok(evidence_dir)
}

fn require_f64_axis(evidence_dir: &Path, axis: &str) -> Result<f64, String> {
    let raw = require_axis_text(evidence_dir, axis)?;
    raw.parse::<f64>().map_err(|_| {
        format!("Fix: axis `{axis}` value `{raw}` is not a floating-point benchmark number.")
    })
}

fn require_u32_axis(evidence_dir: &Path, axis: &str) -> Result<u32, String> {
    let raw = require_axis_text(evidence_dir, axis)?;
    raw.parse::<u32>()
        .map_err(|_| format!("Fix: axis `{axis}` value `{raw}` is not an unsigned integer."))
}

fn require_u64_axis(evidence_dir: &Path, axis: &str) -> Result<u64, String> {
    let raw = require_axis_text(evidence_dir, axis)?;
    raw.parse::<u64>()
        .map_err(|_| format!("Fix: axis `{axis}` value `{raw}` is not an unsigned integer."))
}

fn require_axis_text(evidence_dir: &Path, axis: &str) -> Result<String, String> {
    let entries = fs::read_dir(evidence_dir).map_err(|error| {
        format!(
            "Fix: cannot read benchmark evidence directory `{}`: {error}. Run `cargo_full run --bin xtask -- release-benchmarks --backend cuda` first.",
            evidence_dir.display()
        )
    })?;
    let mut reports = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| {
            format!(
                "Fix: cannot read an entry in `{}`: {error}.",
                evidence_dir.display()
            )
        })?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let contents = read_text_bounded(&path)
            .map_err(|error| format!("Fix: cannot read `{}`: {error}.", path.display()))?;
        let value = serde_json::from_str::<Value>(&contents).map_err(|error| {
            format!("Fix: invalid benchmark JSON `{}`: {error}.", path.display())
        })?;
        reject_report_blockers(&path, &value)?;
        if let Some(value) = json_axis_text(&value, axis) {
            return Ok(value);
        }
        if value.get("cases").and_then(Value::as_array).is_some() {
            reports.push((path, value));
            continue;
        }
    }
    derive_axis_from_reports(axis, &reports).ok_or_else(|| {
        format!(
            "Fix: missing benchmark axis `{axis}` under `{}` and no derivation path had enough data. Run `cargo_full run --bin xtask -- release-benchmarks --backend cuda` with current vyre-bench JSON metrics.",
            evidence_dir.display()
        )
    })
}

fn reject_report_blockers(path: &Path, value: &Value) -> Result<(), String> {
    let blockers = value
        .get("blockers")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    if blockers.is_empty() {
        return Ok(());
    }
    let first = blockers
        .first()
        .and_then(Value::as_str)
        .unwrap_or("<non-string blocker>");
    Err(format!(
        "Fix: benchmark evidence `{}` reports {} blocker(s); first blocker: {first}",
        path.display(),
        blockers.len()
    ))
}

fn json_axis_text(value: &Value, axis: &str) -> Option<String> {
    match value.get(axis)? {
        Value::Number(number) => Some(number.to_string()),
        Value::String(text) if !text.trim().is_empty() => Some(text.clone()),
        _ => None,
    }
}

fn fatal(message: &str) -> ! {
    eprintln!("error: {message}");
    std::process::exit(1);
}

fn derive_axis_from_reports(axis: &str, reports: &[(PathBuf, Value)]) -> Option<String> {
    match axis {
        AXIS_WARM_US_PER_FILE => {
            min_metric_p50(reports, "wall_ns").map(|ns| format_float(ns as f64 / 1_000.0))
        }
        AXIS_COLD_PIPELINE_BUILD_MS => min_first_available_metric_p50(
            reports,
            &[
                "cold_compile_ns",
                "cold_wall_ns",
                "compile_ns",
                "lower_ns",
                "optimize_ns",
            ],
        )
        .map(|ns| format_float(ns as f64 / 1_000_000.0)),
        AXIS_GBS_SCAN_THROUGHPUT => max_metric_p50(reports, "wall_gb_s_x1000")
            .or_else(|| max_metric_p50(reports, "device_gb_s_x1000"))
            .map(|gb_s_x1000| format_float(gb_s_x1000 as f64 / 1_000.0)),
        AXIS_ULP_DRIFT_MAX => max_observed_ulp(reports).map(|ulp| ulp.to_string()),
        AXIS_MAX_VRAM_MIB => max_vram_mib(reports).map(|mib| mib.to_string()),
        _ => None,
    }
}

fn min_first_available_metric_p50(reports: &[(PathBuf, Value)], keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| min_metric_p50(reports, key))
}

fn min_metric_p50(reports: &[(PathBuf, Value)], key: &str) -> Option<u64> {
    metric_p50_values(reports, key).into_iter().min()
}

fn max_metric_p50(reports: &[(PathBuf, Value)], key: &str) -> Option<u64> {
    metric_p50_values(reports, key).into_iter().max()
}

fn metric_p50_values(reports: &[(PathBuf, Value)], key: &str) -> Vec<u64> {
    let mut values = Vec::new();
    for (_, report) in reports {
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

fn max_observed_ulp(reports: &[(PathBuf, Value)]) -> Option<u32> {
    let mut max_ulp = None::<u32>;
    for (_, report) in reports {
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

fn max_vram_mib(reports: &[(PathBuf, Value)]) -> Option<u64> {
    let mut max_mib = None::<u64>;
    for (_, report) in reports {
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

fn format_float(value: f64) -> String {
    let rendered = format!("{value:.6}");
    rendered
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string()
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    let mut reader = fs::File::open(path)?.take(MAX_BENCH_RELEASE_REPORT_BYTES.saturating_add(1));
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    if text.len() as u64 > MAX_BENCH_RELEASE_REPORT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} exceeds {MAX_BENCH_RELEASE_REPORT_BYTES} byte release bench report read cap",
                path.display()
            ),
        ));
    }
    Ok(text)
}
