//! Per-case execution driver. Calls the case `prepare`, runs the
//! measured iterations, harvests metrics, and evaluates the
//! performance contract.

use std::collections::BTreeMap;
use std::time::Instant;

use crate::api::case::{
    BenchContext, Correctness, DeviceBound, PerformanceContract, PerformanceEvaluation,
};
use crate::api::metric::{digest64_buffers, elapsed_ns, MetricStats};
use crate::api::suite::SuiteKind;
use crate::report::json::{benchmark_device_signature, benchmark_held_out_corpus_id, CaseReport};

use super::collect::collect_samples;
use super::stats::{compute_stats, percentile};
use super::target_samples;
use super::RunConfig;

pub(super) fn run_case(
    case: &'static dyn crate::api::case::BenchCase,
    ctx: &mut BenchContext,
    prepared: &mut crate::api::case::PreparedCase,
    suite: &SuiteKind,
    config: &RunConfig,
) -> Result<CaseReport, String> {
    let meta = case.metadata();
    let target_samples = config.measured_samples.unwrap_or_else(|| {
        let base = target_samples(suite);
        if base < 30 {
            30
        } else {
            base
        }
    });
    if matches!(suite, SuiteKind::Release) && target_samples < 30 {
        return Err(format!(
            "release suite measured_samples must unconditionally be >= 30 for CLT validity; got {target_samples}. Fix: pass --measured-samples 30 or higher."
        ));
    }
    if std::env::var("VYRE_ALLOW_FEW_SAMPLES").is_err() && target_samples < 30 {
        return Err(format!(
            "measured_samples must be >= 30 for CLT validity; got {target_samples}. Fix: pass --measured-samples 30 or set VYRE_ALLOW_FEW_SAMPLES=1 for local smoke-only debugging."
        ));
    }
    let mut samples: BTreeMap<&'static str, Vec<u64>> = BTreeMap::new();
    let mut correctness = None;
    // Cold-vs-warm separation: capture the first warmup
    // sample's wall-clock and per-stage breakdown so the report can
    // attribute time to cold-start (compile / cache miss / first
    // dispatch) versus warm steady-state. Subsequent warmup runs are
    // discarded as before.
    let mut cold_metrics: Option<crate::api::metric::BenchMetrics> = None;
    let mut cold_wall_ns: Option<u64> = None;
    let effective_warmup_samples = if matches!(suite, SuiteKind::Release) {
        config.warmup_samples.max(300)
    } else {
        config.warmup_samples
    };

    for warmup_index in 0..effective_warmup_samples {
        let started = Instant::now();
        ctx.include_baseline_outputs = warmup_index == 0;
        let run_result = case
            .run(ctx, prepared)
            .map_err(|error| format!("Warmup error on sample {warmup_index}: {error}"))?;
        let elapsed_ns = elapsed_ns(started);
        if warmup_index == 0 {
            case.verify(ctx, &run_result)
                .map_err(|error| format!("Warmup verify error: {error}"))?;
            cold_wall_ns = Some(elapsed_ns);
            cold_metrics = Some(run_result.metrics.clone());
        }
        if started.elapsed() > config.sample_timeout {
            return Err(format!(
                "Warmup sample {warmup_index} exceeded timeout {:?}",
                config.sample_timeout
            ));
        }
    }

    let mut determinism_p50s = Vec::new();
    let mut cpu_digest = None;
    let mut gpu_digest = None;

    for _d_run in 0..config.determinism_runs {
        let mut d_samples: BTreeMap<&'static str, Vec<u64>> = BTreeMap::new();
        for sample_index in 0..target_samples {
            let started = Instant::now();
            let alloc_region = vyre_alloc_probe::Region::new();
            ctx.include_baseline_outputs = sample_index == 0;
            let mut run_result = case
                .run(ctx, prepared)
                .map_err(|error| format!("Run error on sample {sample_index}: {error}"))?;
            let allocated = alloc_region.change();
            run_result
                .metrics
                .alloc_bytes
                .get_or_insert(allocated.bytes_allocated as u64);
            run_result
                .metrics
                .alloc_count
                .get_or_insert(allocated.allocations as u64);

            let (read, written) = case.bytes_touched(prepared);
            if read > 0 || written > 0 {
                run_result.metrics.bytes_read.get_or_insert(read);
                run_result.metrics.bytes_written.get_or_insert(written);
                run_result
                    .metrics
                    .bytes_touched
                    .get_or_insert(read + written);
            }
            if started.elapsed() > config.sample_timeout {
                return Err(format!(
                    "Measured sample {sample_index} exceeded timeout {:?}",
                    config.sample_timeout
                ));
            }
            if sample_index == 0 {
                correctness = Some(
                    case.verify(ctx, &run_result)
                        .map_err(|error| format!("Verify error: {error}"))?,
                );
                if let Some(outputs) = run_result.baseline_outputs.as_ref() {
                    cpu_digest.get_or_insert_with(|| digest64_buffers(outputs));
                }
                gpu_digest.get_or_insert_with(|| digest64_buffers(&run_result.outputs));
            }

            // Only capture NVIDIA hardware telemetry on the final CUDA sample to avoid jitter.
            if sample_index == target_samples - 1 && ctx.preferred_backend.id() == "cuda" {
                let nvml_counters = crate::probes::capture_nvml_telemetry().map_err(|error| {
                    format!("NVML telemetry error on sample {sample_index}: {error}")
                })?;
                run_result.metrics.gpu_counter.extend(nvml_counters);
            }

            let collect_baseline = sample_index >= config.baseline_warmup_runs
                || target_samples <= config.baseline_warmup_runs;
            collect_samples(&run_result, &mut d_samples, collect_baseline);
            collect_samples(&run_result, &mut samples, collect_baseline);
        }

        // B-4: Ensure we got enough samples before timing out
        let actual_samples = samples.get("wall_ns").map(|v| v.len()).unwrap_or(0);
        let allow_few =
            !matches!(suite, SuiteKind::Release) && std::env::var("VYRE_ALLOW_FEW_SAMPLES").is_ok();
        if actual_samples < 30 && !allow_few {
            let requirements = case.requirements();
            let case_id = meta.id.0;
            let workload_fingerprint = workload_fingerprint(case_id.as_str(), None);
            return Ok(CaseReport {
                id: case_id.clone(),
                workload_fingerprint: workload_fingerprint.clone(),
                name: meta.name,
                owner_crate: meta.owner_crate,
                workload_class: format!("{:?}", meta.workload),
                tags: meta.tags,
                backend_id: Some(ctx.preferred_backend.id().to_string()),
                device_signature: Some(benchmark_device_signature(
                    ctx.preferred_backend.device_profile(),
                )),
                held_out_corpus_id: Some(benchmark_held_out_corpus_id(&workload_fingerprint)),
                needs_gpu: requirements.needs_gpu,
                min_vram_bytes: requirements.min_vram_bytes,
                min_input_bytes: requirements.min_input_bytes,
                required_features: requirements.feature_set,
                status: "failed".to_string(),
                wall_ns: None,
                correctness: Correctness::Invalid {
                    reason: format!(
                        "insufficient samples due to timeout ({} < 30)",
                        actual_samples
                    ),
                },
                contract: None,
                performance: None,
                metrics: BTreeMap::new(),
                optimization_passes_applied: vec![],
                artifacts: vec![],
            });
        }

        if let Some(active_ns) = d_samples
            .get("dispatch_ns")
            .filter(|samples| !samples.is_empty())
            .or_else(|| d_samples.get("wall_ns"))
        {
            let mut sorted = active_ns.clone();
            sorted.sort_unstable();
            determinism_p50s.push(percentile(&sorted, 50.0));
        }
    }
    let cached_fingerprint = ctx
        .take_artifact_session()
        .map_err(|error| error.to_string())?;
    let program_fingerprint = case
        .workload_fingerprint_bytes(prepared)
        .or(cached_fingerprint);

    let correctness = correctness.ok_or_else(|| {
        "benchmark produced no samples; target sample count must be greater than zero".to_string()
    })?;
    if let Correctness::Invalid { reason } = correctness {
        return Err(reason);
    }
    if samples.get("wall_ns").is_none_or(Vec::is_empty) {
        return Err("benchmark produced no wall_ns samples".to_string());
    }

    let mut metrics = BTreeMap::new();
    for (name, values) in samples {
        if !values.is_empty() {
            metrics.insert(name.to_string(), compute_stats(&values));
        }
    }
    // surface the cold (first-warmup) sample as
    // synthetic-stat rows under `cold_*` keys. Stats are degenerate
    // (one sample → min == p50 == max) but they share the
    // MetricStats schema so downstream consumers (flamegraph
    // emitter, JSON report, sqlite writer) treat them uniformly.
    if let Some(cold_wall) = cold_wall_ns {
        metrics
            .entry("cold_wall_ns".to_string())
            .or_insert_with(|| single_sample_stats(cold_wall));
    }
    if let Some(cold) = cold_metrics.as_ref() {
        let cold_pairs: [(&str, Option<u64>); 6] = [
            ("cold_compile_ns", cold.compile_ns),
            ("cold_optimize_ns", cold.optimize_ns),
            ("cold_lower_ns", cold.lower_ns),
            ("cold_cache_lookup_ns", cold.cache_lookup_ns),
            ("cold_dispatch_ns", cold.dispatch_ns),
            ("cold_readback_ns", cold.readback_ns),
        ];
        for (key, value) in cold_pairs {
            if let Some(v) = value {
                metrics
                    .entry(key.to_string())
                    .or_insert_with(|| single_sample_stats(v));
            }
        }
    }
    for (name, value) in ctx.preferred_backend.backend_metric_snapshot() {
        metrics
            .entry(name.to_string())
            .or_insert_with(|| single_sample_stats(value));
    }
    normalize_release_evidence_metrics(&mut metrics, ctx.preferred_backend.id());
    normalize_benchmark_evidence_metrics(&mut metrics, cpu_digest, gpu_digest);

    let contract = case.performance_contract();
    let performance = contract
        .as_ref()
        .map(|contract| evaluate_contract(contract, &metrics, ctx.preferred_backend.id()));

    let mut status = "pass".to_string();
    if determinism_p50s.len() > 1 {
        let sum: u64 = determinism_p50s.iter().sum();
        let mean = sum as f64 / determinism_p50s.len() as f64;
        let variance = determinism_p50s
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / determinism_p50s.len() as f64;
        let stddev = variance.sqrt();
        let cv = stddev / mean;

        // Populate determinism_cv on the active metric
        let target_metric = if metrics.contains_key("kernel_execute_ns") {
            "kernel_execute_ns"
        } else if metrics.contains_key("dispatch_ns") {
            "dispatch_ns"
        } else {
            "wall_ns"
        };
        if let Some(stats) = metrics.get_mut(target_metric) {
            stats.determinism_cv = Some(cv);
        }

        if cv > 0.05 {
            status = "unstable".to_string(); // Variance > 5%
        }
    }
    let requirements = case.requirements();
    if thermal_status_applies(&metrics, requirements.needs_gpu) {
        status = "thermal_unstable".to_string();
    }
    status = final_case_status(&status, performance.as_ref());

    let wall_ns = metrics.get("wall_ns").map(|s| s.mean);

    let optimization_passes_applied =
        infer_optimization_passes_applied(&metrics, ctx.preferred_backend.id());
    let case_id = meta.id.0;
    let workload_fingerprint = workload_fingerprint(case_id.as_str(), program_fingerprint);
    Ok(CaseReport {
        id: case_id.clone(),
        workload_fingerprint: workload_fingerprint.clone(),
        name: meta.name,
        owner_crate: meta.owner_crate,
        workload_class: format!("{:?}", meta.workload),
        tags: meta.tags,
        backend_id: Some(ctx.preferred_backend.id().to_string()),
        device_signature: Some(benchmark_device_signature(
            ctx.preferred_backend.device_profile(),
        )),
        held_out_corpus_id: Some(benchmark_held_out_corpus_id(&workload_fingerprint)),
        needs_gpu: requirements.needs_gpu,
        min_vram_bytes: requirements.min_vram_bytes,
        min_input_bytes: requirements.min_input_bytes,
        required_features: requirements.feature_set,
        status,
        wall_ns,
        correctness,
        contract,
        performance,
        metrics,
        optimization_passes_applied,
        artifacts: vec![],
    })
}

/// State the case verdict.
///
/// WHY: a failed performance contract is a failed case whatever the producing
/// command asked for, so the status a reader tallies agrees with
/// `CaseReport::passes_summary_evidence` and with the printed pair.
fn final_case_status(provisional: &str, performance: Option<&PerformanceEvaluation>) -> String {
    if performance.is_some_and(|performance| !performance.contract_passed) {
        "failed".to_string()
    } else {
        provisional.to_string()
    }
}

fn thermal_status_applies(
    metrics: &BTreeMap<String, MetricStats>,
    workload_needs_gpu: bool,
) -> bool {
    workload_needs_gpu
        && metrics
            .get("thermal_unstable")
            .is_some_and(|stats| stats.max > 0)
}

fn normalize_benchmark_evidence_metrics(
    metrics: &mut BTreeMap<String, MetricStats>,
    cpu_digest: Option<u64>,
    gpu_digest: Option<u64>,
) {
    if let Some(cpu_digest) = cpu_digest {
        metrics
            .entry("cpu_digest".to_string())
            .or_insert_with(|| single_sample_stats(cpu_digest));
    }
    if let Some(gpu_digest) = gpu_digest {
        metrics
            .entry("gpu_digest".to_string())
            .or_insert_with(|| single_sample_stats(gpu_digest));
    }
    if let Some(active_time) = metrics
        .get("kernel_execute_ns")
        .or_else(|| metrics.get("dispatch_ns"))
        .or_else(|| metrics.get("wall_ns"))
        .cloned()
    {
        metrics
            .entry("active_time_ns".to_string())
            .or_insert(active_time);
    }
    if let (Some(input), Some(output)) = (
        metrics.get("host_to_device_bytes").cloned(),
        metrics.get("device_to_host_bytes").cloned(),
    ) {
        metrics
            .entry("transfer_bytes".to_string())
            .or_insert_with(|| sum_metric_stats(&input, &output));
    } else if let Some(bytes) = metrics
        .get("bytes_touched")
        .or_else(|| metrics.get("bytes_read"))
        .or_else(|| metrics.get("bytes_written"))
        .cloned()
    {
        metrics.entry("transfer_bytes".to_string()).or_insert(bytes);
    }
}

fn sum_metric_stats(left: &MetricStats, right: &MetricStats) -> MetricStats {
    MetricStats {
        min: left.min.saturating_add(right.min),
        p50: left.p50.saturating_add(right.p50),
        p90: left.p90.saturating_add(right.p90),
        p95: left.p95.saturating_add(right.p95),
        p99: left.p99.saturating_add(right.p99),
        p999: left.p999.saturating_add(right.p999),
        p9999: left.p9999.saturating_add(right.p9999),
        max: left.max.saturating_add(right.max),
        mean: left.mean + right.mean,
        stddev: (left
            .stddev
            .mul_add(left.stddev, right.stddev * right.stddev))
        .sqrt(),
        samples: left.samples.min(right.samples),
        determinism_cv: None,
    }
}

/// Produce a degenerate `MetricStats` for a single
/// observation. Used to surface the cold (first-warmup) sample
/// alongside the warm-batch stats without inventing a separate
/// schema. min == p50 == max, samples == 1, stddev == 0.
fn single_sample_stats(value: u64) -> MetricStats {
    MetricStats::single(value)
}

fn normalize_release_evidence_metrics(
    metrics: &mut BTreeMap<String, MetricStats>,
    backend_id: &str,
) {
    if backend_id == "cuda" {
        if let Some(input) = metrics
            .get("cuda_host_to_device_bytes")
            .filter(|stats| stats.max > 0)
            .cloned()
        {
            metrics.insert("host_to_device_bytes".to_string(), input);
        }
        if let Some(output) = metrics
            .get("cuda_device_to_host_bytes")
            .filter(|stats| stats.max > 0)
            .cloned()
        {
            metrics.insert("device_to_host_bytes".to_string(), output);
        }
    }
    if let Some(input) = metrics
        .get("input_bytes")
        .or_else(|| metrics.get("bytes_read"))
        .or_else(|| metrics.get("bytes_touched"))
        .cloned()
    {
        metrics
            .entry("host_to_device_bytes".to_string())
            .or_insert(input);
    }
    metrics
        .entry("host_to_device_bytes".to_string())
        .or_insert_with(|| single_sample_stats(0));
    if let Some(output) = metrics
        .get("output_bytes")
        .or_else(|| metrics.get("bytes_written"))
        .or_else(|| metrics.get("bytes_touched"))
        .cloned()
    {
        metrics
            .entry("device_to_host_bytes".to_string())
            .or_insert(output);
    }
    metrics
        .entry("device_to_host_bytes".to_string())
        .or_insert_with(|| single_sample_stats(0));
    if backend_id == "cuda" {
        if let Some(launches) = metrics
            .get("cuda_kernel_launches")
            .filter(|stats| stats.max > 0)
            .cloned()
        {
            metrics
                .entry("kernel_launches".to_string())
                .or_insert(launches);
        }
    }
    if backend_id != "cpu-ref" {
        metrics
            .entry("kernel_launches".to_string())
            .or_insert_with(|| single_sample_stats(1));
    }
}

fn infer_optimization_passes_applied(
    metrics: &BTreeMap<String, MetricStats>,
    backend_id: &str,
) -> Vec<String> {
    let mut passes = Vec::new();
    let metric_positive = |name: &str| metrics.get(name).is_some_and(|stats| stats.max > 0);
    if backend_id == "cuda" {
        passes.push("cuda-explicit-backend-selection".to_string());
    }
    if metrics.contains_key("cache_hit") || metrics.contains_key("cold_cache_lookup_ns") {
        passes.push("pipeline-cache-lookup".to_string());
    }
    if metric_positive("cuda_ptx_source_cache_entries")
        || metric_positive("cuda_ptx_source_cache_hits")
        || metric_positive("cuda_ptx_source_cache_misses")
    {
        passes.push("cuda-ptx-source-cache".to_string());
    }
    if metric_positive("cuda_graph_launches") {
        passes.push("cuda-graph-replay".to_string());
    }
    if metric_positive("cuda_graph_materialized_cache_hits") {
        passes.push("cuda-graph-materialized-output-cache".to_string());
    }
    if metric_positive("cuda_host_upload_operations")
        || metric_positive("cuda_device_readback_operations")
    {
        passes.push("cuda-transfer-operation-telemetry".to_string());
    }
    if metrics.contains_key("optimize_ns") || metrics.contains_key("cold_optimize_ns") {
        passes.push("optimizer-pipeline".to_string());
    }
    if metrics.contains_key("lower_ns") || metrics.contains_key("cold_lower_ns") {
        passes.push("backend-lowering".to_string());
    }
    if metrics
        .get("kernel_launches")
        .is_some_and(|stats| stats.max == 1)
    {
        passes.push("single-dispatch-launch-plan".to_string());
    } else if metrics
        .get("kernel_launches")
        .is_some_and(|stats| stats.max > 1)
    {
        passes.push("multi-dispatch-launch-plan".to_string());
    }
    if metrics.keys().any(|key| {
        key.starts_with("lower_") || key.starts_with("alias_") || key.starts_with("egraph_")
    }) {
        passes.push("measured-lower-optimization-family".to_string());
    }
    passes.sort();
    passes.dedup();
    passes
}

fn workload_fingerprint(case_id: &str, program_fingerprint: Option<[u8; 32]>) -> String {
    let Some(fingerprint) = program_fingerprint else {
        return format!("bench-case:{case_id}");
    };
    let mut encoded = String::with_capacity("program:".len() + 64);
    encoded.push_str("program:");
    for byte in fingerprint {
        use std::fmt::Write as _;
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

/// The device's own active time for one sample, in the order of preference the
/// backends report it.
///
/// `wall_ns` is the last resort rather than the first: it includes host
/// overhead and readback, so a case that reports device time is judged on
/// device time and only a case that reports none falls back to its wall clock.
fn device_active_time(metrics: &BTreeMap<String, MetricStats>) -> Option<&MetricStats> {
    metrics
        .get("dispatch_ns")
        .filter(|stats| stats.p50 > 0)
        .or_else(|| {
            metrics
                .get("kernel_execute_ns")
                .filter(|stats| stats.p50 > 0)
        })
        .or_else(|| metrics.get("wall_ns").filter(|stats| stats.p50 > 0))
}

pub(crate) fn evaluate_contract(
    contract: &PerformanceContract,
    metrics: &BTreeMap<String, MetricStats>,
    backend_id: &str,
) -> PerformanceEvaluation {
    let active_gpu = device_active_time(metrics);
    let speedup_x = match (active_gpu, metrics.get("baseline_wall_ns")) {
        (Some(gpu), Some(cpu)) => Some(cpu.p50 as f64 / gpu.p50 as f64),
        _ => None,
    };
    let mut violations = Vec::new();
    let mut applicable_baselines = 0usize;
    for baseline in &contract.baselines {
        if !baseline.backend_ids.is_empty()
            && !baseline
                .backend_ids
                .iter()
                .any(|candidate| candidate == backend_id)
        {
            continue;
        }
        applicable_baselines += 1;
        match speedup_x {
            Some(speedup) if speedup >= baseline.min_speedup_x => {}
            Some(speedup) => violations.push(format!(
                "{} requires {:.2}x over {}, observed {:.2}x",
                contract.primitive, baseline.min_speedup_x, baseline.name, speedup
            )),
            None => violations.push(format!(
                "{} requires a measured steady-state speedup over {}, but dispatch_ns/kernel_execute_ns/wall_ns or baseline_wall_ns were incomplete",
                contract.primitive, baseline.name
            )),
        }
    }
    let applicable_bounds = evaluate_device_bounds(contract, metrics, &mut violations);
    if applicable_baselines == 0 && applicable_bounds == 0 {
        violations.push(format!(
            "{} has no performance baseline or device bound that applies to backend `{backend_id}`",
            contract.primitive
        ));
    }
    PerformanceEvaluation {
        speedup_x,
        contract_passed: violations.is_empty(),
        violations,
    }
}

/// Judge every device bound the contract carries, appending one violation per
/// bound that is missed or cannot be judged.
///
/// Returns how many bounds were judged. A bound whose metric is absent counts
/// as judged and violated: a device bound that silently does not apply is a
/// contract that certifies nothing.
fn evaluate_device_bounds(
    contract: &PerformanceContract,
    metrics: &BTreeMap<String, MetricStats>,
    violations: &mut Vec<String>,
) -> usize {
    for bound in &contract.device_bounds {
        match bound {
            DeviceBound::MemoryBandwidthFractionOfPeak {
                min_fraction,
                derivation,
            } => {
                // `roofline_mem_pct_x1000` is a percentage scaled by 1000, so
                // 39174 is 39.174% of peak.
                match metrics.get("roofline_mem_pct_x1000") {
                    Some(stats) if stats.p50 > 0 => {
                        let observed = stats.p50 as f64 / 100_000.0;
                        if observed < *min_fraction {
                            violations.push(format!(
                                "{} requires at least {:.1}% of device memory bandwidth, observed {:.1}%. {derivation}",
                                contract.primitive,
                                min_fraction * 100.0,
                                observed * 100.0
                            ));
                        }
                    }
                    _ => violations.push(format!(
                        "{} requires a measured device memory bandwidth fraction, but `roofline_mem_pct_x1000` was absent: the case must state `device_bytes_moved` and the run must carry device memory-peak telemetry",
                        contract.primitive
                    )),
                }
            }
            DeviceBound::ActiveTimeCeilingNs {
                max_active_ns,
                derivation,
            } => match device_active_time(metrics) {
                Some(stats) => {
                    if stats.p50 > *max_active_ns {
                        violations.push(format!(
                            "{} requires device active time at or under {max_active_ns} ns, observed {} ns. {derivation}",
                            contract.primitive, stats.p50
                        ));
                    }
                }
                None => violations.push(format!(
                    "{} requires a measured device active time, but dispatch_ns/kernel_execute_ns/wall_ns were incomplete",
                    contract.primitive
                )),
            },
        }
    }
    contract.device_bounds.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(value: u64) -> MetricStats {
        single_sample_stats(value)
    }

    fn contract_for_backends(backends: &[&str], min_speedup_x: f64) -> PerformanceContract {
        PerformanceContract {
            primitive: "release workload".to_string(),
            baselines: vec![crate::api::case::BaselineTarget {
                name: "cpu sota".to_string(),
                crate_name: "vyre".to_string(),
                class: crate::api::case::BaselineClass::CpuSota,
                min_speedup_x,
                backend_ids: backends.iter().map(|backend| backend.to_string()).collect(),
            }],
            device_bounds: Vec::new(),
        }
    }

    fn bandwidth_contract(min_fraction: f64) -> PerformanceContract {
        PerformanceContract::memory_bandwidth_fraction(
            "bandwidth-bound workload",
            min_fraction,
            "measured 44.4% of peak; the gather floor is 74.9%",
        )
    }

    /// WHY: for a kernel already at the device's streaming ceiling, the
    /// property under test is how much of that bandwidth it uses. The bound
    /// must go red when the kernel stops being bandwidth-bound and must not
    /// move when the host baseline does, which is exactly what a CPU multiple
    /// on the same case could not do.
    ///
    /// Does not catch a wrong `device_bytes_moved`; that is the case's own
    /// accounting, asserted where the case builds it.
    #[test]
    fn a_memory_bandwidth_bound_is_judged_against_the_measured_fraction_of_peak() {
        let mut metrics = BTreeMap::new();
        metrics.insert("dispatch_ns".to_string(), stats(50_496));
        metrics.insert("roofline_mem_pct_x1000".to_string(), stats(44_431));

        let passing = evaluate_contract(&bandwidth_contract(0.30), &metrics, "cuda");
        assert!(
            passing.contract_passed,
            "Fix: 44.431% of peak must satisfy a 30% floor: {:?}",
            passing.violations
        );

        // The kernel loses coalescing and moves the same bytes 1.5x slower.
        let mut regressed = metrics.clone();
        regressed.insert("roofline_mem_pct_x1000".to_string(), stats(29_620));
        let failing = evaluate_contract(&bandwidth_contract(0.30), &regressed, "cuda");
        assert!(
            !failing.contract_passed,
            "Fix: 29.62% of peak must miss a 30% floor"
        );
        assert!(
            failing.violations[0].contains("29.6%")
                && failing.violations[0].contains("gather floor"),
            "Fix: the violation must state the observed fraction and the derivation: {:?}",
            failing.violations
        );

        // A host baseline moving by 100x cannot change the verdict.
        let mut with_slow_host = metrics.clone();
        with_slow_host.insert("baseline_wall_ns".to_string(), stats(50_496));
        assert!(
            evaluate_contract(&bandwidth_contract(0.30), &with_slow_host, "cuda").contract_passed,
            "Fix: a device bound must not read the host baseline."
        );
    }

    /// WHY: a bound whose metric is absent must fail, not silently pass. A
    /// resident case that states no `device_bytes_moved` publishes no
    /// bandwidth fraction, and a contract that certifies what it never
    /// measured is worse than no contract.
    #[test]
    fn a_device_bound_with_no_measurement_fails_closed() {
        let mut metrics = BTreeMap::new();
        metrics.insert("dispatch_ns".to_string(), stats(50_496));

        let evaluation = evaluate_contract(&bandwidth_contract(0.30), &metrics, "cuda");

        assert!(!evaluation.contract_passed);
        assert!(
            evaluation.violations[0].contains("roofline_mem_pct_x1000"),
            "Fix: the violation must name the missing metric: {:?}",
            evaluation.violations
        );
    }

    /// WHY: a latency case's quantity is nanoseconds, so its contract asserts
    /// nanoseconds. Against a host simulator faster than one launch, a ratio
    /// contract can never be met however fast the launch becomes.
    #[test]
    fn an_active_time_ceiling_is_judged_against_device_active_time() {
        let contract = PerformanceContract::active_time_ceiling_ns(
            "resident megakernel slot dispatch",
            5_000,
            "measured floor 2848 ns",
        );
        let mut metrics = BTreeMap::new();
        metrics.insert("dispatch_ns".to_string(), stats(3_904));
        // A host loop faster than a launch, which a ratio contract would read.
        metrics.insert("baseline_wall_ns".to_string(), stats(290));

        let passing = evaluate_contract(&contract, &metrics, "cuda");
        assert!(
            passing.contract_passed,
            "Fix: 3904 ns must satisfy a 5000 ns ceiling: {:?}",
            passing.violations
        );
        assert_eq!(
            passing.speedup_x,
            Some(290.0 / 3_904.0),
            "Fix: the measured ratio is still reported, it is just not asserted."
        );

        let mut regressed = metrics.clone();
        regressed.insert("dispatch_ns".to_string(), stats(6_000));
        let failing = evaluate_contract(&contract, &regressed, "cuda");
        assert!(!failing.contract_passed);
        assert!(
            failing.violations[0].contains("6000 ns")
                && failing.violations[0].contains("2848 ns"),
            "Fix: the violation must state the observed time and the derivation: {:?}",
            failing.violations
        );
    }

    /// WHY: the status a reader tallies in `cases[]` has to agree with
    /// `CaseReport::passes_summary_evidence`, which rejects a failed contract
    /// whatever the producing command asked for. A verdict that depended on a
    /// producer flag printed one pass count and recorded another.
    ///
    /// This does not catch a provisional status added without a decision: the
    /// provisional set is string literals at the call site, not a type.
    #[test]
    fn a_failed_performance_contract_is_a_failed_case_in_every_run() {
        let failed = PerformanceEvaluation {
            speedup_x: Some(99.0),
            contract_passed: false,
            violations: vec!["below contract".to_string()],
        };
        let passed = PerformanceEvaluation {
            speedup_x: Some(101.0),
            contract_passed: true,
            violations: Vec::new(),
        };

        for provisional in ["pass", "unstable", "thermal_unstable"] {
            assert_eq!(
                final_case_status(provisional, Some(&failed)),
                "failed",
                "Fix: a case that missed its performance contract must report status `failed` so the case list and the printed pass count state the same verdict."
            );
            assert_eq!(
                final_case_status(provisional, Some(&passed)),
                provisional,
                "Fix: a passing performance contract must preserve the provisional measurement status."
            );
            assert_eq!(
                final_case_status(provisional, None),
                provisional,
                "Fix: a case without a performance contract must preserve the provisional measurement status."
            );
        }
    }

    #[test]
    fn contract_fails_when_no_baseline_applies_to_backend() {
        let mut metrics = BTreeMap::new();
        metrics.insert("dispatch_ns".to_string(), stats(1));
        metrics.insert("baseline_wall_ns".to_string(), stats(1_000));

        let evaluation =
            evaluate_contract(&contract_for_backends(&["cuda"], 100.0), &metrics, "wgpu");

        assert_eq!(
            evaluation.speedup_x,
            Some(1_000.0),
            "Fix: speedup measurement should still be reported when the contract backend set is wrong."
        );
        assert!(
            !evaluation.contract_passed,
            "Fix: WGPU benchmark evidence must not pass by skipping a CUDA-only baseline."
        );
        assert!(
            evaluation
                .violations
                .iter()
                .any(|violation| violation.contains("no performance baseline")),
            "Fix: contract failures must explain that no baseline applies to the active backend."
        );
    }

    #[test]
    fn contract_with_empty_backend_ids_applies_to_every_backend() {
        let mut metrics = BTreeMap::new();
        metrics.insert("dispatch_ns".to_string(), stats(10));
        metrics.insert("baseline_wall_ns".to_string(), stats(1_000));

        let evaluation = evaluate_contract(&contract_for_backends(&[], 50.0), &metrics, "wgpu");

        assert!(
            evaluation.contract_passed,
            "Fix: backend-agnostic baselines must still apply to WGPU and other backends."
        );
        assert!(
            evaluation.violations.is_empty(),
            "Fix: backend-agnostic passing contracts must not accumulate baseline applicability violations."
        );
    }

    /// WHY: some CUDA event paths report a zero device duration while preserving a measured
    /// host wall duration. A zero dispatch sample is absence, not a zero-cost kernel, and must
    /// not hide the bounded wall-clock fallback used by the performance contract.
    #[test]
    fn contract_uses_wall_time_when_dispatch_duration_is_zero() {
        let mut metrics = BTreeMap::new();
        metrics.insert("dispatch_ns".to_string(), stats(0));
        metrics.insert("wall_ns".to_string(), stats(10));
        metrics.insert("baseline_wall_ns".to_string(), stats(1_000));

        let evaluation =
            evaluate_contract(&contract_for_backends(&["cuda"], 100.0), &metrics, "cuda");

        assert_eq!(evaluation.speedup_x, Some(100.0));
        assert!(evaluation.contract_passed, "{:?}", evaluation.violations);
    }

    #[test]
    fn cuda_graph_backend_metrics_are_reported_as_release_path_passes() {
        let mut metrics = BTreeMap::new();
        metrics.insert("cuda_ptx_source_cache_misses".to_string(), stats(1));
        metrics.insert("cuda_graph_launches".to_string(), stats(3));
        metrics.insert("cuda_graph_materialized_cache_hits".to_string(), stats(2));
        metrics.insert("cuda_host_upload_operations".to_string(), stats(4));
        metrics.insert("cuda_device_readback_operations".to_string(), stats(1));

        let passes = infer_optimization_passes_applied(&metrics, "cuda");

        for expected in [
            "cuda-explicit-backend-selection",
            "cuda-graph-replay",
            "cuda-graph-materialized-output-cache",
            "cuda-ptx-source-cache",
            "cuda-transfer-operation-telemetry",
        ] {
            assert!(
                passes.iter().any(|pass| pass == expected),
                "Fix: CUDA benchmark reports must label `{expected}` when backend telemetry exposes the release-path metric."
            );
        }
    }

    #[test]
    fn cuda_graph_backend_passes_require_positive_release_path_counters() {
        let mut metrics = BTreeMap::new();
        metrics.insert("cuda_ptx_source_cache_hits".to_string(), stats(0));
        metrics.insert("cuda_graph_launches".to_string(), stats(0));
        metrics.insert("cuda_graph_materialized_cache_hits".to_string(), stats(0));
        metrics.insert("cuda_host_upload_operations".to_string(), stats(0));
        metrics.insert("cuda_device_readback_operations".to_string(), stats(0));

        let passes = infer_optimization_passes_applied(&metrics, "cuda");

        for absent in [
            "cuda-graph-replay",
            "cuda-graph-materialized-output-cache",
            "cuda-ptx-source-cache",
            "cuda-transfer-operation-telemetry",
        ] {
            assert!(
                !passes.iter().any(|pass| pass == absent),
                "Fix: CUDA benchmark reports must not label `{absent}` when telemetry exposes only zero observations."
            );
        }
        assert!(
            passes
                .iter()
                .any(|pass| pass == "cuda-explicit-backend-selection"),
            "Fix: explicit CUDA backend selection is independent of per-counter activity."
        );
    }

    #[test]
    fn release_metrics_use_cuda_launch_counter_before_single_launch_fallback() {
        let mut metrics = BTreeMap::new();
        metrics.insert("cuda_kernel_launches".to_string(), stats(4));

        normalize_release_evidence_metrics(&mut metrics, "cuda");

        let launch_stats = metrics
            .get("kernel_launches")
            .expect("Fix: CUDA release reports must expose canonical kernel_launches.");
        assert_eq!(
            launch_stats.p50, 4,
            "Fix: canonical kernel_launches must preserve CUDA telemetry instead of reporting the synthetic single-launch fallback."
        );
    }

    /// WHY: artifact submissions may bypass the lower-level CUDA telemetry
    /// object. A zero observation is unavailable telemetry, while a successful
    /// measured GPU sample proves that at least one kernel was launched.
    #[test]
    fn zero_cuda_launch_counter_uses_single_submission_fallback() {
        let mut metrics = BTreeMap::new();
        metrics.insert("cuda_kernel_launches".to_string(), stats(0));

        normalize_release_evidence_metrics(&mut metrics, "cuda");

        assert_eq!(metrics["kernel_launches"].p50, 1);
    }

    /// WHY: an explicit zero identifies compiler-only evidence. It is a real
    /// observation, unlike a zero backend counter that means telemetry is absent.
    #[test]
    fn explicit_zero_launch_metric_bypasses_single_submission_fallback() {
        let mut metrics = BTreeMap::new();
        metrics.insert("kernel_launches".to_string(), stats(0));

        normalize_release_evidence_metrics(&mut metrics, "cuda");

        assert_eq!(metrics["kernel_launches"].p50, 0);
    }

    #[test]
    fn release_metrics_keep_single_launch_fallback_when_backend_has_no_counter() {
        let mut metrics = BTreeMap::new();

        normalize_release_evidence_metrics(&mut metrics, "wgpu");

        let launch_stats = metrics.get("kernel_launches").expect(
            "Fix: non-CPU release reports without backend counters still need launch evidence.",
        );
        assert_eq!(
            launch_stats.p50, 1,
            "Fix: launch fallback must remain for backends that do not expose a backend-specific launch counter."
        );
    }

    #[test]
    fn launch_plan_labels_match_measured_kernel_launch_count() {
        let mut single = BTreeMap::new();
        single.insert("kernel_launches".to_string(), stats(1));

        let single_passes = infer_optimization_passes_applied(&single, "wgpu");
        assert!(
            single_passes
                .iter()
                .any(|pass| pass == "single-dispatch-launch-plan"),
            "Fix: one measured kernel launch must keep the single-dispatch launch-plan label."
        );
        assert!(
            !single_passes
                .iter()
                .any(|pass| pass == "multi-dispatch-launch-plan"),
            "Fix: one measured kernel launch must not be reported as a multi-dispatch plan."
        );

        let mut multi = BTreeMap::new();
        multi.insert("kernel_launches".to_string(), stats(4));

        let multi_passes = infer_optimization_passes_applied(&multi, "cuda");
        assert!(
            multi_passes
                .iter()
                .any(|pass| pass == "multi-dispatch-launch-plan"),
            "Fix: more than one measured kernel launch must be labeled as a multi-dispatch launch plan."
        );
        assert!(
            !multi_passes
                .iter()
                .any(|pass| pass == "single-dispatch-launch-plan"),
            "Fix: multi-launch CUDA evidence must not claim the single-dispatch launch-plan label."
        );
    }

    #[test]
    fn release_metrics_use_cuda_transfer_counters_before_logical_byte_fallbacks() {
        let mut metrics = BTreeMap::new();
        metrics.insert("bytes_read".to_string(), stats(12));
        metrics.insert("bytes_written".to_string(), stats(4));
        metrics.insert("cuda_host_to_device_bytes".to_string(), stats(48));
        metrics.insert("cuda_device_to_host_bytes".to_string(), stats(16));

        normalize_release_evidence_metrics(&mut metrics, "cuda");

        let host_to_device = metrics
            .get("host_to_device_bytes")
            .expect("Fix: CUDA release reports must expose canonical host_to_device_bytes.");
        assert_eq!(
            host_to_device.p50, 48,
            "Fix: canonical host_to_device_bytes must preserve CUDA transfer telemetry instead of logical input bytes."
        );
        let device_to_host = metrics
            .get("device_to_host_bytes")
            .expect("Fix: CUDA release reports must expose canonical device_to_host_bytes.");
        assert_eq!(
            device_to_host.p50, 16,
            "Fix: canonical device_to_host_bytes must preserve CUDA transfer telemetry instead of logical output bytes."
        );
    }

    /// WHY: artifact materializers may not expose backend telemetry through the
    /// lower-level dispatch object. Zero counters must not erase measured byte
    /// accounting and produce a false missing-transfer release blocker.
    #[test]
    fn zero_cuda_transfer_counters_fall_back_to_measured_byte_accounting() {
        let mut metrics = BTreeMap::new();
        metrics.insert("bytes_read".to_string(), stats(12));
        metrics.insert("bytes_written".to_string(), stats(4));
        metrics.insert("cuda_host_to_device_bytes".to_string(), stats(0));
        metrics.insert("cuda_device_to_host_bytes".to_string(), stats(0));

        normalize_release_evidence_metrics(&mut metrics, "cuda");
        normalize_benchmark_evidence_metrics(&mut metrics, None, None);

        assert_eq!(metrics["host_to_device_bytes"].p50, 12);
        assert_eq!(metrics["device_to_host_bytes"].p50, 4);
        assert_eq!(metrics["transfer_bytes"].p50, 16);
    }

    #[test]
    fn release_metrics_keep_logical_transfer_fallback_when_backend_has_no_transfer_counter() {
        let mut metrics = BTreeMap::new();
        metrics.insert("bytes_read".to_string(), stats(12));
        metrics.insert("bytes_written".to_string(), stats(4));

        normalize_release_evidence_metrics(&mut metrics, "wgpu");

        let host_to_device = metrics
            .get("host_to_device_bytes")
            .expect("Fix: non-CPU release reports still need host_to_device_bytes.");
        assert_eq!(
            host_to_device.p50, 12,
            "Fix: logical input-byte fallback must remain for backends without transfer telemetry."
        );
        let device_to_host = metrics
            .get("device_to_host_bytes")
            .expect("Fix: non-CPU release reports still need device_to_host_bytes.");
        assert_eq!(
            device_to_host.p50, 4,
            "Fix: logical output-byte fallback must remain for backends without transfer telemetry."
        );
    }

    /// CPU-only optimizer benchmarks ignore idle GPU clocks even under a CUDA release run.
    #[test]
    fn cpu_only_workloads_ignore_gpu_thermal_status() {
        let metrics = BTreeMap::from([("thermal_unstable".to_string(), stats(1))]);

        assert!(!thermal_status_applies(&metrics, false));
    }

    /// GPU workloads retain the fail-closed thermal stability gate.
    #[test]
    fn gpu_workloads_preserve_thermal_status() {
        let metrics = BTreeMap::from([("thermal_unstable".to_string(), stats(1))]);

        assert!(thermal_status_applies(&metrics, true));
    }

    /// Stable or absent GPU telemetry never produces a thermal failure.
    #[test]
    fn stable_and_absent_gpu_telemetry_passes() {
        assert!(!thermal_status_applies(&BTreeMap::new(), true));
        assert!(!thermal_status_applies(
            &BTreeMap::from([("thermal_unstable".to_string(), stats(0))]),
            true,
        ));
    }
}
