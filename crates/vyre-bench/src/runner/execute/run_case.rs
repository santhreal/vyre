//! Per-case execution driver. Calls the case `prepare`, runs the
//! measured iterations, harvests metrics, and evaluates the
//! performance contract.

use std::collections::BTreeMap;
use std::time::Instant;

use crate::api::case::{BenchContext, Correctness};
use crate::api::metric::{digest64_buffers, elapsed_ns};
use crate::api::suite::SuiteKind;
use crate::report::json::{benchmark_device_signature, benchmark_held_out_corpus_id, CaseReport};

use super::collect::{collect_samples, derive_roofline_fractions};
use super::contract_eval::{evaluate_contract, final_case_status};
use super::metric_normalize::{
    infer_optimization_passes_applied, normalize_benchmark_evidence_metrics,
    normalize_release_evidence_metrics, single_sample_stats, thermal_status_applies,
    workload_fingerprint,
};
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

    derive_roofline_fractions(&mut samples);

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
