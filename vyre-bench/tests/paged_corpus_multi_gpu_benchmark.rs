//! Physical multi-device speed and parity proof for paged corpus sharding.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::json;
use vyre::scan::{
    scan_paged_fused, scan_paged_fused_on, scan_sharded_fused_weighted_timed, GpuLiteralSet,
    PagedScanResult, ScanTarget, ShardedScanTiming,
};
use vyre_bench::probes::{capture_git_info_at, source_fingerprint, source_tree_fingerprint_at};
use vyre_driver_wgpu::engine::multi_gpu::MultiGpuExecutor;
use vyre_driver_wgpu::WgpuBackend;

const FILE_COUNT: usize = 96;
const FILE_BYTES: usize = 1024 * 1024;
const SAMPLE_COUNT: usize = 11;
const MAX_MATCHES: u32 = 4096;

fn corpus() -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let patterns: Vec<Vec<u8>> = (0..32)
        .map(|index| format!("VYRE_MULTI_GPU_TOKEN_{index:02}_END").into_bytes())
        .collect();
    let mut files = Vec::with_capacity(FILE_COUNT);
    for file_index in 0..FILE_COUNT {
        let mut file = vec![b'x'; FILE_BYTES];
        let pattern = &patterns[file_index % patterns.len()];
        let offset = 4096 + (file_index * 7919) % (FILE_BYTES - pattern.len() - 4096);
        file[offset..offset + pattern.len()].copy_from_slice(pattern);
        files.push(file);
    }
    (patterns, files)
}

fn elapsed_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

fn median(values: &[u64]) -> u64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

fn run_single(
    matcher: &GpuLiteralSet,
    target: &ScanTarget,
    files: &[&[u8]],
) -> (PagedScanResult, u64) {
    let start = Instant::now();
    let result = scan_paged_fused_on(matcher, target, files, FILE_BYTES, MAX_MATCHES)
        .expect("single-device paged scan must complete");
    (result, elapsed_ns(start.elapsed()))
}

fn run_multi(
    matcher: &GpuLiteralSet,
    targets: &[ScanTarget],
    weights: &[u32],
    files: &[&[u8]],
) -> (PagedScanResult, ShardedScanTiming, u64) {
    let start = Instant::now();
    let (result, timing) = scan_sharded_fused_weighted_timed(
        matcher,
        targets,
        weights,
        files,
        FILE_BYTES,
        MAX_MATCHES,
    )
    .expect("weighted physical multi-device scan must complete");
    (result, timing, elapsed_ns(start.elapsed()))
}

/// Requires two distinct physical adapters and proves exact parity plus significant speedup.
#[test]
#[ignore = "requires at least two physical WGPU adapters and records benchmark evidence"]
fn two_physical_gpus_preserve_results_and_accelerate_the_same_corpus() {
    let mut live = MultiGpuExecutor::enumerate_live_gpus();
    live.sort_by_key(|gpu| gpu.adapter_index);
    live.dedup_by(|left, right| {
        left.info.vendor == right.info.vendor
            && left.info.device == right.info.device
            && left.info.name == right.info.name
    });
    assert!(
        live.len() >= 2,
        "physical multi-device benchmark requires at least two distinct GPU adapters; found {live:?}"
    );

    let registration =
        vyre_driver::backend::backend_registration("wgpu").expect("registered WGPU compiler");
    let targets: Vec<ScanTarget> = live
        .iter()
        .map(|gpu| {
            let backend = WgpuBackend::acquire_adapter(gpu.adapter_index).unwrap_or_else(|error| {
                panic!(
                    "adapter {} ({}) failed acquisition: {error}",
                    gpu.adapter_index, gpu.info.name
                )
            });
            let materializer = Arc::from(
                vyre_driver_wgpu::artifact_materializer(backend)
                    .expect("selected adapter artifact materializer"),
            );
            ScanTarget::with_materializer(registration, materializer)
        })
        .collect();

    let (pattern_storage, file_storage) = corpus();
    let pattern_refs: Vec<&[u8]> = pattern_storage.iter().map(Vec::as_slice).collect();
    let file_refs: Vec<&[u8]> = file_storage.iter().map(Vec::as_slice).collect();
    let matcher = GpuLiteralSet::compile(&pattern_refs);

    let mut calibration_ns = Vec::with_capacity(targets.len());
    let mut single_results = Vec::with_capacity(targets.len());
    for target in &targets {
        let _ = run_single(&matcher, target, &file_refs);
        let mut samples = Vec::with_capacity(3);
        let mut last = None;
        for _ in 0..3 {
            let (result, ns) = run_single(&matcher, target, &file_refs);
            samples.push(ns);
            last = Some(result);
        }
        calibration_ns.push(median(&samples));
        single_results.push(last.unwrap());
    }
    let fastest_index = calibration_ns
        .iter()
        .enumerate()
        .min_by_key(|(_, ns)| *ns)
        .map(|(index, _)| index)
        .unwrap();
    let fastest_ns = calibration_ns[fastest_index];
    let weights: Vec<u32> = calibration_ns
        .iter()
        .map(|&ns| {
            ((u128::from(fastest_ns) * 10_000 / u128::from(ns.max(1)))
                .clamp(1, u128::from(u32::MAX))) as u32
        })
        .collect();

    let expected = &single_results[fastest_index];
    let (warm_multi, _, _) = run_multi(&matcher, &targets, &weights, &file_refs);
    assert_eq!(
        &warm_multi, expected,
        "multi-device warmup must match the fastest single GPU"
    );

    let mut single_samples = Vec::with_capacity(SAMPLE_COUNT);
    let mut multi_samples = Vec::with_capacity(SAMPLE_COUNT);
    let mut final_timing = None;
    for sample in 0..SAMPLE_COUNT {
        if sample % 2 == 0 {
            let (single, single_ns) = run_single(&matcher, &targets[fastest_index], &file_refs);
            let (multi, timing, multi_ns) = run_multi(&matcher, &targets, &weights, &file_refs);
            assert_eq!(single, *expected);
            assert_eq!(multi, *expected);
            single_samples.push(single_ns);
            multi_samples.push(multi_ns);
            final_timing = Some(timing);
        } else {
            let (multi, timing, multi_ns) = run_multi(&matcher, &targets, &weights, &file_refs);
            let (single, single_ns) = run_single(&matcher, &targets[fastest_index], &file_refs);
            assert_eq!(multi, *expected);
            assert_eq!(single, *expected);
            single_samples.push(single_ns);
            multi_samples.push(multi_ns);
            final_timing = Some(timing);
        }
    }

    let timing = final_timing.unwrap();
    assert_eq!(timing.shards.len(), targets.len());
    assert!(
        timing
            .shards
            .iter()
            .all(|shard| shard.windows > 0 && shard.bytes_scanned > 0),
        "every physical device must receive real byte-work: {:?}",
        timing.shards
    );
    let single_p50 = median(&single_samples);
    let multi_p50 = median(&multi_samples);
    let wins = single_samples
        .iter()
        .zip(&multi_samples)
        .filter(|(single, multi)| single > multi)
        .count();
    let speedup = single_p50 as f64 / multi_p50.max(1) as f64;

    let adapters: Vec<_> = live
        .iter()
        .map(|gpu| {
            json!({
                "adapter_index": gpu.adapter_index,
                "name": gpu.info.name,
                "vendor": gpu.info.vendor,
                "device": gpu.info.device,
                "device_type": format!("{:?}", gpu.info.device_type),
                "backend": format!("{:?}", gpu.info.backend),
                "driver": gpu.info.driver,
                "driver_info": gpu.info.driver_info,
            })
        })
        .collect();
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("vyre-libs must live under the workspace root");
    let git = capture_git_info_at(workspace_root);
    let generated_unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after the Unix epoch")
        .as_secs();
    let shards: Vec<_> = timing
        .shards
        .iter()
        .map(|shard| {
            json!({
                "shard": shard.shard,
                "windows": shard.windows,
                "bytes_scanned": shard.bytes_scanned,
                "wall_ns": shard.wall_ns,
                "device_ns": shard.device_ns,
                "host_overhead_ns": shard
                    .device_ns
                    .map(|device_ns| shard.wall_ns.saturating_sub(device_ns)),
            })
        })
        .collect();
    let sign_test_p_upper = (0..=SAMPLE_COUNT - wins)
        .map(|k| binomial(SAMPLE_COUNT, k))
        .sum::<u64>() as f64
        / (1u64 << SAMPLE_COUNT) as f64;
    let evidence = json!({
        "schema": "vyre-bench.multi-device.v1",
        "benchmark_id": "libs.scan.paged_corpus.multi_gpu",
        "generated_unix_seconds": generated_unix_seconds,
        "source_fingerprint": source_fingerprint(&git),
        "source_tree_fingerprint": source_tree_fingerprint_at(workspace_root),
        "topology": {
            "host": "single Linux host",
            "dispatch": "one OS thread and one resident session per physical adapter",
            "aggregation": "host globalization plus deterministic stable sort",
            "peer_transfer": "none; each shard stages its own corpus windows",
        },
        "adapters": adapters,
        "corpus": {
            "files": FILE_COUNT,
            "bytes": FILE_COUNT * FILE_BYTES,
            "window_budget_bytes": FILE_BYTES,
            "patterns": pattern_refs.len(),
        },
        "fastest_single_adapter": fastest_index,
        "calibration_ns_p50": calibration_ns,
        "weights": weights,
        "single_samples_ns": single_samples,
        "multi_samples_ns": multi_samples,
        "single_ns_p50": single_p50,
        "multi_ns_p50": multi_p50,
        "speedup": speedup,
        "paired_speedup_wins": wins,
        "paired_samples": SAMPLE_COUNT,
        "one_sided_sign_test_p_upper": sign_test_p_upper,
        "shards": shards,
        "parity": "exact",
    });
    eprintln!("MULTI_GPU_EVIDENCE_JSON={evidence}");

    assert!(
        speedup > 1.0,
        "two physical GPUs must beat the fastest single GPU at p50; evidence={evidence}"
    );
    assert!(
        wins >= 9,
        "at least 9/11 paired samples must favor multi-device execution (one-sided sign test p<0.05); evidence={evidence}"
    );
    if let Ok(path) = std::env::var("VYRE_MULTI_GPU_EVIDENCE_PATH") {
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&evidence).expect("serialize benchmark evidence"),
        )
        .unwrap_or_else(|error| panic!("write multi-GPU evidence to {path}: {error}"));
    }
}

fn binomial(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    (0..k).fold(1u64, |value, index| {
        value * (n - index) as u64 / (index + 1) as u64
    })
}

fn registered_targets(count: usize) -> Vec<ScanTarget> {
    (0..count)
        .map(|_| ScanTarget::registered("wgpu").expect("registered WGPU scan target"))
        .collect()
}

fn small_fixture() -> (GpuLiteralSet, Vec<Vec<u8>>) {
    (
        GpuLiteralSet::compile(&[b"abc", b"secret"]),
        vec![
            b"abc---".to_vec(),
            b"nothing".to_vec(),
            b"secret".to_vec(),
            b"abc-secret".to_vec(),
        ],
    )
}

/// Proves the weighted timed API preserves exact output and accounts every shard.
#[test]
fn weighted_timed_scan_reports_real_work_per_shard() {
    let devices = registered_targets(2);
    let (matcher, files) = small_fixture();
    let file_refs: Vec<&[u8]> = files.iter().map(Vec::as_slice).collect();
    let expected = scan_paged_fused(&matcher, "wgpu", &file_refs, 8, 64).unwrap();
    let (actual, timing) =
        scan_sharded_fused_weighted_timed(&matcher, &devices, &[2, 1], &file_refs, 8, 64).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(timing.shards.len(), 2);
    assert!(timing.shards.iter().all(|shard| shard.windows > 0));
    assert_eq!(
        timing
            .shards
            .iter()
            .map(|shard| shard.bytes_scanned)
            .sum::<u64>(),
        files.iter().map(Vec::len).sum::<usize>() as u64
    );
}

/// Prevents a missing device weight from silently changing shard assignment.
#[test]
fn weighted_timed_scan_rejects_weight_count_mismatch() {
    let devices = registered_targets(2);
    let (matcher, files) = small_fixture();
    let file_refs: Vec<&[u8]> = files.iter().map(Vec::as_slice).collect();
    let error =
        scan_sharded_fused_weighted_timed(&matcher, &devices, &[1], &file_refs, 8, 64).unwrap_err();
    assert!(error.to_string().contains("1 weights for 2 targets"));
}

/// Keeps empty-corpus timing explicit for every configured device.
#[test]
fn weighted_timed_empty_corpus_reports_zero_work_for_each_shard() {
    let devices = registered_targets(2);
    let matcher = GpuLiteralSet::compile(&[b"abc"]);
    let (result, timing) =
        scan_sharded_fused_weighted_timed(&matcher, &devices, &[1, 1], &[], 8, 64).unwrap();
    assert_eq!(result.region_count, 0);
    assert!(result.matches.is_empty());
    assert_eq!(timing.shards.len(), 2);
    assert!(timing.shards.iter().all(|shard| {
        shard.windows == 0
            && shard.bytes_scanned == 0
            && shard.wall_ns == 0
            && shard.device_ns == Some(0)
    }));
}

/// Locks the exact one-sided sign-test threshold used by the benchmark gate.
#[test]
fn nine_of_eleven_paired_wins_is_significant() {
    let tail = (9..=11).map(|wins| binomial(11, wins)).sum::<u64>();
    assert_eq!(tail, 67);
    assert!(tail as f64 / 2048.0 < 0.05);
}
