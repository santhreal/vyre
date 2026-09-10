//! Sample collection helpers used by `run_case` to harvest metric data
//! from `BenchMetrics` after each measured iteration.

use std::collections::BTreeMap;

use crate::api::case::BenchRun;

use super::metric_keys::{
    custom_metric_key, custom_metric_value, derived_metric_key, metric_key,
    rate_per_second_x1000,
};

pub(super) fn collect_samples(
    run_result: &BenchRun,
    samples: &mut BTreeMap<&'static str, Vec<u64>>,
    collect_baseline: bool,
) {
    collect_metric_fields("", &run_result.metrics, samples);
    collect_custom_metrics("", &run_result.metrics, samples);
    collect_gpu_counters("", &run_result.metrics, samples);
    collect_derived_metrics("", &run_result.metrics, samples);
    if collect_baseline {
        if let Some(baseline) = &run_result.baseline_metrics {
            collect_metric_fields("baseline_", baseline, samples);
            collect_custom_metrics("baseline_", baseline, samples);
            collect_gpu_counters("baseline_", baseline, samples);
            collect_derived_metrics("baseline_", baseline, samples);
        }
    }
}

pub(super) fn collect_metric_fields(
    prefix: &'static str,
    metrics: &crate::api::metric::BenchMetrics,
    samples: &mut BTreeMap<&'static str, Vec<u64>>,
) {
    // The cold_* fields of BenchMetrics are never populated by case::run(), they
    // are filled by run_case.rs directly from the first warmup wall-clock and the
    // stored cold_metrics snapshot, bypassing collect_metric_fields entirely and
    // inserting single-sample MetricStats rows into the final BTreeMap.  Entries
    // here for cold_* names would be permanently inert: metric_key() has no
    // cold_* arms (returns None), so the inner `if let (Some(_), Some(_))` guard
    // would never fire.  Keeping them misleads anyone adding a new cold_* metric
    // into thinking this table is the right place.
    #[allow(clippy::type_complexity)]
    const FIELDS: [(&str, fn(&crate::api::metric::BenchMetrics) -> Option<u64>); 25] = [
        ("wall_ns", |m| m.wall_ns),
        ("cpu_ns", |m| m.cpu_ns),
        ("compile_ns", |m| m.compile_ns),
        ("validate_ns", |m| m.validate_ns),
        ("optimize_ns", |m| m.optimize_ns),
        ("lower_ns", |m| m.lower_ns),
        ("cache_lookup_ns", |m| m.cache_lookup_ns),
        ("cache_hit", |m| m.cache_hit.map(|b| if b { 1 } else { 0 })),
        ("upload_ns", |m| m.upload_ns),
        ("dispatch_ns", |m| m.dispatch_ns),
        ("kernel_queue_submit_ns", |m| m.kernel_queue_submit_ns),
        ("kernel_execute_ns", |m| m.kernel_execute_ns),
        ("device_sync_ns", |m| m.device_sync_ns),
        ("readback_ns", |m| m.readback_ns),
        ("verify_ns", |m| m.verify_ns),
        ("alloc_count", |m| m.alloc_count),
        ("alloc_bytes", |m| m.alloc_bytes),
        ("peak_rss_bytes", |m| m.peak_rss_bytes),
        ("input_bytes", |m| m.input_bytes),
        ("output_bytes", |m| m.output_bytes),
        ("bytes_touched", |m| m.bytes_touched),
        ("bytes_read", |m| m.bytes_read),
        ("bytes_written", |m| m.bytes_written),
        ("atomic_op_count", |m| m.atomic_op_count),
        ("wire_bytes", |m| m.wire_bytes),
    ];
    for (name, getter) in FIELDS {
        if let (Some(value), Some(key)) = (getter(metrics), metric_key(prefix, name)) {
            samples.entry(key).or_default().push(value);
        }
    }
}

pub(super) fn collect_custom_metrics(
    prefix: &'static str,
    metrics: &crate::api::metric::BenchMetrics,
    samples: &mut BTreeMap<&'static str, Vec<u64>>,
) {
    for point in &metrics.custom {
        if let Some(key) = custom_metric_key(prefix, point.name.as_str()) {
            samples.entry(key).or_default().push(point.value);
        }
    }
}

pub(super) fn collect_gpu_counters(
    prefix: &'static str,
    metrics: &crate::api::metric::BenchMetrics,
    samples: &mut BTreeMap<&'static str, Vec<u64>>,
) {
    for counter in &metrics.gpu_counter {
        // use custom_metric_key to leak the names into the standard space safely
        if let Some(key) = custom_metric_key(prefix, counter.name.as_str()) {
            samples.entry(key).or_default().push(counter.value);
        }
    }
}

pub(super) fn collect_derived_metrics(
    prefix: &'static str,
    metrics: &crate::api::metric::BenchMetrics,
    samples: &mut BTreeMap<&'static str, Vec<u64>>,
) {
    let host_bytes = metrics.bytes_touched.unwrap_or_else(|| {
        metrics
            .input_bytes
            .unwrap_or(0)
            .saturating_add(metrics.output_bytes.unwrap_or(0))
    });
    // device_gb_s_x1000 states a device bandwidth, so it is computed from the
    // bytes the device moves. A case that states `device_bytes_moved` is taken
    // at its word. Otherwise the host transfer total stands in, which is exact
    // for a transfer case whose host traffic IS its device traffic, and the
    // metric is omitted entirely when neither is set rather than substituting
    // `host_bytes`: that substitution reported a device bandwidth computed from
    // host I/O.
    let device_bytes = metrics.device_bytes_moved.filter(|bytes| *bytes > 0).or_else(
        || match (metrics.bytes_read, metrics.bytes_written) {
            (Some(r), Some(w)) => {
                let total = r.saturating_add(w);
                if total > 0 {
                    Some(total)
                } else {
                    None
                }
            }
            (Some(r), None) if r > 0 => Some(r),
            (None, Some(w)) if w > 0 => Some(w),
            _ => None,
        },
    );

    if let Some(wall_ns) = metrics.wall_ns.filter(|ns| *ns > 0) {
        if host_bytes > 0 {
            if let Some(key) = derived_metric_key(prefix, "wall_gb_s_x1000") {
                samples.entry(key).or_default().push(rate_per_second_x1000(
                    host_bytes,
                    wall_ns,
                    1_000_000_000,
                ));
            }
        }
    }

    // The device's own active time, which both device-side rates below are
    // computed against. A roofline fraction states how much of the device's
    // bandwidth the kernel used while it was running, so wall time is the wrong
    // denominator: it includes readback and host overhead, and on a resident
    // case whose sample wall time is ten times its kernel time it reported a
    // bandwidth-bound kernel as using 0.4% of the device.
    let device_ns = metrics.dispatch_ns.or(metrics.wall_ns).filter(|ns| *ns > 0);
    if let (Some(dev_bytes), Some(device_ns)) = (device_bytes, device_ns) {
        if let Some(key) = derived_metric_key(prefix, "device_gb_s_x1000") {
            samples.entry(key).or_default().push(rate_per_second_x1000(
                dev_bytes,
                device_ns,
                1_000_000_000,
            ));
        }
    }

    if let Some(flop_count) = custom_metric_value(metrics, "flop_count") {
        if let Some(active_ns) = device_ns {
            if let Some(key) = derived_metric_key(prefix, "gflops_x1000") {
                samples.entry(key).or_default().push(rate_per_second_x1000(
                    flop_count,
                    active_ns,
                    1_000_000_000,
                ));
            }
        }
    }
}

/// Derive the roofline memory fraction from the whole achieved-rate series.
///
/// The device memory peak is a device property, and NVML telemetry is captured
/// on one sample per run, so a per-sample derivation produced exactly one
/// roofline sample: the one paired with that telemetry. When that sample was a
/// scheduling outlier, the fraction stated a small percentage of the device
/// while the achieved-rate p50 over the same 200 samples stated 39% of peak,
/// and a bandwidth contract evaluated against it flipped between runs on one
/// binary. The fraction is computed once per achieved-rate sample against the
/// run's peak instead, so its p50 is the p50 of the rate.
///
/// Does not catch a wrong `memory_peak_gb_s_x1000`; that is the device
/// telemetry's own figure.
pub(super) fn derive_roofline_fractions(samples: &mut BTreeMap<&'static str, Vec<u64>>) {
    let Some(peak_gb_s_x1000) = samples
        .get("memory_peak_gb_s_x1000")
        .and_then(|values| values.iter().copied().find(|value| *value > 0))
    else {
        return;
    };
    let Some(fractions) = samples.get("device_gb_s_x1000").map(|rates| {
        rates
            .iter()
            .map(|achieved_gb_s_x1000| {
                ((u128::from(*achieved_gb_s_x1000) * 100_000) / u128::from(peak_gb_s_x1000))
                    .min(u128::from(u64::MAX)) as u64
            })
            .collect::<Vec<u64>>()
    }) else {
        return;
    };
    if fractions.is_empty() {
        return;
    }
    if let Some(key) = derived_metric_key("", "roofline_mem_pct_x1000") {
        samples.insert(key, fractions);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::api::metric::{BenchMetrics, MetricPoint};

    use super::{
        collect_derived_metrics, collect_gpu_counters, collect_metric_fields,
        derive_roofline_fractions,
    };

    fn metrics_with_host_only(input_bytes: u64, wall_ns: u64) -> BenchMetrics {
        BenchMetrics {
            input_bytes: Some(input_bytes),
            wall_ns: Some(wall_ns),
            ..Default::default()
        }
    }

    fn metrics_with_device(bytes_read: u64, bytes_written: u64, wall_ns: u64) -> BenchMetrics {
        BenchMetrics {
            bytes_read: Some(bytes_read),
            bytes_written: Some(bytes_written),
            wall_ns: Some(wall_ns),
            ..Default::default()
        }
    }

    /// Regression for device-bytes-silent-host-fallback: when a case sets only
    /// input_bytes/output_bytes (host-side I/O) and no bytes_read/bytes_written,
    /// device_gb_s_x1000 must be absent from the sample map.  Before the fix,
    /// device_bytes silently fell back to host_bytes and device_gb_s_x1000 was
    /// emitted as if the GPU had transferred 512 MiB of device memory, a metric
    /// miscompile that cannot be distinguished from a real device bandwidth measurement.
    #[test]
    fn device_gb_s_x1000_absent_when_no_device_bytes_set() {
        // 512 MiB of host input, 1 second wall time (no bytes_read/bytes_written).
        let metrics = metrics_with_host_only(512 * 1024 * 1024, 1_000_000_000);
        let mut samples: BTreeMap<&'static str, Vec<u64>> = BTreeMap::new();
        collect_derived_metrics("", &metrics, &mut samples);

        assert!(
            !samples.contains_key("device_gb_s_x1000"),
            "Fix: device_gb_s_x1000 must NOT be emitted when bytes_read/bytes_written are absent \
             (512 MiB host input must not be misreported as device bandwidth). \
             Got samples: {:?}",
            samples.keys().collect::<Vec<_>>()
        );
        // wall_gb_s_x1000 MUST still be present (host bandwidth is unaffected).
        assert!(
            samples.contains_key("wall_gb_s_x1000"),
            "Fix: wall_gb_s_x1000 must still be emitted when host bytes are present."
        );
        let wall_val = samples["wall_gb_s_x1000"][0];
        // 512 MiB = 536870912 bytes, 1 s = 1_000_000_000 ns, scale = 1_000_000_000.
        // rate_per_second_x1000 = (units * 1e12) / (wall_ns * scale)
        //   = (536870912 * 1_000_000_000_000) / (1_000_000_000 * 1_000_000_000)
        //   = 536870912_000_000_000_000 / 1_000_000_000_000_000_000
        //   = 536 (integer division).
        // 512 MiB / 1 s ≈ 0.537 GB/s → 537 x1000-units; floor = 536.
        assert_eq!(
            wall_val, 536,
            "Fix: wall_gb_s_x1000 must equal 536 for 512 MiB / 1 s; got {wall_val}"
        );
    }

    /// device_gb_s_x1000 MUST be present and correct when both bytes_read and
    /// bytes_written are explicitly set by the case.
    #[test]
    fn device_gb_s_x1000_present_when_device_bytes_set() {
        // 256 MiB read + 256 MiB written = 512 MiB device transfer, 1 second.
        let metrics = metrics_with_device(256 * 1024 * 1024, 256 * 1024 * 1024, 1_000_000_000);
        let mut samples: BTreeMap<&'static str, Vec<u64>> = BTreeMap::new();
        collect_derived_metrics("", &metrics, &mut samples);

        assert!(
            samples.contains_key("device_gb_s_x1000"),
            "Fix: device_gb_s_x1000 must be emitted when bytes_read and bytes_written are set."
        );
        let val = samples["device_gb_s_x1000"][0];
        // 512 MiB device transfer / 1 s → same calculation as above: 536 x1000-units.
        assert_eq!(
            val, 536,
            "Fix: device_gb_s_x1000 must equal 536 for 512 MiB device transfer / 1 s; got {val}"
        );
    }

    /// WHY: `bytes_read` and `bytes_written` are HOST transfer bytes. On a
    /// resident dispatch they are 0 and the readback size, so a device rate
    /// derived from them states host I/O as device bandwidth. The release
    /// scatter case published 12.047 GB/s from a 131072-byte readback while its
    /// kernel moved 8650752 bytes. A case that states `device_bytes_moved` must
    /// have that figure used instead.
    ///
    /// Does not catch a case that states a wrong `device_bytes_moved`; that is
    /// the case's own accounting, asserted where the case builds it.
    #[test]
    fn device_rate_prefers_stated_device_traffic_over_host_transfers() {
        // A resident dispatch: no host read, a 131072-byte readback, and a
        // kernel that moved 8650752 bytes in 10828 ns.
        let metrics = BenchMetrics {
            bytes_read: Some(0),
            bytes_written: Some(131_072),
            device_bytes_moved: Some(8_650_752),
            dispatch_ns: Some(10_828),
            wall_ns: Some(94_088),
            ..Default::default()
        };
        let mut samples: BTreeMap<&'static str, Vec<u64>> = BTreeMap::new();
        collect_derived_metrics("", &metrics, &mut samples);

        // 8650752 B / 10828 ns = 798.9 GB/s. The host-transfer figure would
        // have been 131072 / 10828 = 12.1 GB/s.
        assert_eq!(samples["device_gb_s_x1000"][0], 798_924);

        // A case that states no device traffic keeps the host transfer total,
        // which is exact when the host transfer IS the device traffic.
        let transfer_only = BenchMetrics {
            device_bytes_moved: None,
            ..metrics
        };
        let mut transfer_samples: BTreeMap<&'static str, Vec<u64>> = BTreeMap::new();
        collect_derived_metrics("", &transfer_only, &mut transfer_samples);
        assert_eq!(transfer_samples["device_gb_s_x1000"][0], 12_104);
    }

    /// WHY: a roofline percentage states how much of the device's bandwidth the
    /// kernel used while it was running, so it is computed against device
    /// active time. Computed against sample wall time it charges readback and
    /// host overhead to the device: the resident conditional case published
    /// 0.45% of a 1792 GB/s device for a kernel using 44%, because its wall
    /// time is ten times its kernel time.
    ///
    /// Does not catch a wrong `memory_peak_gb_s_x1000`; that is the device
    /// telemetry's own figure.
    #[test]
    fn roofline_fraction_is_computed_against_device_active_time() {
        let metrics = BenchMetrics {
            device_bytes_moved: Some(8_650_752),
            dispatch_ns: Some(10_828),
            wall_ns: Some(94_088),
            gpu_counter: vec![crate::api::metric::GpuCounter {
                name: "memory_peak_gb_s_x1000".to_string(),
                value: 1_792_000,
            }],
            ..Default::default()
        };
        let mut samples: BTreeMap<&'static str, Vec<u64>> = BTreeMap::new();
        collect_gpu_counters("", &metrics, &mut samples);
        collect_derived_metrics("", &metrics, &mut samples);
        derive_roofline_fractions(&mut samples);

        // 8650752 B / 10828 ns = 798.924 GB/s, which is 44.582% of 1792 GB/s.
        // Against wall time it would have been 91.943 GB/s, or 5.131%.
        assert_eq!(
            samples["roofline_mem_pct_x1000"][0], 44_582,
            "Fix: roofline_mem_pct_x1000 must divide device bytes by device active time."
        );
    }

    /// WHY: the device memory peak arrives with NVML telemetry, which is
    /// captured on one sample per run. A fraction derived per sample therefore
    /// existed for exactly that one sample, and its p50 was that sample's
    /// value: a scheduling outlier on the telemetry sample stated 2.907% of a
    /// 1792 GB/s device for a run whose achieved-rate p50 was 39%, and a
    /// bandwidth contract read against it failed on one run of the same binary
    /// and passed on the next two. The fraction must span the whole
    /// achieved-rate series so its p50 tracks the rate's p50.
    ///
    /// Does not catch a peak that changes mid-run; the first positive reading
    /// stands for the run.
    #[test]
    fn roofline_fraction_spans_every_achieved_rate_sample() {
        let mut samples: BTreeMap<&'static str, Vec<u64>> = BTreeMap::new();
        // Three fast samples and one outlier, with telemetry captured on the
        // outlier: 798.924, 798.924, 798.924, and 52.1 GB/s.
        samples.insert(
            "device_gb_s_x1000",
            vec![798_924, 798_924, 798_924, 52_100],
        );
        samples.insert("memory_peak_gb_s_x1000", vec![1_792_000]);
        derive_roofline_fractions(&mut samples);

        assert_eq!(
            samples["roofline_mem_pct_x1000"],
            vec![44_582, 44_582, 44_582, 2_907],
            "Fix: the roofline fraction must be derived once per achieved-rate sample."
        );
    }

    /// WHY: without device memory-peak telemetry there is no denominator, and
    /// a case whose contract states a bandwidth fraction must fail closed on
    /// the absent metric rather than read a fabricated one.
    #[test]
    fn roofline_fraction_is_absent_without_a_device_peak() {
        let mut samples: BTreeMap<&'static str, Vec<u64>> = BTreeMap::new();
        samples.insert("device_gb_s_x1000", vec![798_924]);
        derive_roofline_fractions(&mut samples);

        assert!(
            !samples.contains_key("roofline_mem_pct_x1000"),
            "Fix: a roofline fraction must not be published without a measured device peak."
        );

        let mut rate_less: BTreeMap<&'static str, Vec<u64>> = BTreeMap::new();
        rate_less.insert("memory_peak_gb_s_x1000", vec![1_792_000]);
        derive_roofline_fractions(&mut rate_less);
        assert!(
            !rate_less.contains_key("roofline_mem_pct_x1000"),
            "Fix: a roofline fraction must not be published without a measured device rate."
        );
    }

    /// Regression for dead-cold-fields-in-collect-fields-array: the 7 cold_* names
    /// previously listed in FIELDS are permanently inert because metric_key() returns
    /// None for them.  After the fix the FIELDS array no longer contains cold_* entries,
    /// so even if BenchMetrics.cold_wall_ns is set, collect_metric_fields must not emit
    /// a "cold_wall_ns" sample (the cold path in run_case.rs is the authoritative emitter).
    #[test]
    fn collect_metric_fields_does_not_emit_cold_samples() {
        let mut metrics = BenchMetrics::default();
        metrics.cold_wall_ns = Some(12_345_678);
        metrics.cold_compile_ns = Some(9_000_000);
        metrics.cold_optimize_ns = Some(1_000_000);
        metrics.cold_lower_ns = Some(500_000);
        metrics.cold_cache_lookup_ns = Some(200_000);
        metrics.cold_dispatch_ns = Some(100_000);
        metrics.cold_readback_ns = Some(50_000);

        let mut samples: BTreeMap<&'static str, Vec<u64>> = BTreeMap::new();
        collect_metric_fields("", &metrics, &mut samples);

        for cold_key in [
            "cold_wall_ns",
            "cold_compile_ns",
            "cold_optimize_ns",
            "cold_lower_ns",
            "cold_cache_lookup_ns",
            "cold_dispatch_ns",
            "cold_readback_ns",
        ] {
            assert!(
                !samples.contains_key(cold_key),
                "Fix: collect_metric_fields must not emit `{cold_key}`: cold-path metrics are \
                 populated by run_case.rs directly and do not route through collect_metric_fields."
            );
        }
    }

    /// Verify that the cold_* entries were removed from FIELDS (not just silenced) by
    /// checking the sample map is entirely empty when only cold_* fields are set on
    /// BenchMetrics.  Previously the entries were present but always inert; now they
    /// should simply not be there.
    #[test]
    fn collect_metric_fields_cold_only_metrics_produce_empty_samples() {
        let mut metrics = BenchMetrics::default();
        metrics.cold_wall_ns = Some(999);
        let mut samples: BTreeMap<&'static str, Vec<u64>> = BTreeMap::new();
        collect_metric_fields("", &metrics, &mut samples);
        assert!(
            samples.is_empty(),
            "Fix: a BenchMetrics with only cold_wall_ns set must produce zero samples from \
             collect_metric_fields; got {:?}",
            samples.keys().collect::<Vec<_>>()
        );
    }

    /// collect_custom_metrics must emit a sample for every SyntheticCountWorkload
    /// metric_name that passes through custom_metric_key.
    #[test]
    fn collect_custom_metrics_emits_synthetic_count_workload_metrics() {
        use super::collect_custom_metrics;
        // Verify the collect path: build BenchMetrics with custom points for each
        // synthetic workload name and confirm they appear in the sample map.
        let names = crate::runner::execute::metric_keys::SYNTHETIC_COUNT_METRIC_NAMES;
        let mut metrics = BenchMetrics::default();
        for (i, name) in names.iter().enumerate() {
            metrics.custom.push(MetricPoint {
                name: (*name).to_string(),
                value: (i as u64 + 1) * 1_000_000,
            });
        }
        let mut samples: BTreeMap<&'static str, Vec<u64>> = BTreeMap::new();
        collect_custom_metrics("", &metrics, &mut samples);
        for (i, name) in names.iter().enumerate() {
            let expected_value = (i as u64 + 1) * 1_000_000;
            let actual = samples.get(*name).and_then(|v| v.first()).copied();
            assert_eq!(
                actual,
                Some(expected_value),
                "Fix: custom metric `{name}` must be collected with value {expected_value}; \
                 got {actual:?}"
            );
        }
    }
    /// WHY: a non-dispatch evidence case reports an explicit zero launch count.
    /// Dropping zero here makes release normalization invent one launch.
    #[test]
    fn collect_custom_metrics_preserves_explicit_zero_kernel_launches() {
        use super::collect_custom_metrics;

        let metrics = BenchMetrics {
            custom: vec![MetricPoint {
                name: "kernel_launches".to_string(),
                value: 0,
            }],
            ..Default::default()
        };
        let mut samples = BTreeMap::new();

        collect_custom_metrics("", &metrics, &mut samples);

        assert_eq!(samples.get("kernel_launches"), Some(&vec![0]));
    }
}
