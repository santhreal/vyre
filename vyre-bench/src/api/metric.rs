use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStats {
    pub min: u64,
    pub p50: u64,
    pub p90: u64,
    pub p95: u64,
    pub p99: u64,
    pub p999: u64,
    pub p9999: u64,
    pub max: u64,
    pub mean: f64,
    pub stddev: f64,
    pub samples: u32,
    pub determinism_cv: Option<f64>,
}

impl MetricStats {
    /// Produce a degenerate `MetricStats` for a single observation.
    #[must_use]
    pub fn single(value: u64) -> Self {
        Self {
            min: value,
            p50: value,
            p90: value,
            p95: value,
            p99: value,
            p999: value,
            p9999: value,
            max: value,
            mean: value as f64,
            stddev: 0.0,
            samples: 1,
            determinism_cv: None,
        }
    }

    /// MetricStats with custom summary parameters.
    #[must_use]
    pub fn point(p50: u64, mean: f64, stddev: f64, samples: u32) -> Self {
        Self {
            min: p50,
            p50,
            p90: p50,
            p95: p50,
            p99: p50,
            p999: p50,
            p9999: p50,
            max: p50,
            mean,
            stddev,
            samples,
            determinism_cv: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCounter {
    pub name: String,
    pub value: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    pub name: String,
    pub value: u64,
}

/// Nanoseconds elapsed since `started`, saturating at `u64::MAX`.
///
/// Every timed span in the benchmark reports nanoseconds as a `u64`, and
/// `Duration::as_nanos` is a `u128`. Three spellings of the narrowing coexisted:
/// a bare `as u64` cast, a `min(u64::MAX)` clamp, and a `try_from().unwrap_or`.
/// The bare cast truncates instead of saturating: a count past `u64::MAX`
/// nanoseconds is reported as an unrelated small number rather than clamped at
/// the maximum. This is the single spelling, and it saturates.
#[must_use]
pub fn elapsed_ns(started: std::time::Instant) -> u64 {
    narrow_nanos(started.elapsed().as_nanos())
}

/// The one narrowing of a nanosecond count to the `u64` every metric reports.
///
/// Split out from `elapsed_ns` so the saturation boundary is reachable without
/// an `Instant` far enough in the past to overflow, which no monotonic clock
/// can supply.
#[must_use]
fn narrow_nanos(nanos: u128) -> u64 {
    u64::try_from(nanos).unwrap_or(u64::MAX)
}

#[must_use]
pub fn digest64_buffers(buffers: &[Vec<u8>]) -> u64 {
    let mut hasher = blake3::Hasher::new();
    for buffer in buffers {
        hasher.update(&(buffer.len() as u64).to_le_bytes());
        hasher.update(buffer);
    }
    let hash = hasher.finalize();
    let mut digest = [0u8; 8];
    digest.copy_from_slice(&hash.as_bytes()[..8]);
    u64::from_le_bytes(digest)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchMetrics {
    pub wall_ns: Option<u64>,
    pub cpu_ns: Option<u64>,
    pub compile_ns: Option<u64>,
    pub validate_ns: Option<u64>,
    pub optimize_ns: Option<u64>,
    pub lower_ns: Option<u64>,
    pub cache_lookup_ns: Option<u64>,
    pub cache_hit: Option<bool>,
    pub upload_ns: Option<u64>,
    pub dispatch_ns: Option<u64>,
    pub kernel_queue_submit_ns: Option<u64>,
    pub kernel_execute_ns: Option<u64>,
    pub device_sync_ns: Option<u64>,
    pub readback_ns: Option<u64>,
    pub verify_ns: Option<u64>,
    pub alloc_count: Option<u64>,
    pub alloc_bytes: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    pub input_bytes: Option<u64>,
    pub output_bytes: Option<u64>,
    pub bytes_touched: Option<u64>,
    pub bytes_read: Option<u64>,
    pub bytes_written: Option<u64>,
    pub atomic_op_count: Option<u64>,
    pub wall_throughput_gb_s: Option<f64>,
    pub device_throughput_gb_s: Option<f64>,
    pub peak_bandwidth_gb_s: Option<f64>,
    pub achieved_bandwidth_gb_s: Option<f64>,
    pub roofline_pct: Option<f64>,
    pub throughput_gflops: Option<f64>,
    pub ir_nodes: Option<u64>,
    pub wire_bytes: Option<u64>,
    pub gpu_counter: Vec<GpuCounter>,
    pub custom: Vec<MetricPoint>,
    /// cold-vs-warm separation. Wall-clock (ns) of the
    /// first warmup sample for this case, captured before any pipeline
    /// cache hits, before any naga module cache hits, and before the
    /// GPU adapter has memoised the kernel. Compare against `wall_ns`
    /// (the warm steady-state median) to attribute time to cold-start
    /// work versus per-dispatch work.
    pub cold_wall_ns: Option<u64>,
    /// First-warmup compile-time stage breakdown. Mirrors
    /// `compile_ns` / `lower_ns` / `optimize_ns` etc. but only for the
    /// cold sample. None for stages the cold path did not measure.
    pub cold_compile_ns: Option<u64>,
    pub cold_optimize_ns: Option<u64>,
    pub cold_lower_ns: Option<u64>,
    pub cold_cache_lookup_ns: Option<u64>,
    pub cold_dispatch_ns: Option<u64>,
    pub cold_readback_ns: Option<u64>,
}

impl BenchMetrics {
    /// CPU-side achieved memory bandwidth probe.
    ///
    /// Returns `bytes_touched / wall_ns * 1e9 / 1e9` (= `bytes_touched / wall_ns`)
    /// in GB/s when both `bytes_touched` and `wall_ns` are present and
    /// `wall_ns` is non-zero. Returns `None` when either field is missing
    /// or `wall_ns == 0` (to avoid division by zero).
    ///
    /// The backend-counter half (reading hardware bandwidth counters from
    /// the GPU) needs concrete driver wiring.
    #[must_use]
    pub fn achieved_bandwidth_gb_s(&self) -> Option<f64> {
        let bytes = self.bytes_touched?;
        let wall = self.wall_ns?;
        if wall == 0 {
            return None;
        }
        // bytes / wall_ns gives bytes/ns = GB/s
        Some(bytes as f64 / wall as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics_with(bytes_touched: Option<u64>, wall_ns: Option<u64>) -> BenchMetrics {
        BenchMetrics {
            bytes_touched,
            wall_ns,
            ..Default::default()
        }
    }

    #[test]
    fn achieved_bandwidth_both_present() {
        let m = metrics_with(Some(1_000_000_000), Some(1_000_000_000));
        let bw = m
            .achieved_bandwidth_gb_s()
            .expect("Fix: both fields present");
        assert!((bw - 1.0).abs() < 1e-9, "1GB / 1s = 1 GB/s; got {bw}");
    }

    #[test]
    fn achieved_bandwidth_missing_wall_ns() {
        let m = metrics_with(Some(1_000_000_000), None);
        assert!(
            m.achieved_bandwidth_gb_s().is_none(),
            "missing wall_ns must return None"
        );
    }

    #[test]
    fn achieved_bandwidth_missing_bytes_touched() {
        let m = metrics_with(None, Some(1_000_000_000));
        assert!(
            m.achieved_bandwidth_gb_s().is_none(),
            "missing bytes_touched must return None"
        );
    }

    #[test]
    fn achieved_bandwidth_zero_wall_ns() {
        let m = metrics_with(Some(1_000_000_000), Some(0));
        assert!(
            m.achieved_bandwidth_gb_s().is_none(),
            "zero wall_ns must return None to avoid div-by-zero"
        );
    }

    #[test]
    fn digest64_buffers_is_length_delimited() {
        let joined = digest64_buffers(&[b"ab".to_vec(), b"c".to_vec()]);
        let split = digest64_buffers(&[b"a".to_vec(), b"bc".to_vec()]);
        assert_ne!(
            joined, split,
            "Fix: benchmark output digests must include buffer boundaries."
        );
    }

    /// A span longer than `u64::MAX` nanoseconds reports as the maximum rather
    /// than wrapping to a small number. No monotonic clock can produce an
    /// `Instant` 585 years in the past, so the boundary is driven through
    /// `narrow_nanos`, the function `elapsed_ns` delegates the cast to. The
    /// bare `as u64` cast this replaced wrapped `u64::MAX + 1` to 0.
    #[test]
    fn nanosecond_narrowing_saturates_rather_than_wrapping() {
        assert_eq!(narrow_nanos(u128::from(u64::MAX) + 1), u64::MAX);
        assert_eq!(narrow_nanos(u128::from(u64::MAX) * 4), u64::MAX);
        assert_eq!(narrow_nanos(u128::from(u64::MAX)), u64::MAX);
        assert_eq!(narrow_nanos(1_000_000), 1_000_000);
        assert_eq!(narrow_nanos(0), 0);
    }

    /// A real span narrows to a plausible nonzero count without saturating.
    #[test]
    fn real_span_reports_between_zero_and_saturation() {
        let started = std::time::Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let observed = elapsed_ns(started);
        assert!(
            observed >= 1_000_000,
            "2ms sleep must exceed 1ms: {observed}"
        );
        assert!(observed < u64::MAX, "a real span must not saturate");
    }
}
