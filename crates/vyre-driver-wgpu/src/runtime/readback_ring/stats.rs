//! Runtime counters for one readback ring.

use std::sync::atomic::{AtomicU64, Ordering};
use vyre_driver::accounting::{atomic_max_u64, rebasing_atomic_next_u64};

/// Statistics collected by the ring at runtime.
#[derive(Debug, Default)]
pub struct RingStats {
    /// Total dispatches queued.
    pub dispatches: AtomicU64,
    /// Readbacks that blocked waiting on map_async.
    pub readback_stalls: AtomicU64,
    /// Max outstanding (in-flight) copies.
    pub peak_inflight: AtomicU64,
}

impl RingStats {
    /// Record one dispatch; returns the monotonic dispatch index.
    pub fn record_dispatch(&self) -> u64 {
        rebasing_atomic_next_u64(
            &self.dispatches,
            0,
            Ordering::Relaxed,
            Ordering::Relaxed,
            Ordering::Relaxed,
            |_, _| {
                tracing::error!(
                    "readback ring dispatch counter reached u64::MAX and was rebased to zero. Fix: shard readback rings or scrape counters before wrap."
                );
            },
        )
    }

    /// Record a stall.
    pub fn record_stall(&self) {
        rebasing_atomic_next_u64(
            &self.readback_stalls,
            0,
            Ordering::Relaxed,
            Ordering::Relaxed,
            Ordering::Relaxed,
            |_, _| {
                tracing::error!(
                    "readback ring stall counter reached u64::MAX and was rebased to zero. Fix: shard readback rings or scrape counters before wrap."
                );
            },
        );
    }

    /// Update the peak-in-flight watermark.
    pub fn update_peak(&self, current: u64) {
        atomic_max_u64(&self.peak_inflight, current, Ordering::AcqRel);
    }
}
