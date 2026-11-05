//! Hot/cold tiered metadata layered over the power-of-two buffer pool.
//!
//! The pool owns allocation and reuse. This module owns the bounded event
//! queue that carries retain and access notices to a metadata worker, so the
//! acquire and release paths never take the cache mutex.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use vyre_driver::BackendError;

const TIERING_EVENT_CAPACITY_MIN: usize = 1024;
const TIERING_EVENT_CAPACITY_MAX: usize = 65_536;

/// Opt-in hot/cold tiered metadata layered over the power-of-two pool.
///
/// Off by default. Consumers that batch many small dispatches (inference
/// servers, Karyx streaming scanners, Soleno batched probes) wire one
/// via [`super::BufferPool::with_tiering`] and tag hot allocations through the
/// returned handle. The tiering layer records allocation reuse through
/// a bounded non-blocking event queue and drains it into `TieredCache`
/// on a dedicated metadata worker. This keeps acquire/release free of
/// a global mutex while preserving the cache policy's per-tier O(1)
/// LRU accounting.
///
/// Kept as `pub(crate) Option<Arc<...>>` so the absence of a tiering
/// policy costs exactly one `Option::is_none()` branch on the hot
/// acquire path.
pub(crate) struct PoolTiering {
    events: Sender<TieringEvent>,
    pending_events: Arc<AtomicUsize>,
    dropped_events: AtomicUsize,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum TieringEvent {
    Retain { key: u64, size: u64 },
    Access { key: u64 },
}

impl PoolTiering {
    pub(super) fn new(
        cache: crate::runtime::cache::TieredCache,
        capacity: usize,
    ) -> Result<Self, BackendError> {
        let capacity = capacity.clamp(TIERING_EVENT_CAPACITY_MIN, TIERING_EVENT_CAPACITY_MAX);
        let (events, receiver) = bounded(capacity);
        let pending_events = Arc::new(AtomicUsize::new(0));
        let worker_pending = Arc::clone(&pending_events);
        std::thread::Builder::new()
            .name("vyre-buffer-tiering".to_string())
            .spawn(move || drain_tiering_events(cache, receiver, worker_pending))
            .map_err(|error| {
                BackendError::new(format!(
                    "failed to spawn vyre buffer tiering worker: {error}. Fix: raise process thread limits or disable buffer-pool tiering."
                ))
            })?;
        Ok(Self::from_sender(events, pending_events))
    }

    /// The counting half of construction, without the worker thread.
    ///
    /// `new` owns the drain thread, whose scheduling is what makes a queue
    /// saturate at all. Keeping the counters separable lets the full-queue
    /// accounting be driven from a held receiver instead of from a race
    /// against that thread.
    pub(super) fn from_sender(
        events: Sender<TieringEvent>,
        pending_events: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            events,
            pending_events,
            dropped_events: AtomicUsize::new(0),
        }
    }

    /// Tiering metadata events discarded since construction.
    pub(super) fn dropped_events(&self) -> usize {
        self.dropped_events.load(Ordering::Relaxed)
    }

    #[inline]
    pub(super) fn record_retained(&self, key: u64, size: u64) {
        self.enqueue(TieringEvent::Retain { key, size });
    }

    #[inline]
    pub(super) fn record_access(&self, key: u64) {
        self.enqueue(TieringEvent::Access { key });
    }

    #[inline]
    fn enqueue(&self, event: TieringEvent) {
        self.pending_events.fetch_add(1, Ordering::Release);
        match self.events.try_send(event) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.pending_events.fetch_sub(1, Ordering::AcqRel);
                self.dropped_events.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    #[cfg(all(test, feature = "device-tests"))]
    pub(super) fn drain_all_for_test(&self) {
        // The metadata worker is woken via crossbeam channel; under
        // contention from multiple acquire/release threads it can
        // accumulate a backlog before the OS schedules it. Use bounded
        // adaptive parking rather than a fixed millisecond sleep; fixed
        // sleeps create thundering-herd latency under high test fanout.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut backoff = crate::wait_backoff::AdaptiveWaitBackoff::from_micros(64, 2, 50, 5);
        while std::time::Instant::now() < deadline {
            if self.pending_events.load(Ordering::Acquire) == 0 {
                return;
            }
            backoff.idle_until(deadline);
        }
        panic!("Fix: tiering metadata worker did not drain pending buffer-pool events");
    }
}

fn drain_tiering_events(
    mut cache: crate::runtime::cache::TieredCache,
    receiver: Receiver<TieringEvent>,
    pending_events: Arc<AtomicUsize>,
) {
    while let Ok(event) = receiver.recv() {
        match event {
            TieringEvent::Retain { key, size } => {
                if cache.get(key).is_none() {
                    if let Err(error) = cache.insert(key, size) {
                        tracing::warn!(
                            "buffer pool tiering rejected retained buffer {key} ({size} bytes): {error}. Fix: increase tier capacity or disable tiering for oversized buffers."
                        );
                        pending_events.fetch_sub(1, Ordering::AcqRel);
                        continue;
                    }
                }
                cache.record_access(key);
                if let Err(error) = cache.promote(key) {
                    tracing::warn!(
                        "buffer pool tier promotion failed for retained buffer {key}: {error}. Fix: repair tier sizing or promotion accounting."
                    );
                }
            }
            TieringEvent::Access { key } => {
                cache.record_access(key);
                if let Err(error) = cache.promote(key) {
                    tracing::warn!(
                        "buffer pool tier promotion failed for accessed buffer {key}: {error}. Fix: repair tier sizing or promotion accounting."
                    );
                }
            }
        }
        pending_events.fetch_sub(1, Ordering::AcqRel);
    }
}
