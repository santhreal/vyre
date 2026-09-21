//! Bounded causal tracer and scoped span recording.

use std::sync::atomic::{AtomicU64, Ordering};

use super::event::CausalEvent;
use super::id::{CausalSpanId, TraceId};
use super::mode::CausalTraceMode;
use super::receipt::CausalReceipt;

static GLOBAL_SPAN_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Monotonic span id generator.
#[inline]
pub fn next_span_id() -> CausalSpanId {
    CausalSpanId::new(GLOBAL_SPAN_COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// Bounded causal tracer that records events when enabled and skips allocations when disabled.
#[derive(Clone, Debug)]
pub struct CausalTracer {
    trace_id: TraceId,
    root_span_id: CausalSpanId,
    mode: CausalTraceMode,
    events: Vec<CausalEvent>,
    capacity_limit: usize,
}

impl CausalTracer {
    /// Create a disabled tracer with zero allocated buffer.
    #[inline]
    pub fn disabled() -> Self {
        Self {
            trace_id: TraceId::new(0),
            root_span_id: CausalSpanId::new(0),
            mode: CausalTraceMode::Off,
            events: Vec::new(),
            capacity_limit: 0,
        }
    }

    /// Create a new tracer for a given trace id and mode.
    pub fn new(trace_id: TraceId, root_span_id: CausalSpanId, mode: CausalTraceMode) -> Self {
        let capacity = match mode {
            CausalTraceMode::Off => 0,
            CausalTraceMode::Sampled { .. } | CausalTraceMode::Full => 256,
        };
        Self {
            trace_id,
            root_span_id,
            mode,
            events: Vec::with_capacity(capacity),
            capacity_limit: 65536,
        }
    }

    /// Returns whether this tracer is actively recording events.
    #[inline(always)]
    pub fn is_active(&self) -> bool {
        !self.mode.is_off()
    }

    /// Return the active trace mode.
    #[inline]
    pub const fn mode(&self) -> CausalTraceMode {
        self.mode
    }

    /// Record a completed causal event.
    #[inline]
    pub fn record_event(&mut self, event: CausalEvent) {
        if self.mode.is_off() {
            return;
        }
        if self.events.len() < self.capacity_limit {
            self.events.push(event);
        }
    }

    /// Finalize and build an end-to-end CausalReceipt.
    pub fn finalize(self) -> CausalReceipt {
        let mut receipt = CausalReceipt::new(self.trace_id, self.root_span_id);
        for event in self.events {
            receipt.record_event(event);
        }
        let _ = receipt.reconstruct_critical_path();
        receipt
    }
}
