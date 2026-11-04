use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Instant;

thread_local! {
    // FxHashMap (FxHashBuilder) replaces the default SipHash-backed
    // HashMap so per-PerfScope::finish hashes cost ~10ns instead of
    // ~50ns. Optimizer pipelines call into roughly two perf scopes per
    // pass × ~120 passes per optimize()  -  the savings compound.
    static METRICS: RefCell<FxHashMap<&'static str, u64>> = RefCell::new(FxHashMap::default());
}

/// RAII guard returned by [`span`].
pub struct SpanGuard {
    name: &'static str,
    start: Instant,
}

/// Completed performance timing sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerfMeasurement {
    /// Elapsed nanoseconds.
    pub elapsed_ns: u64,
}

/// Explicitly-finished performance scope used by hot paths that need the
/// elapsed value in addition to thread-local accumulation.
pub struct PerfScope {
    name: &'static str,
    start: Instant,
}

impl PerfScope {
    /// Start a named performance scope.
    #[must_use]
    pub fn start(_crate_name: &'static str, name: &'static str) -> Self {
        Self {
            name,
            start: Instant::now(),
        }
    }

    /// Finish this scope and accumulate its elapsed time.
    #[must_use]
    pub fn finish(self) -> PerfMeasurement {
        let elapsed_ns = narrow_nanos(self.start.elapsed().as_nanos());
        METRICS.with(|metrics| {
            let mut map = metrics.borrow_mut();
            *map.entry(self.name).or_insert(0) += elapsed_ns;
        });
        PerfMeasurement { elapsed_ns }
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        let elapsed = narrow_nanos(self.start.elapsed().as_nanos());
        METRICS.with(|metrics| {
            let mut map = metrics.borrow_mut();
            *map.entry(self.name).or_insert(0) += elapsed;
        });
    }
}

/// The one narrowing of a nanosecond count to the `u64` a metric carries.
///
/// `Duration::as_nanos` answers a `u128`. Both accumulation sites cast it, and a
/// cast discards the high bits instead of clamping, so a count past `u64::MAX`
/// nanoseconds would be recorded as an unrelated small number and read as a fast
/// span. Saturating keeps the ordering of the recorded numbers, which is the only
/// property a metric is read for.
fn narrow_nanos(nanos: u128) -> u64 {
    u64::try_from(nanos).unwrap_or(u64::MAX)
}

/// Start a performance span. When the returned guard is dropped, the elapsed time
/// is accumulated into the thread-local metric named `name`.
pub fn span(name: &'static str) -> SpanGuard {
    SpanGuard {
        name,
        start: Instant::now(),
    }
}

/// Retrieve all accumulated metrics for the current thread.
pub fn get_metrics() -> HashMap<&'static str, u64> {
    METRICS.with(|metrics| metrics.borrow().iter().map(|(k, v)| (*k, *v)).collect())
}

/// Reset all accumulated metrics for the current thread.
pub fn reset_metrics() {
    METRICS.with(|metrics| metrics.borrow_mut().clear());
}
