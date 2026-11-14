use std::time::Instant;

/// Completed performance timing sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerfMeasurement {
    /// Elapsed nanoseconds.
    pub elapsed_ns: u64,
}

/// Explicitly-finished performance scope used by hot paths.
pub struct PerfScope {
    start: Instant,
}

impl PerfScope {
    /// Start a named performance scope.
    #[must_use]
    #[inline]
    pub fn start(_crate_name: &'static str, _name: &'static str) -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Finish this scope and return its elapsed time.
    #[must_use]
    #[inline]
    pub fn finish(self) -> PerfMeasurement {
        let elapsed_ns = u64::try_from(self.start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        PerfMeasurement { elapsed_ns }
    }
}
