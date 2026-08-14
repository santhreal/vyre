//! How a CPU reference sample is timed and reported.
//!
//! Every case that reports a GPU sample against a host answer needs the same two
//! things: the host time the reference took, and the metrics record that time is
//! published in. Both were hand-rolled per case, and the hand-rolled timers did
//! not agree: some cast `as_nanos()` straight to `u64`, which truncates a
//! reference slower than 18 seconds into a small number instead of a large one.
//! This module saturates instead, so a slow reference reports as slow.

use crate::api::metric::BenchMetrics;
use std::time::Instant;

/// Run a CPU reference and report how long it took.
///
/// The reference produces both the expected output and the host time the GPU
/// sample is reported against, so one call covers both. The closure is the
/// caller's own: nothing here decides what the reference computes, only how long
/// it is measured to take.
pub(crate) fn timed_reference<T>(reference: impl FnOnce() -> T) -> (T, u64) {
    let started = Instant::now();
    let value = reference();
    let wall_ns = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    (value, wall_ns)
}

/// Metrics for a CPU reference sample that rewrites host buffers.
///
/// The reference reads its whole input set and writes whatever it produced, and
/// never dispatches, so it carries no device time. A case whose reference does
/// not consume its inputs, or reports its own counters, builds its own record
/// instead of calling this.
pub(crate) fn reference_metrics(wall_ns: u64, input_bytes: u64, output_bytes: u64) -> BenchMetrics {
    BenchMetrics {
        wall_ns: Some(wall_ns),
        input_bytes: Some(input_bytes),
        output_bytes: Some(output_bytes),
        bytes_touched: Some(input_bytes.saturating_add(output_bytes)),
        bytes_read: Some(input_bytes),
        bytes_written: Some(output_bytes),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::{reference_metrics, timed_reference};

    /// A reference slower than `u64::MAX` nanoseconds saturates instead of
    /// truncating, so a pathologically slow host answer cannot be reported as a
    /// fast one. The hand-rolled `as_nanos() as u64` casts this replaced wrapped.
    #[test]
    fn elapsed_saturates_rather_than_wrapping() {
        let huge = u128::from(u64::MAX) + 1;
        assert_eq!(huge.min(u128::from(u64::MAX)) as u64, u64::MAX);
        assert_ne!(huge as u64, u64::MAX);
    }

    /// The closure's value is returned untouched and its side effects happen
    /// exactly once, so timing a reference cannot change what it computed.
    #[test]
    fn reference_value_passes_through_once() {
        let mut calls = 0;
        let (value, wall_ns) = timed_reference(|| {
            calls += 1;
            vec![7_u8, 9]
        });

        assert_eq!(calls, 1);
        assert_eq!(value, vec![7, 9]);
        assert!(wall_ns < 1_000_000_000);
    }

    /// The reference sample touches its inputs and its outputs and nothing else.
    #[test]
    fn reference_metrics_account_both_directions() {
        let metrics = reference_metrics(77, 1_024, 16);

        assert_eq!(metrics.wall_ns, Some(77));
        assert_eq!(metrics.dispatch_ns, None);
        assert_eq!(metrics.input_bytes, Some(1_024));
        assert_eq!(metrics.output_bytes, Some(16));
        assert_eq!(metrics.bytes_read, Some(1_024));
        assert_eq!(metrics.bytes_written, Some(16));
        assert_eq!(metrics.bytes_touched, Some(1_040));
    }

    /// Byte totals saturate rather than wrapping, so an absurd accounting figure
    /// stays absurd instead of becoming a plausible small one.
    #[test]
    fn byte_totals_saturate() {
        let metrics = reference_metrics(1, u64::MAX, 8);

        assert_eq!(metrics.bytes_touched, Some(u64::MAX));
    }
}
