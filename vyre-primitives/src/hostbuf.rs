//! Host-side output-buffer utilities shared by the CPU-reference / parity-oracle
//! paths.
//!
//! ONE-PLACE owner for the "reset a reused output buffer so it can hold exactly
//! `target` elements without reallocating during the following fill" idiom.
//!
//! Before this existed, the idiom was hand-rolled ~20× across the `bitset` and
//! `reduce` CPU-reference helpers as
//!
//! ```ignore
//! out.clear();
//! if target > out.capacity() {
//!     out.try_reserve(target - out.capacity()).map_err(..)?;
//! }
//! ```
//!
//! which UNDER-reserves on a warm (reused) buffer: after `clear()` the length is
//! `0`, so `try_reserve(target - capacity)` only guarantees `target - capacity`
//! free slots, and the subsequent fill reallocates whenever
//! `0 < capacity < target`. Computing the reservation from the true post-clear
//! length (`0`), i.e. reserving `target` outright, makes a single fill
//! allocation-free. Centralizing it also gives the reservation logic exactly one
//! owner (ONE PLACE) instead of a byte-for-byte copy in every oracle.

use std::collections::TryReserveError;

/// Clear `buf` and ensure it can hold at least `target` elements without
/// reallocating during a subsequent single fill (`extend`/`resize`/`push`-to-`target`).
///
/// Returns the raw [`TryReserveError`] on allocation failure so each caller can
/// map it into its own domain error type and message (the historical sites each
/// attach a bespoke context string).
///
/// This deliberately reserves the FULL `target`, not `target - capacity`: after
/// the `clear()` the buffer length is `0`, so the old `- capacity` form
/// under-reserved on a warm buffer and forced an avoidable reallocation.
pub(crate) fn reserve_exact_cleared<T>(
    buf: &mut Vec<T>,
    target: usize,
) -> Result<(), TryReserveError> {
    buf.clear();
    // `buf.len() == 0` here, so `try_reserve_exact(target)` guarantees room for a
    // full `target`-element fill with no reallocation (the whole point of the fix).
    buf.try_reserve_exact(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A WARM buffer (existing capacity between `target/2` and `target`) must end
    /// up with capacity `>= target` so the following fill never reallocates. This
    /// is exactly the case the old `try_reserve(target - capacity)` form got wrong
    /// (it left capacity unchanged when `capacity >= target - capacity`).
    #[test]
    fn warm_buffer_reaches_target_capacity_without_realloc_during_fill() {
        let target = 1000usize;

        // Warm the buffer to a partial capacity strictly between target/2 and target.
        let mut buf: Vec<u32> = Vec::with_capacity(600);
        buf.extend(0..600);
        assert!(buf.capacity() >= 600 && buf.capacity() < target);

        reserve_exact_cleared(&mut buf, target).expect("reservation must succeed");

        assert_eq!(buf.len(), 0, "buffer must be cleared");
        assert!(
            buf.capacity() >= target,
            "warm buffer must reach target capacity (got {}, want >= {target})",
            buf.capacity()
        );

        // The following fill must not reallocate: capacity stays put.
        let cap_before_fill = buf.capacity();
        buf.extend(0..target as u32);
        assert_eq!(
            buf.capacity(),
            cap_before_fill,
            "a single target-sized fill must not reallocate after reserve_exact_cleared"
        );
    }

    /// A COLD buffer (no prior capacity) must also reach `>= target`.
    #[test]
    fn cold_buffer_reaches_target_capacity() {
        let mut buf: Vec<u8> = Vec::new();
        reserve_exact_cleared(&mut buf, 256).expect("reservation must succeed");
        assert_eq!(buf.len(), 0);
        assert!(buf.capacity() >= 256);
    }

    /// `target == 0` clears without demanding any allocation.
    #[test]
    fn zero_target_just_clears() {
        let mut buf: Vec<u64> = vec![1, 2, 3];
        reserve_exact_cleared(&mut buf, 0).expect("zero reservation must succeed");
        assert!(buf.is_empty());
    }
}
