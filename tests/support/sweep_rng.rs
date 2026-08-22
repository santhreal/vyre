//! The deterministic xorshift the sweep oracle matrices draw cases from.
//!
//! A sweep matrix is reproducible only while its generator is: the seed in the
//! failure message has to name one sequence. Two matrices carried a
//! byte-identical copy of this generator, so a change to either one silently
//! made the two corpora incomparable. Consumers include this file with
//! `#[path]`, the same way `tests/support/artifact_fixtures.rs` is shared.

#![allow(dead_code)]

/// Deterministic 64-bit xorshift, seeded per case.
#[derive(Clone, Copy)]
pub(crate) struct Rng(u64);

impl Rng {
    /// Start a sequence at `seed`. A zero seed is not a fixed point of this
    /// shift triple in either direction that matters here, but it emits zero
    /// forever, so it is rejected rather than silently producing a constant
    /// corpus.
    pub(crate) fn new(seed: u64) -> Self {
        assert_ne!(
            seed, 0,
            "Fix: seed the sweep generator with a non-zero value; xorshift emits only zero from a zero state."
        );
        Self(seed)
    }

    /// Next 32 bits of the sequence.
    pub(crate) fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        self.0 = x;
        (x >> 16) as u32
    }

    /// Next value in `0..upper`, or zero when `upper` is zero.
    pub(crate) fn range(&mut self, upper: u32) -> u32 {
        if upper == 0 {
            0
        } else {
            self.next_u32() % upper
        }
    }

    /// Pick one `Copy` item.
    pub(crate) fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.index(items.len())]
    }

    /// Pick one item by clone, for variant types that are not `Copy`.
    pub(crate) fn pick_cloned<T: Clone>(&mut self, items: &[T]) -> T {
        items[self.index(items.len())].clone()
    }

    /// Pick one string slice, keeping the caller's lifetime.
    pub(crate) fn pick_str<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.index(items.len())]
    }

    fn index(&mut self, len: usize) -> usize {
        assert!(
            len > 0,
            "Fix: a sweep case table must not be empty; there is nothing to pick."
        );
        let len = u32::try_from(len).expect("Fix: sweep case tables stay under u32::MAX entries.");
        self.range(len) as usize
    }
}
