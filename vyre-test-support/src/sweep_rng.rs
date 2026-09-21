//! The deterministic generators sweep matrices and generated-case suites draw
//! cases from.
//!
//! A sweep matrix is reproducible only while its generator is: the seed in the
//! failure message has to name one sequence. Two matrices carried a
//! byte-identical copy of this generator, so a change to either one silently
//! made the two corpora incomparable. One owner in this crate is what keeps the
//! seed in a failure message naming the same sequence in every consumer.

/// Deterministic 64-bit xorshift, seeded per case.
#[derive(Clone, Copy)]
pub struct Rng(u64);

impl Rng {
    /// Start a sequence at `seed`. A zero seed is not a fixed point of this
    /// shift triple in either direction that matters here, but it emits zero
    /// forever, so it is rejected rather than silently producing a constant
    /// corpus.
    pub fn new(seed: u64) -> Self {
        assert_ne!(
            seed, 0,
            "Fix: seed the sweep generator with a non-zero value; xorshift emits only zero from a zero state."
        );
        Self(seed)
    }

    /// Next 32 bits of the sequence.
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        self.0 = x;
        (x >> 16) as u32
    }

    /// Next value in `0..upper`, or zero when `upper` is zero.
    pub fn range(&mut self, upper: u32) -> u32 {
        if upper == 0 {
            0
        } else {
            self.next_u32() % upper
        }
    }

    /// Pick one `Copy` item.
    pub fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.index(items.len())]
    }

    /// Pick one item by clone, for variant types that are not `Copy`.
    pub fn pick_cloned<T: Clone>(&mut self, items: &[T]) -> T {
        items[self.index(items.len())].clone()
    }

    /// Pick one string slice, keeping the caller's lifetime.
    pub fn pick_str<'a>(&mut self, items: &[&'a str]) -> &'a str {
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

/// Advance a 64-bit linear congruential state and return it.
///
/// The multiplier and increment are the MMIX constants. Generated-case suites
/// in several crates carried a byte-identical copy of this step, and a copy
/// that drifts makes the `case_index` in two failure messages name different
/// inputs while both suites claim to sweep the same space.
pub fn next_case_u64(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}
