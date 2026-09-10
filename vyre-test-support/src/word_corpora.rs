//! Deterministic `u32` word corpora for wire and buffer tests.
//!
//! Eleven test trees carried a `tests/wire_words/mod.rs` holding these
//! generators, textually included into one or more targets each. The copies had
//! drifted: `lcg_u32` took `(count, seed)` in six of them and `(seed, len)` in
//! four, and two different recurrences answered to that one name, one stepping a
//! 64-bit state and one a 32-bit state mixed with the loop index. `ramp` added
//! one per element in some copies and the golden-ratio constant in others. A
//! corpus is only a corpus while its seed names one sequence, so each recurrence
//! is named for what it computes and there is one definition of each.
//!
//! Nothing here reads IR or a device, so the module carries no feature gate and
//! a consumer of it links no compiler crate.

/// The 64-bit linear congruential stream, stepped one value at a time.
///
/// A caller that draws a variable number of values, or draws bounded indices
/// while walking a structure, holds the state; [`pcg_words`] is the fixed-length
/// form over the same sequence.
pub struct Lcg(pub u64);

impl Lcg {
    /// Start the sequence at `seed`.
    ///
    /// Every state is reachable and none is absorbing, zero included, so no seed
    /// is rejected.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Step the state and return the high bits of the new one.
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    /// Next value in `0..n`, or zero when `n` is zero.
    pub fn below(&mut self, n: u32) -> u32 {
        if n == 0 { 0 } else { self.next_u32() % n }
    }
}

/// `count` values of the 64-bit stream [`Lcg`] produces from `seed`.
#[must_use]
pub fn pcg_words(seed: u64, count: usize) -> Vec<u32> {
    let mut rng = Lcg::new(seed);
    (0..count).map(|_| rng.next_u32()).collect()
}

/// `count` values of the 32-bit stream, each mixed with its own index.
///
/// A distinct sequence from [`pcg_words`], not a narrower spelling of it: the
/// index term means the value at position `i` depends on `i` as well as on the
/// state, which is what makes a truncated corpus differ from a prefix of a
/// longer one.
#[must_use]
pub fn lcg32_words(seed: u32, count: usize) -> Vec<u32> {
    let mut state = seed;
    (0..count)
        .map(|idx| {
            state = state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223)
                .wrapping_add(idx as u32);
            state
        })
        .collect()
}

/// `count` values from `start`, advancing by `step` each element.
///
/// `step` is a parameter because the copies disagreed on it: one added one per
/// element and produced a dense run, another added `0x9E37_79B9` and produced a
/// sequence that covers the word range. Both are wanted, and a default would
/// have silently changed whichever call site did not state it.
#[must_use]
pub fn ramp_words(count: usize, start: u32, step: u32) -> Vec<u32> {
    (0..count)
        .map(|idx| start.wrapping_add((idx as u32).wrapping_mul(step)))
        .collect()
}

/// `count` values alternating `even` at even indices and `odd` at odd ones.
#[must_use]
pub fn alternating_words(count: usize, even: u32, odd: u32) -> Vec<u32> {
    (0..count)
        .map(|idx| if idx % 2 == 0 { even } else { odd })
        .collect()
}
