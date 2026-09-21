//! Subgroup simulator for the CPU reference interpreter.
//!
//! The reference interpreter executes one invocation at a time, but Cat-C
//! subgroup ops need lane-collective semantics to stay oracle-worthy once the
//! lowering path emits backend subgroup expressions. This module provides a small,
//! deterministic simulator over a logical subgroup of lanes.

/// Deterministic CPU model of a hardware subgroup/wave.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubgroupSimulator {
    width: usize,
}

impl Default for SubgroupSimulator {
    fn default() -> Self {
        Self { width: 32 }
    }
}

impl SubgroupSimulator {
    /// Construct a simulator with a fixed subgroup width.
    #[must_use]
    pub fn new(width: usize) -> Self {
        Self {
            width: width.max(1),
        }
    }

    /// Configured subgroup width.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Encode lane predicates as a ballot bitmask.
    #[must_use]
    pub fn ballot<const N: usize>(&self, mask: &[bool; N]) -> u32 {
        self.ballot_slice(mask)
    }

    /// Encode an arbitrary lane predicate slice as a ballot bitmask.
    #[must_use]
    pub fn ballot_slice(&self, mask: &[bool]) -> u32 {
        let active = mask.len().min(self.width).min(32);
        let mut bits = 0u32;
        for (lane, &flag) in mask.iter().take(active).enumerate() {
            if flag {
                bits |= 1u32 << lane;
            }
        }
        bits
    }

    /// Permute values by source-lane indices.
    #[must_use]
    pub fn shuffle(&self, values: &[u32], src_lanes: &[u32]) -> Vec<u32> {
        let active = values.len().min(src_lanes.len()).min(self.width);
        src_lanes
            .iter()
            .take(active)
            .map(|&src| values.get(src as usize).copied().unwrap_or(0))
            .collect()
    }

    /// Wrapping sum reduction across active lanes.
    #[must_use]
    pub fn add(&self, values: &[u32]) -> u32 {
        values
            .iter()
            .take(self.width)
            .copied()
            .fold(0u32, u32::wrapping_add)
    }

    /// Bounds of the subgroup containing `lane_index` within `lane_count`.
    #[must_use]
    pub fn subgroup_bounds(&self, lane_count: usize, lane_index: usize) -> (usize, usize) {
        let start = (lane_index / self.width) * self.width;
        let end = lane_count.min(start + self.width);
        (start, end)
    }
}
