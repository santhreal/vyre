//! The value-range result type.
//!
//! Owns [`IntRange`], the inclusive integer interval the analysis derives,
//! and [`ValueRangeReport`], the per-result-id map plus the staleness
//! positions a consumer must respect before trusting a range. It derives
//! nothing; the walk that fills it lives in `analysis`.

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

/// Inclusive integer range. Represented as i64 internally so it can
/// hold both U32 and I32 bounds without overflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IntRange {
    /// Inclusive lower bound.
    pub min: i64,
    /// Inclusive upper bound.
    pub max: i64,
}

impl IntRange {
    /// Singleton range `[v, v]`.
    pub fn singleton(v: i64) -> Self {
        Self { min: v, max: v }
    }

    /// True iff this range contains exactly one value.
    pub fn is_singleton(&self) -> bool {
        self.min == self.max
    }

    /// Inclusive containment.
    pub fn contains(&self, v: i64) -> bool {
        v >= self.min && v <= self.max
    }

    /// Union of two ranges (smallest range that contains both).
    /// Useful for joining branch arms or Min/Max alternatives.
    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

/// Statically derived integer ranges for one descriptor body.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ValueRangeReport {
    /// `result_id → IntRange` for ids in the TOP-LEVEL body whose
    /// range is statically derivable. Per-body id space  -  child
    /// bodies are walked separately via `analyze_body` if callers
    /// need it.
    pub ranges: FxHashMap<u32, IntRange>,
    /// `result_id → op index` for ids whose value is a snapshot of a
    /// named carrier slot that a later op can overwrite.
    ///
    /// The entry is the index of the first op in this body's `ops` from
    /// which the id must be treated as **unknown**, even though `ranges`
    /// still holds a derived range for it. Reads strictly before that
    /// index observe the snapshot and are sound; reads at or after it may
    /// observe a value written by a `LoopCarrierEnd` in a nested body.
    ///
    /// Consult this through [`ValueRangeReport::get_at`] rather than
    /// reading `ranges` directly whenever the answer feeds a decision that
    /// changes program semantics (branch collapse in particular). It
    /// cannot be folded into `ranges`: the same id is legitimately known
    /// before the mutating construct and unknown after it, so collapsing
    /// the two would either lose a sound optimization or keep an unsound
    /// one. See the internal `carrier_snapshot_invalidations` analysis.
    pub invalidated_from: FxHashMap<u32, usize>,
}

impl ValueRangeReport {
    /// Return the number of result identifiers with known ranges.
    pub fn known_count(&self) -> usize {
        self.ranges.len()
    }

    /// Range for `result_id`, or `None` if not known.
    pub fn get(&self, result_id: u32) -> Option<IntRange> {
        self.ranges.get(&result_id).copied()
    }

    /// Range for `result_id` **as observed by the op at `op_index`**, or
    /// `None` when it is not known there.
    ///
    /// This is [`Self::get`] plus the carrier-snapshot check. Any rewrite
    /// whose decision changes program semantics MUST use this form: `get`
    /// answers "was a range ever derived for this id", which is a strictly
    /// weaker question than "does that range hold at the point I am about
    /// to act on it". A variable mutated inside a nested body has both
    /// answers, and they differ.
    #[must_use]
    pub fn get_at(&self, result_id: u32, op_index: usize) -> Option<IntRange> {
        if let Some(&invalid_from) = self.invalidated_from.get(&result_id) {
            if op_index >= invalid_from {
                return None;
            }
        }
        self.ranges.get(&result_id).copied()
    }

    /// True iff the value at `result_id` is provably equal to `target`.
    /// Returns `None` if the range isn't known (caller may want to
    /// treat that differently from a known-unequal).
    pub fn is_definitely(&self, result_id: u32, target: i64) -> Option<bool> {
        self.ranges
            .get(&result_id)
            .map(|r| r.is_singleton() && r.min == target)
    }

    /// True iff every value in `result_id`'s range is `< target`.
    /// `None` if range unknown.
    pub fn is_definitely_below(&self, result_id: u32, target: i64) -> Option<bool> {
        self.ranges.get(&result_id).map(|r| r.max < target)
    }

    /// True iff every value in `result_id`'s range is `>= target`.
    /// `None` if range unknown.
    pub fn is_definitely_at_least(&self, result_id: u32, target: i64) -> Option<bool> {
        self.ranges.get(&result_id).map(|r| r.min >= target)
    }

    /// If the range for `result_id` is a singleton, return that value.
    /// Useful for downstream rewrites that want to know "is this id
    /// known to be exactly some constant?". Returns `None` for both
    /// "range unknown" and "range non-singleton".
    pub fn as_constant(&self, result_id: u32) -> Option<i64> {
        self.ranges
            .get(&result_id)
            .filter(|r| r.is_singleton())
            .map(|r| r.min)
    }
}
