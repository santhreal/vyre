//! Backend-neutral monotonic ordering helpers for staging hot paths.

/// Return whether an iterator's keys are already nondecreasing.
pub fn iter_is_monotonic_by_key<I, K, F>(items: I, mut key: F) -> bool
where
    I: IntoIterator,
    K: Ord,
    F: FnMut(I::Item) -> K,
{
    let mut previous = None;
    for item in items {
        let current = key(item);
        if let Some(previous) = previous {
            if current < previous {
                return false;
            }
        }
        previous = Some(current);
    }
    true
}

/// Sort only when `items` are not already nondecreasing by `key`.
pub fn sort_by_key_if_needed<T, K, F>(items: &mut [T], mut key: F)
where
    K: Ord,
    F: FnMut(&T) -> K,
{
    let mut previous = None;
    for index in 0..items.len() {
        let current = key(&items[index]);
        if let Some(previous) = previous {
            if current < previous {
                items.sort_by_key(key);
                return;
            }
        }
        previous = Some(current);
    }
}

/// Unstable-sort only when `items` are not already nondecreasing by `key`.
pub fn sort_unstable_by_key_if_needed<T, K, F>(items: &mut [T], mut key: F)
where
    K: Ord,
    F: FnMut(&T) -> K,
{
    let mut previous = None;
    for index in 0..items.len() {
        let current = key(&items[index]);
        if let Some(previous) = previous {
            if current < previous {
                items.sort_unstable_by_key(key);
                return;
            }
        }
        previous = Some(current);
    }
}

/// Unstable-sort only when `items` are not already nondecreasing.
pub fn sort_unstable_if_needed<T>(items: &mut [T])
where
    T: Ord,
{
    for index in 1..items.len() {
        if items[index] < items[index - 1] {
            items.sort_unstable();
            return;
        }
    }
}

/// The first way a sorted index slice fails to be a dense permutation of
/// `0..expected_len`. Distinguishing these lets callers emit a remediation that
/// names the actual defect (a duplicate aliases two descriptors onto one logical
/// slot; a sparse map skips one; a length mismatch has the wrong cardinality)
/// instead of a generic "not dense".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DensePermutationDefect {
    /// After sorting, `index` sits at `slot` with `index < slot`: a value
    /// repeated earlier, so two descriptors alias one logical slot.
    Duplicate {
        /// The repeated value found below its sorted slot position.
        index: usize,
        /// The sorted slot position at which the duplicate surfaced.
        slot: usize,
    },
    /// After sorting, `index` sits at `slot` with `index > slot`: a gap, so a
    /// logical slot in `0..expected_len` is never mapped.
    Sparse {
        /// The value found above its sorted slot position.
        index: usize,
        /// The sorted slot position whose dense value (`slot`) is missing.
        slot: usize,
    },
    /// Every present index was dense but the cardinality is wrong (the map is
    /// truncated or over-long relative to `expected_len`).
    LengthMismatch {
        /// The number of indices actually present.
        resolved: usize,
        /// The dense cardinality the map was required to cover.
        expected: usize,
    },
}

/// Classify whether `sorted_indices` is a dense permutation of `0..expected_len`
///: each value in `0..expected_len` present exactly once.
///
/// Callers MUST pass indices already sorted ascending (e.g. via
/// [`sort_unstable_if_needed`]); the classification is defined on sorted slot
/// position. This is the single source of the dense-index-map invariant shared
/// by every resident/graph descriptor→logical-slot map; format the returned
/// [`DensePermutationDefect`] into a context-specific message at the call site.
pub fn classify_dense_permutation(
    sorted_indices: &[usize],
    expected_len: usize,
) -> Result<(), DensePermutationDefect> {
    for (slot, &index) in sorted_indices.iter().enumerate() {
        if index != slot {
            return Err(if index < slot {
                DensePermutationDefect::Duplicate { index, slot }
            } else {
                DensePermutationDefect::Sparse { index, slot }
            });
        }
    }
    if sorted_indices.len() != expected_len {
        return Err(DensePermutationDefect::LengthMismatch {
            resolved: sorted_indices.len(),
            expected: expected_len,
        });
    }
    Ok(())
}
