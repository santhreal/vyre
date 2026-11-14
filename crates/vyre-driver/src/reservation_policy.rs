//! Backend-neutral reservation policy adapters.
//!
//! Concrete backends own their wording, but hot dispatch paths should share one
//! reservation policy for Vec, SmallVec, hash collections, and output slots.

use std::collections::hash_map::RandomState;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hash};

use smallvec::{Array, SmallVec};

use crate::BackendError;

/// Domain wording for a family of bounded reservations.
#[derive(Clone, Copy, Debug)]
pub struct ReservationPolicy {
    context: &'static str,
    fix: &'static str,
}

impl ReservationPolicy {
    /// Create a reservation policy with a stable error context and fix.
    #[must_use]
    pub const fn new(context: &'static str, fix: &'static str) -> Self {
        Self { context, fix }
    }

    /// Ensure a Vec reaches `target_capacity` without changing length.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the Vec cannot reserve memory.
    pub fn reserve_vec_to_capacity<T>(
        self,
        vec: &mut Vec<T>,
        target_capacity: usize,
        item: &'static str,
    ) -> Result<(), BackendError> {
        crate::allocation::reserve_vec_to_capacity(
            vec,
            target_capacity,
            self.context,
            item,
            self.fix,
        )
    }

    /// Allocate an empty Vec with `target_capacity` reserved.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the Vec cannot reserve memory.
    pub fn reserved_vec<T>(
        self,
        target_capacity: usize,
        item: &'static str,
    ) -> Result<Vec<T>, BackendError> {
        let mut vec = Vec::new();
        self.reserve_vec_to_capacity(&mut vec, target_capacity, item)?;
        Ok(vec)
    }

    /// Reserve `additional` more Vec elements without changing length.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the Vec cannot reserve memory.
    pub fn reserve_vec_additional<T>(
        self,
        vec: &mut Vec<T>,
        additional: usize,
        item: &'static str,
    ) -> Result<(), BackendError> {
        crate::allocation::reserve_vec_additional(vec, additional, self.context, item, self.fix)
    }

    /// Reserve enough Vec storage for `target_len` elements without resizing.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the Vec cannot reserve memory.
    pub fn reserve_vec_exact_for_len<T>(
        self,
        vec: &mut Vec<T>,
        target_len: usize,
        item: &'static str,
    ) -> Result<(), BackendError> {
        crate::output_slots::reserve_vec_exact_for_len(
            vec,
            target_len,
            self.context,
            item,
            self.fix,
        )
    }

    /// Ensure a Vec of output slots has at least `slot_count` slots.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the outer Vec cannot reserve memory.
    pub fn ensure_vec_slots_at_least<T>(
        self,
        slots: &mut Vec<Vec<T>>,
        slot_count: usize,
        item: &'static str,
    ) -> Result<(), BackendError> {
        crate::output_slots::ensure_vec_slots_at_least(
            slots,
            slot_count,
            self.context,
            item,
            self.fix,
        )
    }

    /// Resize a Vec of output slots while preserving existing prefixes.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the outer Vec cannot reserve memory.
    pub fn resize_vec_slots<T>(
        self,
        slots: &mut Vec<Vec<T>>,
        slot_count: usize,
        item: &'static str,
    ) -> Result<(), BackendError> {
        crate::output_slots::resize_vec_slots(slots, slot_count, self.context, item, self.fix)
    }

    /// Clear inner output buffers without changing slot count.
    pub fn clear_vec_slots<T>(slots: &mut [Vec<T>]) {
        crate::output_slots::clear_vec_slots(slots);
    }

    /// Ensure a SmallVec reaches `target_capacity` without changing length.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the SmallVec cannot reserve memory.
    pub fn reserve_smallvec_to_capacity<A>(
        self,
        vec: &mut SmallVec<A>,
        target_capacity: usize,
        item: &'static str,
    ) -> Result<(), BackendError>
    where
        A: Array,
    {
        let additional = target_capacity.saturating_sub(vec.len());
        self.reserve_smallvec_additional(vec, additional, item)
    }

    /// Reserve `additional` more SmallVec elements without changing length.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the SmallVec cannot reserve memory.
    pub fn reserve_smallvec_additional<A>(
        self,
        vec: &mut SmallVec<A>,
        additional: usize,
        item: &'static str,
    ) -> Result<(), BackendError>
    where
        A: Array,
    {
        crate::allocation::reserve_smallvec_additional(
            vec,
            additional,
            self.context,
            item,
            self.fix,
        )
    }

    /// Ensure a HashSet reaches `target_capacity` without changing length.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the HashSet cannot reserve memory.
    pub fn reserve_hash_set_to_capacity<T, S>(
        self,
        set: &mut HashSet<T, S>,
        target_capacity: usize,
        item: &'static str,
    ) -> Result<(), BackendError>
    where
        T: Eq + Hash,
        S: BuildHasher,
    {
        crate::allocation::reserve_hash_set_to_capacity(
            set,
            target_capacity,
            self.context,
            item,
            self.fix,
        )
    }

    /// Ensure a HashMap reaches `target_capacity` without changing length.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the HashMap cannot reserve memory.
    pub fn reserve_hash_map_to_capacity<K, V, S>(
        self,
        map: &mut HashMap<K, V, S>,
        target_capacity: usize,
        item: &'static str,
    ) -> Result<(), BackendError>
    where
        K: Eq + Hash,
        S: BuildHasher,
    {
        crate::allocation::reserve_hash_map_to_capacity(
            map,
            target_capacity,
            self.context,
            item,
            self.fix,
        )
    }
}

/// Convert a shared reservation failure into a caller-domain error.
pub type StagingReservationFailureAdapter<E> = fn(&'static str, usize, String) -> E;

/// Declare a planner's storage-reservation failure adapter.
///
/// Every planner in this crate reserves its scratch and its result vectors
/// before it decides anything, and reports a failure in its own error type so a
/// caller keeps one error to match on. The conversion carries the same three
/// facts every time: which field was being reserved, how many entries it wanted,
/// and what the allocator said. Six planners wrote that function out, so a
/// fourth fact added here had to be threaded through six identical copies, and a
/// planner that was missed would report a reservation failure with less context
/// than the shared layer already had.
///
/// `$error` must be an enum with a `StorageReserveFailed { field, requested,
/// message }` variant; the message it renders stays that planner's own, because
/// it names the planner and the sharding that fixes it.
macro_rules! storage_reserve_failure_adapter {
    ($error:ident) => {
        fn storage_reserve_failed(
            field: &'static str,
            requested: usize,
            message: String,
        ) -> $error {
            $error::StorageReserveFailed {
                field,
                requested,
                message,
            }
        }
    };
}

pub(crate) use storage_reserve_failure_adapter;

/// Reserve Vec capacity and map failures into a caller-domain typed error.
///
/// # Errors
///
/// Returns `E` when the Vec cannot reserve memory.
pub fn reserve_typed_vec_to_capacity<T, E>(
    policy: ReservationPolicy,
    vec: &mut Vec<T>,
    target_capacity: usize,
    item: &'static str,
    failure: StagingReservationFailureAdapter<E>,
) -> Result<(), E> {
    policy
        .reserve_vec_to_capacity(vec, target_capacity, item)
        .map_err(|error| failure(item, target_capacity, error.to_string()))
}

/// Allocate an empty Vec with reserved capacity and typed failure mapping.
///
/// # Errors
///
/// Returns `E` when the Vec cannot reserve memory.
pub fn reserved_typed_vec<T, E>(
    policy: ReservationPolicy,
    target_capacity: usize,
    item: &'static str,
    failure: StagingReservationFailureAdapter<E>,
) -> Result<Vec<T>, E> {
    let mut vec = Vec::new();
    reserve_typed_vec_to_capacity(policy, &mut vec, target_capacity, item, failure)?;
    Ok(vec)
}

/// Reserve HashSet capacity and map failures into a caller-domain typed error.
///
/// # Errors
///
/// Returns `E` when the HashSet cannot reserve memory.
pub fn reserve_typed_hash_set_to_capacity<T, S, E>(
    policy: ReservationPolicy,
    set: &mut HashSet<T, S>,
    target_capacity: usize,
    item: &'static str,
    failure: StagingReservationFailureAdapter<E>,
) -> Result<(), E>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    policy
        .reserve_hash_set_to_capacity(set, target_capacity, item)
        .map_err(|error| failure(item, target_capacity, error.to_string()))
}

/// Reserve HashMap capacity and map failures into a caller-domain typed error.
///
/// # Errors
///
/// Returns `E` when the HashMap cannot reserve memory.
pub fn reserve_typed_hash_map_to_capacity<K, V, S, E>(
    policy: ReservationPolicy,
    map: &mut HashMap<K, V, S>,
    target_capacity: usize,
    item: &'static str,
    failure: StagingReservationFailureAdapter<E>,
) -> Result<(), E>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    policy
        .reserve_hash_map_to_capacity(map, target_capacity, item)
        .map_err(|error| failure(item, target_capacity, error.to_string()))
}

/// Reserve paired duplicate-detection and stable-order buffers with one typed failure adapter.
///
/// # Errors
///
/// Returns `E` when either staging collection cannot reserve memory.
pub fn reserve_typed_hash_set_and_vec_to_capacity<K, V, S, E>(
    policy: ReservationPolicy,
    set: &mut HashSet<K, S>,
    vec: &mut Vec<V>,
    target_capacity: usize,
    set_item: &'static str,
    vec_item: &'static str,
    failure: StagingReservationFailureAdapter<E>,
) -> Result<(), E>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    reserve_typed_hash_set_to_capacity(policy, set, target_capacity, set_item, failure)?;
    reserve_typed_vec_to_capacity(policy, vec, target_capacity, vec_item, failure)
}

/// Reusable duplicate-detection plus stable-order scratch for planner hot paths.
pub struct ReusableIndexScratch<K, S = RandomState> {
    seen: HashSet<K, S>,
    ordered_indices: Vec<usize>,
}

impl<K, S> std::fmt::Debug for ReusableIndexScratch<K, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReusableIndexScratch")
            .field("seen_capacity", &self.seen.capacity())
            .field("ordered_index_capacity", &self.ordered_indices.capacity())
            .finish()
    }
}

impl<K, S> Default for ReusableIndexScratch<K, S>
where
    S: Default,
{
    fn default() -> Self {
        Self {
            seen: HashSet::with_hasher(S::default()),
            ordered_indices: Vec::new(),
        }
    }
}

impl<K, S> ReusableIndexScratch<K, S>
where
    K: Eq + Hash,
    S: BuildHasher + Default,
{
    /// Create empty reusable index scratch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear retained scratch entries without releasing retained capacity.
    pub fn clear(&mut self) {
        self.seen.clear();
        self.ordered_indices.clear();
    }

    /// Reserve duplicate-detection and ordering scratch to the requested capacity.
    ///
    /// # Errors
    ///
    /// Returns `E` when either retained scratch collection cannot reserve memory.
    pub fn try_reserve_with<E>(
        &mut self,
        policy: ReservationPolicy,
        capacity: usize,
        seen_item: &'static str,
        ordered_indices_item: &'static str,
        failure: StagingReservationFailureAdapter<E>,
    ) -> Result<(), E> {
        reserve_typed_hash_set_and_vec_to_capacity(
            policy,
            &mut self.seen,
            &mut self.ordered_indices,
            capacity,
            seen_item,
            ordered_indices_item,
            failure,
        )
    }

    /// Insert a duplicate-detection key.
    pub fn insert_seen(&mut self, key: K) -> bool {
        self.seen.insert(key)
    }

    /// Append an input index to the reusable ordering buffer.
    pub fn push_index(&mut self, index: usize) {
        self.ordered_indices.push(index);
    }

    /// Mutable ordering buffer for planner-specific sort keys.
    pub fn ordered_indices_mut(&mut self) -> &mut Vec<usize> {
        &mut self.ordered_indices
    }

    /// Sort ordered indices only when the current key order is not already monotonic.
    pub fn sort_indices_unstable_by_key_if_needed<Key, F>(&mut self, mut key: F)
    where
        Key: Ord,
        F: FnMut(usize) -> Key,
    {
        let needs_sort = self
            .ordered_indices
            .windows(2)
            .any(|pair| key(pair[0]) > key(pair[1]));
        if needs_sort {
            self.ordered_indices
                .sort_unstable_by_key(|&index| key(index));
        }
    }

    /// Ordered input indices after planner-specific sorting.
    #[must_use]
    pub fn ordered_indices(&self) -> &[usize] {
        &self.ordered_indices
    }

    /// Retained duplicate-detection capacity.
    #[must_use]
    pub fn seen_capacity(&self) -> usize {
        self.seen.capacity()
    }

    /// Retained ordering capacity.
    #[must_use]
    pub fn ordered_index_capacity(&self) -> usize {
        self.ordered_indices.capacity()
    }
}
