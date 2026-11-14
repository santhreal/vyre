//! Backend-neutral reservation policy adapters.
//!
//! Concrete backends own their wording, but hot dispatch paths should share one
//! reservation policy for Vec, SmallVec, hash collections, and output slots.

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
