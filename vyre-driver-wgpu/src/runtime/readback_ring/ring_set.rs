//! Size-classed collection of rings, one class per staging capacity.

use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use rustc_hash::FxHasher;
use std::hash::BuildHasherDefault;
use std::sync::Arc;
use vyre_driver::BackendError;

use super::capacity::{
    readback_ring_slots_from_env, readback_ring_slots_from_raw, ring_capacity_class,
};
use super::ring::ReadbackRing;

/// Size-classed collection of readback rings for direct dispatch.
pub struct ReadbackRingSet {
    rings: DashMap<u64, Arc<ReadbackRing>, BuildHasherDefault<FxHasher>>,
    slots_per_ring: usize,
}

impl Default for ReadbackRingSet {
    fn default() -> Self {
        Self::new()
    }
}

impl ReadbackRingSet {
    /// Construct an empty ring set using the default slot count.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rings: DashMap::with_hasher(BuildHasherDefault::<FxHasher>::default()),
            slots_per_ring: readback_ring_slots_from_env(),
        }
    }

    /// Construct an empty ring set from a raw slot-count setting.
    ///
    /// Passing `None` uses the production default. This keeps test and embedded
    /// callers off process-global environment mutation while preserving the same
    /// parser and clamping semantics as [`Self::new`].
    #[must_use]
    pub fn with_requested_slots(raw_slots: Option<&str>) -> Self {
        Self {
            rings: DashMap::with_hasher(BuildHasherDefault::<FxHasher>::default()),
            slots_per_ring: readback_ring_slots_from_raw(raw_slots),
        }
    }

    /// Return the ring whose staging slots can hold `byte_len`.
    ///
    /// # Errors
    ///
    /// Returns a backend error if the requested byte length overflows wgpu copy
    /// alignment.
    pub fn ring_for(
        &self,
        device: &wgpu::Device,
        byte_len: u64,
    ) -> Result<Arc<ReadbackRing>, BackendError> {
        let capacity = Self::capacity_class_for(byte_len)?;
        self.ring_for_capacity(device, capacity)
    }

    /// Return a ring for an already-normalized capacity class.
    #[inline]
    pub(crate) fn ring_for_capacity(
        &self,
        device: &wgpu::Device,
        capacity: u64,
    ) -> Result<Arc<ReadbackRing>, BackendError> {
        Ok(match self.rings.entry(capacity) {
            Entry::Occupied(entry) => Arc::clone(entry.get()),
            Entry::Vacant(entry) => {
                let ring = Arc::new(ReadbackRing::new(device, self.slots_per_ring, capacity)?);
                entry.insert(Arc::clone(&ring));
                ring
            }
        })
    }

    /// Convert an arbitrary byte length to the ring capacity class used for
    /// ring sizing.
    #[inline]
    pub(crate) fn capacity_class(byte_len: u64) -> Result<u64, BackendError> {
        Self::capacity_class_for(byte_len)
    }

    /// Convert an arbitrary byte length to the ring capacity class used for
    /// ring sizing.
    #[inline]
    pub(crate) fn capacity_class_for(byte_len: u64) -> Result<u64, BackendError> {
        ring_capacity_class(byte_len)
    }

    /// Return an existing size-classed ring without taking exclusive access.
    ///
    /// # Errors
    ///
    /// Returns a backend error if the requested byte length overflows wgpu copy
    /// alignment.
    pub fn existing_ring_for(
        &self,
        byte_len: u64,
    ) -> Result<Option<Arc<ReadbackRing>>, BackendError> {
        let capacity = Self::capacity_class(byte_len)?;
        Ok(self.existing_ring_for_capacity(capacity))
    }

    /// Return an existing size-classed ring without taking exclusive access.
    #[inline]
    pub(crate) fn existing_ring_for_capacity(&self, capacity: u64) -> Option<Arc<ReadbackRing>> {
        self.rings
            .get(&capacity)
            .map(|ring| Arc::clone(ring.value()))
    }

    /// Number of slots configured for each runtime ring instance.
    #[must_use]
    pub fn slots_per_ring(&self) -> usize {
        self.slots_per_ring
    }
}
