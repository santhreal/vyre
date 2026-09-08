//! Shared fallible staging reservations for CUDA backend hot paths.

use std::hash::Hash;

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::{Array, SmallVec};
use vyre_driver::{
    reservation_policy::{
        reserve_typed_hash_map_to_capacity, reserve_typed_vec_to_capacity,
        reserved_typed_vec as driver_reserved_typed_vec, ReservationPolicy,
    },
    BackendError,
};

const CUDA_STAGING: ReservationPolicy = ReservationPolicy::new(
    "CUDA backend staging",
    "split the dispatch batch or lower CUDA staging fan-out before retrying",
);

pub(crate) fn reserve_vec<T>(
    vec: &mut Vec<T>,
    capacity: usize,
    field: &'static str,
) -> Result<(), BackendError> {
    CUDA_STAGING.reserve_vec_to_capacity(vec, capacity, field)
}

pub(crate) fn reserved_vec<T>(
    capacity: usize,
    field: &'static str,
) -> Result<Vec<T>, BackendError> {
    let mut vec = Vec::new();
    reserve_vec(&mut vec, capacity, field)?;
    Ok(vec)
}

pub(crate) fn resize_vec_slots<T>(
    slots: &mut Vec<Vec<T>>,
    slot_count: usize,
    field: &'static str,
) -> Result<(), BackendError> {
    CUDA_STAGING.resize_vec_slots(slots, slot_count, field)
}

pub(crate) fn clear_vec_slots<T>(slots: &mut [Vec<T>]) {
    ReservationPolicy::clear_vec_slots(slots);
}

pub(crate) fn reserve_smallvec<A>(
    vec: &mut SmallVec<A>,
    capacity: usize,
    field: &'static str,
) -> Result<(), BackendError>
where
    A: Array,
{
    CUDA_STAGING.reserve_smallvec_to_capacity(vec, capacity, field)
}

pub(crate) fn reserve_hash_set<T>(
    set: &mut FxHashSet<T>,
    capacity: usize,
    field: &'static str,
) -> Result<(), BackendError>
where
    T: Eq + Hash,
{
    CUDA_STAGING.reserve_hash_set_to_capacity(set, capacity, field)
}

pub(crate) fn reserve_hash_map<K, V>(
    map: &mut FxHashMap<K, V>,
    capacity: usize,
    field: &'static str,
) -> Result<(), BackendError>
where
    K: Eq + Hash,
{
    CUDA_STAGING.reserve_hash_map_to_capacity(map, capacity, field)
}

/// Domain error adapter for CUDA planners that use typed reservation failures.
pub(crate) trait CudaStorageReserveFailure: Sized {
    /// Build the planner-specific error for a failed staging reservation.
    fn storage_reserve_failed(field: &'static str, requested: usize, message: String) -> Self;
}

pub(crate) fn reserve_typed_vec<T, E>(
    vec: &mut Vec<T>,
    capacity: usize,
    field: &'static str,
) -> Result<(), E>
where
    E: CudaStorageReserveFailure,
{
    reserve_typed_vec_to_capacity(
        CUDA_STAGING,
        vec,
        capacity,
        field,
        E::storage_reserve_failed,
    )
}

pub(crate) fn reserved_typed_vec<T, E>(capacity: usize, field: &'static str) -> Result<Vec<T>, E>
where
    E: CudaStorageReserveFailure,
{
    driver_reserved_typed_vec(CUDA_STAGING, capacity, field, E::storage_reserve_failed)
}

pub(crate) fn reserve_typed_hash_map<K, V, E>(
    map: &mut FxHashMap<K, V>,
    capacity: usize,
    field: &'static str,
) -> Result<(), E>
where
    K: Eq + Hash,
    E: CudaStorageReserveFailure,
{
    reserve_typed_hash_map_to_capacity(
        CUDA_STAGING,
        map,
        capacity,
        field,
        E::storage_reserve_failed,
    )
}

// Inline: covers `CudaStorageReserveFailure`, `reserve_smallvec`, `reserve_typed_hash_map`,
// `reserve_typed_vec` and 3 more items this module keeps private, which no integration test can
// name.
#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap;
    use smallvec::SmallVec;

    use super::{
        reserve_smallvec, reserve_typed_hash_map, reserve_typed_vec, reserve_vec, resize_vec_slots,
        CudaStorageReserveFailure,
    };

    #[derive(Debug, Eq, PartialEq)]
    enum TypedReserveError {
        Reserve {
            field: &'static str,
            requested: usize,
            message: String,
        },
    }

    impl CudaStorageReserveFailure for TypedReserveError {
        fn storage_reserve_failed(field: &'static str, requested: usize, message: String) -> Self {
            Self::Reserve {
                field,
                requested,
                message,
            }
        }
    }

    #[test]
    fn reserve_vec_grows_to_target_capacity_after_clear() {
        let mut bytes = Vec::with_capacity(16);
        bytes.extend_from_slice(&[1_u8; 12]);
        bytes.clear();

        reserve_vec(&mut bytes, 20, "test bytes").unwrap();

        assert!(
            bytes.capacity() >= 20,
            "Fix: reserve_vec must request target_capacity - len, not target_capacity - current_capacity."
        );
    }

    #[test]
    fn reserve_smallvec_grows_to_target_capacity_after_clear() {
        let mut words = SmallVec::<[u32; 4]>::new();
        words.extend_from_slice(&[1, 2, 3, 4]);
        words.clear();

        reserve_smallvec(&mut words, 8, "test words").unwrap();

        assert!(
            words.capacity() >= 8,
            "Fix: reserve_smallvec must request target_capacity - len, not target_capacity - current_capacity."
        );
    }

    #[test]
    fn typed_cuda_reservations_share_vec_and_map_growth() {
        let mut bytes = Vec::<u8>::new();
        let mut map = FxHashMap::<u32, u32>::default();

        reserve_typed_vec::<_, TypedReserveError>(&mut bytes, 32, "typed bytes").unwrap();
        reserve_typed_hash_map::<_, _, TypedReserveError>(&mut map, 32, "typed map").unwrap();

        assert!(bytes.capacity() >= 32);
        assert!(map.capacity() >= 32);
    }

    #[test]
    fn resize_vec_slots_grows_and_truncates_through_shared_policy() {
        let mut slots = Vec::<Vec<u8>>::with_capacity(4);
        slots.push(vec![1, 2, 3]);
        let outer_ptr = slots.as_ptr();

        resize_vec_slots(&mut slots, 3, "cuda replay outputs").unwrap();
        assert_eq!(slots.len(), 3);
        assert_eq!(slots[0], vec![1, 2, 3]);
        assert!(slots[1].is_empty());
        assert!(slots[2].is_empty());
        assert_eq!(slots.as_ptr(), outer_ptr);

        resize_vec_slots(&mut slots, 1, "cuda replay outputs").unwrap();
        assert_eq!(slots, vec![vec![1, 2, 3]]);
        assert_eq!(slots.as_ptr(), outer_ptr);
    }
}
