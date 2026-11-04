//! Substrate-neutral fallible target-capacity reservation helpers.
//!
//! These helpers are deliberately below driver/runtime/self-substrate crates so
//! hot paths can share the same capacity arithmetic without creating dependency
//! cycles or backend coupling. Domain crates still own their error wording.

use std::collections::{BinaryHeap, HashMap, HashSet, TryReserveError};
use std::hash::{BuildHasher, Hash};

use smallvec::{Array, SmallVec};

/// Ensure a [`Vec`] can hold `target_capacity` items without changing length.
///
/// # Errors
///
/// Returns [`TryReserveError`] when allocation fails.
pub fn try_reserve_vec_to_capacity<T>(
    vec: &mut Vec<T>,
    target_capacity: usize,
) -> Result<(), TryReserveError> {
    if target_capacity > vec.capacity() {
        vec.try_reserve_exact(target_capacity - vec.len())?;
    }
    Ok(())
}

/// Clear `buf` and ensure it can hold at least `target` elements without
/// reallocating during a subsequent single fill (`extend`/`resize`/`push`-to-`target`).
///
/// ONE-PLACE owner for the "reset a reused output buffer so it can hold exactly
/// `target` elements without reallocating during the following fill" idiom.
///
/// Before this existed, the idiom was hand-rolled across CPU-reference oracles,
/// driver readback paths, and wire decoders as
///
/// ```ignore
/// out.clear();
/// if target > out.capacity() {
///     out.try_reserve(target - out.capacity()).map_err(..)?;
/// }
/// ```
///
/// which UNDER-reserves on a warm (reused) buffer: after `clear()` the length is
/// `0`, so `try_reserve(target - capacity)` only guarantees `target - capacity`
/// free slots, and the subsequent fill reallocates whenever
/// `0 < capacity < target`. Computing the reservation from the true post-clear
/// length (`0`), i.e. reserving `target` outright, makes a single fill
/// allocation-free.
///
/// Use [`try_reserve_vec_to_capacity`] instead when the buffer must keep its
/// current contents; this function is only for the clear-then-refill shape.
///
/// # Errors
///
/// Returns the raw [`TryReserveError`] on allocation failure so each caller can
/// map it into its own domain error type and message (the historical sites each
/// attach a bespoke context string).
pub fn reserve_exact_cleared<T>(buf: &mut Vec<T>, target: usize) -> Result<(), TryReserveError> {
    buf.clear();
    // `buf.len() == 0` here, so `try_reserve_exact(target)` guarantees room for a
    // full `target`-element fill with no reallocation (the whole point of the fix).
    buf.try_reserve_exact(target)
}

/// Ensure a [`String`] can hold `target_capacity` bytes without changing
/// length.
///
/// # Errors
///
/// Returns [`TryReserveError`] when allocation fails.
pub fn try_reserve_string_to_capacity(
    string: &mut String,
    target_capacity: usize,
) -> Result<(), TryReserveError> {
    if target_capacity > string.capacity() {
        string.try_reserve(target_capacity - string.len())?;
    }
    Ok(())
}

/// Ensure a [`SmallVec`] can hold `target_capacity` items without changing
/// length.
///
/// # Errors
///
/// Returns [`TryReserveError`] when allocation fails.
pub fn try_reserve_smallvec_to_capacity<A>(
    vec: &mut SmallVec<A>,
    target_capacity: usize,
) -> Result<(), smallvec::CollectionAllocErr>
where
    A: Array,
{
    if target_capacity > vec.capacity() {
        vec.try_reserve(target_capacity - vec.len())?;
    }
    Ok(())
}

/// Ensure a [`HashMap`] can hold `target_capacity` entries without changing
/// length.
///
/// # Errors
///
/// Returns [`TryReserveError`] when allocation fails.
pub fn try_reserve_hash_map_to_capacity<K, V, S>(
    map: &mut HashMap<K, V, S>,
    target_capacity: usize,
) -> Result<(), TryReserveError>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    if target_capacity > map.capacity() {
        map.try_reserve(target_capacity - map.len())?;
    }
    Ok(())
}

/// Ensure a [`HashSet`] can hold `target_capacity` entries without changing
/// length.
///
/// # Errors
///
/// Returns [`TryReserveError`] when allocation fails.
pub fn try_reserve_hash_set_to_capacity<T, S>(
    set: &mut HashSet<T, S>,
    target_capacity: usize,
) -> Result<(), TryReserveError>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    if target_capacity > set.capacity() {
        set.try_reserve(target_capacity - set.len())?;
    }
    Ok(())
}

/// Ensure a [`BinaryHeap`] can hold `target_capacity` entries without changing
/// length.
///
/// # Errors
///
/// Returns [`TryReserveError`] when allocation fails.
pub fn try_reserve_binary_heap_to_capacity<T>(
    heap: &mut BinaryHeap<T>,
    target_capacity: usize,
) -> Result<(), TryReserveError> {
    if target_capacity > heap.capacity() {
        heap.try_reserve(target_capacity - heap.len())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::{BinaryHeap, HashMap, HashSet};

    use smallvec::SmallVec;

    use super::{
        reserve_exact_cleared, try_reserve_binary_heap_to_capacity,
        try_reserve_hash_map_to_capacity, try_reserve_hash_set_to_capacity,
        try_reserve_smallvec_to_capacity, try_reserve_string_to_capacity,
        try_reserve_vec_to_capacity,
    };

    /// CLASS GATE. Both `Vec` choke points every migrated reserve site routes
    /// through must guarantee `capacity >= target` from ANY warm starting
    /// capacity, and the single fill that follows must not reallocate.
    ///
    /// The defect class is `try_reserve*(target - capacity())`: with a warm
    /// buffer it asks for less than the fill needs and silently leaves capacity
    /// short. One representative case would not catch it, because the wrong
    /// form is accidentally sufficient whenever `capacity <= target / 2`. This
    /// sweeps every warm capacity from `0` to `target` against both owners, so
    /// any owner that derives `additional` from `capacity()` goes red here.
    ///
    /// What it does not catch: allocator-failure paths, non-`Vec` collections
    /// (covered by the sibling target-capacity test), and callers that reserve
    /// correctly and then fill past `target`.
    #[test]
    fn vec_reservation_owners_reach_target_from_every_warm_capacity() {
        const TARGET: usize = 512;

        for warm in 0..=TARGET {
            // Shape A: clear-then-refill sites (`reserve_exact_cleared`).
            let mut cleared: Vec<u32> = Vec::with_capacity(warm);
            cleared.extend(0..warm as u32);
            reserve_exact_cleared(&mut cleared, TARGET)
                .expect("Fix: cleared-buffer reservation must succeed for a 512-word target");
            assert!(
                cleared.is_empty(),
                "Fix: reserve_exact_cleared must clear the buffer (warm capacity {warm})"
            );
            assert!(
                cleared.capacity() >= TARGET,
                "Fix: reserve_exact_cleared under-reserved from warm capacity {warm} (got {}, want >= {TARGET})",
                cleared.capacity()
            );
            let cleared_capacity = cleared.capacity();
            cleared.extend(0..TARGET as u32);
            assert_eq!(
                cleared.capacity(),
                cleared_capacity,
                "Fix: the fill after reserve_exact_cleared reallocated from warm capacity {warm}"
            );

            // Shape B: contents-preserving sites (`try_reserve_vec_to_capacity`).
            let keep_len = warm.min(TARGET / 4);
            let mut retained: Vec<u32> = Vec::with_capacity(warm);
            retained.extend(0..keep_len as u32);
            try_reserve_vec_to_capacity(&mut retained, TARGET)
                .expect("Fix: retained-buffer reservation must succeed for a 512-word target");
            assert_eq!(
                retained.len(),
                keep_len,
                "Fix: try_reserve_vec_to_capacity must not change length (warm capacity {warm})"
            );
            assert!(
                retained.capacity() >= TARGET,
                "Fix: try_reserve_vec_to_capacity under-reserved from warm capacity {warm} (got {}, want >= {TARGET})",
                retained.capacity()
            );
            let retained_capacity = retained.capacity();
            retained.clear();
            retained.extend(0..TARGET as u32);
            assert_eq!(
                retained.capacity(),
                retained_capacity,
                "Fix: the fill after try_reserve_vec_to_capacity reallocated from warm capacity {warm}"
            );
        }
    }

    /// A WARM buffer (existing capacity between `target/2` and `target`) must end
    /// up with capacity `>= target` so the following fill never reallocates. This
    /// is exactly the case the old `try_reserve(target - capacity)` form got wrong
    /// (it left capacity unchanged when `capacity >= target - capacity`).
    #[test]
    fn warm_buffer_reaches_target_capacity_without_realloc_during_fill() {
        let target = 1000usize;

        // Warm the buffer to a partial capacity strictly between target/2 and target.
        let mut buf: Vec<u32> = Vec::with_capacity(600);
        buf.extend(0..600);
        assert!(buf.capacity() >= 600 && buf.capacity() < target);

        reserve_exact_cleared(&mut buf, target).expect("reservation must succeed");

        assert_eq!(buf.len(), 0, "buffer must be cleared");
        assert!(
            buf.capacity() >= target,
            "warm buffer must reach target capacity (got {}, want >= {target})",
            buf.capacity()
        );

        // The following fill must not reallocate: capacity stays put.
        let cap_before_fill = buf.capacity();
        buf.extend(0..target as u32);
        assert_eq!(
            buf.capacity(),
            cap_before_fill,
            "a single target-sized fill must not reallocate after reserve_exact_cleared"
        );
    }

    /// A COLD buffer (no prior capacity) must also reach `>= target`.
    #[test]
    fn cold_buffer_reaches_target_capacity() {
        let mut buf: Vec<u8> = Vec::new();
        reserve_exact_cleared(&mut buf, 256).expect("reservation must succeed");
        assert_eq!(buf.len(), 0);
        assert!(buf.capacity() >= 256);
    }

    /// `target == 0` clears without demanding any allocation.
    #[test]
    fn zero_target_just_clears() {
        let mut buf: Vec<u64> = vec![1, 2, 3];
        reserve_exact_cleared(&mut buf, 0).expect("zero reservation must succeed");
        assert!(buf.is_empty());
    }

    #[test]
    fn target_capacity_helpers_grow_after_clear_without_mutating_lengths() {
        let mut vec = Vec::<u32>::with_capacity(4);
        let mut small = SmallVec::<[u32; 2]>::new();
        let mut string = String::with_capacity(4);
        let mut map = HashMap::<u32, u32>::with_capacity(4);
        let mut set = HashSet::<u32>::with_capacity(4);
        let mut heap = BinaryHeap::<u32>::with_capacity(4);
        for value in 0..4 {
            vec.push(value);
            small.push(value);
            string.push(char::from(b'a' + value as u8));
            map.insert(value, value * 10);
            set.insert(value);
            heap.push(value);
        }
        vec.clear();
        small.clear();
        string.clear();
        map.clear();
        set.clear();
        heap.clear();

        for target in [8, 32, 128, 1024] {
            try_reserve_vec_to_capacity(&mut vec, target)
                .expect("Fix: foundation Vec target reservation must grow cleared Vecs");
            try_reserve_smallvec_to_capacity(&mut small, target)
                .expect("Fix: foundation SmallVec target reservation must grow cleared SmallVecs");
            try_reserve_string_to_capacity(&mut string, target)
                .expect("Fix: foundation String target reservation must grow cleared strings");
            try_reserve_hash_map_to_capacity(&mut map, target)
                .expect("Fix: foundation HashMap target reservation must grow cleared maps");
            try_reserve_hash_set_to_capacity(&mut set, target)
                .expect("Fix: foundation HashSet target reservation must grow cleared sets");
            try_reserve_binary_heap_to_capacity(&mut heap, target)
                .expect("Fix: foundation BinaryHeap target reservation must grow cleared heaps");

            assert!(vec.capacity() >= target);
            assert!(small.capacity() >= target);
            assert!(string.capacity() >= target);
            assert!(map.capacity() >= target);
            assert!(set.capacity() >= target);
            assert!(heap.capacity() >= target);
            assert!(vec.is_empty());
            assert!(small.is_empty());
            assert!(string.is_empty());
            assert!(map.is_empty());
            assert!(set.is_empty());
            assert!(heap.is_empty());
        }
    }

    #[test]
    fn target_capacity_helpers_reject_usize_max_without_mutating_lengths() {
        let mut vec = vec![1u8, 2, 3];
        let mut small = SmallVec::<[u8; 2]>::from_slice(&[1, 2, 3]);
        let mut string = String::from("abc");
        let mut map = HashMap::<u8, u8>::new();
        let mut set = HashSet::<u8>::new();
        let mut heap = BinaryHeap::<u8>::from([3, 1, 2]);

        assert!(try_reserve_vec_to_capacity(&mut vec, usize::MAX).is_err());
        assert!(try_reserve_smallvec_to_capacity(&mut small, usize::MAX).is_err());
        assert!(try_reserve_string_to_capacity(&mut string, usize::MAX).is_err());
        assert!(try_reserve_hash_map_to_capacity(&mut map, usize::MAX).is_err());
        assert!(try_reserve_hash_set_to_capacity(&mut set, usize::MAX).is_err());
        assert!(try_reserve_binary_heap_to_capacity(&mut heap, usize::MAX).is_err());
        assert_eq!(vec, vec![1, 2, 3]);
        assert_eq!(small.as_slice(), &[1, 2, 3]);
        assert_eq!(string, "abc");
        assert!(map.is_empty());
        assert!(set.is_empty());
        assert_eq!(heap.len(), 3);
    }
}
