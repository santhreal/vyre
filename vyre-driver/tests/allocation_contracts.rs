//! Contracts for `vyre_driver::allocation`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use smallvec::{Array, SmallVec};
use std::collections::{HashMap, HashSet, TryReserveError};
use vyre_driver::allocation::{
    reserve_hash_map_to_capacity, reserve_hash_set_to_capacity, reserve_smallvec_additional,
    reserve_smallvec_to_capacity, reserve_vec_additional, reserve_vec_to_capacity,
};

use std::collections::{HashMap, HashSet};

use smallvec::SmallVec;

use vyre_driver::allocation::{
    reserve_hash_map_to_capacity, reserve_hash_set_to_capacity, reserve_smallvec_additional,
    reserve_smallvec_to_capacity, reserve_vec_additional, reserve_vec_to_capacity,
};

#[test]
fn reserve_vec_to_capacity_grows_after_clear() {
    let mut bytes = Vec::with_capacity(16);
    bytes.extend_from_slice(&[1_u8; 12]);
    bytes.clear();

    reserve_vec_to_capacity(
        &mut bytes,
        20,
        "generated reserve test",
        "byte",
        "split generated dispatch",
    )
    .expect("Fix: reserve_vec_to_capacity should grow cleared vectors");

    assert!(bytes.capacity() >= 20);
    assert!(bytes.is_empty());
}

#[test]
fn reserve_smallvec_to_capacity_grows_after_clear() {
    let mut words = SmallVec::<[u32; 4]>::new();
    words.extend_from_slice(&[1, 2, 3, 4]);
    words.clear();

    reserve_smallvec_to_capacity(
        &mut words,
        8,
        "generated reserve test",
        "word",
        "split generated dispatch",
    )
    .expect("Fix: reserve_smallvec_to_capacity should grow cleared smallvecs");

    assert!(words.capacity() >= 8);
    assert!(words.is_empty());
}

#[test]
fn additional_reservations_preserve_length() {
    let mut bytes = vec![1_u8, 2, 3];
    reserve_vec_additional(
        &mut bytes,
        10,
        "generated reserve test",
        "byte",
        "split generated dispatch",
    )
    .expect("Fix: reserve_vec_additional should not mutate length");
    assert_eq!(bytes, vec![1, 2, 3]);

    let mut small = SmallVec::<[u8; 2]>::new();
    small.push(9);
    reserve_smallvec_additional(
        &mut small,
        10,
        "generated reserve test",
        "byte",
        "split generated dispatch",
    )
    .expect("Fix: reserve_smallvec_additional should not mutate length");
    assert_eq!(small.as_slice(), &[9]);
}

#[test]
fn hash_collection_reservations_grow_after_clear_without_reinserting() {
    let mut map = HashMap::<u32, u32>::with_capacity(4);
    let mut set = HashSet::<u32>::with_capacity(4);
    for value in 0..4 {
        map.insert(value, value * 10);
        set.insert(value);
    }
    map.clear();
    set.clear();

    for target in [8, 32, 128, 1024] {
        reserve_hash_map_to_capacity(
            &mut map,
            target,
            "generated reserve test",
            "entry",
            "split generated dispatch",
        )
        .expect("Fix: hash map target reservation should grow cleared maps");
        reserve_hash_set_to_capacity(
            &mut set,
            target,
            "generated reserve test",
            "entry",
            "split generated dispatch",
        )
        .expect("Fix: hash set target reservation should grow cleared sets");

        assert!(map.capacity() >= target);
        assert!(set.capacity() >= target);
        assert!(map.is_empty());
        assert!(set.is_empty());
    }
}
