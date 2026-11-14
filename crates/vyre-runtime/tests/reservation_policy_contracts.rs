//! Contracts for `vyre_runtime::reservation_policy`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
use vyre_runtime::reservation_policy::ReservationPolicy;

const TEST_POLICY: ReservationPolicy =
    ReservationPolicy::new("generated staging reserve", "split generated dispatch");

#[test]
fn policy_reserves_vec_smallvec_and_hash_collections_to_target_capacity() {
    let mut vec = Vec::<u8>::with_capacity(4);
    let mut small = SmallVec::<[u8; 2]>::new();
    let mut map = HashMap::<u32, u32>::with_capacity(4);
    let mut set = HashSet::<u32>::with_capacity(4);

    vec.extend_from_slice(&[1, 2, 3, 4]);
    small.extend_from_slice(&[1, 2, 3, 4]);
    for value in 0..4 {
        map.insert(value, value);
        set.insert(value);
    }
    vec.clear();
    small.clear();
    map.clear();
    set.clear();

    TEST_POLICY
        .reserve_vec_to_capacity(&mut vec, 32, "byte")
        .expect("Fix: Vec target reservation should grow");
    TEST_POLICY
        .reserve_smallvec_to_capacity(&mut small, 32, "byte")
        .expect("Fix: SmallVec target reservation should grow");
    TEST_POLICY
        .reserve_hash_map_to_capacity(&mut map, 32, "entry")
        .expect("Fix: HashMap target reservation should grow");
    TEST_POLICY
        .reserve_hash_set_to_capacity(&mut set, 32, "entry")
        .expect("Fix: HashSet target reservation should grow");

    assert!(vec.capacity() >= 32);
    assert!(small.capacity() >= 32);
    assert!(map.capacity() >= 32);
    assert!(set.capacity() >= 32);
    assert!(vec.is_empty());
    assert!(small.is_empty());
    assert!(map.is_empty());
    assert!(set.is_empty());
}

#[test]
fn policy_manages_output_slot_vectors_without_dropping_live_prefixes() {
    let mut slots = vec![vec![1_u8], vec![2, 3]];

    TEST_POLICY
        .ensure_vec_slots_at_least(&mut slots, 4, "slot")
        .expect("Fix: slot reservation should grow");
    assert_eq!(slots.len(), 4);
    assert_eq!(slots[0], vec![1]);
    assert_eq!(slots[1], vec![2, 3]);

    TEST_POLICY
        .resize_vec_slots(&mut slots, 1, "slot")
        .expect("Fix: slot resize should truncate without allocation");
    assert_eq!(slots, vec![vec![1]]);

    ReservationPolicy::clear_vec_slots(&mut slots);
    assert_eq!(slots, vec![Vec::<u8>::new()]);
}
