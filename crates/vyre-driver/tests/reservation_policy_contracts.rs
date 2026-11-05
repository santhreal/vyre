//! Contracts for `vyre_driver::reservation_policy`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use smallvec::SmallVec;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use vyre_driver::reservation_policy::{
    reserve_typed_hash_map_to_capacity, reserve_typed_hash_set_and_vec_to_capacity,
    reserve_typed_hash_set_to_capacity, reserve_typed_vec_to_capacity, reserved_typed_vec,
    ReservationPolicy, ReusableIndexScratch,
};

const TEST_POLICY: ReservationPolicy =
    ReservationPolicy::new("generated staging reserve", "split generated dispatch");

#[derive(Debug, Eq, PartialEq)]
enum TypedReserveError {
    Reserve {
        field: &'static str,
        requested: usize,
        message: String,
    },
}

fn typed_reserve_error(
    field: &'static str,
    requested: usize,
    message: String,
) -> TypedReserveError {
    TypedReserveError::Reserve {
        field,
        requested,
        message,
    }
}

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

#[test]
fn typed_policy_reservations_share_vec_set_and_map_growth() {
    let mut vec = Vec::<u8>::new();
    let mut set = HashSet::<u32>::new();
    let mut map = HashMap::<u32, u32>::new();

    reserve_typed_vec_to_capacity(TEST_POLICY, &mut vec, 32, "typed byte", typed_reserve_error)
        .expect("Fix: typed Vec reservation should grow");
    reserve_typed_hash_set_to_capacity(
        TEST_POLICY,
        &mut set,
        32,
        "typed set entry",
        typed_reserve_error,
    )
    .expect("Fix: typed HashSet reservation should grow");
    reserve_typed_hash_map_to_capacity(
        TEST_POLICY,
        &mut map,
        32,
        "typed map entry",
        typed_reserve_error,
    )
    .expect("Fix: typed HashMap reservation should grow");
    reserve_typed_hash_set_and_vec_to_capacity(
        TEST_POLICY,
        &mut set,
        &mut vec,
        64,
        "paired set entry",
        "paired byte",
        typed_reserve_error,
    )
    .expect("Fix: paired typed reservations should share one adapter");
    let reserved = reserved_typed_vec::<u16, _>(TEST_POLICY, 16, "typed word", typed_reserve_error)
        .expect("Fix: typed Vec allocation should reserve");

    assert!(vec.capacity() >= 64);
    assert!(set.capacity() >= 64);
    assert!(map.capacity() >= 32);
    assert!(reserved.capacity() >= 16);
    assert!(reserved.is_empty());
}

#[test]
fn typed_policy_reservation_reports_domain_failure_on_overflow() {
    let mut bytes = Vec::<u8>::new();
    let err = reserve_typed_vec_to_capacity(
        TEST_POLICY,
        &mut bytes,
        usize::MAX,
        "oversized typed byte",
        typed_reserve_error,
    )
    .expect_err("oversized typed reservation should fail");

    match err {
        TypedReserveError::Reserve {
            field,
            requested,
            message,
        } => {
            assert_eq!(field, "oversized typed byte");
            assert_eq!(requested, usize::MAX);
            assert!(message.contains("oversized typed byte"));
            assert!(message.contains("Fix:"));
        }
    }
}

#[test]
fn reusable_index_scratch_preserves_capacity_and_orders_only_when_needed() {
    let mut scratch = ReusableIndexScratch::<u32>::new();

    scratch
        .try_reserve_with(
            TEST_POLICY,
            64,
            "scratch seen",
            "scratch ordered",
            typed_reserve_error,
        )
        .expect("Fix: reusable scratch should reserve through shared policy");
    assert!(scratch.insert_seen(7));
    assert!(!scratch.insert_seen(7));
    scratch.push_index(0);
    scratch.push_index(1);
    scratch.push_index(2);

    let key_calls = Cell::new(0);
    scratch.sort_indices_unstable_by_key_if_needed(|index| {
        key_calls.set(key_calls.get() + 1);
        [10_u32, 20, 30][index]
    });
    assert_eq!(scratch.ordered_indices(), &[0, 1, 2]);
    assert_eq!(
        key_calls.get(),
        4,
        "Fix: monotonic planner indices must skip sort_unstable_by_key."
    );

    let seen_capacity = scratch.seen_capacity();
    let ordered_capacity = scratch.ordered_index_capacity();
    scratch.clear();
    scratch.push_index(2);
    scratch.push_index(0);
    scratch.push_index(1);
    scratch.sort_indices_unstable_by_key_if_needed(|index| [10_u32, 20, 30][index]);

    assert_eq!(scratch.ordered_indices(), &[0, 1, 2]);
    assert!(scratch.seen_capacity() >= seen_capacity);
    assert!(scratch.ordered_index_capacity() >= ordered_capacity);
}

#[test]
fn generated_reusable_index_scratch_matrix_keeps_exact_order_contract() {
    for len in 0..=96 {
        let mut scratch = ReusableIndexScratch::<usize>::new();
        scratch
            .try_reserve_with(
                TEST_POLICY,
                len,
                "generated seen",
                "generated ordered",
                typed_reserve_error,
            )
            .expect("Fix: generated scratch reservation should succeed");

        for index in (0..len).rev() {
            assert!(scratch.insert_seen(index));
            scratch.push_index(index);
        }
        scratch.sort_indices_unstable_by_key_if_needed(|index| index);

        assert_eq!(scratch.ordered_indices().len(), len);
        for (expected, actual) in scratch.ordered_indices().iter().copied().enumerate() {
            assert_eq!(actual, expected);
        }
        let seen_capacity = scratch.seen_capacity();
        let ordered_capacity = scratch.ordered_index_capacity();
        scratch.clear();
        scratch
            .try_reserve_with(
                TEST_POLICY,
                len / 2,
                "generated seen shrink",
                "generated ordered shrink",
                typed_reserve_error,
            )
            .expect("Fix: generated scratch reuse should keep retained storage");
        assert!(scratch.seen_capacity() >= seen_capacity);
        assert!(scratch.ordered_index_capacity() >= ordered_capacity);
        assert!(scratch.ordered_indices().is_empty());
    }
}
