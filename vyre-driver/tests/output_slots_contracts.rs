//! Contracts for `vyre_driver::output_slots`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::output_slots::{
    clear_vec_slots, ensure_vec_slots_at_least, resize_vec_slots, resize_vec_with,
};

#[test]
fn resize_vec_with_preserves_prefix_and_initializes_new_slots() {
    for case in 0..4096 {
        let initial_len = case % 17;
        let target_len = (case * 7 + 3) % 23;
        let mut slots = Vec::new();
        slots
            .try_reserve(initial_len)
            .expect("Fix: generated resize test must reserve initial slots");
        for idx in 0..initial_len {
            slots.push(vec![idx as u8; (idx % 5) + 1]);
        }
        let expected_prefix: Vec<Vec<u8>> = slots.iter().take(target_len).cloned().collect();

        resize_vec_with(
            &mut slots,
            target_len,
            Vec::new,
            "generated output slots",
            "slot",
            "split generated dispatch",
        )
        .expect("Fix: generated output slot resize should be fallible but successful");

        assert_eq!(
            slots.len(),
            target_len,
            "generated resize case {case} must match target length"
        );
        assert_eq!(
            &slots[..expected_prefix.len()],
            expected_prefix.as_slice(),
            "generated resize case {case} must preserve existing output slots"
        );
        for slot in slots.iter().skip(initial_len.min(target_len)) {
            assert!(
                slot.is_empty(),
                "generated resize case {case} must initialize new output slots as empty Vecs"
            );
        }
    }
}

#[test]
fn vec_slot_helpers_can_grow_truncate_and_clear() {
    let mut slots = vec![vec![1_u8], vec![2, 3]];
    ensure_vec_slots_at_least(
        &mut slots,
        4,
        "generated slots",
        "slot",
        "split generated dispatch",
    )
    .expect("Fix: slot growth should reserve successfully");
    assert_eq!(slots.len(), 4);
    assert_eq!(slots[0], vec![1]);
    assert_eq!(slots[1], vec![2, 3]);
    assert!(slots[2].is_empty());
    assert!(slots[3].is_empty());

    resize_vec_slots(
        &mut slots,
        1,
        "generated slots",
        "slot",
        "split generated dispatch",
    )
    .expect("Fix: slot truncation should not allocate");
    assert_eq!(slots, vec![vec![1]]);

    clear_vec_slots(&mut slots);
    assert_eq!(slots, vec![Vec::<u8>::new()]);
}
