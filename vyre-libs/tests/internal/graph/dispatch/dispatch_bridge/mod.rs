use super::*;

#[test]
fn program_cache_reuses_same_key_and_rebuilds_on_shape_change() {
    let mut cache = ProgramCache::default();

    assert_eq!(*cache.get_or_insert_with(7_u32, || 11_u32), 11);
    assert_eq!(*cache.get_or_insert_with(7_u32, || 99_u32), 11);
    assert_eq!(cache.builds(), 1);

    assert_eq!(*cache.get_or_insert_with(8_u32, || 22_u32), 22);
    assert_eq!(cache.builds(), 2);
}

#[test]
fn dispatch_input_writer_encodes_zero_and_little_endian_slots_in_place() {
    let mut slot = Vec::with_capacity(32);

    super::inputs::write_dispatch_input(
        &mut slot,
        DispatchInput::u32_slice_or_zero_words(&[], 3, "zero words"),
    )
    .expect("Fix: zero-word dispatch input should encode");
    assert_eq!(slot, vec![0; 12]);

    super::inputs::write_dispatch_input(&mut slot, DispatchInput::u32_slice(&[1, 0xAABB_CCDD]))
        .expect("Fix: u32 dispatch input should encode little-endian bytes");
    assert_eq!(slot, vec![1, 0, 0, 0, 0xDD, 0xCC, 0xBB, 0xAA]);
}

#[test]
fn input_slot_shell_drops_stale_dispatch_slots_on_shape_shrink() {
    let mut inputs = vec![vec![0xAA; 4], vec![0xBB; 8], vec![0xCC; 12], vec![0xDD; 16]];

    crate::dispatch_buffers::ensure_input_slots(&mut inputs, 2);

    assert_eq!(inputs.len(), 2);
    assert_eq!(inputs[0], vec![0xAA; 4]);
    assert_eq!(inputs[1], vec![0xBB; 8]);
}

#[test]
fn prepared_dispatch_inputs_never_forward_stale_slots_after_shrink() {
    let mut inputs = vec![vec![0xAA; 4], vec![0xBB; 8], vec![0xCC; 12], vec![0xDD; 16]];

    super::inputs::prepare_dispatch_inputs(&mut inputs, &[DispatchInput::u32_slice(&[9])])
        .expect("Fix: prepared dispatch input shrink should encode");

    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0], vec![9, 0, 0, 0]);
}

#[test]
fn u32_slice_fingerprint_tracks_width_order_and_content() {
    let base = fingerprint_u32_slice(&[1, 2, 3, 4]);

    assert_eq!(base, fingerprint_u32_slice(&[1, 2, 3, 4]));
    assert_ne!(base, fingerprint_u32_slice(&[4, 3, 2, 1]));
    assert_ne!(base, fingerprint_u32_slice(&[1, 2, 3, 4, 0]));
    assert_ne!(base, fingerprint_u32_slice(&[1, 2, 3, 5]));
}

#[test]
fn keyed_dispatch_refresh_reuses_static_slots_and_updates_mutable_slots() {
    let mut inputs = Vec::new();
    let mut key = None;

    refresh_keyed_dispatch_inputs(
        &mut inputs,
        &mut key,
        7_u32,
        &[
            DispatchInput::u32_slice(&[10, 11]),
            DispatchInput::u32_slice(&[20]),
            DispatchInput::zero_u32_words(2, "mutable out"),
        ],
        &[(2, DispatchInput::zero_u32_words(2, "mutable out"))],
    )
    .expect("Fix: first keyed dispatch refresh should stage every slot");
    assert_eq!(inputs[0], vec![10, 0, 0, 0, 11, 0, 0, 0]);
    assert_eq!(inputs[1], vec![20, 0, 0, 0]);
    inputs[2].fill(0xA5);

    refresh_keyed_dispatch_inputs(
        &mut inputs,
        &mut key,
        7_u32,
        &[
            DispatchInput::u32_slice(&[99, 100]),
            DispatchInput::u32_slice(&[88]),
            DispatchInput::zero_u32_words(2, "mutable out"),
        ],
        &[(2, DispatchInput::zero_u32_words(2, "mutable out"))],
    )
    .expect("Fix: same-key refresh should only rewrite mutable slots");
    assert_eq!(inputs[0], vec![10, 0, 0, 0, 11, 0, 0, 0]);
    assert_eq!(inputs[1], vec![20, 0, 0, 0]);
    assert_eq!(inputs[2], vec![0; 8]);

    refresh_keyed_dispatch_inputs(
        &mut inputs,
        &mut key,
        8_u32,
        &[
            DispatchInput::u32_slice(&[99, 100]),
            DispatchInput::u32_slice(&[88]),
            DispatchInput::zero_u32_words(2, "mutable out"),
        ],
        &[(2, DispatchInput::zero_u32_words(2, "mutable out"))],
    )
    .expect("Fix: changed key should restage every slot");
    assert_eq!(inputs[0], vec![99, 0, 0, 0, 100, 0, 0, 0]);
    assert_eq!(inputs[1], vec![88, 0, 0, 0]);
    assert_eq!(inputs[2], vec![0; 8]);
}
