#![cfg(feature = "device-tests")]

use super::*;

#[test]
fn optimizer_combined_upload_sequence_read_fences_once() {
    let backend = acquire();
    let dispatcher = CudaProgramDispatcher::new(&backend);
    let program = add_program("input", "out", 7);
    let input = dispatcher_lane(&dispatcher, "input");
    let output = dispatcher_lane(&dispatcher, "output");
    let input_bytes = seed_bytes();
    let handle_ids = [input, output];
    let steps = [dispatcher_step(&program, &handle_ids)];

    backend.reset_telemetry();
    let mut outputs = vec![Vec::with_capacity(64)];
    let outer_ptr = outputs.as_ptr();
    let first_slot_ptr = outputs[0].as_ptr();
    dispatcher
        .upload_resident_many_sequence_read_many_into(
            &[(input, input_bytes.as_slice())],
            &steps,
            &[output],
            &mut outputs,
        )
        .expect("Fix: CUDA optimizer combined upload/sequence/read path must succeed.");

    assert_eq!(bytes_u32(&outputs[0]), vec![8, 9, 10, 11]);
    assert_eq!(
        outputs.as_ptr(),
        outer_ptr,
        "Fix: combined resident into path must preserve caller-owned outer output slots."
    );
    assert_eq!(
        outputs[0].as_ptr(),
        first_slot_ptr,
        "Fix: combined resident into path must preserve caller-owned byte capacity."
    );
    let telemetry = backend.telemetry_snapshot();
    assert_eq!(
        telemetry.kernel_launches, 1,
        "Fix: combined resident path must record the queued kernel launch."
    );
    assert!(
        telemetry.sync_points > 0,
        "Fix: combined resident path must fence exactly once for upload + kernel + readback."
    );
    assert!(
        telemetry.host_to_device_bytes >= input_bytes.len() as u64,
        "Fix: combined resident path must include H2D upload telemetry."
    );
    assert_eq!(
        telemetry.readback_bytes,
        expected_readback_bytes(16, 48),
        "Fix: combined resident path must count the final resident readback bytes."
    );

    free_dispatcher_lanes(&dispatcher, &[(input, "input"), (output, "output")]);
}

#[test]
fn optimizer_combined_duplicate_sequence_uploads_fuse_before_kernel_launch() {
    let backend = acquire();
    let dispatcher = CudaProgramDispatcher::new(&backend);
    let program = add_program("input", "out", 7);
    let input = dispatcher_lane(&dispatcher, "input");
    let output = dispatcher_lane(&dispatcher, "output");
    let first_input = seed_bytes();
    let second_input = u32_bytes(&[10, 11, 12, 13]);
    let handle_ids = [input, output];
    let steps = [dispatcher_step(&program, &handle_ids)];

    backend.reset_telemetry();
    let mut outputs = vec![Vec::with_capacity(64)];
    dispatcher
        .upload_resident_many_sequence_read_many_into(
            &[
                (input, first_input.as_slice()),
                (input, second_input.as_slice()),
            ],
            &steps,
            &[output],
            &mut outputs,
        )
        .expect("Fix: CUDA optimizer duplicate upload sequence path must succeed.");

    assert_eq!(
        bytes_u32(&outputs[0]),
        vec![17, 18, 19, 20],
        "Fix: duplicate sequence uploads to the same handle must preserve later-write semantics before kernel launch."
    );
    let telemetry = backend.telemetry_snapshot();
    assert!(
        telemetry.host_upload_operations <= 2,
        "Fix: duplicate full resident sequence uploads must fuse before H2D; observed {} host upload operation(s).",
        telemetry.host_upload_operations
    );
    assert!(
        telemetry.host_to_device_bytes <= (first_input.len() + second_input.len()) as u64,
        "Fix: duplicate full resident sequence uploads must not copy both full payloads; observed {} H2D byte(s).",
        telemetry.host_to_device_bytes
    );

    free_dispatcher_lanes(&dispatcher, &[(input, "input"), (output, "output")]);
}

#[test]
fn optimizer_combined_duplicate_fills_keep_last_value_before_readback() {
    let backend = acquire();
    let dispatcher = CudaProgramDispatcher::new(&backend);
    let handle = dispatcher_lane(&dispatcher, "fill target");

    backend.reset_telemetry();
    let mut outputs = vec![Vec::with_capacity(64)];
    dispatcher
        .fill_upload_resident_many_sequence_read_many_into(
            &[(handle, LANE_BYTES, 0x11), (handle, LANE_BYTES, 0xA5)],
            &[],
            &[],
            &[handle],
            &mut outputs,
        )
        .expect("Fix: CUDA optimizer duplicate fill sequence path must succeed.");

    assert_eq!(
        outputs,
        vec![vec![0xA5; LANE_BYTES]],
        "Fix: duplicate sequence fills to the same handle must preserve last-fill semantics."
    );
    assert_eq!(
        backend.telemetry_snapshot().host_to_device_bytes,
        0,
        "Fix: duplicate resident fills must remain device-side memset work, not H2D uploads."
    );

    free_dispatcher_lanes(&dispatcher, &[(handle, "fill target")]);
}

#[test]
fn optimizer_combined_upload_sequence_read_ranges_compacts_d2h_bytes() {
    let backend = acquire();
    let dispatcher = CudaProgramDispatcher::new(&backend);
    let program = add_program("input", "out", 7);
    let input = dispatcher_lane(&dispatcher, "input");
    let output = dispatcher_lane(&dispatcher, "output");
    let input_bytes = seed_bytes();
    let handle_ids = [input, output];
    let steps = [dispatcher_step(&program, &handle_ids)];
    let read_ranges = [ResidentReadRange {
        handle_id: output,
        byte_offset: 4,
        byte_len: 8,
    }];

    backend.reset_telemetry();
    let mut outputs = vec![Vec::with_capacity(64)];
    let output_ptr = outputs[0].as_ptr();
    dispatcher
        .upload_resident_many_sequence_read_ranges_into(
            &[(input, input_bytes.as_slice())],
            &steps,
            &read_ranges,
            &mut outputs,
        )
        .expect("Fix: CUDA optimizer compact readback path must succeed.");

    assert_eq!(bytes_u32(&outputs[0]), vec![9, 10]);
    assert_eq!(
        outputs[0].as_ptr(),
        output_ptr,
        "Fix: compact resident readback must preserve caller-owned byte capacity."
    );
    let telemetry = backend.telemetry_snapshot();
    assert!(
        telemetry.sync_points > 0,
        "Fix: compact resident readback must keep the one-fence combined path."
    );
    assert_eq!(
        telemetry.readback_bytes,
        expected_readback_bytes(8, 40),
        "Fix: compact resident readback must transfer only requested bytes, not the full 16-byte output buffer."
    );

    free_dispatcher_lanes(&dispatcher, &[(input, "input"), (output, "output")]);
}
