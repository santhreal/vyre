#![cfg(feature = "device-tests")]

use super::*;

#[test]
fn backend_sequence_read_ranges_runs_dependent_steps_with_one_fence() {
    let backend = acquire_registration();
    let add_seven = add_program("input", "tmp", 7);
    let double = mul_program("tmp", "out", 2);
    let input = seeded_resource_lane(&backend, "input");
    let tmp = resource_lane(&backend, "tmp");
    let output = resource_lane(&backend, "output");

    let first_resources = [input.clone(), tmp.clone()];
    let second_resources = [tmp.clone(), output.clone()];
    // The first step repeats deliberately: two launches of the same program over
    // the same bindings must both run, so `tmp` is 1+7+7 before `double`.
    let steps = [
        step(&add_seven, &first_resources),
        step(&add_seven, &first_resources),
        step(&double, &second_resources),
    ];
    let read_ranges = [read_range(&output, 4, 8)];
    let mut compact = Vec::with_capacity(64);
    let compact_ptr = compact.as_ptr();

    backend.reset_telemetry();
    VyreBackend::dispatch_resident_sequence_read_ranges_into(
        &backend,
        &steps,
        &read_ranges,
        &mut [&mut compact],
    )
    .expect("Fix: CUDA backend resident sequence-read API must execute dependent kernels.");

    assert_eq!(bytes_u32(&compact), vec![18, 20]);
    assert_eq!(
        compact.as_ptr(),
        compact_ptr,
        "Fix: CUDA backend resident sequence-read API must preserve caller-owned output capacity."
    );
    let telemetry = backend.telemetry_snapshot();
    assert_eq!(
        telemetry.kernel_launches, 3,
        "Fix: CUDA backend resident sequence-read API must launch every dependent sequence step."
    );
    assert!(telemetry.sync_points > 0, "Fix: CUDA backend resident sequence-read API must fence once for the whole dependent window plus readback.");
    assert_eq!(
        telemetry.readback_bytes,
        expected_readback_bytes(8, 104),
        "Fix: CUDA backend resident sequence-read API must compact readback to the requested byte range."
    );
    assert!(
        telemetry.param_upload_bytes <= 128,
        "Fix: CUDA backend resident sequence-read API must hoist duplicate launch parameter blocks instead of uploading parameters once per sequence step; observed {} bytes.",
        telemetry.param_upload_bytes
    );

    free_resource_lanes(
        &backend,
        vec![(input, "input"), (tmp, "tmp"), (output, "output")],
    );
}

#[test]
fn backend_sequence_read_ranges_coalesces_duplicate_d2h_copies() {
    let backend = acquire_registration();
    let add_seven = add_program("input", "out", 7);
    let input = seeded_resource_lane(&backend, "input");
    let output = resource_lane(&backend, "output");

    let resources = [input.clone(), output.clone()];
    let steps = [step(&add_seven, &resources)];
    let read_ranges = (0..16)
        .map(|_| read_range(&output, 4, 8))
        .collect::<Vec<_>>();
    let mut outputs = (0..16).map(|_| Vec::with_capacity(64)).collect::<Vec<_>>();
    let output_ptrs = outputs.iter().map(Vec::as_ptr).collect::<Vec<_>>();

    backend.reset_telemetry();
    {
        let mut output_refs = outputs.iter_mut().collect::<Vec<_>>();
        VyreBackend::dispatch_resident_sequence_read_ranges_into(
            &backend,
            &steps,
            &read_ranges,
            &mut output_refs,
        )
        .expect("Fix: CUDA backend resident sequence-read API must coalesce duplicate readbacks without losing output slots.");
    }

    for (index, output) in outputs.iter().enumerate() {
        assert_eq!(bytes_u32(output), vec![9, 10]);
        assert_eq!(
            output.as_ptr(),
            output_ptrs[index],
            "Fix: duplicate compact readback must preserve caller-owned byte capacity for output slot {index}."
        );
    }
    assert_native_compact_readback(
        &backend.telemetry_snapshot(),
        8,
        "coalesce duplicate ranges",
    );

    free_resource_lanes(&backend, vec![(input, "input"), (output, "output")]);
}

#[test]
fn backend_sequence_read_ranges_fuses_overlapping_and_adjacent_d2h_intervals() {
    let backend = acquire_registration();
    let add_seven = add_program("input", "out", 7);
    let input = seeded_resource_lane(&backend, "input");
    let output = resource_lane(&backend, "output");

    let resources = [input.clone(), output.clone()];
    let steps = [step(&add_seven, &resources)];
    // Overlapping (0..8, 4..12) plus adjacent (12..16): one fused 16-byte copy.
    let read_ranges = [
        read_range(&output, 0, 8),
        read_range(&output, 4, 8),
        read_range(&output, 12, 4),
    ];
    let mut first = Vec::with_capacity(64);
    let mut second = Vec::with_capacity(64);
    let mut third = Vec::with_capacity(64);

    backend.reset_telemetry();
    VyreBackend::dispatch_resident_sequence_read_ranges_into(
        &backend,
        &steps,
        &read_ranges,
        &mut [&mut first, &mut second, &mut third],
    )
    .expect("Fix: CUDA backend resident sequence-read API must fuse overlapping and adjacent readbacks without changing caller output ordering.");

    assert_eq!(bytes_u32(&first), vec![8, 9]);
    assert_eq!(bytes_u32(&second), vec![9, 10]);
    assert_eq!(bytes_u32(&third), vec![11]);
    assert_native_compact_readback(
        &backend.telemetry_snapshot(),
        16,
        "fuse overlapping and adjacent ranges",
    );

    free_resource_lanes(&backend, vec![(input, "input"), (output, "output")]);
}

#[test]
fn backend_repeated_sequence_read_ranges_runs_without_expanded_host_window() {
    let backend = acquire_registration();
    let add_seven = add_program("input", "tmp", 7);
    let double = mul_program("tmp", "out", 2);
    let input = seeded_resource_lane(&backend, "input");
    let tmp = resource_lane(&backend, "tmp");
    let output = resource_lane(&backend, "output");

    let prefix_resources = [input.clone(), tmp.clone()];
    let repeated_resources = [tmp.clone(), output.clone()];
    let prefix_steps = [step(&add_seven, &prefix_resources)];
    let repeated_steps = [step(&double, &repeated_resources)];
    let read_ranges = [read_range(&output, 0, LANE_BYTES)];
    let mut readback = Vec::with_capacity(64);
    let readback_ptr = readback.as_ptr();

    backend.reset_telemetry();
    VyreBackend::dispatch_resident_repeated_sequence_read_ranges_into(
        &backend,
        &prefix_steps,
        &repeated_steps,
        4,
        &read_ranges,
        &mut [&mut readback],
    )
    .expect("Fix: CUDA backend repeated resident sequence-read API must execute without materializing an expanded caller sequence.");

    assert_eq!(bytes_u32(&readback), vec![16, 18, 20, 22]);
    assert_eq!(
        readback.as_ptr(),
        readback_ptr,
        "Fix: CUDA repeated resident sequence-read API must preserve caller-owned output capacity."
    );
    let telemetry = backend.telemetry_snapshot();
    assert_eq!(
        telemetry.kernel_launches, 5,
        "Fix: CUDA repeated resident sequence-read API must launch prefix plus every repeated step."
    );
    assert!(telemetry.sync_points > 0, "Fix: CUDA repeated resident sequence-read API must fence once for the whole repeated window plus readback.");
    assert_eq!(
        telemetry.readback_bytes,
        expected_readback_bytes(16, 176),
        "Fix: CUDA repeated resident sequence-read API must compact readback to the requested byte range."
    );
    assert!(
        telemetry.param_upload_bytes <= 128,
        "Fix: CUDA repeated resident sequence-read API must hoist repeated launch parameter blocks instead of uploading parameters once per repeated step; observed {} bytes.",
        telemetry.param_upload_bytes
    );

    free_resource_lanes(
        &backend,
        vec![(input, "input"), (tmp, "tmp"), (output, "output")],
    );
}
