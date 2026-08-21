use super::*;

#[test]
fn zero_repeat_resident_sequence_does_not_prepare_dead_repeated_steps() {
    let backend = acquire_registration();
    let add_seven = add_program("input", "tmp", 7);
    let dead_repeated = copy_program("dead_in", "dead_out");
    let input = seeded_resource_lane(&backend, "input");
    let tmp = resource_lane(&backend, "tmp");

    let prefix_resources = [input.clone(), tmp.clone()];
    // Deliberately unresolvable bindings: a zero-repeat window must never touch
    // them, and resolving them would fail before the prefix could read back.
    let invalid_repeated_resources = [Resource::default(), Resource::default()];
    let prefix_steps = [step(&add_seven, &prefix_resources)];
    let repeated_steps = [step(&dead_repeated, &invalid_repeated_resources)];
    let read_ranges = [read_range(&tmp, 0, LANE_BYTES)];
    let mut readback = Vec::new();

    backend.reset_telemetry();
    VyreBackend::dispatch_resident_repeated_sequence_read_ranges_into(
        &backend,
        &prefix_steps,
        &repeated_steps,
        0,
        &read_ranges,
        &mut [&mut readback],
    )
    .expect("Fix: CUDA zero-repeat resident sequence must not resolve or prepare repeated steps that cannot launch.");

    assert_eq!(bytes_u32(&readback), vec![8, 9, 10, 11]);
    let telemetry = backend.telemetry_snapshot();
    assert_eq!(
        telemetry.kernel_launches, 1,
        "Fix: CUDA zero-repeat resident sequence must launch only the prefix step."
    );
    assert!(
        telemetry.sync_points > 0,
        "Fix: CUDA zero-repeat resident sequence should still use one compact readback fence."
    );

    free_resource_lanes(&backend, vec![(input, "input"), (tmp, "tmp")]);
}

#[test]
fn golden_fixed_graph_replay_keeps_host_overhead_sublinear() {
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
    let mut baseline_param_upload_bytes = None;

    for repeat_count in [1_u32, 8, 64] {
        backend.reset_telemetry();
        VyreBackend::dispatch_resident_repeated_sequence_read_ranges_into(
            &backend,
            &prefix_steps,
            &repeated_steps,
            repeat_count,
            &read_ranges,
            &mut [&mut readback],
        )
        .expect("Fix: CUDA golden fixed-graph replay must execute without expanding host orchestration.");

        assert_eq!(bytes_u32(&readback), vec![16, 18, 20, 22]);
        assert_eq!(
            readback.as_ptr(),
            readback_ptr,
            "Fix: CUDA golden fixed-graph replay must preserve caller-owned readback capacity across repeat counts."
        );
        let telemetry = backend.telemetry_snapshot();
        assert_eq!(
            telemetry.kernel_launches,
            u64::from(repeat_count) + 1,
            "Fix: CUDA golden replay should launch only prefix plus required repeated device work."
        );
        assert!(
            telemetry.sync_points > 0,
            "Fix: CUDA golden replay must keep host fences constant as repeat count grows."
        );
        assert!(
            telemetry.readback_bytes <= u64::from(repeat_count + 1) * 64,
            "Fix: CUDA golden replay fallback must keep readback bytes bounded by launched work; observed {} bytes.",
            telemetry.readback_bytes
        );
        let _baseline = baseline_param_upload_bytes.get_or_insert(telemetry.param_upload_bytes);
        assert!(
            telemetry.param_upload_bytes <= u64::from(repeat_count + 1) * 128,
            "Fix: CUDA golden replay fallback must keep parameter uploads bounded by launched work; observed {} bytes.",
            telemetry.param_upload_bytes
        );
    }

    free_resource_lanes(
        &backend,
        vec![(input, "input"), (tmp, "tmp"), (output, "output")],
    );
}
