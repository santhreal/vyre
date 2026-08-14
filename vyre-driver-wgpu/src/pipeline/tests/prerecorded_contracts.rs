use super::*;

/// Pre-recording a persistent dispatch builds bind groups and records the
/// compute pass through the same code the direct persistent path uses, only
/// under its own wgpu labels. Replaying the recorded command buffer must
/// therefore land the same bytes in the output buffer that a direct dispatch
/// of the same program lands.
#[test]
fn prerecorded_replay_writes_the_same_output_as_direct_dispatch() {
    let harness = PipelineHarness::new("pre-recorded dispatch replay test");
    let arena = harness.arena();

    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(4)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
    );

    let pipeline = harness
        .compile_on_arena(&program, &arena)
        .expect("Fix: pre-recorded dispatch test pipeline must compile.");

    let direct = record_once(
        &pipeline,
        &arena,
        false,
        DispatchLabels {
            bind_group: "vyre prerecord parity direct bind group",
            encoder: "vyre prerecord parity direct",
            compute: "vyre prerecord parity direct compute",
        },
    )
    .expect("Fix: direct persistent dispatch must succeed before comparing against replay.");

    let prerecorded = pipeline
        .prerecord_borrowed_dispatch(&[], [1, 1, 1])
        .expect("Fix: pre-recording a persistent dispatch must succeed.");
    prerecorded
        .replay(&harness.device_queue.1)
        .expect("Fix: first replay of a pre-recorded command buffer must succeed.");
    let replayed = prerecorded
        .read_output(0)
        .expect("Fix: reading a replayed output buffer must succeed.");

    assert_eq!(
        u32::from_le_bytes(replayed[0..4].try_into().unwrap()),
        7,
        "Fix: replayed pre-recorded dispatch must write the program's stored value."
    );
    assert_eq!(
        replayed[0..16],
        direct[0][0..16],
        "Fix: pre-recorded replay and direct persistent dispatch must produce identical output bytes."
    );
}

/// A wgpu command buffer is single-submit. The second replay must be a
/// structured error rather than a raw wgpu panic.
#[test]
fn prerecorded_second_replay_is_a_structured_error() {
    let harness = PipelineHarness::new("pre-recorded dispatch resubmit test");
    let arena = harness.arena();

    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(3))],
    );

    let pipeline = harness
        .compile_on_arena(&program, &arena)
        .expect("Fix: pre-recorded resubmit test pipeline must compile.");
    let prerecorded = pipeline
        .prerecord_borrowed_dispatch(&[], [1, 1, 1])
        .expect("Fix: pre-recording a persistent dispatch must succeed.");

    prerecorded
        .replay(&harness.device_queue.1)
        .expect("Fix: first replay of a pre-recorded command buffer must succeed.");
    let error = prerecorded
        .replay(&harness.device_queue.1)
        .expect_err("Fix: a pre-recorded command buffer must refuse a second submission.");
    assert!(
        error.to_string().contains("already submitted"),
        "Fix: expected the single-submit diagnostic, got: {error}"
    );
}
