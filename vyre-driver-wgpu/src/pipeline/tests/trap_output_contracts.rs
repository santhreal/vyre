use super::*;

#[test]
fn direct_record_and_readback_trap_with_output_preserves_ring_fast_path() {
    let harness = PipelineHarness::new("trap+output readback allocation contract test");
    let with_rings_arena = harness.arena();
    let without_rings_arena = harness.arena();
    let with_rings_pool = with_rings_arena.pool().clone();
    let without_rings_pool = without_rings_arena.pool().clone();

    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![
            Node::store("out", Expr::u32(0), Expr::u32(99)),
            Node::trap(Expr::u32(9), "mixed-output-ring-trap"),
        ],
    );

    let pipeline = harness
        .compile_on_arena(&program, &with_rings_arena)
        .expect("Fix: trapped program with output compile must succeed; restore this invariant before continuing.");

    let with_rings_before = with_rings_pool.stats().allocations;
    let with_rings_error = record_once(
        &pipeline,
        &with_rings_arena,
        true,
        DispatchLabels {
            bind_group: "vyre mixed output ring test bind group",
            encoder: "vyre mixed output ring test",
            compute: "vyre mixed output ring test compute",
        },
    )
    .expect_err(
        "Fix: trapped dispatch with output and rings must still surface trap errors and not succeed",
    );
    let with_rings_after = with_rings_pool.stats().allocations;

    assert!(
        with_rings_error
            .to_string()
            .contains("wgpu dispatch trapped"),
        "Fix: expected trap dispatch to surface a backend trap error, got: {with_rings_error}"
    );
    assert!(
        with_rings_error.to_string().contains("mixed-output-ring-trap"),
        "Fix: expected trap tag to be preserved through mixed-output ring path, got: {with_rings_error}"
    );
    assert_eq!(
        with_rings_after,
        with_rings_before + 2,
        "Fix: ring-backed mixed output+trap path should add only output + trap buffer allocations from pool before first successful mapping.",
    );

    let without_rings_before = without_rings_pool.stats().allocations;
    let without_rings_error = record_once(
        &pipeline,
        &without_rings_arena,
        false,
        DispatchLabels {
            bind_group: "vyre mixed output no-ring test bind group",
            encoder: "vyre mixed output no-ring test",
            compute: "vyre mixed output no-ring test compute",
        },
    )
    .expect_err(
        "Fix: trapped dispatch without rings should surface the trap error and not succeed",
    );
    let without_rings_after = without_rings_pool.stats().allocations;

    assert!(
        without_rings_error
            .to_string()
            .contains("wgpu dispatch trapped"),
        "Fix: expected trap dispatch to surface a backend trap error, got: {without_rings_error}"
    );
    assert!(
        without_rings_error.to_string().contains("mixed-output-ring-trap"),
        "Fix: expected trap tag to be preserved through mixed-output fallback path, got: {without_rings_error}"
    );
    assert_eq!(
        without_rings_after,
        without_rings_before + 4,
        "Fix: no-ring mixed output+trap path should allocate output storage, trap storage, output readback, and trap readback buffers; ring-backed dispatch must be the path that avoids the two pooled readback allocations.",
    );
}
