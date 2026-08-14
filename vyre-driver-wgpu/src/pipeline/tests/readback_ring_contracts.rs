use super::*;

#[test]
fn direct_record_and_readback_reuses_bind_groups() {
    let harness = PipelineHarness::new("direct cache test");
    let arena = harness.arena();

    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(4)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
    );

    let pipeline = harness
        .compile_on_arena(&program, &arena)
        .expect("Fix: compile must succeed; restore this invariant before continuing.");

    for _ in 0..2 {
        let outputs = record_once(
            &pipeline,
            &arena,
            false,
            DispatchLabels {
                bind_group: "vyre direct cache test bind group",
                encoder: "vyre direct cache test",
                compute: "vyre direct cache test compute",
            },
        )
        .expect(
            "Fix: direct record/readback must succeed; restore this invariant before continuing.",
        );
        assert_eq!(u32::from_le_bytes(outputs[0][0..4].try_into().unwrap()), 7);
    }

    let stats = pipeline.bind_group_cache_stats();
    // The pool may or may not return the same buffer Arc across two
    // back-to-back readbacks (the prior readback's pinning, plus
    // size-class bucketing, decides). What we DO require: the cache
    // is exercised on every dispatch (misses + hits >= 2) and never
    // reports a negative-cost path (no double-build for a given Arc).
    let total = stats.misses + stats.hits;
    assert!(
        total >= 2,
        "two dispatches should each consult the bind-group cache (got misses={}, hits={})",
        stats.misses,
        stats.hits,
    );
    assert!(
        stats.misses <= 2,
        "no more than one bind-group build per distinct buffer identity (got misses={})",
        stats.misses,
    );
}

#[test]
fn direct_record_and_readback_trap_uses_readback_rings_only() {
    let harness = PipelineHarness::new("trap-sidecar allocation test");
    let with_rings_arena = harness.arena();
    let with_rings_pool = with_rings_arena.pool().clone();

    let program = Program::wrapped(
        vec![],
        [1, 1, 1],
        vec![Node::trap(Expr::u32(3), "direct-readback-ring-trap")],
    );

    let pipeline = harness
        .compile_on_arena(&program, &with_rings_arena)
        .expect(
            "Fix: trapped program compile must succeed; restore this invariant before continuing.",
        );

    let before_allocations = with_rings_pool.stats().allocations;
    let error = record_once(
        &pipeline,
        &with_rings_arena,
        true,
        DispatchLabels {
            bind_group: "vyre direct trap readback ring test bind group",
            encoder: "vyre direct trap readback ring test",
            compute: "vyre direct trap readback ring test compute",
        },
    )
    .expect_err(
        "Fix: trapped dispatch with readback rings must return the underlying trap sidecar error and not succeed",
    );
    let after_allocations = with_rings_pool.stats().allocations;

    assert!(
        error.to_string().contains("wgpu dispatch trapped"),
        "Fix: expected trap dispatch to surface a backend trap error, got: {error}"
    );
    assert!(
        error.to_string().contains("direct-readback-ring-trap"),
        "Fix: expected trap dispatch to surface a backend trap error, got: {error}"
    );
    assert_eq!(
        after_allocations,
        before_allocations + 1,
        "Fix: readback-ring trap path must use the dedicated trap sidecar buffer only (no pooled full-sidecar readback buffer allocation).",
    );
}

#[test]

fn direct_record_and_readback_trap_without_readback_rings_allocates_full_sidecar_copy() {
    let harness = PipelineHarness::new("trap-sidecar allocation delta test");
    let arena = harness.arena();
    let pool = arena.pool().clone();

    let program = Program::wrapped(
        vec![],
        [1, 1, 1],
        vec![Node::trap(Expr::u32(5), "direct-readback-no-ring-trap")],
    );

    let pipeline = harness.compile_on_arena(&program, &arena).expect(
        "Fix: trapped program compile must succeed; restore this invariant before continuing.",
    );

    let before_allocations = pool.stats().allocations;
    let error = record_once(
        &pipeline,
        &arena,
        false,
        DispatchLabels {
            bind_group: "vyre direct trap readback no-ring test bind group",
            encoder: "vyre direct trap readback no-ring test",
            compute: "vyre direct trap readback no-ring test compute",
        },
    )
    .expect_err(
        "Fix: trapped dispatch without rings must still return the underlying trap sidecar error and not succeed",
    );
    let after_allocations = pool.stats().allocations;

    assert!(
        error.to_string().contains("wgpu dispatch trapped"),
        "Fix: expected trap dispatch to surface a backend trap error, got: {error}"
    );
    assert!(
        error.to_string().contains("direct-readback-no-ring-trap"),
        "Fix: expected the trap tag to be preserved across fallback sidecar decode, got: {error}"
    );
    assert_eq!(
        after_allocations,
        before_allocations + 2,
        "Fix: non-ring trap path must allocate exactly the full-sidecar pooled readback buffer plus trap sidecar allocation (before={before_allocations}, after={after_allocations})."
    );
}
