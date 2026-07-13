//! W3-4 pool-hit-rate evidence, on a consumer-shaped re-dispatch workload.
//!
//! The whole point of the transient `DeviceAllocationPool` is that a steady-state
//! re-dispatch loop (the same Program shape scanned over batch after batch) should
//! serve its per-dispatch device buffers from the free-list instead of calling
//! `cuMemAlloc_v2` every time. This test proves that on the real GPU: it warms the
//! pool, resets the telemetry epoch, runs an identical-shape dispatch loop, and
//! asserts the telemetry snapshot reports a HIGH pool hit rate, the evidence the
//! plan asks for, not just that the counters exist.

use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// A minimal fixed-shape Program: one thread storing a constant. Every dispatch
/// requests the same device buffers, so after warm-up the pool should serve them
/// all from cache.
fn no_op_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(0))],
    )
}

#[test]
fn steady_state_redispatch_loop_reports_high_device_pool_hit_rate() {
    let backend = CudaBackend::acquire()
        .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host.");

    let program = no_op_program();
    let inputs: Vec<Vec<u8>> = vec![vec![0u8; 4]];
    let config = DispatchConfig::default();

    // Warm the transient pool: the first couple of dispatches populate the
    // free-list buckets (their acquisitions are the unavoidable cold misses).
    for _ in 0..3 {
        backend
            .dispatch(&program, &inputs, &config)
            .expect("Fix: CUDA no-op dispatch must succeed while warming the pool.");
    }

    // Start a clean telemetry epoch so the hit rate reflects the steady state,
    // not the cold warm-up misses. reset_telemetry also resets the pool counters.
    backend.reset_telemetry();

    // The steady-state re-dispatch loop: identical shape, over and over.
    const STEADY_DISPATCHES: usize = 32;
    for _ in 0..STEADY_DISPATCHES {
        let outputs = backend
            .dispatch(&program, &inputs, &config)
            .expect("Fix: CUDA no-op dispatch must succeed in the steady-state loop.");
        assert_eq!(
            outputs.len(),
            1,
            "the no-op dispatch must return exactly one output buffer"
        );
        assert_eq!(
            outputs[0].as_slice(),
            0u32.to_le_bytes().as_slice(),
            "the no-op dispatch must write the zero word"
        );
    }

    let snapshot = backend.telemetry_snapshot();
    let hits = snapshot.device_pool_hits;
    let misses = snapshot.device_pool_misses;
    let hit_rate_bps = snapshot.device_pool_hit_rate_bps();
    println!(
        "device pool over {STEADY_DISPATCHES} steady dispatches: hits={hits} misses={misses} hit_rate={hit_rate_bps}bps"
    );

    // The loop actually exercised the pool.
    assert!(
        hits + misses >= STEADY_DISPATCHES as u64,
        "each steady dispatch must acquire at least one pooled buffer; saw {hits} hits + {misses} misses over {STEADY_DISPATCHES} dispatches"
    );
    // Steady state: the pool predominantly serves from cache (this is the whole
    // reason the pool exists). A regression that broke bucket reuse would drop the
    // hit rate and fail here.
    assert!(
        hits > misses,
        "steady-state re-dispatch must be majority pool hits, not fresh cuMemAlloc; hits={hits} misses={misses}"
    );
    assert!(
        hit_rate_bps >= 5_000,
        "steady-state pool hit rate must be at least 50% (5000 bps); got {hit_rate_bps} bps"
    );

    // The evidence is operator-visible in the Prometheus exposition.
    let text = snapshot.to_prometheus_text();
    assert!(
        text.contains("vyre_cuda_device_pool_hits_total")
            && text.contains("vyre_cuda_device_pool_hit_rate_bps"),
        "pool hit-rate evidence must be exported in the Prometheus text"
    );

    // The hit rate is exactly consistent with the raw counters (no drift between
    // the derived ratio and the numbers it is derived from).
    let expected_bps = ((u128::from(hits) * 10_000) / u128::from(hits + misses)) as u32;
    assert_eq!(
        hit_rate_bps, expected_bps,
        "reported hit-rate bps must equal hits*10000/(hits+misses)"
    );
}
