//! Contracts for `vyre_driver_cuda::synthetic_device_caps`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver_cuda::synthetic_device_caps::{
    synthetic_sm120_envelope, synthetic_sm120_envelope_default,
};

#[test]
fn synthetic_envelope_preserves_architecture_fields() {
    let caps = synthetic_sm120_envelope_default();

    assert_eq!(caps.compute_capability, (12, 0));
    assert_eq!(caps.warp_size, 32);
    assert_eq!(caps.multi_processor_count, 170);
    assert_eq!(caps.shared_memory_per_block, 128 * 1024);
    assert_eq!(caps.shared_memory_per_sm, 256 * 1024);
    assert_eq!(caps.l2_cache_bytes, 96 * 1024 * 1024);
    assert!(caps.cooperative_launch);
    assert!(caps.concurrent_kernels);
}

#[test]
fn synthetic_envelope_peak_compute_matches_scheduler_issue_model() {
    let caps = synthetic_sm120_envelope_default();
    // SM_count × 4 warp schedulers × warp_size × core_clock_hz.
    let expected = 170u64 * 4 * 32 * 2_410_000 * 1_000;
    assert_eq!(
        caps.peak_compute_ops_per_sec(),
        expected,
        "peak compute must follow the universal 4-scheduler issue model exactly"
    );
    // Sanity bound on the envelope's own arithmetic (about 52 TOPS at these
    // fixed clocks), not a claim about any real part.
    let tops = caps.peak_compute_ops_per_sec() as f64 / 1e12;
    assert!(
        (40.0..80.0).contains(&tops),
        "peak int throughput {tops:.1} TOPS is outside the range this envelope's fixed clocks \
         and SM count can produce"
    );
}

#[test]
fn synthetic_envelope_keeps_memory_pressure_parametric() {
    let low_vram = synthetic_sm120_envelope(512 * 1024 * 1024);
    let high_vram = synthetic_sm120_envelope_default();

    assert_eq!(low_vram.total_memory, 512 * 1024 * 1024);
    assert_eq!(high_vram.total_memory, 32 * 1024 * 1024 * 1024);
    assert_eq!(low_vram.compute_capability, high_vram.compute_capability);
    assert_eq!(
        low_vram.max_threads_per_block,
        high_vram.max_threads_per_block
    );
}
