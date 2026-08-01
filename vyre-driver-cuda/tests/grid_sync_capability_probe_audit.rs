//! Audit: every cooperative grid-sync capability claim must be backed by a live
//! device probe, and the cooperative residency ceiling must be derived from that
//! probe rather than from a host-side constant.
//!
//! Defect class this locks out: a backend capability accessor that returns a
//! hardcoded value instead of reading the probed device. Such an accessor makes
//! the backend claim a capability it may not have, and callers size their work
//! against a number that never moves when the hardware does. The specific
//! accessors under audit are
//! `CudaBackend::hardware_supports_grid_sync` (must equal the freshly probed
//! `compute_capability >= (6, 0) && cooperative_launch`) and
//! `occupancy::cooperative_thread_residency_block_limit` (must equal
//! `(max_threads_per_sm / workgroup) * multi_processor_count` computed from the
//! same fresh probe).
//!
//! The residency ceiling is load-bearing beyond this crate: it is the maximum
//! whole-grid block count a `MemoryOrdering::GridSync` program can launch with
//! in ONE cooperative dispatch, so it is the real upper bound on any
//! per-dispatch throughput claim that depends on a grid-synchronized kernel.

use vyre_driver_cuda::occupancy::cooperative_thread_residency_block_limit;
use vyre_driver_cuda::{CudaBackend, CudaDeviceCaps};

/// Workgroup widths the primitive layer actually declares: 256 is the
/// `PERSISTENT_FIXPOINT_WORKGROUP_SIZE` / `SCALLOP_JOIN_WORKGROUP_SIZE` width,
/// 1024 is the DCE persistent-BFS width, and the rest bracket them.
const AUDITED_WORKGROUP_WIDTHS: [u32; 5] = [64, 128, 256, 512, 1024];

/// `hardware_supports_grid_sync()` must be a function of the live probe, not a
/// constant. Recomputes the predicate from an independent `CudaDeviceCaps::probe`
/// so a hardcoded `true`/`false` in the accessor diverges from the device and
/// fails here.
#[test]
fn hardware_grid_sync_claim_equals_the_independently_probed_predicate() {
    let probed = CudaDeviceCaps::probe(0)
        .expect("Fix: direct CUDA capability probe of device 0 must succeed on the GPU fleet.");
    let backend = CudaBackend::acquire()
        .expect("Fix: CudaBackend::acquire must succeed on the GPU-required test host.");

    let expected = probed.compute_capability >= (6, 0) && probed.cooperative_launch;
    assert_eq!(
        backend.hardware_supports_grid_sync(),
        expected,
        "Fix: hardware_supports_grid_sync() must be derived from the probed \
         compute_capability ({:?}) and cooperative_launch ({}), not hardcoded.",
        probed.compute_capability,
        probed.cooperative_launch,
    );

    // On this fleet the probe must report a cooperative-capable device. A false
    // here means the capability is genuinely absent and every GridSync program
    // routes to the kernel-split path, which is a configuration failure on a
    // host that has an RTX 5090, not an acceptable outcome.
    assert!(
        probed.cooperative_launch,
        "Fix: CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH probed false on `{}`; native \
         cooperative grid-sync cannot run and multi-workgroup GridSync dispatch is \
         unavailable on this host.",
        probed.name,
    );
    assert!(
        backend.hardware_supports_grid_sync(),
        "Fix: cooperative_launch probed true but hardware_supports_grid_sync() \
         reported false; the compute-capability floor rejected {:?}.",
        probed.compute_capability,
    );
}

/// The cooperative residency ceiling must track probed SM geometry exactly.
///
/// Defect this locks out: a residency bound that is a host-side constant (or
/// derived from the wrong probe field) silently caps or over-promises the grid a
/// cooperative launch may use. Over-promising is the dangerous direction, since
/// `cuLaunchCooperativeKernel` then fails at launch instead of at plan time.
#[test]
fn cooperative_residency_ceiling_is_computed_from_probed_sm_geometry() {
    let probed = CudaDeviceCaps::probe(0)
        .expect("Fix: direct CUDA capability probe of device 0 must succeed on the GPU fleet.");

    let threads_per_sm = probed.max_threads_per_sm_u32();
    let sm_count = probed.multi_processor_count_u32();
    assert!(
        threads_per_sm > 0 && sm_count > 0,
        "Fix: probed SM geometry is degenerate (max_threads_per_sm={threads_per_sm}, \
         multi_processor_count={sm_count}); the cooperative residency bound cannot be \
         derived from it."
    );

    for width in AUDITED_WORKGROUP_WIDTHS {
        let expected = u64::from(threads_per_sm / width) * u64::from(sm_count);
        let actual = cooperative_thread_residency_block_limit(&probed, width);
        assert_eq!(
            actual, expected,
            "Fix: cooperative_thread_residency_block_limit({width}) must equal \
             (max_threads_per_sm / workgroup) * multi_processor_count = \
             ({threads_per_sm} / {width}) * {sm_count} = {expected}, got {actual}.",
        );
        assert!(
            actual > 0,
            "Fix: no cooperative grid of {width}-thread blocks is resident on `{}`, so \
             every GridSync program at that width is unlaunchable.",
            probed.name,
        );
    }

    // Zero-width must be rejected rather than divide by zero or return a bound.
    assert_eq!(
        cooperative_thread_residency_block_limit(&probed, 0),
        0,
        "Fix: a zero workgroup width must yield a zero residency bound."
    );

    // Print the audited ceiling so the number that bounds per-dispatch
    // grid-synchronized work is recorded in the test log rather than re-derived
    // by every consumer.
    println!(
        "device={} cc={:?} sm_count={} max_threads_per_sm={} cooperative_launch={}",
        probed.name, probed.compute_capability, sm_count, threads_per_sm, probed.cooperative_launch,
    );
    for width in AUDITED_WORKGROUP_WIDTHS {
        println!(
            "cooperative_resident_blocks[wg={}] = {} ({} threads)",
            width,
            cooperative_thread_residency_block_limit(&probed, width),
            cooperative_thread_residency_block_limit(&probed, width) * u64::from(width),
        );
    }
}

/// `supports_grid_sync()` is the conjunction the dispatch path gates on, so it
/// must reduce to the hardware predicate exactly while `lowers_grid_sync()` is
/// unconditional. This pins the two apart: if lowering ever becomes conditional,
/// this test forces the conditional to be reflected here rather than leaving
/// `supports_grid_sync()` silently over-claiming.
#[test]
fn supports_grid_sync_reduces_to_the_hardware_predicate_while_lowering_is_unconditional() {
    let backend = CudaBackend::acquire()
        .expect("Fix: CudaBackend::acquire must succeed on the GPU-required test host.");

    assert!(
        backend.lowers_grid_sync(),
        "Fix: the CUDA PTX emitter lowers MemoryOrdering::GridSync to a cooperative \
         grid barrier, so lowers_grid_sync() must report it."
    );
    assert_eq!(
        backend.supports_grid_sync(),
        backend.hardware_supports_grid_sync(),
        "Fix: with unconditional native lowering, supports_grid_sync() must equal \
         hardware_supports_grid_sync(); any divergence means one of the two is not \
         reading the live probe."
    );
}
