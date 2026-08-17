//! What one launch charges the compiled-module cache.
//!
//! WHY: closes the class "one launch counted twice in the module cache". A
//! dispatch resolves its launch function from the module cache, then asks the same
//! cache for the module's globals so it can hold the trap sidecar and grid-barrier
//! lease. Both lookups charged a hit, so `pipeline_cache_snapshot` reported two
//! module resolutions per launch and roughly double the real hit count. That
//! number is not decoration: it is the evidence an operator reads to decide
//! whether the pipeline cache is working, and a doubled counter makes a cache that
//! is thrashing look like a cache that is being used.
//!
//! The contract asserted is exact rather than directional. A test that only
//! checked the count went up would have passed against the defect, and against a
//! future third lookup on the same key.
//!
//! What it does not catch: a miss counted twice, which cannot happen through this
//! path because the first lookup loads the module and the second then hits; a
//! launch that resolves more than one module, which is a multi-module payload and
//! is charged per module by design; and whether the hit-to-miss ratio is any good,
//! which is a tuning question and not an accounting one.

mod harness;
use harness::{add_one_program, u32_bytes};
use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;

/// Repeated dispatches measured after the cache is warm.
const LAUNCHES: u64 = 4;


#[test]
fn a_warm_launch_charges_the_module_cache_exactly_once() {
    let backend = CudaBackend::acquire()
        .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host.");
    let program = add_one_program();
    let inputs = [u32_bytes(&[0, 1, 2, 3, 9, 10, 99, u32::MAX - 1])];

    // Warm the cache, so the measured launches are hits and not loads.
    backend
        .dispatch(&program, &inputs, &DispatchConfig::default())
        .expect("Fix: the add-one program must dispatch on the test host.");

    let before = backend.pipeline_cache_snapshot();
    for _ in 0..LAUNCHES {
        backend
            .dispatch(&program, &inputs, &DispatchConfig::default())
            .expect("Fix: a repeated dispatch of a warm program must succeed.");
    }
    let after = backend.pipeline_cache_snapshot();

    assert_eq!(
        after.misses, before.misses,
        "Fix: a warm program must not reload its module; a miss here means the cache evicted between launches and the hit accounting below is measuring the wrong thing."
    );
    assert_eq!(
        after.hits - before.hits,
        LAUNCHES,
        "Fix: one launch of a one-module program must charge the module cache exactly once. Resolving the launch function and resolving the module globals are two lookups of the same key within one dispatch, and only the first is a module resolution to report."
    );
}
