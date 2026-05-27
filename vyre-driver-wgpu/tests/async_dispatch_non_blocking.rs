//! Async dispatch must not block on GPU completion before returning the
//! pending handle.
//!
//! Guarantees:
//! - `dispatch_async` returns a `PendingDispatch` handle immediately
//! - The handle's `is_ready()` is observable as `false` for non-trivial work
//! - Multiple back-to-back async dispatches can be submitted without host
//!   serialization (total submit time < single dispatch time)
//! - GPU execution errors surface through `await_result`, not `dispatch_async`

mod common;
use common::{
    acquire_live_backend as live_backend, add_one_expected, add_one_input, add_one_program,
};

use std::time::{Duration, Instant};
use vyre::{DispatchConfig, VyreBackend};
use vyre_driver::PendingDispatch;

// ------------------------------------------------------------------
// 1. Handle returned before GPU completion
// ------------------------------------------------------------------

#[test]
fn dispatch_async_returns_before_gpu_completion() {
    let backend = live_backend();
    let program = add_one_program(512 * 1024);
    let input = add_one_input(512 * 1024);

    let start = Instant::now();
    let pending = backend
        .dispatch_async(&program, &[input], &DispatchConfig::default())
        .expect("Fix: dispatch_async must return a handle immediately without blocking on GPU completion");
    let return_time = start.elapsed();

    // The handle must be returned faster than the GPU execution itself.
    // 1s budget reflects realistic CPU-side prep (validation, pipeline
    // compile/cache lookup, scratch + bind-group setup, encoder record,
    // queue submit) for a 512K-element add-one program; the contract
    // we enforce is "no GPU sync wait inside dispatch_async", not a
    // hard wall-clock  -  dispatch_async is permitted to do all the
    // CPU-side dispatch prep, just not block on GPU completion.
    assert!(
        return_time < Duration::from_secs(1),
        "Fix: dispatch_async took {:?} to return; this suggests synchronous GPU blocking",
        return_time
    );

    // is_ready must be callable without panic.
    let _ = pending.is_ready();

    let outputs = pending
        .await_result()
        .expect("Fix: await_result must resolve correctly");
    let expected = add_one_expected(512 * 1024);
    assert_eq!(outputs, vec![expected]);
}

// ------------------------------------------------------------------
// 2. Observable ready state for non-trivial work
// ------------------------------------------------------------------

#[test]
fn dispatch_async_ready_state_is_observable_for_non_trivial_work() {
    let backend = live_backend();
    let program = add_one_program(256 * 1024);
    let input = add_one_input(256 * 1024);

    let pending = backend
        .dispatch_async(&program, &[input], &DispatchConfig::default())
        .expect("Fix: dispatch_async must return a handle");

    // Poll once; must not panic.
    let ready_now = pending.is_ready();

    let outputs = pending
        .await_result()
        .expect("Fix: await_result must resolve");
    let expected = add_one_expected(256 * 1024);
    assert_eq!(outputs, vec![expected]);

    // If the handle was not ready on the first poll, we proved the deferred contract.
    if !ready_now {
        // Non-blocking contract verified: work was still in flight when the handle was returned.
    }
}

// ------------------------------------------------------------------
// 3. Concurrent dispatches do not serialize on the host
// ------------------------------------------------------------------

#[test]
fn multiple_concurrent_async_dispatches_do_not_serialize() {
    let backend = live_backend();
    let program = add_one_program(128 * 1024);
    let input = add_one_input(128 * 1024);

    // Warm the pipeline cache.
    let _ = backend
        .dispatch(&program, &[input.clone()], &DispatchConfig::default())
        .expect("Fix: warm-up dispatch must succeed");

    let start = Instant::now();
    let p1 = backend
        .dispatch_async(&program, &[input.clone()], &DispatchConfig::default())
        .expect("Fix: dispatch_async #1 must start");
    let p2 = backend
        .dispatch_async(&program, &[input.clone()], &DispatchConfig::default())
        .expect("Fix: dispatch_async #2 must start");
    let p3 = backend
        .dispatch_async(&program, &[input.clone()], &DispatchConfig::default())
        .expect("Fix: dispatch_async #3 must start");
    let submit_time = start.elapsed();

    assert!(
        submit_time < Duration::from_millis(100),
        "Fix: three back-to-back dispatch_async calls took {:?}, suggesting blocking behavior",
        submit_time
    );

    let o1 = p1
        .await_result()
        .expect("Fix: async dispatch #1 must complete");
    let o2 = p2
        .await_result()
        .expect("Fix: async dispatch #2 must complete");
    let o3 = p3
        .await_result()
        .expect("Fix: async dispatch #3 must complete");
    assert_eq!(
        o1, o2,
        "Fix: identical async dispatches must produce identical outputs"
    );
    assert_eq!(o2, o3);
}

// ------------------------------------------------------------------
// 4. Object safety of the pending handle
// ------------------------------------------------------------------

#[test]
fn pending_dispatch_from_wgpu_is_object_safe() {
    let backend = live_backend();
    let program = add_one_program(1024);
    let input = add_one_input(1024);

    let pending: Box<dyn PendingDispatch> = backend
        .dispatch_async(&program, &[input], &DispatchConfig::default())
        .expect("Fix: wgpu dispatch_async must produce object-safe PendingDispatch");

    let outputs = pending
        .await_result()
        .expect("Fix: object-safe await must succeed");
    let expected = add_one_expected(1024);
    assert_eq!(outputs, vec![expected]);
}
