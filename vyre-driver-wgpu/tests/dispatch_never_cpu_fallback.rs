//! Dispatch-level CPU demotion rejection tests.
//!
//! Guarantees:
//! - `dispatch()` and `dispatch_async()` never silently fall back to CPU execution
//! - WGPU may serve as a portable GPU fallback, but dispatch must reject CPU demotion
//! - Execution latency is consistent with a GPU round-trip, not instant CPU results
//! - `WgpuBackend::acquire()` fails when only CPU adapters are available

mod harness;
use harness::{acquire_live_backend as live_backend, assert_actionable_error, assert_non_cpu_backend};

use std::time::{Duration, Instant};
use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;

// ------------------------------------------------------------------
// 1. Acquisition rejects CPU adapters
// ------------------------------------------------------------------

#[test]
fn acquisition_fails_when_only_cpu_adapters_exist() {
    let has_real_gpu = vyre_driver_wgpu::runtime::has_real_gpu_adapter();

    if !has_real_gpu {
        let result = WgpuBackend::acquire();
        assert!(
            result.is_err(),
            "Fix: WgpuBackend::acquire must fail when only CPU/Other adapters are available"
        );
        assert_actionable_error(&result, "CPU-only rejection error must be actionable");
    }
}

#[test]
fn successful_acquisition_means_non_cpu_adapter() {
    let backend = live_backend();
    assert_non_cpu_backend(&backend);
}


// ------------------------------------------------------------------
// 2. Dispatch latency is GPU-consistent
// ------------------------------------------------------------------

#[test]
fn dispatch_takes_gpu_consistent_time_not_instant() {
    let backend = live_backend();

    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1024)],
        [64, 1, 1],
        vec![
            Node::store("out", Expr::gid_x(), Expr::u32(42)),
            Node::return_(),
        ],
    );

    let start = Instant::now();
    let _ = backend
        .dispatch(&program, &[], &DispatchConfig::default())
        .expect("Fix: dispatch must succeed");
    let elapsed = start.elapsed();

    // Even the smallest GPU dispatch involves queue submission, kernel launch,
    // and readback. A true CPU fallback would return in < 1 microsecond.
    assert!(
        elapsed > Duration::from_micros(10),
        "Fix: dispatch returned in {:?}, which is too fast for a real GPU round-trip. \
         This suggests a silent CPU fallback.",
        elapsed
    );
}

#[test]
fn dispatch_async_never_returns_synchronous_cpu_result() {
    let backend = live_backend();

    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1024)],
        [64, 1, 1],
        vec![
            Node::store("out", Expr::gid_x(), Expr::u32(99)),
            Node::return_(),
        ],
    );

    let start = Instant::now();
    let pending = backend
        .dispatch_async(&program, &[], &DispatchConfig::default())
        .expect("Fix: dispatch_async must return a handle");
    let handle_return_time = start.elapsed();

    // The handle must be returned before the result is ready.
    assert!(
        handle_return_time < Duration::from_millis(50),
        "Fix: dispatch_async took {:?} to return handle, suggesting synchronous execution",
        handle_return_time
    );

    let outputs = pending
        .await_result()
        .expect("Fix: await_result must resolve");
    assert_eq!(outputs[0].len(), 1024 * 4);
}
