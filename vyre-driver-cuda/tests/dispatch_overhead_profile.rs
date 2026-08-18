//! Fine-grained CUDA dispatch overhead attribution.
//!
//! `dispatch_overhead_breakdown` reports the steady-state per-dispatch wall
//! time; this test splits that wall time into its phases (host enqueue vs
//! completion wait vs device kernel) so optimization targets the real headroom
//! instead of guessing. A no-op program isolates the fixed per-dispatch cost:
//! the GPU work is ~nothing, so whatever remains is overhead we can cut.

mod harness;
use harness::no_op_program;
use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

#[test]
fn cuda_steady_state_phase_attribution() {
    let backend = CudaBackend::acquire()
        .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host.");
    let program = no_op_program();
    let input = vec![0u8; 4];
    let inputs: [&[u8]; 1] = [input.as_slice()];
    let config = DispatchConfig::default();

    // Warm the PTX/module/launch-resource caches so the steady-state loop below
    // measures only the recurring per-dispatch cost.
    for _ in 0..3 {
        let _ = backend
            .dispatch_borrowed_timed(&program, &inputs, &config)
            .expect("warm dispatch must succeed");
    }

    const RUNS: u64 = 200;
    let (mut wall, mut enqueue, mut wait, mut device) = (0u64, 0u64, 0u64, 0u64);
    let mut enqueue_samples = 0u64;
    let mut wait_samples = 0u64;
    let mut device_samples = 0u64;
    for _ in 0..RUNS {
        let r = backend
            .dispatch_borrowed_timed(&program, &inputs, &config)
            .expect("steady-state dispatch must succeed");
        wall += r.wall_ns;
        if let Some(e) = r.enqueue_ns {
            enqueue += e;
            enqueue_samples += 1;
        }
        if let Some(w) = r.wait_ns {
            wait += w;
            wait_samples += 1;
        }
        if let Some(d) = r.device_ns {
            device += d;
            device_samples += 1;
        }
    }

    let div = |sum: u64, n: u64| if n == 0 { 0 } else { sum / n };
    println!();
    println!("=== CUDA steady-state dispatch phase attribution ({RUNS} runs) ===");
    println!("wall_ns/dispatch     {:>10}", wall / RUNS);
    println!(
        "enqueue_ns/dispatch  {:>10}  (host prep + launch enqueue, {enqueue_samples} samples)",
        div(enqueue, enqueue_samples)
    );
    println!(
        "wait_ns/dispatch     {:>10}  (sync + readback, {wait_samples} samples)",
        div(wait, wait_samples)
    );
    if device_samples > 0 {
        println!(
            "device_ns/dispatch   {:>10}  (GPU kernel, {device_samples} samples)",
            div(device, device_samples)
        );
    } else {
        println!("device_ns/dispatch          n/a  (no device timer exposed on this path)");
    }
    println!("===");
}
