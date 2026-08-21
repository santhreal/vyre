//! Scaling bench: GPU optimizer wall clock against the CPU optimizer over the
//! shared chain and wide fixtures.
//!
//! The fixtures, the CPU oracle, its worker stack and the table live in
//! `vyre_driver::self_optimizer_bench`, so this backend's numbers can be read
//! against another backend's row for row. What is here is the wgpu device and
//! the three-pass borrowed-dispatch sequence this backend supports.
//!
//! The bench asserts nothing. A single-thread sequential kernel loses to the CPU
//! at small sizes on dispatch overhead; the signal is whether the GPU column
//! stays flat while the CPU column grows.

#![cfg(all(test, feature = "device-tests"))]

mod harness;
use harness::self_optimizer::WgpuProgramDispatcher;

use vyre::ir::Program;
use vyre_driver::self_optimizer_bench::report_scaling;
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::program_dispatch::ProgramDispatcher;
use vyre_pass_engine::optimizer::pipeline::gpu_sequential_three_pass;

/// wgpu implements the borrowed `dispatch` surface only, so the passes run
/// sequentially with each re-encoding its input.
fn sequential_gpu_pipeline(program: Program, dispatcher: &dyn ProgramDispatcher) -> Program {
    gpu_sequential_three_pass(program, dispatcher).expect("sequential three-pass pipeline")
}

#[test]
fn scaling_bench_gpu_vs_cpu_pipeline() {
    let backend = WgpuBackend::acquire().expect("WgpuBackend acquire");
    let dispatcher = WgpuProgramDispatcher::new(&backend);
    report_scaling("wgpu", &dispatcher, sequential_gpu_pipeline);
}
