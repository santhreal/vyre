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

#![cfg(test)]

mod harness;
use harness::self_optimizer::WgpuProgramDispatcher;

use vyre::ir::Program;
use vyre_driver::self_optimizer_bench::report_scaling;
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::program_dispatch::ProgramDispatcher;
use vyre_pass_engine::optimizer::canonicalize_via_encoded::gpu_canonicalize;
use vyre_pass_engine::optimizer::const_fold_via_encoded::gpu_const_fold;
use vyre_pass_engine::optimizer::dce_via_encoded::gpu_dce;

/// wgpu implements the borrowed `dispatch` surface only, so the passes run
/// sequentially with each re-encoding its input.
fn sequential_gpu_pipeline(program: Program, dispatcher: &dyn ProgramDispatcher) -> Program {
    let program = gpu_canonicalize(program, dispatcher).expect("canonicalize");
    let program = gpu_const_fold(program, dispatcher).expect("const-fold");
    gpu_dce(program, dispatcher).expect("dce")
}

#[test]
fn scaling_bench_gpu_vs_cpu_pipeline() {
    let backend = WgpuBackend::acquire().expect("WgpuBackend acquire");
    let dispatcher = WgpuProgramDispatcher::new(&backend);
    report_scaling("wgpu", &dispatcher, sequential_gpu_pipeline);
}
