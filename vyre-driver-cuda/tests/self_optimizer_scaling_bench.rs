//! Scaling bench: GPU optimizer wall clock on real CUDA hardware against the
//! CPU optimizer, over the shared chain and wide fixtures.
//!
//! The fixtures, the CPU oracle, its worker stack and the table live in
//! `vyre_driver::self_optimizer_bench`, so this backend's numbers can be read
//! against another backend's row for row. What is here is the CUDA device and
//! the three-pass sequence.
//!
//! This measures the sequential per-pass path deliberately, not
//! `gpu_optimize`'s persistent-resident route: the resident path has its own
//! end-to-end suite, and comparing the same pass sequence on both backends is
//! the point of the table.

#![cfg(feature = "device-tests")]
#![cfg(test)]

mod harness;

use harness::CudaProgramDispatcher;
use std::thread;

use vyre::ir::Program;
use vyre_driver::self_optimizer_bench::report_scaling;
use vyre_driver_cuda::CudaBackend;
use vyre_foundation::program_dispatch::ProgramDispatcher;
use vyre_pass_engine::optimizer::canonicalize_via_encoded::gpu_canonicalize;
use vyre_pass_engine::optimizer::const_fold_via_encoded::gpu_const_fold;
use vyre_pass_engine::optimizer::dce_via_encoded::gpu_dce;

fn sequential_gpu_pipeline(program: Program, dispatcher: &dyn ProgramDispatcher) -> Program {
    let program = gpu_canonicalize(program, dispatcher).expect("canonicalize");
    let program = gpu_const_fold(program, dispatcher).expect("const-fold");
    gpu_dce(program, dispatcher).expect("dce")
}

#[test]
fn cuda_scaling_bench_gpu_vs_cpu_pipeline() {
    thread::Builder::new()
        .name("cuda_scaling_bench_worker".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(body)
        .expect("spawn cuda scaling bench worker with expanded stack")
        .join()
        .expect("cuda scaling bench worker panicked");
}

fn body() {
    let backend = CudaBackend::acquire().expect("CudaBackend acquire");
    let dispatcher = CudaProgramDispatcher::new(&backend);
    report_scaling("cuda", &dispatcher, sequential_gpu_pipeline);
}
