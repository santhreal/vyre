//! Scaling bench: GPU optimizer wall clock against the CPU optimizer over the
//! shared chain and wide fixtures.
//!
//! The shared benchmark owns the fixtures, CPU oracle, worker stack, and table.
//! This caller supplies one registered executor and explicit immutable policy.

#![cfg(all(test, feature = "device-tests"))]

mod harness;
use harness::self_optimizer::semantic_execution;

use vyre::ir::Program;
use vyre_driver::self_optimizer_bench::report_scaling;
use vyre_driver_wgpu::WgpuBackend;
use vyre_megakernel::{SemanticExecutionPolicy, SemanticExecutor};
use vyre_pass_engine::optimizer::pipeline::gpu_optimize;

fn semantic_gpu_pipeline(
    program: Program,
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
) -> Program {
    gpu_optimize(program, executor, policy).expect("semantic optimizer pipeline")
}

#[test]
fn scaling_bench_gpu_vs_cpu_pipeline() {
    let backend = WgpuBackend::acquire().expect("WgpuBackend acquire");
    let (executor, policy) = semantic_execution(&backend);
    report_scaling("wgpu", &executor, &policy, semantic_gpu_pipeline);
}
