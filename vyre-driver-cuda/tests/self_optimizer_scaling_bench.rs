//! Scaling bench: GPU optimizer wall clock on real CUDA hardware against the
//! CPU optimizer, over the shared chain and wide fixtures.
//!
//! The shared benchmark owns the fixtures, CPU oracle, worker stack, and table.
//! This caller supplies one registered executor and explicit immutable policy.

#![cfg(all(test, feature = "device-tests"))]

use std::collections::BTreeMap;
use std::thread;

use vyre::ir::Program;
use vyre_driver::self_optimizer_bench::report_scaling;
use vyre_driver_cuda::{registered_backend_id, CUDA_BACKEND_ID};
use vyre_megakernel::{
    CompileObjective, Digest, ExternalFacts, SearchBudget, SemanticExecutionPolicy,
    SemanticExecutor,
};
use vyre_pass_engine::optimizer::pipeline::gpu_optimize;
use vyre_runtime::RegisteredSemanticExecutor;

fn semantic_gpu_pipeline(
    program: Program,
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
) -> Program {
    gpu_optimize(program, executor, policy).expect("semantic optimizer pipeline")
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
    let _ = registered_backend_id();
    let registration =
        vyre_driver::backend_registration(CUDA_BACKEND_ID).expect("registered CUDA backend");
    let device = registration.acquire().expect("live CUDA backend");
    let executor = RegisteredSemanticExecutor::new(registration);
    let policy = SemanticExecutionPolicy::new(
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        device.device_profile().compile_facts(),
        CompileObjective::MinimizeLatency,
        SearchBudget::new(128, 128, 0, 0, 128),
        60_000,
    );
    report_scaling("cuda", &executor, &policy, semantic_gpu_pipeline);
}
