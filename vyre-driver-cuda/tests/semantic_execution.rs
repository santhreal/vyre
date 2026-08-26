//! CUDA artifact-backed semantic execution contracts.

#![cfg(all(test, feature = "device-tests"))]

use vyre_driver_cuda::{registered_backend_id, CUDA_BACKEND_ID};
use vyre_megakernel::DeviceFacts;
use vyre_runtime::RegisteredSemanticExecutor;
use vyre_test_support::semantic_requests::{
    assert_executes_add, assert_refuses_zero_artifact_limit,
};

fn live_executor() -> (RegisteredSemanticExecutor, DeviceFacts) {
    let _ = registered_backend_id();
    let registration =
        vyre_driver::backend_registration(CUDA_BACKEND_ID).expect("registered CUDA backend");
    let device = registration.acquire().expect("live CUDA backend");
    let facts = device.device_profile().compile_facts();
    (RegisteredSemanticExecutor::new(registration), facts)
}

#[test]
fn cuda_executes_graph_values_through_registered_artifact() {
    let (executor, facts) = live_executor();
    assert_executes_add(&executor, facts, "cuda-add");
}

#[test]
fn cuda_rejects_hostile_artifact_limit_before_submission() {
    let (executor, facts) = live_executor();
    assert_refuses_zero_artifact_limit(&executor, facts, "cuda-add");
}
