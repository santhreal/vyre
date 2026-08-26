//! WGPU artifact-backed semantic execution contracts.

#![cfg(all(test, feature = "device-tests"))]

mod harness;

use harness::acquire_live_backend;
use vyre_driver_wgpu::{registered_backend_id, WGPU_BACKEND_ID};
use vyre_megakernel::DeviceFacts;
use vyre_runtime::RegisteredSemanticExecutor;
use vyre_test_support::semantic_requests::{
    assert_executes_add, assert_refuses_zero_artifact_limit,
};

fn live_executor() -> (RegisteredSemanticExecutor, DeviceFacts) {
    let backend = acquire_live_backend();
    let facts = backend.device_profile().compile_facts();
    let _ = registered_backend_id();
    let registration =
        vyre_driver::backend_registration(WGPU_BACKEND_ID).expect("registered WGPU backend");
    (RegisteredSemanticExecutor::new(registration), facts)
}

#[test]
fn wgpu_executes_graph_values_through_registered_artifact() {
    let (executor, facts) = live_executor();
    assert_executes_add(&executor, facts, "wgpu-add");
}

#[test]
fn wgpu_rejects_hostile_artifact_limit_before_submission() {
    let (executor, facts) = live_executor();
    assert_refuses_zero_artifact_limit(&executor, facts, "wgpu-add");
}
