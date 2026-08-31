//! Shared semantic executor policy for the self-hosted optimizer suites.

use std::collections::BTreeMap;

use vyre_driver_wgpu::{WgpuBackend, WGPU_BACKEND_ID};
use vyre_megakernel::{
    CompileObjective, Digest, ExternalFacts, ObjectiveMetric, SearchBudget, SemanticExecutionPolicy,
};
use vyre_runtime::RegisteredSemanticExecutor;

pub(crate) fn semantic_execution(
    backend: &WgpuBackend,
) -> (RegisteredSemanticExecutor, SemanticExecutionPolicy) {
    let _ = vyre_driver_wgpu::registered_backend_id();
    let registration =
        vyre_driver::backend_registration(WGPU_BACKEND_ID).expect("registered WGPU backend");
    let executor = RegisteredSemanticExecutor::new(registration);
    let policy = SemanticExecutionPolicy::new(
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        backend.device_profile().compile_facts(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 60_000),
        SearchBudget::new(128, 128, 0, 0, 128),
    );
    (executor, policy)
}

/// The program shape every optimizer pass suite feeds in, and the reader that
/// peels the `Region` wrapper back off. Owned by `vyre-test-support` so the wgpu
/// and CUDA suites cannot disagree about either.
pub(crate) use vyre_test_support::pass_programs::{first_let_value, wrapped};
