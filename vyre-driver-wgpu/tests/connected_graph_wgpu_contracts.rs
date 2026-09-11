//! WGPU connected-graph production route contracts.
//!
//! A representative connected graph must execute through
//! `CompileRequest -> ArtifactEnvelope -> TargetPayload -> ArtifactInstance -> BindingSet -> Completion`
//! and match independent semantics.
//!
//! The graph and its host oracle belong to
//! [`vyre_test_support::graph_fixtures`], because every concrete driver is
//! asked the same question and an answer that differs by which suite built the
//! graph proves nothing.

#![cfg(all(test, feature = "device-tests"))]

use std::collections::BTreeMap;

use vyre_driver::BoundResource;
use vyre_driver_wgpu::{registered_backend_id, WGPU_BACKEND_ID};
use vyre_megakernel::{
    attach_target, compile, CompileObjective, CompileRequest, Digest, ExternalFacts,
    ObjectiveMetric, SearchBudget,
};
use vyre_runtime::artifact_admission::ArtifactSession;
use vyre_test_support::graph_fixtures::{pure_dataflow_graph, pure_dataflow_oracle};

/// The four lanes the graph is driven with.
const INPUT_LANES: [u32; 4] = [1, 2, 3, 4];

#[test]
fn wgpu_executes_pure_dataflow_connected_graph() {
    let _ = registered_backend_id();
    let registration =
        vyre_driver::backend_registration(WGPU_BACKEND_ID).expect("registered WGPU backend");
    let device = registration
        .acquire()
        .expect("WGPU device acquisition must succeed");
    assert!(!device.device_lost(), "WGPU device must not be lost");
    let device_facts = device.device_profile().compile_facts();

    let request = CompileRequest::new(
        pure_dataflow_graph(),
        ExternalFacts::new(Digest([0x56; 32]), BTreeMap::new()),
        device_facts,
        SearchBudget::new(64, 1_000_000, 4, 0, 10_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
    )
    .validate()
    .expect("compile request must validate");

    let compiler = registration.target_compiler().expect("target compiler");
    let artifact = compile(&request).expect("compile must succeed");
    assert_eq!(artifact.nodes().len(), 3);
    assert_ne!(artifact.digest(), Digest([0; 32]));

    let envelope = attach_target(artifact, compiler.as_ref()).expect("attach target");
    let payloads = envelope.target_payloads();
    assert_eq!(payloads.len(), 1, "attach_target must attach one payload");
    assert_ne!(payloads[0].digest(), Digest([0; 32]));

    let session = ArtifactSession::from_envelope(registration, envelope).expect("materialization");
    let mut bindings = session.bindings().expect("binding set");

    let input_bytes = INPUT_LANES
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    if let Ok(in_val) = session.resource("in_x") {
        bindings.insert(in_val, BoundResource::Host(input_bytes));
    }
    if let Ok(out_val) = session.resource("z_out") {
        bindings.insert(out_val, BoundResource::Host(vec![0_u8; 16]));
    }

    let completion = session
        .submit_and_wait(bindings)
        .expect("execution must complete");

    assert_ne!(completion.artifact, Digest([0; 32]));
    assert_eq!(completion.artifact, session.artifact().unwrap());
    assert_ne!(session.payload().unwrap(), Digest([0; 32]));

    let expected = pure_dataflow_oracle(INPUT_LANES)
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let out_res = session.resource("z_out").expect("z_out resource");
    let actual = completion
        .outputs
        .get(&out_res)
        .expect("z_out output bytes");
    assert_eq!(actual, &expected, "output must match independent oracle");
}
