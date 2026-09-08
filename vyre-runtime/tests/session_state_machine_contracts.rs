//! Session state machine, retained generation, error classification, and recovery contracts.
//!
//! BACKLOG row 54 requires a typed domain-neutral execution session that owns artifact
//! identity, authenticated resources, retained generations, bounded step/control state,
//! IO readiness, and recovery across one-shot, iterative, and event-driven graphs.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use vyre_driver::{BackendError, BackendRegistration, ErrorCode};
use vyre_foundation::diagnostics::RetryClass;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ShapeDim, ValueContract, ValueLifetime,
};
use vyre_megakernel::{Artifact, ArtifactEnvelope, ArtifactValueId};
use vyre_runtime::artifact_admission::{ArtifactSession, RetainedArtifactSession};
use vyre_runtime::recovery::classify_backend_error;

use vyre_test_support::artifact_fixtures;

use crate::artifact_session_fixtures::{
    fixture_backend_registration, fixture_target_payload, SessionFixtureMaterializer,
};

const FORMAT: &str = "state_machine.target";
static SM_REGISTRATION: BackendRegistration = fixture_backend_registration("sm-artifact");

fn stateful_accumulator_artifact() -> Artifact {
    let mut graph = ProgramGraph::new();

    let delta = graph
        .add_external_value(
            "delta",
            ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(1)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
        )
        .expect("delta");

    let counter_0 = graph
        .add_external_value(
            "counter.0",
            ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(1)],
                access: BufferAccess::ReadWrite,
                lifetime: ValueLifetime::Retained,
            },
        )
        .expect("counter");

    graph
        .add_node(
            "accumulate",
            Program::wrapped(
                vec![
                    BufferDecl::read("delta", 0, DataType::U32).with_count(1),
                    BufferDecl::storage("counter", 1, BufferAccess::ReadWrite, DataType::U32)
                        .with_count(1),
                    BufferDecl::output("snapshot", 2, DataType::U32).with_count(1),
                ],
                [1, 1, 1],
                vec![
                    Node::store(
                        "counter",
                        Expr::u32(0),
                        Expr::add(
                            Expr::load("counter", Expr::u32(0)),
                            Expr::load("delta", Expr::u32(0)),
                        ),
                    ),
                    Node::store(
                        "snapshot",
                        Expr::u32(0),
                        Expr::load("counter", Expr::u32(0)),
                    ),
                ],
            ),
            vec![
                GraphInput {
                    buffer: "delta".into(),
                    value: delta,
                    contract: ValueContract {
                        dtype: DataType::U32,
                        shape: vec![ShapeDim::Known(1)],
                        access: BufferAccess::ReadOnly,
                        lifetime: ValueLifetime::Invocation,
                    },
                },
                GraphInput {
                    buffer: "counter".into(),
                    value: counter_0,
                    contract: ValueContract {
                        dtype: DataType::U32,
                        shape: vec![ShapeDim::Known(1)],
                        access: BufferAccess::ReadWrite,
                        lifetime: ValueLifetime::Retained,
                    },
                },
            ],
            vec![
                GraphOutput {
                    buffer: "counter".into(),
                    name: "counter.1".into(),
                    contract: ValueContract {
                        dtype: DataType::U32,
                        shape: vec![ShapeDim::Known(1)],
                        access: BufferAccess::ReadWrite,
                        lifetime: ValueLifetime::Retained,
                    },
                    retained_successor_of: Some(counter_0),
                },
                GraphOutput {
                    buffer: "snapshot".into(),
                    name: "accum_out".into(),
                    contract: ValueContract {
                        dtype: DataType::U32,
                        shape: vec![ShapeDim::Known(1)],
                        access: BufferAccess::WriteOnly,
                        lifetime: ValueLifetime::Output,
                    },
                    retained_successor_of: None,
                },
            ],
        )
        .expect("accumulate node");

    artifact_fixtures::compile_graph(graph, 0)
}

#[test]
fn retained_session_manages_state_machine_generations_atomically() {
    let artifact = stateful_accumulator_artifact();
    let payload = fixture_target_payload(&artifact, FORMAT, vec![10, 20, 30]);
    let mut envelope = ArtifactEnvelope::new(artifact.clone());
    envelope
        .attach_target_payload(payload)
        .expect("envelope payload");
    let materializer = SessionFixtureMaterializer::new("sm-backend", "sm-device", FORMAT);

    let session = ArtifactSession::from_envelope_with_materializer(
        &SM_REGISTRATION,
        envelope,
        materializer.clone(),
    )
    .expect("session");

    let counter_0_id = session.resource("counter.0").expect("counter.0 resource");
    let counter_1_id = session.resource("counter.1").expect("counter.1 resource");

    // 1. Missing initial retained state must fail
    let mut bad_envelope = ArtifactEnvelope::new(artifact.clone());
    bad_envelope
        .attach_target_payload(fixture_target_payload(&artifact, FORMAT, vec![10, 20, 30]))
        .expect("bad envelope payload");
    let bad_init = RetainedArtifactSession::new(
        ArtifactSession::from_envelope_with_materializer(
            &SM_REGISTRATION,
            bad_envelope,
            materializer,
        )
        .expect("bad session"),
        BTreeMap::from([(counter_0_id, 0_u32.to_le_bytes().to_vec())]),
    );
    assert!(
        bad_init.is_err(),
        "must reject incomplete initial retained state"
    );

    // 2. Proper initialization
    let initial_state = BTreeMap::from([
        (counter_0_id, 0_u32.to_le_bytes().to_vec()),
        (counter_1_id, 0_u32.to_le_bytes().to_vec()),
    ]);
    let retained_session = RetainedArtifactSession::new(session, initial_state)
        .expect("valid retained session initialization");

    assert_eq!(retained_session.artifact().unwrap(), artifact.digest());

    // 3. State replacement validation
    let replacement = BTreeMap::from([
        (counter_0_id, 100_u32.to_le_bytes().to_vec()),
        (counter_1_id, 100_u32.to_le_bytes().to_vec()),
    ]);
    retained_session
        .replace_retained(replacement)
        .expect("valid state replacement");

    let bad_replacement = BTreeMap::from([(ArtifactValueId(999), vec![0])]);
    assert!(
        retained_session.replace_retained(bad_replacement).is_err(),
        "must reject invalid retained replacement keys"
    );
}

#[test]
fn backend_error_classification_exhaustive_closure() {
    let test_cases = [
        (
            BackendError::DeviceLost {
                backend: "test".into(),
                device: "gpu".into(),
                generation: 1,
                message: "lost".into(),
            },
            ErrorCode::DeviceLost,
            RetryClass::NewDevice,
        ),
        (
            BackendError::DeviceOutOfMemory {
                requested: 1024,
                available: 512,
            },
            ErrorCode::DeviceOutOfMemory,
            RetryClass::SameDevice,
        ),
        (
            BackendError::PoisonedLock {
                lock_error: "state".into(),
            },
            ErrorCode::PoisonedLock,
            RetryClass::SameDevice,
        ),
        (
            BackendError::UnsupportedFeature {
                name: "feature".into(),
                backend: "test".into(),
            },
            ErrorCode::UnsupportedFeature,
            RetryClass::Never,
        ),
        (
            BackendError::KernelCompileFailed {
                backend: "test".into(),
                compiler_message: "compiler syntax error".into(),
            },
            ErrorCode::KernelCompileFailed,
            RetryClass::Never,
        ),
        (
            BackendError::DispatchFailed {
                code: Some(1),
                message: "timeout".into(),
            },
            ErrorCode::DispatchFailed,
            RetryClass::Never,
        ),
        (
            BackendError::InvalidProgram {
                fix: "invalid".into(),
            },
            ErrorCode::InvalidProgram,
            RetryClass::Never,
        ),
        (
            BackendError::CooperativeResidencyExceeded {
                grid_blocks: 128,
                resident_limit: 64,
                detail: "cooperative limits".into(),
            },
            ErrorCode::CooperativeResidencyExceeded,
            RetryClass::Never,
        ),
        (
            BackendError::ExecutionAborted {
                stage: "execution",
                reason: "cancelled".into(),
            },
            ErrorCode::ExecutionAborted,
            RetryClass::Never,
        ),
        (
            BackendError::new("unclassified failure"),
            ErrorCode::Unknown,
            RetryClass::Never,
        ),
    ];

    let mut tested_codes = Vec::new();
    for (error, expected_code, expected_class) in test_cases {
        assert_eq!(error.code(), expected_code);
        assert_eq!(classify_backend_error(&error), expected_class);
        tested_codes.push(expected_code);
    }

    for &code in ErrorCode::ALL {
        assert!(
            tested_codes.contains(&code),
            "ErrorCode::{code:?} must be explicitly classified in test_cases"
        );
    }
}
