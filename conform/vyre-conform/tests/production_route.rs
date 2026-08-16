//! Production conformance must exercise compiler, payload, materializer, ABI, and submission.

#![cfg(feature = "gpu")]

use std::time::{Duration, Instant};

use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_conform::production::{run_bounded_step, ProductionError};
use vyre_conform::ProductionSession;
use vyre_driver::backend_dispatches;
use vyre_registry_link::backend::live_backend_registry;

/// Ceiling on one whole session lifecycle, compile through drop.
///
/// Tighter than [`vyre_conform::production::PRODUCTION_STEP_DEADLINE`] so the
/// lifecycle bound is what reports a wedged drop, and far above the milliseconds
/// a one-store program needs on any linked backend.
const LIFECYCLE_DEADLINE: Duration = Duration::from_secs(90);

/// Value the single-store lifecycle program writes.
const LIFECYCLE_OUTPUT: u32 = 7;

/// Operation identity the lifecycle programs carry, so a bound that expires
/// names something a reader can find.
const LIFECYCLE_OP_ID: &str = "vyre-conform::production_route::session_lifecycle";

#[test]
fn wgpu_production_route_executes_canonical_artifact() {
    let registration = live_backend_registry()
        .expect("valid backend registry")
        .iter()
        .find(|registration| registration.id == "wgpu")
        .expect("Fix: the gpu feature must link the wgpu registration");
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
    );

    let session = ProductionSession::compile(&program, registration)
        .expect("Fix: production conformance compilation and materialization must succeed");
    let outputs = session
        .submit(&[])
        .expect("Fix: typed artifact submission must succeed");

    assert_eq!(outputs, vec![7_u32.to_le_bytes().to_vec()]);
    assert_ne!(session.artifact_digest().0, [0; 32]);
    assert_ne!(session.payload_digest().0, [0; 32]);
}

#[test]
fn cuda_production_route_executes_line_index() {
    let registrations = live_backend_registry().expect("valid backend registry");
    let Some(registration) = registrations.iter().find(|reg| reg.id == "cuda") else {
        return;
    };
    if !backend_dispatches("cuda").expect("valid backend registry") {
        return;
    }
    let source = b"ab\ncd";
    let n = source.len() as u32;
    let program = vyre_libs::text::line_index("source", "lines", n);

    let session = ProductionSession::compile(&program, registration)
        .expect("Fix: production conformance compilation and materialization must succeed for line_index on CUDA");
    let mut u32_input = Vec::with_capacity(source.len() * 4);
    for &b in source {
        u32_input.extend_from_slice(&(b as u32).to_le_bytes());
    }
    let outputs = session
        .submit(&[&u32_input])
        .expect("Fix: line_index artifact submission must succeed on CUDA");

    let lines_index = program
        .output_buffer_indices()
        .iter()
        .position(|&idx| program.buffers()[idx as usize].name() == "lines")
        .expect("lines output index");
    let out_u32s: Vec<u32> = outputs[lines_index]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    assert_eq!(&out_u32s[..source.len()], &[0, 0, 0, 1, 1]);
}

#[test]
fn cuda_production_route_reports_traps_on_malformed_inputs() {
    let registrations = live_backend_registry().expect("valid backend registry");
    let Some(registration) = registrations.iter().find(|reg| reg.id == "cuda") else {
        return;
    };
    if !backend_dispatches("cuda").expect("valid backend registry") {
        return;
    }
    let program = Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(1),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![
            Node::if_then(
                Expr::ne(Expr::load("input", Expr::u32(0)), Expr::u32(0)),
                vec![Node::trap(
                    Expr::load("input", Expr::u32(0)),
                    "deliberate malformed-input trap",
                )],
            ),
            Node::store("out", Expr::u32(0), Expr::u32(0)),
        ],
    )
    .with_entry_op_id("vyre-conform::production_route::conditional_trap");

    let session = ProductionSession::compile(&program, registration)
        .expect("Fix: safe finalist inputs must compile a trap-declaring program on CUDA");
    let malformed = 1_u32.to_le_bytes();
    let error = session
        .submit(&[&malformed])
        .expect_err("malformed input must trap on CUDA");
    let message = error.to_string();
    assert!(
        message.contains("cuda dispatch trapped")
            && message.contains("deliberate malformed-input trap"),
        "Fix: CUDA production execution must report the device trap and its tag, got: {message}"
    );
}

/// Every registered artifact backend finishes a whole session lifecycle, drop
/// included, inside [`LIFECYCLE_DEADLINE`].
///
/// Compilation and submission return values a test can compare; releasing the
/// session returns nothing, so a defect there shows up only as a process that
/// never comes back. The CUDA graph guards took their context liveness from a
/// sibling field of the structs holding them, and both structs declare that
/// field first, so dropping the session released the context before
/// `cuGraphExecDestroy` ran: the destroy then blocked forever on freed driver
/// memory. This asserts the lifecycle terminates, which is the only observable
/// that failure has, and it asserts it for every backend that materializes an
/// artifact rather than the one that happened to break.
#[test]
fn every_artifact_backend_finishes_a_session_lifecycle_including_drop() {
    let registrations = live_backend_registry()
        .expect("valid backend registry")
        .iter()
        .filter(|registration| {
            !registration.reference_oracle
                && registration.target_compiler.is_some()
                && registration.materializer.is_some()
        })
        .collect::<Vec<_>>();
    assert!(
        !registrations.is_empty(),
        "Fix: the gpu feature must link at least one backend that compiles and materializes an artifact."
    );

    let mut exercised = Vec::new();
    for registration in registrations {
        if !backend_dispatches(registration.id).expect("valid backend registry") {
            continue;
        }
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![
                Node::store("out", Expr::u32(0), Expr::u32(LIFECYCLE_OUTPUT)),
                Node::Return,
            ],
        )
        .with_entry_op_id(LIFECYCLE_OP_ID);
        let outputs = run_bounded_step(
            "session lifecycle",
            LIFECYCLE_OP_ID,
            registration.id,
            LIFECYCLE_DEADLINE,
            move || {
                let session = ProductionSession::compile(&program, registration)?;
                let outputs = session.submit(&[])?;
                // The release is the subject: it has to run inside the bound.
                drop(session);
                Ok(outputs)
            },
        )
        .unwrap_or_else(|error| {
            panic!("Fix: {error}");
        });
        assert_eq!(
            outputs,
            vec![LIFECYCLE_OUTPUT.to_le_bytes().to_vec()],
            "Fix: {} must return the single stored word from the lifecycle program.",
            registration.id
        );
        exercised.push(registration.id);
    }

    assert!(
        !exercised.is_empty(),
        "Fix: no linked artifact backend could dispatch on this host, so the session lifecycle was never exercised. Provide a dispatch-capable device or unlink the backend."
    );
}

/// A step that never returns is reported, not awaited.
///
/// The bound is the whole mechanism the lifecycle test rests on, so it is
/// exercised against work that is guaranteed never to finish: without the
/// deadline this test would hang, which is the failure it exists to convert into
/// a named error.
#[test]
fn a_bounded_step_that_never_returns_is_reported_with_its_operation_and_backend() {
    let backend = live_backend_registry()
        .expect("valid backend registry")
        .iter()
        .map(|registration| registration.id)
        .next()
        .expect("Fix: at least one backend must be linked to name in a bounded step.");
    let deadline = Duration::from_millis(50);
    let started = Instant::now();
    let error =
        run_bounded_step::<()>("wedged step", LIFECYCLE_OP_ID, backend, deadline, || loop {
            std::thread::park();
        })
        .expect_err(
            "Fix: a bounded step whose work never returns must fail, not block the caller.",
        );

    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(5),
        "Fix: a bounded step must return once its deadline elapses; this one took {elapsed:?} for a {deadline:?} ceiling."
    );
    match error {
        ProductionError::Deadline {
            step,
            op_id,
            backend: reported_backend,
            deadline: reported_deadline,
        } => {
            assert_eq!(step, "wedged step");
            assert_eq!(op_id, LIFECYCLE_OP_ID);
            assert_eq!(reported_backend, backend);
            assert_eq!(reported_deadline, deadline);
        }
        other => panic!(
            "Fix: an expired bounded step must report ProductionError::Deadline, got {other:?}."
        ),
    }
}
