//! `bundle_cert` contracts over the public `vyre_conform` surface.

use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_conform::{issue_bundle_cert, verify_bundle_against_reference, BundleCertError};
use vyre_conform_spec::ConformanceCase;
use vyre_primitives::wire::pack_u32_slice as bytes_u32;

/// Smallest non-trivial Program we can dispatch on the reference:
/// copy the first element of a read-only u32 buffer into a
/// read-write buffer. Good enough to exercise the byte-identity
/// pipeline without leaning on a specific feature gate.
fn copy_first_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::storage("output", 1, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "output",
            Expr::u32(0),
            Expr::load("input", Expr::u32(0)),
        )],
    )
}

fn output_first_copy_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::output("output", 0, DataType::U32).with_count(1),
            BufferDecl::storage("input", 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "output",
            Expr::u32(0),
            Expr::load("input", Expr::u32(0)),
        )],
    )
}

fn sample_corpus() -> Vec<ConformanceCase> {
    vec![
        ConformanceCase {
            name: "alpha".into(),
            inputs: vec![bytes_u32(&[1, 2, 3, 4]), bytes_u32(&[0, 0])],
        },
        ConformanceCase {
            name: "beta".into(),
            inputs: vec![bytes_u32(&[7, 8, 9, 10]), bytes_u32(&[0, 0])],
        },
    ]
}

#[test]
fn issue_populates_all_fields() {
    let program = copy_first_program();
    let cert = issue_bundle_cert(
        &program,
        &sample_corpus(),
        "2026-04-21T00:00:00Z",
        "sig",
        "pub",
    )
    .expect("Fix: issue; restore this invariant before continuing.");
    assert_eq!(cert.witness_count, 2);
    assert_eq!(cert.bundle_blake3.len(), 64);
    assert_eq!(cert.corpus_blake3.len(), 64);
    assert_eq!(cert.reference_output_blake3.len(), 64);
}

#[test]
fn rejects_empty_corpus() {
    let program = copy_first_program();
    let err =
        issue_bundle_cert(&program, &[], "t", "s", "p").expect_err("empty corpus must reject");
    assert!(matches!(err, BundleCertError::EmptyCorpus));
}

#[test]
fn corpus_hash_is_order_independent() {
    let program = copy_first_program();
    let forward = sample_corpus();
    let reversed: Vec<ConformanceCase> = forward.iter().cloned().rev().collect();
    let cert_a = issue_bundle_cert(&program, &forward, "t", "s", "p").unwrap();
    let cert_b = issue_bundle_cert(&program, &reversed, "t", "s", "p").unwrap();
    assert_eq!(cert_a.corpus_blake3, cert_b.corpus_blake3);
    assert_eq!(
        cert_a.reference_output_blake3,
        cert_b.reference_output_blake3
    );
}

#[test]
fn changing_input_changes_cert() {
    let program = copy_first_program();
    let corpus_a = sample_corpus();
    let mut corpus_b = sample_corpus();
    corpus_b[0].inputs[0] = bytes_u32(&[99, 99, 99, 99]);
    let cert_a = issue_bundle_cert(&program, &corpus_a, "t", "s", "p").unwrap();
    let cert_b = issue_bundle_cert(&program, &corpus_b, "t", "s", "p").unwrap();
    assert_ne!(cert_a.corpus_blake3, cert_b.corpus_blake3);
    assert_ne!(
        cert_a.reference_output_blake3,
        cert_b.reference_output_blake3
    );
}

#[test]
fn changing_program_changes_bundle_hash() {
    let prog_a = copy_first_program();
    let prog_b = {
        // Copy with a different entry node (store to output[1])
        // to produce a different wire hash.
        Program::wrapped(
            vec![
                BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(4),
                BufferDecl::storage("output", 1, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(2),
            ],
            [1, 1, 1],
            vec![Node::store(
                "output",
                Expr::u32(1),
                Expr::load("input", Expr::u32(0)),
            )],
        )
    };
    let cert_a = issue_bundle_cert(&prog_a, &sample_corpus(), "t", "s", "p").unwrap();
    let cert_b = issue_bundle_cert(&prog_b, &sample_corpus(), "t", "s", "p").unwrap();
    assert_ne!(cert_a.bundle_blake3, cert_b.bundle_blake3);
}

#[test]
fn reference_self_verifies() {
    let program = copy_first_program();
    let corpus = sample_corpus();
    let cert = issue_bundle_cert(&program, &corpus, "t", "s", "p").unwrap();
    verify_bundle_against_reference(&cert, &program, &corpus)
        .expect("Fix: reference self-verifies; restore this invariant before continuing.");
}

#[test]
fn bundle_cert_accepts_logical_witness_order_after_output_buffer() {
    let program = output_first_copy_program();
    let corpus = vec![ConformanceCase {
        name: "logical-input-only".into(),
        inputs: vec![bytes_u32(&[0xA5A5_5A5A])],
    }];

    let cert = issue_bundle_cert(&program, &corpus, "t", "s", "p")
        .expect("Fix: bundle issue must plan logical witness inputs, not raw buffer order.");
    verify_bundle_against_reference(&cert, &program, &corpus)
        .expect("Fix: bundle verify must reuse the same planned witness stream as issue.");
}

#[test]
fn bundle_reference_verifier_accepts_logical_witness_order_after_output_buffer() {
    let program = output_first_copy_program();
    let corpus = vec![ConformanceCase {
        name: "backend-logical-input-only".into(),
        inputs: vec![bytes_u32(&[0xFEED_FACE])],
    }];
    let cert = issue_bundle_cert(&program, &corpus, "t", "s", "p")
        .expect("Fix: bundle issue must certify planned logical witness inputs.");
    verify_bundle_against_reference(&cert, &program, &corpus).expect(
        "Fix: bundle reference verification must dispatch the planned logical witness stream.",
    );
}

#[test]
fn bundle_cert_rejects_omitted_runtime_sized_read_write_witness() {
    let program = Program::wrapped(
        vec![BufferDecl::storage(
            "scratch",
            0,
            BufferAccess::ReadWrite,
            DataType::U32,
        )],
        [1, 1, 1],
        Vec::<Node>::new(),
    );
    let corpus = vec![ConformanceCase {
        name: "missing-runtime-scratch".into(),
        inputs: Vec::new(),
    }];

    let error = issue_bundle_cert(&program, &corpus, "t", "s", "p")
        .expect_err("Fix: omitted runtime-sized read-write witnesses must reject.");

    assert!(
        matches!(error, BundleCertError::WitnessPlanningFailed { .. }),
        "Fix: planner errors must stay explicit, got: {error}"
    );
    assert!(
        error
            .to_string()
            .contains("runtime-sized read-write buffer"),
        "Fix: planner error must name the dynamic read-write contract, got: {error}"
    );
}

#[test]
fn verify_catches_bundle_drift() {
    let program = copy_first_program();
    let corpus = sample_corpus();
    let cert = issue_bundle_cert(&program, &corpus, "t", "s", "p").unwrap();

    let drifted = Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::storage("output", 1, BufferAccess::ReadWrite, DataType::U32).with_count(2),
        ],
        [1, 1, 1],
        vec![Node::store(
            "output",
            Expr::u32(1),
            Expr::load("input", Expr::u32(0)),
        )],
    );
    let err = verify_bundle_against_reference(&cert, &drifted, &corpus)
        .expect_err("bundle drift must reject");
    assert!(matches!(err, BundleCertError::BundleHashMismatch { .. }));
}

#[test]
fn verify_catches_corpus_drift() {
    let program = copy_first_program();
    let corpus = sample_corpus();
    let cert = issue_bundle_cert(&program, &corpus, "t", "s", "p").unwrap();

    let mut drifted_corpus = corpus.clone();
    drifted_corpus[0].inputs[0] = bytes_u32(&[42, 42, 42, 42]);
    let err = verify_bundle_against_reference(&cert, &program, &drifted_corpus)
        .expect_err("corpus drift must reject");
    assert!(matches!(err, BundleCertError::CorpusHashMismatch { .. }));
}

#[test]
fn verify_catches_output_drift() {
    // Craft a cert whose reference_output_blake3 is wrong, then
    // assert verify surfaces OutputHashMismatch  -  not
    // BundleHashMismatch / CorpusHashMismatch.
    let program = copy_first_program();
    let corpus = sample_corpus();
    let mut cert = issue_bundle_cert(&program, &corpus, "t", "s", "p").unwrap();
    cert.reference_output_blake3 = "00".repeat(32);
    let err = verify_bundle_against_reference(&cert, &program, &corpus)
        .expect_err("output drift must reject");
    assert!(matches!(err, BundleCertError::OutputHashMismatch { .. }));
}
