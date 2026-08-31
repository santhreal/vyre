//! Parity contracts that need no device.
//!
//! Canonical library operations run through fixture witnesses, and the parity
//! comparison itself is judged on buffers this file produces. The two contracts
//! that acquire a real backend live in `lens_parity_device`, which is admitted
//! by `device-tests`.

#![forbid(unsafe_code)]

use vyre_conform::lens::outcome::LensOutcome;
use vyre_conform::lens::witness;

fn report(op_id: &str, lens_name: &'static str, outcome: LensOutcome, failures: &mut Vec<String>) {
    match outcome {
        LensOutcome::Pass { cases } => {
            println!("  [{lens_name}] {op_id}: pass ({cases} cases)");
        }
        LensOutcome::Fail { case_index, detail } => {
            failures.push(format!("{lens_name} / {op_id} case {case_index}: {detail}"));
        }
    }
}

#[test]
fn every_op_passes_the_witness_lens() {
    let entries = vyre_libs::operation_catalog::fixture_entries();
    let (failure_capacity, _) = entries.size_hint();
    let mut failures = Vec::with_capacity(failure_capacity);
    let mut passed = 0usize;
    for entry in entries {
        let outcome = witness::run(&entry);
        if outcome.is_pass() {
            passed += 1;
        }
        report(entry.id, "witness", outcome, &mut failures);
    }
    assert!(
        failures.is_empty(),
        "witness lens failures:\n  - {}",
        failures.join("\n  - ")
    );
    assert!(
        passed > 0,
        "witness lens covered zero ops  -  every registered entry must provide test_inputs and expected_output."
    );
}

/// A device result that disagrees with the reference fails parity, and the
/// reference value is not handed back in its place.
///
/// WHY: the deleted `vyre-driver::shadow` path could evaluate a user program on
/// the host and return that, so a wrong device answer could be masked by a
/// right host one. Nothing may substitute now: a divergence is a refusal.
///
/// The corruption is applied to the buffer the backend produced rather than
/// obtained from a real device, because a device that returns a wrong answer on
/// demand is not something a test can arrange. What is under test is the
/// decision the lens makes about the two buffers, which is this comparison.
///
/// Does not catch: a caller that ignores the outcome and reads a buffer anyway.
/// `LensOutcome` carries none, which `an_outcome_carries_no_buffers` pins.
#[test]
fn a_corrupted_device_result_fails_parity_instead_of_being_replaced() {
    use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};
    use vyre_foundation::fp_parity::{compare_output_buffers, BufferParity};

    // U32 under the exact policy: one wrong byte is a divergence, with no
    // tolerance for a transcendental to hide behind.
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(4)],
        [4, 1, 1],
        vec![Node::store(
            "out",
            Expr::InvocationId { axis: 0 },
            Expr::mul(Expr::InvocationId { axis: 0 }, Expr::u32(7)),
        )],
    );

    let reference = vyre_conform::lens::execution::run_cpu(&program, &[])
        .expect("Fix: the reference interpreter must execute this program.");
    assert_eq!(
        reference,
        vec![vyre_primitives::wire::pack_u32_slice(&[0, 7, 14, 21])],
        "Fix: the fixture must pin a known reference result, or the corruption below proves nothing."
    );

    let honest = reference.clone();
    assert!(
        matches!(
            compare_output_buffers(&program, &reference, &honest),
            BufferParity::Ok
        ),
        "Fix: an identical device result must pass, or this test fails for the wrong reason."
    );

    // One lane wrong, in the last element, which a length-only check misses.
    let mut corrupted = reference.clone();
    corrupted[0][12] ^= 0x01;
    let BufferParity::Mismatch(detail) = compare_output_buffers(&program, &reference, &corrupted)
    else {
        panic!(
            "Fix: a device buffer that differs from the reference must be reported as a mismatch."
        );
    };
    assert!(
        !detail.is_empty(),
        "Fix: a mismatch must say which buffer diverged."
    );

    // The refusal names the divergence; it does not carry a buffer for a
    // caller to use instead.
    let outcome = LensOutcome::Fail {
        case_index: 0,
        detail,
    };
    assert!(
        !outcome.is_pass(),
        "Fix: a divergence must not be reported as a pass."
    );
}

/// No lens outcome carries output buffers, so no caller can be handed a
/// reference value in place of a device one.
///
/// WHY: substitution needs somewhere to put the substituted value. The match
/// below has no catch-all arm, so a variant added with a buffer field stops
/// this compiling rather than quietly reopening the path row 69 closed.
#[test]
fn an_outcome_carries_no_buffers() {
    for outcome in [
        LensOutcome::Pass { cases: 1 },
        LensOutcome::Fail {
            case_index: 0,
            detail: "diverged".to_string(),
        },
    ] {
        match outcome {
            LensOutcome::Pass { cases: _ } => {}
            LensOutcome::Fail {
                case_index: _,
                detail: _,
            } => {}
        }
    }
}
