//! Unified parity harness using the `lens` module.
//!
//! Canonical library operations run through fixture witnesses and any
//! registered stateful convergence contract.

#![forbid(unsafe_code)]

use vyre_conform::lens::outcome::LensOutcome;
use vyre_conform::lens::{backend_parity, witness};

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

// If no artifact-capable target is linked, the test fails loudly.
#[test]
fn convergence_contract_reachable_for_every_registered_op() {
    // Discover every op with a ConvergenceContract and verify structural
    // invariants plus CPU-side and registered-target convergence.
    let entries = vyre_libs::operation_catalog::fixture_entries();
    let (failure_capacity, _) = entries.size_hint();
    let mut cpu_failures = Vec::with_capacity(failure_capacity);
    let backend = build_registered_backend();

    for entry in entries {
        let Some(contract) = vyre_libs::operation_catalog::convergence_contract(entry.id) else {
            continue;
        };
        assert!(
            contract.max_iterations > 0,
            "convergence contract for `{}` has max_iterations=0",
            entry.id
        );

        let Some(test_inputs) = entry.test_inputs else {
            cpu_failures.push(format!(
                "{}: no test_inputs  -  convergence lens has nothing to run.",
                entry.id
            ));
            continue;
        };
        let program = entry
            .program()
            .expect("Fix: conformance operation must provide a neutral builder");
        let cases = test_inputs();
        if cases.is_empty() {
            cpu_failures.push(format!(
                "{}: empty test_inputs fixture. Fix: convergence parity requires at least one initial state.",
                entry.id
            ));
            continue;
        }

        for (case_index, inputs) in cases.iter().enumerate() {
            match vyre_conform::convergence_lens::run_cpu_fixpoint_to_convergence(
                &program,
                inputs,
                contract.max_iterations,
            ) {
                Ok(_) => {}
                Err(error) => {
                    cpu_failures.push(format!(
                        "{} case {}: CPU convergence loop failed: {error}",
                        entry.id, case_index
                    ));
                }
            }
            {
                match vyre_conform::convergence_lens::run_fixpoint_to_convergence(
                    backend,
                    &program,
                    inputs,
                    contract.max_iterations,
                ) {
                    Ok(_) => {}
                    Err(error) => {
                        cpu_failures.push(format!(
                            "{} case {}: backend convergence loop failed: {error}",
                            entry.id, case_index
                        ));
                    }
                }
            }
        }
    }

    assert!(
        cpu_failures.is_empty(),
        "convergence lens failures:\n  - {}",
        cpu_failures.join("\n  - ")
    );
}

/// H5: verify that the cpu_vs_backend lens accepts small ULP divergence
/// for F32 transcendentals instead of failing with raw byte comparison.
#[test]
fn cpu_vs_backend_accepts_transcendental_ulp_divergence() {
    fn build_sin_program() -> vyre::ir::Program {
        use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};
        Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::F32).with_count(1)],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), Expr::sin(Expr::f32(1.0)))],
        )
    }

    fn sin_inputs() -> Vec<Vec<Vec<u8>>> {
        vec![vec![]]
    }

    let entry = vyre_foundation::operation::SemanticOperation {
        id: "vyre-conform::synthetic::sin_ulp_probe",
        semantic_version: 1,
        signature: None,
        tier: vyre_foundation::operation::OperationTier::External,
        category: Some("conform"),
        build: Some(build_sin_program),
        test_inputs: Some(sin_inputs),
        expected_output: None,
        laws: &[],
        tolerance: vyre_foundation::operation::TolerancePolicy::EXACT,
        geometry_requirements: None,
        source_file: file!(),
        explicit_effects: None,
        explicit_capabilities: None,
    };

    let backend = build_registered_backend();
    let outcome = backend_parity::run(&entry, backend);
    assert!(
        outcome.is_pass(),
        "cpu_vs_backend lens should accept small ULP divergence for sin(1.0), but got: {outcome:?}"
    );
}

fn build_registered_backend() -> &'static vyre_driver::BackendRegistration {
    vyre_conform::production::live_test_backend().expect(
        "Fix: a dispatch-capable backend must be registered for convergence lens. \
         Link a concrete driver crate into the test binary.",
    )
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
