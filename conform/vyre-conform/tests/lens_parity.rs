//! Unified parity harness using the `lens` module.
//!
//! Canonical library operations run through fixture witnesses and any
//! registered stateful convergence contract.

#![forbid(unsafe_code)]

use vyre_conform::lens::{self, LensOutcome};

#[cfg(feature = "gpu")]
use vyre_driver_metal as _;
#[cfg(feature = "gpu")]
use vyre_driver_wgpu as _;

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
        let outcome = lens::witness(&entry);
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
    };

    let backend = build_registered_backend();
    let outcome = lens::cpu_vs_backend(&entry, backend);
    assert!(
        outcome.is_pass(),
        "cpu_vs_backend lens should accept small ULP divergence for sin(1.0), but got: {outcome:?}"
    );
}

fn build_registered_backend() -> &'static vyre_driver::BackendRegistration {
    force_link_backend_inventory();
    let selected = std::env::var("VYRE_BACKEND")
        .ok()
        .filter(|value| !value.trim().is_empty());
    vyre_driver::backend::registered_backends()
        .expect("valid backend registry")
        .iter()
        .find(|registration| {
            vyre_driver::backend::backend_dispatches(registration.id)
                .expect("valid backend registry")
                && selected
                    .as_deref()
                    .is_none_or(|backend| registration.id == backend)
        })
        .expect(
            "Fix: a dispatch-capable backend must be registered for convergence lens. \
             Link a concrete driver crate into the test binary.",
        )
}

fn force_link_backend_inventory() {
    #[cfg(feature = "gpu")]
    {
        std::hint::black_box(vyre_driver_metal::METAL_BACKEND_ID);
    }
}
