//! The lens contracts that acquire a real device.
//!
//! Split out of `lens_parity` because both tests here reach hardware through
//! `vyre_conform::production::live_test_backend`, which the reference-only
//! contracts in that file do not. A default `cargo test -p vyre-conform` runs
//! on every hosted matrix leg, and acquiring CUDA where no driver exists aborts
//! the process from inside cudarc rather than returning an error, so the whole
//! leg went red naming a third-party source line.
//!
//! The target is admitted by `device-tests`, which `conform.yml` turns on for
//! the runner that has the device.

#![forbid(unsafe_code)]

use vyre_conform::lens::backend_parity;

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
