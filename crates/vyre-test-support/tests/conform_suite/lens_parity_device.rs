//! The lens contracts that acquire a real device.
//!
//! Split out of `lens_parity` because this contract reaches hardware through
//! `vyre_test_support::conform::production::live_test_backend`, which the reference-only
//! contracts in that file do not. A default `cargo test -p vyre-test-support` runs
//! on every hosted matrix leg, and acquiring CUDA where no driver exists aborts
//! the process from inside cudarc rather than returning an error, so the whole
//! leg went red naming a third-party source line.
//!
//! The target is admitted by `device-tests`, which `conform.yml` turns on for
//! the runner that has the device.

#![forbid(unsafe_code)]

use vyre_test_support::conform::lens::backend_parity;

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
        numeric: vyre_foundation::numeric::NumericContract::EXACT,
        geometry_requirements: vyre_foundation::GeometryRequirements::agnostic(),
        source_file: file!(),
        explicit_effects: None,
        explicit_capabilities: None,
        absence: Some(vyre_foundation::operation::AbsenceDecision::Uncharacterized),
    };

    let backend = build_registered_backend();
    let outcome = backend_parity::run(&entry, backend);
    assert!(
        outcome.is_pass(),
        "cpu_vs_backend lens should accept small ULP divergence for sin(1.0), but got: {outcome:?}"
    );
}

fn build_registered_backend() -> &'static vyre_runtime::BackendRegistration {
    vyre_test_support::conform::production::live_test_backend().expect(
        "Fix: a dispatch-capable backend must be registered for backend_parity lens. \
         Link a concrete backend into the test binary.",
    )
}
