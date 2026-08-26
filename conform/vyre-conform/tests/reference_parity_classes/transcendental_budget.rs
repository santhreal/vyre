//! WHY: closes the class "the production artifact route refuses a transcendental
//! for want of a parity window", which took 21 ops out of the conformance
//! certificate on cuda with `CUDA PTX ``Exp``/``tanh`` lowering requires
//! approximate transcendental instructions, but ulp_budget is not positive`.
//!
//! The owner turned out to be the PTX emitter, not the routes that call it. PTX
//! has only an approximate `tanh`, `ex2`, `lg2`, `sin` and `cos`, so gating
//! admission on a budget named a choice that does not exist and made every route
//! compensate for it. The refusal is gone, and the budget now governs only the
//! ops PTX offers in two forms. `vyre_foundation::fp_parity` still owns the
//! window itself, which is what `prove` holds the numbers to.
//!
//! The roster is the operation registry crossed with the backend registry: every
//! op whose program carries the transcendental window, compiled through every
//! backend that registers a target compiler and a materializer, which is the
//! route a caller's program takes in a release. A new transcendental op, or a new
//! backend whose dialect forgets the window, turns this red without anyone
//! editing a list.
//!
//! What it does not catch: numerical accuracy. This proves the payload builds,
//! not that the approximate instruction lands inside the window it was granted;
//! `prove` compares the output and the per-op ULP audit measures the distance.

use vyre_conform::production::ProductionSession;
use vyre_conform::witness_plan::{plan_witness_inputs_into, WitnessInputPlan};
use vyre_driver::BackendRegistration;
use vyre_foundation::fp_parity::{f32_ulp_tolerance, BACKEND_TRANSCENDENTAL_ULP_BUDGET};
use vyre_foundation::ir::Program;
use vyre_registry_link::backend::live_backend_registry;
use vyre_registry_link::operation::live_operation_registry;

/// Whether this program is held to the transcendental window rather than the
/// elementary one. The classification is `fp_parity`'s, read rather than copied.
fn carries_the_transcendental_window(program: &Program) -> bool {
    f32_ulp_tolerance(program) == BACKEND_TRANSCENDENTAL_ULP_BUDGET
}

fn artifact_route_backends() -> Vec<&'static BackendRegistration> {
    let registrations =
        live_backend_registry().expect("Fix: the backend registry must start before it is judged.");
    let backends = registrations
        .iter()
        .filter(|registration| {
            registration.target_compiler.is_some() && registration.materializer.is_some()
        })
        .collect::<Vec<_>>();
    assert!(
        !backends.is_empty(),
        "Fix: no linked backend registers the artifact route, so this test judges nothing. Link a concrete driver crate."
    );
    backends
}

#[test]
fn every_transcendental_op_builds_a_payload_on_every_artifact_route_backend() {
    let backends = artifact_route_backends();
    let programs = live_operation_registry()
        .iter()
        .filter_map(|operation| {
            let program = operation.program()?;
            if !carries_the_transcendental_window(&program) {
                return None;
            }
            let test_inputs = operation.test_inputs?;
            let cases = test_inputs();
            let first_case = cases.into_iter().next()?;
            Some((operation.id, program, first_case))
        })
        .collect::<Vec<_>>();
    assert!(
        !programs.is_empty(),
        "Fix: no registered op carries the transcendental parity window, so this class has no members left to hold. Read the classification from vyre-foundation/src/fp_parity.rs before deleting this."
    );

    let mut refused = Vec::new();
    for (op_id, program, first_case) in &programs {
        let input_plan = match WitnessInputPlan::for_program(program) {
            Ok(plan) => plan,
            Err(error) => {
                refused.push(format!("(plan, {op_id}): {error}"));
                continue;
            }
        };
        let mut backend_inputs = Vec::with_capacity(input_plan.source_count());
        if let Err(error) = plan_witness_inputs_into(first_case, &input_plan, &mut backend_inputs) {
            refused.push(format!("(witness, {op_id}): {error}"));
            continue;
        }
        for backend in &backends {
            match ProductionSession::from_registration(program, backend)
                .and_then(|session| session.submit(&backend_inputs))
            {
                Ok(_) => {}
                Err(error) => refused.push(format!("({}, {op_id}): {error}", backend.id)),
            }
        }
    }
    assert!(
        refused.is_empty(),
        "Fix: the production artifact route cannot build a payload for a transcendental op. A dialect emitter must not refuse a descriptor for want of a parity window: the window chooses between two instructions where the target offers both, and is never an admission gate. See vyre-emit-ptx/tests/ulp_budget_is_not_an_admission_gate.rs.\n{}",
        refused.join("\n")
    );
}

#[test]
fn the_parity_window_is_positive_for_every_registered_op() {
    let mut zero = Vec::new();
    for operation in live_operation_registry().iter() {
        let Some(program) = operation.program() else {
            continue;
        };
        if f32_ulp_tolerance(&program) == 0 {
            zero.push(operation.id);
        }
    }
    assert!(
        zero.is_empty(),
        "Fix: a zero parity window makes an emitter refuse every approximate lowering, and contraction is a documented backend right, so the floor is positive for every program. Ops with a zero window: {}",
        zero.join(", ")
    );
}
