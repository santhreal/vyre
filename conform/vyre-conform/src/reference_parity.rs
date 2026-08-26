//! Execution of one prepared entry on one backend and comparison against the CPU
//! reference outputs.

use crate::operation_selection::PreparedEntry;
use crate::proof_scheduler::panic_message;
use crate::replay_capsule::build_replay_capsule;
use vyre_conform::witness_plan::plan_witness_inputs_into;
use vyre_conform::{convergence_lens, ProductionSession};
use vyre_conform_spec::ConformanceResult;
use vyre_foundation::fp_parity::{compare_output_buffers, BufferParity};

/// How one prepared entry reaches the backend.
///
/// A fixpoint entry has no route of its own: `convergence_lens` opens one per
/// iteration. A direct entry holds one route open across every case. Holding the
/// route inside the variant that has one is what makes the pairing of a route
/// with an iteration bound unrepresentable, so no case body has to assert it.
enum Execution {
    /// Fixpoint entry, converged by `convergence_lens` under this bound.
    Fixpoint { max_iterations: u32 },
    /// Direct entry, submitted once per case on this route.
    Direct(ProductionSession),
}

pub(crate) fn compare_backend_against_reference(
    backend: &'static vyre_driver::BackendRegistration,
    prepared: &PreparedEntry,
) -> ConformanceResult {
    let backend_id = backend.id.to_string();
    if backend.reference_oracle && !prepared.expected_is_recorded {
        return ConformanceResult {
            op_id: prepared.id.into(),
            backend_id,
            passed: false,
            message: "the backend under test is the reference interpreter and this operation records no expected outputs, so the comparison would be that interpreter against itself. Fix: record expected_output bytes for this operation, or prove it on a backend that is not the oracle.".to_string(),
            replay_capsule: None,
        };
    }
    let mut checked_cases = 0usize;
    let execution = if let Some(max_iterations) = prepared.convergence_max_iterations {
        Execution::Fixpoint { max_iterations }
    } else {
        let mut first_inputs: Vec<&[u8]> = Vec::with_capacity(prepared.input_plan.source_count());
        if let Some(first_case) = prepared.cases.first() {
            if let Err(error) =
                plan_witness_inputs_into(first_case, &prepared.input_plan, &mut first_inputs)
            {
                return ConformanceResult {
                    op_id: prepared.id.into(),
                    backend_id,
                    passed: false,
                    message: format!(
                        "witness input planning failed for first case: {error}. Fix: fixture cases must match Program buffer declarations."
                    ),
                    replay_capsule: None,
                };
            }
        }
        match ProductionSession::from_registration(&prepared.program, backend) {
            Ok(route) => Execution::Direct(route),
            Err(error) => {
                return ConformanceResult {
                    op_id: prepared.id.into(),
                    backend_id,
                    passed: false,
                    message: format!(
                        "execution route failed before case execution: {error}. Fix: repair graph compilation, target payload emission, materialization, or backend dispatch."
                    ),
                    replay_capsule: None,
                };
            }
        }
    };
    let mut backend_inputs: Vec<&[u8]> = Vec::with_capacity(prepared.input_plan.source_count());

    for (case_index, inputs) in prepared.cases.iter().enumerate() {
        let reference = &prepared.reference_cases[case_index];
        match &execution {
            Execution::Fixpoint { max_iterations } => {
                let outputs = match convergence_lens::run_fixpoint_to_convergence(
                    backend,
                    &prepared.program,
                    inputs,
                    *max_iterations,
                ) {
                    Ok(outputs) => outputs,
                    Err(error) => {
                        return ConformanceResult {
                        op_id: prepared.id.into(),
                        backend_id: backend_id.clone(),
                        passed: false,
                        message: format!(
                            "production fixpoint artifact route failed on case {case_index}: {error}. Fix: repair compiler, payload, materialization, retained bindings, or submission."
                        ),
                        replay_capsule: None,
                    };
                    }
                };

                if let BufferParity::Mismatch(detail) =
                    compare_output_buffers(&prepared.program, &outputs, reference)
                {
                    return ConformanceResult {
                    op_id: prepared.id.into(),
                    backend_id: backend_id.clone(),
                    passed: false,
                    message: format!(
                        "backend output diverged from vyre-reference after fixpoint convergence on case {case_index}: {detail}. Fix: align backend.dispatch with vyre-reference under the backend-transcendental-aware ULP window (byte-exact for non-F32, <= program-derived ULP cap for F32)."
                    ),
                    replay_capsule: Some(build_replay_capsule(
                        &backend_id,
                        prepared,
                        case_index,
                        inputs,
                        &outputs,
                        reference,
                    )),
                };
                }
            }
            Execution::Direct(route) => {
                if let Err(error) =
                    plan_witness_inputs_into(inputs, &prepared.input_plan, &mut backend_inputs)
                {
                    return ConformanceResult {
                        op_id: prepared.id.into(),
                        backend_id: backend_id.clone(),
                        passed: false,
                        message: error,
                        replay_capsule: None,
                    };
                }
                let dispatch_result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        route.submit(&backend_inputs)
                    }));
                match dispatch_result {
                    Ok(Ok(execution)) => {
                        let outputs = execution.outputs;
                        if let BufferParity::Mismatch(detail) =
                            compare_output_buffers(&prepared.program, &outputs, reference)
                        {
                            return ConformanceResult {
                            op_id: prepared.id.into(),
                            backend_id: backend_id.clone(),
                            passed: false,
                            message: format!(
                                "backend output diverged from vyre-reference on case {case_index}: {detail}. Fix: align backend.dispatch with vyre-reference under the backend-transcendental-aware ULP window (byte-exact for non-F32, <= program-derived ULP cap for F32)."
                            ),
                            replay_capsule: Some(build_replay_capsule(
                                &backend_id,
                                prepared,
                                case_index,
                                inputs,
                                &outputs,
                                reference,
                            )),
                        };
                        }
                    }
                    Ok(Err(error)) => {
                        return ConformanceResult {
                        op_id: prepared.id.into(),
                        backend_id: backend_id.clone(),
                        passed: false,
                        message: format!(
                            "backend dispatch failed on case {case_index}: {error}. Fix: make backend.dispatch execute this witness."
                        ),
                        replay_capsule: None,
                    };
                    }
                    Err(payload) => {
                        return ConformanceResult {
                        op_id: prepared.id.into(),
                        backend_id: backend_id.clone(),
                        passed: false,
                        message: format!(
                            "backend dispatch panicked on case {case_index}: {}. Fix: backend.dispatch must return BackendError instead of unwinding, then execute this witness.",
                            panic_message(payload)
                        ),
                        replay_capsule: None,
                    };
                    }
                }
            }
        }
        checked_cases += 1;
    }

    let proof = match &execution {
        Execution::Fixpoint { .. } => "through the production fixpoint artifact route",
        Execution::Direct(route) => route.proof(),
    };
    ConformanceResult {
        op_id: prepared.id.into(),
        backend_id,
        passed: true,
        message: format!("{checked_cases} witness case(s) matched vyre-reference {proof}"),
        replay_capsule: None,
    }
}
