//! Execution of one prepared entry on one backend and comparison against the CPU
//! reference outputs.

use crate::operation_selection::PreparedEntry;
use crate::proof_scheduler::panic_message;
use crate::replay_capsule::build_replay_capsule;
use vyre_conform::fp_parity::{compare_output_buffers, BufferParity};
use vyre_conform::witness_plan::plan_witness_inputs_into;
use vyre_conform::{convergence_lens, ProductionSession};
use vyre_conform_spec::ConformanceResult;

pub(crate) fn compare_backend_against_reference(
    backend: &'static vyre_driver::BackendRegistration,
    prepared: &PreparedEntry,
) -> ConformanceResult {
    let backend_id = backend.id.to_string();
    let mut checked_cases = 0usize;
    let production = if prepared.convergence_max_iterations.is_none() {
        match ProductionSession::compile(&prepared.program, backend) {
            Ok(session) => Some(session),
            Err(error) => {
                return ConformanceResult {
                    op_id: prepared.id.into(),
                    backend_id,
                    passed: false,
                    message: format!(
                        "production compiler route failed before case execution: {error}. Fix: repair graph compilation, target payload emission, or materialization."
                    ),
                    replay_capsule: None,
                };
            }
        }
    } else {
        None
    };
    let mut backend_inputs: Vec<&[u8]> = Vec::with_capacity(prepared.input_plan.source_count());

    for (case_index, inputs) in prepared.cases.iter().enumerate() {
        let reference = &prepared.reference_cases[case_index];
        if let Some(max_iterations) = prepared.convergence_max_iterations {
            let outputs = match convergence_lens::run_fixpoint_to_convergence(
                backend,
                &prepared.program,
                inputs,
                max_iterations,
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
        } else {
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
            let dispatch_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                production
                    .as_ref()
                    .expect("production session initialized")
                    .submit(&backend_inputs)
            }));
            match dispatch_result {
                Ok(Ok(outputs)) => {
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
        checked_cases += 1;
    }

    ConformanceResult {
        op_id: prepared.id.into(),
        backend_id,
        passed: true,
        message: format!(
            "{checked_cases} witness case(s) matched vyre-reference through canonical artifact submission"
        ),
        replay_capsule: None,
    }
}
