//! The `dispatch` subcommand: sequential single-backend conformance over selected ops.

use crate::backend_selection::backend_registration;
use crate::operation_selection::{prepare_entry, select_entries, unified_entries, PreparedEntry};
use crate::reference_parity::compare_backend_against_reference;
use vyre_conform::oracle::OracleSession;
use vyre_conform::witness_plan::plan_witness_inputs_into;
use vyre_conform_spec::ConformanceResult;
use vyre_foundation::fp_parity::{compare_output_buffers, BufferParity};
pub(crate) fn dispatch_pairs(
    backend_id: &str,
    ops: &str,
) -> Result<Vec<ConformanceResult>, String> {
    let entries = unified_entries();
    let selected_entries = select_entries(&entries, ops, None)?;
    let mut pairs = Vec::with_capacity(selected_entries.len());
    let backend_id = backend_id.to_string();

    for entry in selected_entries {
        let prepared = match prepare_entry(entry) {
            Ok(prepared) => prepared,
            Err(error) => {
                pairs.push(ConformanceResult {
                    op_id: entry.id.into(),
                    backend_id: backend_id.clone(),
                    passed: false,
                    message: error,
                    replay_capsule: None,
                });
                continue;
            }
        };
        if is_oracle_backend(&backend_id) {
            pairs.push(dispatch_oracle(&prepared, &backend_id));
            continue;
        }
        let backend = match backend_registration(&backend_id) {
            Ok(backend) => backend,
            Err(error) => {
                pairs.push(ConformanceResult {
                    op_id: entry.id.into(),
                    backend_id: backend_id.clone(),
                    passed: false,
                    message: format!(
                        "backend acquisition failed before dispatch: {error}. Fix: isolate or reset the backend after the preceding failing op, then repair the op that poisoned device state."
                    ),
                    replay_capsule: None,
                });
                continue;
            }
        };
        pairs.push(compare_backend_against_reference(backend, &prepared));
    }

    Ok(pairs)
}

fn is_oracle_backend(backend_id: &str) -> bool {
    backend_id == "cpu-ref" || backend_id == "reference" || backend_id == "oracle"
}

fn dispatch_oracle(prepared: &PreparedEntry, backend_id: &str) -> ConformanceResult {
    let session = OracleSession::new(prepared.program.clone());
    let mut planned_inputs = Vec::with_capacity(prepared.input_plan.source_count());
    for (case_index, inputs) in prepared.cases.iter().enumerate() {
        if let Err(error) =
            plan_witness_inputs_into(inputs, &prepared.input_plan, &mut planned_inputs)
        {
            return ConformanceResult {
                op_id: prepared.id.into(),
                backend_id: backend_id.to_string(),
                passed: false,
                message: format!("witness input planning failed for case {case_index}: {error}"),
                replay_capsule: None,
            };
        }
        let outputs = if let Some(max_iterations) = prepared.convergence_max_iterations {
            let planned_owned: Vec<Vec<u8>> =
                planned_inputs.iter().map(|slice| slice.to_vec()).collect();
            match vyre_conform::convergence_lens::run_cpu_fixpoint_to_convergence(
                &prepared.program,
                &planned_owned,
                max_iterations,
            ) {
                Ok(outputs) => outputs,
                Err(error) => {
                    return ConformanceResult {
                        op_id: prepared.id.into(),
                        backend_id: backend_id.to_string(),
                        passed: false,
                        message: format!("oracle fixpoint failed on case {case_index}: {error}"),
                        replay_capsule: None,
                    };
                }
            }
        } else {
            match session.execute(&planned_inputs) {
                Ok(outputs) => outputs,
                Err(error) => {
                    return ConformanceResult {
                        op_id: prepared.id.into(),
                        backend_id: backend_id.to_string(),
                        passed: false,
                        message: format!("oracle evaluation failed on case {case_index}: {error}"),
                        replay_capsule: None,
                    };
                }
            }
        };
        let expected = &prepared.reference_cases[case_index];
        if let BufferParity::Mismatch(detail) =
            compare_output_buffers(&prepared.program, &outputs, expected)
        {
            return ConformanceResult {
                op_id: prepared.id.into(),
                backend_id: backend_id.to_string(),
                passed: false,
                message: format!(
                    "oracle output diverged on case {case_index}: {detail}. Fix: align reference implementation with expected output fixture."
                ),
                replay_capsule: None,
            };
        }
    }
    ConformanceResult {
        op_id: prepared.id.into(),
        backend_id: backend_id.to_string(),
        passed: true,
        message: format!(
            "{} witness case(s) passed through reference oracle",
            prepared.cases.len()
        ),
        replay_capsule: None,
    }
}
