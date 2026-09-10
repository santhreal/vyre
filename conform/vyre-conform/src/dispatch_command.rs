//! The `dispatch` subcommand: sequential single-backend conformance over selected ops.

use crate::operation_selection::{prepare_entry, select_entries, unified_entries, PreparedEntry};
use crate::reference_parity::compare_backend_against_reference;
use vyre_conform::backend_selection::backend_registration;
use vyre_conform::oracle::OracleSession;
use vyre_conform::witness_plan::plan_witness_inputs_into;
use vyre_conform_spec::ConformanceResult;
use vyre_driver_reference::ORACLE_EXECUTOR_ID;
use vyre_foundation::fp_parity::{compare_output_buffers, BufferParity};

/// What a dispatch run executes against.
///
/// The oracle is not a backend, so it is not reachable by spelling a backend
/// id. A caller selects one or the other and the two carry different executor
/// ids into every row.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Executor<'a> {
    Backend(&'a str),
    Oracle,
}

impl<'a> Executor<'a> {
    fn executor_id(self) -> &'a str {
        match self {
            Executor::Backend(id) => id,
            Executor::Oracle => ORACLE_EXECUTOR_ID,
        }
    }
}

pub(crate) fn dispatch_pairs(
    executor: Executor<'_>,
    ops: &str,
) -> Result<Vec<ConformanceResult>, String> {
    let entries = unified_entries();
    let selected_entries = select_entries(&entries, ops, None)?;
    // The registry is a static built at link time, so a lookup inside the loop
    // answers the same question once per op. An id no linked driver registers
    // used to become one identical conformance failure per op: a spirv run
    // wrote 349 rows saying `unknown backend`, and a reader counting rows saw a
    // judged backend with 349 defects rather than a backend nothing judged.
    // The oracle answers from the reference interpreter and registers no device
    // backend, so the route is chosen once rather than re-tested per op.
    let backend = match executor {
        Executor::Oracle => None,
        Executor::Backend(id) => Some(backend_registration(id)?),
    };
    let mut pairs = Vec::with_capacity(selected_entries.len());

    for entry in selected_entries {
        let prepared = match prepare_entry(entry) {
            Ok(prepared) => prepared,
            Err(error) => {
                pairs.push(ConformanceResult {
                    op_id: entry.id.into(),
                    executor_id: executor.executor_id().into(),
                    passed: false,
                    message: error,
                    replay_capsule: None,
                });
                continue;
            }
        };

        pairs.push(match backend {
            Some(backend) => compare_backend_against_reference(backend, &prepared),
            None => dispatch_oracle(&prepared),
        });
    }

    Ok(pairs)
}

fn dispatch_oracle(prepared: &PreparedEntry) -> ConformanceResult {
    let session = OracleSession::new(prepared.program.clone());
    let mut planned_inputs = Vec::with_capacity(prepared.input_plan.source_count());
    for (case_index, inputs) in prepared.cases.iter().enumerate() {
        if let Err(error) =
            plan_witness_inputs_into(inputs, &prepared.input_plan, &mut planned_inputs)
        {
            return ConformanceResult {
                op_id: prepared.id.into(),
                executor_id: ORACLE_EXECUTOR_ID.to_string(),
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
                        executor_id: ORACLE_EXECUTOR_ID.to_string(),
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
                        executor_id: ORACLE_EXECUTOR_ID.to_string(),
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
                executor_id: ORACLE_EXECUTOR_ID.to_string(),
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
        executor_id: ORACLE_EXECUTOR_ID.to_string(),
        passed: true,
        message: format!(
            "{} witness case(s) passed through reference oracle",
            prepared.cases.len()
        ),
        replay_capsule: None,
    }
}
