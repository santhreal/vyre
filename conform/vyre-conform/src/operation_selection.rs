//! Registered operation entries, shard selection over them, and preparation of each
//! entry into an executable case with CPU reference outputs.

use crate::proof_options::ShardSpec;
use crate::witness_fixtures::{
    backend_dispatch_inputs_with_plan_into, backend_dispatch_plan, synthesize_witness_cases,
    BackendDispatchPlan, FixtureCases, FixtureFn,
};
use vyre_conform::{convergence_lens, dispatch_grid};
use vyre_reference::value::Value;

#[derive(Clone, Copy)]
pub(crate) struct UnifiedEntry {
    pub(crate) id: &'static str,
    build: Option<fn() -> vyre::Program>,
    test_inputs: Option<FixtureFn>,
    expected_output: Option<FixtureFn>,
}

pub(crate) struct PreparedEntry {
    pub(crate) id: &'static str,
    pub(crate) program: vyre::Program,
    pub(crate) dispatch_config: vyre_driver::DispatchConfig,
    pub(crate) cases: FixtureCases,
    pub(crate) reference_cases: FixtureCases,
    pub(crate) input_plan: BackendDispatchPlan,
    pub(crate) convergence_max_iterations: Option<u32>,
}

pub(crate) fn select_entries(
    all_entries: &[UnifiedEntry],
    ops_filter: &str,
    shard: Option<ShardSpec>,
) -> Result<Vec<UnifiedEntry>, String> {
    let mut selected = Vec::new();
    let mut matched_ops_filter = false;
    let mut executable_index = 0usize;
    for entry in all_entries.iter().copied() {
        if ops_filter != "all" && entry.id != ops_filter {
            continue;
        }
        matched_ops_filter = true;
        if entry.build.is_none() {
            continue;
        }
        if let Some(shard) = shard {
            if executable_index % shard.count != shard.index {
                executable_index += 1;
                continue;
            }
        }
        executable_index += 1;
        selected.push(entry);
    }
    if ops_filter != "all" && !matched_ops_filter {
        return Err(format!(
            "unknown op `{ops_filter}`. Fix: pass `--ops all` or one registered semantic operation id."
        ));
    }
    if ops_filter != "all" && selected.is_empty() {
        return Err(format!(
            "operation `{ops_filter}` is registered as signature-only and has no neutral Program builder, so it is not executable conformance input. Fix: select an operation with a canonical Program builder."
        ));
    }
    if selected.is_empty() {
        return Err(
            "proof selection matched zero executable ops. Fix: choose a shard that contains at least one operation with a canonical Program builder or remove `--shard`."
                .to_string(),
        );
    }
    Ok(selected)
}

pub(crate) fn unified_entries() -> Vec<UnifiedEntry> {
    vyre_registry_link::operation::live_operation_registry()
        .iter()
        .map(|entry| UnifiedEntry {
            id: entry.id,
            build: entry.build,
            test_inputs: entry.test_inputs,
            expected_output: entry.expected_output,
        })
        .collect()
}

pub(crate) fn prepare_entry(entry: UnifiedEntry) -> Result<PreparedEntry, String> {
    let program = entry
        .build
        .ok_or_else(|| format!("{} has no neutral Program builder", entry.id))?();
    let dispatch_config = dispatch_grid::config_for_program(&program)?;
    let cases = match entry.test_inputs {
        Some(test_inputs) => test_inputs(),
        None => synthesize_witness_cases(&program)?,
    };
    // CRITIQUE_CONFORM_2026-04-23 H4: `compare_backend_against_reference`
    // returned `passed: true` with message "0 witness case(s) matched"
    // when test_inputs() produced an empty vector  -  an op that registered
    // a witness-input function returning `vec![]` received a passing
    // certificate with zero coverage, defeating the entire witness
    // discipline. Reject up front with a named Fix: hint so the author
    // fixes the fixture.
    if cases.is_empty() {
        return Err("empty witness fixture. Fix: op has zero witness cases  -  empty fixtures are not coverage. Populate test_inputs() with at least one case before running `vyre-conform dispatch`.".to_string());
    }
    let expected_cases = entry
        .expected_output
        .map(|expected_output| expected_output());
    if let Some(expected_cases) = &expected_cases {
        if expected_cases.len() != cases.len() {
            return Err(format!(
                "expected_output case count {} does not match test_inputs case count {}. Fix: every witness case must have exactly one oracle case.",
                expected_cases.len(),
                cases.len()
            ));
        }
    }
    let input_plan = backend_dispatch_plan(&program)?;
    let convergence_max_iterations = vyre_libs::operation_catalog::convergence_contract(entry.id)
        .map(|contract| contract.max_iterations);
    let reference_cases = prepare_reference_cases(
        entry.id,
        &program,
        &cases,
        &input_plan,
        expected_cases,
        convergence_max_iterations,
    )?;

    Ok(PreparedEntry {
        id: entry.id,
        program,
        dispatch_config,
        cases,
        reference_cases,
        input_plan,
        convergence_max_iterations,
    })
}

fn prepare_reference_cases(
    op_id: &str,
    program: &vyre::Program,
    cases: &FixtureCases,
    input_plan: &BackendDispatchPlan,
    expected_cases: Option<FixtureCases>,
    convergence_max_iterations: Option<u32>,
) -> Result<FixtureCases, String> {
    let mut reference_cases = Vec::with_capacity(cases.len());
    if let Some(max_iterations) = convergence_max_iterations {
        for (case_index, inputs) in cases.iter().enumerate() {
            let outputs = convergence_lens::run_cpu_fixpoint_to_convergence(
                program,
                inputs,
                max_iterations,
            )
            .map_err(|error| {
                format!(
                    "{op_id}: CPU reference fixpoint loop failed while preparing case {case_index}: {error}. Fix: repair the witness or CPU reference before running backend parity."
                )
            })?;
            reference_cases.push(outputs);
        }
        return Ok(reference_cases);
    }

    if let Some(expected_cases) = expected_cases {
        return Ok(expected_cases);
    }

    let mut reference_values = Vec::with_capacity(program.buffers().len());
    let mut planned_inputs: Vec<&[u8]> = Vec::with_capacity(input_plan.source_count());
    for (case_index, inputs) in cases.iter().enumerate() {
        backend_dispatch_inputs_with_plan_into(inputs, input_plan, &mut planned_inputs).map_err(
            |error| {
                format!(
                    "{op_id}: reference input planning failed while preparing case {case_index}: {error}"
                )
            },
        )?;
        reference_values.clear();
        for input in &planned_inputs {
            reference_values.push(Value::from(*input));
        }
        let outputs = vyre_reference::reference_eval(program, &reference_values)
            .map_err(|error| {
                format!(
                    "{op_id}: reference dispatch failed while preparing case {case_index}: {error}. Fix: repair the witness or CPU reference before running backend parity."
                )
            })?
            .into_iter()
            .map(|value| value.to_bytes())
            .collect::<Vec<_>>();
        reference_cases.push(outputs);
    }
    Ok(reference_cases)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre::ir::{BufferAccess, BufferDecl, DataType, Node, Program};

    fn test_program() -> Program {
        Program::wrapped(Vec::new(), [1, 1, 1], Vec::new())
    }

    /// WHY: signature-only semantic records belong in catalogs but cannot
    /// become false conformance failures or executable shard members.
    #[test]
    fn selection_excludes_signature_only_operations() {
        let entries = [
            UnifiedEntry {
                id: "core.signature_only",
                build: None,
                test_inputs: None,
                expected_output: None,
            },
            UnifiedEntry {
                id: "core.executable",
                build: Some(test_program),
                test_inputs: None,
                expected_output: None,
            },
        ];

        let selected = select_entries(&entries, "all", None).expect("one executable operation");
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].id, "core.executable");

        let error = match select_entries(&entries, "core.signature_only", None) {
            Ok(_) => panic!("signature-only operations must remain non-executable"),
            Err(error) => error,
        };
        assert!(error.contains("registered as signature-only"), "{error}");
    }

    #[test]
    fn prepare_reference_cases_uses_planned_zeroed_read_write_inputs() {
        let program = Program::wrapped(
            vec![
                BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(1),
                BufferDecl::storage("scratch", 1, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(1),
            ],
            [1, 1, 1],
            vec![Node::store(
                "scratch",
                vyre::ir::Expr::u32(0),
                vyre::ir::Expr::load("input", vyre::ir::Expr::u32(0)),
            )],
        );
        let input_plan = backend_dispatch_plan(&program)
            .expect("Fix: static read-write zero-fill planning must succeed.");
        let cases = vec![vec![1u32.to_le_bytes().to_vec()]];

        let reference_cases = prepare_reference_cases(
            "test.planned_reference",
            &program,
            &cases,
            &input_plan,
            None,
            None,
        )
        .expect("Fix: reference preparation must use planned zeroed read-write inputs.");

        assert_eq!(
            reference_cases,
            vec![vec![1u32.to_le_bytes().to_vec()]],
            "Fix: prove reference preparation must match the input stream used by backend dispatch."
        );
    }
}
