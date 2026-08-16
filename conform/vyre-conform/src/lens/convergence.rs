//! Convergence lens: dispatch until the read-write state stabilises.

use vyre_driver::{BackendError, BackendRegistration, DispatchConfig};
use vyre_foundation::ir::Program;
use vyre_foundation::operation::SemanticOperation;
use vyre_libs::operation_catalog::convergence_contract;

use crate::lens::buffer_state::{index_of_buffer, merge_rw};
use crate::lens::execution::{production_session, run_cpu, LoopError};
use crate::lens::fixpoint::infer_fixpoint_buffers;
use crate::lens::iterative::{compare_iterative_lens_cases, prepare_iterative_lens};
use crate::lens::outcome::LensOutcome;

/// Run the convergence lens: dispatch the op repeatedly until the RW state
/// stabilises, then compare the final state to the CPU reference.
///
/// Used for ops that register a [`vyre_libs::operation_catalog::ConvergenceContract`],
/// such as graph-traversal steps whose program performs one transfer step.
/// The lens infers the `current` (RO input) and `next` (RW output)
/// buffers, copies `next` → `current` between iterations, and stops
/// when `next` stops changing.
pub fn run(entry: &SemanticOperation, backend: &'static BackendRegistration) -> LensOutcome {
    let Some(contract) = convergence_contract(entry.id) else {
        return LensOutcome::Fail {
            case_index: 0,
            detail: format!(
                "{}: no ConvergenceContract registered for this op. Fix: register a contract or use the cpu_vs_backend lens.",
                entry.id
            ),
        };
    };
    let setup = match prepare_iterative_lens(entry, "convergence") {
        Ok(setup) => setup,
        Err(outcome) => return outcome,
    };
    let Ok((current_name, next_name, _words)) = infer_fixpoint_buffers(&setup.program) else {
        return LensOutcome::Fail {
            case_index: 0,
            detail: format!(
                "{}: could not infer fixpoint current/next buffers from program layout. Fix: ensure one RO buffer matches the last RW buffer in count.",
                entry.id
            ),
        };
    };
    let Some(current_idx) = index_of_buffer(&setup.program, current_name) else {
        return LensOutcome::Fail {
            case_index: 0,
            detail: format!(
                "{}: inferred current buffer '{current_name}' is absent from the program buffer table. Fix: keep fixpoint inference and buffer declarations in the same program contract.",
                entry.id
            ),
        };
    };
    let Some(next_idx) = index_of_buffer(&setup.program, next_name) else {
        return LensOutcome::Fail {
            case_index: 0,
            detail: format!(
                "{}: inferred next buffer '{next_name}' is absent from the program buffer table. Fix: keep fixpoint inference and buffer declarations in the same program contract.",
                entry.id
            ),
        };
    };
    compare_iterative_lens_cases(
        &setup,
        backend,
        "convergence",
        contract.max_iterations,
        |inputs| {
            cpu_convergence(
                &setup.program,
                inputs,
                contract.max_iterations,
                current_idx,
                next_idx,
            )
        },
        |inputs| {
            gpu_convergence(
                backend,
                &setup.program,
                inputs,
                contract.max_iterations,
                current_idx,
                next_idx,
                &setup.config,
            )
        },
    )
}

/// Iterate the reference interpreter until `next` stops changing.
fn cpu_convergence(
    program: &Program,
    initial_inputs: &[Vec<u8>],
    max_iterations: u32,
    current_idx: usize,
    next_idx: usize,
) -> Result<Vec<Vec<u8>>, LoopError> {
    let mut state: Vec<Vec<u8>> = initial_inputs.to_vec();
    let mut prev_next: Vec<u8> = Vec::new();
    for _ in 0..max_iterations {
        let outputs = run_cpu(program, &state).map_err(LoopError::Reference)?;
        merge_rw(&mut state, &outputs, program);
        if state.get(next_idx) == Some(&prev_next) {
            return Ok(state);
        }
        prev_next = state[next_idx].clone();
        state[current_idx] = state[next_idx].clone();
    }
    Err(LoopError::DidNotConverge)
}

/// Iterate one compiled backend artifact until `next` stops changing.
fn gpu_convergence(
    backend: &'static BackendRegistration,
    program: &Program,
    initial_inputs: &[Vec<u8>],
    max_iterations: u32,
    current_idx: usize,
    next_idx: usize,
    _config: &DispatchConfig,
) -> Result<Vec<Vec<u8>>, LoopError> {
    let mut state: Vec<Vec<u8>> = initial_inputs.to_vec();
    let mut prev_next: Vec<u8> = Vec::new();
    let production = production_session(backend, program)?;
    for _ in 0..max_iterations {
        let borrowed_state: Vec<&[u8]> = state.iter().map(Vec::as_slice).collect();
        let outputs = production
            .submit(&borrowed_state)
            .map_err(|error| LoopError::Backend(BackendError::new(error.to_string())))?;
        merge_rw(&mut state, &outputs, program);
        if state.get(next_idx) == Some(&prev_next) {
            return Ok(state);
        }
        prev_next = state[next_idx].clone();
        state[current_idx] = state[next_idx].clone();
    }
    Err(LoopError::DidNotConverge)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node};

    #[test]
    fn convergence_contract_ops_are_discoverable() {
        let convergent_ids: Vec<&str> = vyre_libs::operation_catalog::all_entries()
            .filter_map(|entry| convergence_contract(entry.id).map(|_| entry.id))
            .collect();
        assert!(
            !convergent_ids.is_empty(),
            "expected at least one ConvergenceContract-registered op"
        );
    }

    #[test]
    fn cpu_convergence_reaches_fixpoint_on_accumulating_or() {
        // Synthetic program: each invocation ORs current into next.
        // Iteration 1: next = current | next = 1 | 2 = 3
        // Iteration 2: current = 3, next = 3 | 3 = 3 → converged
        let program = Program::wrapped(
            vec![
                BufferDecl::storage("current", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(1),
                BufferDecl::storage("next", 1, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(1),
            ],
            [1, 1, 1],
            vec![Node::store(
                "next",
                Expr::u32(0),
                Expr::bitor(
                    Expr::load("current", Expr::u32(0)),
                    Expr::load("next", Expr::u32(0)),
                ),
            )],
        );
        let initial = vec![
            vec![1u8, 0, 0, 0], // current = 1
            vec![2u8, 0, 0, 0], // next = 2
        ];
        let result = cpu_convergence(&program, &initial, 10, 0, 1).unwrap();
        let final_next =
            u32::from_le_bytes([result[1][0], result[1][1], result[1][2], result[1][3]]);
        assert_eq!(final_next, 3, "should converge to stable OR of all inputs");
    }

    #[test]
    fn cpu_convergence_respects_max_iterations() {
        // Program that never converges: next = next + 1
        let program = Program::wrapped(
            vec![
                BufferDecl::storage("current", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(1),
                BufferDecl::storage("next", 1, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(1),
            ],
            [1, 1, 1],
            vec![Node::store(
                "next",
                Expr::u32(0),
                Expr::add(Expr::load("next", Expr::u32(0)), Expr::u32(1)),
            )],
        );
        let initial = vec![vec![0u8; 4], vec![0u8; 4]];
        assert!(
            cpu_convergence(&program, &initial, 5, 0, 1).is_err(),
            "non-convergent program should exhaust max_iterations"
        );
    }
}
