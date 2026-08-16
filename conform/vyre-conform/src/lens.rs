//! Reusable conform lenses: ways of comparing backend output to a truth
//! oracle, one primitive per semantic.
//!
//! Every parity test runs a fixture witness, compares reference and target
//! execution, or drives a stateful operation to its registered convergence
//! bound.

use vyre_driver::{BackendError, BackendRegistration, DispatchConfig};
use vyre_foundation::ir::{BufferAccess, Program};
use vyre_foundation::operation::SemanticOperation;
use vyre_libs::operation_catalog::convergence_contract;
use vyre_reference::value::Value;
use vyre_reference::ReferenceError;

use vyre_foundation::fp_parity::{compare_operation_outputs, compare_output_buffers, BufferParity};
use crate::production::ProductionSession;

/// Outcome of running one lens against one op.
#[derive(Debug)]
pub enum LensOutcome {
    /// Lens passed  -  op output matched the oracle for every case.
    Pass {
        /// Number of input cases that were compared.
        cases: usize,
    },
    /// Lens failed  -  op diverged from the oracle on the referenced case.
    Fail {
        /// Zero-based case index of the first divergence.
        case_index: usize,
        /// Rendered failure detail.
        detail: String,
    },
}

impl LensOutcome {
    /// True only when the lens passed (ran and matched the oracle).
    ///
    /// Missing coverage is represented as [`LensOutcome::Fail`], so a
    /// passing lens always performed real comparisons.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        matches!(self, LensOutcome::Pass { .. })
    }

    /// True only when the lens actually ran and matched the oracle.
    #[must_use]
    pub fn is_pass(&self) -> bool {
        matches!(self, LensOutcome::Pass { .. })
    }
}

fn run_cpu(program: &Program, inputs: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, ReferenceError> {
    let values: Vec<Value> = inputs.iter().cloned().map(Value::from).collect();
    let outputs = vyre_reference::reference_eval(program, &values)?;
    Ok(outputs.into_iter().map(|value| value.to_bytes()).collect())
}

fn dispatch_config_for(program: &Program) -> Result<DispatchConfig, String> {
    let mut config = DispatchConfig::default();
    let workgroup = program.workgroup_size();
    for (axis, size) in workgroup.into_iter().enumerate() {
        if size == 0 {
            return Err(format!(
                "workgroup_size[{axis}] is 0. Fix: parity dispatch requires every workgroup dimension to be >= 1 before backend dispatch."
            ));
        }
    }
    if workgroup[1] == 1 && workgroup[2] == 1 {
        return Ok(config);
    }

    let lanes = u64::from(workgroup[0])
        .checked_mul(u64::from(workgroup[1]))
        .and_then(|lanes| lanes.checked_mul(u64::from(workgroup[2])))
        .ok_or_else(|| {
            format!(
                "workgroup_size {workgroup:?} overflows u64 lane accounting. Fix: use a valid backend workgroup shape."
            )
        })?;
    let max_writable_count = program
        .buffers()
        .iter()
        .filter(|decl| matches!(decl.access(), BufferAccess::ReadWrite) || decl.is_output())
        .map(|decl| u64::from(decl.count()))
        .max()
        .unwrap_or(1);

    if max_writable_count > lanes {
        return Err(format!(
            "non-1D workgroup_size {workgroup:?} has {lanes} lanes but the largest writable buffer has {max_writable_count} elements. Fix: register an explicit dispatch grid for this op instead of relying on the one-workgroup parity fixture path."
        ));
    }

    config.grid_override = Some([1, 1, 1]);
    Ok(config)
}

/// CPU-only witness lens.
///
/// Executes the op's `test_inputs` through `vyre_reference::reference_eval` and
/// compares the result byte-for-byte against its declared
/// `expected_output`. The oracle lives next to the op; the lens just
/// runs it.
pub fn witness(entry: &SemanticOperation) -> LensOutcome {
    let Some(test_inputs) = entry.test_inputs else {
        return LensOutcome::Fail {
            case_index: 0,
            detail: format!(
                "{}: no test_inputs  -  witness lens has nothing to run. Fix: register a fixture.",
                entry.id
            ),
        };
    };
    let Some(expected_fn) = entry.expected_output else {
        return LensOutcome::Fail {
            case_index: 0,
            detail: format!(
                "{}: no expected_output  -  witness lens has no oracle. Fix: register a fixture.",
                entry.id
            ),
        };
    };

    let Some(program) = entry.program() else {
        return LensOutcome::Fail {
            case_index: 0,
            detail: format!(
                "{}: no neutral builder  -  witness lens has no program to run. Fix: register a builder.",
                entry.id
            ),
        };
    };
    let cases = test_inputs();
    let expected = expected_fn();
    if cases.is_empty() {
        return LensOutcome::Fail {
            case_index: 0,
            detail: format!(
                "{}: empty test_inputs fixture. Fix: empty fixtures are zero coverage.",
                entry.id
            ),
        };
    }
    if expected.is_empty() {
        return LensOutcome::Fail {
            case_index: 0,
            detail: format!(
                "{}: empty expected_output fixture. Fix: empty oracles are zero coverage.",
                entry.id
            ),
        };
    }
    if cases.len() != expected.len() {
        return LensOutcome::Fail {
            case_index: 0,
            detail: format!(
                "witness vector count mismatch: {} test_inputs vs {} expected_output sets.",
                cases.len(),
                expected.len()
            ),
        };
    }

    for (index, (inputs, expected_buffers)) in cases.iter().zip(expected.iter()).enumerate() {
        match run_cpu(&program, inputs) {
            Ok(outputs) => {
                if let BufferParity::Mismatch(detail) =
                    compare_operation_outputs(entry.id, &program, &outputs, expected_buffers)
                {
                    return LensOutcome::Fail {
                        case_index: index,
                        detail: format!(
                            "CPU reference output diverged from declared expected_output: {detail}\n\
                             ACTUAL:\n{:?}\nEXPECTED:\n{:?}\n\
                             Fix: regenerate the witness via `./cargo_full run --bin xtask -- trace-f32 {}` or repair the reference.",
                            outputs, expected_buffers, entry.id
                        ),
                    };
                }
            }
            Err(error) => {
                return LensOutcome::Fail {
                    case_index: index,
                    detail: format!("CPU reference failed: {error}"),
                };
            }
        }
    }

    LensOutcome::Pass { cases: cases.len() }
}

/// CPU-vs-backend byte-identity lens.
///
/// Compiles the operation for the supplied registered target, executes the
/// authenticated artifact and CPU reference, and compares outputs under the
/// operation's declared tolerance. Missing fixtures and target failures are
/// hard failures. Stateful operations route to [`convergence`].
pub fn cpu_vs_backend(
    entry: &SemanticOperation,
    backend: &'static BackendRegistration,
) -> LensOutcome {
    // Convergence-contract ops need iterative dispatch until the state
    // stabilises; route them to the convergence lens.
    if convergence_contract(entry.id).is_some() {
        return convergence(entry, backend);
    }
    let Some(test_inputs) = entry.test_inputs else {
        return LensOutcome::Fail {
            case_index: 0,
            detail: format!("{}: no test_inputs  -  byte-identity lens has nothing to run. Fix: register a fixture.", entry.id),
        };
    };

    let Some(program) = entry.program() else {
        return LensOutcome::Fail {
            case_index: 0,
            detail: format!(
                "{}: no neutral builder  -  byte-identity lens has no program to run. Fix: register a builder.",
                entry.id
            ),
        };
    };

    let cases = test_inputs();
    if cases.is_empty() {
        return LensOutcome::Fail {
            case_index: 0,
            detail: format!(
                "{}: empty test_inputs fixture. Fix: byte-identity parity requires at least one backend witness.",
                entry.id
            ),
        };
    }
    let production = match ProductionSession::compile(&program, backend) {
        Ok(session) => session,
        Err(error) => {
            return LensOutcome::Fail {
                case_index: 0,
                detail: format!(
                    "backend `{}` production compilation failed: {error}",
                    backend.id
                ),
            };
        }
    };
    for (index, inputs) in cases.iter().enumerate() {
        let cpu = match run_cpu(&program, inputs) {
            Ok(outputs) => outputs,
            Err(error) => {
                return LensOutcome::Fail {
                    case_index: index,
                    detail: format!("CPU reference failed: {error}"),
                };
            }
        };
        let borrowed_inputs: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
        let gpu = match production.submit(&borrowed_inputs) {
            Ok(outputs) => outputs,
            Err(error) => {
                return LensOutcome::Fail {
                    case_index: index,
                    detail: format!(
                        "backend `{}` artifact submission failed: {error}",
                        backend.id
                    ),
                };
            }
        };
        if let BufferParity::Mismatch(detail) = compare_output_buffers(&program, &cpu, &gpu) {
            return LensOutcome::Fail {
                case_index: index,
                detail: format!(
                    "backend `{}` diverged from CPU reference on case {index}: {detail}",
                    backend.id,
                ),
            };
        }
    }

    LensOutcome::Pass { cases: cases.len() }
}

struct IterativeLensSetup {
    program: Program,
    config: DispatchConfig,
    cases: Vec<Vec<Vec<u8>>>,
}

fn prepare_iterative_lens(
    entry: &SemanticOperation,
    lens_name: &str,
) -> Result<IterativeLensSetup, LensOutcome> {
    let Some(test_inputs) = entry.test_inputs else {
        return Err(LensOutcome::Fail {
            case_index: 0,
            detail: format!(
                "{}: no test_inputs - {lens_name} lens has nothing to run. Fix: register a fixture.",
                entry.id
            ),
        });
    };
    let Some(program) = entry.program() else {
        return Err(LensOutcome::Fail {
            case_index: 0,
            detail: format!(
                "{}: no neutral builder - {lens_name} lens has no program to run. Fix: register a builder.",
                entry.id
            ),
        });
    };
    let config = dispatch_config_for(&program).map_err(|detail| LensOutcome::Fail {
        case_index: 0,
        detail,
    })?;
    let cases = test_inputs();
    if cases.is_empty() {
        return Err(LensOutcome::Fail {
            case_index: 0,
            detail: format!(
                "{}: empty test_inputs fixture. Fix: {lens_name} parity requires at least one initial state.",
                entry.id
            ),
        });
    }
    Ok(IterativeLensSetup {
        program,
        config,
        cases,
    })
}

fn compare_iterative_lens_cases(
    setup: &IterativeLensSetup,
    backend: &'static BackendRegistration,
    loop_name: &str,
    max_iterations: u32,
    mut run_cpu_loop: impl FnMut(&[Vec<u8>]) -> Result<Vec<Vec<u8>>, LoopError>,
    mut run_backend_loop: impl FnMut(&[Vec<u8>]) -> Result<Vec<Vec<u8>>, LoopError>,
) -> LensOutcome {
    for (index, inputs) in setup.cases.iter().enumerate() {
        let cpu_final = match run_cpu_loop(inputs) {
            Ok(outputs) => outputs,
            Err(LoopError::Reference(error)) => {
                return LensOutcome::Fail {
                    case_index: index,
                    detail: format!("CPU reference failed inside {loop_name} loop: {error}"),
                };
            }
            Err(LoopError::DidNotConverge) => {
                return LensOutcome::Fail {
                    case_index: index,
                    detail: format!(
                        "CPU reference did not converge in {max_iterations} iterations. Fix: raise the contract max_iterations or shrink the fixture."
                    ),
                };
            }
            Err(LoopError::Backend(error)) => {
                return LensOutcome::Fail {
                    case_index: index,
                    detail: format!("backend failed inside {loop_name} loop: {error}"),
                };
            }
        };
        let gpu_final = match run_backend_loop(inputs) {
            Ok(outputs) => outputs,
            Err(LoopError::Reference(error)) => {
                return LensOutcome::Fail {
                    case_index: index,
                    detail: format!("CPU reference failed inside {loop_name} loop: {error}"),
                };
            }
            Err(LoopError::DidNotConverge) => {
                return LensOutcome::Fail {
                    case_index: index,
                    detail: format!(
                        "backend `{}` did not converge in {max_iterations} iterations.",
                        backend.id
                    ),
                };
            }
            Err(LoopError::Backend(error)) => {
                return LensOutcome::Fail {
                    case_index: index,
                    detail: format!(
                        "backend `{}` {loop_name} dispatch failed: {error}",
                        backend.id
                    ),
                };
            }
        };
        let cpu_outputs = match project_output_buffers(&setup.program, &cpu_final) {
            Ok(outputs) => outputs,
            Err(detail) => {
                return LensOutcome::Fail {
                    case_index: index,
                    detail: format!("CPU reference {detail}"),
                };
            }
        };
        let gpu_outputs = match project_output_buffers(&setup.program, &gpu_final) {
            Ok(outputs) => outputs,
            Err(detail) => {
                return LensOutcome::Fail {
                    case_index: index,
                    detail: format!("backend `{}` {detail}", backend.id),
                };
            }
        };
        if let BufferParity::Mismatch(detail) =
            compare_output_buffers(&setup.program, &cpu_outputs, &gpu_outputs)
        {
            return LensOutcome::Fail {
                case_index: index,
                detail: format!(
                    "backend `{}` final state diverged from CPU reference after {loop_name} loop: {detail}",
                    backend.id
                ),
            };
        }
    }
    LensOutcome::Pass {
        cases: setup.cases.len(),
    }
}

/// Convergence lens: dispatch the op repeatedly until the RW state
/// stabilises, then compare the final state to the CPU reference.
///
/// Used for ops that register a [`vyre_libs::operation_catalog::ConvergenceContract`],
/// such as graph-traversal steps whose program performs one transfer step.
/// The lens infers the `current` (RO input) and `next` (RW output)
/// buffers, copies `next` → `current` between iterations, and stops
/// when `next` stops changing.
pub fn convergence(
    entry: &SemanticOperation,
    backend: &'static BackendRegistration,
) -> LensOutcome {
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

fn infer_fixpoint_buffers(program: &Program) -> Result<(&str, &str, u32), String> {
    let ro_buffers: Vec<_> = program
        .buffers()
        .iter()
        .filter(|d| d.access() == BufferAccess::ReadOnly)
        .collect();
    let rw_buffers: Vec<_> = program
        .buffers()
        .iter()
        .filter(|d| d.access() == BufferAccess::ReadWrite)
        .collect();

    let next = rw_buffers
        .last()
        .ok_or_else(|| "no ReadWrite buffer found for fixpoint next".to_string())?
        .name();

    let next_count = rw_buffers
        .last()
        .ok_or_else(|| "no ReadWrite buffer found for fixpoint next".to_string())?
        .count();

    let current_decl = ro_buffers
        .iter()
        .copied()
        .filter(|decl| decl.count() == next_count)
        .min_by_key(|decl| fixpoint_current_score(decl.name(), next))
        .ok_or_else(|| {
            format!("no ReadOnly fixpoint current buffer matches `{next}` count={next_count}")
        })?;
    let current = current_decl.name();
    let current_count = current_decl.count();

    if current_count != next_count {
        return Err(format!(
            "fixpoint buffers `{current}` (count={current_count}) and `{next}` (count={next_count}) must match",
        ));
    }

    Ok((current, next, current_count))
}

/// Rank a candidate current buffer against the selected fixpoint next buffer.
#[must_use]
pub fn fixpoint_current_score(current: &str, next: &str) -> u8 {
    if let Some(expected) = next.strip_suffix("out").map(|prefix| format!("{prefix}in")) {
        if current == expected {
            return 0;
        }
    }
    let expected_current = next.replace("next", "current");
    if expected_current != next && current == expected_current {
        return 0;
    }
    if current.contains("current") || current.contains("frontier") || current.ends_with("in") {
        return 1;
    }
    if current.contains("tag") || current.contains("kind") || current.contains("offset") {
        return 8;
    }
    4
}

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

#[derive(Debug)]
enum LoopError {
    Reference(ReferenceError),
    Backend(BackendError),
    DidNotConverge,
}

fn production_session(
    backend: &'static BackendRegistration,
    program: &Program,
) -> Result<ProductionSession, LoopError> {
    ProductionSession::compile(program, backend)
        .map_err(|error| LoopError::Backend(BackendError::new(error.to_string())))
}

fn merge_rw(state: &mut [Vec<u8>], outputs: &[Vec<u8>], program: &Program) {
    // Reference and production artifact execution return writable buffers in
    // canonical binding order. Walk the declarations in the same order.
    let mut out_iter = outputs.iter();
    for (slot, decl) in state.iter_mut().zip(program.buffers().iter()) {
        if matches!(decl.access(), BufferAccess::ReadWrite) {
            if let Some(next) = out_iter.next() {
                *slot = next.clone();
            }
        }
    }
}

fn index_of_buffer(program: &Program, name: &str) -> Option<usize> {
    program
        .buffers()
        .iter()
        .position(|decl| decl.name() == name)
}

/// Project a full convergence/fixpoint state vector down to just the
/// program's declared output buffers (`ReadWrite`/`WriteOnly`), in
/// `output_buffer_indices` order, the exact shape
/// [`compare_output_buffers`] requires (it asserts the comparison vectors
/// have one entry per declared output).
///
/// The iterative lenses carry the FULL buffer state across iterations
/// (every read-only input plus the read-write frontier) so they can copy
/// `next` → `current` between steps. The only buffers the backend
/// actually computes are the outputs; the read-only inputs are
/// host-managed and identical on both sides by construction, so comparing
/// them adds nothing. Returns an explicit `Err` (never a silently short
/// vector) if `state` is missing a declared output slot, so a malformed
/// fixture surfaces loudly instead of degrading into a confusing
/// length-mismatch downstream.
fn project_output_buffers(program: &Program, state: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, String> {
    program
        .output_buffer_indices()
        .iter()
        .map(|&index| {
            let slot = index as usize;
            state.get(slot).cloned().ok_or_else(|| {
                format!(
                    "convergence state has {} buffer(s) but the program declares an output at \
                     index {slot}; the fixture must supply an initial value for every program buffer.",
                    state.len()
                )
            })
        })
        .collect()
}

// Inline: covers the private `cpu_convergence`, `infer_fixpoint_buffers` and `project_output_buffers`, which no integration test can reach.
#[cfg(test)]
mod convergence_tests {
    use super::*;
    use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

    #[test]
    fn infer_fixpoint_buffers_rejects_no_rw() {
        let program = Program::wrapped(
            vec![BufferDecl::storage("x", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![],
        );
        assert!(infer_fixpoint_buffers(&program).is_err());
    }

    #[test]
    fn infer_fixpoint_buffers_matches_in_out_pair() {
        // Simulate the buffer layout of flows_to / sanitized_by.
        let program = Program::wrapped(
            vec![
                BufferDecl::storage("pg_nodes", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(4),
                BufferDecl::storage("fin", 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
                BufferDecl::storage("fout", 2, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(1),
            ],
            [1, 1, 1],
            vec![],
        );
        let (current, next, count) = infer_fixpoint_buffers(&program).expect("Fix: inference");
        assert_eq!(current, "fin");
        assert_eq!(next, "fout");
        assert_eq!(count, 1);
    }

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

    /// Mirrors the real `vyre-libs::security::flows_to` layout: many
    /// read-only graph/input buffers plus a single read-write frontier.
    /// The convergence loop carries the FULL state (7 buffers), but
    /// `compare_output_buffers` is contractually sized to the program's
    /// declared outputs. Before the projection fix the lens fed the
    /// 7-buffer state straight into the comparator, which rejected it with
    /// "program declares 1 output buffer(s), compared 7 result buffer(s)"
    /// even when CPU and GPU agreed byte-for-byte, a false parity
    /// failure. These two assertions pin both halves: the raw full state
    /// is rejected, the projected state is accepted.
    fn flows_to_shaped_program() -> Program {
        Program::wrapped(
            vec![
                BufferDecl::storage("pg_nodes", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(4),
                BufferDecl::storage("pg_edges", 1, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(4),
                BufferDecl::storage("edge_offsets", 2, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(4),
                BufferDecl::storage("sources", 3, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(4),
                BufferDecl::storage("dims", 4, BufferAccess::ReadOnly, DataType::U32).with_count(1),
                BufferDecl::storage("reach_in", 5, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(4),
                BufferDecl::storage("reach_out", 6, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(4),
            ],
            [1, 1, 1],
            vec![],
        )
    }

    #[test]
    fn project_output_buffers_selects_only_the_declared_output() {
        let program = flows_to_shaped_program();
        // Full convergence state: a distinct value per buffer so a wrong
        // projection is detectable, not coincidentally equal.
        let full_state: Vec<Vec<u8>> = (0..7u32)
            .map(|index| {
                let byte = (index + 1) as u8;
                vec![byte; if index == 4 { 4 } else { 16 }]
            })
            .collect();
        assert_eq!(program.output_buffer_indices(), &[6]);

        let projected =
            project_output_buffers(&program, &full_state).expect("Fix: projection must succeed");
        assert_eq!(
            projected.len(),
            1,
            "exactly one declared output (reach_out) must survive projection"
        );
        assert_eq!(
            projected[0], full_state[6],
            "the surviving buffer must be reach_out (the RW frontier), byte-for-byte"
        );
    }

    #[test]
    fn full_state_breaks_comparator_but_projection_restores_parity() {
        let program = flows_to_shaped_program();
        let full_state: Vec<Vec<u8>> = (0..7u32)
            .map(|index| vec![(index + 1) as u8; if index == 4 { 4 } else { 16 }])
            .collect();

        // The raw full state (CPU == GPU, both 7 buffers) is STILL rejected
        // by the comparator because it is sized to declared outputs (1).
        // This is the exact false failure the GPU parity test hit.
        match compare_output_buffers(&program, &full_state, &full_state) {
            BufferParity::Mismatch(detail) => assert!(
                detail.contains("declares 1 output buffer(s), compared 7 result buffer(s)"),
                "expected the output-count mismatch, got: {detail}"
            ),
            BufferParity::Ok => {
                panic!("full 7-buffer state must NOT satisfy a 1-output comparator")
            }
        }

        // Projecting to declared outputs first makes identical states pass.
        let cpu = project_output_buffers(&program, &full_state).expect("Fix: projection");
        let gpu = project_output_buffers(&program, &full_state).expect("Fix: projection");
        assert!(
            matches!(
                compare_output_buffers(&program, &cpu, &gpu),
                BufferParity::Ok
            ),
            "projected identical outputs must compare equal"
        );
    }

    #[test]
    fn project_output_buffers_errs_loudly_on_missing_output_slot() {
        let program = flows_to_shaped_program();
        // State truncated before the RW output index (6), a malformed
        // fixture. Projection must surface this, never silently drop it.
        let short_state: Vec<Vec<u8>> = vec![vec![0u8; 16]; 3];
        let err = project_output_buffers(&program, &short_state)
            .expect_err("missing output slot must be an error, not a silent short vector");
        assert!(
            err.contains("declares an output at index 6"),
            "error must name the missing output index: {err}"
        );
    }
}
