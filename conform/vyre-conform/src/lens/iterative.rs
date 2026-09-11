//! The shape every iterative lens shares: prepare a fixture, then run the
//! reference and the backend loop over each case and compare the final states.

use vyre_driver::BackendRegistration;
use vyre_foundation::fp_parity::{compare_output_buffers, BufferParity};
use vyre_foundation::ir::Program;
use vyre_foundation::operation::SemanticOperation;

use crate::lens::buffer_state::project_output_buffers;
use crate::lens::execution::LoopError;
use crate::lens::outcome::LensOutcome;

/// One fixture prepared for an iterative lens.
pub struct IterativeLensSetup {
    /// The op program every iteration dispatches.
    pub program: Program,
    /// One initial state vector per fixture case.
    pub cases: Vec<Vec<Vec<u8>>>,
}

/// Read the fixture and semantic program for `entry`.
pub fn prepare_iterative_lens(
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
    Ok(IterativeLensSetup { program, cases })
}

/// Run both loops over every case and compare the projected final states.
pub fn compare_iterative_lens_cases(
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
