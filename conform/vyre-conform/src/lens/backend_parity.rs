//! CPU-vs-backend byte-identity lens.

use vyre_driver::BackendRegistration;
use vyre_foundation::fp_parity::{compare_output_buffers, BufferParity};
use vyre_foundation::operation::SemanticOperation;
use vyre_libs::operation_catalog::convergence_contract;

use crate::lens::convergence;
use crate::lens::execution::run_cpu;
use crate::lens::outcome::LensOutcome;
use crate::production::ProductionSession;

/// Run the byte-identity lens over every registered fixture case.
///
/// Compiles the operation for the supplied registered target, executes the
/// authenticated artifact and CPU reference, and compares outputs under the
/// operation's declared tolerance. Missing fixtures and target failures are
/// hard failures. Stateful operations route to [`convergence::run`].
pub fn run(entry: &SemanticOperation, backend: &'static BackendRegistration) -> LensOutcome {
    // Convergence-contract ops need iterative dispatch until the state
    // stabilises; route them to the convergence lens.
    if convergence_contract(entry.id).is_some() {
        return convergence::run(entry, backend);
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
    let first_borrowed: Vec<&[u8]> = cases[0].iter().map(Vec::as_slice).collect();
    let production = match ProductionSession::compile_with_representative_inputs(
        &program,
        &first_borrowed,
        backend,
    ) {
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
