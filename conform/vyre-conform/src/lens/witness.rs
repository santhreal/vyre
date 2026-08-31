//! CPU-only witness lens.

use vyre_foundation::fp_parity::{compare_operation_outputs, BufferParity};
use vyre_foundation::operation::SemanticOperation;

use crate::lens::execution::run_cpu;
use crate::lens::outcome::LensOutcome;

/// Run the witness lens over every registered fixture case.
///
/// Executes the op's `test_inputs` through `vyre_reference::reference_eval` and
/// compares the result byte-for-byte against its declared
/// `expected_output`. The oracle lives next to the op; the lens just
/// runs it.
pub fn run(entry: &SemanticOperation) -> LensOutcome {
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
