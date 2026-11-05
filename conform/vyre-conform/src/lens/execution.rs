//! How a lens executes one program: on the reference interpreter, on a
//! registered backend, and the dispatch shape both sides agree on.

use vyre_driver::{BackendError, BackendRegistration};
use vyre_foundation::ir::Program;
use vyre_reference::ReferenceError;

use crate::production::ProductionSession;

/// Execute `program` on the reference interpreter and return its output bytes.
pub fn run_cpu(program: &Program, inputs: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, ReferenceError> {
    let inputs_slices: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
    let expected = program
        .buffers()
        .iter()
        .filter(|decl| vyre_reference::is_reference_input(decl))
        .count();
    let effective_slices = if inputs_slices.len() > expected && expected > 0 {
        &inputs_slices[..expected]
    } else {
        &inputs_slices[..]
    };
    let values = vyre_reference::reference_input_values(program, effective_slices)
        .map_err(|m| ReferenceError::new(format!("input mismatch: {m}")))?;
    let outputs = vyre_reference::ReferenceRequest::standard(program, &values).outputs()?;
    Ok(outputs.into_iter().map(|value| value.to_bytes()).collect())
}

/// What went wrong inside one iteration of an iterative lens.
#[derive(Debug)]
pub enum LoopError {
    /// The reference interpreter refused the program or its state.
    Reference(ReferenceError),
    /// The backend refused the artifact or its submission.
    Backend(BackendError),
    /// The loop hit its registered iteration bound without stabilising.
    DidNotConverge,
}

/// Construct the semantic execution boundary for one iterative backend program.
pub fn production_session(
    backend: &'static BackendRegistration,
    program: &Program,
) -> Result<ProductionSession, LoopError> {
    ProductionSession::from_registration(program, backend)
        .map_err(|error| LoopError::Backend(BackendError::new(error.to_string())))
}
