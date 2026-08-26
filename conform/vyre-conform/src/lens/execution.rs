//! How a lens executes one program: on the reference interpreter, on a
//! registered backend, and the dispatch shape both sides agree on.

use vyre_driver::{BackendError, BackendRegistration};
use vyre_foundation::ir::Program;
use vyre_reference::value::Value;
use vyre_reference::ReferenceError;

use crate::production::ProductionSession;

/// Execute `program` on the reference interpreter and return its output bytes.
pub fn run_cpu(program: &Program, inputs: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, ReferenceError> {
    let values: Vec<Value> = inputs.iter().cloned().map(Value::from).collect();
    let outputs = vyre_reference::reference_eval(program, &values)?;
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
