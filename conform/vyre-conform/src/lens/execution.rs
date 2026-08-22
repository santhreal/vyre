//! How a lens executes one program: on the reference interpreter, on a
//! registered backend, and the dispatch shape both sides agree on.

use vyre_driver::{BackendError, BackendRegistration, DispatchConfig};
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

/// The dispatch configuration a parity fixture runs one program under.
///
/// A 1D workgroup needs no override. A non-1D one is accepted only when a
/// single workgroup covers every writable element, because the fixture path
/// dispatches exactly one.
pub fn dispatch_config_for(program: &Program) -> Result<DispatchConfig, String> {
    crate::dispatch_grid::config_for_program(program)
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

/// Compile `program` for `backend` once, for a loop that submits repeatedly.
pub fn production_session(
    backend: &'static BackendRegistration,
    program: &Program,
    representative_inputs: &[&[u8]],
) -> Result<ProductionSession, LoopError> {
    ProductionSession::compile_with_representative_inputs(program, representative_inputs, backend)
        .map_err(|error| LoopError::Backend(BackendError::new(error.to_string())))
}
