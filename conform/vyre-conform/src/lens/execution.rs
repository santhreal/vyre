//! How a lens executes one program: on the reference interpreter, on a
//! registered backend, and the dispatch shape both sides agree on.

use vyre_driver::{BackendError, BackendRegistration, DispatchConfig};
use vyre_foundation::ir::{BufferAccess, Program};
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
