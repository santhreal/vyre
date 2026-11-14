//! Reference-interpreter execution helper shared across conform lenses.

use vyre_foundation::ir::Program;
use vyre_reference::ReferenceError;
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
