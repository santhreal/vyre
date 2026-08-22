//! Flat-byte dispatch through reference-owned facets.

use crate::execution::call::invoke_cpu_ref;
use crate::ReferenceError;
use vyre_foundation::operation::OperationRegistry;

/// Run one canonical semantic operation through its registered reference facet.
///
/// # Errors
///
/// Returns [`ReferenceError`] when the operation is unknown or has
/// no portable reference implementation.
pub fn dispatch_op(op_id: &str, input: &[u8], output: &mut Vec<u8>) -> Result<(), ReferenceError> {
    if OperationRegistry::global().get(op_id).is_none() {
        return Err(ReferenceError::new(format!(
            "reference interpreter: operation `{op_id}` is not registered. Fix: link the crate that submits its canonical OperationRegistration."
        )));
    }
    let execute = crate::reference_fn(op_id).ok_or_else(|| {
        ReferenceError::new(format!(
            "unsupported reference dispatch for `{op_id}`: no ReferenceFacet is registered. Fix: submit one reference-owned facet or route to a target that declares support."
        ))
    })?;
    invoke_cpu_ref(op_id, execute, input, output)
}
