//! Flat-byte dispatch through reference-owned facets.

use crate::execution::call::invoke_cpu_ref;
use crate::ReferenceError;
use vyre_foundation::operation::{OperationRegistry, OperationTier};

/// Run one canonical semantic operation through its registered reference facet.
///
/// # Errors
///
/// Returns [`ReferenceError`] when the operation is unknown, target-only, or has
/// no portable reference implementation.
pub fn dispatch_op(op_id: &str, input: &[u8], output: &mut Vec<u8>) -> Result<(), ReferenceError> {
    let operation = OperationRegistry::global().get(op_id).ok_or_else(|| {
        ReferenceError::new(format!(
            "reference interpreter: operation `{op_id}` is not registered. Fix: link the crate that submits its canonical OperationRegistration."
        ))
    })?;
    if operation.tier == OperationTier::Runtime {
        return Err(ReferenceError::new(format!(
            "unsupported capability for `{op_id}` on the reference backend: runtime operations require a target facet. Fix: select a target that advertises this operation."
        )));
    }
    let execute = crate::reference_fn(op_id).ok_or_else(|| {
        ReferenceError::new(format!(
            "unsupported reference dispatch for `{op_id}`: no ReferenceFacet is registered. Fix: submit one reference-owned facet or route to a target that declares support."
        ))
    })?;
    invoke_cpu_ref(op_id, execute, input, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ReferenceFacet;
    use vyre_foundation::dialect_lookup::Signature;
    use vyre_foundation::operation::{OperationRegistration, OperationTier};

    const ECHO_ID: &str = "test::reference_echo";
    const PANIC_ID: &str = "test::reference_panic";
    const MISSING_ID: &str = "test::reference_missing";
    const EMPTY_SIGNATURE: Signature = Signature {
        inputs: &[],
        outputs: &[],
        attrs: &[],
        bytes_extraction: false,
    };

    inventory::submit! {
        OperationRegistration::new(ECHO_ID, OperationTier::External, None, None, None)
            .with_signature(EMPTY_SIGNATURE)
            .with_category("test")
    }
    inventory::submit! {
        OperationRegistration::new(PANIC_ID, OperationTier::External, None, None, None)
            .with_signature(EMPTY_SIGNATURE)
            .with_category("test")
    }
    inventory::submit! {
        OperationRegistration::new(MISSING_ID, OperationTier::External, None, None, None)
            .with_signature(EMPTY_SIGNATURE)
            .with_category("test")
    }

    fn echo(input: &[u8], output: &mut Vec<u8>) {
        output.extend_from_slice(input);
    }

    fn panic_after_output(_: &[u8], output: &mut Vec<u8>) {
        output.extend_from_slice(&[0xde, 0xad]);
        panic!("malformed reference input");
    }

    inventory::submit! { ReferenceFacet::new(ECHO_ID, echo) }
    inventory::submit! { ReferenceFacet::new(PANIC_ID, panic_after_output) }

    #[test]
    fn unknown_operation_fails_closed() {
        let error = dispatch_op("missing::operation", &[], &mut Vec::new())
            .expect_err("unknown operation must fail");
        assert!(error.to_string().contains("OperationRegistration"));
    }

    #[test]
    fn registered_reference_facet_dispatches() {
        let mut output = Vec::new();
        dispatch_op(ECHO_ID, &[9, 8, 7], &mut output).expect("echo facet");
        assert_eq!(output, [9, 8, 7]);
    }

    #[test]
    fn missing_reference_facet_is_typed_absence() {
        let mut output = vec![0xaa];
        let error = dispatch_op(MISSING_ID, &[], &mut output).expect_err("missing facet");
        assert!(error.to_string().contains("no ReferenceFacet"));
        assert_eq!(output, [0xaa]);
    }

    #[test]
    fn panicking_facet_does_not_publish_partial_output() {
        let mut output = vec![0xaa];
        let error = dispatch_op(PANIC_ID, &[], &mut output).expect_err("panic is contained");
        assert!(error.to_string().contains("panicked"));
        assert_eq!(output, [0xaa]);
    }
}
