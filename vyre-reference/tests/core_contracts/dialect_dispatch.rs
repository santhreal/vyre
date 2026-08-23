//! Reference facet dispatch contracts.
//!
//! Covers the four outcomes `dispatch_op` distinguishes: an operation that was
//! never registered, a registered operation with a facet, a registered
//! operation without one, and a facet that panics part way through writing.

use vyre_foundation::dialect_lookup::Signature;
use vyre_foundation::operation::{OperationRegistration, OperationTier};
use vyre_reference::dialect_dispatch::dispatch_op;
use vyre_reference::ReferenceFacet;

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
    OperationRegistration::new_unconstrained(ECHO_ID, OperationTier::External, None, None, None)
        .with_signature(EMPTY_SIGNATURE)
        .with_category("test")
}
inventory::submit! {
    OperationRegistration::new_unconstrained(PANIC_ID, OperationTier::External, None, None, None)
        .with_signature(EMPTY_SIGNATURE)
        .with_category("test")
}
inventory::submit! {
    OperationRegistration::new_unconstrained(MISSING_ID, OperationTier::External, None, None, None)
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
