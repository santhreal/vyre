//! Descriptor verification entry-point contracts.

use vyre_lower::descriptor_builder::{body, descriptor, lit};
use vyre_lower::*;

#[test]
fn valid_input_returns_descriptor_directly() {
    let desc = KernelDescriptor {
        id: "k".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops: vec![lit(0, 0)],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(7)],
        },
    };
    let out = verify_descriptor(&desc).unwrap();
    assert_eq!(out, desc);
}

#[test]
fn invalid_input_returns_input_failure() {
    // Descriptor with zero workgroup_size dim  -  caught by verify.
    let desc = descriptor("bad").dispatch(0, 1, 1).body(body()).build();
    let r = verify_descriptor(&desc);
    assert!(matches!(r, Err(VerifyFailure::Input(_))));
}

#[test]
fn errors_accessor_yields_underlying() {
    let desc = descriptor("bad").dispatch(0, 1, 1).body(body()).build();
    let f = verify_descriptor(&desc).unwrap_err();
    assert_ne!(f.errors().len(), 0);
}
