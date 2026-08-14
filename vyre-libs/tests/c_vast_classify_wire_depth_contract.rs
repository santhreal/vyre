//! Wire-depth contract for the GPU C VAST classifier program.

#![cfg(feature = "c-parser")]

mod support;

use support::optimizer::assert_optimizer_is_idempotent;
use vyre::ir::{Expr, Program};
use vyre_libs::parsing::c::parse::vast::c11_classify_vast_node_kinds;

#[test]
fn c_vast_classifier_wire_roundtrip_respects_decode_depth_contract() {
    let program = c11_classify_vast_node_kinds("vast_nodes", Expr::u32(9), "out_typed_vast_nodes");
    let wire = program
        .to_wire()
        .expect("VAST classifier must encode to the canonical wire format");
    let decoded = Program::from_wire(&wire)
        .expect("VAST classifier wire form must stay within canonical decode depth");
    assert_eq!(program, decoded);
}

#[test]
fn c_vast_classifier_optimizer_is_idempotent() {
    assert_optimizer_is_idempotent(c11_classify_vast_node_kinds(
        "vast_nodes",
        Expr::u32(9),
        "out_typed_vast_nodes",
    ));
}
