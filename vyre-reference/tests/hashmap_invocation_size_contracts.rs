//! Hashmap reference interpreter invocation-count contracts.

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

/// WHY: validation refuses a zero workgroup extent but nothing caps the product,
/// so a program whose extents multiply past `u32::MAX` is valid IR and reaches
/// the interpreter. The product used to saturate: the interpreter reported
/// `u32::MAX` invocations per workgroup, divided the dispatch by it, collapsed
/// to a single workgroup, and ran a grid the program does not describe. Each
/// case below overflows at a different factor, and the last wraps the product to
/// a small number rather than a large one, so a wrapping or saturating multiply
/// goes red instead of dispatching something plausible.
#[test]
fn a_workgroup_whose_extents_pass_what_a_u32_counts_is_refused() {
    for extents in [
        [u32::MAX, 2, 1],
        [1, u32::MAX, 2],
        [2, 1, u32::MAX],
        [1 << 16, 1 << 16, 1],
        [1 << 11, 1 << 11, 1 << 11],
    ] {
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            extents,
            vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
        );

        let refusal = vyre_reference::ReferenceRequest::standard(&program, &[])
            .outputs()
            .expect_err("Fix: a workgroup this size must be refused, not dispatched");
        assert_eq!(
            refusal.error_class(),
            vyre_reference::ReferenceErrorClass::Overflow,
            "Fix: a product that passes its type is an overflow, got: {refusal}"
        );
        assert!(
            refusal
                .to_string()
                .contains("invocations per workgroup than a u32 counts"),
            "Fix: a workgroup of {extents:?} must be refused by name, got: {refusal}"
        );
    }
}
