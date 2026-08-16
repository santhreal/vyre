//! Contract test for `vyre_lower::lower` and `vyre_lower::WORKGROUP_SLOT_BASE` public visibility.

use vyre_foundation::ir::Program;
use vyre_lower::{lower, WORKGROUP_SLOT_BASE};

#[test]
fn lower_function_is_publicly_callable() {
    let program = Program::wrapped(vec![], [1, 1, 1], vec![]);

    // Re-exported at crate root
    let desc = lower(&program).expect("lower valid empty program");
    assert_eq!(desc.dispatch.workgroup_size, [1, 1, 1]);
}

#[test]
fn workgroup_slot_base_is_accessible() {
    assert_eq!(WORKGROUP_SLOT_BASE, 1 << 24);
}
