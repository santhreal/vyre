//! What a resident launch takes a resource for, and what it does not.
//!
//! The projection lived in two backends, each with its own copy of the filter,
//! and only one of them was tested. It is now one function in `vyre-driver`, so
//! the rule is proved once for every backend that reads a binding order off a
//! plan.

use vyre_driver::materialize::resident_buffer_names;
use vyre_driver::BindingPlan;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Program};

/// WHY: workgroup scratch is module-internal memory, not an artifact value. A
/// resident launch must bind every host-visible role and no shared scratch, or
/// the caller is asked for a handle to memory the launch allocates itself.
#[test]
fn a_resident_launch_takes_no_resource_for_workgroup_scratch() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::F32).with_count(16),
            BufferDecl::workgroup("scratch", 16, DataType::F32),
            BufferDecl::output("output", 1, DataType::F32).with_count(16),
        ],
        [16, 1, 1],
        Vec::new(),
    );
    let plan = BindingPlan::build(&program)
        .expect("Fix: the resident projection fixture must build a binding plan.");

    let names = resident_buffer_names(&plan, &program).collect::<Vec<_>>();

    assert_eq!(names, ["input", "output"]);
}

/// WHY: the order is the binding plan's, not the declaration's, and a launch
/// handed resources in the wrong order reads the wrong memory without failing.
#[test]
fn the_names_follow_binding_order() {
    let program = Program::wrapped(
        vec![
            BufferDecl::output("late", 3, DataType::F32).with_count(4),
            BufferDecl::storage("early", 1, BufferAccess::ReadOnly, DataType::F32).with_count(4),
        ],
        [4, 1, 1],
        Vec::new(),
    );
    let plan = BindingPlan::build(&program)
        .expect("Fix: the resident projection fixture must build a binding plan.");

    let names = resident_buffer_names(&plan, &program).collect::<Vec<_>>();
    let ordered = plan
        .bindings
        .iter()
        .filter(|binding| binding.role != vyre_driver::BindingRole::Shared)
        .map(|binding| binding.name.as_ref())
        .collect::<Vec<_>>();

    assert_eq!(names, ordered);
}
