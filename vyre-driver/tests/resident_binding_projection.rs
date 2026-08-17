//! What a resident launch takes a resource for, and what it does not.
//!
//! The projection lived in two backends, each with its own copy of the filter,
//! and only one of them was tested. It is now one function in `vyre-driver`, so
//! the rule is proved once for every backend that reads a binding order off a
//! plan.

use std::collections::BTreeMap;

use vyre_driver::materialize::{project_resources, resident_buffer_names};
use vyre_driver::BindingPlan;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ShapeDim, ValueContract, ValueLifetime,
};
use vyre_megakernel::{
    compile, CompileRequest, DeviceFacts, Digest, ExternalFacts, SearchBudget,
};

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

/// WHY: buffer declarations in reverse or arbitrary order must still produce
/// deterministic binding-order names for the resident launch path.
#[test]
fn resident_buffer_names_with_reordered_declarations() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("buf_c", 2, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage("buf_a", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::output("buf_b", 1, DataType::U32),
        ],
        [1, 1, 1],
        Vec::new(),
    );
    let plan = BindingPlan::build(&program)
        .expect("Fix: binding plan must build for reordered declarations.");

    let names = resident_buffer_names(&plan, &program).collect::<Vec<_>>();
    assert_eq!(names, ["buf_a", "buf_b", "buf_c"]);
}

/// WHY: `project_resources` must derive resource values from `Artifact::canonical_value_by_name`,
/// ensuring the consumer calls the owner rather than implementing divergent name resolution.
#[test]
fn project_resources_derives_values_from_canonical_values_by_name() {
    let mut graph = ProgramGraph::new();
    let input = graph
        .add_external_value(
            "input",
            ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(16)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
        )
        .expect("adding external input value must succeed");
    graph
        .add_node(
            "test_node",
            Program::wrapped(
                vec![
                    BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32)
                        .with_count(16),
                    BufferDecl::output("output", 1, DataType::U32).with_count(16),
                ],
                [16, 1, 1],
                vec![Node::store(
                    "output",
                    Expr::u32(0),
                    Expr::load("input", Expr::u32(0)),
                )],
            ),
            vec![GraphInput {
                buffer: "input".into(),
                value: input,
                contract: ValueContract {
                    dtype: DataType::U32,
                    shape: vec![ShapeDim::Known(16)],
                    access: BufferAccess::ReadOnly,
                    lifetime: ValueLifetime::Invocation,
                },
            }],
            vec![GraphOutput {
                buffer: "output".into(),
                name: "output".into(),
                contract: ValueContract {
                    dtype: DataType::U32,
                    shape: vec![ShapeDim::Known(16)],
                    access: BufferAccess::ReadWrite,
                    lifetime: ValueLifetime::Output,
                },
                retained_successor_of: None,
            }],
        )
        .expect("adding test node must succeed");

    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0xA5; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(1, 100_000, 1, 0, 100_000_000),
        1_000_000,
    )
    .validate()
    .expect("compile request must validate");

    let artifact = compile(&request).expect("artifact compilation must succeed");
    let canonical = artifact
        .canonical_value_by_name()
        .expect("canonical_value_by_name must succeed");
    let projection = project_resources(&artifact);

    assert!(!canonical.is_empty(), "canonical values must not be empty");
    for (name, val) in canonical {
        assert_eq!(
            projection.values.get(name).copied(),
            Some(val),
            "projected value for {name} must match canonical owner"
        );
    }
    assert!(
        !projection.outputs.is_empty(),
        "projected outputs must not be empty"
    );
}
