//! RFC-0004 collective IR node contracts.

use vyre_foundation::ir::stats::{
    NODE_KIND_ALL_GATHER, NODE_KIND_ALL_REDUCE, NODE_KIND_BROADCAST, NODE_KIND_REDUCE_SCATTER,
};
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, CollectiveOp, CommGroup, DataType, Node, Program,
};
use vyre_foundation::program_caps::scan;
use vyre_foundation::transform::collectives::{
    collective_transport_plan, lower_single_rank_collectives,
};
use vyre_foundation::validate::{
    validate, validate_with_options, BackendCapabilities, ValidationOptions,
};
use vyre_test_support::collective_programs::{collective_buffers, collective_nodes};

fn collective_program() -> Program {
    Program::wrapped(
        collective_buffers(),
        [64, 1, 1],
        collective_nodes(CommGroup(3), 1).into(),
    )
}

fn collective_options() -> ValidationOptions<'static> {
    ValidationOptions::default().with_backend_capabilities(BackendCapabilities {
        supports_distributed_collectives: true,
        ..BackendCapabilities::default()
    })
}

#[test]
fn collective_nodes_require_explicit_backend_capability() {
    let errors = validate(&collective_program());
    assert!(
        errors
            .iter()
            .any(|error| error.code().as_str() == "V046"
                && error.message().contains("distributed collective nodes require backend collective support")),
        "Fix: collectives must not silently validate on scalar or single-device backends: {errors:?}"
    );
}

#[test]
fn collective_nodes_validate_when_backend_declares_support() {
    let report = validate_with_options(&collective_program(), collective_options());
    assert!(
        report.errors.is_empty(),
        "Fix: supported RFC-0004 collectives must validate cleanly, got {:?}",
        report.errors
    );
}

#[test]
fn split_collectives_reject_element_type_mismatch() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(8),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::F32).with_count(8),
        ],
        [64, 1, 1],
        vec![Node::AllGather {
            input: "input".into(),
            output: "out".into(),
            group: CommGroup::WORLD,
        }],
    );

    let report = validate_with_options(&program, collective_options());
    assert!(
        report
            .errors
            .iter()
            .any(|error| error.message().contains("element")),
        "Fix: split collectives must reject mismatched input/output element types: {:?}",
        report.errors
    );
}

#[test]
fn collective_nodes_roundtrip_through_program_wire() {
    let program = collective_program();
    let wire = program
        .to_wire()
        .expect("Fix: collective program must encode.");
    let decoded = Program::from_wire(&wire).expect("Fix: collective program must decode.");
    let nodes = flatten_nodes(decoded.entry());

    assert_eq!(nodes.len(), 5);
    assert!(matches!(
        nodes[1],
        Node::AllReduce {
            op: CollectiveOp::Sum,
            group: CommGroup(0),
            ..
        }
    ));
    assert!(matches!(
        nodes[2],
        Node::AllGather {
            group: CommGroup(0),
            ..
        }
    ));
    assert!(matches!(
        nodes[3],
        Node::ReduceScatter {
            op: CollectiveOp::Max,
            group: CommGroup(3),
            ..
        }
    ));
    assert!(matches!(
        nodes[4],
        Node::Broadcast {
            root: 1,
            group: CommGroup(3),
            ..
        }
    ));
}

fn flatten_nodes(nodes: &[Node]) -> Vec<&Node> {
    let mut out = Vec::new();
    for node in nodes {
        out.push(node);
        match node {
            Node::If {
                then, otherwise, ..
            } => {
                out.extend(flatten_nodes(then));
                out.extend(flatten_nodes(otherwise));
            }
            Node::Loop { body, .. } | Node::Block(body) => out.extend(flatten_nodes(body)),
            Node::Region { body, .. } => out.extend(flatten_nodes(body)),
            _ => {}
        }
    }
    out
}

#[test]
fn collective_nodes_are_visible_to_program_stats() {
    let program = collective_program();
    let stats = program.stats();
    let mask = NODE_KIND_ALL_REDUCE
        | NODE_KIND_ALL_GATHER
        | NODE_KIND_REDUCE_SCATTER
        | NODE_KIND_BROADCAST;
    assert!(
        stats.has_any_node_kind(mask),
        "Fix: optimizer skip gates must see collective node kinds."
    );
    assert_eq!(
        stats.node_kinds_present & mask,
        mask,
        "Fix: every RFC-0004 collective node kind must have a stable ProgramStats bit."
    );
}

#[test]
fn collective_nodes_are_visible_to_required_capability_scan() {
    let required = scan(&collective_program());
    assert!(
        required.distributed_collectives,
        "Fix: dispatch admission must see RFC-0004 collective requirements before backend launch."
    );
}

/// Every collective kind on the world group lowers to a local rewrite, and the
/// plan counts one of each before it does.
///
/// A kind that stopped lowering would leave a distributed collective in a
/// program the single-rank path reported as fully local.
#[test]
fn local_single_rank_lowering_covers_all_collective_node_kinds() {
    let program = Program::wrapped(
        collective_buffers(),
        [64, 1, 1],
        vec![Node::Block(collective_nodes(CommGroup::WORLD, 0).into())],
    );

    let plan = collective_transport_plan(&program);
    assert_eq!(plan.local_single_rank_collectives(), 4);
    assert_eq!(plan.transport_collectives(), 0);
    assert_eq!(plan.local_ops().all_reduce(), 1);
    assert_eq!(plan.local_ops().all_gather(), 1);
    assert_eq!(plan.local_ops().reduce_scatter(), 1);
    assert_eq!(plan.local_ops().broadcast(), 1);

    let lowered = lower_single_rank_collectives(&program)
        .expect("Fix: all WORLD single-rank collective kinds must lower locally")
        .expect("Fix: local collective lowering must rewrite the program");

    assert!(!lowered.stats().distributed_collectives());
    assert!(validate(&lowered).is_empty());
}
