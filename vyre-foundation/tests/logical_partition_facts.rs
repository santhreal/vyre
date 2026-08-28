//! What the logical stage states about distributing a region.
//!
//! WHY: BACKLOG row 64 requires the logical stage to express partitionable
//! values and semantic exchanges without naming a device. Schedule selection
//! reads exactly these facts to place shards on a mesh, so a wrong axis kind or
//! a missing payload size becomes a placement that computes the wrong values or
//! a transfer priced at zero.
//!
//! What these cases do not prove: which device holds a shard. That is a target
//! fact and a schedule decision, proven in the megakernel placement tests.

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, CollectiveOp, CommGroup, DataType, Expr, GraphInput, GraphOutput,
    Node, Program, ProgramGraph, ShapeDim, ValueContract, ValueLifetime,
};
use vyre_foundation::logical::{
    LogicalExchangeKind, LogicalPartitionAxisKind, LogicalProgramGraph,
};

fn map_program(count: u32) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(count),
            BufferDecl::output("output", 1, DataType::U32).with_count(count),
        ],
        [64, 1, 1],
        vec![Node::store(
            "output",
            Expr::gid_x(),
            Expr::add(Expr::load("input", Expr::gid_x()), Expr::u32(1)),
        )],
    )
}

fn collective_program(count: u32, group: CommGroup) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(count),
            BufferDecl::output("output", 1, DataType::U32).with_count(count),
        ],
        [64, 1, 1],
        vec![
            Node::store("output", Expr::gid_x(), Expr::load("input", Expr::gid_x())),
            Node::AllReduce {
                buffer: "output".into(),
                op: CollectiveOp::Sum,
                group,
            },
        ],
    )
}

fn atomic_program(count: u32) -> Program {
    Program::wrapped(
        vec![BufferDecl::read_write("counter", 0, DataType::U32).with_count(count)],
        [64, 1, 1],
        vec![Node::store(
            "counter",
            Expr::u32(0),
            Expr::atomic_add("counter", Expr::u32(0), Expr::u32(1)),
        )],
    )
}

fn logical_of(program: Program) -> ProgramGraph {
    ProgramGraph::from_program("region", program).expect("fixture graph must validate")
}

/// A graph whose node reads the caller-supplied value it also writes.
fn in_place_graph(count: u32) -> ProgramGraph {
    let state = ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Known(u64::from(count))],
        access: BufferAccess::ReadWrite,
        lifetime: ValueLifetime::Invocation,
    };
    let mut graph = ProgramGraph::new();
    let value = graph
        .add_external_value("state", state.clone())
        .expect("fixture external value must be valid");
    graph
        .add_node(
            "relax",
            Program::wrapped(
                vec![BufferDecl::read_write("state", 0, DataType::U32).with_count(count)],
                [64, 1, 1],
                vec![Node::store(
                    "state",
                    Expr::gid_x(),
                    Expr::add(Expr::load("state", Expr::gid_x()), Expr::u32(1)),
                )],
            ),
            vec![GraphInput {
                buffer: "state".into(),
                value,
                contract: state,
            }],
            Vec::new(),
        )
        .expect("fixture in-place node must be valid");
    graph
}

/// A graph whose second node consumes the value the first produces.
fn chained_graph(count: u32) -> ProgramGraph {
    let carried = ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Known(u64::from(count))],
        access: BufferAccess::ReadWrite,
        lifetime: ValueLifetime::Invocation,
    };
    let mut graph = ProgramGraph::new();
    let (_, produced) = graph
        .add_node(
            "producer",
            Program::wrapped(
                vec![BufferDecl::read_write("mid", 0, DataType::U32).with_count(count)],
                [64, 1, 1],
                vec![Node::store("mid", Expr::gid_x(), Expr::u32(1))],
            ),
            Vec::new(),
            vec![GraphOutput {
                buffer: "mid".into(),
                name: "mid".into(),
                contract: carried.clone(),
                retained_successor_of: None,
            }],
        )
        .expect("fixture producer node must be valid");
    graph
        .add_node(
            "consumer",
            Program::wrapped(
                vec![
                    BufferDecl::read_write("mid", 0, DataType::U32).with_count(count),
                    BufferDecl::output("result", 1, DataType::U32).with_count(count),
                ],
                [64, 1, 1],
                vec![Node::store(
                    "result",
                    Expr::gid_x(),
                    Expr::load("mid", Expr::gid_x()),
                )],
            ),
            vec![GraphInput {
                buffer: "mid".into(),
                value: produced[0],
                contract: carried,
            }],
            vec![GraphOutput {
                buffer: "result".into(),
                name: "result".into(),
                contract: ValueContract {
                    dtype: DataType::U32,
                    shape: vec![ShapeDim::Known(u64::from(count))],
                    access: BufferAccess::WriteOnly,
                    lifetime: ValueLifetime::Output,
                },
                retained_successor_of: None,
            }],
        )
        .expect("fixture consumer node must be valid");
    graph
}

/// A graph whose node atomically updates a caller-visible value.
///
/// The value is written for the invocation rather than retained across
/// submissions, so the region is atomic without being ordered.
fn atomic_graph(count: u32) -> ProgramGraph {
    let counters = ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Known(u64::from(count))],
        access: BufferAccess::ReadWrite,
        lifetime: ValueLifetime::Invocation,
    };
    let mut graph = ProgramGraph::new();
    graph
        .add_node(
            "scatter",
            Program::wrapped(
                vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(count)],
                [64, 1, 1],
                vec![Node::let_bind(
                    "prior",
                    Expr::atomic_add("out", Expr::gid_x(), Expr::u32(1)),
                )],
            ),
            Vec::new(),
            vec![GraphOutput {
                buffer: "out".into(),
                name: "out".into(),
                contract: counters,
                retained_successor_of: None,
            }],
        )
        .expect("fixture atomic node must be valid");
    graph
}

#[test]
fn an_independent_region_may_split_every_axis() {
    let graph = logical_of(map_program(256));
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect("fixture logical stage must validate");
    let facts = &logical.regions()[0].partition;
    assert!(facts.replicable);
    assert_eq!(facts.axes.len(), 1);
    assert_eq!(facts.axes[0].axis, 0);
    assert_eq!(facts.axes[0].kind, LogicalPartitionAxisKind::Elementwise);
    assert_eq!(facts.axes[0].bound, 256);
}

#[test]
fn a_region_with_atomics_is_never_replicated() {
    let graph = logical_of(atomic_program(64));
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect("atomic logical stage must validate");
    assert!(
        !logical.regions()[0].partition.replicable,
        "two participants holding one atomic region would each advance it"
    );
}

#[test]
fn an_exchange_states_its_semantics_group_and_payload_bytes() {
    let graph = logical_of(collective_program(256, CommGroup(3)));
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect("collective logical stage must validate");
    let exchanges = logical.exchanges();
    assert_eq!(exchanges.len(), 1);
    let exchange = &exchanges[0];
    assert_eq!(exchange.kind, LogicalExchangeKind::AllReduce);
    assert_eq!(exchange.group, 3);
    assert_eq!(exchange.combine, Some(CollectiveOp::Sum));
    assert_eq!(exchange.bytes, 256 * 4);
    assert_eq!(exchange.values.len(), 1);
}

#[test]
fn a_graph_without_an_exchange_states_none() {
    let graph = logical_of(map_program(64));
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect("fixture logical stage must validate");
    assert!(logical.exchanges().is_empty());
}

#[test]
fn a_symbolic_payload_is_sized_from_its_binding() {
    let program = Program::wrapped(
        vec![BufferDecl::output("output", 0, DataType::U32).with_count(8)],
        [64, 1, 1],
        vec![
            Node::store("output", Expr::gid_x(), Expr::u32(1)),
            Node::AllReduce {
                buffer: "output".into(),
                op: CollectiveOp::Max,
                group: CommGroup::WORLD,
            },
        ],
    );
    let mut graph = ProgramGraph::new();
    graph
        .add_node(
            "region",
            program,
            Vec::<GraphInput>::new(),
            vec![GraphOutput {
                buffer: "output".to_owned(),
                name: "result".to_owned(),
                contract: ValueContract {
                    dtype: DataType::U32,
                    shape: vec![ShapeDim::Symbol("items".to_owned())],
                    access: BufferAccess::WriteOnly,
                    lifetime: ValueLifetime::Output,
                },
                retained_successor_of: None,
            }],
        )
        .expect("symbolic graph node must validate");
    let logical =
        LogicalProgramGraph::validate(&graph, &BTreeMap::from([("items".to_owned(), 12)]))
            .expect("symbolic logical stage must validate");
    assert_eq!(logical.exchanges()[0].bytes, 12 * 4);
}

/// WHY: an atomic update names its destination at run time, so a shard computes
/// contributions the other shards own. Stating that axis as an ordered sequence
/// would forbid cutting it at all, and stating it as independent elements would
/// lose every contribution that crossed a shard boundary.
#[test]
fn a_region_with_atomics_routes_its_points() {
    let graph = atomic_graph(64);
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect("atomic logical stage must validate");
    let facts = &logical.regions()[0].partition;
    assert_eq!(facts.axes.len(), 1);
    assert_eq!(facts.axes[0].kind, LogicalPartitionAxisKind::Routed);
}

/// WHY: a region that reads a value it also writes may read a point another
/// shard computed. Calling that axis elementwise states that no shard needs a
/// neighbour's bytes, which is what makes a halo exchange go missing.
#[test]
fn a_region_that_reads_what_it_writes_addresses_a_spatial_domain() {
    let graph = in_place_graph(128);
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect("in-place logical stage must validate");
    let facts = &logical.regions()[0].partition;
    assert_eq!(facts.axes.len(), 1);
    assert_eq!(facts.axes[0].kind, LogicalPartitionAxisKind::Spatial);
    assert!(
        facts.replicable,
        "reading what it writes does not make a region unsafe to hold whole"
    );
}

/// WHY: a placement prices a cross-device handoff and a routed update from these
/// two figures. A zero would price a transfer that moves real bytes at nothing,
/// and the placement that needs it would rank as free.
#[test]
fn a_region_states_the_bytes_it_writes_and_the_bytes_it_waits_for() {
    let graph = chained_graph(256);
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect("chained logical stage must validate");
    let regions = logical.regions();
    assert_eq!(regions.len(), 2);
    assert_eq!(regions[0].written_bytes, 256 * 4);
    assert_eq!(regions[1].dependencies.len(), 1);
    assert_eq!(regions[1].dependencies[0].bytes, 256 * 4);
}
