//! A schedule may reorder a reduction only when the reduction reassociates.
//!
//! A spatial partition, a persistent queue, a pipeline, and an asymmetric join
//! all let independent workers reach a shared accumulator in an order the
//! schedule does not fix. Over integer addition that is the same number; over
//! floating-point addition it is a different one, and the difference is
//! data-dependent, so it reaches a caller as an accuracy report rather than as a
//! failure. Before this constraint the search admitted those candidates for a
//! rounding reduction and could select one.
//!
//! The two graphs here are the same kernel organization over two element types,
//! so the only variable is whether the reduction reassociates. The last case
//! derives the whole production vocabulary from `ScheduleProduction::ALL` and
//! requires every production the rounding graph loses to state `Numerical`,
//! which is what a production added later has to satisfy.

#[path = "support/search_fixtures.rs"]
mod search_fixtures;

use search_fixtures::{facts, rich_device};

use vyre_foundation::ir::{
    BinOp, BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program,
    ProgramGraph, ShapeDim, SubgroupReduceOp, ValueContract, ValueLifetime,
};
use vyre_megakernel::{
    compile, CompileObjective, CompileRequest, ObjectiveMetric, PruneReason, ScheduleProduction,
    SearchBudget, SearchCertificate,
};

/// Productions whose transforms change the order invocations combine in.
const REORDERING: [ScheduleProduction; 5] = [
    ScheduleProduction::SpatialPartition,
    ScheduleProduction::PersistentQueue,
    ScheduleProduction::Pipeline,
    ScheduleProduction::AsymmetricJoin,
    ScheduleProduction::AxisReorder,
];

/// Productions that move work without changing which invocations combine.
const ORDER_PRESERVING: [ScheduleProduction; 4] = [
    ScheduleProduction::DispatchCut,
    ScheduleProduction::Synchronization,
    ScheduleProduction::MemoryPlacement,
    ScheduleProduction::Prefetch,
];

fn contract(element: &DataType, access: BufferAccess, lifetime: ValueLifetime) -> ValueContract {
    ValueContract {
        dtype: element.clone(),
        shape: vec![
            ShapeDim::Symbol("rows".into()),
            ShapeDim::Symbol("cols".into()),
        ],
        access,
        lifetime,
    }
}

fn invocation(element: &DataType) -> ValueContract {
    contract(element, BufferAccess::ReadWrite, ValueLifetime::Invocation)
}

/// One subgroup reduction of a loaded element, stored back.
fn reduce_program(input: &str, output: &str, element: &DataType) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read_write(input, 0, element.clone()),
            BufferDecl::read_write(output, 1, element.clone()),
        ],
        [32, 1, 1],
        vec![Node::store(
            output,
            Expr::LocalId { axis: 0 },
            Expr::SubgroupReduce {
                op: SubgroupReduceOp::Add,
                value: Box::new(Expr::load(input, Expr::LocalId { axis: 0 })),
            },
        )],
    )
}

/// The same reduction over the sum of two loaded elements.
fn join_program(left: &str, right: &str, output: &str, element: &DataType) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read_write(left, 0, element.clone()),
            BufferDecl::read_write(right, 1, element.clone()),
            BufferDecl::read_write(output, 2, element.clone()),
        ],
        [32, 1, 1],
        vec![Node::store(
            output,
            Expr::LocalId { axis: 0 },
            Expr::SubgroupReduce {
                op: SubgroupReduceOp::Add,
                value: Box::new(Expr::BinOp {
                    op: BinOp::Add,
                    left: Box::new(Expr::load(left, Expr::LocalId { axis: 0 })),
                    right: Box::new(Expr::load(right, Expr::LocalId { axis: 0 })),
                }),
            },
        )],
    )
}

/// A chain, an independent arm, and a fan-in, every stage reducing.
///
/// The independent arm is what gives the concurrency productions an operand, and
/// the fan-in is what gives the joining production one, so one graph exercises
/// every production that reorders a combine.
fn reducing_graph(element: &DataType) -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let in_a = graph
        .add_external_value("in_a", invocation(element))
        .expect("external input");
    let in_b = graph
        .add_external_value("in_b", invocation(element))
        .expect("external input");
    let (_, mid_a) = graph
        .add_node(
            "n0",
            reduce_program("in_a", "mid_a", element),
            vec![GraphInput {
                buffer: "in_a".into(),
                value: in_a,
                contract: invocation(element),
            }],
            vec![GraphOutput {
                buffer: "mid_a".into(),
                name: "mid_a".into(),
                contract: invocation(element),
                retained_successor_of: None,
            }],
        )
        .expect("first stage");
    let (_, mid_b) = graph
        .add_node(
            "n1",
            reduce_program("mid_a", "mid_b", element),
            vec![GraphInput {
                buffer: "mid_a".into(),
                value: mid_a[0],
                contract: invocation(element),
            }],
            vec![GraphOutput {
                buffer: "mid_b".into(),
                name: "mid_b".into(),
                contract: invocation(element),
                retained_successor_of: None,
            }],
        )
        .expect("second stage");
    let (_, mid_c) = graph
        .add_node(
            "n2",
            reduce_program("in_b", "mid_c", element),
            vec![GraphInput {
                buffer: "in_b".into(),
                value: in_b,
                contract: invocation(element),
            }],
            vec![GraphOutput {
                buffer: "mid_c".into(),
                name: "mid_c".into(),
                contract: invocation(element),
                retained_successor_of: None,
            }],
        )
        .expect("independent arm");
    graph
        .add_node(
            "n3",
            join_program("mid_b", "mid_c", "out", element),
            vec![
                GraphInput {
                    buffer: "mid_b".into(),
                    value: mid_b[0],
                    contract: invocation(element),
                },
                GraphInput {
                    buffer: "mid_c".into(),
                    value: mid_c[0],
                    contract: invocation(element),
                },
            ],
            vec![GraphOutput {
                buffer: "out".into(),
                name: "out".into(),
                contract: contract(element, BufferAccess::ReadWrite, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .expect("fan-in stage");
    graph
}

fn certificate(element: &DataType) -> SearchCertificate {
    let request = CompileRequest::new(
        reducing_graph(element),
        facts(),
        rich_device(),
        SearchBudget::new(512, 200_000, 4, 0, 1_000_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 4_000_000),
    )
    .validate()
    .expect("request must validate");
    compile(&request)
        .expect("compilation must succeed")
        .selected_plan()
        .certificate
        .clone()
}

/// Candidates of one production eliminated for one reason.
fn pruned(certificate: &SearchCertificate, production: ScheduleProduction) -> u32 {
    certificate
        .pruned
        .iter()
        .filter(|family| family.production == production && family.reason == PruneReason::Numerical)
        .map(|family| family.count)
        .sum()
}

#[test]
fn an_exact_reduction_keeps_every_reordering_production() {
    let exact = certificate(&DataType::U32);

    for production in REORDERING {
        assert!(
            exact.admitted_by(production) > 0,
            "{production:?} admitted nothing over an exact reduction: {exact:#?}"
        );
        assert_eq!(
            pruned(&exact, production),
            0,
            "{production:?} was called numerical over an exact reduction"
        );
    }
}

#[test]
fn a_rounding_reduction_eliminates_every_reordering_production() {
    let rounding = certificate(&DataType::F32);

    for production in REORDERING {
        assert_eq!(
            rounding.admitted_by(production),
            0,
            "{production:?} admitted a candidate that reorders a rounding reduction"
        );
        assert!(
            pruned(&rounding, production) > 0,
            "{production:?} lost its candidates without stating Numerical: {rounding:#?}"
        );
    }
}

#[test]
fn a_rounding_reduction_still_compiles_through_the_order_preserving_productions() {
    let rounding = certificate(&DataType::F32);

    for production in ORDER_PRESERVING {
        assert!(
            rounding.admitted_by(production) > 0,
            "{production:?} admitted nothing, so the rounding graph lost more than reordering"
        );
    }
}

#[test]
fn every_production_an_exact_reduction_admits_is_kept_or_stated_numerical() {
    let exact = certificate(&DataType::U32);
    let rounding = certificate(&DataType::F32);

    for production in ScheduleProduction::ALL.iter().copied() {
        if exact.admitted_by(production) == 0 {
            continue;
        }
        if rounding.admitted_by(production) > 0 {
            continue;
        }
        assert!(
            pruned(&rounding, production) > 0,
            "{production:?} disappeared over a rounding reduction without stating Numerical"
        );
    }
}
