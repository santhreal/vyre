//! Graphs, facts, and device fact sets the candidate-search contracts share.
//!
//! The grammar contracts and the evaluation-ladder contracts compile the same
//! graphs under the same facts and differ only in what they assert, so the
//! inputs are defined once. A second copy would have to be kept identical for
//! assertions that compare derivations across the two suites to keep meaning
//! what they say.
//!
//! Included with `#[path]` the same way as `tests/support/artifact_fixtures.rs`.

#![allow(dead_code)]

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BinOp, BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, GraphValueId, Node,
    Program, ProgramGraph, ShapeDim, SubgroupReduceOp, ValueContract, ValueLifetime,
};
use vyre_foundation::numeric::NumericContract;
use vyre_foundation::validate::BackendCapabilities;
use vyre_megakernel::{
    compile, Artifact, CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts,
    ObjectiveMetric, PruneReason, ScheduleProduction, SearchBudget, SearchCertificate,
    ValidatedCompileRequest,
};
use vyre_test_support::pass_programs::{add_program, copy_program};

pub(crate) fn contract(
    element: &DataType,
    access: BufferAccess,
    lifetime: ValueLifetime,
) -> ValueContract {
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

pub(crate) fn invocation(element: &DataType) -> ValueContract {
    contract(element, BufferAccess::ReadWrite, ValueLifetime::Invocation)
}

/// One subgroup reduction of `value`, stored to `output`.
///
/// The buffer pair and the reduction shell are the same for every reducing
/// fixture, so a suite that needs a particular combine states only the
/// expression that combine applies.
pub(crate) fn reduction_over(
    input: &str,
    output: &str,
    element: &DataType,
    value: Expr,
) -> Program {
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
                value: Box::new(value),
            },
        )],
    )
}

/// One subgroup reduction of a loaded element, stored back.
pub(crate) fn reduce_program(input: &str, output: &str, element: &DataType) -> Program {
    reduction_over(
        input,
        output,
        element,
        Expr::load(input, Expr::LocalId { axis: 0 }),
    )
}

/// The same reduction over the sum of two loaded elements.
pub(crate) fn join_program(left: &str, right: &str, output: &str, element: &DataType) -> Program {
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

/// Append one single-input, single-output stage to `graph`.
///
/// Every chained fixture graph here is stages wired in sequence over one value
/// contract, so the wiring is stated once. Returns the value the stage writes,
/// which is the next stage's input.
pub(crate) fn stage(
    graph: &mut ProgramGraph,
    name: &str,
    program: Program,
    input: (&str, GraphValueId),
    output: (&str, ValueContract),
) -> GraphValueId {
    let (buffer, value) = input;
    let (written, produced) = output;
    let (_, ids) = graph
        .add_node(
            name,
            program,
            vec![GraphInput {
                buffer: buffer.into(),
                value,
                contract: graph
                    .values()
                    .iter()
                    .find(|held| held.id == value)
                    .map(|held| held.contract.clone())
                    .expect("Fix: state a stage input the graph already declares"),
            }],
            vec![GraphOutput {
                buffer: written.into(),
                name: written.into(),
                contract: produced,
                retained_successor_of: None,
            }],
        )
        .unwrap_or_else(|error| panic!("Fix: stage `{name}` must be admitted: {error}"));
    ids[0]
}

/// A chain, an independent arm, and a fan-in, every stage reducing.
///
/// The independent arm is what gives the concurrency productions an operand, and
/// the fan-in is what gives the joining production one, so one graph exercises
/// every production that reorders a combine.
pub(crate) fn reducing_graph(element: &DataType) -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let in_a = graph
        .add_external_value("in_a", invocation(element))
        .expect("external input");
    let in_b = graph
        .add_external_value("in_b", invocation(element))
        .expect("external input");
    let mid_a = stage(
        &mut graph,
        "n0",
        reduce_program("in_a", "mid_a", element),
        ("in_a", in_a),
        ("mid_a", invocation(element)),
    );
    let mid_b = stage(
        &mut graph,
        "n1",
        reduce_program("mid_a", "mid_b", element),
        ("mid_a", mid_a),
        ("mid_b", invocation(element)),
    );
    let mid_c = stage(
        &mut graph,
        "n2",
        reduce_program("in_b", "mid_c", element),
        ("in_b", in_b),
        ("mid_c", invocation(element)),
    );
    graph
        .add_node(
            "n3",
            join_program("mid_b", "mid_c", "out", element),
            vec![
                GraphInput {
                    buffer: "mid_b".into(),
                    value: mid_b,
                    contract: invocation(element),
                },
                GraphInput {
                    buffer: "mid_c".into(),
                    value: mid_c,
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

/// A two-dimensional `u32` value, so a phase carries more than one axis.
pub(crate) fn matrix(lifetime: ValueLifetime) -> ValueContract {
    ValueContract {
        dtype: DataType::U32,
        shape: vec![
            ShapeDim::Symbol("rows".into()),
            ShapeDim::Symbol("cols".into()),
        ],
        access: BufferAccess::ReadWrite,
        lifetime,
    }
}

/// The same value read by a consumer that only loads it.
pub(crate) fn read_matrix(lifetime: ValueLifetime) -> ValueContract {
    ValueContract {
        access: BufferAccess::ReadOnly,
        ..matrix(lifetime)
    }
}

/// One stage, so no production has anything to fuse, overlap, or cut.
pub(crate) fn single_stage_graph() -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let input = graph
        .add_external_value("in_a", matrix(ValueLifetime::Invocation))
        .expect("external input");
    graph
        .add_node(
            "n0",
            copy_program("in_a", "out"),
            vec![GraphInput {
                buffer: "in_a".into(),
                value: input,
                contract: matrix(ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "out".into(),
                name: "out".into(),
                contract: matrix(ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .expect("only stage");
    graph
}

/// A chain, an independent arm, and a fan-in.
///
/// `n0 -> n1` is a single-consumer chain the grammar may fuse, `n2` is an
/// independent arm, and `n3` reads both `n1` and `n2`, so one phase carries two
/// predecessors and the joining productions have an operand.
pub(crate) fn joined_graph() -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let in_a = graph
        .add_external_value("in_a", matrix(ValueLifetime::Invocation))
        .expect("external input");
    let in_b = graph
        .add_external_value("in_b", matrix(ValueLifetime::Invocation))
        .expect("external input");
    let (_, mid_a) = graph
        .add_node(
            "n0",
            copy_program("in_a", "mid_a"),
            vec![GraphInput {
                buffer: "in_a".into(),
                value: in_a,
                contract: matrix(ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "mid_a".into(),
                name: "mid_a".into(),
                contract: matrix(ValueLifetime::Invocation),
                retained_successor_of: None,
            }],
        )
        .expect("first stage");
    let (_, mid_b) = graph
        .add_node(
            "n1",
            copy_program("mid_a", "mid_b"),
            vec![GraphInput {
                buffer: "mid_a".into(),
                value: mid_a[0],
                contract: matrix(ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "mid_b".into(),
                name: "mid_b".into(),
                contract: matrix(ValueLifetime::Invocation),
                retained_successor_of: None,
            }],
        )
        .expect("second stage");
    let (_, mid_c) = graph
        .add_node(
            "n2",
            copy_program("in_b", "mid_c"),
            vec![GraphInput {
                buffer: "in_b".into(),
                value: in_b,
                contract: matrix(ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "mid_c".into(),
                name: "mid_c".into(),
                contract: matrix(ValueLifetime::Invocation),
                retained_successor_of: None,
            }],
        )
        .expect("independent arm");
    graph
        .add_node(
            "n3",
            add_program("mid_b", "mid_c", "out"),
            vec![
                GraphInput {
                    buffer: "mid_b".into(),
                    value: mid_b[0],
                    contract: read_matrix(ValueLifetime::Invocation),
                },
                GraphInput {
                    buffer: "mid_c".into(),
                    value: mid_c[0],
                    contract: read_matrix(ValueLifetime::Invocation),
                },
            ],
            vec![GraphOutput {
                buffer: "out".into(),
                name: "out".into(),
                contract: matrix(ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .expect("fan-in stage");
    graph
}

pub(crate) fn facts() -> ExternalFacts {
    ExternalFacts::new(
        Digest([0x5a; 32]),
        BTreeMap::from([("rows".into(), 64), ("cols".into(), 64)]),
    )
}

/// A device that grants every capability the grammar can ask an operand for.
pub(crate) fn rich_device() -> DeviceFacts {
    DeviceFacts::new(
        BackendCapabilities {
            supports_subgroup_ops: true,
            has_subgroup_shuffle: true,
            ..BackendCapabilities::default()
        },
        256,
    )
    .with_occupancy(128, 64 * 1024)
    .with_compute_units(8)
    .with_concurrent_queues(4)
    .with_spatial_partitioning(true)
    .with_cooperative_launch(true)
    .with_subgroup_size(32)
    .with_launch_costs(4224, 1000)
    .with_bandwidth_facts(3788, 3788)
}

/// A device that reports nothing beyond a launch limit.
pub(crate) fn bare_device() -> DeviceFacts {
    DeviceFacts::new(BackendCapabilities::default(), 256)
}

/// A device where issuing one launch dominates every other cost.
pub(crate) fn launch_bound_device() -> DeviceFacts {
    DeviceFacts::new(BackendCapabilities::default(), 256).with_launch_costs(200_000, 1_000)
}

/// A device where launches are nearly free and resident state is scarce.
pub(crate) fn occupancy_bound_device() -> DeviceFacts {
    DeviceFacts::new(BackendCapabilities::default(), 256)
        .with_launch_costs(1, 1)
        .with_occupancy(1, 1)
        .with_bandwidth_facts(1, 1)
}

/// A device that partitions but guarantees no forward progress.
pub(crate) fn no_progress_device() -> DeviceFacts {
    DeviceFacts::new(BackendCapabilities::default(), 256)
        .with_compute_units(8)
        .with_spatial_partitioning(true)
        .with_launch_costs(4224, 1000)
}

pub(crate) fn budget() -> SearchBudget {
    SearchBudget::new(512, 200_000, 4, 0, 1_000_000_000)
}

/// Artifact byte ceiling every fixture request states, well above what the
/// fixture graphs compile to.
pub(crate) const ARTIFACT_BYTES: u64 = 4_000_000;

/// Minimize the latency of one submission within the fixture byte ceiling.
pub(crate) fn latency_objective() -> CompileObjective {
    CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, ARTIFACT_BYTES)
}

/// The fixture graph and facts as an unvalidated request.
///
/// Every suite here compiles the same graph under the same external facts and
/// differs only in device, budget, and objective. A suite that also states
/// recorded measurement evidence adds it before validating.
pub(crate) fn fixture_request(
    device: DeviceFacts,
    budget: SearchBudget,
    objective: CompileObjective,
) -> CompileRequest {
    CompileRequest::new(joined_graph(), facts(), device, budget, objective)
}

/// The fixture request validated for `device`, `budget` and `objective`.
pub(crate) fn validated(
    device: DeviceFacts,
    budget: SearchBudget,
    objective: CompileObjective,
) -> ValidatedCompileRequest {
    fixture_request(device, budget, objective)
        .validate()
        .expect("Fix: state fixture facts the compiler accepts")
}

pub(crate) fn compiled(device: DeviceFacts, budget: SearchBudget) -> Artifact {
    compile(&validated(device, budget, latency_objective())).expect("compilation must succeed")
}

/// The field path a refusal names, when it names one.
///
/// A diagnostic that reports the right code against the wrong field sends a
/// caller to edit something that was already correct, so every suite asserts the
/// path and none restates how to read it.
pub(crate) fn refused_field(error: &vyre_megakernel::CompileError) -> Option<&str> {
    error
        .diagnostic
        .location
        .as_ref()
        .and_then(|location| location.path.as_deref())
}

/// Productions whose transforms change the order invocations combine in.
///
/// Two suites ask what a reordering production does to a rounding reduction and
/// what a stated budget changes about it, so the roster they both range over is
/// stated once.
pub(crate) const REORDERING_PRODUCTIONS: [ScheduleProduction; 5] = [
    ScheduleProduction::SpatialPartition,
    ScheduleProduction::PersistentQueue,
    ScheduleProduction::Pipeline,
    ScheduleProduction::AsymmetricJoin,
    ScheduleProduction::AxisReorder,
];

/// `graph` compiled on the rich device under `numeric` when stated.
///
/// Every suite here compiles a fixture graph under the same external facts,
/// device, budget and objective, and differs only in the graph and the numeric
/// contract, so the request is built once.
pub(crate) fn artifact_of(graph: ProgramGraph, numeric: Option<NumericContract>) -> Artifact {
    artifact_of_within(graph, numeric, budget())
}

/// `graph` compiled on the rich device under `numeric` within `budget`.
///
/// A suite that asks what a bound does to the derived set states the bound and
/// reads the same request every other suite compiles.
pub(crate) fn artifact_of_within(
    graph: ProgramGraph,
    numeric: Option<NumericContract>,
    budget: SearchBudget,
) -> Artifact {
    let mut request =
        CompileRequest::new(graph, facts(), rich_device(), budget, latency_objective());
    if let Some(numeric) = numeric {
        request = request.with_numeric_budget(numeric);
    }
    compile(&request.validate().expect("request must validate")).expect("compilation must succeed")
}

/// The reducing graph over `element`, compiled under `numeric` when stated.
pub(crate) fn reducing_artifact(element: &DataType, numeric: Option<NumericContract>) -> Artifact {
    artifact_of(reducing_graph(element), numeric)
}

/// Candidates of one production the search eliminated as numerically illegal.
pub(crate) fn numerically_pruned(
    certificate: &SearchCertificate,
    production: ScheduleProduction,
) -> u32 {
    certificate
        .pruned
        .iter()
        .filter(|family| family.production == production && family.reason == PruneReason::Numerical)
        .map(|family| family.count)
        .sum()
}
