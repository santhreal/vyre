//! The graph shapes the whole-program planning suites compile.
//!
//! Three suites ask the planner about the same two-node graph: a producer and a
//! consumer joined by one invocation-scoped value. Each had written its own copy
//! of the builder, identical but for the program each node runs. One owner means
//! a suite that changes the shape changes it for every suite that reads it, which
//! is the point of asking the same planner the same question.

use vyre_foundation::ir::{
    BufferAccess, DataType, Expr, GraphInput, GraphOutput, GraphValueId, Node, Program,
    ProgramGraph, ShapeDim, ValueContract, ValueLifetime,
};

use crate::graph_values::u32_symbolic;
use crate::pass_programs::copy_program;

/// A `u32` value of symbolic length that lives for one invocation.
pub fn invocation_contract() -> ValueContract {
    ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Symbol("items".into())],
        access: BufferAccess::ReadWrite,
        lifetime: ValueLifetime::Invocation,
    }
}

/// A producer and a consumer joined by one invocation-scoped value.
///
/// `producer` reads `input` and writes `intermediate`; `consumer` reads
/// `intermediate` and writes `output`. The caller supplies both programs, which
/// is the only thing the suites disagree about.
pub fn producer_consumer_pair(producer: Program, consumer: Program) -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let input = graph
        .add_external_value("input", invocation_contract())
        .unwrap();
    let (_, intermediate) = graph
        .add_node(
            "producer",
            producer,
            vec![GraphInput {
                buffer: "input".into(),
                value: input,
                contract: invocation_contract(),
            }],
            vec![GraphOutput {
                buffer: "intermediate".into(),
                name: "intermediate".into(),
                contract: invocation_contract(),
                retained_successor_of: None,
            }],
        )
        .unwrap();
    graph
        .add_node(
            "consumer",
            consumer,
            vec![GraphInput {
                buffer: "intermediate".into(),
                value: intermediate[0],
                contract: invocation_contract(),
            }],
            vec![GraphOutput {
                buffer: "output".into(),
                name: "output".into(),
                contract: ValueContract {
                    lifetime: ValueLifetime::Output,
                    ..invocation_contract()
                },
                retained_successor_of: None,
            }],
        )
        .unwrap();
    graph
}
/// Two independent arms joined into one graph.
pub fn two_arm_graph(arm_a: Program, arm_b: Program) -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let in_a = graph
        .add_external_value("in_a", invocation_contract())
        .unwrap();
    let in_b = graph
        .add_external_value("in_b", invocation_contract())
        .unwrap();
    graph
        .add_node(
            "arm_a",
            arm_a,
            vec![GraphInput {
                buffer: "in_a".into(),
                value: in_a,
                contract: invocation_contract(),
            }],
            vec![GraphOutput {
                buffer: "out_a".into(),
                name: "out_a".into(),
                contract: ValueContract {
                    lifetime: ValueLifetime::Output,
                    ..invocation_contract()
                },
                retained_successor_of: None,
            }],
        )
        .unwrap();
    graph
        .add_node(
            "arm_b",
            arm_b,
            vec![GraphInput {
                buffer: "in_b".into(),
                value: in_b,
                contract: invocation_contract(),
            }],
            vec![GraphOutput {
                buffer: "out_b".into(),
                name: "out_b".into(),
                contract: ValueContract {
                    lifetime: ValueLifetime::Output,
                    ..invocation_contract()
                },
                retained_successor_of: None,
            }],
        )
        .unwrap();
    graph
}

/// Independent two-arm graph: two parallel copy operations reading separate inputs and writing separate outputs.
pub fn independent_two_arm_graph() -> ProgramGraph {
    two_arm_graph(copy_program("in_a", "out_a"), copy_program("in_b", "out_b"))
}

/// RAW conflicting two-arm graph: arm B reads the intermediate output of arm A within the same graph.
pub fn raw_conflict_two_arm_graph() -> ProgramGraph {
    producer_consumer_pair(
        copy_program("input", "intermediate"),
        copy_program("intermediate", "output"),
    )
}

/// Asymmetric join graph: Node 0 feeds Node 1 and Node 2.
pub fn asymmetric_join_graph() -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let in0 = graph
        .add_external_value("in0", invocation_contract())
        .unwrap();
    let (_, out0) = graph
        .add_node(
            "n0",
            copy_program("in0", "out0"),
            vec![GraphInput {
                buffer: "in0".into(),
                value: in0,
                contract: invocation_contract(),
            }],
            vec![GraphOutput {
                buffer: "out0".into(),
                name: "out0".into(),
                contract: invocation_contract(),
                retained_successor_of: None,
            }],
        )
        .unwrap();
    graph
        .add_node(
            "n1",
            copy_program("in1", "out1"),
            vec![GraphInput {
                buffer: "in1".into(),
                value: out0[0],
                contract: invocation_contract(),
            }],
            vec![GraphOutput {
                buffer: "out1".into(),
                name: "out1".into(),
                contract: invocation_contract(),
                retained_successor_of: None,
            }],
        )
        .unwrap();
    graph
        .add_node(
            "n2",
            copy_program("in2", "out2"),
            vec![GraphInput {
                buffer: "in2".into(),
                value: out0[0],
                contract: invocation_contract(),
            }],
            vec![GraphOutput {
                buffer: "out2".into(),
                name: "out2".into(),
                contract: ValueContract {
                    lifetime: ValueLifetime::Output,
                    ..invocation_contract()
                },
                retained_successor_of: None,
            }],
        )
        .unwrap();
    graph
}

/// The read-only ports a node declares over one caller value and one constant.
///
/// Two suites build a node that adds a caller input to a constant, and the port
/// declarations were the same eleven lines in both. Only the program and the
/// output around them differ, so those stay at the call site.
pub fn value_and_constant_ports(input: GraphValueId, constant: GraphValueId) -> Vec<GraphInput> {
    vec![
        GraphInput {
            buffer: "input".into(),
            value: input,
            contract: u32_symbolic(BufferAccess::ReadOnly, ValueLifetime::Invocation),
        },
        GraphInput {
            buffer: "constant".into(),
            value: constant,
            contract: u32_symbolic(BufferAccess::ReadOnly, ValueLifetime::Constant),
        },
    ]
}

/// The recurrence a retained-state connected-graph contract steps.
///
/// `s[0] = 2 * s[0] + u[0]`, over the read binding `u` and the read-write
/// binding `s`. Two connected-graph suites state this recurrence and differ
/// only in what they publish out of the updated state, so the recurrence has
/// one owner and the publication stays at the call site.
#[must_use]
pub fn retained_accumulate_node() -> Node {
    Node::store(
        "s",
        Expr::u32(0),
        Expr::add(
            Expr::mul(Expr::load("s", Expr::u32(0)), Expr::u32(2)),
            Expr::load("u", Expr::u32(0)),
        ),
    )
}

/// The `u` and `s` ports a node running [`retained_accumulate_node`] declares.
///
/// `sample` arrives once per invocation and `state` survives the step, which
/// is what makes the recurrence retained rather than a fold over one call.
#[must_use]
pub fn sample_and_retained_ports(sample: GraphValueId, state: GraphValueId) -> Vec<GraphInput> {
    vec![
        GraphInput {
            buffer: "u".into(),
            value: sample,
            contract: dense_u32(BufferAccess::ReadOnly, ValueLifetime::Invocation, 1),
        },
        GraphInput {
            buffer: "s".into(),
            value: state,
            contract: dense_u32(BufferAccess::ReadWrite, ValueLifetime::Retained, 1),
        },
    ]
}

/// Lanes the pure-dataflow graph runs over.
pub const PURE_DATAFLOW_LANES: u64 = 4;

/// A `u32` value of `count` dense lanes.
fn dense_u32(access: BufferAccess, lifetime: ValueLifetime, count: u64) -> ValueContract {
    ValueContract::dense_1d(DataType::U32, count, access, lifetime)
}

/// The three-stage pure-dataflow graph every concrete driver's connected-graph
/// contract compiles.
///
/// `scale_node` writes `y = 3x + 5` over [`PURE_DATAFLOW_LANES`] lanes,
/// `sum_node` reduces `y` to a single `s`, and `norm_node` writes
/// `z = y + s`. Two value edges leave `scale_node`, so a planner that
/// collapses the graph to a chain produces a different answer. The external
/// input is `in_x` and the graph output is `z_out`.
///
/// Three suites built this graph line for line. The shape is the question every
/// backend is asked, so a suite that changes it changes it for all of them.
#[must_use]
pub fn pure_dataflow_graph() -> ProgramGraph {
    use vyre_foundation::ir::{BufferDecl, Expr, Node};

    let count = PURE_DATAFLOW_LANES;
    let lanes = u32::try_from(count).expect("the lane count fits a dispatch dimension");
    let mut graph = ProgramGraph::new();

    let in_x = graph
        .add_external_value(
            "in_x",
            dense_u32(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
        )
        .expect("the external input value is the first value in an empty graph");

    let scale = Program::wrapped(
        vec![
            BufferDecl::read("x", 0, DataType::U32).with_count(lanes),
            BufferDecl::written("y", 1, BufferAccess::WriteOnly, DataType::U32).with_count(lanes),
        ],
        [lanes, 1, 1],
        vec![Node::store(
            "y",
            Expr::gid_x(),
            Expr::add(
                Expr::mul(Expr::load("x", Expr::gid_x()), Expr::u32(3)),
                Expr::u32(5),
            ),
        )],
    );
    let (_, val_y) = graph
        .add_node(
            "scale_node",
            scale,
            vec![GraphInput {
                buffer: "x".into(),
                value: in_x,
                contract: dense_u32(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
            }],
            vec![GraphOutput {
                buffer: "y".into(),
                name: "y".into(),
                contract: dense_u32(BufferAccess::WriteOnly, ValueLifetime::Invocation, count),
                retained_successor_of: None,
            }],
        )
        .expect("the scale node declares one input and one output");

    let sum = Program::wrapped(
        vec![
            BufferDecl::read("y_in", 0, DataType::U32).with_count(lanes),
            BufferDecl::written("sum_out", 1, BufferAccess::WriteOnly, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "sum_out",
            Expr::u32(0),
            Expr::add(
                Expr::add(
                    Expr::load("y_in", Expr::u32(0)),
                    Expr::load("y_in", Expr::u32(1)),
                ),
                Expr::add(
                    Expr::load("y_in", Expr::u32(2)),
                    Expr::load("y_in", Expr::u32(3)),
                ),
            ),
        )],
    );
    let (_, val_s) = graph
        .add_node(
            "sum_node",
            sum,
            vec![GraphInput {
                buffer: "y_in".into(),
                value: val_y[0],
                contract: dense_u32(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
            }],
            vec![GraphOutput {
                buffer: "sum_out".into(),
                name: "sum_out".into(),
                contract: dense_u32(BufferAccess::WriteOnly, ValueLifetime::Invocation, 1),
                retained_successor_of: None,
            }],
        )
        .expect("the sum node reads the scale node's output");

    let norm = Program::wrapped(
        vec![
            BufferDecl::read("y_norm_in", 0, DataType::U32).with_count(lanes),
            BufferDecl::read("s_in", 1, DataType::U32).with_count(1),
            BufferDecl::output("z_out", 2, DataType::U32).with_count(lanes),
        ],
        [lanes, 1, 1],
        vec![Node::store(
            "z_out",
            Expr::gid_x(),
            Expr::add(
                Expr::load("y_norm_in", Expr::gid_x()),
                Expr::load("s_in", Expr::u32(0)),
            ),
        )],
    );
    graph
        .add_node(
            "norm_node",
            norm,
            vec![
                GraphInput {
                    buffer: "y_norm_in".into(),
                    value: val_y[0],
                    contract: dense_u32(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                },
                GraphInput {
                    buffer: "s_in".into(),
                    value: val_s[0],
                    contract: dense_u32(BufferAccess::ReadOnly, ValueLifetime::Invocation, 1),
                },
            ],
            vec![GraphOutput {
                buffer: "z_out".into(),
                name: "z_out".into(),
                contract: dense_u32(BufferAccess::WriteOnly, ValueLifetime::Output, count),
                retained_successor_of: None,
            }],
        )
        .expect("the norm node joins both intra-graph values");

    graph
}

/// An independent host evaluation of [`pure_dataflow_graph`] over `input`.
///
/// Written from the mathematics the graph states, not from its IR, so a lowering
/// that computes something else is separated from one that computes the answer.
#[must_use]
pub fn pure_dataflow_oracle(input: [u32; 4]) -> [u32; 4] {
    let scaled = input.map(|lane| lane.wrapping_mul(3).wrapping_add(5));
    let sum = scaled
        .iter()
        .fold(0_u32, |total, lane| total.wrapping_add(*lane));
    scaled.map(|lane| lane.wrapping_add(sum))
}
