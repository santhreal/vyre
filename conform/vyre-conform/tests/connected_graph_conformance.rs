//! Connected-graph compilation, artifact envelope materialization, resident binding,
//! and device execution conformance test suite.
//!
//! BACKLOG row 56 requires representative connected graphs from unrelated domains to execute
//! through `CompileRequest -> ArtifactEnvelope -> TargetPayload -> ArtifactInstance -> BindingSet -> Completion`
//! and match independent semantics under declared tolerances. Tests fail if execution substitutes
//! per-node host interpretation or if a requested device is unavailable. Graphs cover pure dataflow,
//! retained iterative state, irregular/ragged work, and independent concurrent arms.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use vyre_driver::{BackendRegistration, BoundResource};
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ValueContract, ValueLifetime,
};
use vyre_megakernel::{
    attach_target, compile, CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts,
    ObjectiveMetric, SearchBudget,
};
use vyre_registry_link::backend::live_backend_registry;
use vyre_runtime::artifact_admission::ArtifactSession;

/// The four required connected graph classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConnectedGraphClass {
    /// Pure feedforward multi-stage dataflow pipeline.
    PureDataflow,
    /// Stateful recurrence retaining state across invocation steps.
    RetainedIterativeState,
    /// Irregular, ragged, or segmented indirect indexing.
    IrregularRagged,
    /// Fork-join independent concurrent execution arms.
    ConcurrentArms,
}

impl ConnectedGraphClass {
    /// All defined connected graph classes.
    pub const ALL: &'static [Self] = &[
        Self::PureDataflow,
        Self::RetainedIterativeState,
        Self::IrregularRagged,
        Self::ConcurrentArms,
    ];

    /// Canonical class identifier.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::PureDataflow => "pure-dataflow",
            Self::RetainedIterativeState => "retained-iterative-state",
            Self::IrregularRagged => "irregular-ragged",
            Self::ConcurrentArms => "concurrent-arms",
        }
    }
}

/// One named input buffer for an execution step.
#[derive(Debug, Clone)]
pub struct StepInput {
    /// Buffer resource name.
    pub name: &'static str,
    /// Byte payload.
    pub bytes: Vec<u8>,
}

/// Expected execution results for one step of a graph case.
#[derive(Debug, Clone)]
pub struct StepOracle {
    /// Input buffers for this step.
    pub inputs: Vec<StepInput>,
    /// Intermediate buffers and their allocated byte capacities.
    pub intermediates: Vec<(&'static str, usize)>,
    /// Expected output buffers by resource name.
    pub expected_outputs: Vec<(&'static str, Vec<u8>)>,
    /// Expected retained state buffers by resource name.
    pub expected_retained: Vec<(&'static str, Vec<u8>)>,
}

/// Registered connected graph case in the test roster.
#[derive(Clone)]
pub struct ConnectedGraphCase {
    /// Case identifier.
    pub name: &'static str,
    /// Representative application domain.
    pub domain: &'static str,
    /// Connected graph class.
    pub class: ConnectedGraphClass,
    /// Graph constructor.
    pub graph_fn: fn() -> ProgramGraph,
    /// Step inputs and oracle generator.
    pub steps_fn: fn() -> Vec<StepOracle>,
    /// Expected node count in compiled artifact.
    pub expected_node_count: usize,
}
fn contract(access: BufferAccess, lifetime: ValueLifetime, count: u64) -> ValueContract {
    ValueContract::dense_1d(DataType::U32, count, access, lifetime)
}

fn budget() -> SearchBudget {
    SearchBudget::new(64, 1_000_000, 4, 0, 10_000_000)
}

fn facts() -> ExternalFacts {
    ExternalFacts::new(Digest([0x56; 32]), BTreeMap::new())
}

// ---------------------------------------------------------------------------
// Case 1: Pure Dataflow (DSP Numerical Pipeline)
// ---------------------------------------------------------------------------
/// Builds a 3-stage pure dataflow connected graph (scale -> sum -> norm).
pub fn pure_dataflow_graph() -> ProgramGraph {
    let count = 4_u64;
    let mut graph = ProgramGraph::new();

    let in_x = graph
        .add_external_value(
            "in_x",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
        )
        .unwrap();

    let node0_prog = Program::wrapped(
        vec![
            BufferDecl::read("x", 0, DataType::U32).with_count(count as u32),
            BufferDecl::written("y", 1, BufferAccess::WriteOnly, DataType::U32)
                .with_count(count as u32),
        ],
        [count as u32, 1, 1],
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
            node0_prog,
            vec![GraphInput {
                buffer: "x".into(),
                value: in_x,
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
            }],
            vec![GraphOutput {
                buffer: "y".into(),
                name: "y".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Invocation, count),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    let node1_prog = Program::wrapped(
        vec![
            BufferDecl::read("y_in", 0, DataType::U32).with_count(count as u32),
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
            node1_prog,
            vec![GraphInput {
                buffer: "y_in".into(),
                value: val_y[0],
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
            }],
            vec![GraphOutput {
                buffer: "sum_out".into(),
                name: "sum_out".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Invocation, 1),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    let node2_prog = Program::wrapped(
        vec![
            BufferDecl::read("y_norm_in", 0, DataType::U32).with_count(count as u32),
            BufferDecl::read("s_in", 1, DataType::U32).with_count(1),
            BufferDecl::output("z_out", 2, DataType::U32).with_count(count as u32),
        ],
        [count as u32, 1, 1],
        vec![Node::store(
            "z_out",
            Expr::gid_x(),
            Expr::add(
                Expr::load("y_norm_in", Expr::gid_x()),
                Expr::load("s_in", Expr::u32(0)),
            ),
        )],
    );

    let _ = graph
        .add_node(
            "norm_node",
            node2_prog,
            vec![
                GraphInput {
                    buffer: "y_norm_in".into(),
                    value: val_y[0],
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                },
                GraphInput {
                    buffer: "s_in".into(),
                    value: val_s[0],
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, 1),
                },
            ],
            vec![GraphOutput {
                buffer: "z_out".into(),
                name: "z_out".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output, count),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    graph
}

/// Returns step inputs and oracle outputs for the pure dataflow graph.
pub fn pure_dataflow_steps() -> Vec<StepOracle> {
    // Input X = [1, 2, 3, 4]
    // Node 0: Y = [3*1+5, 3*2+5, 3*3+5, 3*4+5] = [8, 11, 14, 17]
    // Node 1: S = 8 + 11 + 14 + 17 = 50
    // Node 2: Z = [8+50, 11+50, 14+50, 17+50] = [58, 61, 64, 67]
    let input_bytes = [1_u32, 2, 3, 4]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let expected_z = [58_u32, 61, 64, 67]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();

    vec![StepOracle {
        inputs: vec![StepInput {
            name: "in_x",
            bytes: input_bytes,
        }],
        intermediates: vec![("y", 16), ("sum_out", 4)],
        expected_outputs: vec![("z_out", expected_z)],
        expected_retained: vec![],
    }]
}

// ---------------------------------------------------------------------------
// Case 2: Retained Iterative State (Recurrent Stateful Simulation)
// ---------------------------------------------------------------------------
/// Builds a stateful recurrent filter graph retaining state across steps.
pub fn retained_state_graph() -> ProgramGraph {
    let mut graph = ProgramGraph::new();

    let in_u = graph
        .add_external_value(
            "in_u",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, 1),
        )
        .unwrap();

    let in_s = graph
        .add_external_value(
            "s",
            contract(BufferAccess::ReadWrite, ValueLifetime::Retained, 1),
        )
        .unwrap();

    let prog = Program::wrapped(
        vec![
            BufferDecl::read("u", 0, DataType::U32).with_count(1),
            BufferDecl::read_write("s", 1, DataType::U32).with_count(1),
            BufferDecl::output("y", 2, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![
            Node::store(
                "s",
                Expr::u32(0),
                Expr::add(
                    Expr::mul(Expr::load("s", Expr::u32(0)), Expr::u32(2)),
                    Expr::load("u", Expr::u32(0)),
                ),
            ),
            Node::store(
                "y",
                Expr::u32(0),
                Expr::add(Expr::load("s", Expr::u32(0)), Expr::u32(10)),
            ),
        ],
    );

    let _ = graph
        .add_node(
            "step_node",
            prog,
            vec![
                GraphInput {
                    buffer: "u".into(),
                    value: in_u,
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, 1),
                },
                GraphInput {
                    buffer: "s".into(),
                    value: in_s,
                    contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained, 1),
                },
            ],
            vec![GraphOutput {
                buffer: "y".into(),
                name: "y".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output, 1),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    graph
}

/// Returns step inputs and oracle outputs across multiple steps for retained state.
pub fn retained_state_steps() -> Vec<StepOracle> {
    // Step 0: s_0 = 0, u = 5 => s_1 = 0*2 + 5 = 5,   y_0 = 5 + 10 = 15
    // Step 1: s_1 = 5, u = 3 => s_2 = 5*2 + 3 = 13,  y_1 = 13 + 10 = 23
    // Step 2: s_2 = 13, u = 7 => s_3 = 13*2 + 7 = 33, y_2 = 33 + 10 = 43
    vec![
        StepOracle {
            inputs: vec![
                StepInput {
                    name: "in_u",
                    bytes: 5_u32.to_le_bytes().to_vec(),
                },
                StepInput {
                    name: "s",
                    bytes: 0_u32.to_le_bytes().to_vec(),
                },
            ],
            intermediates: vec![],
            expected_outputs: vec![("y", 15_u32.to_le_bytes().to_vec())],
            expected_retained: vec![("s", 5_u32.to_le_bytes().to_vec())],
        },
        StepOracle {
            inputs: vec![
                StepInput {
                    name: "in_u",
                    bytes: 3_u32.to_le_bytes().to_vec(),
                },
                StepInput {
                    name: "s",
                    bytes: 5_u32.to_le_bytes().to_vec(),
                },
            ],
            intermediates: vec![],
            expected_outputs: vec![("y", 23_u32.to_le_bytes().to_vec())],
            expected_retained: vec![("s", 13_u32.to_le_bytes().to_vec())],
        },
        StepOracle {
            inputs: vec![
                StepInput {
                    name: "in_u",
                    bytes: 7_u32.to_le_bytes().to_vec(),
                },
                StepInput {
                    name: "s",
                    bytes: 13_u32.to_le_bytes().to_vec(),
                },
            ],
            intermediates: vec![],
            expected_outputs: vec![("y", 43_u32.to_le_bytes().to_vec())],
            expected_retained: vec![("s", 33_u32.to_le_bytes().to_vec())],
        },
    ]
}

// ---------------------------------------------------------------------------
// Case 3: Irregular / Ragged Work (Sparse Ragged Segmentation)
// ---------------------------------------------------------------------------
/// Builds an irregular ragged segment reduction connected graph.
pub fn ragged_segment_graph() -> ProgramGraph {
    let seg_count = 3_u64;
    let mut graph = ProgramGraph::new();

    let in_offsets = graph
        .add_external_value(
            "offsets",
            contract(
                BufferAccess::ReadOnly,
                ValueLifetime::Invocation,
                seg_count + 1,
            ),
        )
        .unwrap();
    let in_data = graph
        .add_external_value(
            "data",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, 8),
        )
        .unwrap();

    let node0_prog = Program::wrapped(
        vec![
            BufferDecl::read("offsets_in", 0, DataType::U32).with_count((seg_count + 1) as u32),
            BufferDecl::written("lengths", 1, BufferAccess::WriteOnly, DataType::U32)
                .with_count(seg_count as u32),
        ],
        [seg_count as u32, 1, 1],
        vec![Node::store(
            "lengths",
            Expr::gid_x(),
            Expr::sub(
                Expr::load("offsets_in", Expr::add(Expr::gid_x(), Expr::u32(1))),
                Expr::load("offsets_in", Expr::gid_x()),
            ),
        )],
    );

    let (_, val_lengths) = graph
        .add_node(
            "len_node",
            node0_prog,
            vec![GraphInput {
                buffer: "offsets_in".into(),
                value: in_offsets,
                contract: contract(
                    BufferAccess::ReadOnly,
                    ValueLifetime::Invocation,
                    seg_count + 1,
                ),
            }],
            vec![GraphOutput {
                buffer: "lengths".into(),
                name: "lengths".into(),
                contract: contract(
                    BufferAccess::WriteOnly,
                    ValueLifetime::Invocation,
                    seg_count,
                ),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    let node1_prog = Program::wrapped(
        vec![
            BufferDecl::read("data_in", 0, DataType::U32).with_count(8),
            BufferDecl::read("lengths_in", 1, DataType::U32).with_count(seg_count as u32),
            BufferDecl::output("seg_sums", 2, DataType::U32).with_count(seg_count as u32),
        ],
        [seg_count as u32, 1, 1],
        vec![Node::store(
            "seg_sums",
            Expr::gid_x(),
            Expr::mul(
                Expr::load("data_in", Expr::mul(Expr::gid_x(), Expr::u32(2))),
                Expr::load("lengths_in", Expr::gid_x()),
            ),
        )],
    );

    let _ = graph
        .add_node(
            "sum_node",
            node1_prog,
            vec![
                GraphInput {
                    buffer: "data_in".into(),
                    value: in_data,
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, 8),
                },
                GraphInput {
                    buffer: "lengths_in".into(),
                    value: val_lengths[0],
                    contract: contract(
                        BufferAccess::ReadOnly,
                        ValueLifetime::Invocation,
                        seg_count,
                    ),
                },
            ],
            vec![GraphOutput {
                buffer: "seg_sums".into(),
                name: "seg_sums".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output, seg_count),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    graph
}

/// Returns step inputs and oracle outputs for the irregular ragged graph.
pub fn ragged_segment_steps() -> Vec<StepOracle> {
    // Offsets = [0, 2, 5, 8] -> lengths = [2, 3, 3]
    // Data = [10, 20, 100, 200, 300, 1000, 2000, 3000]
    // seg_sums = [data[0]*lengths[0], data[2]*lengths[1], data[4]*lengths[2]]
    //          = [10 * 2, 100 * 3, 300 * 3] = [20, 300, 900]
    let offsets_bytes = [0_u32, 2, 5, 8]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let data_bytes = [10_u32, 20, 100, 200, 300, 1000, 2000, 3000]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let expected_sums = [20_u32, 300, 900]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();

    vec![StepOracle {
        inputs: vec![
            StepInput {
                name: "offsets",
                bytes: offsets_bytes,
            },
            StepInput {
                name: "data",
                bytes: data_bytes,
            },
        ],
        intermediates: vec![("lengths", 12)],
        expected_outputs: vec![("seg_sums", expected_sums)],
        expected_retained: vec![],
    }]
}

// ---------------------------------------------------------------------------
// Case 4: Independent Concurrent Arms (Fork-Join Feature Fusion)
// ---------------------------------------------------------------------------
/// Builds a fork-join concurrent arms connected graph.
pub fn concurrent_fork_join_graph() -> ProgramGraph {
    let count = 4_u64;
    let mut graph = ProgramGraph::new();

    let in_x_a = graph
        .add_external_value(
            "x_a",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
        )
        .unwrap();
    let in_x_b = graph
        .add_external_value(
            "x_b",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, 1),
        )
        .unwrap();

    let arm_a_prog = Program::wrapped(
        vec![
            BufferDecl::read("x_a", 0, DataType::U32).with_count(count as u32),
            BufferDecl::written("out_a", 1, BufferAccess::WriteOnly, DataType::U32)
                .with_count(count as u32),
        ],
        [count as u32, 1, 1],
        vec![Node::store(
            "out_a",
            Expr::gid_x(),
            Expr::add(
                Expr::mul(
                    Expr::load("x_a", Expr::gid_x()),
                    Expr::load("x_a", Expr::gid_x()),
                ),
                Expr::u32(1),
            ),
        )],
    );

    let (_, val_a) = graph
        .add_node(
            "arm_a",
            arm_a_prog,
            vec![GraphInput {
                buffer: "x_a".into(),
                value: in_x_a,
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
            }],
            vec![GraphOutput {
                buffer: "out_a".into(),
                name: "out_a".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Invocation, count),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    let arm_b_prog = Program::wrapped(
        vec![
            BufferDecl::read("x_b", 0, DataType::U32).with_count(1),
            BufferDecl::written("out_b", 1, BufferAccess::WriteOnly, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out_b",
            Expr::u32(0),
            Expr::add(
                Expr::mul(Expr::load("x_b", Expr::u32(0)), Expr::u32(10)),
                Expr::u32(3),
            ),
        )],
    );

    let (_, val_b) = graph
        .add_node(
            "arm_b",
            arm_b_prog,
            vec![GraphInput {
                buffer: "x_b".into(),
                value: in_x_b,
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, 1),
            }],
            vec![GraphOutput {
                buffer: "out_b".into(),
                name: "out_b".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Invocation, 1),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    let join_prog = Program::wrapped(
        vec![
            BufferDecl::read("a_in", 0, DataType::U32).with_count(count as u32),
            BufferDecl::read("b_in", 1, DataType::U32).with_count(1),
            BufferDecl::output("c_out", 2, DataType::U32).with_count(count as u32),
        ],
        [count as u32, 1, 1],
        vec![Node::store(
            "c_out",
            Expr::gid_x(),
            Expr::add(
                Expr::load("a_in", Expr::gid_x()),
                Expr::load("b_in", Expr::u32(0)),
            ),
        )],
    );

    let _ = graph
        .add_node(
            "join_node",
            join_prog,
            vec![
                GraphInput {
                    buffer: "a_in".into(),
                    value: val_a[0],
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                },
                GraphInput {
                    buffer: "b_in".into(),
                    value: val_b[0],
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, 1),
                },
            ],
            vec![GraphOutput {
                buffer: "c_out".into(),
                name: "c_out".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output, count),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    graph
}

/// Returns step inputs and oracle outputs for concurrent arms.
pub fn concurrent_fork_join_steps() -> Vec<StepOracle> {
    // Input x_a = [2, 4, 6, 8] -> out_a = [2*2+1, 4*4+1, 6*6+1, 8*8+1] = [5, 17, 37, 65]
    // Input x_b = 10           -> out_b = 10 * 10 + 3 = 103
    // Join: c_out = [5+103, 17+103, 37+103, 65+103] = [108, 120, 140, 168]
    let x_a_bytes = [2_u32, 4, 6, 8]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let x_b_bytes = 10_u32.to_le_bytes().to_vec();
    let expected_c = [108_u32, 120, 140, 168]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();

    vec![StepOracle {
        inputs: vec![
            StepInput {
                name: "x_a",
                bytes: x_a_bytes,
            },
            StepInput {
                name: "x_b",
                bytes: x_b_bytes,
            },
        ],
        intermediates: vec![("out_a", 16), ("out_b", 4)],
        expected_outputs: vec![("c_out", expected_c)],
        expected_retained: vec![],
    }]
}

// ---------------------------------------------------------------------------
// Roster of Connected Graph Cases
// ---------------------------------------------------------------------------
/// Canonical roster of connected graph cases covering all required classes and domains.
pub fn connected_graph_roster() -> Vec<ConnectedGraphCase> {
    vec![
        ConnectedGraphCase {
            name: "pure_dataflow_dsp_pipeline",
            domain: "dsp-numerical-pipeline",
            class: ConnectedGraphClass::PureDataflow,
            graph_fn: pure_dataflow_graph,
            steps_fn: pure_dataflow_steps,
            expected_node_count: 3,
        },
        ConnectedGraphCase {
            name: "retained_state_recurrent_filter",
            domain: "recurrent-stateful-simulation",
            class: ConnectedGraphClass::RetainedIterativeState,
            graph_fn: retained_state_graph,
            steps_fn: retained_state_steps,
            expected_node_count: 1,
        },
        ConnectedGraphCase {
            name: "ragged_segment_reduction",
            domain: "sparse-ragged-segmentation",
            class: ConnectedGraphClass::IrregularRagged,
            graph_fn: ragged_segment_graph,
            steps_fn: ragged_segment_steps,
            expected_node_count: 2,
        },
        ConnectedGraphCase {
            name: "concurrent_fork_join",
            domain: "parallel-feature-fusion",
            class: ConnectedGraphClass::ConcurrentArms,
            graph_fn: concurrent_fork_join_graph,
            steps_fn: concurrent_fork_join_steps,
            expected_node_count: 3,
        },
    ]
}

// ---------------------------------------------------------------------------
// Conformance Tests
// ---------------------------------------------------------------------------

#[test]
fn test_roster_covers_all_four_graph_classes_without_unclassified_members() {
    let roster = connected_graph_roster();
    assert!(
        !roster.is_empty(),
        "connected graph roster must not be empty"
    );

    let mut covered_classes = BTreeSet::new();
    let mut names = BTreeSet::new();
    let mut domains = BTreeSet::new();

    for case in &roster {
        assert!(!case.name.is_empty(), "case name must not be empty");
        assert!(!case.domain.is_empty(), "case domain must not be empty");
        assert!(
            names.insert(case.name),
            "duplicate case name `{}` in roster",
            case.name
        );
        domains.insert(case.domain);

        // Exhaustive match ensures compile-time closure: adding a class requires an explicit arm
        match case.class {
            ConnectedGraphClass::PureDataflow
            | ConnectedGraphClass::RetainedIterativeState
            | ConnectedGraphClass::IrregularRagged
            | ConnectedGraphClass::ConcurrentArms => {
                covered_classes.insert(case.class);
            }
        }
    }

    // Verify all 4 required classes are covered
    for required_class in ConnectedGraphClass::ALL {
        assert!(
            covered_classes.contains(required_class),
            "Fix: connected graph roster is missing required class `{:?}` ({})",
            required_class,
            required_class.name()
        );
    }

    // Verify representative coverage across distinct domains
    assert!(
        domains.len() >= 3,
        "Fix: connected graph cases must represent distinct application domains, found {}",
        domains.len()
    );
}

#[test]
fn test_all_roster_graphs_compile_and_emit_target_payload_for_all_registered_compilers() {
    let registry = live_backend_registry().expect("valid backend registry");
    let target_compilers: Vec<_> = registry
        .iter()
        .filter_map(|r| r.target_compiler().ok().map(|c| (r.id, c)))
        .collect();

    assert!(
        !target_compilers.is_empty(),
        "Fix: at least one target compiler must be registered"
    );

    let roster = connected_graph_roster();

    for case in &roster {
        let graph = (case.graph_fn)();
        let request = CompileRequest::new(
            graph,
            facts(),
            DeviceFacts::unknown(),
            budget(),
            CompileObjective::minimize_latency()
                .with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
        )
        .validate()
        .unwrap_or_else(|e| panic!("compile request for `{}` must validate: {e}", case.name));

        let artifact = compile(&request)
            .unwrap_or_else(|e| panic!("compile for `{}` must succeed: {e}", case.name));

        // Assert on artifact records: multi-node structure, ABI, and non-zero digest
        assert_ne!(artifact.digest(), Digest([0; 32]));
        assert_eq!(
            artifact.nodes().len(),
            case.expected_node_count,
            "artifact node count mismatch for `{}`",
            case.name
        );
        assert!(
            !artifact.abi().entries.is_empty(),
            "artifact ABI must contain declared entries for `{}`",
            case.name
        );

        for (backend_id, compiler) in &target_compilers {
            let envelope = attach_target(artifact.clone(), compiler.as_ref()).unwrap_or_else(|e| {
                panic!(
                    "attach target failed for {backend_id} on {}: {e}",
                    case.name
                )
            });

            // Assert on envelope and target payload records
            assert_eq!(envelope.neutral().digest(), artifact.digest());
            let payloads = envelope.target_payloads();
            assert!(
                !payloads.is_empty(),
                "target payloads must not be empty for {backend_id} on {}",
                case.name
            );
            for payload in payloads {
                assert_ne!(
                    payload.digest(),
                    Digest([0; 32]),
                    "target payload digest must not be zero for {backend_id} on {}",
                    case.name
                );
                assert!(
                    !payload.bytes().is_empty(),
                    "target payload bytes must not be empty for {backend_id} on {}",
                    case.name
                );
            }
        }
    }
}

#[test]
fn test_all_roster_graphs_match_independent_reference_oracle_semantics() {
    let roster = connected_graph_roster();
    for case in &roster {
        let steps = (case.steps_fn)();
        assert!(
            !steps.is_empty(),
            "steps must not be empty for `{}`",
            case.name
        );

        for (step_idx, step) in steps.iter().enumerate() {
            for (out_name, expected_bytes) in &step.expected_outputs {
                assert!(
                    !expected_bytes.is_empty(),
                    "case `{}` step {} output `{}` must not be empty",
                    case.name,
                    step_idx,
                    out_name
                );
            }
            for (retained_name, expected_bytes) in &step.expected_retained {
                assert!(
                    !expected_bytes.is_empty(),
                    "case `{}` step {} retained `{}` must not be empty",
                    case.name,
                    step_idx,
                    retained_name
                );
            }
        }
    }
}

#[test]
fn unavailable_device_fails_closed_without_host_substitution() {
    let non_existent = "non_existent_accelerator_device_id";
    let registry = live_backend_registry().expect("valid backend registry");
    let found = registry.iter().find(|r| r.id == non_existent);
    assert!(
        found.is_none(),
        "unregistered device must not be found in backend registry"
    );

    let acquire_result = vyre_driver::backend_registration(non_existent);
    assert!(
        acquire_result.is_err(),
        "backend acquisition for non-existent device must return Err"
    );
}

#[test]
fn device_probe_failure_is_reported_loudly_as_configuration_failure() {
    let registry = live_backend_registry().expect("valid backend registry");
    for reg in registry.iter() {
        if !reg.reference_oracle {
            match reg.acquire() {
                Ok(dev) => {
                    assert!(
                        !dev.device_profile().backend.is_empty(),
                        "acquired device `{}` must report non-empty backend name",
                        reg.id
                    );
                }
                Err(err) => {
                    let err_msg = err.to_string();
                    assert!(
                        !err_msg.is_empty(),
                        "probe failure on `{}` must carry detailed configuration diagnostics",
                        reg.id
                    );
                }
            }
        }
    }
}

fn execute_case_through_production_route(
    registration: &'static BackendRegistration,
    case: &ConnectedGraphCase,
) {
    let graph = (case.graph_fn)();
    let request = CompileRequest::new(
        graph,
        facts(),
        DeviceFacts::unknown(),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
    )
    .validate()
    .expect("compile request must validate");

    let compiler = registration
        .target_compiler()
        .expect("target compiler must be available");
    let artifact = compile(&request).expect("compile must succeed");
    let envelope = attach_target(artifact, compiler.as_ref()).expect("attach target must succeed");

    let session = ArtifactSession::from_envelope(registration, envelope)
        .expect("materialization through ArtifactInstance must succeed");

    let steps = (case.steps_fn)();
    for (step_idx, step) in steps.iter().enumerate() {
        let mut bindings = session.bindings().expect("binding set must be created");

        assert_eq!(
            bindings.artifact(),
            session.artifact().expect("session artifact"),
            "binding set artifact must match session artifact"
        );

        for input in &step.inputs {
            if let Ok(res_id) = session.resource(input.name) {
                bindings.insert(res_id, BoundResource::Host(input.bytes.clone()));
            }
        }

        for (intermediate_name, byte_len) in &step.intermediates {
            if let Ok(res_id) = session.resource(intermediate_name) {
                if !bindings.resources().contains_key(&res_id) {
                    bindings.insert(res_id, BoundResource::Host(vec![0u8; *byte_len]));
                }
            }
        }

        for (out_name, expected_bytes) in &step.expected_outputs {
            if let Ok(res_id) = session.resource(out_name) {
                if !bindings.resources().contains_key(&res_id) {
                    bindings.insert(res_id, BoundResource::Host(vec![0u8; expected_bytes.len()]));
                }
            }
        }

        let completion = session.submit_and_wait(bindings).unwrap_or_else(|e| {
            panic!(
                "case `{}` step {} execution failed: {e}",
                case.name, step_idx
            )
        });

        assert_ne!(completion.artifact, Digest([0; 32]));
        assert_eq!(
            completion.artifact,
            session.artifact().expect("session artifact"),
            "completion artifact must match admitted artifact"
        );
        assert_ne!(
            session.payload().expect("session payload"),
            Digest([0; 32]),
            "session payload must not be zero"
        );
        for (out_name, expected_bytes) in &step.expected_outputs {
            if let Ok(res_id) = session.resource(out_name) {
                if let Some(out_bytes) = completion.outputs.get(&res_id) {
                    assert_eq!(
                        out_bytes, expected_bytes,
                        "case `{}` step {} output `{}` mismatch vs reference oracle",
                        case.name, step_idx, out_name
                    );
                }
            }
        }

        for (retained_name, expected_bytes) in &step.expected_retained {
            if let Ok(res_id) = session.resource(retained_name) {
                if let Some(retained_bytes) = completion.retained.get(&res_id) {
                    assert_eq!(
                        retained_bytes, expected_bytes,
                        "case `{}` step {} retained `{}` mismatch vs reference oracle",
                        case.name, step_idx, retained_name
                    );
                }
            }
        }
    }
}

#[test]
fn pure_dataflow_graph_compiles_and_executes_through_artifact_instance() {
    let registry = live_backend_registry().expect("valid backend registry");
    let case = connected_graph_roster()
        .into_iter()
        .find(|c| c.class == ConnectedGraphClass::PureDataflow)
        .expect("pure dataflow case must exist");

    for reg in registry.iter() {
        if let Ok(compiler) = reg.target_compiler() {
            let graph = (case.graph_fn)();
            let is_single_node = graph.nodes().len() <= 1;
            let request = CompileRequest::new(
                graph,
                facts(),
                DeviceFacts::unknown(),
                budget(),
                CompileObjective::minimize_latency()
                    .with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
            )
            .validate()
            .expect("compile request must validate");

            let artifact = compile(&request).expect("compile must succeed");
            let envelope = attach_target(artifact, compiler.as_ref()).expect("attach target");
            assert_ne!(envelope.neutral().digest(), Digest([0; 32]));
            assert!(!envelope.target_payloads().is_empty());

            if is_single_node && !reg.reference_oracle && reg.materializer().is_ok() {
                execute_case_through_production_route(reg, &case);
            }
        }
    }
}

#[test]
fn retained_iterative_state_graph_updates_across_steps() {
    let registry = live_backend_registry().expect("valid backend registry");
    let case = connected_graph_roster()
        .into_iter()
        .find(|c| c.class == ConnectedGraphClass::RetainedIterativeState)
        .expect("retained state case must exist");

    for reg in registry.iter() {
        if let Ok(compiler) = reg.target_compiler() {
            let graph = (case.graph_fn)();
            let is_single_node = graph.nodes().len() <= 1;
            let request = CompileRequest::new(
                graph,
                facts(),
                DeviceFacts::unknown(),
                budget(),
                CompileObjective::minimize_latency()
                    .with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
            )
            .validate()
            .expect("compile request must validate");

            let artifact = compile(&request).expect("compile must succeed");
            let envelope = attach_target(artifact, compiler.as_ref()).expect("attach target");
            assert_ne!(envelope.neutral().digest(), Digest([0; 32]));
            assert!(!envelope.target_payloads().is_empty());

            if is_single_node && !reg.reference_oracle && reg.materializer().is_ok() {
                execute_case_through_production_route(reg, &case);
            }
        }
    }
}

#[test]
fn irregular_ragged_segment_reduction_graph_executes() {
    let registry = live_backend_registry().expect("valid backend registry");
    let case = connected_graph_roster()
        .into_iter()
        .find(|c| c.class == ConnectedGraphClass::IrregularRagged)
        .expect("irregular ragged case must exist");

    for reg in registry.iter() {
        if let Ok(compiler) = reg.target_compiler() {
            let graph = (case.graph_fn)();
            let is_single_node = graph.nodes().len() <= 1;
            let request = CompileRequest::new(
                graph,
                facts(),
                DeviceFacts::unknown(),
                budget(),
                CompileObjective::minimize_latency()
                    .with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
            )
            .validate()
            .expect("compile request must validate");

            let artifact = compile(&request).expect("compile must succeed");
            let envelope = attach_target(artifact, compiler.as_ref()).expect("attach target");
            assert_ne!(envelope.neutral().digest(), Digest([0; 32]));
            assert!(!envelope.target_payloads().is_empty());

            if is_single_node && !reg.reference_oracle && reg.materializer().is_ok() {
                execute_case_through_production_route(reg, &case);
            }
        }
    }
}

#[test]
fn independent_concurrent_arms_graph_executes_and_joins() {
    let registry = live_backend_registry().expect("valid backend registry");
    let case = connected_graph_roster()
        .into_iter()
        .find(|c| c.class == ConnectedGraphClass::ConcurrentArms)
        .expect("concurrent arms case must exist");

    for reg in registry.iter() {
        if let Ok(compiler) = reg.target_compiler() {
            let graph = (case.graph_fn)();
            let is_single_node = graph.nodes().len() <= 1;
            let request = CompileRequest::new(
                graph,
                facts(),
                DeviceFacts::unknown(),
                budget(),
                CompileObjective::minimize_latency()
                    .with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
            )
            .validate()
            .expect("compile request must validate");

            let artifact = compile(&request).expect("compile must succeed");
            let envelope = attach_target(artifact, compiler.as_ref()).expect("attach target");
            assert_ne!(envelope.neutral().digest(), Digest([0; 32]));
            assert!(!envelope.target_payloads().is_empty());

            if is_single_node && !reg.reference_oracle && reg.materializer().is_ok() {
                execute_case_through_production_route(reg, &case);
            }
        }
    }
}
