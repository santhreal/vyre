//! Typed multi-Program graph contracts.

#![forbid(unsafe_code)]

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, GraphInput, GraphOutput, GraphValueId, Node, Program,
    ProgramGraph, ProgramGraphError, ShapeDim, TensorContract, ValueLifetime,
};

fn contract(access: BufferAccess, lifetime: ValueLifetime) -> TensorContract {
    TensorContract {
        dtype: DataType::F32,
        shape: vec![ShapeDim::Symbol("tokens".into()), ShapeDim::Known(8)],
        access,
        lifetime,
    }
}

fn copy_program(input: &str, output: &str) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::F32),
            BufferDecl::storage(output, 1, BufferAccess::ReadWrite, DataType::F32),
        ],
        [1, 1, 1],
        vec![Node::store(
            output,
            vyre_foundation::ir::Expr::u32(0),
            vyre_foundation::ir::Expr::load(input, vyre_foundation::ir::Expr::u32(0)),
        )],
    )
}

fn two_output_program(input: &str, first: &str, second: &str) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::F32),
            BufferDecl::storage(first, 1, BufferAccess::ReadWrite, DataType::F32),
            BufferDecl::storage(second, 2, BufferAccess::ReadWrite, DataType::F32),
        ],
        [1, 1, 1],
        Vec::new(),
    )
}

fn two_input_program(first: &str, second: &str) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(first, 0, BufferAccess::ReadOnly, DataType::F32),
            BufferDecl::storage(second, 1, BufferAccess::ReadOnly, DataType::F32),
        ],
        [1, 1, 1],
        Vec::new(),
    )
}

fn stateful_wire_graph() -> ProgramGraph {
    let state_contract = TensorContract {
        dtype: DataType::F32,
        shape: vec![ShapeDim::Symbol("batch".into()), ShapeDim::Known(8)],
        access: BufferAccess::ReadWrite,
        lifetime: ValueLifetime::SequenceState,
    };
    let mut graph = ProgramGraph::new();
    graph
        .add_external_value(
            "weight",
            TensorContract {
                dtype: DataType::BF16,
                shape: vec![ShapeDim::Known(2), ShapeDim::Known(4)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::ImmutableWeight,
            },
        )
        .expect("Fix: wire fixture weight must register");
    let state = graph
        .add_external_value("cache.0", state_contract.clone())
        .expect("Fix: wire fixture state must register");
    graph
        .add_node(
            "decode",
            Program::wrapped(
                vec![BufferDecl::storage(
                    "cache",
                    0,
                    BufferAccess::ReadWrite,
                    DataType::F32,
                )],
                [1, 1, 1],
                Vec::new(),
            ),
            vec![GraphInput {
                buffer: "cache".into(),
                value: state,
                contract: state_contract.clone(),
            }],
            vec![GraphOutput {
                buffer: "cache".into(),
                name: "cache.1".into(),
                contract: state_contract,
                state_successor_of: Some(state),
            }],
        )
        .expect("Fix: wire fixture state transition must connect");
    graph
}

/// Proves typed values connect ordinary Programs without replacing their executable IR.
#[test]
fn connected_programs_have_exact_schedule_and_liveness() {
    let mut graph = ProgramGraph::new();
    let tokens = graph
        .add_external_value(
            "tokens",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
        )
        .expect("Fix: graph input must be valid");
    let (_, hidden) = graph
        .add_node(
            "embed",
            copy_program("tokens", "hidden"),
            vec![GraphInput {
                buffer: "tokens".into(),
                value: tokens,
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "hidden".into(),
                name: "hidden.0".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation),
                state_successor_of: None,
            }],
        )
        .expect("Fix: first graph node must connect");
    let (_, logits) = graph
        .add_node(
            "head",
            copy_program("hidden", "logits"),
            vec![GraphInput {
                buffer: "hidden".into(),
                value: hidden[0],
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "logits".into(),
                name: "logits".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Output),
                state_successor_of: None,
            }],
        )
        .expect("Fix: second graph node must connect");

    assert_eq!(
        graph.schedule().iter().map(|id| id.0).collect::<Vec<_>>(),
        [0, 1]
    );
    assert_eq!(logits, [GraphValueId(2)]);
    assert_eq!(
        graph
            .liveness_intervals()
            .iter()
            .map(|interval| (interval.value.0, interval.start, interval.end))
            .collect::<Vec<_>>(),
        [(0, 0, 0), (1, 0, 1), (2, 1, 1)]
    );
    assert_eq!(graph.values()[1].consumers[0].0, 1);
}

/// Prevents a model edge from silently binding to a misspelled Program buffer.
#[test]
fn missing_program_buffer_fails_closed() {
    let mut graph = ProgramGraph::new();
    let value = graph
        .add_external_value(
            "input",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
        )
        .expect("Fix: graph input must be valid");
    let error = graph
        .add_node(
            "layer",
            copy_program("input", "output"),
            vec![GraphInput {
                buffer: "misspelled".into(),
                value,
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
            }],
            Vec::new(),
        )
        .expect_err("Fix: unknown buffer names must fail");
    assert_eq!(
        error,
        ProgramGraphError::MissingBuffer {
            node: "layer".into(),
            buffer: "misspelled".into(),
        }
    );
}

/// Prevents dtype or access drift between model ports and executable Program buffers.
#[test]
fn incompatible_tensor_contract_fails_closed() {
    let mut graph = ProgramGraph::new();
    let value = graph
        .add_external_value(
            "input",
            TensorContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(8)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
        )
        .expect("Fix: external values may use any supported dtype");
    let error = graph
        .add_node(
            "layer",
            copy_program("input", "output"),
            vec![GraphInput {
                buffer: "input".into(),
                value,
                contract: TensorContract {
                    dtype: DataType::U32,
                    shape: vec![ShapeDim::Known(8)],
                    access: BufferAccess::ReadOnly,
                    lifetime: ValueLifetime::Invocation,
                },
            }],
            Vec::new(),
        )
        .expect_err("Fix: mismatched dtypes must fail");
    assert!(matches!(error, ProgramGraphError::BufferContract { .. }));
    assert!(error
        .to_string()
        .contains("Program uses F32, graph uses U32"));
}

/// Locks decode state to an explicit type-preserving successor edge.
#[test]
fn sequence_state_transition_preserves_contract() {
    let state_contract = contract(BufferAccess::ReadWrite, ValueLifetime::SequenceState);
    let mut graph = ProgramGraph::new();
    let state = graph
        .add_external_value("cache.0", state_contract.clone())
        .expect("Fix: initial state must be valid");
    let program = Program::wrapped(
        vec![BufferDecl::storage(
            "cache",
            0,
            BufferAccess::ReadWrite,
            DataType::F32,
        )],
        [1, 1, 1],
        Vec::new(),
    );
    let (_, outputs) = graph
        .add_node(
            "decode.0",
            program,
            vec![GraphInput {
                buffer: "cache".into(),
                value: state,
                contract: state_contract.clone(),
            }],
            vec![GraphOutput {
                buffer: "cache".into(),
                name: "cache.1".into(),
                contract: state_contract,
                state_successor_of: Some(state),
            }],
        )
        .expect("Fix: exact state successors must connect");
    assert_eq!(
        graph.values()[outputs[0].0 as usize].state_successor_of,
        Some(state)
    );
}

/// Prevents model layers from changing cache shape or lifetime across decode steps.
#[test]
fn incompatible_sequence_state_transition_fails_closed() {
    let mut graph = ProgramGraph::new();
    let state = graph
        .add_external_value(
            "cache.0",
            contract(BufferAccess::ReadWrite, ValueLifetime::SequenceState),
        )
        .expect("Fix: initial state must be valid");
    let program = Program::wrapped(
        vec![BufferDecl::storage(
            "cache",
            0,
            BufferAccess::ReadWrite,
            DataType::F32,
        )],
        [1, 1, 1],
        Vec::new(),
    );
    let error = graph
        .add_node(
            "decode.0",
            program,
            vec![GraphInput {
                buffer: "cache".into(),
                value: state,
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::SequenceState),
            }],
            vec![GraphOutput {
                buffer: "cache".into(),
                name: "cache.1".into(),
                contract: TensorContract {
                    shape: vec![ShapeDim::Known(9)],
                    ..contract(BufferAccess::ReadWrite, ValueLifetime::SequenceState)
                },
                state_successor_of: Some(state),
            }],
        )
        .expect_err("Fix: state shape changes must fail");
    assert_eq!(
        error,
        ProgramGraphError::InvalidStateTransition {
            output: "cache.1".into(),
            prior: state,
        }
    );
}

/// Prevents ambiguous graph identities from producing unstable fingerprints or wiring.
#[test]
fn duplicate_names_fail_closed() {
    let mut graph = ProgramGraph::new();
    graph
        .add_external_value(
            "shared",
            contract(BufferAccess::ReadOnly, ValueLifetime::ImmutableWeight),
        )
        .expect("Fix: first name must be accepted");
    assert_eq!(
        graph
            .add_external_value(
                "shared",
                contract(BufferAccess::ReadOnly, ValueLifetime::ImmutableWeight),
            )
            .expect_err("Fix: duplicate value names must fail"),
        ProgramGraphError::DuplicateName("shared".into())
    );
}

/// Locks failed node validation to a transaction so its node name remains reusable.
#[test]
fn failed_node_validation_does_not_reserve_node_identity() {
    let mut graph = ProgramGraph::new();
    let input = graph
        .add_external_value(
            "input",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
        )
        .expect("Fix: graph input must be valid");
    let before = graph.values().to_vec();
    graph
        .add_node(
            "layer",
            copy_program("input", "output"),
            vec![GraphInput {
                buffer: "missing".into(),
                value: input,
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
            }],
            Vec::new(),
        )
        .expect_err("Fix: missing buffer must fail");
    assert_eq!(graph.nodes().len(), 0);
    assert_eq!(graph.values(), before);

    let (node, outputs) = graph
        .add_node(
            "layer",
            copy_program("input", "output"),
            vec![GraphInput {
                buffer: "input".into(),
                value: input,
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "output".into(),
                name: "output".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Output),
                state_successor_of: None,
            }],
        )
        .expect("Fix: failed construction must leave its node name reusable");
    assert_eq!(node.0, 0);
    assert_eq!(outputs, [GraphValueId(1)]);
}

/// Prevents a late output-contract failure from adding values, consumers, or names.
#[test]
fn failed_output_validation_rolls_back_every_graph_collection() {
    let mut graph = ProgramGraph::new();
    let input = graph
        .add_external_value(
            "input",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
        )
        .expect("Fix: graph input must be valid");
    let before = graph.values().to_vec();
    let error = graph
        .add_node(
            "split",
            two_output_program("input", "first", "second"),
            vec![GraphInput {
                buffer: "input".into(),
                value: input,
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
            }],
            vec![
                GraphOutput {
                    buffer: "first".into(),
                    name: "first.value".into(),
                    contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation),
                    state_successor_of: None,
                },
                GraphOutput {
                    buffer: "second".into(),
                    name: "second.value".into(),
                    contract: TensorContract {
                        dtype: DataType::U32,
                        ..contract(BufferAccess::ReadWrite, ValueLifetime::Invocation)
                    },
                    state_successor_of: None,
                },
            ],
        )
        .expect_err("Fix: second output dtype mismatch must fail");
    assert!(matches!(error, ProgramGraphError::BufferContract { .. }));
    assert_eq!(graph.nodes().len(), 0);
    assert_eq!(graph.values(), before);
    assert!(graph.values()[input.0 as usize].consumers.is_empty());
    assert_eq!(
        graph
            .add_external_value(
                "first.value",
                contract(BufferAccess::ReadOnly, ValueLifetime::ImmutableWeight),
            )
            .expect("Fix: failed outputs must not reserve names"),
        GraphValueId(1)
    );
    graph
        .add_external_value(
            "split",
            contract(BufferAccess::ReadOnly, ValueLifetime::ImmutableWeight),
        )
        .expect("Fix: failed node must not reserve its name");
}

/// Prevents duplicate output names from partially committing the first output.
#[test]
fn duplicate_output_names_leave_graph_unchanged() {
    let mut graph = ProgramGraph::new();
    let input = graph
        .add_external_value(
            "input",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
        )
        .expect("Fix: graph input must be valid");
    let before = graph.values().to_vec();
    assert_eq!(
        graph
            .add_node(
                "split",
                two_output_program("input", "first", "second"),
                vec![GraphInput {
                    buffer: "input".into(),
                    value: input,
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
                }],
                vec![
                    GraphOutput {
                        buffer: "first".into(),
                        name: "duplicate".into(),
                        contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation,),
                        state_successor_of: None,
                    },
                    GraphOutput {
                        buffer: "second".into(),
                        name: "duplicate".into(),
                        contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation,),
                        state_successor_of: None,
                    },
                ],
            )
            .expect_err("Fix: duplicate outputs must fail before mutation"),
        ProgramGraphError::DuplicateName("duplicate".into())
    );
    assert_eq!(graph.nodes().len(), 0);
    assert_eq!(graph.values(), before);
    assert!(graph.values()[input.0 as usize].consumers.is_empty());
    assert_eq!(
        graph
            .add_external_value(
                "duplicate",
                contract(BufferAccess::ReadOnly, ValueLifetime::ImmutableWeight),
            )
            .expect("Fix: duplicate failed outputs must not reserve names"),
        GraphValueId(1)
    );
}

/// Proves a complete external parameter batch receives contiguous canonical identities.
#[test]
fn external_value_batch_commits_in_declaration_order() {
    let mut graph = ProgramGraph::new();
    let ids = graph
        .add_external_values(vec![
            (
                "weight.a".into(),
                contract(BufferAccess::ReadOnly, ValueLifetime::ImmutableWeight),
            ),
            (
                "weight.b".into(),
                contract(BufferAccess::ReadOnly, ValueLifetime::ImmutableWeight),
            ),
        ])
        .expect("Fix: unique external batch must register");
    assert_eq!(ids, [GraphValueId(0), GraphValueId(1)]);
    assert_eq!(
        graph
            .values()
            .iter()
            .map(|value| value.name.as_str())
            .collect::<Vec<_>>(),
        ["weight.a", "weight.b"]
    );
}

/// Prevents an intra-batch duplicate from committing any earlier parameter.
#[test]
fn duplicate_external_batch_name_rolls_back_all_values() {
    let mut graph = ProgramGraph::new();
    assert_eq!(
        graph
            .add_external_values(vec![
                (
                    "weight".into(),
                    contract(BufferAccess::ReadOnly, ValueLifetime::ImmutableWeight),
                ),
                (
                    "weight".into(),
                    contract(BufferAccess::ReadOnly, ValueLifetime::ImmutableWeight),
                ),
            ])
            .expect_err("Fix: duplicate batch name must fail"),
        ProgramGraphError::DuplicateName("weight".into())
    );
    assert!(graph.values().is_empty());
    assert_eq!(
        graph
            .add_external_value(
                "weight",
                contract(BufferAccess::ReadOnly, ValueLifetime::ImmutableWeight),
            )
            .expect("Fix: rejected batch must not reserve its first name"),
        GraphValueId(0)
    );
}

/// Prevents a late collision with existing graph state from partially extending the batch.
#[test]
fn existing_name_collision_rolls_back_external_batch() {
    let mut graph = ProgramGraph::new();
    graph
        .add_external_value(
            "existing",
            contract(BufferAccess::ReadOnly, ValueLifetime::ImmutableWeight),
        )
        .expect("Fix: existing fixture value must register");
    let before = graph.values().to_vec();
    assert_eq!(
        graph
            .add_external_values(vec![
                (
                    "new".into(),
                    contract(BufferAccess::ReadOnly, ValueLifetime::ImmutableWeight),
                ),
                (
                    "existing".into(),
                    contract(BufferAccess::ReadOnly, ValueLifetime::ImmutableWeight),
                ),
            ])
            .expect_err("Fix: existing name collision must reject the whole batch"),
        ProgramGraphError::DuplicateName("existing".into())
    );
    assert_eq!(graph.values(), before);
    assert_eq!(
        graph
            .add_external_value(
                "new",
                contract(BufferAccess::ReadOnly, ValueLifetime::ImmutableWeight),
            )
            .expect("Fix: rejected batch must leave earlier names reusable"),
        GraphValueId(1)
    );
}

/// Prevents a consumer from flattening or reshaping a connected value implicitly.
#[test]
fn consumer_rank_drift_fails_before_graph_mutation() {
    let producer_contract = TensorContract {
        dtype: DataType::F32,
        shape: vec![ShapeDim::Known(2), ShapeDim::Known(4)],
        access: BufferAccess::ReadOnly,
        lifetime: ValueLifetime::Invocation,
    };
    let mut graph = ProgramGraph::new();
    let value = graph
        .add_external_value("matrix", producer_contract.clone())
        .expect("Fix: matrix fixture must register");
    let before = graph.values().to_vec();
    let error = graph
        .add_node(
            "flatten",
            Program::wrapped(
                vec![
                    BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::F32)
                        .with_count(8),
                ],
                [1, 1, 1],
                Vec::new(),
            ),
            vec![GraphInput {
                buffer: "input".into(),
                value,
                contract: TensorContract {
                    shape: vec![ShapeDim::Known(8)],
                    ..producer_contract.clone()
                },
            }],
            Vec::new(),
        )
        .expect_err("Fix: implicit rank drift must fail");
    assert!(matches!(
        error,
        ProgramGraphError::InputContract {
            value: GraphValueId(0),
            ..
        }
    ));
    assert_eq!(graph.values(), before);
    assert!(graph.nodes().is_empty());
}

/// Prevents a statically sized graph port from binding a shorter Program buffer.
#[test]
fn static_shape_element_count_must_match_program_buffer() {
    let exact = TensorContract {
        dtype: DataType::F32,
        shape: vec![ShapeDim::Known(8)],
        access: BufferAccess::ReadOnly,
        lifetime: ValueLifetime::Invocation,
    };
    let mut graph = ProgramGraph::new();
    let value = graph
        .add_external_value("input", exact.clone())
        .expect("Fix: static fixture must register");
    let error = graph
        .add_node(
            "short",
            Program::wrapped(
                vec![
                    BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::F32)
                        .with_count(7),
                ],
                [1, 1, 1],
                Vec::new(),
            ),
            vec![GraphInput {
                buffer: "input".into(),
                value,
                contract: exact,
            }],
            Vec::new(),
        )
        .expect_err("Fix: short Program buffer must fail");
    assert!(matches!(error, ProgramGraphError::BufferContract { .. }));
    assert!(error
        .to_string()
        .contains("Program declares 7 elements, graph shape requires 8"));
}

/// Prevents one graph value from aliasing two Program input bindings implicitly.
#[test]
fn duplicate_input_value_alias_fails_closed() {
    let input_contract = contract(BufferAccess::ReadOnly, ValueLifetime::Invocation);
    let mut graph = ProgramGraph::new();
    let value = graph
        .add_external_value("shared", input_contract.clone())
        .expect("Fix: alias fixture must register");
    assert_eq!(
        graph
            .add_node(
                "alias",
                two_input_program("first", "second"),
                vec![
                    GraphInput {
                        buffer: "first".into(),
                        value,
                        contract: input_contract.clone(),
                    },
                    GraphInput {
                        buffer: "second".into(),
                        value,
                        contract: input_contract,
                    },
                ],
                Vec::new(),
            )
            .expect_err("Fix: implicit value alias must fail"),
        ProgramGraphError::DuplicateValueInput {
            node: "alias".into(),
            value,
        }
    );
}

/// Prevents a state output from replacing state that the node never consumed.
#[test]
fn dangling_state_successor_fails_closed() {
    let state_contract = contract(BufferAccess::ReadWrite, ValueLifetime::SequenceState);
    let mut graph = ProgramGraph::new();
    let consumed = graph
        .add_external_value("state.consumed", state_contract.clone())
        .expect("Fix: consumed state must register");
    let dangling = graph
        .add_external_value("state.dangling", state_contract.clone())
        .expect("Fix: dangling fixture state must register");
    assert_eq!(
        graph
            .add_node(
                "state.step",
                copy_program("input", "output"),
                vec![GraphInput {
                    buffer: "input".into(),
                    value: consumed,
                    contract: TensorContract {
                        access: BufferAccess::ReadOnly,
                        ..state_contract.clone()
                    },
                }],
                vec![GraphOutput {
                    buffer: "output".into(),
                    name: "state.next".into(),
                    contract: state_contract,
                    state_successor_of: Some(dangling),
                }],
            )
            .expect_err("Fix: unconsumed state successor must fail"),
        ProgramGraphError::DanglingStateTransition {
            output: "state.next".into(),
            prior: dangling,
        }
    );
}

/// Prevents a writable graph contract from binding a read-only Program input.
#[test]
fn program_access_must_satisfy_consumer_contract() {
    let writable = contract(BufferAccess::ReadWrite, ValueLifetime::Invocation);
    let mut graph = ProgramGraph::new();
    let value = graph
        .add_external_value("input", writable.clone())
        .expect("Fix: writable fixture must register");
    let error = graph
        .add_node(
            "write-required",
            copy_program("input", "output"),
            vec![GraphInput {
                buffer: "input".into(),
                value,
                contract: writable,
            }],
            Vec::new(),
        )
        .expect_err("Fix: read-only Program input must not satisfy writable contract");
    assert!(matches!(error, ProgramGraphError::BufferContract { .. }));
    assert!(error
        .to_string()
        .contains("Program access ReadOnly does not satisfy graph access ReadWrite"));
}

/// Prevents hostile static dimensions from overflowing shape arithmetic.
#[test]
fn static_shape_product_overflow_fails_closed() {
    let overflowing = TensorContract {
        dtype: DataType::F32,
        shape: vec![ShapeDim::Known(u64::MAX), ShapeDim::Known(2)],
        access: BufferAccess::ReadOnly,
        lifetime: ValueLifetime::Invocation,
    };
    let mut graph = ProgramGraph::new();
    let value = graph
        .add_external_value("input", overflowing.clone())
        .expect("Fix: metadata registration remains allocation-free");
    let error = graph
        .add_node(
            "overflow",
            copy_program("input", "output"),
            vec![GraphInput {
                buffer: "input".into(),
                value,
                contract: overflowing,
            }],
            Vec::new(),
        )
        .expect_err("Fix: overflowing shape must fail before scheduling");
    assert!(matches!(error, ProgramGraphError::BufferContract { .. }));
    assert!(error.to_string().contains("overflows u64"));
}

/// Proves symbolic and concrete contracts survive canonical graph wire round-trip.
#[test]
fn graph_wire_round_trip_preserves_typed_topology_and_programs() {
    let graph = stateful_wire_graph();
    let bytes = graph.to_wire().expect("Fix: valid graph must encode");
    let decoded = ProgramGraph::from_wire(&bytes).expect("Fix: valid graph wire must decode");
    assert_eq!(
        decoded
            .to_wire()
            .expect("Fix: decoded graph must re-encode canonically"),
        bytes
    );
    assert_eq!(decoded.values().len(), 3);
    assert_eq!(decoded.nodes().len(), 1);
    assert_eq!(
        decoded.values()[0].contract.shape,
        [ShapeDim::Known(2), ShapeDim::Known(4)]
    );
    assert_eq!(
        decoded.values()[1].contract.shape,
        [ShapeDim::Symbol("batch".into()), ShapeDim::Known(8)]
    );
    assert_eq!(decoded.nodes()[0].output_ports[0].buffer, "cache");
    assert_eq!(
        decoded.nodes()[0].output_ports[0].state_successor_of,
        Some(GraphValueId(1))
    );
}

/// Locks canonical graph bytes to graph content rather than allocation history.
#[test]
fn independently_built_equal_graphs_have_identical_wire_bytes() {
    assert_eq!(
        stateful_wire_graph()
            .to_wire()
            .expect("Fix: first graph must encode"),
        stateful_wire_graph()
            .to_wire()
            .expect("Fix: second graph must encode")
    );
}

/// Prevents unknown framing, truncation, and trailing bytes from being accepted.
#[test]
fn malformed_graph_wire_frames_fail_closed() {
    let bytes = stateful_wire_graph()
        .to_wire()
        .expect("Fix: wire fixture must encode");
    let mut bad_magic = bytes.clone();
    bad_magic[0] = b'X';
    assert!(ProgramGraph::from_wire(&bad_magic)
        .expect_err("Fix: bad magic must fail")
        .to_string()
        .contains("magic mismatch"));

    let mut bad_version = bytes.clone();
    bad_version[4..6].copy_from_slice(&2_u16.to_le_bytes());
    assert!(ProgramGraph::from_wire(&bad_version)
        .expect_err("Fix: unknown version must fail")
        .to_string()
        .contains("unsupported graph wire version 2"));

    assert!(ProgramGraph::from_wire(&bytes[..bytes.len() - 1])
        .expect_err("Fix: truncated state identity must fail")
        .to_string()
        .contains("truncated graph wire input"));

    let mut trailing = bytes;
    trailing.push(0);
    assert!(ProgramGraph::from_wire(&trailing)
        .expect_err("Fix: trailing bytes must fail")
        .to_string()
        .contains("trailing bytes"));
}

/// Prevents hostile count fields from reserving unbounded graph collections.
#[test]
fn oversized_graph_wire_counts_fail_before_allocation() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"VGR0");
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    let error = ProgramGraph::from_wire(&bytes)
        .expect_err("Fix: hostile external count must fail before allocation");
    assert!(error
        .to_string()
        .contains("external value count is 4294967295; maximum is 1000000"));
}

/// Prevents wire data from introducing a state edge to a nonexistent value.
#[test]
fn graph_wire_dangling_state_identity_fails_validation() {
    let mut bytes = stateful_wire_graph()
        .to_wire()
        .expect("Fix: wire fixture must encode");
    let prior_start = bytes.len() - 4;
    bytes[prior_start..].copy_from_slice(&99_u32.to_le_bytes());
    assert_eq!(
        ProgramGraph::from_wire(&bytes).expect_err("Fix: nonexistent state identity must fail"),
        ProgramGraphError::MissingValue(GraphValueId(99))
    );
}
