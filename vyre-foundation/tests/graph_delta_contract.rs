//! Contract and regression tests for canonical transactional [`GraphDelta`].
//!
//! Interactive applications modify localized graph subtrees. These tests prove:
//! 1. Atomic operations (insert, replace, delete, shape bounds, resource generation, state transitions).
//! 2. Transactional safety (aborted mutations leave base graph completely untouched).
//! 3. Transitive dependency derivation (dirty vs unchanged nodes/values).
//! 4. Versioned wire serialization and rejection of stale or corrupted versions.
//! 5. Scale closure bounding (mutating one node only dirties its downstream closure).

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, GraphDelta, GraphDeltaError, GraphDeltaOp, GraphInput,
    GraphNodeId, GraphOutput, GraphValueId, Program, ProgramGraph, ShapeDim, ValueContract,
    ValueLifetime, GRAPH_DELTA_VERSION,
};
use vyre_test_support::monorepo::vyre_workspace_root;
use vyre_test_support::{braced_body, read_source_file_bounded, top_level_variant_names};

fn tensor(
    dtype: DataType,
    shape: Vec<ShapeDim>,
    access: BufferAccess,
    lifetime: ValueLifetime,
) -> ValueContract {
    ValueContract {
        dtype,
        shape,
        access,
        lifetime,
    }
}

/// A single-input single-output program.
///
/// `Program::wrapped` carries no name, so the workgroup width is what makes two
/// otherwise identical programs distinguishable. A replacement test needs that:
/// installing a program equal to the one already there proves nothing.
fn make_unary_node(in_name: &str, out_name: &str) -> Program {
    make_unary_node_sized(in_name, out_name, 1)
}

fn make_unary_node_sized(in_name: &str, out_name: &str, workgroup_x: u32) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(in_name, 0, BufferAccess::ReadOnly, DataType::F32),
            BufferDecl::storage(out_name, 1, BufferAccess::ReadWrite, DataType::F32),
        ],
        [workgroup_x, 1, 1],
        Vec::new(),
    )
}

fn build_pipeline_graph() -> (ProgramGraph, GraphValueId, GraphNodeId, GraphNodeId) {
    let mut graph = ProgramGraph::new();
    let input = graph
        .add_external_value(
            "input_pixels",
            tensor(
                DataType::F32,
                vec![ShapeDim::Symbol("width".into()), ShapeDim::Known(1080)],
                BufferAccess::ReadOnly,
                ValueLifetime::Invocation,
            ),
        )
        .expect("Fix: external input must register");

    let (node1, out1) = graph
        .add_node(
            "blur_stage",
            make_unary_node("blur.in", "blur.out"),
            vec![GraphInput {
                buffer: "blur.in".into(),
                value: input,
                contract: tensor(
                    DataType::F32,
                    vec![ShapeDim::Symbol("width".into()), ShapeDim::Known(1080)],
                    BufferAccess::ReadOnly,
                    ValueLifetime::Invocation,
                ),
            }],
            vec![GraphOutput {
                buffer: "blur.out".into(),
                name: "blur_output".into(),
                contract: tensor(
                    DataType::F32,
                    vec![ShapeDim::Symbol("width".into()), ShapeDim::Known(1080)],
                    BufferAccess::ReadWrite,
                    ValueLifetime::Invocation,
                ),
                retained_successor_of: None,
            }],
        )
        .expect("Fix: node 1 must connect");

    let (node2, _) = graph
        .add_node(
            "composite_stage",
            make_unary_node("composite.in", "composite.out"),
            vec![GraphInput {
                buffer: "composite.in".into(),
                value: out1[0],
                contract: tensor(
                    DataType::F32,
                    vec![ShapeDim::Symbol("width".into()), ShapeDim::Known(1080)],
                    BufferAccess::ReadOnly,
                    ValueLifetime::Invocation,
                ),
            }],
            vec![GraphOutput {
                buffer: "composite.out".into(),
                name: "final_output".into(),
                contract: tensor(
                    DataType::F32,
                    vec![ShapeDim::Symbol("width".into()), ShapeDim::Known(1080)],
                    BufferAccess::ReadWrite,
                    ValueLifetime::Output,
                ),
                retained_successor_of: None,
            }],
        )
        .expect("Fix: node 2 must connect");

    (graph, input, node1, node2)
}

#[test]
fn graph_delta_empty_fails_validation() {
    let (graph, _, _, _) = build_pipeline_graph();
    let empty_delta = GraphDelta::new();
    assert!(empty_delta.is_empty());
    assert_eq!(empty_delta.len(), 0);
    assert_eq!(empty_delta.version(), GRAPH_DELTA_VERSION);

    let err = empty_delta
        .validate(&graph)
        .expect_err("Fix: empty delta must fail validation");
    assert!(matches!(err, GraphDeltaError::EmptyDelta));
}

#[test]
fn graph_delta_insert_external_value_and_node() {
    let (graph, _, _, _) = build_pipeline_graph();
    let initial_node_count = graph.nodes().len();
    let initial_value_count = graph.values().len();

    let mut delta = GraphDelta::new();
    delta.push(GraphDeltaOp::InsertExternalValue {
        name: "overlay_alpha".into(),
        contract: tensor(
            DataType::F32,
            vec![ShapeDim::Known(1080)],
            BufferAccess::ReadOnly,
            ValueLifetime::Constant,
        ),
    });

    let (new_graph, closure) = delta
        .apply_transactional(&graph)
        .expect("Fix: insert value delta must apply");

    assert_eq!(new_graph.values().len(), initial_value_count + 1);
    assert_eq!(new_graph.nodes().len(), initial_node_count);
    assert_eq!(closure.dirty_values.len(), 1);
    assert!(closure.affected_resource_names.contains("overlay_alpha"));
    assert!(!closure.is_pure_generation_bump);
}

#[test]
fn graph_delta_replace_node_propagates_dirty_closure() {
    let (graph, input, node1, node2) = build_pipeline_graph();

    let mut delta = GraphDelta::new();
    delta.push(GraphDeltaOp::ReplaceNode {
        node_id: node1,
        program: make_unary_node_sized("blur.in", "blur.out", 64),
        inputs: vec![GraphInput {
            buffer: "blur.in".into(),
            value: input,
            contract: tensor(
                DataType::F32,
                vec![ShapeDim::Symbol("width".into()), ShapeDim::Known(1080)],
                BufferAccess::ReadOnly,
                ValueLifetime::Invocation,
            ),
        }],
        outputs: vec![GraphOutput {
            buffer: "blur.out".into(),
            name: "blur_output".into(),
            contract: tensor(
                DataType::F32,
                vec![ShapeDim::Symbol("width".into()), ShapeDim::Known(1080)],
                BufferAccess::ReadWrite,
                ValueLifetime::Invocation,
            ),
            retained_successor_of: None,
        }],
    });

    let (mutated_graph, closure) = delta
        .apply_transactional(&graph)
        .expect("Fix: replace node must succeed");

    // Both node1 and downstream node2 must be dirty
    assert!(closure.dirty_nodes.contains(&node1));
    assert!(closure.dirty_nodes.contains(&node2));
    assert_eq!(closure.unchanged_nodes.len(), 0);
    assert_eq!(mutated_graph.nodes().len(), graph.nodes().len());
}

#[test]
fn graph_delta_delete_node_with_dependents_fails_transactional() {
    let (graph, _, node1, node2) = build_pipeline_graph();

    let mut delta = GraphDelta::new();
    delta.push(GraphDeltaOp::DeleteNode { node_id: node1 });

    // Node 1 cannot be deleted while Node 2 consumes its output
    let err = delta
        .apply_transactional(&graph)
        .expect_err("Fix: deleting node with active consumers must fail");

    assert!(
        matches!(
            err,
            GraphDeltaError::DependencyViolation { node, dependent }
                if node == node1 && dependent == node2
        ),
        "the refusal must name both the node and the consumer that blocks it, got {err}"
    );

    // Base graph is completely untouched
    assert_eq!(graph.nodes().len(), 2);
}

#[test]
fn graph_delta_shape_bound_update_identifies_affected_closure() {
    let (graph, input, node1, node2) = build_pipeline_graph();

    let mut delta = GraphDelta::new();
    delta.push(GraphDeltaOp::UpdateShapeBound {
        symbol: "width".into(),
        old_bound: 1920,
        new_bound: 2560,
    });

    let (_, closure) = delta
        .apply_transactional(&graph)
        .expect("Fix: shape bound update must succeed");

    assert!(closure.is_pure_shape_update);
    assert!(!closure.is_pure_generation_bump);
    assert!(closure.dirty_values.contains(&input));
    assert!(closure.dirty_nodes.contains(&node1));
    assert!(closure.dirty_nodes.contains(&node2));
    assert!(closure.affected_resource_names.contains("width"));
}

#[test]
fn graph_delta_resource_generation_bump_tracks_resource() {
    let (graph, _, _, _) = build_pipeline_graph();

    let mut delta = GraphDelta::new();
    delta.push(GraphDeltaOp::UpdateResourceGeneration {
        resource_name: "retained_atlas".into(),
        prior_generation: 4,
        new_generation: 5,
    });

    let (_, closure) = delta
        .apply_transactional(&graph)
        .expect("Fix: valid generation bump must succeed");

    assert!(closure.is_pure_generation_bump);
    assert!(!closure.is_pure_shape_update);
    assert!(closure.affected_resource_names.contains("retained_atlas"));

    // Stale generation bump (not strictly greater) fails
    let stale_delta = GraphDelta::new().with_op(GraphDeltaOp::UpdateResourceGeneration {
        resource_name: "retained_atlas".into(),
        prior_generation: 4,
        new_generation: 4,
    });
    assert!(matches!(
        stale_delta.apply_transactional(&graph),
        Err(GraphDeltaError::ResourceGenerationMismatch { .. })
    ));
}

#[test]
fn graph_delta_wire_roundtrip_preserves_semantics() {
    let mut delta = GraphDelta::new();
    delta.push(GraphDeltaOp::InsertExternalValue {
        name: "glyph_cache".into(),
        contract: tensor(
            DataType::U8,
            vec![ShapeDim::Known(4096), ShapeDim::Known(4096)],
            BufferAccess::ReadOnly,
            ValueLifetime::Constant,
        ),
    });
    delta.push(GraphDeltaOp::UpdateShapeBound {
        symbol: "viewport_w".into(),
        old_bound: 1080,
        new_bound: 1440,
    });
    delta.push(GraphDeltaOp::UpdateResourceGeneration {
        resource_name: "glyph_cache".into(),
        prior_generation: 1,
        new_generation: 2,
    });

    let wire_bytes = delta
        .to_wire()
        .expect("Fix: delta wire encode must succeed");
    let decoded = GraphDelta::from_wire(&wire_bytes).expect("Fix: delta wire decode must succeed");

    assert_eq!(delta, decoded);
    assert_eq!(decoded.version(), GRAPH_DELTA_VERSION);
    assert_eq!(decoded.len(), 3);
}

#[test]
fn stale_or_corrupt_graph_delta_version_is_rejected() {
    let mut delta = GraphDelta::new();
    delta.push(GraphDeltaOp::UpdateResourceGeneration {
        resource_name: "font_atlas".into(),
        prior_generation: 1,
        new_generation: 2,
    });
    let mut wire_bytes = delta
        .to_wire()
        .expect("Fix: delta wire encode must succeed");

    // Corrupt the version field (bytes 4..6 in little-endian)
    wire_bytes[4] = 99;
    wire_bytes[5] = 0;

    let err = GraphDelta::from_wire(&wire_bytes).expect_err("Fix: stale version must be rejected");
    assert!(matches!(
        err,
        GraphDeltaError::VersionMismatch {
            expected: 1,
            found: 99
        }
    ));

    // Corrupt the magic header
    wire_bytes[0] = b'X';
    let err_magic =
        GraphDelta::from_wire(&wire_bytes).expect_err("Fix: corrupt magic must be rejected");
    assert!(matches!(err_magic, GraphDeltaError::Wire(_)));
}

/// Prevents hostile operation count fields from allocating unbounded memory.
#[test]
fn oversized_graph_delta_operation_count_fails_before_allocation() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"VGD0");
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    let err = GraphDelta::from_wire(&bytes)
        .expect_err("Fix: hostile delta operation count must fail before allocation");
    let msg = match err {
        GraphDeltaError::Wire(m) => m,
        other => panic!("expected Wire error, got {other:?}"),
    };
    assert!(
        msg.contains("operation count is 4294967295; maximum is 1000000"),
        "got: {msg}"
    );
}

/// Prevents hostile string length fields from triggering huge allocations.
#[test]
fn oversized_graph_delta_string_length_fails() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"VGD0");
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&1_u32.to_le_bytes()); // 1 op
    bytes.push(1); // InsertExternalValue
    bytes.extend_from_slice(&100_000_u32.to_le_bytes()); // string len exceeds MAX_NAME_BYTES (4096)
    let err = GraphDelta::from_wire(&bytes)
        .expect_err("Fix: hostile string length must be rejected");
    let msg = match err {
        GraphDeltaError::Wire(m) => m,
        other => panic!("expected Wire error, got {other:?}"),
    };
    assert!(msg.contains("string length 100000 exceeds limit 4096"), "got: {msg}");
}

/// Prevents hostile port counts in graph delta operations from allocating unbounded memory.
#[test]
fn oversized_graph_delta_port_count_fails() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"VGD0");
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&1_u32.to_le_bytes()); // 1 op
    bytes.push(2); // InsertNode
    bytes.extend_from_slice(&4_u32.to_le_bytes()); // name len
    bytes.extend_from_slice(b"node");
    let prog = vyre_foundation::ir::Program::wrapped(vec![], [1, 1, 1], vec![]);
    let prog_bytes = prog.to_wire().expect("program wire");
    bytes.extend_from_slice(&(prog_bytes.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&prog_bytes);
    bytes.extend_from_slice(&2_000_000_u32.to_le_bytes()); // in_count exceeds MAX_PORTS_PER_NODE
    let err = GraphDelta::from_wire(&bytes)
        .expect_err("Fix: hostile port count must be rejected");
    let msg = match err {
        GraphDeltaError::Wire(m) => m,
        other => panic!("expected Wire error, got {other:?}"),
    };
    assert!(msg.contains("input port count is 2000000; maximum is 1000000"), "got: {msg}");
}

#[test]
fn scale_closure_bounding_one_node_in_large_graph() {
    let mut graph = ProgramGraph::new();
    let mut prev_val = graph
        .add_external_value(
            "root_input",
            tensor(
                DataType::F32,
                vec![ShapeDim::Known(64)],
                BufferAccess::ReadOnly,
                ValueLifetime::Invocation,
            ),
        )
        .expect("Fix: root input must register");

    let num_nodes = 20;
    let mut node_ids = Vec::new();

    for i in 0..num_nodes {
        let node_name = format!("stage_{i}");
        let in_buf = format!("in_{i}");
        let out_buf = format!("out_{i}");
        let val_name = format!("val_{i}");

        let (nid, out_vids) = graph
            .add_node(
                &node_name,
                make_unary_node(&in_buf, &out_buf),
                vec![GraphInput {
                    buffer: in_buf,
                    value: prev_val,
                    contract: tensor(
                        DataType::F32,
                        vec![ShapeDim::Known(64)],
                        BufferAccess::ReadOnly,
                        ValueLifetime::Invocation,
                    ),
                }],
                vec![GraphOutput {
                    buffer: out_buf,
                    name: val_name,
                    contract: tensor(
                        DataType::F32,
                        vec![ShapeDim::Known(64)],
                        BufferAccess::ReadWrite,
                        if i == num_nodes - 1 {
                            ValueLifetime::Output
                        } else {
                            ValueLifetime::Invocation
                        },
                    ),
                    retained_successor_of: None,
                }],
            )
            .expect("Fix: sequential node must connect");

        node_ids.push(nid);
        prev_val = out_vids[0];
    }

    // Now mutate only node 15 in a 20-node sequential chain
    let target_node = node_ids[15];
    let target_in = graph.nodes()[15].inputs[0].value;

    let mut delta = GraphDelta::new();
    delta.push(GraphDeltaOp::ReplaceNode {
        node_id: target_node,
        program: make_unary_node_sized("in_15", "out_15", 64),
        inputs: vec![GraphInput {
            buffer: "in_15".into(),
            value: target_in,
            contract: tensor(
                DataType::F32,
                vec![ShapeDim::Known(64)],
                BufferAccess::ReadOnly,
                ValueLifetime::Invocation,
            ),
        }],
        outputs: vec![GraphOutput {
            buffer: "out_15".into(),
            name: "val_15".into(),
            contract: tensor(
                DataType::F32,
                vec![ShapeDim::Known(64)],
                BufferAccess::ReadWrite,
                ValueLifetime::Invocation,
            ),
            retained_successor_of: None,
        }],
    });

    let (_, closure) = delta
        .apply_transactional(&graph)
        .expect("Fix: mutation must apply");

    // Nodes 0..14 MUST be unchanged
    for i in 0..15 {
        assert!(
            closure.unchanged_nodes.contains(&node_ids[i]),
            "node {i} before mutation point must remain unchanged"
        );
        assert!(
            !closure.dirty_nodes.contains(&node_ids[i]),
            "node {i} must not be dirty"
        );
    }

    // Nodes 15..19 MUST be dirty
    for i in 15..num_nodes {
        assert!(
            closure.dirty_nodes.contains(&node_ids[i]),
            "downstream node {i} must be dirty"
        );
    }
}

#[test]
fn replacing_a_node_installs_the_new_program() {
    let (graph, input, node1, _) = build_pipeline_graph();
    let replacement = make_unary_node_sized("blur.in", "blur.out", 64);
    assert_ne!(
        replacement,
        graph.nodes()[node1.0 as usize].program,
        "the replacement program must differ from the installed one, or this test proves nothing"
    );

    let ports = graph.nodes()[node1.0 as usize].output_ports.clone();
    let delta = GraphDelta::new().with_op(GraphDeltaOp::ReplaceNode {
        node_id: node1,
        program: replacement.clone(),
        inputs: vec![GraphInput {
            buffer: "blur.in".into(),
            value: input,
            contract: tensor(
                DataType::F32,
                vec![ShapeDim::Symbol("width".into()), ShapeDim::Known(1080)],
                BufferAccess::ReadOnly,
                ValueLifetime::Invocation,
            ),
        }],
        outputs: ports,
    });

    let (mutated, _) = delta
        .apply_transactional(&graph)
        .expect("Fix: a replacement matching the node's ports must apply");

    assert_eq!(
        mutated.nodes()[node1.0 as usize].program,
        replacement,
        "the replaced node must execute the new program"
    );
    assert_ne!(
        mutated.nodes()[node1.0 as usize].program,
        graph.nodes()[node1.0 as usize].program,
        "the base graph's program must not survive the replacement"
    );
    assert_eq!(
        mutated.nodes()[node1.0 as usize].outputs,
        graph.nodes()[node1.0 as usize].outputs,
        "output value ids must survive so consumers stay connected"
    );
}

#[test]
fn replacing_a_node_rewires_the_values_it_reads() {
    let (graph, input, node1, node2) = build_pipeline_graph();
    let blur_output = graph.nodes()[node1.0 as usize].outputs[0];
    assert!(
        graph.values()[blur_output.0 as usize]
            .consumers
            .contains(&node2),
        "the base graph must connect node2 to node1's output"
    );

    // Repoint node2 at the external input instead of node1's output. Both
    // values carry the same contract, so only the edge changes.
    let ports = graph.nodes()[node2.0 as usize].output_ports.clone();
    let delta = GraphDelta::new().with_op(GraphDeltaOp::ReplaceNode {
        node_id: node2,
        program: make_unary_node("composite.in", "composite.out"),
        inputs: vec![GraphInput {
            buffer: "composite.in".into(),
            value: input,
            contract: tensor(
                DataType::F32,
                vec![ShapeDim::Symbol("width".into()), ShapeDim::Known(1080)],
                BufferAccess::ReadOnly,
                ValueLifetime::Invocation,
            ),
        }],
        outputs: ports,
    });

    let (mutated, _) = delta
        .apply_transactional(&graph)
        .expect("Fix: repointing an input to an equally typed value must apply");

    assert!(
        !mutated.values()[blur_output.0 as usize]
            .consumers
            .contains(&node2),
        "a value the node no longer reads must drop it as a consumer"
    );
    assert!(
        mutated.values()[input.0 as usize]
            .consumers
            .contains(&node2),
        "a value the node now reads must record it as a consumer"
    );
    assert_eq!(
        mutated.nodes()[node2.0 as usize].inputs[0].value,
        input,
        "the node must read the value the replacement named"
    );
}

#[test]
fn a_replacement_that_changes_output_ports_is_refused() {
    let (graph, input, node1, _) = build_pipeline_graph();

    // Consumers read every field of the output port, so each field is refused
    // on its own rather than only the buffer name.
    let base = graph.nodes()[node1.0 as usize].output_ports[0].clone();
    let mut renamed = base.clone();
    renamed.name = "blur_output_v2".into();
    let mut retyped = base.clone();
    retyped.contract = tensor(
        DataType::F16,
        vec![ShapeDim::Symbol("width".into()), ShapeDim::Known(1080)],
        BufferAccess::ReadWrite,
        ValueLifetime::Invocation,
    );
    let mut reshaped = base.clone();
    reshaped.contract.shape = vec![ShapeDim::Known(64)];

    for (label, outputs) in [
        ("renamed output value", vec![renamed]),
        ("changed output dtype", vec![retyped]),
        ("changed output shape", vec![reshaped]),
        ("dropped the output entirely", Vec::new()),
        ("added a second output", vec![base.clone(), base.clone()]),
    ] {
        let delta = GraphDelta::new().with_op(GraphDeltaOp::ReplaceNode {
            node_id: node1,
            program: make_unary_node_sized("blur.in", "blur.out", 64),
            inputs: vec![GraphInput {
                buffer: "blur.in".into(),
                value: input,
                contract: tensor(
                    DataType::F32,
                    vec![ShapeDim::Symbol("width".into()), ShapeDim::Known(1080)],
                    BufferAccess::ReadOnly,
                    ValueLifetime::Invocation,
                ),
            }],
            outputs,
        });

        let err = delta
            .apply_transactional(&graph)
            .expect_err(&format!("Fix: a replacement that {label} must be refused"));
        assert!(
            matches!(&err, GraphDeltaError::Wire(reason) if reason.contains("different output ports")),
            "a replacement that {label} must be refused as an output-port change, got {err}"
        );
    }
}

#[test]
fn a_refused_replacement_leaves_the_node_untouched() {
    let (graph, input, node1, _) = build_pipeline_graph();

    // The port names a buffer the replacement program does not declare, so
    // validation fails after the graph has already been cloned.
    let ports = graph.nodes()[node1.0 as usize].output_ports.clone();
    let delta = GraphDelta::new().with_op(GraphDeltaOp::ReplaceNode {
        node_id: node1,
        program: make_unary_node_sized("other.in", "blur.out", 64),
        inputs: vec![GraphInput {
            buffer: "blur.in".into(),
            value: input,
            contract: tensor(
                DataType::F32,
                vec![ShapeDim::Symbol("width".into()), ShapeDim::Known(1080)],
                BufferAccess::ReadOnly,
                ValueLifetime::Invocation,
            ),
        }],
        outputs: ports,
    });

    assert!(
        delta.apply_transactional(&graph).is_err(),
        "a port naming an undeclared buffer must be refused"
    );
    assert_eq!(
        graph.nodes()[node1.0 as usize].program,
        make_unary_node("blur.in", "blur.out"),
        "the base graph must be untouched by a refused replacement"
    );
}

#[test]
fn a_shape_bound_update_that_states_no_change_is_refused() {
    let (graph, _, _, _) = build_pipeline_graph();

    for (label, old_bound, new_bound) in [
        ("does not move the bound", 1920, 1920),
        ("states a zero extent", 1920, 0),
        ("states zero for both", 0, 0),
    ] {
        let delta = GraphDelta::new().with_op(GraphDeltaOp::UpdateShapeBound {
            symbol: "width".into(),
            old_bound,
            new_bound,
        });
        let err = delta
            .apply_transactional(&graph)
            .expect_err(&format!("Fix: a bound update that {label} must be refused"));
        assert!(
            matches!(err, GraphDeltaError::IllegalShapeBound { .. }),
            "a bound update that {label} must be refused as illegal, got {err}"
        );
    }
}

#[test]
fn a_shape_bound_update_naming_an_undeclared_symbol_is_refused() {
    let (graph, _, _, _) = build_pipeline_graph();

    let delta = GraphDelta::new().with_op(GraphDeltaOp::UpdateShapeBound {
        symbol: "height".into(),
        old_bound: 1080,
        new_bound: 1440,
    });

    let err = delta
        .apply_transactional(&graph)
        .expect_err("Fix: a symbol no value declares must be refused");
    assert!(
        matches!(&err, GraphDeltaError::UnknownShapeSymbol { symbol } if symbol == "height"),
        "the refusal must name the undeclared symbol, got {err}"
    );
}

#[test]
fn every_buffer_access_survives_a_delta_wire_round_trip() {
    // The encoder wrote its own access table while the decoder read the
    // canonical one, so `ReadOnly` went out as tag 1 and came back
    // `ReadWrite`.
    //
    // `BufferAccess` is non-exhaustive, so the case list is read from the enum
    // declaration at run time. Adding a variant turns this red until it is
    // wired into the round trip rather than passing on a stale list.
    let cases: Vec<(&str, BufferAccess)> = vec![
        ("ReadOnly", BufferAccess::ReadOnly),
        ("ReadWrite", BufferAccess::ReadWrite),
        ("Uniform", BufferAccess::Uniform),
        ("Workgroup", BufferAccess::Workgroup),
        ("WriteOnly", BufferAccess::WriteOnly),
    ];

    let source =
        read_source_file_bounded(&vyre_workspace_root().join("vyre-spec/src/buffer_access.rs"))
            .expect("Fix: the BufferAccess declaration must be readable");
    let declared = top_level_variant_names(
        braced_body(&source, "pub enum BufferAccess {")
            .expect("Fix: `pub enum BufferAccess {` must be present in vyre-spec"),
    );
    let covered: BTreeSet<String> = cases.iter().map(|(name, _)| (*name).to_string()).collect();
    assert_eq!(
        declared,
        covered,
        "every declared BufferAccess must be round-tripped; missing {:?}, unknown {:?}",
        declared.difference(&covered).collect::<Vec<_>>(),
        covered.difference(&declared).collect::<Vec<_>>()
    );

    for (name, access) in cases {
        let delta = GraphDelta::new().with_op(GraphDeltaOp::InsertExternalValue {
            name: "probe".into(),
            contract: tensor(
                DataType::U32,
                vec![ShapeDim::Known(8), ShapeDim::Symbol("n".into())],
                access,
                ValueLifetime::Constant,
            ),
        });

        let bytes = delta
            .to_wire()
            .expect("Fix: delta wire encode must succeed");
        let decoded = GraphDelta::from_wire(&bytes).expect("Fix: delta wire decode must succeed");
        assert_eq!(
            decoded, delta,
            "access {name} must decode to the access that was encoded"
        );
    }
}
