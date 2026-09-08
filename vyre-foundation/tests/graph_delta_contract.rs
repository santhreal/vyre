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
    AffectedGraphClosure, BufferAccess, BufferDecl, DataType, GraphDelta, GraphDeltaError,
    GraphDeltaOp, GraphInput, GraphNodeId, GraphOutput, GraphValueId, Program, ProgramGraph,
    ShapeDim, ValueContract, ValueLifetime, GRAPH_DELTA_VERSION,
};

fn tensor(dtype: DataType, shape: Vec<ShapeDim>, access: BufferAccess, lifetime: ValueLifetime) -> ValueContract {
    ValueContract {
        dtype,
        shape,
        access,
        lifetime,
    }
}

fn make_unary_node(name: &str, in_name: &str, out_name: &str) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(in_name, 0, BufferAccess::ReadOnly, DataType::F32),
            BufferDecl::storage(out_name, 1, BufferAccess::ReadWrite, DataType::F32),
        ],
        [1, 1, 1],
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
            make_unary_node("blur", "blur.in", "blur.out"),
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
            make_unary_node("composite", "composite.in", "composite.out"),
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
        program: make_unary_node("blur_fast", "blur.in", "blur.out"),
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
    let (graph, _, node1, _) = build_pipeline_graph();

    let mut delta = GraphDelta::new();
    delta.push(GraphDeltaOp::DeleteNode { node_id: node1 });

    // Node 1 cannot be deleted while Node 2 consumes its output
    let err = delta
        .apply_transactional(&graph)
        .expect_err("Fix: deleting node with active consumers must fail");

    assert!(matches!(
        err,
        GraphDeltaError::DependencyViolation { node, dependent } if node == node1
    ));

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

    let wire_bytes = delta.to_wire().expect("Fix: delta wire encode must succeed");
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
    let mut wire_bytes = delta.to_wire().expect("Fix: delta wire encode must succeed");

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
    let err_magic = GraphDelta::from_wire(&wire_bytes).expect_err("Fix: corrupt magic must be rejected");
    assert!(matches!(err_magic, GraphDeltaError::Wire(_)));
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
                make_unary_node(&node_name, &in_buf, &out_buf),
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
        program: make_unary_node("stage_15_fast", "in_15", "out_15"),
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
