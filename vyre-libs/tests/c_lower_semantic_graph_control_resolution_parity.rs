//! Differential parity for the two registry-closure orphan lowering builders
//! `c_lower_ast_to_pg_semantic_graph_with_pg` and its `_no_control_resolution` twin. Both
//! delegate to `c_lower_ast_to_pg_semantic_graph_impl`, differing ONLY in the
//! `resolve_control_edges` flag (semantic_resolution_nodes vs unresolved_control_edge_slots).
//! Their documented contract: identical output when the token stream has NO control-flow
//! target constructs (goto/switch/case/default). We pin that with a control-flow-free VAST 
//! the two builders' three output buffers (plain PG nodes, semantic PG nodes, PG edges) must be
//! byte-identical through `reference_eval`, AND non-trivial (real PG rows + structural edges).
//!
//! Drains the vyre-libs slice of BACKLOG.md WIRING-tautology-closure-25crates.
#![cfg(feature = "c-parser")]
#![forbid(unsafe_code)]

use vyre::ir::Expr;
use vyre_libs::parsing::c::lower::ast_to_pg_nodes::{
    c_lower_ast_to_pg_semantic_graph_with_pg,
    c_lower_ast_to_pg_semantic_graph_with_pg_no_control_resolution,
};
use vyre_reference::value::Value;

// VAST wire layout consumed by the lowering pass (mirrors the private IDX_* consts):
// field 0=kind, 1=parent, 2=first_child, 3=next_sibling, 5=src_byte_off, 6=src_byte_len.
const VAST_STRIDE: usize = 10;
const SENTINEL: u32 = u32::MAX;

// PG output strides (mirror parse/c/lower/ast_to_pg_nodes/mod.rs).
const PLAIN_PG_STRIDE: usize = 6; // PG_NODE_STRIDE_U32
const SEMANTIC_PG_STRIDE: usize = 10; // C_AST_PG_SEMANTIC_NODE_STRIDE_U32
const PG_EDGE_STRIDE: usize = 6; // C_AST_PG_EDGE_STRIDE_U32
const PG_EDGE_ROWS_PER_NODE: usize = 5; // C_AST_PG_EDGE_ROWS_PER_NODE

fn pack(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn unpack(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Lower `nodes` with the given builder; returns (plain_pg, semantic_pg, pg_edges) words.
fn lower(program: &vyre::ir::Program, n: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let flat: Vec<u32> = VAST_ROWS.iter().flat_map(|r| r.iter().copied()).collect();
    debug_assert_eq!(flat.len(), n * VAST_STRIDE);
    let plain_init = vec![0u32; n * PLAIN_PG_STRIDE];
    let semantic_init = vec![0u32; n * SEMANTIC_PG_STRIDE];
    let edges_init = vec![0u32; n * PG_EDGE_ROWS_PER_NODE * PG_EDGE_STRIDE];

    let outputs = vyre_reference::reference_eval(
        program,
        &[
            Value::from(pack(&flat)),
            Value::from(pack(&plain_init)),
            Value::from(pack(&semantic_init)),
            Value::from(pack(&edges_init)),
        ],
    )
    .expect("lowering program must execute under reference_eval");
    (
        unpack(&outputs[0].to_bytes()),
        unpack(&outputs[1].to_bytes()),
        unpack(&outputs[2].to_bytes()),
    )
}

// A tiny control-flow-free tree: node0 -> {node1 -> node2 siblings}.
const VAST_ROWS: [[u32; VAST_STRIDE]; 3] = {
    // Built via const-fn-free literals so it can be a const; mirrors vast_row().
    // [kind, parent, first_child, next_sibling, _, src_off, src_len=1, _, _, _]
    [
        [10, SENTINEL, 1, SENTINEL, 0, 0, 1, 0, 0, 0], // root
        [20, 0, SENTINEL, 2, 0, 1, 1, 0, 0, 0],        // first child
        [30, 0, SENTINEL, SENTINEL, 0, 2, 1, 0, 0, 0], // sibling
    ]
};

#[test]
fn semantic_graph_with_and_without_control_resolution_agree_on_control_free_input() {
    let n = VAST_ROWS.len();
    let num_nodes = Expr::u32(n as u32);

    let resolved = c_lower_ast_to_pg_semantic_graph_with_pg(
        "vast_nodes",
        num_nodes.clone(),
        "out_plain_pg_nodes",
        "out_pg_nodes",
        "out_pg_edges",
    );
    let unresolved = c_lower_ast_to_pg_semantic_graph_with_pg_no_control_resolution(
        "vast_nodes",
        num_nodes,
        "out_plain_pg_nodes",
        "out_pg_nodes",
        "out_pg_edges",
    );

    let (plain_a, semantic_a, edges_a) = lower(&resolved, n);
    let (plain_b, semantic_b, edges_b) = lower(&unresolved, n);

    // The documented equivalence: byte-identical on all three outputs for control-flow-free input.
    assert_eq!(
        plain_a, plain_b,
        "plain PG nodes must match: control resolution changes nothing without control-flow constructs"
    );
    assert_eq!(semantic_a, semantic_b, "semantic PG nodes must match");
    assert_eq!(edges_a, edges_b, "PG edges must match");

    // Non-trivial: the semantic PG carries each node's kind + copied parent/first_child links.
    let node = |i: usize| i * SEMANTIC_PG_STRIDE;
    assert_eq!(semantic_a[node(0)], 10, "node0 kind copied to PG");
    assert_eq!(semantic_a[node(1)], 20, "node1 kind copied to PG");
    assert_eq!(semantic_a[node(2)], 30, "node2 kind copied to PG");
    // out_pg_nodes field 3 = parent_idx, field 4 = first_child_idx (see the pass' stores).
    assert_eq!(semantic_a[node(0) + 3], SENTINEL, "root has no parent");
    assert_eq!(semantic_a[node(0) + 4], 1, "root's first child is node1");
    assert_eq!(semantic_a[node(1) + 3], 0, "node1's parent is node0");

    // Non-trivial: a structural parent edge from node1 to its parent node0 exists in PG edges.
    // Edge row 0 per node is the PARENT edge (store_semantic_edge slot 0).
    let node1_edge_base = PG_EDGE_ROWS_PER_NODE * PG_EDGE_STRIDE;
    assert!(
        edges_a[node1_edge_base..node1_edge_base + PG_EDGE_STRIDE]
            .iter()
            .any(|&w| w != 0),
        "node1 must emit a non-empty parent edge row"
    );
}
