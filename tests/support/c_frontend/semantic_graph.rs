//! Accessors and assertions for the C frontend's semantic property graph.
//!
//! `reference_ast_to_pg_semantic_graph` returns two packed buffers: node rows
//! with a kind/category/role triple, and a fixed number of edge rows per node.
//! Reading either one means knowing its stride and field offsets, so those live
//! here once instead of in every family that inspects the graph.

use super::rows::{word_at, VAST_STRIDE_U32};
use vyre_libs::parsing::c::lower::{
    C_AST_PG_EDGE_PARENT, C_AST_PG_EDGE_ROWS_PER_NODE, C_AST_PG_EDGE_STRIDE_U32,
    C_AST_PG_SEMANTIC_NODE_STRIDE_U32,
};

pub(crate) fn semantic_node_word(nodes: &[u8], idx: usize, field: usize) -> u32 {
    word_at(
        nodes,
        idx * C_AST_PG_SEMANTIC_NODE_STRIDE_U32 as usize + field,
    )
}

pub(crate) fn semantic_edge_word(
    edges: &[u8],
    node_idx: usize,
    edge_slot: usize,
    field: usize,
) -> u32 {
    let edge_idx = node_idx * C_AST_PG_EDGE_ROWS_PER_NODE as usize + edge_slot;
    word_at(edges, edge_idx * C_AST_PG_EDGE_STRIDE_U32 as usize + field)
}

pub(crate) fn vast_word(rows: &[u8], idx: usize, field: usize) -> u32 {
    word_at(rows, idx * VAST_STRIDE_U32 + field)
}

/// Assert the semantic node at `idx` carries the expected kind, category, role.
pub(crate) fn assert_semantic_node(nodes: &[u8], idx: usize, kind: u32, category: u32, role: u32) {
    assert_eq!(semantic_node_word(nodes, idx, 0), kind, "kind[{idx}]");
    assert_eq!(
        semantic_node_word(nodes, idx, 6),
        category,
        "category[{idx}]"
    );
    assert_eq!(semantic_node_word(nodes, idx, 7), role, "role[{idx}]");
}

/// Assert edge slot 0 of `node_idx` is the parent edge with the expected shape.
pub(crate) fn assert_parent_edge(
    edges: &[u8],
    node_idx: usize,
    parent_idx: u32,
    role: u32,
    category: u32,
) {
    assert_eq!(
        semantic_edge_word(edges, node_idx, 0, 0),
        C_AST_PG_EDGE_PARENT,
        "parent edge kind[{node_idx}]"
    );
    assert_eq!(semantic_edge_word(edges, node_idx, 0, 1), parent_idx);
    assert_eq!(semantic_edge_word(edges, node_idx, 0, 2), node_idx as u32);
    assert_eq!(semantic_edge_word(edges, node_idx, 0, 4), role);
    assert_eq!(semantic_edge_word(edges, node_idx, 0, 5), category);
}

pub(crate) fn assert_semantic_edge(
    edges: &[u8],
    node_idx: usize,
    edge_slot: usize,
    edge_kind: u32,
    src_idx: u32,
    dst_idx: u32,
) {
    assert_eq!(
        semantic_edge_word(edges, node_idx, edge_slot, 0),
        edge_kind,
        "semantic edge kind node={node_idx} slot={edge_slot}"
    );
    assert_eq!(
        semantic_edge_word(edges, node_idx, edge_slot, 1),
        src_idx,
        "semantic edge src node={node_idx} slot={edge_slot}"
    );
    assert_eq!(
        semantic_edge_word(edges, node_idx, edge_slot, 2),
        dst_idx,
        "semantic edge dst node={node_idx} slot={edge_slot}"
    );
}
