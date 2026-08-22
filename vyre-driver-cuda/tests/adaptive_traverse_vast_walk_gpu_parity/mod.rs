//! Parity tests for vyre-primitives graph::adaptive_traverse and
//! graph::vast_tree_walk preorder.

#![cfg(all(test, feature = "device-tests"))]

mod auto_selector_contracts;
mod dense_sparse_contracts;
#[path = "../harness/mod.rs"]
mod harness;
mod resident_sparse_dense_contracts;
mod resident_sparse_queue_contracts;
mod vast_walk_contracts;

use harness::{bytes_u32, live_dispatcher, pack_nodes, u32_bytes};
use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_foundation::vast::{walk_preorder_indices, VastNode, NODE_STRIDE_U32, SENTINEL};
use vyre_libs::graph::adaptive_traverse::{
    adaptive_dense_step, adaptive_node_dispatch_grid, adaptive_sparse_dense_step,
    AdaptiveTraversalMode,
};
use vyre_libs::graph::dispatch::adaptive_traverse::{
    adaptive_traverse_resident_graph_auto_step_with_scratch_into,
    adaptive_traverse_resident_graph_sparse_queue_step_with_scratch_into,
    adaptive_traverse_resident_graph_step_with_scratch_into,
    adaptive_traverse_resident_sparse_queue_step_with_scratch_into,
    upload_resident_adaptive_sparse_queue_graph, upload_resident_adaptive_traversal_graph,
    AdaptiveTraversalPlanCacheSnapshot, AdaptiveTraversalResidentScratch,
};
use vyre_libs::graph::vast_tree_walk::ast_walk_preorder;
use vyre_libs::reduce::count::reduce_count;

fn cpu_dense_step(frontier_in: &[u32], adj_rows_dense: &[u32], node_count: u32) -> Vec<u32> {
    vyre_reference::composition_witness::dense_bitmatrix_step_witness(
        frontier_in,
        adj_rows_dense,
        node_count,
    )
}

#[allow(clippy::too_many_arguments)]
fn cpu_sparse_dense_step(
    frontier_in: &[u32],
    selector_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    adj_rows_dense: &[u32],
    node_count: u32,
    allow_mask: u32,
    dense_threshold_pct: u32,
) -> Vec<u32> {
    let cutover = (u64::from(node_count) * u64::from(dense_threshold_pct)).div_ceil(100) as u32;
    if selector_count >= cutover {
        cpu_dense_step(frontier_in, adj_rows_dense, node_count)
    } else {
        vyre_reference::composition_witness::csr_forward_traverse_witness(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            frontier_in,
            allow_mask,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn adaptive_traverse_step(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    adj_rows_dense: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
    dense_threshold_pct: u32,
) -> Result<Vec<u32>, String> {
    let popcount = vyre_reference::composition_witness::frontier_popcount_witness(frontier_in)?;
    Ok(cpu_sparse_dense_step(
        frontier_in,
        popcount,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        adj_rows_dense,
        node_count,
        allow_mask,
        dense_threshold_pct,
    ))
}

fn bitset_words(node_count: u32) -> u32 {
    node_count.div_ceil(32).max(1)
}

fn run_dense_step(
    backend: &CudaBackend,
    frontier_in: &[u32],
    adj_rows_dense: &[u32],
    node_count: u32,
) -> Vec<u32> {
    let words = bitset_words(node_count);
    let program = adaptive_dense_step("frontier_in", "frontier_out", "adj", node_count);
    let inputs: Vec<Vec<u8>> = vec![
        u32_bytes(frontier_in),
        vec![0u8; words as usize * 4],
        u32_bytes(adj_rows_dense),
    ];
    let mut config = DispatchConfig::default();
    config.grid_override = Some(adaptive_node_dispatch_grid(node_count));
    let outputs = backend
        .dispatch(&program, &inputs, &config)
        .expect("dispatch");
    let mut out = bytes_u32(&outputs[0]);
    out.truncate(words as usize);
    out
}

fn build_dense_adj(edges: &[(u32, u32)], node_count: u32) -> Vec<u32> {
    let words = bitset_words(node_count) as usize;
    let mut rows = vec![0_u32; node_count as usize * words];
    for &(src, dst) in edges {
        rows[dst as usize * words + src as usize / 32] |= 1 << (src % 32);
    }
    rows
}

fn run_reduce_count(backend: &CudaBackend, frontier_in: &[u32]) -> Vec<u8> {
    let program = reduce_count("frontier_in", "frontier_popcount", frontier_in.len() as u32);
    let inputs = vec![u32_bytes(frontier_in), vec![0u8; 4]];
    let mut config = DispatchConfig::default();
    config.grid_override = Some([1, 1, 1]);
    let outputs = backend
        .dispatch(&program, &inputs, &config)
        .expect("reduce_count dispatch");
    outputs[0].clone()
}

fn run_sparse_dense_step(
    backend: &CudaBackend,
    frontier_in: &[u32],
    frontier_popcount: Vec<u8>,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    adj_rows_dense: &[u32],
    node_count: u32,
    dense_threshold_pct: u32,
) -> Vec<u32> {
    let words = bitset_words(node_count);
    let program = adaptive_sparse_dense_step(
        "frontier_in",
        "frontier_out",
        "frontier_popcount",
        "edge_offsets",
        "edge_targets",
        "edge_kind_mask",
        "adj_rows_dense",
        node_count,
        edge_targets.len() as u32,
        1,
        dense_threshold_pct,
    );
    let inputs: Vec<Vec<u8>> = vec![
        u32_bytes(frontier_in),
        vec![0u8; words as usize * 4],
        frontier_popcount,
        u32_bytes(edge_offsets),
        u32_bytes(edge_targets),
        u32_bytes(edge_kind_mask),
        u32_bytes(adj_rows_dense),
    ];
    let mut config = DispatchConfig::default();
    config.grid_override = Some(adaptive_node_dispatch_grid(node_count));
    let outputs = backend
        .dispatch(&program, &inputs, &config)
        .expect("adaptive hybrid dispatch");
    let mut out = bytes_u32(&outputs[0]);
    out.truncate(words as usize);
    out
}

fn pack_vast(nodes: &[VastNode]) -> (Vec<u8>, Vec<u32>) {
    let mut bytes = Vec::with_capacity(nodes.len() * NODE_STRIDE_U32 * 4);
    for n in nodes {
        bytes.extend_from_slice(&n.to_bytes());
    }
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (bytes, words)
}

fn run_preorder(
    backend: &CudaBackend,
    nodes_words: &[u32],
    node_count: u32,
    out_cap: u32,
) -> Vec<u32> {
    let program = ast_walk_preorder("nodes", "out", node_count, out_cap);
    let inputs: Vec<Vec<u8>> = vec![u32_bytes(nodes_words), vec![0u8; out_cap as usize * 4]];
    let mut config = DispatchConfig::default();
    // workgroup [1,1,1]; preorder is a single-threaded walk.
    config.grid_override = Some([1, 1, 1]);
    let outputs = backend
        .dispatch(&program, &inputs, &config)
        .expect("dispatch");
    let mut out = bytes_u32(&outputs[0]);
    out.truncate(out_cap as usize);
    out
}

fn make_node(parent: u32, first_child: u32, next_sibling: u32) -> VastNode {
    VastNode {
        kind: 0,
        parent_idx: parent,
        first_child,
        next_sibling,
        src_file: 0,
        src_byte_off: 0,
        src_byte_len: 0,
        attr_off: 0,
        attr_len: 0,
        reserved: 0,
    }
}
