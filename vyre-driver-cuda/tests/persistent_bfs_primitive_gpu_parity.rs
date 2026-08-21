//! Parity test: vyre-primitives persistent_bfs Program matches CPU oracle.

#![cfg(all(test, feature = "device-tests"))]

mod harness;

use harness::{bytes_u32, u32_bytes, with_live_backend};
use vyre_driver::DispatchConfig;
use vyre_foundation::ir::BufferAccess;
use vyre_libs::graph::persistent_bfs::{
    persistent_bfs, persistent_bfs_batch, persistent_bfs_batch_dispatch_grid,
    persistent_bfs_single_dispatch_grid, validate_persistent_bfs_converged_flag,
};
use vyre_libs::graph::program_graph::ProgramGraphShape;
use vyre_reference::composition_witness::csr_persistent_closure_detailed_witness;

fn run(
    node_count: u32,
    edge_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
    max_iters: u32,
) -> (Vec<u32>, u32, u32) {
    let words = ((node_count + 31) / 32).max(1);
    let pg_nodes = vec![0u32; node_count as usize];
    let pg_node_tags = vec![0u32; node_count as usize];
    let program = persistent_bfs(
        ProgramGraphShape::new(node_count, edge_count.max(1)),
        "frontier_in",
        "frontier_out",
        allow_mask,
        max_iters,
    );
    let inputs: Vec<Vec<u8>> = program
        .buffers()
        .iter()
        .filter(|buffer| buffer.access() != BufferAccess::Workgroup)
        .map(|buffer| {
            let declared_words = buffer.count().max(1) as usize;
            match buffer.name() {
                "pg_nodes" => u32_bytes(&pg_nodes),
                "pg_edge_offsets" => u32_bytes(edge_offsets),
                "pg_edge_targets" => u32_bytes(edge_targets),
                "pg_edge_kind_mask" => u32_bytes(edge_kind_mask),
                "pg_node_tags" => u32_bytes(&pg_node_tags),
                "frontier_in" => u32_bytes(frontier),
                "frontier_out" | "changed" | "converged" => vec![0u8; declared_words * 4],
                other => panic!("Unexpected buffer in persistent_bfs program: {other}"),
            }
        })
        .collect();
    let mut config = DispatchConfig::default();
    config.grid_override = Some(persistent_bfs_single_dispatch_grid(node_count));
    let outputs = with_live_backend("persistent BFS primitive", |backend| {
        backend
            .dispatch(&program, &inputs, &config)
            .unwrap_or_else(|error| {
                panic!("Fix: CUDA persistent BFS primitive dispatch failed: {error}")
            })
    });
    let mut frontier_out = bytes_u32(&outputs[0]);
    frontier_out.truncate(words as usize);
    let changed = bytes_u32(&outputs[1])[0];
    let converged = bytes_u32(&outputs[2])[0];
    validate_persistent_bfs_converged_flag(converged)
        .expect("Fix: CUDA persistent BFS must write 0 or 1 to the converged output");
    (frontier_out, changed, converged)
}

/// Expected outcome for the single-query persistent-BFS program.
///
/// The program writes three outputs since 0.7.0: the accumulated frontier, the
/// sticky `changed` flag, and `converged`. This test used to compare only the
/// first two, so a device that reported a fixpoint it never reached, or one
/// that never wrote the flag at all, would still pass. Comparing the flag
/// against the reference source of truth is what makes an under-approximated closure
/// a test failure instead of a silently truncated answer.
fn expected_persistent_bfs_single(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
    max_iters: u32,
) -> (Vec<u32>, u32, u32) {
    let result = csr_persistent_closure_detailed_witness(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_in,
        allow_mask,
        max_iters,
    );
    (result.frontier, result.changed, u32::from(result.converged))
}

fn run_batch(
    node_count: u32,
    edge_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontiers: &[u32],
    query_count: u32,
    allow_mask: u32,
    max_iters: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let words = ((node_count + 31) / 32).max(1);
    let total_words = words as usize * query_count.max(1) as usize;
    let pg_nodes = vec![0u32; node_count as usize];
    let pg_node_tags = vec![0u32; node_count as usize];
    let program = persistent_bfs_batch(
        ProgramGraphShape::new(node_count, edge_count.max(1)),
        "frontier_in",
        "frontier_out",
        "changed",
        "converged",
        query_count,
        allow_mask,
        max_iters,
    );
    let inputs: Vec<Vec<u8>> = vec![
        u32_bytes(&pg_nodes),
        u32_bytes(edge_offsets),
        u32_bytes(edge_targets),
        u32_bytes(edge_kind_mask),
        u32_bytes(&pg_node_tags),
        u32_bytes(frontiers),
        vec![0u8; total_words * 4],
        vec![0u8; query_count.max(1) as usize * 4],
        vec![0u8; query_count.max(1) as usize * 4],
    ];
    let mut config = DispatchConfig::default();
    config.grid_override = Some(persistent_bfs_batch_dispatch_grid(node_count, query_count));
    let outputs = with_live_backend("persistent BFS primitive batch", |backend| {
        backend
            .dispatch(&program, &inputs, &config)
            .unwrap_or_else(|error| {
                panic!("Fix: CUDA persistent BFS primitive batch dispatch failed: {error}")
            })
    });
    let mut frontier_out = bytes_u32(&outputs[0]);
    frontier_out.truncate(total_words);
    let mut changed = bytes_u32(&outputs[1]);
    changed.truncate(query_count as usize);
    let mut converged = bytes_u32(&outputs[2]);
    converged.truncate(query_count as usize);
    (frontier_out, changed, converged)
}

fn expected_persistent_bfs_batch(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontiers: &[u32],
    query_count: u32,
    allow_mask: u32,
    max_iters: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let words = ((node_count + 31) / 32).max(1) as usize;
    let mut frontier_out = Vec::with_capacity(words * query_count as usize);
    let mut changed_out = Vec::with_capacity(query_count as usize);
    let mut converged_out = Vec::with_capacity(query_count as usize);
    for query in 0..query_count as usize {
        let start = query * words;
        let end = start + words;
        let result = csr_persistent_closure_detailed_witness(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            &frontiers[start..end],
            allow_mask,
            max_iters,
        );
        frontier_out.extend_from_slice(&result.frontier);
        changed_out.push(result.changed);
        converged_out.push(u32::from(result.converged));
    }
    (frontier_out, changed_out, converged_out)
}

#[test]
fn cuda_persistent_bfs_chain_converges_changed_set() {
    let n = 4u32;
    let edge_offsets = vec![0u32, 1, 2, 3, 3];
    let edge_targets = vec![1u32, 2, 3];
    let edge_kind_mask = vec![1u32; 3];
    let frontier = vec![0b0001u32];
    let cpu = expected_persistent_bfs_single(
        n,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier,
        0xFFFF_FFFF,
        8,
    );
    let gpu = run(
        n,
        3,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier,
        0xFFFF_FFFF,
        8,
    );
    assert_eq!(gpu, cpu);
    assert_eq!(gpu.0, vec![0b1111u32]);
    assert_eq!(gpu.1, 1);
    assert_eq!(
        gpu.2, 1,
        "Fix: a 4-node chain closes inside an 8-iteration budget, so the device must report a fixpoint."
    );
}

#[test]
fn cuda_persistent_bfs_diamond_converges() {
    let n = 4u32;
    let edge_offsets = vec![0u32, 2, 3, 4, 4];
    let edge_targets = vec![1u32, 2, 3, 3];
    let edge_kind_mask = vec![1u32; 4];
    let frontier = vec![0b0001u32];
    let cpu = expected_persistent_bfs_single(
        n,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier,
        0xFFFF_FFFF,
        8,
    );
    let gpu = run(
        n,
        4,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier,
        0xFFFF_FFFF,
        8,
    );
    assert_eq!(gpu, cpu);
    assert_eq!(gpu.0, vec![0b1111u32]);
}

#[test]
fn cuda_persistent_bfs_isolated_seed_unchanged() {
    let n = 3u32;
    let edge_offsets = vec![0u32, 0, 0, 0];
    let edge_targets: Vec<u32> = Vec::new();
    let edge_kind_mask: Vec<u32> = Vec::new();
    let padded_edge_targets = vec![0u32; 1];
    let padded_edge_kind_mask = vec![0u32; 1];
    let frontier = vec![0b001u32];
    let cpu = expected_persistent_bfs_single(
        n,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier,
        0xFFFF_FFFF,
        8,
    );
    let gpu = run(
        n,
        0,
        &edge_offsets,
        &padded_edge_targets,
        &padded_edge_kind_mask,
        &frontier,
        0xFFFF_FFFF,
        8,
    );
    assert_eq!(gpu, cpu);
    assert_eq!(gpu.0, vec![0b001u32]);
    assert_eq!(gpu.1, 0);
    assert_eq!(
        gpu.2, 1,
        "Fix: an isolated seed reaches its fixpoint on the first step, so converged must be 1."
    );
}

#[test]
fn cuda_persistent_bfs_large_no_edges_converges_without_changed() {
    let n = 513u32;
    let edge_offsets = vec![0u32; n as usize + 1];
    let edge_targets: Vec<u32> = Vec::new();
    let edge_kind_mask: Vec<u32> = Vec::new();
    let padded_edge_targets = vec![0u32; 1];
    let padded_edge_kind_mask = vec![0u32; 1];
    let words = ((n + 31) / 32) as usize;
    let mut frontier = vec![0u32; words];
    frontier[8] = 1;

    let cpu = expected_persistent_bfs_single(
        n,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier,
        0xFFFF_FFFF,
        64,
    );
    let gpu = run(
        n,
        0,
        &edge_offsets,
        &padded_edge_targets,
        &padded_edge_kind_mask,
        &frontier,
        0xFFFF_FFFF,
        64,
    );

    assert_eq!(gpu, cpu);
    assert_eq!(gpu.0[8], 1);
    assert!(gpu.0[..8].iter().all(|word| *word == 0));
    assert!(gpu.0[9..].iter().all(|word| *word == 0));
    assert_eq!(gpu.1, 0);
}

#[test]
fn cuda_persistent_bfs_large_graph_crosses_workgroup_boundary() {
    let n = 513u32;
    let mut edge_offsets = vec![0u32; n as usize + 1];
    let mut edge_targets = Vec::new();
    let mut edge_kind_mask = Vec::new();
    for src in 0..n {
        edge_offsets[src as usize] = edge_targets.len() as u32;
        if src == 256 {
            edge_targets.push(512);
            edge_kind_mask.push(1);
        }
    }
    edge_offsets[n as usize] = edge_targets.len() as u32;
    let mut frontier = vec![0u32; ((n + 31) / 32) as usize];
    frontier[8] = 1;

    let cpu = expected_persistent_bfs_single(
        n,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier,
        0xFFFF_FFFF,
        1,
    );
    let gpu = run(
        n,
        edge_targets.len() as u32,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier,
        0xFFFF_FFFF,
        1,
    );

    assert_eq!(gpu, cpu);
    assert_eq!(gpu.0[8], 1);
    assert_eq!(gpu.0[16], 1);
    assert_eq!(gpu.1, 1);
}

#[test]
fn cuda_persistent_bfs_large_chain_honors_one_step_cap() {
    let n = 513u32;
    let mut edge_offsets = Vec::with_capacity(n as usize + 1);
    let mut edge_targets = Vec::with_capacity(n as usize - 1);
    let mut edge_kind_mask = Vec::with_capacity(n as usize - 1);
    edge_offsets.push(0);
    for src in 0..n {
        if src + 1 < n {
            edge_targets.push(src + 1);
            edge_kind_mask.push(1);
        }
        edge_offsets.push(edge_targets.len() as u32);
    }
    let mut frontier = vec![0u32; ((n + 31) / 32) as usize];
    frontier[0] = 1;

    let cpu = expected_persistent_bfs_single(
        n,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier,
        0xFFFF_FFFF,
        1,
    );
    let gpu = run(
        n,
        edge_targets.len() as u32,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier,
        0xFFFF_FFFF,
        1,
    );

    assert_eq!(gpu, cpu);
    assert_eq!(gpu.0[0], 0b11);
    assert!(
        gpu.0[1..].iter().all(|word| *word == 0),
        "Fix: one persistent-BFS iteration must not cascade past node 1 on a long chain."
    );
    assert_eq!(gpu.1, 1);
    assert_eq!(
        gpu.2, 0,
        "Fix: a 513-node chain capped at one iteration is still growing, so the frontier is an \
         under-approximation and converged must be 0."
    );
}

#[test]
fn cuda_persistent_bfs_batch_large_chain_honors_one_step_cap_per_query() {
    let n = 513u32;
    let words = ((n + 31) / 32) as usize;
    let query_count = 2u32;
    let mut edge_offsets = Vec::with_capacity(n as usize + 1);
    let mut edge_targets = Vec::with_capacity(n as usize - 1);
    let mut edge_kind_mask = Vec::with_capacity(n as usize - 1);
    edge_offsets.push(0);
    for src in 0..n {
        if src + 1 < n {
            edge_targets.push(src + 1);
            edge_kind_mask.push(1);
        }
        edge_offsets.push(edge_targets.len() as u32);
    }
    let mut frontiers = vec![0u32; words * query_count as usize];
    frontiers[0] = 1;
    frontiers[words + 8] = 1;

    let cpu = expected_persistent_bfs_batch(
        n,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontiers,
        query_count,
        0xFFFF_FFFF,
        1,
    );
    let gpu = run_batch(
        n,
        edge_targets.len() as u32,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontiers,
        query_count,
        0xFFFF_FFFF,
        1,
    );

    assert_eq!(gpu, cpu);
    assert_eq!(gpu.0[0], 0b11);
    assert!(gpu.0[1..words].iter().all(|word| *word == 0));
    assert_eq!(gpu.0[words + 8], 0b11);
    assert!(gpu.0[words..words + 8].iter().all(|word| *word == 0));
    assert!(gpu.0[words + 9..].iter().all(|word| *word == 0));
    assert_eq!(gpu.1, vec![1, 1]);
}
