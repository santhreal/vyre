//! Device-parity contracts for the persistent-BFS per-iteration density readback.
//!
//! [`persistent_bfs_with_density`] declares one extra `max_iters`-length u32
//! output, `density_active`, whose entry `i` holds the popcount of the frontier
//! after traversal step `i` (flat once the closure converges, since growth is
//! monotone). A host reconstructs every [`FrontierDensityTelemetry`]-style
//! aggregate from this array plus the seed popcount without a per-step device
//! round-trip. [`try_cpu_ref_density`] is the source of truth for that array.
//!
//! These tests dispatch the real density-instrumented IR on the reference
//! interpreter and assert the device density array matches the oracle
//! bit-for-bit on both program variants: the single-workgroup path
//! (`node_count <= 256`, build-time-unrolled leader popcount) and the grid-sync
//! path (`node_count > 256`, per-word atomic-add reduction across three
//! grid-sync barriers). The graphs mirror `converged_device_parity`, so each
//! path advances exactly one hop per iteration, the regime `cpu_ref` models.

use super::*;
use crate::wire::pack_u32_slice;
use vyre_driver::grid_sync::{contains_grid_sync, dispatch_with_grid_sync_split};
use vyre_driver::DispatchConfig;
use vyre_driver_reference::CpuRefBackend;
use vyre_foundation::ir::{BufferAccess, Program};
use vyre_reference::{output_index, reference_eval, reference_eval_with_grid};

/// Build the positional storage-buffer inputs for a density-instrumented
/// persistent_bfs program: the read-only CSR/frontier buffers carry data, every
/// ReadWrite output (frontier_out, changed, converged, density_active) starts
/// zeroed, and interpreter-internal workgroup buffers are skipped.
fn build_inputs(
    program: &Program,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
) -> Vec<Vec<u8>> {
    let mut inputs: Vec<Vec<u8>> = Vec::new();
    for buffer in program.buffers() {
        if buffer.access() == BufferAccess::Workgroup {
            continue;
        }
        let count = buffer.count() as usize;
        let mut data = match buffer.name() {
            "pg_edge_offsets" => edge_offsets.to_vec(),
            "pg_edge_targets" => edge_targets.to_vec(),
            "pg_edge_kind_mask" => edge_kind_mask.to_vec(),
            "frontier_in" => frontier_in.to_vec(),
            _ => Vec::new(),
        };
        data.resize(count, 0);
        inputs.push(pack_u32_slice(&data));
    }
    inputs
}

/// Read a named u32 output buffer in `output_buffer_indices` order.
fn read_named_output(program: &Program, outputs: &[Vec<u8>], name: &str) -> Vec<u32> {
    let idx = output_index(program, name)
        .unwrap_or_else(|| panic!("Fix: persistent_bfs must expose the `{name}` output buffer."));
    outputs[idx]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Dispatch the density-instrumented persistent_bfs and read back the frontier
/// and the `max_iters`-length density array.
///
/// The single-workgroup program runs directly on the interpreter; the grid-sync
/// program carries `GridSync` barriers and routes through
/// [`dispatch_with_grid_sync_split`] on [`CpuRefBackend`], exactly as the
/// converged-parity harness does, so every barrier (including the three the
/// density reduction adds per iteration) becomes a kernel-launch boundary.
fn run_device_density(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
    max_iters: u32,
) -> (Vec<u32>, Vec<u32>) {
    let edge_count = edge_targets.len() as u32;
    let shape = ProgramGraphShape::new(node_count, edge_count.max(1));
    let program = persistent_bfs_with_density(
        shape,
        "frontier_in",
        "frontier_out",
        DENSITY_ACTIVE_BUFFER,
        allow_mask,
        max_iters,
    );
    let words = bitset_words(node_count) as usize;
    let inputs = build_inputs(&program, edge_offsets, edge_targets, edge_kind_mask, frontier_in);

    let outputs: Vec<Vec<u8>> = if contains_grid_sync(&program) {
        let borrowed: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
        dispatch_with_grid_sync_split(
            &CpuRefBackend,
            &program,
            &borrowed,
            &DispatchConfig::default(),
        )
        .expect("Fix: density persistent_bfs grid-sync split dispatch must succeed on a valid graph.")
    } else {
        reference_eval(
            &program,
            &inputs
                .iter()
                .map(|bytes| vyre_reference::value::Value::from(bytes.as_slice()))
                .collect::<Vec<_>>(),
        )
        .expect("Fix: density persistent_bfs reference dispatch must succeed on a valid graph.")
        .into_iter()
        .map(|value| value.to_bytes())
        .collect()
    };
    let mut frontier_out = read_named_output(&program, &outputs, "frontier_out");
    frontier_out.truncate(words);
    let mut density = read_named_output(&program, &outputs, DENSITY_ACTIVE_BUFFER);
    density.truncate(max_iters as usize);
    (frontier_out, density)
}

/// Assert the device density array (and frontier) matches the CPU oracle, and
/// return the oracle's density array for a self-documenting exact-value check.
fn assert_device_density_matches_oracle(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
    max_iters: u32,
) -> Vec<u32> {
    let (frontier, _outcome, active) = try_cpu_ref_density(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_in,
        allow_mask,
        max_iters,
    )
    .expect("Fix: CPU density oracle must accept a valid graph.");
    assert_eq!(
        active.len(),
        max_iters as usize,
        "oracle density array must have exactly max_iters entries"
    );
    let (device_frontier, device_density) = run_device_density(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_in,
        allow_mask,
        max_iters,
    );
    assert_eq!(
        device_frontier, frontier,
        "node_count={node_count} max_iters={max_iters}: device frontier must equal the CPU oracle."
    );
    assert_eq!(
        device_density, active,
        "node_count={node_count} max_iters={max_iters}: device density array must equal the CPU oracle."
    );
    active
}

/// Reverse-numbered chain `(n-1) -> ... -> 0` seeded at the top node; an
/// ascending single-workgroup sweep advances one hop per iteration.
fn reverse_chain(node_count: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut offsets = vec![0u32];
    for i in 0..node_count {
        offsets.push(i);
    }
    let targets: Vec<u32> = (0..node_count.saturating_sub(1)).collect();
    let masks = vec![1u32; targets.len()];
    let words = bitset_words(node_count) as usize;
    let mut seed = vec![0u32; words];
    let top = node_count - 1;
    seed[(top / 32) as usize] = 1 << (top % 32);
    (offsets, targets, masks, seed)
}

/// A two-level fan graph with `node_count > 256` forcing the grid-sync path,
/// diameter 2: node 0 -> nodes 1..=fanout, node 1 -> the final leaf.
fn grid_sync_two_level(fanout: u32) -> (u32, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let node_count = fanout + 2;
    let leaf = node_count - 1;
    let mut targets: Vec<u32> = (1..=fanout).collect();
    targets.push(leaf);
    let mut offsets = vec![0u32, fanout];
    for _ in 2..=node_count {
        offsets.push(fanout + 1);
    }
    let masks = vec![1u32; targets.len()];
    let words = bitset_words(node_count) as usize;
    let mut seed = vec![0u32; words];
    seed[0] = 1;
    (node_count, offsets, targets, masks, seed)
}

#[test]
fn single_workgroup_density_matches_oracle_growing_then_flat_after_convergence() {
    // 4-node reverse chain, diameter 3, seeded at node 3: popcount after each step
    // is 2, 3, 4, then flat at 4 once the closure converges on step 3.
    let (offsets, targets, masks, seed) = reverse_chain(4);

    // Below the diameter: still growing at the budget boundary (both steps
    // unrolled), density = [2, 3].
    let active = assert_device_density_matches_oracle(4, &offsets, &targets, &masks, &seed, 0xFFFF_FFFF, 2);
    assert_eq!(active, vec![2, 3]);

    // Above the diameter: growth then the flat converged tail, exercising both the
    // unrolled steps (0..4) and the trailing bounded loop (4..8).
    let active = assert_device_density_matches_oracle(4, &offsets, &targets, &masks, &seed, 0xFFFF_FFFF, 8);
    assert_eq!(active, vec![2, 3, 4, 4, 4, 4, 4, 4]);
}

#[test]
fn single_workgroup_density_repeats_seed_popcount_when_seed_is_already_a_fixpoint() {
    // 2-node chain 0 -> 1 seeded with both nodes present: the first step adds
    // nothing, so every density entry is the seed popcount 2.
    let offsets = [0u32, 1, 1];
    let targets = [1u32];
    let masks = [1u32];
    let seed = [0b11u32];
    let active =
        assert_device_density_matches_oracle(2, &offsets, &targets, &masks, &seed, 0xFFFF_FFFF, 4);
    assert_eq!(active, vec![2, 2, 2, 2]);
}

#[test]
fn grid_sync_density_matches_oracle_across_the_budget_boundary() {
    // 258 nodes (> 256) forces the grid-sync density program; diameter 2. Seeded
    // at node 0: popcount after step 0 is 257 (root plus its 256 fan targets),
    // after step 1 is 258 (the leaf off node 1), then flat at 258.
    let (node_count, offsets, targets, masks, seed) = grid_sync_two_level(256);
    assert!(node_count > 256, "must exercise the grid-sync density path");

    let active =
        assert_device_density_matches_oracle(node_count, &offsets, &targets, &masks, &seed, 0xFFFF_FFFF, 1);
    assert_eq!(active, vec![257]);

    let active =
        assert_device_density_matches_oracle(node_count, &offsets, &targets, &masks, &seed, 0xFFFF_FFFF, 2);
    assert_eq!(active, vec![257, 258]);

    let active =
        assert_device_density_matches_oracle(node_count, &offsets, &targets, &masks, &seed, 0xFFFF_FFFF, 3);
    assert_eq!(active, vec![257, 258, 258]);
}

/// Dispatch the BATCH density-instrumented program and read back the flat
/// `[query][iter]` density array, mirroring `run_device_batch` in the converged
/// harness.
fn run_device_batch_density(
    node_count: u32,
    query_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
    max_iters: u32,
) -> Vec<u32> {
    let edge_count = edge_targets.len() as u32;
    let shape = ProgramGraphShape::new(node_count, edge_count.max(1));
    let program = persistent_bfs_batch_with_density(
        shape,
        "frontier_in",
        "frontier_out",
        "changed",
        "converged",
        DENSITY_ACTIVE_BUFFER,
        query_count,
        allow_mask,
        max_iters,
    );
    let inputs = build_inputs(&program, edge_offsets, edge_targets, edge_kind_mask, frontier_in);
    let grid = persistent_bfs_batch_dispatch_grid(node_count, query_count);

    let outputs: Vec<Vec<u8>> = if contains_grid_sync(&program) {
        let borrowed: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
        let mut config = DispatchConfig::default();
        config.dispatch_grid = Some(grid);
        dispatch_with_grid_sync_split(&CpuRefBackend, &program, &borrowed, &config).expect(
            "Fix: density persistent_bfs_batch grid-sync split dispatch must succeed on a valid graph.",
        )
    } else {
        reference_eval_with_grid(
            &program,
            &inputs
                .iter()
                .map(|bytes| vyre_reference::value::Value::from(bytes.as_slice()))
                .collect::<Vec<_>>(),
            grid,
        )
        .expect("Fix: density persistent_bfs_batch reference dispatch must succeed on a valid graph.")
        .into_iter()
        .map(|value| value.to_bytes())
        .collect()
    };
    let mut density = read_named_output(&program, &outputs, DENSITY_ACTIVE_BUFFER);
    density.truncate((query_count * max_iters) as usize);
    density
}

/// Assert the batch device density array matches the per-query CPU oracle: each
/// query is an independent single-frontier density run, so the flat
/// `[query][iter]` device array must equal the concatenation of
/// [`try_cpu_ref_density`] over the seeds. Returns that oracle concatenation.
fn assert_batch_device_density_matches_oracle(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    seeds: &[Vec<u32>],
    allow_mask: u32,
    max_iters: u32,
) -> Vec<u32> {
    let words = bitset_words(node_count) as usize;
    let query_count = seeds.len() as u32;
    let mut frontier_in = Vec::with_capacity(words * seeds.len());
    for seed in seeds {
        assert_eq!(seed.len(), words, "each batch seed must be one frontier bitset");
        frontier_in.extend_from_slice(seed);
    }
    let device_density = run_device_batch_density(
        node_count,
        query_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        &frontier_in,
        allow_mask,
        max_iters,
    );

    let mut oracle = Vec::with_capacity(seeds.len() * max_iters as usize);
    for (query, seed) in seeds.iter().enumerate() {
        let (_frontier, _outcome, active) = try_cpu_ref_density(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            seed,
            allow_mask,
            max_iters,
        )
        .expect("Fix: CPU density oracle must accept a valid graph.");
        let start = query * max_iters as usize;
        let end = start + max_iters as usize;
        assert_eq!(
            &device_density[start..end],
            active.as_slice(),
            "query {query}: batch device density must equal the CPU oracle."
        );
        oracle.extend_from_slice(&active);
    }
    oracle
}

#[test]
fn batch_single_workgroup_density_matches_oracle_across_queries() {
    // 4-node reverse chain (diameter 3), single-workgroup batch path. Three queries
    // with distinct seeds and distinct density trajectories at the same budget:
    // node 3 grows 2 then 3; node 1 grows to 2 then converges; a saturated seed
    // sits flat at 4. If grid.y collapsed to 1, queries 1 and 2 would read zeros.
    let (offsets, targets, masks, _seed) = reverse_chain(4);
    let seeds = vec![vec![0b1000u32], vec![0b0010u32], vec![0b1111u32]];
    let oracle =
        assert_batch_device_density_matches_oracle(4, &offsets, &targets, &masks, &seeds, 0xFFFF_FFFF, 2);
    assert_eq!(oracle, vec![2, 3, 2, 2, 4, 4]);
}

#[test]
fn batch_grid_sync_density_matches_oracle_across_queries() {
    // 258 nodes (> 256) forces the grid-sync batch density program; diameter 2.
    // Query 0 seeded at the root grows 257 then 258; query 1 seeded at the leaf is
    // an immediate fixpoint sitting flat at 1.
    let (node_count, offsets, targets, masks, root_seed) = grid_sync_two_level(256);
    assert!(node_count > 256, "must exercise the grid-sync batch density path");
    let words = bitset_words(node_count) as usize;
    let leaf = node_count - 1;
    let mut leaf_seed = vec![0u32; words];
    leaf_seed[(leaf / 32) as usize] = 1 << (leaf % 32);
    let seeds = vec![root_seed, leaf_seed];
    let oracle = assert_batch_device_density_matches_oracle(
        node_count,
        &offsets,
        &targets,
        &masks,
        &seeds,
        0xFFFF_FFFF,
        3,
    );
    assert_eq!(oracle, vec![257, 258, 258, 1, 1, 1]);
}
