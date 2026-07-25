//! Device-parity contracts for the persistent-BFS converged readback.
//!
//! The CPU reference [`try_cpu_ref_converged`] is the source of truth for the
//! converged signal (a run that exhausts `max_iters` while still growing reports
//! `converged == false`; a reached fixpoint reports `true`). These tests dispatch
//! the real IR program on the reference interpreter and assert the device
//! converged word matches that oracle bit-for-bit, on both program variants: the
//! single-workgroup path (`node_count <= 256`) and the grid-sync path
//! (`node_count > 256`).
//!
//! Step semantics differ by path, so the graphs here are chosen to make every
//! path advance exactly one hop per iteration, the regime `cpu_ref` models:
//!
//! * The single-workgroup step is a serial in-place sweep over source nodes in
//!   ascending index order (`csr_forward_or_changed_body_prefixed`). On a
//!   forward-numbered chain one sweep reaches the whole closure (a node set by an
//!   earlier source is seen by a later source in the same sweep). Numbering the
//!   chain in reverse (edges point to lower indices) makes an ascending sweep
//!   advance exactly one hop, matching the oracle.
//! * The grid-sync step snapshots active bits before writing
//!   (`csr_forward_or_changed_parallel_snapshot_*`), so it is one hop per
//!   iteration for any numbering.

use super::*;
use crate::wire::pack_u32_slice;
use vyre_driver::grid_sync::{
    contains_grid_sync, dispatch_with_grid_sync_split, dispatch_with_grid_sync_split_via,
};
use vyre_driver::backend::VyreBackend;
use vyre_driver::DispatchConfig;
use vyre_driver_reference::CpuRefBackend;
use vyre_foundation::ir::{BufferAccess, Program};
use vyre_reference::{output_index, reference_eval, reference_eval_with_grid};

/// Build the positional storage-buffer inputs for a persistent_bfs program: the
/// read-only CSR/frontier buffers carry data, every ReadWrite output
/// (frontier_out, changed, converged) starts zeroed, and interpreter-internal
/// workgroup buffers are skipped.
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

/// Read a named u32 output buffer from a `Vec<Vec<u8>>` returned in
/// `output_buffer_indices` order (identical to [`output_index`]'s ordering).
fn read_named_output(program: &Program, outputs: &[Vec<u8>], name: &str) -> Vec<u32> {
    let idx = output_index(program, name)
        .unwrap_or_else(|| panic!("Fix: persistent_bfs must expose the `{name}` output buffer."));
    outputs[idx]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Dispatch persistent_bfs and read back the frontier, the sticky changed flag,
/// and the device converged word.
///
/// The single-workgroup program (`node_count <= 256`) is one kernel launch, so
/// the reference interpreter runs it directly. The grid-sync program
/// (`node_count > 256`) carries `Node::Barrier { ordering: GridSync }`, which the
/// interpreter cannot execute in one pass (variables bound in one segment are
/// read in the next, carried through buffers). It routes through
/// [`dispatch_with_grid_sync_split`] on [`CpuRefBackend`], the same non-native
/// grid-sync path the conform runner and production drivers use: every barrier
/// becomes a kernel-launch boundary, so prior writes are globally visible to the
/// next segment. Both paths return outputs in `output_buffer_indices` order,
/// which is exactly [`output_index`]'s ordering, so the readback is uniform.
fn run_device(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
    max_iters: u32,
) -> (Vec<u32>, u32, u32) {
    let edge_count = edge_targets.len() as u32;
    let shape = ProgramGraphShape::new(node_count, edge_count.max(1));
    let program = persistent_bfs(shape, "frontier_in", "frontier_out", allow_mask, max_iters);
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
        .expect("Fix: persistent_bfs grid-sync split dispatch must succeed on a valid graph.")
    } else {
        reference_eval(
            &program,
            &inputs
                .iter()
                .map(|bytes| vyre_reference::value::Value::from(bytes.as_slice()))
                .collect::<Vec<_>>(),
        )
        .expect("Fix: persistent_bfs reference dispatch must succeed on a valid graph.")
        .into_iter()
        .map(|value| value.to_bytes())
        .collect()
    };
    let mut frontier_out = read_named_output(&program, &outputs, "frontier_out");
    frontier_out.truncate(words);
    let changed = read_named_output(&program, &outputs, "changed")[0];
    let converged = read_named_output(&program, &outputs, "converged")[0];
    (frontier_out, changed, converged)
}

/// Assert the device frontier/changed/converged triple matches the CPU oracle,
/// returning the oracle's `(changed, converged)` for a self-documenting check.
fn assert_device_matches_oracle(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
    max_iters: u32,
) -> (u32, bool) {
    let (frontier, outcome) = try_cpu_ref_converged(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_in,
        allow_mask,
        max_iters,
    )
    .expect("Fix: CPU oracle must accept a valid graph.");
    let (device_frontier, device_changed, device_converged) = run_device(
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
        device_changed, outcome.changed,
        "node_count={node_count} max_iters={max_iters}: device changed flag must equal the CPU oracle."
    );
    assert_eq!(
        device_converged,
        u32::from(outcome.converged),
        "node_count={node_count} max_iters={max_iters}: device converged word must equal the CPU oracle (converged={}).",
        outcome.converged
    );
    (outcome.changed, outcome.converged)
}

/// Build a reverse-numbered chain `(n-1) -> (n-2) -> ... -> 1 -> 0`, seeded at
/// the top node so an ascending single-workgroup sweep advances one hop per
/// iteration (diameter `n - 1`).
fn reverse_chain(node_count: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    // node i (i >= 1) owns one edge i -> i-1; node 0 owns none.
    let mut offsets = vec![0u32];
    for i in 0..node_count {
        // offsets[i+1] = number of edges owned by nodes 0..=i.
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

#[test]
fn single_workgroup_converged_word_matches_oracle_below_and_above_diameter() {
    // 4-node reverse chain, diameter 3, single-workgroup path.
    let (offsets, targets, masks, seed) = reverse_chain(4);
    // Below diameter: still growing at the budget boundary, converged=false.
    let (changed, converged) =
        assert_device_matches_oracle(4, &offsets, &targets, &masks, &seed, 0xFFFF_FFFF, 2);
    assert_eq!((changed, converged), (1, false));
    // Above diameter: fixpoint reached with a confirming no-change step.
    let (changed, converged) =
        assert_device_matches_oracle(4, &offsets, &targets, &masks, &seed, 0xFFFF_FFFF, 8);
    assert_eq!((changed, converged), (1, true));
}

#[test]
fn single_workgroup_converged_is_false_when_full_set_reached_on_last_allowed_step() {
    // 5-node reverse chain, diameter 4. With max_iters=4 the final node is added
    // on the last allowed step, leaving no confirming step: converged must be
    // false even though the frontier is complete.
    let (offsets, targets, masks, seed) = reverse_chain(5);
    let (device_frontier, device_changed, device_converged) =
        run_device(5, &offsets, &targets, &masks, &seed, 0xFFFF_FFFF, 4);
    assert_eq!(device_frontier, vec![0b11111]);
    assert_eq!(device_changed, 1);
    assert_eq!(
        device_converged, 0,
        "Fix: a full set reached only on the last allowed step must report converged=0 on the device."
    );
    assert_device_matches_oracle(5, &offsets, &targets, &masks, &seed, 0xFFFF_FFFF, 4);
}

#[test]
fn single_workgroup_converged_is_true_when_seed_is_already_a_fixpoint() {
    // A 2-node chain 0 -> 1 seeded with both nodes already present: the first
    // step adds nothing, so the run converges immediately with changed=0.
    let offsets = [0u32, 1, 1];
    let targets = [1u32];
    let masks = [1u32];
    let seed = [0b11u32];
    let (device_frontier, device_changed, device_converged) =
        run_device(2, &offsets, &targets, &masks, &seed, 0xFFFF_FFFF, 4);
    assert_eq!(device_frontier, vec![0b11]);
    assert_eq!(device_changed, 0);
    assert_eq!(device_converged, 1);
    assert_device_matches_oracle(2, &offsets, &targets, &masks, &seed, 0xFFFF_FFFF, 4);
}

/// A two-level fan graph with `node_count > 256` to force the grid-sync path
/// while keeping the diameter at 2 so the interpreter stays cheap:
/// node 0 -> nodes 1..=fanout, node 1 -> the final leaf node.
fn grid_sync_two_level(fanout: u32) -> (u32, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let node_count = fanout + 2; // node 0, the fanout targets, and one leaf off node 1.
    let leaf = node_count - 1;
    let mut targets: Vec<u32> = (1..=fanout).collect();
    targets.push(leaf); // node 1 -> leaf
    // offsets: node 0 owns edges [0, fanout); node 1 owns edge [fanout, fanout+1);
    // every later node has no outgoing edge.
    let mut offsets = vec![0u32, fanout];
    for _ in 2..=node_count {
        offsets.push(fanout + 1);
    }
    let masks = vec![1u32; targets.len()];
    let words = bitset_words(node_count) as usize;
    let mut seed = vec![0u32; words];
    seed[0] = 1; // seed node 0
    (node_count, offsets, targets, masks, seed)
}

#[test]
fn grid_sync_converged_word_matches_oracle_across_the_budget_boundary() {
    // 258 nodes (> 256) forces the grid-sync program; diameter 2 keeps it cheap.
    let (node_count, offsets, targets, masks, seed) = grid_sync_two_level(256);
    assert!(node_count > 256, "must exercise the grid-sync path");

    // max_iters below the diameter: still growing at the boundary, converged=false.
    let (changed, converged) =
        assert_device_matches_oracle(node_count, &offsets, &targets, &masks, &seed, 0xFFFF_FFFF, 1);
    assert_eq!((changed, converged), (1, false));

    // max_iters == diameter: the full set is reached on the last allowed step, so
    // there is no confirming step and converged stays false.
    let (changed, converged) =
        assert_device_matches_oracle(node_count, &offsets, &targets, &masks, &seed, 0xFFFF_FFFF, 2);
    assert_eq!((changed, converged), (1, false));

    // Above the diameter: the confirming no-change step lands, converged=true.
    let (changed, converged) =
        assert_device_matches_oracle(node_count, &offsets, &targets, &masks, &seed, 0xFFFF_FFFF, 3);
    assert_eq!((changed, converged), (1, true));
}

#[test]
fn grid_sync_converged_word_matches_oracle_through_the_closure_split_entry() {
    // This is the exact production call path of a downstream dataflow consumer: a
    // host-loop fixpoint solver holds an opaque single-launch dispatch closure (not
    // a `&dyn VyreBackend`) and routes the grid-sync persistent_bfs program through
    // `dispatch_with_grid_sync_split_via`. The closure wraps `CpuRefBackend`; the
    // shared split core loops the segment sequence to a fixpoint and publishes
    // the converged word, which must match the oracle just as the backend entry
    // does. This proves the converged signal is correct on that closure entry.
    let (node_count, offsets, targets, masks, seed) = grid_sync_two_level(256);
    assert!(node_count > 256, "must exercise the grid-sync path");
    let edge_count = targets.len() as u32;
    let shape = ProgramGraphShape::new(node_count, edge_count.max(1));

    // The opaque closure a host-loop solver supplies: one kernel launch per
    // segment, grid from the caller, per-segment fixpoint fixed at 1 (the shared
    // split core owns the outer iteration count).
    let dispatch = |program: &Program,
                    inputs: &[&[u8]],
                    grid: Option<[u32; 3]>,
                    outputs: &mut Vec<Vec<u8>>|
     -> Result<(), String> {
        let mut config = DispatchConfig::default();
        config.grid_override = grid;
        config.fixpoint_iterations = Some(1);
        CpuRefBackend
            .dispatch_borrowed_into(program, inputs, &config, outputs)
            .map_err(|error| error.to_string())
    };

    for (max_iters, expect_converged) in [(1u32, 0u32), (2, 0), (3, 1)] {
        let program =
            persistent_bfs(shape, "frontier_in", "frontier_out", 0xFFFF_FFFF, max_iters);
        assert!(
            contains_grid_sync(&program),
            "258-node persistent_bfs must be a grid-sync program"
        );
        let inputs = build_inputs(&program, &offsets, &targets, &masks, &seed);
        let borrowed: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
        let outputs =
            dispatch_with_grid_sync_split_via(&program, &borrowed, &DispatchConfig::default(), &dispatch)
                .expect("Fix: persistent_bfs closure-split dispatch must succeed on a valid graph.");
        let converged = read_named_output(&program, &outputs, "converged")[0];

        let (_, oracle) =
            try_cpu_ref_converged(node_count, &offsets, &targets, &masks, &seed, 0xFFFF_FFFF, max_iters)
                .expect("Fix: CPU oracle must accept a valid graph.");
        assert_eq!(
            converged,
            expect_converged,
            "max_iters={max_iters}: closure-split converged word must equal the oracle (converged={}).",
            oracle.converged
        );
        assert_eq!(converged, u32::from(oracle.converged));
    }
}

/// Dispatch the BATCH persistent_bfs program and read back the per-query
/// frontier/changed/converged, exactly as [`run_device`] does for the single
/// program.
///
/// The batch program fans a `[256, 1, 1]` workgroup across `grid.y` (one query
/// per `grid.y` block, `q = gid_y()`), so the interpreter must be told the real
/// grid or it collapses to `grid.y == 1` and computes only query 0. Both paths
/// pass `persistent_bfs_batch_dispatch_grid`: the single-workgroup path
/// (`node_count <= 256`) through [`reference_eval_with_grid`], and the grid-sync
/// path (`node_count > 256`) through [`dispatch_with_grid_sync_split`] with the
/// grid on the config, which the split core clones onto every segment dispatch.
fn run_device_batch(
    node_count: u32,
    query_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
    max_iters: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let edge_count = edge_targets.len() as u32;
    let shape = ProgramGraphShape::new(node_count, edge_count.max(1));
    let program = persistent_bfs_batch(
        shape,
        "frontier_in",
        "frontier_out",
        "changed",
        "converged",
        query_count,
        allow_mask,
        max_iters,
    );
    let words = bitset_words(node_count) as usize;
    let total_words = words * query_count.max(1) as usize;
    let inputs = build_inputs(&program, edge_offsets, edge_targets, edge_kind_mask, frontier_in);
    let grid = persistent_bfs_batch_dispatch_grid(node_count, query_count);

    let outputs: Vec<Vec<u8>> = if contains_grid_sync(&program) {
        let borrowed: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
        let mut config = DispatchConfig::default();
        config.dispatch_grid = Some(grid);
        dispatch_with_grid_sync_split(&CpuRefBackend, &program, &borrowed, &config)
            .expect("Fix: persistent_bfs_batch grid-sync split dispatch must succeed on a valid graph.")
    } else {
        reference_eval_with_grid(
            &program,
            &inputs
                .iter()
                .map(|bytes| vyre_reference::value::Value::from(bytes.as_slice()))
                .collect::<Vec<_>>(),
            grid,
        )
        .expect("Fix: persistent_bfs_batch reference dispatch must succeed on a valid graph.")
        .into_iter()
        .map(|value| value.to_bytes())
        .collect()
    };
    let mut frontier_out = read_named_output(&program, &outputs, "frontier_out");
    frontier_out.truncate(total_words);
    let mut changed = read_named_output(&program, &outputs, "changed");
    changed.truncate(query_count as usize);
    let mut converged = read_named_output(&program, &outputs, "converged");
    converged.truncate(query_count as usize);
    (frontier_out, changed, converged)
}

/// Assert the batch device output matches the CPU oracle on EVERY query, where
/// each query is an independent single-frontier run of the same graph. Returns
/// the per-query `(changed, converged)` the oracle produced.
///
/// The queries are packed as a flat `[query][word]` frontier array, the exact
/// layout the batch program indexes with `base = q * words`. Distinct seeds with
/// distinct convergence outcomes at the same `max_iters` prove the `grid.y`
/// coverage is real: if the interpreter under-fired the query dimension, queries
/// past the first would read back their zeroed slot and diverge from the oracle.
fn assert_batch_device_matches_oracle(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    seeds: &[Vec<u32>],
    allow_mask: u32,
    max_iters: u32,
) -> Vec<(u32, bool)> {
    let words = bitset_words(node_count) as usize;
    let query_count = seeds.len() as u32;
    let mut frontier_in = Vec::with_capacity(words * seeds.len());
    for seed in seeds {
        assert_eq!(
            seed.len(),
            words,
            "each batch seed must be exactly one frontier bitset ({words} words)"
        );
        frontier_in.extend_from_slice(seed);
    }
    let (device_frontier, device_changed, device_converged) = run_device_batch(
        node_count,
        query_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        &frontier_in,
        allow_mask,
        max_iters,
    );

    let mut oracle_outcomes = Vec::with_capacity(seeds.len());
    for (query, seed) in seeds.iter().enumerate() {
        let (frontier, outcome) = try_cpu_ref_converged(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            seed,
            allow_mask,
            max_iters,
        )
        .expect("Fix: CPU oracle must accept a valid graph.");
        let start = query * words;
        let end = start + words;
        assert_eq!(
            &device_frontier[start..end],
            frontier.as_slice(),
            "query {query}: batch device frontier must equal the CPU oracle."
        );
        assert_eq!(
            device_changed[query], outcome.changed,
            "query {query}: batch device changed flag must equal the CPU oracle."
        );
        assert_eq!(
            device_converged[query],
            u32::from(outcome.converged),
            "query {query}: batch device converged word must equal the CPU oracle (converged={}).",
            outcome.converged
        );
        oracle_outcomes.push((outcome.changed, outcome.converged));
    }
    oracle_outcomes
}

#[test]
fn batch_single_workgroup_converged_matches_oracle_across_queries_and_budget() {
    // 4-node reverse chain (diameter 3), single-workgroup path. Three queries
    // with distinct seeds and distinct convergence outcomes at the same budget:
    // node 3 (needs 3 hops), node 1 (needs 1 hop), a saturated seed (immediate
    // fixpoint). If the interpreter collapsed `grid.y` to 1, queries 1 and 2
    // would read back their zeroed frontier and this would fail.
    let (offsets, targets, masks, _seed) = reverse_chain(4);
    let seeds = vec![vec![0b1000u32], vec![0b0010u32], vec![0b1111u32]];

    // Budget 2: node-3 query still growing (converged=false), node-1 query at
    // fixpoint with a confirming step (true), saturated query immediate fixpoint
    // with changed=0 (true).
    let outcomes =
        assert_batch_device_matches_oracle(4, &offsets, &targets, &masks, &seeds, 0xFFFF_FFFF, 2);
    assert_eq!(outcomes, vec![(1, false), (1, true), (0, true)]);

    // Budget 8 (above diameter): every query reaches a confirmed fixpoint.
    let outcomes =
        assert_batch_device_matches_oracle(4, &offsets, &targets, &masks, &seeds, 0xFFFF_FFFF, 8);
    assert_eq!(outcomes, vec![(1, true), (1, true), (0, true)]);
}

#[test]
fn batch_grid_sync_converged_matches_oracle_across_queries_and_budget() {
    // 258 nodes (> 256) forces the grid-sync batch program; diameter 2 keeps the
    // interpreter cheap. Two queries: the seeded root (grows two hops) and a query
    // seeded at the leaf only (an immediate fixpoint, changed=0).
    let (node_count, offsets, targets, masks, root_seed) = grid_sync_two_level(256);
    assert!(node_count > 256, "must exercise the grid-sync batch path");
    let words = bitset_words(node_count) as usize;
    let leaf = node_count - 1;
    let mut leaf_seed = vec![0u32; words];
    leaf_seed[(leaf / 32) as usize] = 1 << (leaf % 32);
    let seeds = vec![root_seed, leaf_seed];

    // Below the diameter: the root query is still growing (converged=false); the
    // leaf query is an immediate fixpoint (changed=0, converged=true).
    let outcomes = assert_batch_device_matches_oracle(
        node_count,
        &offsets,
        &targets,
        &masks,
        &seeds,
        0xFFFF_FFFF,
        1,
    );
    assert_eq!(outcomes, vec![(1, false), (0, true)]);

    // Above the diameter: the root query lands a confirming no-change step.
    let outcomes = assert_batch_device_matches_oracle(
        node_count,
        &offsets,
        &targets,
        &masks,
        &seeds,
        0xFFFF_FFFF,
        3,
    );
    assert_eq!(outcomes, vec![(1, true), (0, true)]);
}
