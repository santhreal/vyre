//! Connectivity parity for the lock-free `graph::union_find` merge program.
//!
//! The registry fixture proves the program merges one hand-checked 4-node graph; this
//! proves the actual lock-free union-with-path-halving IR, run through the reference
//! interpreter from an identity parent over many random edge batches, computes the CORRECT
//! connected components at scale. The program uses ordered root selection (the lower-index
//! root always wins), so `find(x)` converges to the MINIMUM index in x's component, a
//! unique, order-independent representative that an independent union-find oracle reproduces.
//!
//! TWO harness contracts make this real (both were missing before, producing a FALSE
//! "union_find has no connectivity closure" defect):
//!
//! 1. GRID FLOOR: union_find fires one lane PER EDGE (`if lane < edge_count`), but the
//!    reference infers its dispatch grid from buffer SHAPES. When `edge_count > node_count`
//!    the buffer-inferred grid UNDER-fires and silently drops the high-index edges, the hole
//!    `reference_eval_with_dispatch`'s `min_dispatch_elements` floor closes. Passing
//!    `edge_count` as the floor fires every edge lane.
//! 2. RE-DISPATCH TO A FIXPOINT: the reference arena executor models GPU parallelism 
//!    each invocation reads the parent buffer as of the dispatch START (snapshot), only the
//!    atomic RMW (CAS / path-halving `min`) is live. So a SINGLE dispatch of a parallel
//!    union-find does NOT close a general graph: concurrent unions race and only partially
//!    apply, exactly as on real hardware. The contract is the CONVERGED state, feed each
//!    pass's parent output back as the next seed until it stabilizes. Parent values only ever
//!    decrease (ordered CAS + path-halving `min`), so the iteration is monotone and reaches
//!    full closure within `node_count` passes. This is genuine iterative connected-components,
//!    not an algorithm defect.
#![cfg(feature = "graph")]

use vyre_foundation::ir::Program;
use vyre_primitives::graph::union_find::union_find_program;
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

fn next_u32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn out_by_name(program: &Program, outputs: &[Value], name: &str) -> Vec<u32> {
    let index = vyre_reference::output_index(program, name)
        .unwrap_or_else(|| panic!("Fix: union_find program must declare output `{name}`"));
    unpack(&outputs[index].to_bytes())
}

/// Walk `parent` from `node` to a fixed point (a root points to itself). Bounded by
/// `node_count` so a malformed parent array cannot spin.
fn find(parent: &[u32], mut node: u32, node_count: u32) -> u32 {
    for _ in 0..node_count + 1 {
        let p = parent[node as usize];
        if p == node {
            break;
        }
        node = p;
    }
    node
}

/// Independent oracle: component representative = minimum node index reachable through the
/// undirected edge set. Built by a plain iterate-to-stable relaxation over the edge list 
/// a wholly different structure from the on-device path-halving CAS union.
fn component_min(node_count: u32, edge_a: &[u32], edge_b: &[u32]) -> Vec<u32> {
    let mut rep: Vec<u32> = (0..node_count).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for (&a, &b) in edge_a.iter().zip(edge_b.iter()) {
            let m = rep[a as usize].min(rep[b as usize]);
            if rep[a as usize] != m {
                rep[a as usize] = m;
                changed = true;
            }
            if rep[b as usize] != m {
                rep[b as usize] = m;
                changed = true;
            }
        }
    }
    // Collapse each node to its ultimate representative (rep chains can be more than one deep).
    for _ in 0..node_count + 1 {
        let mut moved = false;
        for i in 0..node_count as usize {
            let r = rep[rep[i] as usize];
            if r != rep[i] {
                rep[i] = r;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    rep
}

/// Drive union_find to its converged connectivity: fire every edge lane (the `edge_count`
/// dispatch floor) each pass and feed the parent output back as the next seed until it
/// stabilizes. Snapshot-parallel execution needs re-dispatch; the monotone decrease of parent
/// values bounds convergence by `node_count` passes (one extra as a safety margin).
fn union_find_closure(
    program: &Program,
    node_count: u32,
    edge_count: u32,
    edge_a: &[u32],
    edge_b: &[u32],
) -> Vec<u32> {
    let mut parent: Vec<u32> = (0..node_count).collect();
    for _ in 0..node_count + 1 {
        let outputs = vyre_reference::reference_eval_with_dispatch(
            program,
            &[
                Value::from(pack(&parent)),
                Value::from(pack(edge_a)),
                Value::from(pack(edge_b)),
            ],
            edge_count,
        )
        .expect("union_find reference evaluation must succeed");
        let next = out_by_name(program, &outputs, "parent");
        if next == parent {
            break;
        }
        parent = next;
    }
    parent
}

/// Sharp regression for the `find_root_body` walk bug: seed a parent where one endpoint sits
/// UNDER a multi-hop root, so the union only fires if `find` actually walks the chain. Nodes
/// 1,2,3 are all children of root 3; node 0 is a separate root. Edge `(1,0)` merges the two
/// components ONLY if `find(1)` resolves to 3 (then `CAS(parent[3]=0)` succeeds). A broken
/// find that returns the raw endpoint computes `find(1)=1` and `CAS(parent[1])` fails (live
/// `parent[1]=3`), leaving the graph permanently split (exactly the shipped defect).
#[test]
fn find_root_body_walks_multi_hop_root() {
    let node_count = 4u32;
    let edge_a = vec![1u32];
    let edge_b = vec![0u32];
    let edge_count = 1u32;
    let program = union_find_program("parent", "edge_a", "edge_b", node_count, edge_count);
    // Seed a pre-built tree: {1,2,3} under root 3, {0} alone. NOT the identity, so it directly
    // exercises multi-hop find (union_find_closure would reseed to identity, so call directly).
    let seed = vec![0u32, 3, 3, 3];
    let mut parent = seed;
    for _ in 0..node_count + 1 {
        let outputs = vyre_reference::reference_eval_with_dispatch(
            &program,
            &[
                Value::from(pack(&parent)),
                Value::from(pack(&edge_a)),
                Value::from(pack(&edge_b)),
            ],
            edge_count,
        )
        .expect("union_find reference evaluation must succeed");
        let next = out_by_name(&program, &outputs, "parent");
        if next == parent {
            break;
        }
        parent = next;
    }
    for node in 0..node_count {
        assert_eq!(
            find(&parent, node, node_count),
            0,
            "node {node} not merged into root 0, find failed to walk the multi-hop chain \
             (parent={parent:?})"
        );
    }
}

#[test]
fn ir_union_find_connectivity_matches_component_min_oracle() {
    // 127 diverse random graphs (node_count 2..39, edge_count up to 2*node_count), a
    // systematic union/find defect surfaces in the first handful of seeds (the shipped
    // find-walk bug failed at seed 4), so this is a strong differential while staying fast
    // enough for the feature-gated conform step (each seed re-dispatches to a fixpoint, so
    // per-seed cost is O(passes * reference_eval); the two O(1) sharp tests above pin the
    // specific find-walk + grid-floor cases deterministically).
    for seed in 1..128u32 {
        let node_count = 2 + (seed % 38);
        let mut state = seed ^ 0x51F0_1234;
        let edge_count = 1 + (next_u32(&mut state) % (node_count * 2));
        let mut edge_a = Vec::with_capacity(edge_count as usize);
        let mut edge_b = Vec::with_capacity(edge_count as usize);
        for _ in 0..edge_count {
            edge_a.push(next_u32(&mut state) % node_count);
            edge_b.push(next_u32(&mut state) % node_count);
        }

        let program = union_find_program("parent", "edge_a", "edge_b", node_count, edge_count);
        let parent_out = union_find_closure(&program, node_count, edge_count, &edge_a, &edge_b);

        let oracle = component_min(node_count, &edge_a, &edge_b);
        for node in 0..node_count {
            assert_eq!(
                find(&parent_out, node, node_count),
                oracle[node as usize],
                "seed {seed}: node {node} find-root diverged from component-min oracle \
                 (node_count={node_count}, edges={edge_count})"
            );
        }
    }
}

/// The batch is edge-count-dominated (edge_count can exceed node_count): assert the harness
/// actually fires every edge lane by picking a shape where the wrong (buffer-shape) grid
/// would drop edges. Seed-10's dropped lane 12 was the bridging edge `(2,0)`; here the last
/// edge is the ONLY thing connecting node `n-1` to root 0, so a short grid leaves it split.
#[test]
fn full_edge_grid_processes_trailing_bridge_edge() {
    let node_count = 6u32;
    // Fill edges 0..(node_count) with self-loops (no-ops), then a trailing chain that only
    // closes if the final, highest-index edge lane fires. edge_count (2*node_count) > node_count.
    let mut edge_a = vec![0u32; node_count as usize]; // self-loops on node 0
    let mut edge_b = vec![0u32; node_count as usize];
    // Chain 0-1-2-3-4-5 appended AFTER the self-loops, so the bridges live at high lane ids.
    for k in 0..node_count - 1 {
        edge_a.push(k);
        edge_b.push(k + 1);
    }
    let edge_count = edge_a.len() as u32;
    assert!(
        edge_count > node_count,
        "shape must be edge-dominated to exercise the grid floor"
    );

    let program = union_find_program("parent", "edge_a", "edge_b", node_count, edge_count);
    let parent_out = union_find_closure(&program, node_count, edge_count, &edge_a, &edge_b);
    // Every node collapses to root 0 only if the trailing high-lane chain edges all fired.
    for node in 0..node_count {
        assert_eq!(
            find(&parent_out, node, node_count),
            0,
            "node {node} not merged into root 0, a trailing edge lane was dropped"
        );
    }
}
