//! Value parity for `graph::toposort::toposort_program`: the actual IR PROGRAM, run through
//! `reference_eval`, must compute a correct topological order, not just emit valid IR.
//!
//! `toposort_program` is a single-invocation (`[1,1,1]`, lane0-gated) Kahn's algorithm: it
//! zeroes indegrees, counts them from the CSR edge list, seeds a FIFO queue with every
//! indegree-0 node in ASCENDING id order, then pops/decrements/pushes until the queue drains,
//! writing the pop order into `order_out`. Being single-invocation there is no snapshot/
//! re-dispatch subtlety (unlike the parallel union_find), one dispatch is the whole
//! algorithm, so `order_out` compares directly to an independent oracle.
//!
//! WHY THIS EXISTS: toposort_program is unregistered (order-dependent output slots can't join
//! the race-net registry) and its only prior coverage was the CPU `toposort` function + an
//! IR-validity check, the actual IR was NEVER run through `reference_eval` for VALUE. That is
//! the exact gap the union_find `find`-never-walks bug fell through: valid IR, wrong behavior,
//! CPU-oracle-covered, IR-value-unchecked. This differential closes it: an independent Kahn
//! implementation that MIRRORS the IR's exact policy (ascending indeg-0 seed, FIFO pop, CSR
//! successor order) reproduces the order byte-for-byte, so any deviation (a loop that never
//! advances, a miscounted indegree, a dropped push) fails loudly.
#![cfg(feature = "graph")]

use vyre_foundation::ir::Program;
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

fn next_u32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Independent Kahn's toposort mirroring `toposort_program`'s exact deterministic policy:
/// ascending indegree-0 seed, FIFO pop, successors walked in CSR order. Returns the pop order.
fn kahn_fifo(node_count: u32, offsets: &[u32], targets: &[u32]) -> Vec<u32> {
    let n = node_count as usize;
    let mut indeg = vec![0u32; n];
    for &t in targets {
        indeg[t as usize] += 1;
    }
    let mut queue: Vec<u32> = (0..node_count)
        .filter(|&v| indeg[v as usize] == 0)
        .collect();
    let mut order = Vec::with_capacity(n);
    let mut read = 0usize;
    while read < queue.len() {
        let v = queue[read];
        read += 1;
        order.push(v);
        for e in offsets[v as usize]..offsets[v as usize + 1] {
            let u = targets[e as usize] as usize;
            indeg[u] -= 1;
            if indeg[u] == 0 {
                queue.push(u as u32);
            }
        }
    }
    order
}

/// Build a random DAG's CSR (edges only from lower to higher id => guaranteed acyclic, so the
/// toposort always consumes every node and `order_out[0..n]` is fully written).
fn generated_dag(seed: u32) -> (u32, Vec<u32>, Vec<u32>) {
    let node_count = 2 + (seed % 30);
    let mut state = seed ^ 0x2468_ACE1;
    // adjacency[u] = successors v>u
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); node_count as usize];
    for u in 0..node_count {
        for v in (u + 1)..node_count {
            // ~35% edge density, biased so most graphs have real structure.
            if next_u32(&mut state) % 100 < 35 {
                adjacency[u as usize].push(v);
            }
        }
    }
    let mut offsets = vec![0u32];
    let mut targets = Vec::new();
    for u in 0..node_count as usize {
        for &v in &adjacency[u] {
            targets.push(v);
        }
        offsets.push(targets.len() as u32);
    }
    (node_count, offsets, targets)
}

fn run_toposort_program(node_count: u32, offsets: &[u32], targets: &[u32]) -> Vec<u32> {
    let program: Program = vyre_primitives::graph::toposort::toposort_program(
        node_count, "offsets", "targets", "indeg", "queue", "order",
    );
    let zeros = vec![0u32; node_count.max(1) as usize];
    // Input order = buffer declaration order: offsets(0), targets(1), indeg(2), queue(3),
    // order(4). indeg/queue/order are ReadWrite scratch/output, seeded to zero.
    let targets_in = if targets.is_empty() {
        vec![0u32]
    } else {
        targets.to_vec()
    };
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(offsets)),
            Value::from(pack(&targets_in)),
            Value::from(pack(&zeros)),
            Value::from(pack(&zeros)),
            Value::from(pack(&zeros)),
        ],
    )
    .expect("toposort_program reference evaluation must succeed");
    let index = vyre_reference::output_index(&program, "order")
        .expect("Fix: toposort_program must declare output `order`");
    let full = unpack(&outputs[index].to_bytes());
    full[..node_count as usize].to_vec()
}

fn assert_valid_topo(
    node_count: u32,
    offsets: &[u32],
    targets: &[u32],
    order: &[u32],
    label: &str,
) {
    // Permutation of 0..node_count.
    let mut seen = vec![false; node_count as usize];
    for &node in order {
        assert!(
            node < node_count,
            "{label}: order has out-of-range node {node}"
        );
        assert!(
            !seen[node as usize],
            "{label}: node {node} appears twice in the order"
        );
        seen[node as usize] = true;
    }
    assert!(
        seen.iter().all(|&s| s),
        "{label}: order is not a full permutation ({order:?})"
    );
    // Every edge u->v must have position(u) < position(v).
    let mut pos = vec![0u32; node_count as usize];
    for (i, &node) in order.iter().enumerate() {
        pos[node as usize] = i as u32;
    }
    for u in 0..node_count as usize {
        for e in offsets[u]..offsets[u + 1] {
            let v = targets[e as usize] as usize;
            assert!(
                pos[u] < pos[v],
                "{label}: edge {u}->{v} violated, pos[{u}]={} !< pos[{v}]={} (order={order:?})",
                pos[u],
                pos[v]
            );
        }
    }
}

#[test]
fn toposort_program_matches_independent_kahn_over_generated_dags() {
    for seed in 1..320u32 {
        let (node_count, offsets, targets) = generated_dag(seed);
        let order = run_toposort_program(node_count, &offsets, &targets);
        let oracle = kahn_fifo(node_count, &offsets, &targets);
        assert_eq!(
            order,
            oracle,
            "seed {seed}: toposort_program order diverged from the independent FIFO-Kahn oracle \
             (node_count={node_count}, edges={})",
            targets.len()
        );
        // Independent semantic check on top of the exact-match: it is a valid topological order.
        assert_valid_topo(
            node_count,
            &offsets,
            &targets,
            &order,
            &format!("seed {seed}"),
        );
    }
}

#[test]
fn toposort_program_orders_hand_checked_shapes() {
    // Diamond 0->1, 0->2, 1->3, 2->3: only valid FIFO order is [0,1,2,3].
    let offsets = vec![0u32, 2, 3, 4, 4];
    let targets = vec![1u32, 2, 3, 3];
    let order = run_toposort_program(4, &offsets, &targets);
    assert_eq!(
        order,
        vec![0, 1, 2, 3],
        "diamond DAG must FIFO-sort to [0,1,2,3]"
    );

    // Linear chain 0->1->2->3->4: unique order [0,1,2,3,4].
    let offsets = vec![0u32, 1, 2, 3, 4, 4];
    let targets = vec![1u32, 2, 3, 4];
    let order = run_toposort_program(5, &offsets, &targets);
    assert_eq!(
        order,
        vec![0, 1, 2, 3, 4],
        "chain DAG must sort to [0,1,2,3,4]"
    );

    // Two disconnected chains 0->2 and 1->3: FIFO seeds {0,1} ascending -> [0,1,2,3].
    let offsets = vec![0u32, 1, 2, 2, 2];
    let targets = vec![2u32, 3];
    let order = run_toposort_program(4, &offsets, &targets);
    assert_valid_topo(4, &offsets, &targets, &order, "two-chains");
    assert_eq!(order, vec![0, 1, 2, 3], "disconnected chains FIFO order");
}
