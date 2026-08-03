//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]
#![cfg(all(feature = "graph", feature = "cpu-parity"))]
mod graph_sweep_support;
use graph_sweep_support::next_u32;

use std::collections::{HashSet, VecDeque};

use vyre_primitives::graph::reachable::reachable;

fn generated_edges(seed: u64, node_count: u32) -> Vec<(u32, u32)> {
    let mut rng = seed;
    let n = node_count.max(1);
    let edge_count = 1 + (next_u32(&mut rng) % 32) as usize;
    (0..edge_count)
        .map(|_| (next_u32(&mut rng) % n, next_u32(&mut rng) % n))
        .collect()
}

fn oracle_reachable(node_count: u32, edges: &[(u32, u32)], sources: &[u32]) -> HashSet<u32> {
    let mut seen = HashSet::new();
    let mut queue = VecDeque::new();
    for &s in sources {
        if s < node_count {
            seen.insert(s);
            queue.push_back(s);
        }
    }
    while let Some(u) = queue.pop_front() {
        for &(from, to) in edges {
            if from == u && to < node_count && seen.insert(to) {
                queue.push_back(to);
            }
        }
    }
    seen
}

const CASES: usize = 16384;

#[test]
fn sweep_graph_reachable_volume_oracle_matrix() {
    for case in 0..CASES {
        let seed = case as u64 ^ 0xAEAC4AB1E;
        let mut rng = seed;
        let node_count = 2 + next_u32(&mut rng) % 48;
        let edges = generated_edges(seed.rotate_left(9), node_count);
        let source = next_u32(&mut rng) % node_count;
        let sources = vec![source];
        let expected = oracle_reachable(node_count, &edges, &sources);
        let actual = reachable(node_count, &edges, &sources)
            .expect("Fix: generated reachable volume inputs must be in-range");
        assert_eq!(actual, expected, "Fix: reachable volume case {case}");
    }
}
