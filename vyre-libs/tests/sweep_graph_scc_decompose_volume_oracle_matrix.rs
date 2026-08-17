//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]
#![cfg(feature = "graph")]
mod graph_sweep_fixtures;
use graph_sweep_fixtures::bitset_words;
#[path = "../../tests/support/csr_sweep/mod.rs"]
mod csr_sweep;

use vyre_libs::graph::scc_decompose;
use vyre_reference::composition_witness::scc_decompose_witness;

fn generated_scc_case(seed: u64) -> (u32, Vec<u32>, Vec<u32>, Vec<u32>, u32) {
    let mut rng = csr_sweep::Rng::new((seed) | 1);
    const BOUNDARY_SHAPES: [u32; 27] = [
        0, 1, 2, 31, 32, 33, 63, 64, 65, 95, 96, 127, 128, 129, 255, 256, 257, 300, 511, 512, 513,
        1023, 1024, 1025, 1535, 1536, 1537,
    ];
    let node_count = if rng.next_u32() % 4 == 0 {
        BOUNDARY_SHAPES[(rng.next_u32() as usize) % BOUNDARY_SHAPES.len()]
    } else {
        rng.next_u32() % 2048
    };
    let words = bitset_words(node_count);
    let mut forward = Vec::with_capacity(words);
    let mut backward = Vec::with_capacity(words);
    for _ in 0..words {
        forward.push(rng.next_u32());
        backward.push(rng.next_u32());
    }

    let tail_bits = node_count % 32;
    if tail_bits != 0 && words != 0 {
        let tail_mask = (1u32 << tail_bits) - 1;
        forward[words - 1] &= tail_mask;
        backward[words - 1] &= tail_mask;
    }

    let mut component_in = Vec::with_capacity(node_count as usize);
    for node in 0..node_count {
        let assigned = rng.next_u32() % 7 == 0;
        component_in.push(if assigned {
            rng.next_u32().wrapping_add(node) & 0x7FFF_FFFF
        } else {
            u32::MAX
        });
    }
    let pivot = rng.next_u32();
    (node_count, forward, backward, component_in, pivot)
}

fn oracle_scc(
    node_count: u32,
    forward: &[u32],
    backward: &[u32],
    component_in: &[u32],
    pivot: u32,
) -> Vec<u32> {
    let mut out = component_in.to_vec();
    for v in 0..node_count {
        let word = (v / 32) as usize;
        let bit = 1u32 << (v % 32);
        let fwd = word < forward.len() && forward[word] & bit != 0;
        let bwd = word < backward.len() && backward[word] & bit != 0;
        if fwd && bwd && out[v as usize] == u32::MAX {
            out[v as usize] = pivot;
        }
    }
    out
}

const CASES: usize = 32768;

fn scc_decompose_dispatch_grid(node_count: u32) -> [u32; 3] {
    vyre_primitives::lane_grid(node_count, scc_decompose::SCC_DECOMPOSE_WORKGROUP_SIZE[0])
}

#[test]
fn sweep_graph_scc_decompose_volume_oracle_matrix() {
    for case in 0..CASES {
        let seed = case as u64 ^ 0x5CCDEC0D;
        let (node_count, forward, backward, component_in, pivot) = generated_scc_case(seed);
        let expected = oracle_scc(node_count, &forward, &backward, &component_in, pivot);
        let actual = scc_decompose_witness(node_count, &forward, &backward, &component_in, pivot);
        assert_eq!(actual, expected, "Fix: scc_decompose volume case {case}");

        let grid = scc_decompose_dispatch_grid(node_count);
        assert_eq!(
            grid[1], 1,
            "Fix: SCC grid y dimension drifted at case {case}"
        );
        assert_eq!(
            grid[2], 1,
            "Fix: SCC grid z dimension drifted at case {case}"
        );
        assert!(
            grid[0] >= 1,
            "Fix: SCC dispatch grid must keep an empty graph launchable at case {case}"
        );
        assert!(
            grid[0] * scc_decompose::SCC_DECOMPOSE_WORKGROUP_SIZE[0] >= node_count.max(1),
            "Fix: SCC dispatch grid under-covers node_count={node_count} at case {case}"
        );
    }
}
