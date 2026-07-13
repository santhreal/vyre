//! GPU-IR vs CPU-ref parity for `graph::motif` (compile-time subgraph-motif
//! participation witness).
//!
//! The motif edges are baked into the generated Program as constants. One serial
//! invocation clears both outputs, scans the CSR graph for each motif edge
//! (`edge.from -> edge.to` with `kind & edge.kind_mask != 0`), counts matched
//! edges, marks each motif endpoint in `motif_hits`, and ONLY IF ALL motif edges
//! are present materializes the endpoint union into `witness_out`. The op is a
//! SINGLE dispatch round, so `reference_eval` models it faithfully; because the
//! kernel writes deterministic values from a full re-scan, an over-fired grid is
//! idempotent (every lane writes identical bytes). Every shipped test drives only
//! the CPU oracle or checks Program shape; the actual all-edges-present gate +
//! endpoint materialization IR was never executed. A missing `matched_edges ==
//! edge_count` gate (partial motif spuriously published), a swapped endpoint
//! store, or a broken kind-mask compare all diverge here.
#![forbid(unsafe_code)]
#![cfg(all(feature = "graph", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_primitives::graph::motif::{cpu_ref, motif, MotifEdge};
use vyre_primitives::graph::program_graph::ProgramGraphShape;
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

/// Drive the real motif IR and return the `witness_out` byte-per-node array.
///
/// Buffer binding order (all non-workgroup): pg_nodes(0), pg_edge_offsets(1),
/// pg_edge_targets(2), pg_edge_kind_mask(3), pg_node_tags(4), motif_hits(5, RW
/// scratch), witness_out(6, RW). reference_eval returns the two ReadWrite buffers
/// in binding order, so outputs[1] is witness_out.
fn gpu_witness(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
) -> Vec<u32> {
    let edge_count = *edge_offsets
        .last()
        .expect("offsets has node_count+1 entries");
    let program = motif(
        ProgramGraphShape::new(node_count, edge_count),
        motif_edges,
        "witness_out",
    );
    let padded_edges = edge_count.max(1) as usize;
    let mut targets = edge_targets.to_vec();
    targets.resize(padded_edges, 0);
    let mut kind_mask = edge_kind_mask.to_vec();
    kind_mask.resize(padded_edges, 0);
    let node_tags = vec![0u32; node_count as usize];
    let nodes = vec![0u32; node_count as usize];
    let out_slots = node_count.max(1) as usize;

    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(&nodes)),
            Value::from(pack(edge_offsets)),
            Value::from(pack(&targets)),
            Value::from(pack(&kind_mask)),
            Value::from(pack(&node_tags)),
            Value::from(pack(&vec![0u32; out_slots])), // motif_hits scratch
            Value::from(pack(&vec![0u32; out_slots])), // witness_out
        ],
    )
    .expect("motif reference evaluation must succeed");
    unpack(&outputs[1].to_bytes())
}

/// Build a small random graph and a motif that is present about half the time
/// (mix of real graph edges and possibly-absent random edges).
fn generated_case(seed: u64) -> (u32, Vec<u32>, Vec<u32>, Vec<u32>, Vec<MotifEdge>) {
    let mut rng = seed;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 32) as u32
    };
    let node_count = 3 + next() % 6; // 3..=8
    let mut offsets = Vec::with_capacity(node_count as usize + 1);
    let mut targets = Vec::new();
    let mut kind_mask = Vec::new();
    // Collect real edges so a motif can be assembled from present ones.
    let mut real_edges: Vec<(u32, u32, u32)> = Vec::new();
    offsets.push(0u32);
    for from in 0..node_count {
        let degree = next() % 4;
        for _ in 0..degree {
            let to = next() % node_count;
            let kind = 1u32 << (next() % 4);
            targets.push(to);
            kind_mask.push(kind);
            real_edges.push((from, to, kind));
        }
        offsets.push(targets.len() as u32);
    }
    let motif_len = 1 + next() % 3; // 1..=3 motif edges
    let mut motif_edges = Vec::new();
    for _ in 0..motif_len {
        // 60% chance: reuse a real present edge (kind_mask = its exact kind, so it
        // matches); 40%: a random edge that may be absent (drives the partial /
        // no-match branch).
        if !real_edges.is_empty() && next() % 5 < 3 {
            let (from, to, kind) = real_edges[(next() as usize) % real_edges.len()];
            motif_edges.push(MotifEdge {
                from,
                kind_mask: kind,
                to,
            });
        } else {
            motif_edges.push(MotifEdge {
                from: next() % node_count,
                kind_mask: 1u32 << (next() % 4),
                to: next() % node_count,
            });
        }
    }
    (node_count, offsets, targets, kind_mask, motif_edges)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn ir_matches_cpu_ref_over_random_motifs(seed in any::<u64>()) {
        let (node_count, offsets, targets, kind_mask, motif_edges) = generated_case(seed);
        let expected = cpu_ref(node_count, &offsets, &targets, &kind_mask, &motif_edges);
        let got = gpu_witness(node_count, &offsets, &targets, &kind_mask, &motif_edges);
        prop_assert_eq!(
            got, expected,
            "motif witness IR diverged from cpu_ref: node_count={}, offsets={:?}, targets={:?}, motif={:?}",
            node_count, offsets, targets, motif_edges
        );
    }
}

/// Deterministic anchors: a fully-present 2-edge path motif (witness = its
/// endpoints), the SAME motif with one edge absent (witness must be EMPTY, the
/// all-edges-present gate), and a single-edge motif.
#[test]
fn ir_matches_cpu_ref_on_present_and_partial_motifs() {
    // Graph: 0 -(k1)-> 1 -(k1)-> 2. CSR offsets [0,1,2,2], targets [1,2],
    // kind_mask [1,1].
    let node_count = 3u32;
    let offsets = vec![0u32, 1, 2, 2];
    let targets = vec![1u32, 2];
    let kind_mask = vec![1u32, 1];

    // Present 2-edge path motif {0->1, 1->2}: all present -> witness {0,1,2}.
    let present = vec![
        MotifEdge {
            from: 0,
            kind_mask: 1,
            to: 1,
        },
        MotifEdge {
            from: 1,
            kind_mask: 1,
            to: 2,
        },
    ];
    let expected = cpu_ref(node_count, &offsets, &targets, &kind_mask, &present);
    assert_eq!(
        expected,
        vec![1u32, 1, 1],
        "cpu_ref: all endpoints participate"
    );
    assert_eq!(
        gpu_witness(node_count, &offsets, &targets, &kind_mask, &present),
        expected,
        "fully-present motif witness must match"
    );

    // Partial: {0->1, 1->2, 2->0}; the 2->0 edge does NOT exist -> matched_edges
    // (2) != edge_count (3) -> witness must stay ALL ZERO.
    let partial = vec![
        MotifEdge {
            from: 0,
            kind_mask: 1,
            to: 1,
        },
        MotifEdge {
            from: 1,
            kind_mask: 1,
            to: 2,
        },
        MotifEdge {
            from: 2,
            kind_mask: 1,
            to: 0,
        },
    ];
    let empty = cpu_ref(node_count, &offsets, &targets, &kind_mask, &partial);
    assert_eq!(
        empty,
        vec![0u32, 0, 0],
        "cpu_ref: missing edge -> no publish"
    );
    assert_eq!(
        gpu_witness(node_count, &offsets, &targets, &kind_mask, &partial),
        empty,
        "partial motif must NOT publish a witness (all-edges-present gate)"
    );

    // Kind-mask mismatch: motif edge 0->1 demands kind bit 2, but the graph edge
    // is kind bit 1 -> absent -> empty.
    let wrong_kind = vec![MotifEdge {
        from: 0,
        kind_mask: 1 << 1,
        to: 1,
    }];
    let empty2 = cpu_ref(node_count, &offsets, &targets, &kind_mask, &wrong_kind);
    assert_eq!(empty2, vec![0u32, 0, 0], "cpu_ref: kind mismatch -> absent");
    assert_eq!(
        gpu_witness(node_count, &offsets, &targets, &kind_mask, &wrong_kind),
        empty2,
        "kind-mask mismatch must drop the motif edge in IR too"
    );

    // Single present edge motif {0->1}: witness {0,1}.
    let single = vec![MotifEdge {
        from: 0,
        kind_mask: 1,
        to: 1,
    }];
    let one = cpu_ref(node_count, &offsets, &targets, &kind_mask, &single);
    assert_eq!(one, vec![1u32, 1, 0], "cpu_ref: single-edge endpoints");
    assert_eq!(
        gpu_witness(node_count, &offsets, &targets, &kind_mask, &single),
        one,
        "single-edge motif witness must match"
    );
}
