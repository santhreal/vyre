//! Independent-source contract matrix for the `graph::motif` CSR reference.
//!
//! The oracle here never scans a CSR row. It first expands the CSR into an
//! edge-keyed dictionary of or-ed kind masks, which is a different data
//! structure reached by a different traversal, and then answers the three
//! questions the primitive answers directly from the motif specification: a
//! motif matches when every one of its edges is present with an overlapping
//! kind mask, the witness is the endpoint set of a matched motif, and the
//! participation count is the size of that set.
//!
//! Volume testing.volume - do NOT weaken to shape-only asserts.

#![forbid(unsafe_code)]
#![cfg(feature = "cpu-parity")]

use std::collections::{BTreeMap, BTreeSet};

use vyre_primitives::graph::motif::{self, count_witness_participants, MotifEdge};

#[path = "../../tests/support/csr_sweep/mod.rs"]
mod csr_sweep;

#[test]
fn motif_csr_matches_independent_witness_oracle_matrix() {
    for (case, shape) in csr_sweep::cases(
        "topology_only_all_kinds",
        8192,
        0xA07F_CAFE_0000_0000,
        0x9E37_79B9_7F4A_7C15,
    ) {
        let seed = 0xA07F_CAFE_0000_0000u64 ^ case.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let csr_sweep::CsrSweepCase {
            node_count,
            offsets,
            targets,
            masks,
            ..
        } = shape;
        let motif_edges = generated_motif_edges(seed.rotate_left(13), node_count);
        let adjacency = edge_mask_dictionary(&offsets, &targets, &masks);

        let expected_matches = spec_motif_matches(&adjacency, &motif_edges);
        assert_eq!(
            motif::cpu_ref_matches(&offsets, &targets, &masks, &motif_edges),
            expected_matches,
            "Fix: motif cpu_ref_matches oracle case {case} must agree with the edge-dictionary existence rule."
        );

        let endpoints = spec_matched_endpoints(node_count, &adjacency, &motif_edges);
        let expected = spec_witness(node_count, &endpoints);
        let actual = motif::cpu_ref(node_count, &offsets, &targets, &masks, &motif_edges);
        assert_eq!(
            actual, expected,
            "Fix: motif cpu_ref oracle case {case} node_count={node_count} must mark exactly the endpoint set of a matched motif."
        );

        let expected_count = endpoints.len() as u32;
        assert_eq!(
            motif::cpu_ref_participation_count(
                node_count,
                &offsets,
                &targets,
                &masks,
                &motif_edges
            ),
            expected_count,
            "Fix: motif participation count oracle case {case} must equal the number of distinct matched endpoints."
        );

        let mut reused = vec![0xCAFE_BABE; node_count as usize + 4];
        motif::cpu_ref_into(
            node_count,
            &offsets,
            &targets,
            &masks,
            &motif_edges,
            &mut reused,
        );
        assert_eq!(
            reused, expected,
            "Fix: motif cpu_ref_into oracle case {case} must clear stale witness capacity before writing."
        );

        let witness_count = count_witness_participants(&actual)
            .expect("Fix: generated motif witness must satisfy the boolean contract.");
        assert_eq!(
            witness_count, expected_count,
            "Fix: motif witness participant count oracle case {case} must agree with participation count."
        );
    }
}

/// Expand a CSR into `(from, to) -> or-ed kind mask`.
///
/// Row ranges are read once here, up front, so every later question is a
/// dictionary lookup instead of a row scan. Or-ing the masks of parallel edges
/// is lossless for the question asked of it: some parallel edge overlaps a
/// requested kind mask exactly when the or of their masks does.
fn edge_mask_dictionary(
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
) -> BTreeMap<(u32, u32), u32> {
    let reachable = edge_targets.len().min(edge_kind_mask.len());
    let mut adjacency = BTreeMap::new();
    for (from, bounds) in edge_offsets.windows(2).enumerate() {
        let start = bounds[0] as usize;
        let end = (bounds[1] as usize).min(reachable);
        for edge in start..end {
            *adjacency
                .entry((from as u32, edge_targets[edge]))
                .or_insert(0) |= edge_kind_mask[edge];
        }
    }
    adjacency
}

/// A motif matches when every motif edge exists with an overlapping kind mask.
fn spec_motif_matches(adjacency: &BTreeMap<(u32, u32), u32>, motif_edges: &[MotifEdge]) -> bool {
    motif_edges.iter().all(|edge| {
        adjacency
            .get(&(edge.from, edge.to))
            .is_some_and(|present| present & edge.kind_mask != 0)
    })
}

/// Distinct in-range endpoints of a matched motif; empty when it does not match.
fn spec_matched_endpoints(
    node_count: u32,
    adjacency: &BTreeMap<(u32, u32), u32>,
    motif_edges: &[MotifEdge],
) -> BTreeSet<u32> {
    if !spec_motif_matches(adjacency, motif_edges) {
        return BTreeSet::new();
    }
    motif_edges
        .iter()
        .flat_map(|edge| [edge.from, edge.to])
        .filter(|node| *node < node_count)
        .collect()
}

/// One word per node, `1` for each endpoint in the set.
fn spec_witness(node_count: u32, endpoints: &BTreeSet<u32>) -> Vec<u32> {
    (0..node_count)
        .map(|node| u32::from(endpoints.contains(&node)))
        .collect()
}

fn generated_motif_edges(seed: u64, node_count: u32) -> Vec<MotifEdge> {
    let mut rng = csr_sweep::Rng::new(seed | 1);
    let motif_len = 1 + rng.range(5) as usize;
    let mut motif_edges = Vec::with_capacity(motif_len);
    for _ in 0..motif_len {
        motif_edges.push(MotifEdge {
            from: rng.range(node_count),
            kind_mask: 1u32 << rng.range(5),
            to: rng.range(node_count),
        });
    }
    motif_edges
}
