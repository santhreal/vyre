//! Handwritten oracle matrix for `graph::csr_bidirectional` one-step reach.
//!
//! Compares production bidirectional CSR step against an independent forward+
//! backward union oracle on 1024 generated CSR/frontier shapes.

#![forbid(unsafe_code)]
#![cfg(feature = "cpu-parity")]

use vyre_foundation::ir::Program;
use vyre_primitives::graph::csr_bidirectional;
use vyre_primitives::graph::program_graph::ProgramGraphShape;
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

#[test]
fn csr_bidirectional_matches_independent_union_oracle_matrix() {
    for case in 0..8192usize {
        let seed = case as u64 ^ 0xB1D1_0000_0000_0000;
        let (node_count, offsets, targets, masks, frontier, allow_mask) =
            generated_csr_frontier(seed);

        let expected = oracle_bidirectional_step(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask,
        );
        let actual = csr_bidirectional::cpu_ref(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask,
        );
        assert_eq!(
            actual, expected,
            "Fix: csr_bidirectional cpu_ref oracle case {case} node_count={node_count} allow_mask={allow_mask:#x} must match the independent union oracle."
        );

        let mut reused = vec![0xDEAD_BEEF; bitset_words(node_count) + 3];
        csr_bidirectional::cpu_ref_into(
            node_count,
            &offsets,
            &targets,
            &masks,
            &frontier,
            allow_mask,
            &mut reused,
        );
        assert_eq!(
            reused, expected,
            "Fix: csr_bidirectional cpu_ref_into oracle case {case} must clear stale frontier capacity before writing."
        );
    }
}

/// Position of the writable buffer `name` within `reference_eval`'s returned outputs.
/// Delegates to the interpreter's own output-selection predicate so it cannot drift.
fn output_index(program: &Program, name: &str) -> usize {
    vyre_reference::output_index(program, name)
        .expect("Fix: csr_bidirectional must declare the frontier_out buffer")
}

/// Drive the FUSED `csr_bidirectional` GPU program through `reference_eval` and assert
/// its `frontier_out` bitset equals `cpu_ref`, byte-for-byte, over the same generated
/// CSR/frontier shapes the oracle matrix uses.
///
/// The existing matrix only checks `cpu_ref` vs an independent oracle, the actual GPU
/// IR was never executed. This closes that gap: the fused program is a single Region
/// with the forward step followed by the backward step, both marking ONE shared
/// `frontier_out` bitset via atomic-or. Crucially there is NO inter-arm `GridSync`
/// fence (and none is needed): the store index `dst = edge_targets[e]` is a data value,
/// not a launch-geometry offset, so `fuse_programs` does not promote the boundary, and
/// neither arm reads `frontier_out`, so the two commutative atomic-or passes are order-
/// independent. The test therefore verifies (a) both directions' atomic writes land in
/// the final bitset regardless of arm order, and (b) the reference dispatch covers every
/// node, not just the `node_count / 32` bitset words the output buffer is sized in.
#[test]
fn csr_bidirectional_fused_program_matches_cpu_ref_via_reference_eval() {
    for case in 0..512usize {
        let seed = case as u64 ^ 0xB1D1_0000_0000_0EEF;
        let (node_count, offsets, targets, masks, frontier, allow_mask) =
            generated_csr_frontier(seed);

        // Pad the edge-indexed buffers to the primitive's physical storage width
        // (`edge_count.max(1)`); the pad element sits at index `edge_count`, past every
        // `offsets[node_count]`-bounded edge scan, so neither the GPU program nor
        // `cpu_ref` ever reads it (both stay in exact agreement).
        let edge_count = targets.len() as u32;
        let storage = edge_count.max(1) as usize;
        let mut padded_targets = targets.clone();
        padded_targets.resize(storage, 0);
        let mut padded_masks = masks.clone();
        padded_masks.resize(storage, 0);

        let program = csr_bidirectional::csr_bidirectional(
            ProgramGraphShape::new(node_count, edge_count.max(1)),
            "frontier_in",
            "frontier_out",
            allow_mask,
        );

        let words = bitset_words(node_count);
        let node_words = node_count as usize;
        // Positional inputs in binding order: the five read-only ProgramGraph buffers
        // (pg_nodes, pg_edge_offsets, pg_edge_targets, pg_edge_kind_mask, pg_node_tags),
        // then frontier_in, then the zero-initialised RW frontier_out.
        let inputs = vec![
            Value::from(pack(&vec![0u32; node_words])), // pg_nodes (unused by the step)
            Value::from(pack(&offsets)),                // pg_edge_offsets (node_count+1)
            Value::from(pack(&padded_targets)),         // pg_edge_targets
            Value::from(pack(&padded_masks)),           // pg_edge_kind_mask
            Value::from(pack(&vec![0u32; node_words])), // pg_node_tags (unused by the step)
            Value::from(pack(&frontier)),               // frontier_in
            Value::from(pack(&vec![0u32; words])),      // frontier_out (init)
        ];

        let outputs = vyre_reference::reference_eval(&program, &inputs).unwrap_or_else(|error| {
            panic!("csr_bidirectional case {case} (node_count={node_count}) reference_eval failed: {error}")
        });
        let out_idx = output_index(&program, "frontier_out");
        let mut gpu = unpack(&outputs[out_idx].to_bytes());
        gpu.truncate(words);

        let cpu = csr_bidirectional::cpu_ref(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask,
        );

        assert_eq!(
            gpu, cpu,
            "Fix: csr_bidirectional GPU program diverges from cpu_ref at case {case} \
             (node_count={node_count}, edge_count={edge_count}, allow_mask={allow_mask:#x}). \
             A mismatch means one direction's atomic frontier write was lost, an arm \
             read stale state, or the dispatch under-covered the node range."
        );
    }
}

fn oracle_bidirectional_step(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    let words = bitset_words(node_count);
    let node_words = node_count as usize;
    let mut out = vec![0u32; words];
    for src in 0..node_words {
        let src_word = src / 32;
        let src_bit = 1u32 << (src % 32);
        let src_in_frontier =
            src_word < frontier_in.len() && (frontier_in[src_word] & src_bit) != 0;
        let edge_start = edge_offsets[src] as usize;
        let edge_end = edge_offsets[src + 1] as usize;
        let mut backward_hit = false;
        for edge in edge_start..edge_end.min(edge_targets.len()).min(edge_kind_mask.len()) {
            if edge_kind_mask[edge] & allow_mask == 0 {
                continue;
            }
            let dst = edge_targets[edge] as usize;
            let dst_word = dst / 32;
            let dst_bit = 1u32 << (dst % 32);
            if src_in_frontier && dst < node_words {
                out[dst_word] |= dst_bit;
            }
            if dst_word < frontier_in.len() && (frontier_in[dst_word] & dst_bit) != 0 {
                backward_hit = true;
            }
        }
        if backward_hit && src_word < out.len() {
            out[src_word] |= src_bit;
        }
    }
    out
}

fn generated_csr_frontier(seed: u64) -> (u32, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, u32) {
    let mut rng = seed;
    let node_count = 1 + next_u32(&mut rng) % 96;
    let words = bitset_words(node_count);
    let mut offsets = Vec::with_capacity(node_count as usize + 1);
    let mut targets = Vec::new();
    let mut masks = Vec::new();
    offsets.push(0);
    for _ in 0..node_count {
        let degree = next_u32(&mut rng) % 6;
        for _ in 0..degree {
            targets.push(next_u32(&mut rng) % node_count);
            let bit = 1u32 << (next_u32(&mut rng) % 5);
            let noise = if next_u32(&mut rng) & 7 == 0 {
                1u32 << (next_u32(&mut rng) % 5)
            } else {
                0
            };
            masks.push(bit | noise);
        }
        offsets.push(targets.len() as u32);
    }
    let mut frontier = vec![0u32; words];
    for node in 0..node_count {
        if next_u32(&mut rng) & 3 == 0 {
            frontier[(node / 32) as usize] |= 1u32 << (node % 32);
        }
    }
    if next_u32(&mut rng) & 1 == 0 {
        let word = (node_count - 1) / 32;
        let used = node_count % 32;
        if used != 0 {
            frontier[word as usize] |= !((1u32 << used) - 1);
        }
    }
    let allow_mask = match next_u32(&mut rng) % 6 {
        0 => 0,
        1 => 1,
        2 => 0b10,
        3 => 0b101,
        _ => 0xFFFF_FFFF,
    };
    (node_count, offsets, targets, masks, frontier, allow_mask)
}

fn bitset_words(node_count: u32) -> usize {
    node_count.div_ceil(32) as usize
}

fn next_u32(rng: &mut u64) -> u32 {
    *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*rng >> 16) as u32
}
