//! Row-strided sparse CSR expansion for skewed active queues.
//!
//! `csr_frontier_queue::csr_queue_forward_traverse` maps one lane to one queued
//! source row. That is the right shape for tiny rows, but power-law graphs can
//! put thousands of edges behind one active source and leave the rest of the GPU
//! idle. This primitive keeps the same queue ABI and assigns a fixed lane team
//! to each queued source:
//!
//! ```text
//! queue index = global_lane / 32
//! edge lane   = global_lane % 32
//! for e = row_start + edge_lane; e < row_end; e += 32:
//!     emit edge target
//! ```
//!
//! It is intentionally a separate Program builder so callers can keep the
//! one-lane-per-source kernel for low-degree graphs and select this path only
//! when row skew is large enough to amortize the extra lanes.

use vyre_foundation::ir::Program;

#[cfg(test)]
use crate::bitset::bitset_words;
use crate::graph::csr_frontier_queue::{
    define_csr_queue_forward_entry_point, CsrQueueForwardTraverseParams,
};
use crate::graph::csr_frontier_step::{csr_queue_step_program, CsrQueueLanes};

/// Canonical op id for row-strided queue-driven CSR expansion.
pub const CSR_QUEUE_STRIDED_FORWARD_OP_ID: &str =
    "vyre-libs::graph::csr_queue_strided_forward_traverse";

/// Fixed lane team assigned to each queued source row.
pub const CSR_QUEUE_STRIDED_FORWARD_LANES_PER_SOURCE: u32 = 32;

/// Workgroup shape for row-strided queue-driven CSR expansion.
pub const CSR_QUEUE_STRIDED_FORWARD_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

define_csr_queue_forward_entry_point! {
    /// Build a GPU program that expands queued CSR source rows with a fixed lane
    /// team per row.
    csr_queue_strided_forward_traverse -> csr_queue_strided_forward_traverse_with
}

/// Build a GPU program that expands queued CSR source rows with a fixed lane
/// team per row.
#[must_use]
pub fn csr_queue_strided_forward_traverse_with(
    params: CsrQueueForwardTraverseParams<'_>,
) -> Program {
    if let Some(program) = params.empty_shape_program(
        CSR_QUEUE_STRIDED_FORWARD_OP_ID,
        "csr_queue_strided_forward_traverse",
    ) {
        return program;
    }
    csr_queue_step_program(&params.spec(
        CSR_QUEUE_STRIDED_FORWARD_OP_ID,
        "csr_queue_strided_forward_traverse",
        "qs",
        CSR_QUEUE_STRIDED_FORWARD_WORKGROUP_SIZE,
        CsrQueueLanes::Team {
            lanes: CSR_QUEUE_STRIDED_FORWARD_LANES_PER_SOURCE,
        },
    ))
}

const EXPECTED_CSR_QUEUE_STRIDED_OUTPUT_BYTES: [u8; 4] = [10, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        CSR_QUEUE_STRIDED_FORWARD_OP_ID,
        || csr_queue_strided_forward_traverse(
            "active_queue",
            "queue_len",
            "edge_offsets",
            "edge_targets",
            "edge_kind_mask",
            "frontier_out",
            4,
            4,
            2,
            1,
        ),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[0, 3]),            // active_queue
                to_bytes(&[2]),               // queue_len
                to_bytes(&[0, 3, 3, 4, 4]),   // edge_offsets
                to_bytes(&[1, 2, 3, 0]),      // edge_targets
                to_bytes(&[1, 2, 1, 1]),      // edge_kind_mask
                to_bytes(&[0]),               // frontier_out
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_CSR_QUEUE_STRIDED_OUTPUT_BYTES.to_vec()]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::csr_frontier_queue::assert_offset_overflow_traps;
    use vyre_reference::composition_witness::csr_queue_strided_forward_witness;

    fn try_csr_queue_strided_forward_traverse_cpu(
        active_queue: &[u32],
        queue_len: u32,
        edge_offsets: &[u32],
        edge_targets: &[u32],
        edge_kind_mask: &[u32],
        node_count: u32,
        allow_mask: u32,
    ) -> Result<Vec<u32>, String> {
        if edge_offsets.len() != (node_count as usize) + 1 {
            return Err("Fix: edge offsets length must match node count + 1.".to_owned());
        }
        if edge_targets.len() != edge_kind_mask.len() {
            return Err("Fix: edge targets length must match edge kind mask length.".to_owned());
        }
        Ok(csr_queue_strided_forward_witness(
            active_queue,
            queue_len,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            node_count,
            allow_mask,
        ))
    }

    fn scalar_queue_forward(
        active_queue: &[u32],
        queue_len: u32,
        edge_offsets: &[u32],
        edge_targets: &[u32],
        edge_kind_mask: &[u32],
        node_count: u32,
        allow_mask: u32,
    ) -> Vec<u32> {
        let mut out = vec![0u32; bitset_words(node_count) as usize];
        let take = (queue_len as usize).min(active_queue.len());
        for &src in &active_queue[..take] {
            if src >= node_count {
                continue;
            }
            for edge in edge_offsets[src as usize]..edge_offsets[src as usize + 1] {
                let edge = edge as usize;
                if edge_kind_mask[edge] & allow_mask == 0 {
                    continue;
                }
                let dst = edge_targets[edge];
                out[dst as usize / 32] |= 1u32 << (dst % 32);
            }
        }
        out
    }

    #[test]
    fn build_program_returns_well_formed_program() {
        let program = csr_queue_strided_forward_traverse(
            "queue", "len", "offsets", "targets", "kinds", "out", 64, 4096, 9, 0x55,
        );
        assert_eq!(
            program.workgroup_size(),
            CSR_QUEUE_STRIDED_FORWARD_WORKGROUP_SIZE
        );
        assert_eq!(program.buffers().len(), 6);
        assert!(!program.stats().trap());
    }

    #[test]
    fn generated_strided_cpu_matches_scalar_reference_on_skewed_rows() {
        let mut seed = 0x51A7_7EED_u32;
        for case in 0..4096u32 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let node_count = 33 + (seed % 224);
            let queue_capacity = 1 + (seed.rotate_left(5) % node_count);
            let mut offsets = Vec::with_capacity(node_count as usize + 1);
            let mut targets = Vec::new();
            let mut masks = Vec::new();
            offsets.push(0);
            for src in 0..node_count {
                seed ^= src.wrapping_mul(0x9E37_79B9).rotate_left((src & 15) + 1);
                let degree = if src == case % node_count {
                    96 + (seed % 257)
                } else {
                    seed % 5
                };
                for edge in 0..degree {
                    targets.push(src.wrapping_mul(17).wrapping_add(edge * 3 + seed) % node_count);
                    masks.push(if (edge ^ src ^ seed) & 3 == 0 { 2 } else { 1 });
                }
                offsets.push(targets.len() as u32);
            }
            let mut queue = Vec::with_capacity(queue_capacity as usize);
            for slot in 0..queue_capacity {
                queue.push(slot.wrapping_mul(7).wrapping_add(seed) % node_count);
            }
            let queue_len = queue_capacity.saturating_add(seed % 3);
            let expected =
                scalar_queue_forward(&queue, queue_len, &offsets, &targets, &masks, node_count, 1);

            assert_eq!(
                try_csr_queue_strided_forward_traverse_cpu(
                    &queue, queue_len, &offsets, &targets, &masks, node_count, 1,
                ),
                Ok(expected),
                "generated skewed CSR queue case {case}"
            );
        }
    }

    #[test]
    fn invalid_shape_returns_trap_program() {
        let program = csr_queue_strided_forward_traverse(
            "queue", "len", "offsets", "targets", "kinds", "out", 0, 0, 1, 1,
        );

        assert!(
            program.stats().trap(),
            "invalid node_count must compile to a trap program"
        );
    }

    #[test]
    fn offset_count_overflow_returns_trap_program_without_panic() {
        assert_offset_overflow_traps(
            csr_queue_strided_forward_traverse,
            "csr_queue_strided_forward_traverse",
        );
    }
}
