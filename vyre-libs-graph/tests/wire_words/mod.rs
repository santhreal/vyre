//! Graph oracles and wire access for this crate's tests.

pub(crate) use vyre_primitives::wire::decode_u32_le_bytes_all as decode_u32_words;

/// Advance a 64-bit splitmix state.
pub(crate) fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(feature = "graph")]
pub(crate) fn toposort(
    node_count: u32,
    edges: &[(u32, u32)],
) -> Result<Vec<u32>, vyre_libs_graph::graph::toposort::ToposortError> {
    for (edge_idx, &(from, to)) in edges.iter().enumerate() {
        if from >= node_count {
            return Err(
                vyre_libs_graph::graph::toposort::ToposortError::UnknownNode {
                    edge: edge_idx,
                    node: from,
                },
            );
        }
        if to >= node_count {
            return Err(
                vyre_libs_graph::graph::toposort::ToposortError::UnknownNode {
                    edge: edge_idx,
                    node: to,
                },
            );
        }
    }
    vyre_reference::composition_witness::toposort_witness(node_count, edges).map_err(|err| {
        if let Some(rest) = err.strip_prefix("Cycle detected involving node ") {
            if let Ok(node) = rest.parse::<u32>() {
                return vyre_libs_graph::graph::toposort::ToposortError::Cycle { node };
            }
        }
        vyre_libs_graph::graph::toposort::ToposortError::InconsistentState { message: err }
    })
}

#[cfg(feature = "graph")]
pub(crate) fn queue_forward_oracle(
    active_queue: &[u32],
    queue_len: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    node_count: u32,
    allow_mask: u32,
) -> Vec<u32> {
    let mut out = vec![0u32; vyre_libs_bitset::bitset::bitset_words(node_count) as usize];
    let take = (queue_len as usize).min(active_queue.len());
    for &src in &active_queue[..take] {
        if src >= node_count {
            continue;
        }
        let start = edge_offsets[src as usize] as usize;
        let end = edge_offsets[src as usize + 1] as usize;
        for edge in start..end {
            if edge_kind_mask[edge] & allow_mask != 0 {
                let dst = edge_targets[edge];
                out[dst as usize / 32] |= 1u32 << (dst % 32);
            }
        }
    }
    out
}
