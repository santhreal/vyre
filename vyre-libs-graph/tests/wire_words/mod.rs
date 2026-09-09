//! Wire helpers for tests.
#![allow(dead_code, unused_imports, unused_variables)]

pub(crate) struct Lcg(pub(crate) u64);

impl Lcg {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    pub(crate) fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            0
        } else {
            self.next_u32() % n
        }
    }
}

pub(crate) use vyre_primitives::wire::decode_u32_le_bytes_all as decode_u32_words;
pub(crate) use vyre_primitives::wire::pack_u32_slice as u32_bytes;

pub(crate) fn lcg_u32(count: usize, seed: u64) -> Vec<u32> {
    let mut rng = Lcg::new(seed);
    (0..count).map(|_| rng.next_u32()).collect()
}

pub(crate) fn ramp(count: usize, start: u32, step: u32) -> Vec<u32> {
    (0..count)
        .map(|i| start.wrapping_add((i as u32).wrapping_mul(step)))
        .collect()
}

pub(crate) fn alternating(count: usize, a: u32, b: u32) -> Vec<u32> {
    (0..count).map(|i| if i % 2 == 0 { a } else { b }).collect()
}
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
