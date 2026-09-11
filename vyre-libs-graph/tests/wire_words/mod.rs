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

/// Run a frontier-queue scatter and decode the queue and its length.
///
/// Two suites drive a scatter the same way: the frontier words, whatever
/// prefix buffers the variant needs, then a zeroed queue of `queue_capacity`
/// slots and a single-element length of zero. Each stated that binding tail
/// and its own decode, so a suite that sized the queue in words rather than
/// bytes compared against a buffer the program never filled.
///
/// Outputs are read by declared name, not by position, so a builder that
/// reorders its outputs is caught rather than silently swapping the two.
#[cfg(feature = "graph")]
pub(crate) fn run_frontier_scatter(
    program: &vyre_foundation::ir::Program,
    leading: Vec<vyre_reference::value::Value>,
    queue_capacity: u32,
) -> (Vec<u32>, Vec<u32>) {
    use vyre_reference::value::Value;

    let mut inputs = leading;
    inputs.push(Value::from(vec![
        0_u8;
        queue_capacity as usize * size_of::<u32>()
    ]));
    inputs.push(Value::from(vyre_primitives::wire::pack_u32_slice(&[0])));
    let outputs = vyre_reference::ReferenceRequest::standard(program, &inputs)
        .outputs()
        .expect("Fix: frontier queue scatter must evaluate on the reference oracle");
    let named = |name: &str| {
        let index = vyre_reference::output_index(program, name)
            .unwrap_or_else(|| panic!("Fix: scatter program must declare output `{name}`"));
        decode_u32_words(&outputs[index].to_bytes())
    };
    (named("queue"), named("queue_len"))
}
