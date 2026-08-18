//! Deterministic graph fixtures shared by volume-oracle integration suites.

/// Return the number of words required by a node frontier.
pub(crate) fn bitset_words(node_count: u32) -> usize {
    vyre_libs::bitset::bitset_words(node_count) as usize
}

/// A CSR frontier-step Program builder: `csr_forward_traverse`,
/// `csr_backward_traverse`, and every other one-round step over a
/// `ProgramGraphShape` share this shape.
pub(crate) type FrontierStepBuilder = fn(
    vyre_libs::graph::program_graph::ProgramGraphShape,
    &str,
    &str,
    u32,
) -> vyre_foundation::ir::Program;

/// Build a one-round CSR frontier step, run it through the reference
/// interpreter, and return the `frontier_out` word bitset.
///
/// Every such step binds the same buffers in the same order: pg_nodes(0),
/// pg_edge_offsets(1), pg_edge_targets(2), pg_edge_kind_mask(3),
/// pg_node_tags(4), frontier_in(5), and frontier_out(6), the only ReadWrite
/// buffer, fed a zeroed slot and returned as the single writable output.
///
/// The dispatch floor is `node_count` because the lanes are node-indexed: the
/// interpreter otherwise infers the grid from the largest buffer, which under-
/// fires on a sparse graph. The per-lane `src < node_count` guard drops any
/// over-fire.
pub(crate) fn frontier_step_out(
    build: FrontierStepBuilder,
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    use vyre_libs::graph::program_graph::ProgramGraphShape;
    use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
    use vyre_reference::value::Value;

    let edge_count = *edge_offsets
        .last()
        .expect("offsets has node_count+1 entries");
    let program = build(
        ProgramGraphShape::new(node_count, edge_count),
        "frontier_in",
        "frontier_out",
        allow_mask,
    );
    let padded_edges = edge_count.max(1) as usize;
    let mut targets = edge_targets.to_vec();
    targets.resize(padded_edges, 0);
    let mut kind_mask = edge_kind_mask.to_vec();
    kind_mask.resize(padded_edges, 0);
    let nodes = vec![0u32; node_count as usize];
    let node_tags = vec![0u32; node_count as usize];
    let words = bitset_words(node_count);

    let outputs = vyre_reference::reference_eval_with_dispatch(
        &program,
        &[
            Value::from(pack(&nodes)),
            Value::from(pack(edge_offsets)),
            Value::from(pack(&targets)),
            Value::from(pack(&kind_mask)),
            Value::from(pack(&node_tags)),
            Value::from(pack(frontier_in)),
            Value::from(pack(&vec![0u32; words])),
        ],
        node_count,
    )
    .expect("CSR frontier-step reference evaluation must succeed");
    unpack(&outputs[0].to_bytes())
}

/// Execute a CSR frontier step IR program and unpack its output word bitset.
pub(crate) fn gpu_step(
    build: FrontierStepBuilder,
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    frontier_step_out(
        build,
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_in,
        allow_mask,
    )
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Clone, Debug)]
pub(crate) struct GeneratedCsr {
    pub(crate) edge_offsets: Vec<u32>,
    pub(crate) edge_targets: Vec<u32>,
    pub(crate) edge_kind_mask: Vec<u32>,
}

pub(crate) fn generated_frontier_words(node_count: u32, seed: u64) -> Vec<u32> {

    let words = bitset_words(node_count);
    let mut frontier = Vec::with_capacity(words);
    for word in 0..words {
        frontier.push(mix64(seed ^ (word as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)) as u32);
    }
    if seed & 1 == 0 {
        set_node(&mut frontier, 0);
    }
    if seed & 2 == 0 {
        set_node(&mut frontier, node_count - 1);
    }
    if node_count > 32 && seed & 4 == 0 {
        set_node(&mut frontier, 32);
    }
    frontier
}

pub(crate) fn generated_csr(node_count: u32, seed: u64) -> GeneratedCsr {

    let mut edge_offsets = Vec::with_capacity(node_count as usize + 1);
    let mut edge_targets = Vec::new();
    let mut edge_kind_mask = Vec::new();
    edge_offsets.push(0);
    for src in 0..node_count {
        let row_seed = mix64(seed ^ (src as u64).wrapping_mul(0xd1b5_4a32_d192_ed03));
        let degree = (row_seed % 5) as u32;
        for edge_ordinal in 0..degree {
            let edge_seed =
                mix64(row_seed ^ (edge_ordinal as u64).wrapping_mul(0x94d0_49bb_1331_11eb));
            edge_targets.push((edge_seed % u64::from(node_count)) as u32);
            let mask_bit = ((edge_seed >> 17) % 9) as u32;
            edge_kind_mask.push(if mask_bit == 8 { 0 } else { 1u32 << mask_bit });
        }
        edge_offsets.push(edge_targets.len() as u32);
    }
    GeneratedCsr {
        edge_offsets,
        edge_targets,
        edge_kind_mask,
    }
}

pub(crate) fn active_nodes(frontier: &[u32], node_count: u32) -> Vec<u32> {
    (0..node_count)
        .filter(|&node| frontier_has_node(frontier, node))
        .collect()
}

pub(crate) fn max_row_degree(edge_offsets: &[u32]) -> u32 {
    edge_offsets
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .max()
        .unwrap_or(0)
}

pub(crate) fn frontier_has_node(frontier: &[u32], node: u32) -> bool {
    frontier[node as usize / 32] & (1u32 << (node & 31)) != 0
}

pub(crate) fn set_node(frontier: &mut [u32], node: u32) {
    frontier[node as usize / 32] |= 1u32 << (node & 31);
}
