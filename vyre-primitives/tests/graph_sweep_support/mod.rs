//! Deterministic graph fixtures shared by volume-oracle integration suites.

/// Return the number of words required by a node frontier.
pub(crate) fn bitset_words(node_count: u32) -> usize {
    vyre_primitives::bitset::bitset_words(node_count) as usize
}

/// Advance the deterministic graph fixture generator.
pub(crate) fn next_u32(rng: &mut u64) -> u32 {
    *rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (*rng >> 32) as u32
}

/// Generate a bounded CSR graph, one frontier seed, and an all-kinds mask.
pub(crate) fn generated_csr_frontier(
    seed: u64,
) -> (u32, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, u32) {
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
            masks.push(1u32 << (next_u32(&mut rng) % 5));
        }
        offsets.push(targets.len() as u32);
    }
    let mut frontier = vec![0u32; words];
    let start = next_u32(&mut rng) % node_count;
    frontier[(start / 32) as usize] |= 1u32 << (start % 32);
    (node_count, offsets, targets, masks, frontier, u32::MAX)
}

/// Generate a bounded CSR graph with a MULTI-SOURCE frontier and a randomly
/// restricted kind mask, so the per-edge kind intersection fires both ways and
/// many source lanes contend for the same output word.
///
/// `generated_csr_frontier` seeds a single bit and allows every kind, which
/// leaves both of those branches unexercised.
pub(crate) fn generated_csr_multi_source_frontier(
    seed: u64,
) -> (u32, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, u32) {
    let mut rng = seed;
    let node_count = 1 + next_u32(&mut rng) % 96;
    let mut offsets = Vec::with_capacity(node_count as usize + 1);
    let mut targets = Vec::new();
    let mut masks = Vec::new();
    offsets.push(0u32);
    for _ in 0..node_count {
        let degree = next_u32(&mut rng) % 7;
        for _ in 0..degree {
            targets.push(next_u32(&mut rng) % node_count);
            masks.push(1u32 << (next_u32(&mut rng) % 5));
        }
        offsets.push(targets.len() as u32);
    }
    let mut frontier = vec![0u32; bitset_words(node_count)];
    for node in 0..node_count {
        if next_u32(&mut rng) & 1 == 0 {
            frontier[(node / 32) as usize] |= 1u32 << (node % 32);
        }
    }
    // Never trivially zero: an empty allow mask would empty every frontier and
    // make the case vacuous.
    let allow_mask = 1u32 << (next_u32(&mut rng) % 5) | 1u32 << (next_u32(&mut rng) % 5);
    (node_count, offsets, targets, masks, frontier, allow_mask)
}

/// A CSR frontier-step Program builder: `csr_forward_traverse`,
/// `csr_backward_traverse`, and every other one-round step over a
/// `ProgramGraphShape` share this shape.
pub(crate) type FrontierStepBuilder = fn(
    vyre_primitives::graph::program_graph::ProgramGraphShape,
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
    use vyre_primitives::graph::program_graph::ProgramGraphShape;
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
