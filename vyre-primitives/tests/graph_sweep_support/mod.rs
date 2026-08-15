//! Deterministic graph fixtures shared by volume-oracle integration suites.

/// Return the number of words required by a node frontier.
pub(crate) fn bitset_words(node_count: u32) -> usize {
    vyre_primitives::bitset::bitset_words(node_count) as usize
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
