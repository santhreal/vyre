use super::validate::validate_csr_inputs;
use crate::graph::csr_closure_inputs::CsrClosureInputs;
use vyre_reference::composition_witness::{
    csr_forward_or_changed_closure_with_step_hook_witness_into,
    csr_forward_or_changed_closure_witness, csr_forward_or_changed_closure_witness_into,
    csr_forward_or_changed_witness_into,
};

pub(crate) fn cpu_ref(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
) -> (Vec<u32>, u32) {
    let mut output = Vec::new();
    let changed = cpu_ref_into(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier,
        allow_mask,
        &mut output,
    );
    (output, changed)
}

pub(crate) fn cpu_ref_into(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
    output: &mut Vec<u32>,
) -> u32 {
    let _layout = validate_csr_inputs(node_count, edge_offsets, edge_targets, edge_kind_mask)
        .unwrap_or_else(|error| panic!("invalid CSR forward witness input: {error}"));
    csr_forward_or_changed_witness_into(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier,
        allow_mask,
        output,
    )
}

pub(crate) fn cpu_ref_closure(inputs: CsrClosureInputs<'_>, seed: &[u32]) -> Vec<u32> {
    let graph = inputs.graph;
    let _layout = validate_csr_inputs(
        graph.node_count,
        graph.edge_offsets,
        graph.edge_targets,
        graph.edge_kind_mask,
    )
    .unwrap_or_else(|error| panic!("invalid CSR forward witness input: {error}"));
    csr_forward_or_changed_closure_witness(
        graph.node_count,
        graph.edge_offsets,
        graph.edge_targets,
        graph.edge_kind_mask,
        seed,
        inputs.allow_mask,
        inputs.max_iters,
    )
}
pub(crate) fn cpu_ref_closure_into(
    inputs: CsrClosureInputs<'_>,
    seed: &[u32],
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
) {
    let graph = inputs.graph;
    let _layout = validate_csr_inputs(
        graph.node_count,
        graph.edge_offsets,
        graph.edge_targets,
        graph.edge_kind_mask,
    )
    .unwrap_or_else(|error| panic!("invalid CSR forward witness input: {error}"));
    csr_forward_or_changed_closure_witness_into(
        graph.node_count,
        graph.edge_offsets,
        graph.edge_targets,
        graph.edge_kind_mask,
        seed,
        inputs.allow_mask,
        inputs.max_iters,
        current,
        next,
    );
}

pub(crate) fn cpu_ref_closure_into_with_step_hook(
    inputs: CsrClosureInputs<'_>,
    seed: &[u32],
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
    on_step: impl FnMut(u32),
) {
    let graph = inputs.graph;
    let _layout = validate_csr_inputs(
        graph.node_count,
        graph.edge_offsets,
        graph.edge_targets,
        graph.edge_kind_mask,
    )
    .unwrap_or_else(|error| panic!("invalid CSR forward witness input: {error}"));
    csr_forward_or_changed_closure_with_step_hook_witness_into(
        graph.node_count,
        graph.edge_offsets,
        graph.edge_targets,
        graph.edge_kind_mask,
        seed,
        inputs.allow_mask,
        inputs.max_iters,
        on_step,
        current,
        next,
    );
}
