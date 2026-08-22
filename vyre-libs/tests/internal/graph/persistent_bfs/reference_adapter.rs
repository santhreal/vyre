use crate::graph::csr_closure_inputs::CsrClosureInputs;
use vyre_reference::composition_witness::{
    csr_persistent_closure_detailed_witness, csr_persistent_closure_witness_with_scratch_into,
};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PersistentBfsConvergence {
    pub(crate) changed: u32,
    pub(crate) converged: bool,
    pub(crate) stop_iter: u32,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct PersistentBfsCpuScratch {
    pub(crate) step: Vec<u32>,
}

impl PersistentBfsCpuScratch {
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

pub(crate) fn cpu_ref(inputs: CsrClosureInputs<'_>, frontier_in: &[u32]) -> (Vec<u32>, u32) {
    try_cpu_ref(inputs, frontier_in)
        .unwrap_or_else(|error| panic!("invalid persistent BFS witness input: {error}"))
}

pub(crate) fn try_cpu_ref(
    inputs: CsrClosureInputs<'_>,
    frontier_in: &[u32],
) -> Result<(Vec<u32>, u32), String> {
    let result = validated_witness(inputs, frontier_in)?;
    Ok((result.frontier, result.changed))
}

pub(crate) fn cpu_ref_into(
    inputs: CsrClosureInputs<'_>,
    frontier_in: &[u32],
    frontier_out: &mut Vec<u32>,
) -> u32 {
    try_cpu_ref_into(inputs, frontier_in, frontier_out)
        .unwrap_or_else(|error| panic!("invalid persistent BFS witness input: {error}"))
}

pub(crate) fn try_cpu_ref_into(
    inputs: CsrClosureInputs<'_>,
    frontier_in: &[u32],
    frontier_out: &mut Vec<u32>,
) -> Result<u32, String> {
    let mut scratch = PersistentBfsCpuScratch::default();
    try_cpu_ref_into_with_scratch(inputs, frontier_in, frontier_out, &mut scratch)
}

pub(crate) fn try_cpu_ref_into_with_scratch(
    inputs: CsrClosureInputs<'_>,
    frontier_in: &[u32],
    frontier_out: &mut Vec<u32>,
    scratch: &mut PersistentBfsCpuScratch,
) -> Result<u32, String> {
    let graph = inputs.graph;
    super::validate_persistent_bfs_inputs(
        graph.node_count,
        graph.edge_offsets,
        graph.edge_targets,
        graph.edge_kind_mask,
        frontier_in,
    )?;
    let changed = csr_persistent_closure_witness_with_scratch_into(
        graph.node_count,
        graph.edge_offsets,
        graph.edge_targets,
        graph.edge_kind_mask,
        frontier_in,
        inputs.allow_mask,
        inputs.max_iters,
        frontier_out,
        &mut scratch.step,
    );
    Ok(changed)
}

pub(crate) fn try_cpu_ref_converged(
    inputs: CsrClosureInputs<'_>,
    frontier_in: &[u32],
) -> Result<(Vec<u32>, PersistentBfsConvergence), String> {
    let result = validated_witness(inputs, frontier_in)?;
    let outcome = PersistentBfsConvergence {
        changed: result.changed,
        converged: result.converged,
        stop_iter: result.stop_iteration,
    };
    Ok((result.frontier, outcome))
}

pub(crate) fn try_cpu_ref_density(
    inputs: CsrClosureInputs<'_>,
    frontier_in: &[u32],
) -> Result<(Vec<u32>, PersistentBfsConvergence, Vec<u32>), String> {
    let result = validated_witness(inputs, frontier_in)?;
    let outcome = PersistentBfsConvergence {
        changed: result.changed,
        converged: result.converged,
        stop_iter: result.stop_iteration,
    };
    Ok((result.frontier, outcome, result.active_density))
}

fn validated_witness(
    inputs: CsrClosureInputs<'_>,
    frontier_in: &[u32],
) -> Result<vyre_reference::composition_witness::CsrPersistentClosureWitness, String> {
    let graph = inputs.graph;
    super::validate_persistent_bfs_inputs(
        graph.node_count,
        graph.edge_offsets,
        graph.edge_targets,
        graph.edge_kind_mask,
        frontier_in,
    )?;
    Ok(csr_persistent_closure_detailed_witness(
        graph.node_count,
        graph.edge_offsets,
        graph.edge_targets,
        graph.edge_kind_mask,
        frontier_in,
        inputs.allow_mask,
        inputs.max_iters,
    ))
}
