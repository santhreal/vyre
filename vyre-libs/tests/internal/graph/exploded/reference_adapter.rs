use vyre_reference::composition_witness::{
    try_exploded_ifds_csr_witness_into, ExplodedIfdsScratchWitness,
};

pub(crate) type ExplodedIfdsCpuScratch = ExplodedIfdsScratchWitness;

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_cpu_reference(
    procedure_count: u32,
    blocks_per_procedure: u32,
    facts_per_procedure: u32,
    intra_edges: &[(u32, u32, u32)],
    inter_edges: &[(u32, u32, u32, u32)],
    generated_facts: &[(u32, u32, u32)],
    killed_facts: &[(u32, u32, u32)],
) -> (Vec<u32>, Vec<u32>) {
    try_build_cpu_reference(
        procedure_count,
        blocks_per_procedure,
        facts_per_procedure,
        intra_edges,
        inter_edges,
        generated_facts,
        killed_facts,
    )
    .unwrap_or_else(|error| panic!("exploded IFDS CPU reference received malformed input. {error}"))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_build_cpu_reference(
    procedure_count: u32,
    blocks_per_procedure: u32,
    facts_per_procedure: u32,
    intra_edges: &[(u32, u32, u32)],
    inter_edges: &[(u32, u32, u32, u32)],
    generated_facts: &[(u32, u32, u32)],
    killed_facts: &[(u32, u32, u32)],
) -> Result<(Vec<u32>, Vec<u32>), String> {
    let mut rows = Vec::new();
    let mut columns = Vec::new();
    let mut scratch = ExplodedIfdsCpuScratch::default();
    try_build_cpu_reference_into(
        procedure_count,
        blocks_per_procedure,
        facts_per_procedure,
        intra_edges,
        inter_edges,
        generated_facts,
        killed_facts,
        &mut rows,
        &mut columns,
        &mut scratch,
    )?;
    Ok((rows, columns))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_build_cpu_reference_into(
    procedure_count: u32,
    blocks_per_procedure: u32,
    facts_per_procedure: u32,
    intra_edges: &[(u32, u32, u32)],
    inter_edges: &[(u32, u32, u32, u32)],
    generated_facts: &[(u32, u32, u32)],
    killed_facts: &[(u32, u32, u32)],
    rows: &mut Vec<u32>,
    columns: &mut Vec<u32>,
    scratch: &mut ExplodedIfdsCpuScratch,
) -> Result<(), String> {
    let layout = super::validate_ifds_csr_inputs(
        procedure_count,
        blocks_per_procedure,
        facts_per_procedure,
        intra_edges,
        inter_edges,
        generated_facts,
        killed_facts,
    )?;
    if layout.empty {
        return Err(format!(
            "exploded IFDS CPU reference dimensions must be nonzero, got procs={procedure_count}, blocks={blocks_per_procedure}, facts={facts_per_procedure}. Fix: pass a real exploded-supergraph domain before parity comparison."
        ));
    }
    try_exploded_ifds_csr_witness_into(
        procedure_count,
        blocks_per_procedure,
        facts_per_procedure,
        intra_edges,
        inter_edges,
        generated_facts,
        killed_facts,
        rows,
        columns,
        scratch,
    )
}
