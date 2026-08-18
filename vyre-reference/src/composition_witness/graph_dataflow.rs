//! Sequential mathematical witnesses for exploded IFDS CSR construction and d-DNNF circuit evaluation.

/// Scratch storage for exploded IFDS CSR construction.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExplodedIfdsScratchWitness {
    /// Flattened edge scratch buffer.
    pub edges_flat: Vec<(u32, u32)>,
    /// Dense boolean mask of killed facts.
    pub killed: Vec<bool>,
    /// Generation rule offset scratch buffer.
    pub gen_offsets: Vec<usize>,
    /// Generation rule cursor scratch buffer.
    pub gen_cursor: Vec<usize>,
    /// Generated fact IDs scratch buffer.
    pub gen_facts: Vec<u32>,
    /// Row cursor scratch buffer for CSR materialization.
    pub cursor: Vec<usize>,
}

impl ExplodedIfdsScratchWitness {
    /// Create a new empty scratch workspace.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fallible sequential mathematical witness for building dense-index exploded IFDS CSR into caller storage.
///
/// Validates input bounds before modifying `row_offsets`, `columns`, or `scratch`.
#[allow(clippy::too_many_arguments)]
pub fn try_exploded_ifds_csr_witness_into(
    procedure_count: u32,
    blocks_per_procedure: u32,
    facts_per_procedure: u32,
    intra_edges: &[(u32, u32, u32)],
    inter_edges: &[(u32, u32, u32, u32)],
    generated_facts: &[(u32, u32, u32)],
    killed_facts: &[(u32, u32, u32)],
    row_offsets: &mut Vec<u32>,
    columns: &mut Vec<u32>,
    scratch: &mut ExplodedIfdsScratchWitness,
) -> Result<(), String> {
    if procedure_count == 0 || blocks_per_procedure == 0 || facts_per_procedure == 0 {
        return Err(format!(
            "exploded IFDS CPU reference dimensions must be nonzero, got procs={procedure_count}, blocks={blocks_per_procedure}, facts={facts_per_procedure}. Fix: pass a real exploded-supergraph domain before parity comparison."
        ));
    }
    let Some(slots_per_procedure) =
        (blocks_per_procedure as usize).checked_mul(facts_per_procedure as usize)
    else {
        return Err("exploded IFDS slots_per_procedure overflow".to_string());
    };
    let Some(node_count) = (procedure_count as usize).checked_mul(slots_per_procedure) else {
        return Err("exploded IFDS node_count overflow".to_string());
    };
    if node_count > u32::MAX as usize {
        return Err("exploded IFDS node_count exceeds u32".to_string());
    }

    let index = |procedure: u32, block: u32, fact: u32| {
        procedure as usize * slots_per_procedure
            + block as usize * facts_per_procedure as usize
            + fact as usize
    };
    let in_domain = |procedure: u32, block: u32, fact: u32| {
        procedure < procedure_count && block < blocks_per_procedure && fact < facts_per_procedure
    };

    scratch.killed.clear();
    scratch.killed.resize(node_count, false);
    for &(p, b, f) in killed_facts {
        if in_domain(p, b, f) {
            let idx = index(p, b, f);
            scratch.killed[idx] = true;
        }
    }

    row_offsets.clear();
    row_offsets.resize(node_count + 1, 0);

    for &(procedure, source_block, destination_block) in intra_edges {
        if procedure >= procedure_count
            || source_block >= blocks_per_procedure
            || destination_block >= blocks_per_procedure
        {
            continue;
        }
        for fact in 0..facts_per_procedure {
            if !scratch.killed[index(procedure, source_block, fact)] {
                row_offsets[index(procedure, source_block, fact) + 1] += 1;
            }
        }
        for &(generated_procedure, generated_block, fact) in generated_facts {
            if generated_procedure == procedure
                && generated_block == source_block
                && fact < facts_per_procedure
            {
                row_offsets[index(procedure, source_block, 0) + 1] += 1;
            }
        }
    }
    for &(source_procedure, source_block, destination_procedure, destination_block) in inter_edges {
        if source_procedure >= procedure_count
            || destination_procedure >= procedure_count
            || source_block >= blocks_per_procedure
            || destination_block >= blocks_per_procedure
        {
            continue;
        }
        for fact in 0..facts_per_procedure {
            row_offsets[index(source_procedure, source_block, fact) + 1] += 1;
        }
    }

    let mut total_edges = 0_usize;
    for i in 0..node_count {
        total_edges += row_offsets[i + 1] as usize;
        row_offsets[i + 1] = total_edges as u32;
    }

    columns.clear();
    columns.resize(total_edges, 0);

    scratch.cursor.clear();
    scratch.cursor.extend(
        row_offsets[..node_count]
            .iter()
            .map(|&offset| offset as usize),
    );
    for &(procedure, source_block, destination_block) in intra_edges {
        if procedure >= procedure_count
            || source_block >= blocks_per_procedure
            || destination_block >= blocks_per_procedure
        {
            continue;
        }
        for fact in 0..facts_per_procedure {
            if !scratch.killed[index(procedure, source_block, fact)] {
                let src = index(procedure, source_block, fact);
                let pos = scratch.cursor[src];
                columns[pos] = index(procedure, destination_block, fact) as u32;
                scratch.cursor[src] += 1;
            }
        }
        for &(generated_procedure, generated_block, fact) in generated_facts {
            if generated_procedure == procedure
                && generated_block == source_block
                && fact < facts_per_procedure
            {
                let src = index(procedure, source_block, 0);
                let pos = scratch.cursor[src];
                columns[pos] = index(procedure, destination_block, fact) as u32;
                scratch.cursor[src] += 1;
            }
        }
    }
    for &(source_procedure, source_block, destination_procedure, destination_block) in inter_edges {
        if source_procedure >= procedure_count
            || destination_procedure >= procedure_count
            || source_block >= blocks_per_procedure
            || destination_block >= blocks_per_procedure
        {
            continue;
        }
        for fact in 0..facts_per_procedure {
            let src = index(source_procedure, source_block, fact);
            let pos = scratch.cursor[src];
            columns[pos] = index(destination_procedure, destination_block, fact) as u32;
            scratch.cursor[src] += 1;
        }
    }

    scratch.edges_flat.clear();
    scratch.killed.clear();
    scratch.gen_offsets.clear();
    scratch.gen_cursor.clear();
    scratch.gen_facts.clear();
    scratch.cursor.clear();

    Ok(())
}

/// Build the dense-index exploded IFDS graph as CSR.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn exploded_ifds_csr_witness(
    procedure_count: u32,
    blocks_per_procedure: u32,
    facts_per_procedure: u32,
    intra_edges: &[(u32, u32, u32)],
    inter_edges: &[(u32, u32, u32, u32)],
    generated_facts: &[(u32, u32, u32)],
    killed_facts: &[(u32, u32, u32)],
) -> (Vec<u32>, Vec<u32>) {
    let mut row_offsets = Vec::new();
    let mut columns = Vec::new();
    let mut scratch = ExplodedIfdsScratchWitness::default();
    if try_exploded_ifds_csr_witness_into(
        procedure_count,
        blocks_per_procedure,
        facts_per_procedure,
        intra_edges,
        inter_edges,
        generated_facts,
        killed_facts,
        &mut row_offsets,
        &mut columns,
        &mut scratch,
    )
    .is_err()
    {
        return (vec![0], Vec::new());
    }
    (row_offsets, columns)
}

/// Evaluates a topologically ordered d-DNNF circuit into caller-owned storage.
pub fn try_ddnnf_evaluate_witness_into(
    nodes: &[(u32, u32, u32)],
    node_variables: &[u32],
    children: &[u32],
    variable_assignments: &[u32],
    topological_order: &[u32],
    values: &mut Vec<u32>,
) -> Result<(), String> {
    if node_variables.len() != nodes.len() {
        return Err(format!(
            "d-DNNF node variable count {} does not match node count {}",
            node_variables.len(),
            nodes.len()
        ));
    }
    for &node in topological_order {
        let index = node as usize;
        let &(kind, child_offset, child_count) = nodes
            .get(index)
            .ok_or_else(|| format!("d-DNNF topological node {node} is out of bounds"))?;
        match kind {
            1 | 2 => {
                let variable = node_variables[index] as usize;
                if variable >= variable_assignments.len() {
                    return Err(format!(
                        "d-DNNF literal node {index} variable {variable} is outside assignment_count={}",
                        variable_assignments.len()
                    ));
                }
            }
            3 | 4 => {
                let start = child_offset as usize;
                let end = start
                    .checked_add(child_count as usize)
                    .ok_or_else(|| format!("d-DNNF child range overflows at node {index}"))?;
                let child_ids = children.get(start..end).ok_or_else(|| {
                    format!(
                        "d-DNNF node {index} child range {start}..{end} exceeds child_count={}",
                        children.len()
                    )
                })?;
                for &child in child_ids {
                    if child as usize >= nodes.len() {
                        return Err(format!(
                            "d-DNNF child node {child} is outside node_count={} at node {index}",
                            nodes.len()
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    if values.capacity() < nodes.len() {
        values.reserve(nodes.len().saturating_sub(values.len()));
    }
    values.clear();
    values.resize(nodes.len(), 0);
    for &node in topological_order {
        let index = node as usize;
        let &(kind, child_offset, child_count) = &nodes[index];
        values[index] = match kind {
            1 | 2 => {
                let variable = node_variables[index] as usize;
                let assignment = variable_assignments[variable];
                u32::from(if kind == 1 {
                    assignment == 1 || assignment == u32::MAX
                } else {
                    assignment == 0 || assignment == u32::MAX
                })
            }
            3 | 4 => {
                let start = child_offset as usize;
                let end = start + child_count as usize;
                let child_ids = &children[start..end];
                let mut accumulator = u32::from(kind == 3);
                for &child in child_ids {
                    let child = child as usize;
                    let value = values[child];
                    accumulator = if kind == 3 {
                        accumulator.checked_mul(value)
                    } else {
                        accumulator.checked_add(value)
                    }
                    .ok_or_else(|| format!("d-DNNF model count overflows at node {index}"))?;
                }
                accumulator
            }
            _ => 0,
        };
    }
    Ok(())
}

/// Evaluate a topologically ordered d-DNNF circuit.
///
/// Node tuples contain `(kind, child_offset, child_count)`. Kinds `1` and `2`
/// are positive and negative literals; kinds `3` and `4` are AND and OR.
///
/// # Errors
///
/// Returns an actionable diagnostic for malformed node, variable, child, or
/// topological-order indices and for model-count arithmetic overflow.
pub fn try_ddnnf_evaluate_witness(
    nodes: &[(u32, u32, u32)],
    node_variables: &[u32],
    children: &[u32],
    variable_assignments: &[u32],
    topological_order: &[u32],
) -> Result<Vec<u32>, String> {
    let mut values = Vec::with_capacity(nodes.len());
    try_ddnnf_evaluate_witness_into(
        nodes,
        node_variables,
        children,
        variable_assignments,
        topological_order,
        &mut values,
    )?;
    Ok(values)
}

/// Evaluate a valid topologically ordered d-DNNF circuit.
///
/// # Panics
///
/// Panics if the d-DNNF circuit structure, variable assignments, or topological order are invalid.
#[must_use]
pub fn ddnnf_evaluate_witness(
    nodes: &[(u32, u32, u32)],
    node_variables: &[u32],
    children: &[u32],
    variable_assignments: &[u32],
    topological_order: &[u32],
) -> Vec<u32> {
    try_ddnnf_evaluate_witness(
        nodes,
        node_variables,
        children,
        variable_assignments,
        topological_order,
    )
    .unwrap_or_else(|error| panic!("invalid d-DNNF witness input: {error}"))
}
