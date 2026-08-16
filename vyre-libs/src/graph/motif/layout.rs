//! Validated dispatch layout, and the CSR and witness checks that produce it.

use super::pattern::MotifEdge;

/// Validated motif dispatch layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MotifLayout {
    /// Number of graph nodes and output words.
    pub node_count: u32,
    /// Number of graph nodes and output words, widened for host buffer sizing.
    pub output_words: usize,
    /// Number of physical CSR edges.
    pub edge_count: u32,
    /// Number of u32 words required by physical edge buffers after padding.
    pub edge_storage_words: usize,
    /// Number of requested motif edges.
    pub motif_edge_count: u32,
}

/// Validate the public CSR inputs consumed by the motif primitive.
///
/// Returns the exact edge count declared by `edge_offsets[node_count]`, so
/// dispatch wrappers can pad zero-edge buffers without duplicating CSR
/// validation logic.
///
/// # Errors
///
/// Returns an actionable diagnostic for malformed row offsets, edge arrays, or
/// out-of-range destinations.
pub fn validate_csr_inputs(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
) -> Result<MotifLayout, String> {
    validate_motif_inputs(node_count, edge_offsets, edge_targets, edge_kind_mask, &[])
}

/// Validate the public CSR and motif inputs consumed by the motif primitive.
///
/// # Errors
///
/// Returns an actionable diagnostic for malformed row offsets, edge arrays,
/// out-of-range destinations, or motif edge counts that exceed u32 dispatch
/// metadata.
pub fn validate_motif_inputs(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
) -> Result<MotifLayout, String> {
    let expected_offsets = (node_count as usize).checked_add(1).ok_or_else(|| {
        format!("Fix: motif node_count + 1 overflows usize for node_count={node_count}.")
    })?;
    if edge_offsets.len() != expected_offsets {
        return Err(format!(
            "Fix: motif requires edge_offsets.len() == node_count + 1, got len={}, node_count={node_count}.",
            edge_offsets.len()
        ));
    }
    if edge_targets.len() != edge_kind_mask.len() {
        return Err(format!(
            "Fix: motif requires edge_targets.len() == edge_kind_mask.len(), got {} vs {}.",
            edge_targets.len(),
            edge_kind_mask.len()
        ));
    }
    if let Some(&first) = edge_offsets.first() {
        if first != 0 {
            return Err(format!(
                "Fix: motif requires edge_offsets[0] == 0, got {first}."
            ));
        }
    }
    for (index, pair) in edge_offsets.windows(2).enumerate() {
        if pair[0] > pair[1] {
            return Err(format!(
                "Fix: motif offsets must be monotonic; offsets[{index}]={} > offsets[{}]={}.",
                pair[0],
                index + 1,
                pair[1]
            ));
        }
    }
    let edge_count = edge_offsets[expected_offsets - 1] as usize;
    if edge_targets.len() != edge_count {
        return Err(format!(
            "Fix: motif final offset declares edge_count={edge_count}, but targets_len={} and kind_mask_len={}.",
            edge_targets.len(),
            edge_kind_mask.len()
        ));
    }
    for (index, &target) in edge_targets.iter().enumerate() {
        if target >= node_count {
            return Err(format!(
                "Fix: motif edge_targets[{index}]={target} is outside node_count {node_count}."
            ));
        }
    }
    for (index, motif_edge) in motif_edges.iter().enumerate() {
        if motif_edge.from >= node_count {
            return Err(format!(
                "Fix: motif_edges[{index}].from={} is outside node_count {node_count}.",
                motif_edge.from
            ));
        }
        if motif_edge.to >= node_count {
            return Err(format!(
                "Fix: motif_edges[{index}].to={} is outside node_count {node_count}.",
                motif_edge.to
            ));
        }
    }
    let edge_count = u32::try_from(edge_count)
        .map_err(|_| format!("Fix: motif edge count {edge_count} exceeds u32 index space."))?;
    let motif_edge_count = u32::try_from(motif_edges.len()).map_err(|_| {
        format!(
            "Fix: motif edge pattern count {} exceeds u32 index space.",
            motif_edges.len()
        )
    })?;
    Ok(MotifLayout {
        node_count,
        output_words: node_count as usize,
        edge_count,
        edge_storage_words: edge_targets.len().max(1),
        motif_edge_count,
    })
}

/// Count nonzero witness entries using the primitive's u32 result contract.
///
/// # Errors
///
/// Returns an actionable diagnostic if the witness vector is too large to
/// report with the primitive's u32 count metadata.
pub fn count_witness_participants(witness: &[u32]) -> Result<u32, String> {
    let count = witness.iter().filter(|&&value| value != 0).count();
    u32::try_from(count)
        .map_err(|_| format!("Fix: motif witness participant count {count} exceeds u32::MAX."))
}

/// Validate the primitive's u32 witness output contract.
///
/// # Errors
///
/// Returns an actionable diagnostic if the backend returns the wrong number of
/// witness words or any non-boolean witness entry.
pub fn validate_motif_witness(layout: MotifLayout, witness: &[u32]) -> Result<(), String> {
    if witness.len() != layout.output_words {
        return Err(format!(
            "Fix: motif witness expected {} word(s), got {}.",
            layout.output_words,
            witness.len()
        ));
    }
    for (index, &value) in witness.iter().enumerate() {
        if value > 1 {
            return Err(format!(
                "Fix: motif witness[{index}]={value} is not boolean; expected 0 or 1."
            ));
        }
    }
    Ok(())
}
