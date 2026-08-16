//! Host reference for the motif witness.
//!
//! The module declaration carries the gate. Every item here is a host oracle,
//! so one gate on the declaration says that once instead of repeating it per
//! item, where an omission ships a CPU implementation in a device build.

use super::layout::validate_motif_inputs;
use super::pattern::MotifEdge;

/// CPU reference: return one byte-per-node witness set where `1`
/// means the node participates in a complete motif match.
#[must_use]
pub fn cpu_ref(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
) -> Vec<u32> {
    let mut participants = Vec::new();
    try_cpu_ref_into(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
        &mut participants,
    )
    .unwrap_or_else(|err| panic!("motif CPU oracle received malformed input. {err}"));
    participants
}

/// Fallible CPU reference into caller-owned witness storage.
pub fn try_cpu_ref_into(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
    participants: &mut Vec<u32>,
) -> Result<(), String> {
    let layout = validate_motif_inputs(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
    )?;
    crate::plumbing::host::scratch::reserve_items(
        participants,
        layout.output_words,
        "motif CPU oracle",
        "motif witness output",
    )?;
    participants.clear();
    participants.resize(layout.output_words, 0);
    if !motif_all_edges_present(edge_offsets, edge_targets, edge_kind_mask, motif_edges) {
        return Ok(());
    }
    for motif_edge in motif_edges {
        if let Some(hit) = participants.get_mut(motif_edge.from as usize) {
            *hit = 1;
        }
        if let Some(hit) = participants.get_mut(motif_edge.to as usize) {
            *hit = 1;
        }
    }
    Ok(())
}

/// CPU reference into caller-owned witness storage.
pub fn cpu_ref_into(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
    participants: &mut Vec<u32>,
) {
    try_cpu_ref_into(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
        participants,
    )
    .unwrap_or_else(|err| panic!("motif CPU oracle received malformed input. {err}"));
}

/// Return true iff the complete motif exists.
///
/// This avoids allocating a full witness vector for existence checks.
#[must_use]
pub fn cpu_ref_matches(
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
) -> bool {
    motif_all_edges_present(edge_offsets, edge_targets, edge_kind_mask, motif_edges)
}

/// Count distinct nodes participating in a complete motif match.
///
/// This avoids materializing the witness vector when callers only need a
/// scheduling signal.
#[must_use]
pub fn cpu_ref_participation_count(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
) -> u32 {
    try_cpu_ref_participation_count(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
    )
    .unwrap_or_else(|err| panic!("motif participation oracle received malformed input. {err}"))
}

/// Caller-owned workspace for motif CPU reference helpers.
#[derive(Debug, Default, Clone)]
pub struct MotifCpuScratch {
    /// Distinct endpoint scratch used by participation-count queries.
    pub endpoints: Vec<u32>,
}

impl MotifCpuScratch {
    /// Create an empty reusable motif workspace.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fallible count of distinct nodes participating in a complete motif match.
pub fn try_cpu_ref_participation_count(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
) -> Result<u32, String> {
    let mut scratch = MotifCpuScratch::default();
    try_cpu_ref_participation_count_with_scratch(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
        &mut scratch,
    )
}

/// Fallible participation count using caller-owned endpoint scratch.
///
/// Validation happens before the scratch vector is touched. For valid inputs,
/// the scratch vector is cleared and reused even when the complete motif is not
/// present, so stale endpoints cannot leak into later proof cases.
pub fn try_cpu_ref_participation_count_with_scratch(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
    scratch: &mut MotifCpuScratch,
) -> Result<u32, String> {
    validate_motif_inputs(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
    )?;
    let endpoint_count = motif_edges
        .len()
        .checked_mul(2)
        .ok_or_else(|| "Fix: motif endpoint count overflows usize.".to_string())?;
    scratch
        .endpoints
        .try_reserve(endpoint_count)
        .map_err(|error| {
            format!(
            "Fix: motif participation oracle could not reserve {endpoint_count} endpoints: {error}"
        )
        })?;
    scratch.endpoints.clear();
    if !motif_all_edges_present(edge_offsets, edge_targets, edge_kind_mask, motif_edges) {
        return Ok(0);
    }
    for motif_edge in motif_edges {
        if motif_edge.from < node_count {
            scratch.endpoints.push(motif_edge.from);
        }
        if motif_edge.to < node_count {
            scratch.endpoints.push(motif_edge.to);
        }
    }
    scratch.endpoints.sort_unstable();
    scratch.endpoints.dedup();
    u32::try_from(scratch.endpoints.len()).map_err(|error| {
        format!("Fix: motif participation count does not fit u32 after deduplication: {error}")
    })
}

fn motif_all_edges_present(
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
) -> bool {
    for motif_edge in motif_edges {
        let Some(start) = edge_offsets.get(motif_edge.from as usize).copied() else {
            return false;
        };
        let Some(end) = edge_offsets.get(motif_edge.from as usize + 1).copied() else {
            return false;
        };
        let start = start as usize;
        let end = end as usize;
        let mut found = false;
        for edge_idx in start..end {
            let Some(dst) = edge_targets.get(edge_idx).copied() else {
                break;
            };
            let Some(kind) = edge_kind_mask.get(edge_idx).copied() else {
                break;
            };
            if dst == motif_edge.to && (kind & motif_edge.kind_mask) != 0 {
                found = true;
                break;
            }
        }
        if !found {
            return false;
        }
    }
    true
}
