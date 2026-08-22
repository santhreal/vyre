//! `motif`  -  intersect edge witnesses for a small graph pattern.
//!
//! Each motif edge is checked independently against the canonical
//! ProgramGraph CSR. If every requested motif edge exists, every
//! endpoint participating in the motif is marked in the final witness.

use crate::graph::program_graph::BINDING_PRIMITIVE_START;

mod layout;
mod pattern;
mod plan;
mod program;
mod registry;

pub use layout::{validate_csr_inputs, validate_motif_inputs, validate_motif_witness, MotifLayout};
pub use pattern::{MotifEdge, TWO_EDGE_PATH_MOTIF};
pub use plan::{
    plan_motif_dispatch, plan_motif_launch, MotifDispatchPlan, MotifLaunchPlan,
    MotifProgramCacheKey, MotifStaticInputKey,
};
pub use program::motif;

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::graph::motif";
/// Canonical binding index for motif scratch hits.
pub const MOTIF_HITS_BUFFER: u32 = BINDING_PRIMITIVE_START;
/// Canonical binding index for the public witness output.
pub const MOTIF_WITNESS_OUT_BUFFER: u32 = BINDING_PRIMITIVE_START + 1;
/// Motif matching is serial over the small pattern by construction.
pub const MOTIF_WORKGROUP_SIZE: [u32; 3] = [1, 1, 1];
/// Canonical motif dispatch grid.
pub const MOTIF_DISPATCH_GRID: [u32; 3] = [1, 1, 1];

#[cfg(test)]
fn try_cpu_ref_into(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
    out: &mut Vec<u32>,
) -> Result<(), String> {
    validate_motif_inputs(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
    )?;
    let edges: Vec<(u32, u32, u32)> = motif_edges
        .iter()
        .map(|e| (e.from, e.kind_mask, e.to))
        .collect();
    vyre_reference::composition_witness::motif_witness_into(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        &edges,
        out,
    );
    Ok(())
}

#[cfg(test)]
fn cpu_ref(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
) -> Vec<u32> {
    let mut out = Vec::new();
    try_cpu_ref_into(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
        &mut out,
    )
    .expect("cpu_ref failed");
    out
}

#[cfg(test)]
fn cpu_ref_matches(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
) -> bool {
    motif_edges.is_empty()
        || cpu_ref(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            motif_edges,
        )
        .iter()
        .any(|&x| x != 0)
}

#[cfg(test)]
fn cpu_ref_participation_count(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
) -> u32 {
    let hits = cpu_ref(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
    );
    hits.iter().filter(|&&x| x != 0).count() as u32
}
/// Match motif pattern against graph and return per-node participation vector.
#[cfg(test)]
#[must_use]
pub(crate) fn match_motif(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
) -> Vec<u32> {
    cpu_ref(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
    )
}

/// Fallible match motif pattern returning per-node participation vector.
#[cfg(test)]
pub(crate) fn try_match_motif(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    try_cpu_ref_into(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
        &mut out,
    )?;
    Ok(out)
}

/// Return true if the graph contains any match for the motif pattern.
#[cfg(test)]
#[must_use]
pub(crate) fn motif_matches(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
) -> bool {
    cpu_ref_matches(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
    )
}

/// Fallible check whether the graph contains any match for the motif pattern.
#[cfg(test)]
pub(crate) fn try_motif_matches(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
) -> Result<bool, String> {
    let hits = try_match_motif(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
    )?;
    Ok(hits.iter().any(|&x| x != 0))
}

/// Count nodes participating in a match for the motif pattern.
#[cfg(test)]
#[must_use]
pub(crate) fn motif_participation_count(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
) -> u32 {
    cpu_ref_participation_count(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
    )
}

/// Fallible count of nodes participating in a match for the motif pattern.
#[cfg(test)]
pub(crate) fn try_motif_participation_count(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
) -> Result<u32, String> {
    let hits = try_match_motif(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
    )?;
    Ok(hits.iter().filter(|&&x| x != 0).count() as u32)
}

#[cfg(test)]
mod oracle_contracts {
    use super::*;

    /// WHY: malformed inputs must fail before caller-owned witness storage is changed.
    #[test]
    fn checked_witness_preserves_output_on_validation_failure() {
        let mut output = vec![0xCAFE_BABE, 0xDEAD_BEEF];
        let original = output.clone();
        let error = try_cpu_ref_into(
            2,
            &[0, 2, 1],
            &[1],
            &[1],
            &[MotifEdge {
                from: 0,
                kind_mask: 1,
                to: 1,
            }],
            &mut output,
        )
        .expect_err("non-monotonic CSR offsets must fail");
        assert!(error.contains("monotonic"), "{error}");
        assert_eq!(output, original);
    }
}
