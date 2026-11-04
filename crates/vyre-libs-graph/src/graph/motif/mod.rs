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
