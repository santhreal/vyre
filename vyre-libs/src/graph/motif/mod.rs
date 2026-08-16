//! `motif`  -  intersect edge witnesses for a small graph pattern.
//!
//! Each motif edge is checked independently against the canonical
//! ProgramGraph CSR. If every requested motif edge exists, every
//! endpoint participating in the motif is marked in the final witness.

use crate::graph::program_graph::BINDING_PRIMITIVE_START;

#[cfg(any(test, feature = "cpu-parity"))]
mod cpu_ref;
mod layout;
mod pattern;
mod plan;
mod program;
mod registry;

#[cfg(test)]
mod tests;

pub use layout::{
    count_witness_participants, validate_csr_inputs, validate_motif_inputs,
    validate_motif_witness, MotifLayout,
};
#[cfg(any(test, feature = "cpu-parity"))]
pub use cpu_ref::{
    cpu_ref, cpu_ref_into, cpu_ref_matches, cpu_ref_participation_count, try_cpu_ref_into,
    try_cpu_ref_participation_count, try_cpu_ref_participation_count_with_scratch,
    MotifCpuScratch,
};
pub use pattern::{MotifEdge, TWO_EDGE_PATH_MOTIF};
pub use plan::{
    plan_motif_dispatch, plan_motif_launch, MotifDispatchPlan, MotifLaunchPlan,
    MotifProgramCacheKey, MotifStaticInputKey,
};
pub use program::motif;

/// Canonical op id.
pub const OP_ID: &str = "vyre-primitives::graph::motif";
/// Canonical binding index for motif scratch hits.
pub const MOTIF_HITS_BUFFER: u32 = BINDING_PRIMITIVE_START;
/// Canonical binding index for the public witness output.
pub const MOTIF_WITNESS_OUT_BUFFER: u32 = BINDING_PRIMITIVE_START + 1;
/// Motif matching is serial over the small pattern by construction.
pub const MOTIF_WORKGROUP_SIZE: [u32; 3] = [1, 1, 1];
/// Canonical motif dispatch grid.
pub const MOTIF_DISPATCH_GRID: [u32; 3] = [1, 1, 1];
