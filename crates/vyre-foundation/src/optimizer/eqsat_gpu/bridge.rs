//! The measured host-side bridge across the snapshot, packing and merge seams.
//!
//! The bridge accepts equivalences as if a device kernel produced them, applies
//! them through the canonical merge, and compares the extraction against an
//! independent CPU clone that applied the same equivalences. That proves the
//! seams without claiming a device ran.

use std::time::Instant;

use super::apply::apply_equivalences_to_egraph;
use super::error::GpuEGraphBridgeError;
use super::snapshot::{Equivalence, GpuEGraphSnapshot};
use crate::optimizer::eqsat::{
    try_extract_best_with_budget, EClassId, EGraph, ENodeLang, DEFAULT_EXTRACTION_ITER_BUDGET,
};

/// Measured bridge report for the CPU e-graph to GPU-columnar equivalence path.
///
/// The bridge accepts equivalences as if they were produced by a backend GPU
/// kernel, applies them through the canonical CPU merge API, and compares that
/// result against an independent CPU parity clone. This proves the snapshot,
/// packing, equivalence merge, and extraction seams without fabricating device
/// execution when no backend kernel was launched.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GpuEGraphBridgeReport {
    /// Number of rows in the compact GPU snapshot.
    pub snapshot_rows: usize,
    /// Number of child references in the compact GPU snapshot.
    pub snapshot_children: usize,
    /// Number of u32 words in the packed upload slab.
    pub device_words: usize,
    /// Number of e-class row groups in the packed upload slab.
    pub device_eclass_groups: usize,
    /// Input equivalence count.
    pub equivalences_requested: usize,
    /// Equivalences with valid e-class ids on the GPU-equivalence path.
    pub equivalences_valid: usize,
    /// State-changing merges on the GPU-equivalence path.
    pub equivalences_merged: usize,
    /// Additional rebuild unions on the GPU-equivalence path.
    pub rebuild_unions: usize,
    /// Equivalences with valid e-class ids on the CPU parity path.
    pub cpu_equivalences_valid: usize,
    /// State-changing merges on the CPU parity path.
    pub cpu_equivalences_merged: usize,
    /// Additional rebuild unions on the CPU parity path.
    pub cpu_rebuild_unions: usize,
    /// Snapshot construction time in nanoseconds.
    pub snapshot_ns: u64,
    /// Device-image packing time in nanoseconds.
    pub pack_ns: u64,
    /// CPU parity equivalence application time in nanoseconds.
    pub cpu_apply_ns: u64,
    /// GPU-equivalence application time in nanoseconds.
    pub gpu_apply_ns: u64,
    /// CPU parity extraction time in nanoseconds.
    pub cpu_extraction_ns: u64,
    /// GPU-equivalence extraction time in nanoseconds.
    pub gpu_extraction_ns: u64,
    /// Best extraction cost after the CPU parity apply.
    pub cpu_extraction_cost: Option<u64>,
    /// Best extraction cost after the GPU-equivalence apply.
    pub gpu_extraction_cost: Option<u64>,
    /// True when CPU parity and GPU-equivalence paths converge on the same
    /// apply report and extracted node/cost.
    pub recall_parity: bool,
    /// True when repeated snapshot packing produces identical class grouping
    /// columns for the same CPU e-graph.
    pub class_id_deterministic: bool,
}

/// Build a compact GPU e-graph image, apply backend-discovered equivalences
/// through the canonical CPU merge API, and prove extraction parity against an
/// independent CPU clone.
///
/// This function is the host-side bridge for GPU e-graph work: it measures the
/// CPU snapshot and packing cost, records the time to apply equivalences
/// returned by a backend, and checks that the resulting extraction matches a
/// CPU parity path that applied the same equivalences.
///
/// # Errors
///
/// Returns [`GpuEGraphBridgeError`] if snapshot construction, device packing, or
/// parity extraction fails.
pub fn bridge_equivalence_batch_with_report<L, F, S, C>(
    egraph: &mut EGraph<L>,
    root: EClassId,
    op_name: F,
    equivalences: &[Equivalence],
    cost_fn: C,
) -> Result<GpuEGraphBridgeReport, GpuEGraphBridgeError>
where
    L: ENodeLang,
    F: Fn(&L) -> S,
    S: AsRef<str>,
    C: Fn(&L) -> u64 + Copy,
{
    let snapshot_start = Instant::now();
    let snapshot = GpuEGraphSnapshot::try_from_egraph_with(egraph, |node| op_name(node))?;
    let snapshot_ns = elapsed_nonzero_ns(snapshot_start);

    let deterministic_snapshot =
        GpuEGraphSnapshot::try_from_egraph_with(egraph, |node| op_name(node))?;

    let pack_start = Instant::now();
    let image = snapshot.try_pack_device_image()?;
    let pack_ns = elapsed_nonzero_ns(pack_start);
    let deterministic_image = deterministic_snapshot.try_pack_device_image()?;
    let class_id_deterministic = image.group_eclass_ids() == deterministic_image.group_eclass_ids()
        && image.group_offsets() == deterministic_image.group_offsets()
        && image.group_rows() == deterministic_image.group_rows();

    let mut cpu_parity = egraph.clone();
    let cpu_apply_start = Instant::now();
    let cpu_apply = apply_equivalences_to_egraph(&mut cpu_parity, equivalences);
    let cpu_apply_ns = elapsed_nonzero_ns(cpu_apply_start);

    let gpu_apply_start = Instant::now();
    let gpu_apply = apply_equivalences_to_egraph(egraph, equivalences);
    let gpu_apply_ns = elapsed_nonzero_ns(gpu_apply_start);

    let cpu_extraction_start = Instant::now();
    let cpu_extraction = try_extract_best_with_budget(
        &cpu_parity,
        root,
        |node| cost_fn(node),
        DEFAULT_EXTRACTION_ITER_BUDGET,
    )?;
    let cpu_extraction_ns = elapsed_nonzero_ns(cpu_extraction_start);

    let gpu_extraction_start = Instant::now();
    let gpu_extraction = try_extract_best_with_budget(
        egraph,
        root,
        |node| cost_fn(node),
        DEFAULT_EXTRACTION_ITER_BUDGET,
    )?;
    let gpu_extraction_ns = elapsed_nonzero_ns(gpu_extraction_start);

    let cpu_extraction_cost = cpu_extraction.best.as_ref().map(|(_, cost)| *cost);
    let gpu_extraction_cost = gpu_extraction.best.as_ref().map(|(_, cost)| *cost);
    let recall_parity = cpu_apply == gpu_apply && cpu_extraction.best == gpu_extraction.best;

    Ok(GpuEGraphBridgeReport {
        snapshot_rows: snapshot.node_count(),
        snapshot_children: snapshot.child_count(),
        device_words: image.words().len(),
        device_eclass_groups: image.layout().eclass_group_count(),
        equivalences_requested: gpu_apply.requested,
        equivalences_valid: gpu_apply.valid,
        equivalences_merged: gpu_apply.merged,
        rebuild_unions: gpu_apply.rebuild_unions,
        cpu_equivalences_valid: cpu_apply.valid,
        cpu_equivalences_merged: cpu_apply.merged,
        cpu_rebuild_unions: cpu_apply.rebuild_unions,
        snapshot_ns,
        pack_ns,
        cpu_apply_ns,
        gpu_apply_ns,
        cpu_extraction_ns,
        gpu_extraction_ns,
        cpu_extraction_cost,
        gpu_extraction_cost,
        recall_parity,
        class_id_deterministic,
    })
}

fn elapsed_nonzero_ns(start: Instant) -> u64 {
    let ns = start.elapsed().as_nanos();
    u64::try_from(ns).unwrap_or(u64::MAX).max(1)
}
