//! Region-tree coarse-graining via #58 Mori-Zwanzig projection (#58 self-consumer).
//!
//! Closes the recursion thesis for #58  -  `mz_project_step` ships to
//! user dialects (climate modeling, scientific ML model reduction)
//! AND derives vyre's coarse-grained dispatch view of its own Region
//! tree.
//!
//! # The self-use
//!
//! Vyre's full dispatch graph at workspace scale is millions of
//! Regions. Most optimizer passes don't need that resolution  -  they
//! need a coarse view that preserves the dispatch structure (memory
//! pressure, sync points, fusion eligibility) while dropping leaf
//! detail. Mori-Zwanzig (1965) gives an EXACT reduction with a
//! memory kernel that captures whatever the projection drops.
//!
//! Concretely: cluster Regions into K macro-nodes via #2 sinkhorn
//! divergence (the substrate clustering primitive #31 ships).
//! Construct a projection matrix P that averages within each cluster.
//! Mori-Zwanzig's projection step yields the coarse dispatch
//! dynamics; the memory kernel encodes how within-cluster detail
//! influences cross-cluster decisions over time (= over subsequent
//! optimizer passes).
//!
//! This module owns the coarse-graining projection step. Memory-kernel
//! recursion composes with this projection when a pass framework
//! supplies historical state.
//!
//! # Algorithm
//!
//! ```text
//! P[i,j] = 1/|cluster(i)|  if cluster(i) == cluster(j) else 0
//! coarse_state = P · state  (one mz_project_step dispatch)
//! ```
//!
//! `state[i]` is any per-Region scalar feature (memory residency,
//! dispatch latency, fusion-eligibility score). The projected
//! `coarse_state[i]` is the cluster-averaged value at i.
//!
//! # Why this matters at scale
//!
//! At 1M Regions, naive full-resolution analysis is O(N²) memory in
//! the worst case (#19 polyhedral fusion considers all pairs).
//! Mori-Zwanzig coarsening to K macro-nodes drops that to O(K²) at
//! the cost of an exactly-quantified projection error  -  the memory
//! kernel. Combined with #51 FMM hierarchical compression on the
//! coarse system, full workspace analysis stays tractable.

use crate::dispatch_buffers::{
    ceil_div_u32, checked_square_cells, decode_u32_output_exact, ensure_input_slots,
    write_u32_slice_le_bytes, write_zero_bytes,
};
use crate::math::mori_zwanzig::mz_project_step;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};
#[cfg(test)]
use vyre_reference::composition_witness::{
    cluster_projection_matrix_witness_into, mori_zwanzig_coarsen_via_clustering_witness_into,
    mori_zwanzig_project_witness_into as reference_mz_project_step_into,
};

/// Caller-owned dispatch scratch for fixed-point Mori-Zwanzig projection.
#[derive(Debug, Default)]
pub struct RegionCoarsenGpuScratch {
    inputs: Vec<Vec<u8>>,
}

/// Reusable buffers for Mori-Zwanzig region coarsening.
#[derive(Debug, Default)]
#[cfg(test)]
pub(crate) struct RegionCoarsenScratch {
    cluster_sizes: Vec<u32>,
    projection: Vec<f64>,
    coarse_state: Vec<f64>,
}

#[cfg(test)]
impl RegionCoarsenScratch {
    /// Create empty reusable region-coarsening scratch.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn projection_ptr(&self) -> *const f64 {
        self.projection.as_ptr()
    }
}

/// Build a cluster-projection matrix from a cluster-assignment
/// vector. `assignments[i]` is the cluster id (0..k) of Region i.
/// Returns a row-major n*n matrix where row i is uniform over its
/// cluster's column indices.
///
/// # Panics
///
/// Panics if any assignment exceeds k-1.
#[must_use]
#[cfg(test)]
pub(crate) fn cluster_projection_matrix(assignments: &[u32], n: u32, k: u32) -> Vec<f64> {
    let mut scratch = RegionCoarsenScratch::new();
    cluster_projection_matrix_into(assignments, n, k, &mut scratch).to_vec()
}

/// Build a cluster-projection matrix using caller-owned scratch.
#[must_use]
#[cfg(test)]
pub(crate) fn cluster_projection_matrix_into<'a>(
    assignments: &[u32],
    n: u32,
    k: u32,
    scratch: &'a mut RegionCoarsenScratch,
) -> &'a [f64] {
    use crate::telemetry::{bump, mori_zwanzig_region_coarsen_calls};
    bump(&mori_zwanzig_region_coarsen_calls);
    cluster_projection_matrix_witness_into(
        assignments,
        n,
        k,
        &mut scratch.cluster_sizes,
        &mut scratch.projection,
    );
    &scratch.projection
}

/// Apply Mori-Zwanzig projection to coarse-grain a per-Region scalar
/// feature vector. Returns the coarse-grained state where each
/// Region's value is replaced by its cluster-mean.
///
/// # Panics
///
/// Panics if `state.len() != n`.
#[must_use]
#[cfg(test)]
pub(crate) fn coarsen_region_state(p_matrix: &[f64], state: &[f64], n: u32) -> Vec<f64> {
    let mut out = Vec::new();
    reference_coarsen_region_state_into(p_matrix, state, n, &mut out);
    out
}

/// Apply Mori-Zwanzig projection using caller-owned output storage.
#[cfg(test)]
pub(crate) fn reference_coarsen_region_state_into(
    p_matrix: &[f64],
    state: &[f64],
    n: u32,
    out: &mut Vec<f64>,
) {
    use crate::telemetry::{bump, mori_zwanzig_region_coarsen_calls};
    bump(&mori_zwanzig_region_coarsen_calls);
    reference_mz_project_step_into(p_matrix, state, n, out);
}
/// Primitive-native fixed-point production path for applying a
/// Mori-Zwanzig projection to a per-Region scalar state.
///
/// `p_matrix_fixed` is a row-major `n x n` 16.16 projector and
/// `state_fixed` is a length-`n` 16.16 feature vector. Callers that derive
/// projectors from clustering assignments should quantize once at that
/// representation boundary and keep repeated projection steps on the
/// dispatcher path.
///
/// # Errors
///
/// Returns [`DispatchError`] when shapes are invalid, lane counts overflow,
/// or the backend returns malformed output.
pub fn coarsen_region_state_fixed_via(
    dispatcher: &impl ProgramDispatcher,
    p_matrix_fixed: &[u32],
    state_fixed: &[u32],
    n: u32,
) -> Result<Vec<u32>, DispatchError> {
    let mut out = Vec::new();
    coarsen_region_state_fixed_via_into(dispatcher, p_matrix_fixed, state_fixed, n, &mut out)?;
    Ok(out)
}

/// Primitive-native fixed-point Mori-Zwanzig projection into caller-owned
/// output storage.
///
/// # Errors
///
/// Returns [`DispatchError`] when shape checks or backend execution fail.
pub fn coarsen_region_state_fixed_via_into(
    dispatcher: &impl ProgramDispatcher,
    p_matrix_fixed: &[u32],
    state_fixed: &[u32],
    n: u32,
    out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let mut scratch = RegionCoarsenGpuScratch::default();
    coarsen_region_state_fixed_via_with_scratch_into(
        dispatcher,
        p_matrix_fixed,
        state_fixed,
        n,
        &mut scratch,
        out,
    )
}

/// Primitive-native fixed-point Mori-Zwanzig projection using caller-owned
/// dispatch scratch.
///
/// # Errors
///
/// Returns [`DispatchError`] when shape checks or backend execution fail.
pub fn coarsen_region_state_fixed_via_with_scratch_into(
    dispatcher: &impl ProgramDispatcher,
    p_matrix_fixed: &[u32],
    state_fixed: &[u32],
    n: u32,
    scratch: &mut RegionCoarsenGpuScratch,
    out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    use crate::telemetry::{bump, mori_zwanzig_region_coarsen_calls};
    bump(&mori_zwanzig_region_coarsen_calls);

    let cells = checked_square_cells(n, "coarsen_region_state_fixed_via")?;
    let cells_u32 = u32::try_from(cells).map_err(|_| {
        DispatchError::BadInputs(format!(
            "Fix: coarsen_region_state_fixed_via n*n exceeds the primitive u32 lane limit for n={n}."
        ))
    })?;
    if p_matrix_fixed.len() != cells {
        return Err(DispatchError::BadInputs(format!(
            "Fix: coarsen_region_state_fixed_via requires p_matrix_fixed.len() == n*n, got len={}, n={n}, n*n={cells}.",
            p_matrix_fixed.len()
        )));
    }
    if state_fixed.len() != n as usize {
        return Err(DispatchError::BadInputs(format!(
            "Fix: coarsen_region_state_fixed_via requires state_fixed.len() == n, got len={}, n={n}.",
            state_fixed.len()
        )));
    }

    let program = mz_project_step("p_matrix", "state", "out", n);
    let out_bytes = (n as usize)
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            DispatchError::BadInputs(format!(
                "Fix: coarsen_region_state_fixed_via n={n} overflows output byte count."
            ))
        })?;
    ensure_input_slots(&mut scratch.inputs, 3);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], p_matrix_fixed);
    write_u32_slice_le_bytes(&mut scratch.inputs[1], state_fixed);
    write_zero_bytes(&mut scratch.inputs[2], out_bytes);
    let outputs = dispatcher.dispatch(
        &program,
        &scratch.inputs,
        Some([ceil_div_u32(cells_u32, 256), 1, 1]),
    )?;
    if outputs.is_empty() {
        return Err(DispatchError::BackendError(format!(
            "Fix: coarsen_region_state_fixed_via expected at least one output buffer, got {}.",
            outputs.len()
        )));
    }
    decode_u32_output_exact(
        &outputs[0],
        n as usize,
        "coarsen_region_state_fixed_via",
        out,
    )
}

/// Convenience: derive the projection AND apply it in one step.
#[must_use]
#[cfg(test)]
pub(crate) fn coarsen_via_clustering(
    state: &[f64],
    assignments: &[u32],
    n: u32,
    k: u32,
) -> Vec<f64> {
    let mut scratch = RegionCoarsenScratch::new();
    reference_coarsen_via_clustering_into(state, assignments, n, k, &mut scratch).to_vec()
}

/// Derive and apply the projection using caller-owned scratch.
#[must_use]
#[cfg(test)]
pub(crate) fn reference_coarsen_via_clustering_into<'a>(
    state: &[f64],
    assignments: &[u32],
    n: u32,
    k: u32,
    scratch: &'a mut RegionCoarsenScratch,
) -> &'a [f64] {
    use crate::telemetry::{bump, mori_zwanzig_region_coarsen_calls};
    bump(&mori_zwanzig_region_coarsen_calls);
    mori_zwanzig_coarsen_via_clustering_witness_into(
        state,
        assignments,
        n,
        k,
        &mut scratch.cluster_sizes,
        &mut scratch.projection,
        &mut scratch.coarse_state,
    );
    &scratch.coarse_state
}

#[cfg(test)]
mod tests {
    #![allow(clippy::identity_op, clippy::erasing_op)]
    use super::*;
    use crate::dispatch_buffers::u32_slice_to_le_bytes;
    use vyre_foundation::ir::Program;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn projection_matrix_normalizes_within_cluster() {
        // 4 nodes, 2 clusters: {0,1} and {2,3}.
        let assignments = vec![0u32, 0, 1, 1];
        let p = cluster_projection_matrix(&assignments, 4, 2);
        // Row 0: uniform over cols 0+1, zero on 2+3.
        assert!(approx_eq(p[0], 0.5));
        assert!(approx_eq(p[1], 0.5));
        assert!(approx_eq(p[2], 0.0));
        assert!(approx_eq(p[3], 0.0));
        // Row 2: uniform over cols 2+3.
        assert!(approx_eq(p[2 * 4 + 2], 0.5));
        assert!(approx_eq(p[2 * 4 + 3], 0.5));
        assert!(approx_eq(p[2 * 4 + 0], 0.0));
    }

    #[test]
    fn coarsening_replaces_with_cluster_mean() {
        // 4 nodes, 2 clusters; state values [10, 20, 100, 200].
        // After coarsening: cluster 0 mean = 15, cluster 1 mean = 150.
        let assignments = vec![0u32, 0, 1, 1];
        let state = vec![10.0, 20.0, 100.0, 200.0];
        let coarse = coarsen_via_clustering(&state, &assignments, 4, 2);
        assert!(approx_eq(coarse[0], 15.0));
        assert!(approx_eq(coarse[1], 15.0));
        assert!(approx_eq(coarse[2], 150.0));
        assert!(approx_eq(coarse[3], 150.0));
    }

    #[test]
    fn singleton_clusters_preserve_state() {
        // Each Region is its own cluster  -  projection is identity.
        let assignments = vec![0u32, 1, 2, 3];
        let state = vec![10.0, 20.0, 30.0, 40.0];
        let coarse = coarsen_via_clustering(&state, &assignments, 4, 4);
        for (a, b) in state.iter().zip(coarse.iter()) {
            assert!(approx_eq(*a, *b));
        }
    }

    #[test]
    fn single_global_cluster_yields_uniform_mean() {
        // All Regions in one cluster  -  every coarse cell = global mean.
        let assignments = vec![0u32; 4];
        let state = vec![10.0, 20.0, 30.0, 40.0];
        let coarse = coarsen_via_clustering(&state, &assignments, 4, 1);
        let mean = (10.0 + 20.0 + 30.0 + 40.0) / 4.0;
        for v in coarse {
            assert!(approx_eq(v, mean));
        }
    }

    #[test]
    #[should_panic(expected = "exceeds cluster count")]
    fn rejects_out_of_range_assignment() {
        let assignments = vec![0u32, 1, 5, 0];
        let _projection = cluster_projection_matrix(&assignments, 4, 2);
    }

    #[test]
    fn coarsen_via_clustering_into_reuses_projection_storage() {
        let assignments = vec![0u32, 0, 1, 1];
        let state = vec![10.0, 20.0, 100.0, 200.0];
        let mut scratch = RegionCoarsenScratch::new();

        let first = reference_coarsen_via_clustering_into(&state, &assignments, 4, 2, &mut scratch)
            .to_vec();
        let ptr = scratch.projection_ptr();
        let second =
            reference_coarsen_via_clustering_into(&state, &assignments, 4, 2, &mut scratch)
                .to_vec();

        assert!(approx_eq(first[0], 15.0));
        assert_eq!(first, second);
        assert_eq!(scratch.projection_ptr(), ptr);
    }

    struct MoriDispatcher;

    impl ProgramDispatcher for MoriDispatcher {
        fn dispatch(
            &self,
            _program: &Program,
            inputs: &[Vec<u8>],
            grid_override: Option<[u32; 3]>,
        ) -> Result<Vec<Vec<u8>>, DispatchError> {
            assert_eq!(grid_override, Some([1, 1, 1]));
            assert_eq!(inputs.len(), 3);
            let p = crate::dispatch_buffers::read_u32s(&inputs[0]);
            let state = crate::dispatch_buffers::read_u32s(&inputs[1]);
            assert_eq!(inputs[2].len(), state.len() * std::mem::size_of::<u32>());
            let n = state.len();
            let mut out = vec![0u32; n];
            for i in 0..n {
                let mut acc = 0u32;
                for j in 0..n {
                    acc =
                        acc.saturating_add(((p[i * n + j] as u64 * state[j] as u64) >> 16) as u32);
                }
                out[i] = acc;
            }
            Ok(vec![u32_slice_to_le_bytes(&out)])
        }
    }

    #[test]
    fn fixed_via_dispatches_projection_step() {
        let one = 1u32 << 16;
        let half = 1u32 << 15;
        let out = coarsen_region_state_fixed_via(
            &MoriDispatcher,
            &[half, half, half, half],
            &[10 * one, 20 * one],
            2,
        )
        .unwrap();
        assert_eq!(out, vec![15 * one, 15 * one]);
    }

    #[test]

    fn fixed_via_rejects_shape_mismatch() {
        let err =
            coarsen_region_state_fixed_via(&MoriDispatcher, &[1, 0, 0], &[1, 1], 2).unwrap_err();
        assert!(matches!(err, DispatchError::BadInputs(_)));
    }

    #[test]
    fn fixed_via_with_scratch_reuses_input_buffers() {
        let one = 1u32 << 16;
        let half = 1u32 << 15;
        let mut scratch = RegionCoarsenGpuScratch::default();
        let mut out = Vec::new();

        coarsen_region_state_fixed_via_with_scratch_into(
            &MoriDispatcher,
            &[half, half, half, half],
            &[10 * one, 20 * one],
            2,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        let input_ptrs: Vec<*const u8> = scratch.inputs.iter().map(Vec::as_ptr).collect();
        coarsen_region_state_fixed_via_with_scratch_into(
            &MoriDispatcher,
            &[half, half, half, half],
            &[12 * one, 18 * one],
            2,
            &mut scratch,
            &mut out,
        )
        .unwrap();

        for (before, after) in input_ptrs
            .iter()
            .zip(scratch.inputs.iter().map(Vec::as_ptr))
        {
            assert_eq!(*before, after);
        }
    }
}
