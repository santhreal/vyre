//! Persistent homology loop scale signatures and Vietoris-Rips 1-skeletons.
//!
//! Exposes host dispatcher callers for the topological filtration kernels
//! implemented in [`crate::topology::vietoris_rips`].

use crate::dispatch_buffers::{
    ceil_div_u32, checked_square_cells, decode_u32_output_exact, u32_slice_to_le_bytes,
};

use crate::topology::vietoris_rips::vietoris_rips_edge_filter;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

/// Compute the Vietoris-Rips 1-skeleton through the dispatcher using
/// fixed-point 16.16 distances. This is the production path for callers
/// that only need the mask of active edges at scale `epsilon`.
///
/// `dist_matrix_fixed` is `n × n` in 16.16 fixed-point (u32).
/// `epsilon_fixed` is the distance threshold in 16.16 fixed-point.
/// Returns the `n × n` u32 edge mask (1 = edge present, 0 = absent).
///
/// # Errors
///
/// Returns `DispatchError::BadInputs` if `dist_matrix_fixed.len() != n*n` or `n == 0`.
pub fn region_loop_skeleton_fixed_via(
    dispatcher: &dyn ProgramDispatcher,
    dist_matrix_fixed: &[u32],
    epsilon_fixed: u32,
    n: u32,
) -> Result<Vec<u32>, DispatchError> {
    let cells = checked_square_cells(n, "region_loop_skeleton_fixed_via")?;
    if dist_matrix_fixed.len() != cells {
        return Err(DispatchError::BadInputs(format!(
            "distance matrix len {} != n*n ({cells})",
            dist_matrix_fixed.len()
        )));
    }
    let mut out = Vec::new();
    region_loop_skeleton_fixed_via_into(dispatcher, dist_matrix_fixed, epsilon_fixed, n, &mut out)?;
    Ok(out)
}

/// Compute the Vietoris-Rips 1-skeleton into caller-owned storage.
///
/// # Errors
///
/// Returns `DispatchError` if inputs are malformed or dispatch fails.
pub fn region_loop_skeleton_fixed_via_into(
    dispatcher: &dyn ProgramDispatcher,
    dist_matrix_fixed: &[u32],
    epsilon_fixed: u32,
    n: u32,
    out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let cells = checked_square_cells(n, "region_loop_skeleton_fixed_via_into")?;
    if dist_matrix_fixed.len() != cells {
        return Err(DispatchError::BadInputs(format!(
            "distance matrix len {} != n*n ({cells})",
            dist_matrix_fixed.len()
        )));
    }
    let program = vietoris_rips_edge_filter("dist_matrix", "epsilon", "edge_mask", n);
    let inputs = vec![
        u32_slice_to_le_bytes(dist_matrix_fixed),
        u32_slice_to_le_bytes(&[epsilon_fixed]),
        vec![0u8; cells * 4],
    ];
    let groups = ceil_div_u32(n, 256).max(1);
    let outputs = dispatcher.dispatch(&program, &inputs, Some([groups, 1, 1]))?;
    if outputs.is_empty() {
        return Err(DispatchError::BackendError(
            "Fix: Vietoris-Rips dispatch returned no output buffers".to_string(),
        ));
    }
    decode_u32_output_exact(&outputs[0], cells, "region_loop_skeleton_fixed_via", out)
}

#[cfg(test)]
mod fixed_via_tests {
    use super::*;
    use vyre_foundation::ir::Program;

    struct SkeletonDispatcher;

    impl ProgramDispatcher for SkeletonDispatcher {
        fn dispatch(
            &self,
            _program: &Program,
            inputs: &[Vec<u8>],
            grid_override: Option<[u32; 3]>,
        ) -> Result<Vec<Vec<u8>>, DispatchError> {
            assert_eq!(grid_override, Some([1, 1, 1]));
            assert_eq!(inputs.len(), 3);
            let dist = crate::dispatch_buffers::read_u32s(&inputs[0]);
            let epsilon = crate::dispatch_buffers::read_u32s(&inputs[1])[0];
            let n = integer_sqrt(dist.len());
            let mut mask = vec![0u32; dist.len()];
            for i in 0..n {
                for j in (i + 1)..n {
                    let idx = i * n + j;
                    if dist[idx] <= epsilon {
                        mask[idx] = 1;
                    }
                }
            }
            Ok(vec![u32_slice_to_le_bytes(&mask)])
        }
    }

    #[test]
    fn fixed_via_dispatches_vietoris_rips_mask() {
        let dist = vec![0, 10, 30, 10, 0, 20, 30, 20, 0];
        let mask = region_loop_skeleton_fixed_via(&SkeletonDispatcher, &dist, 20, 3).unwrap();
        assert_eq!(mask, vec![0, 1, 0, 0, 0, 1, 0, 0, 0]);
    }

    #[test]
    fn fixed_via_rejects_bad_matrix_shape() {
        let err =
            region_loop_skeleton_fixed_via(&SkeletonDispatcher, &[0, 1, 2], 1, 2).unwrap_err();
        assert!(matches!(err, DispatchError::BadInputs(_)));
    }

    fn integer_sqrt(n: usize) -> usize {
        let mut root = 0usize;
        while root * root < n {
            root += 1;
        }
        root
    }
}

#[cfg(test)]
mod tests {
    use vyre_reference::composition_witness::betti_persistence_witness as reference_betti_persistence;

    #[derive(Debug, Default)]
    struct LoopTopologyScratch {
        mask: Vec<u32>,
    }

    fn reference_region_loop_skeleton(dist_matrix: &[f64], epsilon: f64, n: u32) -> Vec<u32> {
        let mut out = Vec::new();
        reference_region_loop_skeleton_into(dist_matrix, epsilon, n, &mut out);
        out
    }

    fn reference_region_loop_skeleton_into(
        dist_matrix: &[f64],
        epsilon: f64,
        n: u32,
        out: &mut Vec<u32>,
    ) {
        let n_us = n as usize;
        assert_eq!(dist_matrix.len(), n_us * n_us);
        out.clear();
        out.resize(n_us * n_us, 0);
        for i in 0..n_us {
            for j in (i + 1)..n_us {
                if dist_matrix[i * n_us + j] <= epsilon {
                    out[i * n_us + j] = 1;
                    out[j * n_us + i] = 1;
                }
            }
        }
    }

    fn count_upper_triangle_edges(mask: &[u32], n: u32) -> u32 {
        let n_us = n as usize;
        let mut edges = 0u32;
        for i in 0..n_us {
            for j in (i + 1)..n_us {
                if mask[i * n_us + j] != 0 {
                    edges = edges.saturating_add(1);
                }
            }
        }
        edges
    }

    fn reference_region_loop_edges(dist_matrix: &[f64], epsilon: f64, n: u32) -> Vec<(u32, u32)> {
        let mask = reference_region_loop_skeleton(dist_matrix, epsilon, n);
        let n_us = n as usize;
        let mut edges = Vec::new();
        for i in 0..n_us {
            for j in (i + 1)..n_us {
                if mask[i * n_us + j] != 0 {
                    edges.push((i as u32, j as u32));
                }
            }
        }
        edges
    }

    fn reference_loop_filtration_edge_counts(
        dist_matrix: &[f64],
        epsilons: &[f64],
        n: u32,
    ) -> Vec<u32> {
        let mut scratch = LoopTopologyScratch::default();
        let mut out = Vec::with_capacity(epsilons.len());
        reference_loop_filtration_edge_counts_into(
            dist_matrix,
            epsilons,
            n,
            &mut scratch,
            &mut out,
        );
        out
    }

    fn reference_loop_filtration_edge_counts_into(
        dist_matrix: &[f64],
        epsilons: &[f64],
        n: u32,
        scratch: &mut LoopTopologyScratch,
        out: &mut Vec<u32>,
    ) {
        out.clear();
        for &eps in epsilons {
            reference_region_loop_skeleton_into(dist_matrix, eps, n, &mut scratch.mask);
            out.push(count_upper_triangle_edges(&scratch.mask, n));
        }
    }

    fn reference_loop_filtration_betti(
        dist_matrix: &[f64],
        epsilons: &[f64],
        n: u32,
    ) -> Vec<(u32, u32)> {
        let mut scratch = LoopTopologyScratch::default();
        let mut out = Vec::with_capacity(epsilons.len());
        reference_loop_filtration_betti_into(dist_matrix, epsilons, n, &mut scratch, &mut out);
        out
    }

    fn reference_loop_filtration_betti_into(
        dist_matrix: &[f64],
        epsilons: &[f64],
        n: u32,
        scratch: &mut LoopTopologyScratch,
        out: &mut Vec<(u32, u32)>,
    ) {
        out.clear();
        for &eps in epsilons {
            reference_region_loop_skeleton_into(dist_matrix, eps, n, &mut scratch.mask);
            let (b0, b1, _edges) = reference_betti_persistence(&scratch.mask, n);
            out.push((b0, b1));
        }
    }

    fn reference_h1_birth_scales(dist_matrix: &[f64], epsilons: &[f64], n: u32) -> Vec<(f64, u32)> {
        let mut scratch = LoopTopologyScratch::default();
        let mut births = Vec::new();
        reference_h1_birth_scales_into(dist_matrix, epsilons, n, &mut scratch, &mut births);
        births
    }

    fn reference_h1_birth_scales_into(
        dist_matrix: &[f64],
        epsilons: &[f64],
        n: u32,
        scratch: &mut LoopTopologyScratch,
        births: &mut Vec<(f64, u32)>,
    ) {
        let mut prev_b1 = 0u32;
        births.clear();
        for &eps in epsilons {
            reference_region_loop_skeleton_into(dist_matrix, eps, n, &mut scratch.mask);
            let (_b0, b1, _edges) = reference_betti_persistence(&scratch.mask, n);
            if b1 > prev_b1 {
                births.push((eps, b1));
            }
            prev_b1 = b1;
        }
    }

    #[test]
    fn empty_skeleton_below_threshold() {
        let dist = vec![0.0, 1.0, 1.0, 0.0];
        let mask = reference_region_loop_skeleton(&dist, 0.5, 2);
        assert!(mask.iter().all(|&v| v == 0));
    }

    #[test]
    fn full_skeleton_above_threshold() {
        let dist = vec![0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0];
        let mask = reference_region_loop_skeleton(&dist, 0.6, 3);
        let count = count_upper_triangle_edges(&mask, 3);
        assert_eq!(count, 3);
    }

    #[test]
    fn edges_extracted_in_canonical_order() {
        let dist = vec![0.0, 0.3, 0.7, 0.3, 0.0, 0.4, 0.7, 0.4, 0.0];
        let edges = reference_region_loop_edges(&dist, 0.5, 3);
        assert!(edges.contains(&(0, 1)));
        assert!(edges.contains(&(1, 2)));
        assert!(!edges.contains(&(0, 2)));
    }

    #[test]
    fn filtration_edge_counts_monotone_increasing() {
        let dist = vec![0.0, 0.1, 0.5, 0.1, 0.0, 0.2, 0.5, 0.2, 0.0];
        let epsilons = vec![0.05, 0.15, 0.25, 0.6];
        let counts = reference_loop_filtration_edge_counts(&dist, &epsilons, 3);
        for w in counts.windows(2) {
            assert!(
                w[0] <= w[1],
                "edge counts must be monotone over ε filtration"
            );
        }
        assert_eq!(counts[3], 3);
    }

    #[test]
    fn singleton_dist_yields_no_edges() {
        let dist = vec![0.0];
        let mask = reference_region_loop_skeleton(&dist, 1.0, 1);
        assert!(mask.iter().all(|&v| v == 0));
    }

    #[test]
    fn betti_filtration_below_threshold_no_cycles() {
        let dist = vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let series = reference_loop_filtration_betti(&dist, &[0.5], 3);
        assert_eq!(series, vec![(3, 0)]);
    }

    #[test]
    fn betti_filtration_triangle_has_b1_one() {
        let dist = vec![0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0];
        let series = reference_loop_filtration_betti(&dist, &[0.6], 3);
        assert_eq!(series, vec![(1, 1)]);
    }

    #[test]
    fn betti_filtration_b1_monotone_non_decreasing_on_growing_filtration() {
        let dist = vec![
            0.0, 0.1, 0.2, 0.3, 0.1, 0.0, 0.4, 0.5, 0.2, 0.4, 0.0, 0.6, 0.3, 0.5, 0.6, 0.0,
        ];
        let epsilons = vec![0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65];
        let series = reference_loop_filtration_betti(&dist, &epsilons, 4);
        for w in series.windows(2) {
            assert!(
                w[0].1 <= w[1].1,
                "b1 must be non-decreasing across a growing filtration; got {:?}",
                series
            );
        }
        assert_eq!(series.last().unwrap().1, 3);
    }

    #[test]
    fn betti_h1_birth_scales_pinpoints_first_cycle() {
        let dist = vec![0.0, 0.1, 0.2, 0.1, 0.0, 0.3, 0.2, 0.3, 0.0];
        let epsilons = vec![0.15, 0.25, 0.35];
        let births = reference_h1_birth_scales(&dist, &epsilons, 3);
        assert_eq!(births, vec![(0.35, 1)]);
    }

    #[test]
    fn filtration_into_paths_match_owned_helpers() {
        let dist = vec![0.0, 0.1, 0.2, 0.1, 0.0, 0.3, 0.2, 0.3, 0.0];
        let epsilons = vec![0.15, 0.25, 0.35];
        let mut scratch = LoopTopologyScratch::default();

        let owned_counts = reference_loop_filtration_edge_counts(&dist, &epsilons, 3);
        let mut counts = Vec::new();
        reference_loop_filtration_edge_counts_into(&dist, &epsilons, 3, &mut scratch, &mut counts);
        assert_eq!(counts, owned_counts);

        let owned_betti = reference_loop_filtration_betti(&dist, &epsilons, 3);
        let mut betti = Vec::new();
        reference_loop_filtration_betti_into(&dist, &epsilons, 3, &mut scratch, &mut betti);
        assert_eq!(betti, owned_betti);

        let owned_births = reference_h1_birth_scales(&dist, &epsilons, 3);
        let mut births = Vec::new();
        reference_h1_birth_scales_into(&dist, &epsilons, 3, &mut scratch, &mut births);
        assert_eq!(births, owned_births);
    }

    #[test]
    fn betti_filtration_matches_primitive_on_each_epsilon() {
        let dist = vec![0.0, 0.2, 0.4, 0.2, 0.0, 0.3, 0.4, 0.3, 0.0];
        let epsilons = vec![0.1, 0.25, 0.35, 0.5];
        let series = reference_loop_filtration_betti(&dist, &epsilons, 3);
        for (idx, &eps) in epsilons.iter().enumerate() {
            let mask = reference_region_loop_skeleton(&dist, eps, 3);
            let (b0_p, b1_p, _) = reference_betti_persistence(&mask, 3);
            assert_eq!(series[idx], (b0_p, b1_p));
        }
    }

    #[test]
    fn betti_adversarial_two_disjoint_triangles_has_b1_two() {
        let mut dist = vec![5.0; 36];
        for i in 0..6 {
            dist[i * 6 + i] = 0.0;
        }
        for &(i, j) in &[(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)] {
            dist[i * 6 + j] = 0.4;
            dist[j * 6 + i] = 0.4;
        }
        let series = reference_loop_filtration_betti(&dist, &[0.5], 6);
        let (b0, b1) = series[0];
        assert_eq!((b0, b1), (2, 2));
    }

    #[test]
    fn betti_filtration_empty_epsilons_returns_empty() {
        let dist = vec![0.0, 0.1, 0.1, 0.0];
        let series = reference_loop_filtration_betti(&dist, &[], 2);
        assert!(series.is_empty());
        let births = reference_h1_birth_scales(&dist, &[], 2);
        assert!(births.is_empty());
    }
}
