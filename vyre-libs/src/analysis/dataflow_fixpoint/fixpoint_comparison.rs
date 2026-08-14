//! Side-by-side reachability closure over three engine formulations: the vyre
//! semiring product, an external-style frontier walk, and a GraphBLAS-style
//! sparse product.

use super::dense_matrix::{
    checked_dense_cells, checked_dense_node_count, dense_bool_to_csr, normalize_bool_matrix,
};
use super::{FixpointEngineReport, FixpointEngineTelemetry, StaticAnalysisFixpointComparison};
use vyre_foundation::pass_substrate::dataflow_fixpoint::reachability_closure_into;

/// Compare Vyre semiring, external-style frontier, and GraphBLAS-style sparse
/// reachability closures on one static-analysis adjacency matrix.
///
/// # Errors
///
/// Returns a fix-directed string when dimensions overflow, inputs are empty, or
/// the adjacency matrix is not exactly `n*n`.
pub fn compare_static_analysis_reachability_fixpoints(
    adj: &[u32],
    n: u32,
    max_iters: u32,
) -> Result<StaticAnalysisFixpointComparison, String> {
    let n_us = checked_dense_node_count(n)?;
    let cells = checked_dense_cells(n_us)?;
    if adj.len() != cells {
        return Err(format!(
            "Fix: static-analysis fixpoint comparison expected adj.len() == n*n == {cells}, got {}.",
            adj.len()
        ));
    }
    if max_iters == 0 {
        return Err("Fix: static-analysis fixpoint comparison requires max_iters > 0.".to_string());
    }
    let normalized = normalize_bool_matrix(adj);
    let csr = dense_bool_to_csr(&normalized, n_us);
    let vyre_semiring = vyre_semiring_reachability_report(&normalized, n, max_iters)?;
    let external_frontier = external_frontier_reachability_report(&csr, n_us, max_iters)?;
    let graphblas_sparse = graphblas_sparse_reachability_report(&csr, n_us, max_iters)?;
    let exact_reachability_sets = vyre_semiring.reachability == external_frontier.reachability
        && vyre_semiring.reachability == graphblas_sparse.reachability;
    Ok(StaticAnalysisFixpointComparison {
        node_count: n,
        max_iterations: max_iters,
        vyre_semiring,
        external_frontier,
        graphblas_sparse,
        exact_reachability_sets,
    })
}

fn vyre_semiring_reachability_report(
    adj: &[u32],
    n: u32,
    max_iters: u32,
) -> Result<FixpointEngineReport, String> {
    let started = std::time::Instant::now();
    let mut reachability = Vec::new();
    let mut next = Vec::new();
    reachability_closure_into(adj, n, max_iters, &mut reachability, &mut next);
    let active_time_ns = started.elapsed().as_nanos().max(1);
    let cells = u64::try_from(adj.len())
        .map_err(|_| "Fix: adjacency length does not fit telemetry byte accounting.".to_string())?;
    let active = u64::try_from(reachability.iter().filter(|value| **value != 0).count())
        .map_err(|_| "Fix: reachability count does not fit telemetry.".to_string())?;
    let iterations = max_iters;
    Ok(FixpointEngineReport {
        telemetry: FixpointEngineTelemetry {
            engine_id: "vyre.semiring.bool_or.dense",
            iterations,
            bytes_touched: cells
                .saturating_mul(std::mem::size_of::<u32>() as u64)
                .saturating_mul(u64::from(iterations).saturating_add(2)),
            frontier_density_bps: density_bps(active, cells, iterations),
            active_time_ns,
        },
        reachability,
    })
}

fn external_frontier_reachability_report(
    csr: &[Vec<usize>],
    n_us: usize,
    max_iters: u32,
) -> Result<FixpointEngineReport, String> {
    let started = std::time::Instant::now();
    let cells = checked_dense_cells(n_us)?;
    let mut reachability = vec![0; cells];
    let mut max_layers = 0u32;
    let mut frontier_visits = 0u64;
    let mut edge_visits = 0u64;
    for source in 0..n_us {
        let mut reached = vec![false; n_us];
        let mut frontier = Vec::new();
        for &target in &csr[source] {
            if !reached[target] {
                reached[target] = true;
                reachability[source * n_us + target] = 1;
                frontier.push(target);
            }
        }
        let mut layers = 0u32;
        while !frontier.is_empty() && layers < max_iters {
            frontier_visits = frontier_visits.saturating_add(frontier.len() as u64);
            let mut next_frontier = Vec::new();
            for node in frontier {
                edge_visits = edge_visits.saturating_add(csr[node].len() as u64);
                for &target in &csr[node] {
                    if !reached[target] {
                        reached[target] = true;
                        reachability[source * n_us + target] = 1;
                        next_frontier.push(target);
                    }
                }
            }
            frontier = next_frontier;
            layers = layers.saturating_add(1);
        }
        max_layers = max_layers.max(layers);
    }
    let active_time_ns = started.elapsed().as_nanos().max(1);
    Ok(FixpointEngineReport {
        telemetry: FixpointEngineTelemetry {
            engine_id: "external.csr.frontier",
            iterations: max_layers,
            bytes_touched: edge_visits
                .saturating_add(frontier_visits)
                .saturating_mul(std::mem::size_of::<u32>() as u64),
            frontier_density_bps: density_bps(frontier_visits, cells as u64, max_layers.max(1)),
            active_time_ns,
        },
        reachability,
    })
}

fn graphblas_sparse_reachability_report(
    csr: &[Vec<usize>],
    n_us: usize,
    max_iters: u32,
) -> Result<FixpointEngineReport, String> {
    let started = std::time::Instant::now();
    let cells = checked_dense_cells(n_us)?;
    let mut reached = vec![0; cells];
    let mut frontier = vec![0; cells];
    for row in 0..n_us {
        for &target in &csr[row] {
            reached[row * n_us + target] = 1;
            frontier[row * n_us + target] = 1;
        }
    }
    let mut iterations = 0u32;
    let mut frontier_visits =
        u64::try_from(frontier.iter().filter(|value| **value != 0).count())
            .map_err(|_| "Fix: frontier count does not fit telemetry.".to_string())?;
    let mut edge_visits = 0u64;
    while iterations < max_iters {
        let mut next_frontier = vec![0; cells];
        let mut new_bits = 0u64;
        for row in 0..n_us {
            for mid in 0..n_us {
                if frontier[row * n_us + mid] == 0 {
                    continue;
                }
                edge_visits = edge_visits.saturating_add(csr[mid].len() as u64);
                for &target in &csr[mid] {
                    let slot = row * n_us + target;
                    if reached[slot] == 0 {
                        reached[slot] = 1;
                        next_frontier[slot] = 1;
                        new_bits = new_bits.saturating_add(1);
                    }
                }
            }
        }
        iterations = iterations.saturating_add(1);
        if new_bits == 0 {
            break;
        }
        frontier_visits = frontier_visits.saturating_add(new_bits);
        frontier = next_frontier;
    }
    let active_time_ns = started.elapsed().as_nanos().max(1);
    Ok(FixpointEngineReport {
        telemetry: FixpointEngineTelemetry {
            engine_id: "graphblas.sparse.bool_mxm",
            iterations,
            bytes_touched: edge_visits
                .saturating_add(frontier_visits)
                .saturating_mul(std::mem::size_of::<u32>() as u64),
            frontier_density_bps: density_bps(frontier_visits, cells as u64, iterations.max(1)),
            active_time_ns,
        },
        reachability: reached,
    })
}

fn density_bps(active: u64, slots: u64, iterations: u32) -> u32 {
    if active == 0 || slots == 0 || iterations == 0 {
        return 0;
    }
    let denom = u128::from(slots).saturating_mul(u128::from(iterations));
    let bps = u128::from(active)
        .saturating_mul(10_000)
        .checked_div(denom)
        .unwrap_or(0)
        .min(10_000);
    bps as u32
}

#[cfg(test)]
mod tests {
    use super::compare_static_analysis_reachability_fixpoints;

    #[test]
    fn static_analysis_fixpoint_comparison_matches_vyre_external_and_graphblas_closures() {
        let adj = vec![
            0, 1, 0, 0, 0, //
            0, 0, 1, 1, 0, //
            0, 0, 0, 0, 1, //
            0, 0, 0, 0, 1, //
            0, 1, 0, 0, 0, //
        ];
        let expected = vec![
            0, 1, 1, 1, 1, //
            0, 1, 1, 1, 1, //
            0, 1, 1, 1, 1, //
            0, 1, 1, 1, 1, //
            0, 1, 1, 1, 1, //
        ];

        let report = compare_static_analysis_reachability_fixpoints(&adj, 5, 5)
            .expect("Fix: valid static-analysis corpus fixture should compare");

        assert!(report.exact_reachability_sets);
        assert_eq!(report.vyre_semiring.reachability, expected);
        assert_eq!(report.external_frontier.reachability, expected);
        assert_eq!(report.graphblas_sparse.reachability, expected);
        assert_eq!(
            report.vyre_semiring.telemetry.engine_id,
            "vyre.semiring.bool_or.dense"
        );
        assert_eq!(
            report.external_frontier.telemetry.engine_id,
            "external.csr.frontier"
        );
        assert_eq!(
            report.graphblas_sparse.telemetry.engine_id,
            "graphblas.sparse.bool_mxm"
        );
        for telemetry in [
            &report.vyre_semiring.telemetry,
            &report.external_frontier.telemetry,
            &report.graphblas_sparse.telemetry,
        ] {
            assert!(telemetry.iterations > 0);
            assert!(telemetry.bytes_touched > 0);
            assert!(telemetry.frontier_density_bps <= 10_000);
            assert!(telemetry.active_time_ns > 0);
        }
    }
}
