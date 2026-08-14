//! Host closures over the semirings: matrix product plus the reachability,
//! lineage, and shortest-path fixpoints.

use vyre_foundation::pass_substrate::dataflow_fixpoint as foundation_dataflow;

#[cfg(any(test, feature = "cpu-parity"))]
use super::Semiring;

/// Multiply matrices over the selected semiring through the reference oracle.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn reference_semiring_gemm(
    a: &[u32],
    b: &[u32],
    m: u32,
    n: u32,
    k: u32,
    semiring: Semiring,
) -> Vec<u32> {
    let mut c = Vec::new();
    reference_semiring_gemm_into(a, b, m, n, k, semiring, &mut c);
    c
}

/// Multiply matrices over the selected semiring into caller-owned storage.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn reference_semiring_gemm_into(
    a: &[u32],
    b: &[u32],
    m: u32,
    n: u32,
    k: u32,
    semiring: Semiring,
    c: &mut Vec<u32>,
) {
    use vyre_libs::telemetry::observability::{bump, dataflow_fixpoint_calls};
    bump(&dataflow_fixpoint_calls);
    foundation_dataflow::semiring_gemm_cpu_into(a, b, m, n, k, semiring, c);
}

/// Compute boolean reachability closure on a Region adjacency matrix
/// via repeated `semiring_gemm` iterations under `Semiring::BoolOr`.
/// Iterates until fixpoint (max `max_iters` steps).
#[must_use]
pub fn reachability_closure(adj: &[u32], n: u32, max_iters: u32) -> Vec<u32> {
    let mut current = Vec::new();
    let mut next = Vec::new();
    reachability_closure_into(adj, n, max_iters, &mut current, &mut next);
    current
}

/// Compute boolean reachability closure into caller-owned buffers.
pub fn reachability_closure_into(
    adj: &[u32],
    n: u32,
    max_iters: u32,
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
) {
    foundation_dataflow::reachability_closure_into(adj, n, max_iters, current, next);
}

/// Compute lineage (which-clauses-used) closure under `Semiring::Lineage`.
/// Each entry of `adj` is a bitset of clause/source IDs.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn lineage_closure(adj: &[u32], n: u32, max_iters: u32) -> Vec<u32> {
    let mut current = Vec::new();
    let mut next = Vec::new();
    lineage_closure_into(adj, n, max_iters, &mut current, &mut next);
    current
}

/// Compute lineage closure into caller-owned buffers.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn lineage_closure_into(
    adj: &[u32],
    n: u32,
    max_iters: u32,
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
) {
    foundation_dataflow::lineage_closure_into(adj, n, max_iters, current, next);
}

/// Compute min-cost shortest-path distance matrix under `Semiring::MinPlus`.
/// Use `u32::MAX` for "no edge".
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn shortest_path_closure(adj: &[u32], n: u32, max_iters: u32) -> Vec<u32> {
    let mut current = Vec::new();
    let mut next = Vec::new();
    shortest_path_closure_into(adj, n, max_iters, &mut current, &mut next);
    current
}

/// Compute min-cost shortest-path closure into caller-owned buffers.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn shortest_path_closure_into(
    adj: &[u32],
    n: u32,
    max_iters: u32,
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
) {
    foundation_dataflow::shortest_path_closure_into(adj, n, max_iters, current, next);
}

#[cfg(test)]
#[allow(clippy::erasing_op, clippy::identity_op)]
mod tests {
    use super::super::Semiring;
    use super::{
        lineage_closure, reachability_closure, reachability_closure_into,
        reference_semiring_gemm_into, shortest_path_closure,
    };
    use vyre_foundation::pass_substrate::dataflow_fixpoint as foundation_dataflow;

    #[test]
    fn reachability_chain_graph() {
        // 0 → 1 → 2 → 3
        let adj = vec![0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0];
        let closure = reachability_closure(&adj, 4, 5);
        // After closure: 0 reaches {1, 2, 3}; 1 reaches {2, 3}; 2 reaches {3}.
        assert_eq!(closure[0 * 4 + 1], 1);
        assert_eq!(closure[0 * 4 + 2], 1);
        assert_eq!(closure[0 * 4 + 3], 1);
        assert_eq!(closure[1 * 4 + 3], 1);
        // No reverse edges
        assert_eq!(closure[3 * 4 + 0], 0);
    }

    #[test]
    fn reference_semiring_gemm_into_delegates_to_foundation_authority() {
        let left = vec![1, 2, 3, 4, 5, 6];
        let right = vec![7, 8, 9, 10, 11, 12];
        let mut out = Vec::with_capacity(8);
        let ptr = out.as_ptr();
        reference_semiring_gemm_into(&left, &right, 2, 2, 3, Semiring::Real, &mut out);
        let mut expected = Vec::new();
        foundation_dataflow::semiring_gemm_cpu_into(
            &left,
            &right,
            2,
            2,
            3,
            Semiring::Real,
            &mut expected,
        );
        assert_eq!(out, expected);
        assert_eq!(out.as_ptr(), ptr);
    }

    #[test]
    fn reachability_invalid_shapes_clear_buffers_without_panicking() {
        let mut current = vec![99, 100];
        let mut next = vec![101];
        reachability_closure_into(&[0, 1, 0], 2, 4, &mut current, &mut next);
        assert!(current.is_empty());
        assert!(next.is_empty());
        reachability_closure_into(&[], 0, 4, &mut current, &mut next);
        assert!(current.is_empty());
        assert!(next.is_empty());
    }

    #[test]
    fn reachability_respects_primitive_max_iters_policy() {
        let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
        assert_eq!(reachability_closure(&adj, 3, 0).len(), adj.len());
        assert_eq!(reachability_closure(&adj, 3, 0), adj);
    }

    #[test]
    fn generated_reachability_matches_foundation_authority() {
        for n in 1u32..=8 {
            let cells = (n * n) as usize;
            for seed in 0u32..64 {
                let mut state = seed ^ n.wrapping_mul(0x9E37);
                let mut adj = Vec::with_capacity(cells);
                for _ in 0..cells {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    adj.push((state >> 31) & 1);
                }
                assert_eq!(
                    reachability_closure(&adj, n, n),
                    foundation_dataflow::reachability_closure(&adj, n, n),
                    "n={n} seed={seed}"
                );
            }
        }
    }

    #[test]
    fn reachability_disjoint_components_stay_disjoint() {
        // 0 → 1, 2 → 3, no cross.
        let adj = vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0];
        let closure = reachability_closure(&adj, 4, 5);
        assert_eq!(closure[0 * 4 + 2], 0);
        assert_eq!(closure[2 * 4 + 0], 0);
    }

    #[test]
    fn lineage_closure_unions_clauses_along_paths() {
        // Edge 0→1 used clause f1 = 0b01; edge 1→2 used clause f2 = 0b10.
        // Path 0→2 uses both: 0b11.
        let f1 = 0b01;
        let f2 = 0b10;
        let adj = vec![0, f1, 0, 0, 0, f2, 0, 0, 0];
        let closure = lineage_closure(&adj, 3, 5);
        assert_eq!(closure[0 * 3 + 2], f1 | f2);
    }

    #[test]
    fn shortest_path_closure_finds_two_hop_minimum() {
        let inf = u32::MAX;
        // 0→1 cost 5, 1→2 cost 3, 0→2 cost 100 (slower direct).
        let adj = vec![inf, 5, 100, inf, inf, 3, inf, inf, inf];
        let closure = shortest_path_closure(&adj, 3, 5);
        // Best 0→2 = min(100, 5+3) = 8.
        assert_eq!(closure[0 * 3 + 2], 8);
    }

    #[test]
    fn reachability_self_loop_detected() {
        // 0 → 1, 1 → 0. Closure should mark both.
        let adj = vec![0, 1, 1, 0];
        let closure = reachability_closure(&adj, 2, 5);
        // After 1 iteration: 0 reaches 0 via 0→1→0; 1 reaches 1.
        assert_eq!(closure[0 * 2 + 0], 1);
        assert_eq!(closure[1 * 2 + 1], 1);
    }
}
