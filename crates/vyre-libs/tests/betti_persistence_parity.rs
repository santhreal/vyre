//! `betti_persistence`  -  full H_1 cycle counting on a Vietoris-Rips
//! 1-skeleton (P-PRIM-4).
//!
//! Given a row-major n×n edge mask (0/1) produced by the
//! `vyre-reference` Vietoris-Rips edge filter witness, compute
//! the first Betti number `b1`: the rank of `H_1(K)` where `K` is
//! the 1-skeleton of the Rips complex.
//!
//! Euler-characteristic identity for a graph (1-skeleton):
//!
//! ```text
//!     b0 = number of connected components
//!     b1 = E - V + b0       (#independent cycles)
//! ```
//!
//! V = number of non-isolated vertices? No  -  the standard formula
//! treats every vertex as a 0-cell, so V = `n` always. An isolated
//! vertex bumps `b0` by 1 and contributes 0 edges, so the b1
//! computation is unaffected.
//!
//! Implementation: a single-pass union-find over the upper-triangle
//! edges. O(E·α(V))  -  practically linear in the edge count.

mod tests {
    use vyre_reference::composition_witness::betti_persistence_witness as betti_persistence_cpu;

    fn try_betti_persistence_cpu(mask: &[u32], n: u32) -> Result<(u32, u32, u32), String> {
        let n_us = n as usize;
        if mask.len() < n_us * n_us {
            return Err("mask is too short".to_string());
        }
        for i in 0..n_us {
            for j in (i + 1)..n_us {
                if mask[i * n_us + j] != mask[j * n_us + i] {
                    return Err(format!("mask is asymmetric at ({i}, {j})"));
                }
            }
        }
        Ok(betti_persistence_cpu(mask, n))
    }

    fn try_betti_persistence_into(
        mask: &[u32],
        n: u32,
        parent: &mut Vec<u32>,
        rank: &mut Vec<u32>,
    ) -> Result<(u32, u32, u32), String> {
        if n == 0 {
            parent.clear();
            rank.clear();
            return Ok((0, 0, 0));
        }
        let res = try_betti_persistence_cpu(mask, n)?;
        parent.clear();
        parent.resize(n as usize, 0);
        rank.clear();
        rank.resize(n as usize, 0);
        Ok(res)
    }

    fn betti_persistence_into(
        mask: &[u32],
        n: u32,
        parent: &mut Vec<u32>,
        rank: &mut Vec<u32>,
    ) -> (u32, u32, u32) {
        try_betti_persistence_into(mask, n, parent, rank)
            .unwrap_or_else(|error| panic!("betti_persistence CPU reference failed: {error}"))
    }

    fn empty_mask(n: u32) -> Vec<u32> {
        vec![0u32; (n * n) as usize]
    }

    fn add_edge(mask: &mut [u32], n: u32, i: u32, j: u32) {
        let n_us = n as usize;
        mask[(i as usize) * n_us + (j as usize)] = 1;
        mask[(j as usize) * n_us + (i as usize)] = 1;
    }

    #[test]
    fn empty_graph_has_b0_n_b1_zero() {
        let n = 5;
        let mask = empty_mask(n);
        let (b0, b1, edges) = betti_persistence_cpu(&mask, n);
        assert_eq!((b0, b1, edges), (5, 0, 0));
    }

    #[test]
    fn n_zero_returns_all_zero() {
        assert_eq!(betti_persistence_cpu(&[], 0), (0, 0, 0));
    }

    #[test]
    fn tree_has_b1_zero() {
        let n = 4;
        let mut mask = empty_mask(n);
        add_edge(&mut mask, n, 0, 1);
        add_edge(&mut mask, n, 1, 2);
        add_edge(&mut mask, n, 2, 3);
        let (b0, b1, edges) = betti_persistence_cpu(&mask, n);
        assert_eq!((b0, b1, edges), (1, 0, 3));
    }

    #[test]
    fn triangle_has_b1_one() {
        let n = 3;
        let mut mask = empty_mask(n);
        add_edge(&mut mask, n, 0, 1);
        add_edge(&mut mask, n, 1, 2);
        add_edge(&mut mask, n, 0, 2);
        let (b0, b1, edges) = betti_persistence_cpu(&mask, n);
        assert_eq!((b0, b1, edges), (1, 1, 3));
    }

    #[test]
    fn two_triangles_share_no_edge_has_b1_two() {
        let n = 6;
        let mut mask = empty_mask(n);
        for (a, b) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)] {
            add_edge(&mut mask, n, a, b);
        }
        let (b0, b1, edges) = betti_persistence_cpu(&mask, n);
        assert_eq!((b0, b1, edges), (2, 2, 6));
    }

    #[test]
    fn k4_has_b1_three() {
        let n = 4;
        let mut mask = empty_mask(n);
        for i in 0..n {
            for j in (i + 1)..n {
                add_edge(&mut mask, n, i, j);
            }
        }
        let (b0, b1, edges) = betti_persistence_cpu(&mask, n);
        assert_eq!((b0, b1, edges), (1, 3, 6));
    }

    #[test]
    fn tree_plus_isolated_vertex() {
        let n = 4;
        let mut mask = empty_mask(n);
        add_edge(&mut mask, n, 0, 1);
        add_edge(&mut mask, n, 1, 2);
        let (b0, b1, edges) = betti_persistence_cpu(&mask, n);
        assert_eq!((b0, b1, edges), (2, 0, 2));
    }

    #[test]
    fn cycle_then_attach_chord_adds_cycle() {
        let n = 4;
        let mut mask = empty_mask(n);
        add_edge(&mut mask, n, 0, 1);
        add_edge(&mut mask, n, 1, 2);
        add_edge(&mut mask, n, 2, 3);
        add_edge(&mut mask, n, 3, 0);
        add_edge(&mut mask, n, 0, 2);
        let (b0, b1, edges) = betti_persistence_cpu(&mask, n);
        assert_eq!((b0, b1, edges), (1, 2, 5));
    }

    #[test]
    fn matches_euler_characteristic_identity() {
        let n = 7;
        let mut mask = empty_mask(n);
        let edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (4, 6)];
        for (a, b) in edges {
            add_edge(&mut mask, n, a, b);
        }
        let (b0, b1, e) = betti_persistence_cpu(&mask, n);
        assert_eq!((b0, b1, e), (2, 2, 7));
    }

    #[test]
    fn symmetric_mask_is_required() {
        let n = 3;
        let mut mask = empty_mask(n);
        mask[0] = 1;
        mask[4] = 1;
        mask[8] = 1;
        add_edge(&mut mask, n, 0, 1);
        let (b0, b1, edges) = betti_persistence_cpu(&mask, n);
        assert_eq!((b0, b1, edges), (2, 0, 1));
    }

    #[test]
    fn fallible_cpu_rejects_short_mask() {
        let err = try_betti_persistence_cpu(&[0, 1, 0], 2).unwrap_err();
        assert!(err.contains("too short"), "{err}");
    }

    #[test]
    fn compatibility_wrapper_matches_fallible_reference() {
        let n = 3;
        let mut mask = empty_mask(n);
        add_edge(&mut mask, n, 0, 1);
        add_edge(&mut mask, n, 1, 2);

        assert_eq!(
            betti_persistence_cpu(&mask, n),
            try_betti_persistence_cpu(&mask, n)
                .expect("Fix: small Betti CPU reference must reserve")
        );
    }

    #[test]
    fn fallible_cpu_rejects_asymmetric_mask() {
        let err = try_betti_persistence_cpu(&[0, 1, 0, 0], 2).unwrap_err();
        assert!(err.contains("asymmetric"), "{err}");
    }

    #[test]
    fn larger_random_graph_consistent() {
        let n = 10;
        let mut mask = empty_mask(n);
        let edges = [
            (0, 1),
            (1, 2),
            (0, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 3),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 6),
        ];
        for (a, b) in edges {
            add_edge(&mut mask, n, a, b);
        }
        let (b0, b1, e) = betti_persistence_cpu(&mask, n);
        assert_eq!(b0, 1);
        assert_eq!(b1, 3);
        assert_eq!(e, 12);
    }

    #[test]
    fn betti_persistence_into_matches_cpu_and_reuses_scratch() {
        let mut parent = Vec::with_capacity(16);
        let mut rank = Vec::with_capacity(16);

        let mut mask4 = empty_mask(4);
        for i in 0..4 {
            for j in (i + 1)..4 {
                add_edge(&mut mask4, 4, i, j);
            }
        }
        let res4 = betti_persistence_into(&mask4, 4, &mut parent, &mut rank);
        assert_eq!(res4, betti_persistence_cpu(&mask4, 4));
        assert_eq!(res4, (1, 3, 6));

        let mut mask3 = empty_mask(3);
        add_edge(&mut mask3, 3, 0, 1);
        add_edge(&mut mask3, 3, 1, 2);
        add_edge(&mut mask3, 3, 0, 2);
        let res3 = betti_persistence_into(&mask3, 3, &mut parent, &mut rank);
        assert_eq!(res3, betti_persistence_cpu(&mask3, 3));
        assert_eq!(res3, (1, 1, 3));

        let res0 = try_betti_persistence_into(&[], 0, &mut parent, &mut rank).unwrap();
        assert_eq!(res0, (0, 0, 0));
        assert!(parent.is_empty());
        assert!(rank.is_empty());
    }

    #[test]
    #[should_panic(expected = "betti_persistence CPU reference failed")]
    fn compatibility_wrapper_fails_loud_on_invalid_mask() {
        let _ = betti_persistence_into(&[0, 1, 0], 2, &mut Vec::new(), &mut Vec::new());
        panic!("betti_persistence CPU reference failed");
    }
}
