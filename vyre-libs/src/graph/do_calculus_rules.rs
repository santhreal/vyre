//! Rules 2 and 3 of Pearl's do-calculus: edge reversal and subgraph extraction.

use vyre_foundation::composition::{trap_program, wrap_anonymous_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Rule 2 op id.
pub const RULE2_OP_ID: &str = "vyre-libs::graph::do_rule2_reverse_incoming";
/// Rule 3 op id.
pub const RULE3_OP_ID: &str = "vyre-libs::graph::do_rule3_subgraph";

/// Rule 2 (do-calculus)  -  edge reversal on incoming edges of treatment
/// nodes. Reverses every edge `i → j` where `treatment_mask[j] != 0`
/// to `j → i`. Pre-existing reverse edges are merged via OR.
///
/// Returns the reversed adjacency matrix.
#[must_use]
pub fn rule2_reverse_incoming(
    adjacency: &str,
    treatment_mask: &str,
    out_adjacency: &str,
    n: u32,
) -> Program {
    match try_rule2_reverse_incoming(adjacency, treatment_mask, out_adjacency, n) {
        Ok(program) => program,
        Err(error) => trap_program(RULE2_OP_ID, Some((out_adjacency, DataType::U32)), error),
    }
}

/// Emit a Rule 2 incoming-edge-reversal Program with checked adjacency matrix
/// shape.
pub fn try_rule2_reverse_incoming(
    adjacency: &str,
    treatment_mask: &str,
    out_adjacency: &str,
    n: u32,
) -> Result<Program, String> {
    let cells = crate::plumbing::operand::shape::square_matrix_cells(RULE2_OP_ID, n)?;
    let t = Expr::InvocationId { axis: 0 };
    let row = Expr::div(t.clone(), Expr::u32(n));
    let col = Expr::rem(t.clone(), Expr::u32(n));
    let not_self = Expr::ne(row.clone(), col.clone());
    let original = Expr::load(adjacency, t.clone());
    let col_treated = Expr::ne(Expr::load(treatment_mask, col.clone()), Expr::u32(0));
    let row_treated = Expr::ne(Expr::load(treatment_mask, row.clone()), Expr::u32(0));
    let reverse_idx = Expr::add(Expr::mul(col, Expr::u32(n)), row);
    let kept_original = Expr::select(
        Expr::and(col_treated, not_self.clone()),
        Expr::u32(0),
        original,
    );
    let reversed_in = Expr::select(
        Expr::and(row_treated, not_self),
        Expr::load(adjacency, reverse_idx),
        Expr::u32(0),
    );
    let value = Expr::bitor(kept_original, reversed_in);

    let body = vec![Node::if_then(
        Expr::lt(t.clone(), Expr::u32(cells)),
        vec![Node::store(out_adjacency, t, value)],
    )];

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(adjacency, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(cells),
            BufferDecl::storage(treatment_mask, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(n),
            BufferDecl::storage(out_adjacency, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(cells),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(RULE2_OP_ID, body)],
    ))
}

/// Emit a Program for do-calculus **Rule 3 subgraph extraction**: the GPU/IR
/// counterpart of the `vyre-reference` do-calculus rule 3 witness. Restricts the `n × n` adjacency
/// matrix to the nodes whose `keep_mask` bit is set, laying the result out as a
/// dense `k × k` block (`k = popcount(keep_mask)`), and emits the
/// kept-index → original-index map plus the scalar `k`.
///
/// Inputs:
/// - `adjacency`: row-major `n × n` u32 buffer.
/// - `keep_mask`: `n` u32 lanes, non-zero if the node is retained.
///
/// Outputs:
/// - `reduced`: `n × n` u32 buffer; the first `k × k` cells (row-major, **stride
///   `k`**) hold the extracted subgraph, the remainder is left untouched.
/// - `kept`: `n` u32 buffer; the first `k` cells hold the retained original
///   indices in ascending order.
/// - `kept_len`: single u32 = `k`.
///
/// Unlike the two per-cell-map do-calculus surgeries (intervention / rule 2),
/// Rule 3 has a **data-dependent output size** (`k × k`, stride `k ≠ n`) and so
/// requires a compaction (prefix scan of the kept indices) followed by a gather.
/// The compaction/gather is done by a **single serialized lane** (`InvocationId
/// == 0`), which makes the kept order deterministic (ascending original index,
/// byte-identical to the CPU oracle) rather than the nondeterministic
/// atomic-append order a parallel compaction would produce.
#[must_use]
pub fn rule3_subgraph(
    adjacency: &str,
    keep_mask: &str,
    reduced: &str,
    kept: &str,
    kept_len: &str,
    n: u32,
) -> Program {
    match try_rule3_subgraph(adjacency, keep_mask, reduced, kept, kept_len, n) {
        Ok(program) => program,
        Err(error) => trap_program(RULE3_OP_ID, Some((reduced, DataType::U32)), error),
    }
}

/// Emit a Rule-3 subgraph-extraction Program with checked adjacency shape.
pub fn try_rule3_subgraph(
    adjacency: &str,
    keep_mask: &str,
    reduced: &str,
    kept: &str,
    kept_len: &str,
    n: u32,
) -> Result<Program, String> {
    let cells = crate::plumbing::operand::shape::square_matrix_cells(RULE3_OP_ID, n)?;

    let lane0 = Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0));

    // Pass 1, compaction: walk nodes in ascending order, appending each kept
    // original index to `kept[k]` and counting `k`. Deterministic order.
    let scan = vec![
        Node::let_bind("r3_k", Expr::u32(0)),
        Node::loop_for(
            "r3_i",
            Expr::u32(0),
            Expr::u32(n),
            vec![Node::if_then(
                Expr::ne(Expr::load(keep_mask, Expr::var("r3_i")), Expr::u32(0)),
                vec![
                    Node::store(kept, Expr::var("r3_k"), Expr::var("r3_i")),
                    Node::assign("r3_k", Expr::add(Expr::var("r3_k"), Expr::u32(1))),
                ],
            )],
        ),
        Node::store(kept_len, Expr::u32(0), Expr::var("r3_k")),
    ];

    // Pass 2, gather: for each (new_i, new_j) in the k × k block, copy
    // adjacency[kept[new_i] * n + kept[new_j]] into reduced[new_i * k + new_j].
    // Constant `0..n` loop bounds guarded by `< k` (portable, no dynamic loop
    // trip count); the write stride uses the runtime `r3_k`.
    let gather = vec![Node::loop_for(
        "r3_ni",
        Expr::u32(0),
        Expr::u32(n),
        vec![Node::if_then(
            Expr::lt(Expr::var("r3_ni"), Expr::var("r3_k")),
            vec![
                Node::let_bind("r3_old_i", Expr::load(kept, Expr::var("r3_ni"))),
                Node::loop_for(
                    "r3_nj",
                    Expr::u32(0),
                    Expr::u32(n),
                    vec![Node::if_then(
                        Expr::lt(Expr::var("r3_nj"), Expr::var("r3_k")),
                        vec![
                            Node::let_bind("r3_old_j", Expr::load(kept, Expr::var("r3_nj"))),
                            Node::store(
                                reduced,
                                Expr::add(
                                    Expr::mul(Expr::var("r3_ni"), Expr::var("r3_k")),
                                    Expr::var("r3_nj"),
                                ),
                                Expr::load(
                                    adjacency,
                                    Expr::add(
                                        Expr::mul(Expr::var("r3_old_i"), Expr::u32(n)),
                                        Expr::var("r3_old_j"),
                                    ),
                                ),
                            ),
                        ],
                    )],
                ),
            ],
        )],
    )];

    let mut serial_body = scan;
    serial_body.extend(gather);
    let body = vec![Node::if_then(lane0, serial_body)];

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(adjacency, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(cells),
            BufferDecl::storage(keep_mask, 1, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::storage(reduced, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(cells),
            BufferDecl::storage(kept, 3, BufferAccess::ReadWrite, DataType::U32).with_count(n),
            BufferDecl::storage(kept_len, 4, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(RULE3_OP_ID, body)],
    ))
}

#[cfg(test)]
mod rule2_tests {
    use super::*;
    use vyre_reference::composition_witness::do_rule2_reverse_incoming_witness as do_rule2_reverse_incoming_cpu;

    #[test]
    fn no_treatment_preserves_adjacency() {
        let a = vec![0, 1, 0, 0];
        let mask = vec![0u32, 0];
        let out = do_rule2_reverse_incoming_cpu(&a, &mask, 2);
        assert_eq!(out, a);
    }

    #[test]
    fn single_treatment_reverses_incoming() {
        // 2 nodes; edge 0→1; treat node 1 → reverse to 1→0.
        let a = vec![
            0, 1, // row 0
            0, 0, // row 1
        ];
        let mask = vec![0u32, 1];
        let out = do_rule2_reverse_incoming_cpu(&a, &mask, 2);
        assert_eq!(out, vec![0, 0, 1, 0]);
    }

    #[test]
    fn reversal_or_merges_with_existing_reverse_edge() {
        // Bidirectional 0↔1 (both edges exist).
        // Treat node 1 → 0→1 reversed to 1→0; existing 1→0 stays.
        let a = vec![0, 1, 1, 0];
        let mask = vec![0u32, 1];
        let out = do_rule2_reverse_incoming_cpu(&a, &mask, 2);
        assert_eq!(out, vec![0, 0, 1, 0]);
    }

    #[test]
    fn self_edges_untouched() {
        let a = vec![1, 0, 0, 1];
        let mask = vec![1u32, 1];
        let out = do_rule2_reverse_incoming_cpu(&a, &mask, 2);
        // Self-edges are skipped; still 1 on the diagonal.
        assert_eq!(out, vec![1, 0, 0, 1]);
    }

    #[test]
    fn reversal_is_involution_under_double_treatment() {
        // Reversing twice on the same treatment set yields the
        // original adjacency (when no overlap with reverse edges).
        let a = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
        let mask = vec![1u32, 1, 1];
        let once = do_rule2_reverse_incoming_cpu(&a, &mask, 3);
        let twice = do_rule2_reverse_incoming_cpu(&once, &mask, 3);
        assert_eq!(twice, a);
    }

    #[test]
    fn bidirectional_fully_treated_preserves_both_edges_without_order_loss() {
        let a = vec![0, 1, 1, 0];
        let mask = vec![1u32, 1];
        let out = do_rule2_reverse_incoming_cpu(&a, &mask, 2);
        assert_eq!(out, a);
    }

    #[test]
    fn ir_program_buffer_layout() {
        let p = rule2_reverse_incoming("a", "m", "out", 4);
        assert_eq!(p.workgroup_size, [256, 1, 1]);
        let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
        assert_eq!(names, vec!["a", "m", "out"]);
        assert_eq!(p.buffers[0].count(), 16);
        assert_eq!(p.buffers[1].count(), 4);
        assert_eq!(p.buffers[2].count(), 16);
    }

    #[test]
    fn checked_rule2_builder_rejects_adjacency_cell_overflow() {
        let error = try_rule2_reverse_incoming("a", "m", "out", u32::MAX)
            .expect_err("checked Rule 2 builder must reject n*n overflow");
        assert!(
            error.contains("do_rule2_reverse_incoming shape")
                && error.contains("overflows the u32 cell count"),
            "error should name the op and the shape that overflowed: {error}"
        );
    }

    #[test]
    fn legacy_rule2_builder_does_not_panic_on_adjacency_cell_overflow() {
        let program = rule2_reverse_incoming("a", "m", "out", u32::MAX);
        assert!(program.stats().trap());
    }
}

#[cfg(test)]
mod rule3_tests {
    use vyre_reference::composition_witness::{
        do_rule3_subgraph_witness as do_rule3_subgraph_cpu,
        do_rule3_subgraph_witness_into as do_rule3_subgraph_cpu_into,
    };
    fn try_do_rule3_subgraph_cpu_into(
        adjacency: &[u32],
        keep_mask: &[u32],
        n: u32,
        reduced: &mut Vec<u32>,
        kept: &mut Vec<u32>,
    ) -> Result<(), String> {
        let n_usize = n as usize;
        if adjacency.len() != n_usize * n_usize {
            return Err(format!(
                "Fix: rule3 requires adjacency.len() == n*n, got {} vs {}.",
                adjacency.len(),
                n_usize * n_usize
            ));
        }
        if keep_mask.len() != n_usize {
            return Err(format!(
                "Fix: rule3 requires keep_mask.len() == n, got {} vs {}.",
                keep_mask.len(),
                n_usize
            ));
        }
        do_rule3_subgraph_cpu_into(adjacency, keep_mask, n, reduced, kept);
        Ok(())
    }

    #[test]
    fn keep_all_returns_original() {
        let a = vec![0, 1, 1, 0];
        let mask = vec![1u32, 1];
        let (out, kept) = do_rule3_subgraph_cpu(&a, &mask, 2);
        assert_eq!(out, a);
        assert_eq!(kept, vec![0, 1]);
    }

    #[test]
    fn subgraph_into_reuses_buffers() {
        let a = vec![0, 1, 1, 0];
        let mask = vec![1u32, 1];
        let mut out = Vec::with_capacity(8);
        let mut kept = Vec::with_capacity(4);
        let out_capacity = out.capacity();
        let kept_capacity = kept.capacity();
        out.extend_from_slice(&[99, 98, 97, 96, 95, 94, 93, 92]);
        kept.extend_from_slice(&[9, 8, 7, 6]);
        do_rule3_subgraph_cpu_into(&a, &mask, 2, &mut out, &mut kept);
        assert_eq!(out.capacity(), out_capacity);
        assert_eq!(kept.capacity(), kept_capacity);
        assert_eq!(out, a);
        assert_eq!(kept, vec![0, 1]);

        do_rule3_subgraph_cpu_into(&a, &[1u32, 0], 2, &mut out, &mut kept);
        assert_eq!(out.capacity(), out_capacity);
        assert_eq!(kept.capacity(), kept_capacity);
        assert_eq!(out, vec![0]);
        assert_eq!(kept, vec![0]);
    }

    #[test]
    fn generated_try_rule3_subgraph_matches_kept_shape_contracts() {
        for n in 1u32..=64 {
            let adjacency: Vec<u32> = (0..n)
                .flat_map(|row| {
                    (0..n).map(move |col| {
                        if row == col {
                            0
                        } else {
                            ((row + 1) * 17 + (col + 1) * 31) & 1
                        }
                    })
                })
                .collect();
            for seed in 0u32..64 {
                let keep_mask: Vec<u32> = (0..n)
                    .map(|node| ((node.wrapping_mul(5) + seed) % 3 == 0) as u32)
                    .collect();
                let mut reduced = vec![0xCAFE_BABEu32; 3];
                let mut kept = vec![0xDEAD_BEEFu32; 2];
                try_do_rule3_subgraph_cpu_into(&adjacency, &keep_mask, n, &mut reduced, &mut kept)
                    .unwrap();
                let expected_kept: Vec<u32> = keep_mask
                    .iter()
                    .enumerate()
                    .filter_map(|(index, &keep)| (keep != 0).then_some(index as u32))
                    .collect();
                assert_eq!(kept, expected_kept);
                assert_eq!(reduced.len(), kept.len() * kept.len());
                for (new_i, &old_i) in kept.iter().enumerate() {
                    for (new_j, &old_j) in kept.iter().enumerate() {
                        assert_eq!(
                            reduced[new_i * kept.len() + new_j],
                            adjacency[(old_i as usize) * n as usize + old_j as usize]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn keep_none_returns_empty() {
        let a = vec![0, 1, 1, 0];
        let mask = vec![0u32, 0];
        let (out, kept) = do_rule3_subgraph_cpu(&a, &mask, 2);
        assert!(out.is_empty());
        assert!(kept.is_empty());
    }

    #[test]
    fn keep_one_extracts_self_loop_only() {
        let a = vec![1, 1, 1, 1];
        let mask = vec![1u32, 0];
        let (out, kept) = do_rule3_subgraph_cpu(&a, &mask, 2);
        assert_eq!(out, vec![1]);
        assert_eq!(kept, vec![0]);
    }

    #[test]
    fn keep_two_of_three_drops_middle() {
        // 3-node chain 0→1→2. Keep {0, 2} → 1×... wait k=2.
        // After dropping node 1, 0 and 2 share no edge directly.
        let a = vec![
            0, 1, 0, // row 0
            0, 0, 1, // row 1
            0, 0, 0, // row 2
        ];
        let mask = vec![1u32, 0, 1];
        let (out, kept) = do_rule3_subgraph_cpu(&a, &mask, 3);
        assert_eq!(out, vec![0, 0, 0, 0]);
        assert_eq!(kept, vec![0, 2]);
    }

    #[test]
    fn keep_preserves_edges_between_kept_nodes() {
        // 4-node graph. Keep {1, 3}.
        // Edge 1→3 exists; should appear in 2×2 reduced.
        let n = 4;
        let mut a = vec![0u32; (n * n) as usize];
        a[(1 * n + 3) as usize] = 7;
        a[(3 * n + 1) as usize] = 5;
        let mask = vec![0u32, 1, 0, 1];
        let (out, kept) = do_rule3_subgraph_cpu(&a, &mask, n);
        // Reduced indices: 1 → new 0, 3 → new 1. So 1→3 lands at out[0,1] = 7.
        assert_eq!(out, vec![0, 7, 5, 0]);
        assert_eq!(kept, vec![1, 3]);
    }
}
