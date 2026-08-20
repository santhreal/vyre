//! Pearl's do-calculus  -  graph surgery primitives.
//!
//! Pearl's three rules of do-calculus reduce a do-query `P(Y | do(X))`
//! to an observable-query `P(Y | X)` when the causal graph admits.
//! The Shpitser ID algorithm (2008) automates the rule application;
//! Correa-Bareinboim (2020) extends to multi-treatment identifiability.
//!
//! At the GPU primitive level, do-calculus reduces to **graph
//! surgery**  -  three primitive transformations on the adjacency matrix:
//!
//! 1. **Edge deletion**  -  `do(X = x)` removes incoming edges to X
//!    (parents no longer cause X; X is set externally).
//! 2. **Edge reversal**  -  needed when applying Rule 3 (action /
//!    observation exchange).
//! 3. **Subgraph extraction**  -  restrict to a node subset for backdoor
//!    / frontdoor adjustment.
//!
//! This file ships the **incoming-edge-deletion** primitive  -  the
//! most-used graph surgery, the heart of `do(X = x)`.
//!
//! # Why this primitive is dual-use
//!
//! | Consumer | Use |
//! |---|---|
//! | `vyre-libs::causal` consumers | Pearl-style counterfactuals |
//! | `vyre-libs::security::what_if` consumers | "would finding fire under fix X?" counterfactual analysis |
//! | `vyre-foundation::transform` change-impact analysis | `do(rule_X)` on the rule dependency graph predicts which downstream Programs invalidate. Replaces ad-hoc cache-invalidation tracking with formal causal analysis. |

use vyre_foundation::composition::{trap_program, wrap_anonymous_region};

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Op id.
pub const OP_ID: &str = "vyre-libs::graph::do_intervention_delete_incoming";
#[path = "do_calculus_rules.rs"]
mod do_calculus_rules;
pub use do_calculus_rules::*;

#[cfg(test)]
#[path = "do_calculus_oracle.rs"]
mod do_calculus_oracle;

/// Impact mask op id.
pub(crate) const IMPACT_MASK_OP_ID: &str = "vyre-libs::graph::do_impact_mask_from_closure";

/// Emit a Program that zeros all incoming edges to nodes marked
/// "intervened" in `intervention_mask`. The result is the post-do
/// adjacency matrix.
///
/// Inputs:
/// - `adjacency`: row-major `n × n` u32 buffer (entry `[i, j]` = edge
///   weight or 0/1 for unweighted).
/// - `intervention_mask`: `n` u32 lanes, `1` if node is do-intervened.
///
/// Output:
/// - `out_adjacency`: row-major `n × n` u32 buffer.
///
/// Per-cell rule: `out[i, j] = 0` if `intervention_mask[j] == 1`
/// (column j zeros out  -  incoming edges to j removed). Otherwise
/// `out[i, j] = adjacency[i, j]`.
#[must_use]
pub fn intervention_delete_incoming(
    adjacency: &str,
    intervention_mask: &str,
    out_adjacency: &str,
    n: u32,
) -> Program {
    match try_intervention_delete_incoming(adjacency, intervention_mask, out_adjacency, n) {
        Ok(program) => program,
        Err(error) => trap_program(OP_ID, Some((out_adjacency, DataType::U32)), error),
    }
}

/// Emit an incoming-edge-deletion Program with checked adjacency matrix shape.
pub fn try_intervention_delete_incoming(
    adjacency: &str,
    intervention_mask: &str,
    out_adjacency: &str,
    n: u32,
) -> Result<Program, String> {
    let cells = crate::plumbing::operand::shape::square_matrix_cells(OP_ID, n)?;
    let t = Expr::InvocationId { axis: 0 };

    // Decode (i, j) from flat invocation t = i*n + j; only j matters.
    let j_expr = Expr::rem(t.clone(), Expr::u32(n));
    let intervened = Expr::load(intervention_mask, j_expr);
    let edge = Expr::load(adjacency, t.clone());
    let value = Expr::select(Expr::eq(intervened, Expr::u32(0)), edge, Expr::u32(0));

    let body = vec![Node::if_then(
        Expr::lt(t.clone(), Expr::u32(cells)),
        vec![Node::store(out_adjacency, t, value)],
    )];

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(adjacency, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(cells),
            BufferDecl::storage(intervention_mask, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(n),
            BufferDecl::storage(out_adjacency, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(cells),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_reference::composition_witness::do_intervention_delete_incoming_witness as do_intervention_delete_incoming_cpu;

    #[test]
    fn cpu_no_intervention_preserves_adjacency() {
        let a = vec![1, 2, 3, 4];
        let mask = vec![0, 0];
        let out = do_intervention_delete_incoming_cpu(&a, &mask, 2);
        assert_eq!(out, a);
    }

    #[test]
    fn cpu_intervene_node_zero_zeros_column() {
        // 2-node graph, intervene on node 0.
        // Edge [0->0]=1, [0->1]=2, [1->0]=3, [1->1]=4
        // After do(0): incoming-to-0 zeroed → [0->0]=0, [1->0]=0 stay
        // existing: [0->1]=2, [1->1]=4
        let a = vec![1, 2, 3, 4];
        let mask = vec![1, 0];
        let out = do_intervention_delete_incoming_cpu(&a, &mask, 2);
        // column 0: out[0*2+0] = 0, out[1*2+0] = 0
        // column 1: out[0*2+1] = 2, out[1*2+1] = 4
        assert_eq!(out, vec![0, 2, 0, 4]);
    }

    #[test]
    fn cpu_intervene_all_zeros_all() {
        let a = vec![1, 2, 3, 4];
        let mask = vec![1, 1];
        let out = do_intervention_delete_incoming_cpu(&a, &mask, 2);
        assert_eq!(out, vec![0; 4]);
    }

    #[test]
    fn cpu_chain_graph_intervention_breaks_chain() {
        // Chain: 0 -> 1 -> 2.
        // Adjacency (row=from, col=to):
        //   [0,1]=1, [1,2]=1, others=0
        let a = vec![
            0, 1, 0, // row 0: edge to 1
            0, 0, 1, // row 1: edge to 2
            0, 0, 0, // row 2: no edges out
        ];
        // Intervene on node 1: "set node 1 externally" → break 0→1.
        let mask = vec![0, 1, 0];
        let out = do_intervention_delete_incoming_cpu(&a, &mask, 3);
        // column 1 zeroed: [0,1]=0
        // column 2 untouched: [1,2]=1
        assert_eq!(out[0 * 3 + 1], 0);
        assert_eq!(out[1 * 3 + 2], 1);
    }

    #[test]
    #[should_panic(expected = "complete n*n adjacency matrix")]
    fn cpu_malformed_inputs_fail_loudly() {
        let _ = do_intervention_delete_incoming_cpu(&[1], &[1], 2);
    }

    #[test]
    fn ir_program_buffer_layout() {
        let p = intervention_delete_incoming("a", "m", "out", 4);
        assert_eq!(p.workgroup_size, [256, 1, 1]);
        let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
        assert_eq!(p.buffers[0].count(), 16); // n*n
        assert_eq!(p.buffers[1].count(), 4); // n
        assert_eq!(p.buffers[2].count(), 16); // n*n
    }

    #[test]
    fn zero_n_traps() {
        let p = intervention_delete_incoming("a", "m", "o", 0);
        assert!(p.stats().trap());
    }

    #[test]
    fn checked_delete_incoming_rejects_zero_n() {
        let error = try_intervention_delete_incoming("a", "m", "out", 0)
            .expect_err("checked do-intervention builder must reject n=0");
        assert!(
            error.contains("requires n > 0"),
            "error should describe the invalid causal graph shape: {error}"
        );
    }

    #[test]
    fn checked_delete_incoming_rejects_adjacency_cell_overflow() {
        let error = try_intervention_delete_incoming("a", "m", "out", u32::MAX)
            .expect_err("checked do-intervention builder must reject n*n overflow");
        assert!(
            error.contains("do_intervention_delete_incoming shape")
                && error.contains("overflows the u32 cell count"),
            "error should name the op and the shape that overflowed: {error}"
        );
    }

    #[test]
    fn legacy_delete_incoming_does_not_panic_on_adjacency_cell_overflow() {
        let program = intervention_delete_incoming("a", "m", "out", u32::MAX);
        assert!(program.stats().trap());
    }
}

/// Emit a Program that projects a reachability closure matrix and intervention mask
/// into an n-element impact mask on device.
#[must_use]
pub(crate) fn impact_mask_from_closure(
    intervention_mask: &str,
    closure: &str,
    impact_mask: &str,
    n: u32,
) -> Program {
    match try_impact_mask_from_closure(intervention_mask, closure, impact_mask, n) {
        Ok(program) => program,
        Err(error) => trap_program(IMPACT_MASK_OP_ID, Some((impact_mask, DataType::U32)), error),
    }
}

/// Emit an impact-mask projection Program with checked input shapes.
pub(crate) fn try_impact_mask_from_closure(
    intervention_mask: &str,
    closure: &str,
    impact_mask: &str,
    n: u32,
) -> Result<Program, String> {
    if n == 0 {
        return Err(format!("Fix: {IMPACT_MASK_OP_ID} requires n > 0."));
    }
    let cells = crate::plumbing::operand::shape::square_matrix_cells(IMPACT_MASK_OP_ID, n)?;
    let j = Expr::InvocationId { axis: 0 };
    let body = vec![Node::if_then(
        Expr::lt(j.clone(), Expr::u32(n)),
        vec![
            Node::let_bind(
                "is_impacted",
                Expr::select(
                    Expr::ne(Expr::load(intervention_mask, j.clone()), Expr::u32(0)),
                    Expr::u32(1),
                    Expr::u32(0),
                ),
            ),
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(n),
                vec![
                    Node::let_bind(
                        "src_intervened",
                        Expr::ne(Expr::load(intervention_mask, Expr::var("i")), Expr::u32(0)),
                    ),
                    Node::let_bind(
                        "reach",
                        Expr::ne(
                            Expr::load(
                                closure,
                                Expr::add(Expr::mul(Expr::var("i"), Expr::u32(n)), j.clone()),
                            ),
                            Expr::u32(0),
                        ),
                    ),
                    Node::if_then(
                        Expr::and(Expr::var("src_intervened"), Expr::var("reach")),
                        vec![Node::assign("is_impacted", Expr::u32(1))],
                    ),
                ],
            ),
            Node::store(impact_mask, j, Expr::var("is_impacted")),
        ],
    )];

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(intervention_mask, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(n),
            BufferDecl::storage(closure, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(cells),
            BufferDecl::storage(impact_mask, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(n),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(IMPACT_MASK_OP_ID, body)],
    ))
}

#[cfg(test)]
mod fallible_cpu_reference_tests {
    use super::do_calculus_oracle::{
        try_do_intervention_delete_incoming_cpu_into, try_do_rule2_reverse_incoming_cpu_into,
        try_do_rule3_subgraph_cpu, try_do_rule3_subgraph_cpu_into,
    };

    fn do_intervention_delete_incoming_cpu(
        adjacency: &[u32],
        intervention_mask: &[u32],
        n: u32,
    ) -> Vec<u32> {
        vyre_reference::composition_witness::do_intervention_delete_incoming_witness(
            adjacency,
            intervention_mask,
            n,
        )
    }

    fn do_rule2_reverse_incoming_cpu(
        adjacency: &[u32],
        treatment_mask: &[u32],
        n: u32,
    ) -> Vec<u32> {
        vyre_reference::composition_witness::do_rule2_reverse_incoming_witness(
            adjacency,
            treatment_mask,
            n,
        )
    }

    #[test]
    fn try_intervention_rejects_bad_input_without_clobbering_output() {
        let mut out = vec![42, 7];

        let err = try_do_intervention_delete_incoming_cpu_into(&[1], &[1], 2, &mut out)
            .expect_err("malformed intervention adjacency must return a typed error");

        assert!(
            err.contains("adjacency.len() == n*n"),
            "Fix: intervention shape error must identify the adjacency contract, got: {err}"
        );
        assert_eq!(
            out,
            vec![42, 7],
            "failed intervention preflight must preserve caller-owned diagnostics"
        );
    }

    #[test]
    fn intervention_into_reuses_capacity_and_truncates_stale_tail() {
        let adjacency = vec![1, 2, 3, 4];
        let mask = vec![1, 0];
        let mut out = Vec::with_capacity(8);
        out.extend_from_slice(&[99, 98, 97, 96, 95, 94, 93, 92]);
        let capacity = out.capacity();

        try_do_intervention_delete_incoming_cpu_into(&adjacency, &mask, 2, &mut out)
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - valid intervention matrix should reuse caller-owned output");

        assert_eq!(out, vec![0, 2, 0, 4]);
        assert_eq!(out.capacity(), capacity);

        try_do_intervention_delete_incoming_cpu_into(&[5], &[1], 1, &mut out)
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - smaller intervention matrix should truncate stale output");

        assert_eq!(out, vec![0]);
        assert_eq!(out.capacity(), capacity);
    }

    #[test]
    fn generated_try_intervention_matches_legacy_oracle() {
        for n in 1usize..=6 {
            let adjacency: Vec<u32> = (0..(n * n))
                .map(|idx| u32::from(((idx * 11 + n) % 5) == 0))
                .collect();
            let mask: Vec<u32> = (0..n)
                .map(|idx| u32::from(((idx * 3 + n) % 2) == 0))
                .collect();
            let legacy = do_intervention_delete_incoming_cpu(&adjacency, &mask, n as u32);
            let mut out = vec![u32::MAX];

            try_do_intervention_delete_incoming_cpu_into(&adjacency, &mask, n as u32, &mut out)
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - generated valid intervention matrices must pass fallible oracle");

            assert_eq!(
                out, legacy,
                "fallible intervention oracle diverged at n={n}"
            );
        }
    }

    #[test]
    fn try_rule2_rejects_bad_input_without_clobbering_output() {
        let mut out = vec![7, 11, 13];

        let err = try_do_rule2_reverse_incoming_cpu_into(&[1], &[1], 2, &mut out)
            .expect_err("malformed rule2 adjacency must return a typed error");

        assert!(
            err.contains("adjacency.len() == n*n"),
            "Fix: rule2 shape error must identify the adjacency contract, got: {err}"
        );
        assert_eq!(
            out,
            vec![7, 11, 13],
            "failed rule2 preflight must preserve caller-owned diagnostics"
        );
    }

    #[test]
    fn rule2_into_reuses_capacity_and_truncates_stale_tail() {
        let adjacency = vec![0, 1, 0, 0];
        let mask = vec![0, 1];
        let mut out = Vec::with_capacity(8);
        out.extend_from_slice(&[99, 98, 97, 96, 95, 94, 93, 92]);
        let capacity = out.capacity();

        try_do_rule2_reverse_incoming_cpu_into(&adjacency, &mask, 2, &mut out)
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - valid rule2 matrix should reuse caller-owned output");

        assert_eq!(out, vec![0, 0, 1, 0]);
        assert_eq!(out.capacity(), capacity);

        try_do_rule2_reverse_incoming_cpu_into(&[7], &[1], 1, &mut out)
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - smaller rule2 matrix should truncate stale output");

        assert_eq!(out, vec![7]);
        assert_eq!(out.capacity(), capacity);
    }

    #[test]
    fn generated_try_rule2_matches_legacy_oracle() {
        for n in 1usize..=6 {
            let mut adjacency = vec![0u32; n * n];
            for row in 0..n {
                for col in 0..n {
                    adjacency[row * n + col] = u32::from(((row * 3 + col * 5 + n) % 4) == 0);
                }
            }
            let treatment_mask: Vec<u32> = (0..n)
                .map(|idx| u32::from(((idx * 7 + n) % 3) == 0))
                .collect();
            let legacy = do_rule2_reverse_incoming_cpu(&adjacency, &treatment_mask, n as u32);
            let mut out = vec![u32::MAX];

            try_do_rule2_reverse_incoming_cpu_into(&adjacency, &treatment_mask, n as u32, &mut out)
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - generated valid rule2 matrices must pass fallible oracle");

            assert_eq!(out, legacy, "fallible rule2 oracle diverged at n={n}");
        }
    }

    #[test]
    fn try_rule3_returns_tuple_and_preserves_outputs_on_error() {
        let mut reduced = vec![0xA5, 0x5A];
        let mut kept = vec![3, 1];

        let err = try_do_rule3_subgraph_cpu_into(&[1], &[1, 0], 2, &mut reduced, &mut kept)
            .expect_err("malformed rule3 adjacency must return a typed error");

        assert!(
            err.contains("adjacency.len() == n*n"),
            "Fix: rule3 shape error must identify the adjacency contract, got: {err}"
        );
        assert_eq!(
            reduced,
            vec![0xA5, 0x5A],
            "failed rule3 preflight must preserve reduced adjacency diagnostics"
        );
        assert_eq!(
            kept,
            vec![3, 1],
            "failed rule3 preflight must preserve kept-index diagnostics"
        );

        let (valid_reduced, valid_kept) = try_do_rule3_subgraph_cpu(&[0, 1, 1, 0], &[1, 0], 2)
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - valid rule3 tuple oracle must succeed");
        assert_eq!(valid_reduced, vec![0]);
        assert_eq!(valid_kept, vec![0]);
    }
}
