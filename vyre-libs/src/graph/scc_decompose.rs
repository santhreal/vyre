//! `scc_decompose`  -  Forward-Backward strongly-connected-component
//! decomposition over `super::program_graph::ProgramGraph`.
//!
//! For each pivot node `v`, the set of nodes simultaneously forward-
//! reachable from `v` AND backward-reachable from `v` is exactly the
//! SCC containing `v`. The primitive runs one pass given a pre-
//! computed forward-reach bitset and backward-reach bitset and
//! emits `component[v] = pivot` for every `v` in the pivot's SCC.
//!
//! Driver composition: iterate until every node carries a component
//! id. The CPU reference below shows the composition; the Program
//! ships one pass.

use vyre_foundation::composition::{trap_program, wrap_anonymous_region};

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::bitset::bitset_words;
use crate::graph::frontier_bits::{bind_bit_address, bind_word, bit_is_set, BitAccess};
#[cfg(test)]
use vyre_primitives::lane_grid;

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::graph::scc_decompose";
/// Source-lane workgroup for SCC component stamping.
pub const SCC_DECOMPOSE_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

/// Dispatch grid for one SCC decomposition pass over `node_count` lanes.
#[cfg(test)]
#[must_use]
pub const fn scc_decompose_dispatch_grid(node_count: u32) -> [u32; 3] {
    lane_grid(node_count, SCC_DECOMPOSE_WORKGROUP_SIZE[0])
}

/// Internal operation id for GPU packing of dense pivot-reachability rows.
pub(crate) const DENSE_REACHABILITY_BITSETS_OP_ID: &str =
    "vyre-libs::graph::dense_reachability_bitsets";

/// Build a program that packs one pivot row from each dense closure into bitsets with checked dimensions.
pub(crate) fn try_dense_reachability_bitsets(
    node_count: u32,
    dense_count: u32,
    pivot: u32,
    forward_closure: &str,
    backward_closure: &str,
    forward_bitset: &str,
    backward_bitset: &str,
) -> Result<Program, String> {
    if node_count == 0 {
        return Err(format!(
            "Fix: {DENSE_REACHABILITY_BITSETS_OP_ID} requires node_count > 0, got 0."
        ));
    }
    if pivot >= node_count {
        return Err(format!(
            "Fix: {DENSE_REACHABILITY_BITSETS_OP_ID} requires pivot < node_count, got pivot={pivot}, node_count={node_count}."
        ));
    }
    let expected_dense = node_count.checked_mul(node_count).ok_or_else(|| {
        format!(
            "Fix: {DENSE_REACHABILITY_BITSETS_OP_ID} node_count*node_count overflows u32 for node_count={node_count}."
        )
    })?;
    if dense_count != expected_dense {
        return Err(format!(
            "Fix: {DENSE_REACHABILITY_BITSETS_OP_ID} requires dense_count == node_count*node_count == {expected_dense}, got {dense_count}."
        ));
    }
    let pivot_row_offset = pivot.checked_mul(node_count).ok_or_else(|| {
        format!(
            "Fix: {DENSE_REACHABILITY_BITSETS_OP_ID} pivot*node_count overflows u32 for pivot={pivot}, node_count={node_count}."
        )
    })?;

    let lane = Expr::LogicalIndex { axis: 0 };
    let row_index = Expr::add(Expr::u32(pivot_row_offset), lane.clone());
    let word_index = Expr::div(lane.clone(), Expr::u32(32));
    let bit = Expr::shl(Expr::u32(1), Expr::bitand(lane.clone(), Expr::u32(31)));
    let forward_reachable = Expr::or(
        Expr::eq(lane.clone(), Expr::u32(pivot)),
        Expr::ne(Expr::load(forward_closure, row_index.clone()), Expr::u32(0)),
    );
    let backward_reachable = Expr::or(
        Expr::eq(lane.clone(), Expr::u32(pivot)),
        Expr::ne(Expr::load(backward_closure, row_index), Expr::u32(0)),
    );

    let words = bitset_words(node_count);

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(forward_closure, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(dense_count),
            BufferDecl::storage(backward_closure, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(dense_count),
            BufferDecl::storage(forward_bitset, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(words),
            BufferDecl::storage(backward_bitset, 3, BufferAccess::ReadWrite, DataType::U32)
                .with_count(words),
        ],
        SCC_DECOMPOSE_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(
            DENSE_REACHABILITY_BITSETS_OP_ID,
            vec![Node::if_then(
                Expr::lt(lane, Expr::u32(node_count)),
                vec![
                    Node::if_then(
                        forward_reachable,
                        vec![Node::let_bind(
                            "forward_prior",
                            Expr::atomic_or(forward_bitset, word_index.clone(), bit.clone()),
                        )],
                    ),
                    Node::if_then(
                        backward_reachable,
                        vec![Node::let_bind(
                            "backward_prior",
                            Expr::atomic_or(backward_bitset, word_index, bit),
                        )],
                    ),
                ],
            )],
        )],
    ))
}

/// Build a program that packs one pivot row from each dense closure into bitsets.
#[must_use]
pub(crate) fn dense_reachability_bitsets(
    node_count: u32,
    dense_count: u32,
    pivot: u32,
    forward_closure: &str,
    backward_closure: &str,
    forward_bitset: &str,
    backward_bitset: &str,
) -> Program {
    match try_dense_reachability_bitsets(
        node_count,
        dense_count,
        pivot,
        forward_closure,
        backward_closure,
        forward_bitset,
        backward_bitset,
    ) {
        Ok(program) => program,
        Err(error) => trap_program(
            DENSE_REACHABILITY_BITSETS_OP_ID,
            Some((forward_bitset, DataType::U32)),
            error,
        ),
    }
}

/// Build a Program that marks every node in the intersection of
/// `forward` ∩ `backward` with the pivot id.
///
/// AUDIT_2026-04-24 F-SCC-01: the IR consumes `component_out` as a
/// ReadWrite buffer and only *writes* to slots where both bitsets
/// are set  -  it never reads the prior value. Callers MUST pre-load
/// `component_out` with the initial component assignment before
/// dispatch (typically `vec![u32::MAX; node_count]` for "unassigned"
/// on the first pivot, or the running component vector on
/// subsequent passes). The `cpu_ref` below models this contract by
/// taking `component_in` and cloning it into `out`; the IR expects
/// the dispatcher to supply that seed state in-place. Not binding
/// `component_in` as a separate ReadOnly buffer keeps the primitive
/// composable in a multi-pivot loop without ping-pong copies.
#[must_use]
pub fn scc_decompose(
    node_count: u32,
    forward_bitset: &str,
    backward_bitset: &str,
    component_out: &str,
    pivot: u32,
) -> Program {
    let t = Expr::LogicalIndex { axis: 0 };
    let words = bitset_words(node_count);

    // One bit address serves both bitsets: forward and backward reach share the
    // node domain, so the pivot's SCC is exactly the nodes whose bit is set in
    // both. Addressing them separately is what lets the two sides disagree.
    let mut body = bind_bit_address(&t, "word_idx", "bit", |word| word).to_vec();
    body.extend([
        bind_word(
            forward_bitset,
            BitAccess {
                word: "word_idx",
                mask: "bit",
                value: "fwd_word",
            },
        ),
        bind_word(
            backward_bitset,
            BitAccess {
                word: "word_idx",
                mask: "bit",
                value: "bwd_word",
            },
        ),
        Node::let_bind(
            "fwd_set",
            bit_is_set(BitAccess {
                word: "word_idx",
                mask: "bit",
                value: "fwd_word",
            }),
        ),
        Node::let_bind(
            "bwd_set",
            bit_is_set(BitAccess {
                word: "word_idx",
                mask: "bit",
                value: "bwd_word",
            }),
        ),
        // PHASE7_GRAPH HIGH: previously this stored unconditionally,
        // overwriting any prior pivot's assignment for nodes that
        // happen to be in both. The component_out invariant is "first
        // pivot wins" (the caller iterates pivots in descending reach
        // order). Read first; only write if the slot is still
        // u32::MAX (unassigned). Eliminates the silent
        // pivot-ordering hazard the audit flagged.
        Node::if_then(
            Expr::and(Expr::var("fwd_set"), Expr::var("bwd_set")),
            vec![
                Node::let_bind("prior", Expr::load(component_out, t.clone())),
                Node::if_then(
                    Expr::eq(Expr::var("prior"), Expr::u32(u32::MAX)),
                    vec![Node::store(component_out, t.clone(), Expr::u32(pivot))],
                ),
            ],
        ),
    ]);

    Program::wrapped(
        vec![
            BufferDecl::storage(forward_bitset, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(words),
            BufferDecl::storage(backward_bitset, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(words),
            BufferDecl::storage(component_out, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(node_count),
        ],
        SCC_DECOMPOSE_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::if_then(
                Expr::lt(t.clone(), Expr::u32(node_count)),
                body,
            )],
        )],
    )
}

#[cfg(test)]
mod regression_tests {
    use vyre_reference::composition_witness::scc_decompose_witness as cpu_ref;

    /// PHASE7_GRAPH HIGH regression: two pivots stamping the same
    /// node  -  the first pivot's assignment must survive. Prior
    /// scc_decompose blindly overwrote, so the order of dispatch
    /// determined the outcome.
    #[test]
    fn cpu_ref_first_pivot_wins_when_two_pivots_share_a_node() {
        // Node 0 is in the forward+backward intersection of BOTH
        // pivots. First pivot (5) stamps; second pivot (9) must NOT
        // overwrite.
        let component_in = vec![u32::MAX; 4];
        let forward = vec![0b1111];
        let backward = vec![0b1111];

        let after_first = cpu_ref(4, &forward, &backward, &component_in, 5);
        assert_eq!(after_first, vec![5, 5, 5, 5]);

        let after_second = cpu_ref(4, &forward, &backward, &after_first, 9);
        assert_eq!(
            after_second,
            vec![5, 5, 5, 5],
            "second pivot must NOT overwrite first pivot's assignments"
        );
    }

    /// PHASE7_GRAPH HIGH regression: a node only assigned by the
    /// second pivot still gets stamped (no false-skip).
    #[test]
    fn cpu_ref_unassigned_node_picks_up_second_pivot() {
        // Pivot 5 only sees node 0; pivot 9 only sees node 2. Both
        // must end up stamped.
        let component_in = vec![u32::MAX; 4];

        let after_first = cpu_ref(4, &[0b0001], &[0b0001], &component_in, 5);
        assert_eq!(after_first[0], 5);
        assert_eq!(after_first[2], u32::MAX);

        let after_second = cpu_ref(4, &[0b0100], &[0b0100], &after_first, 9);
        assert_eq!(after_second[0], 5, "first pivot survives");
        assert_eq!(after_second[2], 9, "second pivot stamps unassigned node");
    }
}

const EXPECTED_SCC_DECOMPOSE_OUTPUT_BYTES: [u8; 16] = [
    0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        // AUDIT_2026-04-24 F-SCC-02: fixture differentiates forward
        // from backward so the intersection actually filters. Nodes
        // 0..=2 are forward-reachable from pivot 0; nodes 0, 2, 3
        // reach pivot 0 backward. Intersection = {0, 2}  -  node 1 is
        // forward-only, node 3 is backward-only, neither gets
        // stamped. Prior fixture fed identical bitsets and therefore
        // never exercised the AND gate.
        || scc_decompose(4, "fwd", "bwd", "comp", 0),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[0b0111]),                           // forward = {0,1,2}
                to_bytes(&[0b1101]),                           // backward = {0,2,3}
                to_bytes(&[u32::MAX, u32::MAX, u32::MAX, u32::MAX]),
            ]]
        }),
        Some(|| {
            // forward ∩ backward = 0b0101 → nodes 0 and 2 stamped.
            vec![vec![EXPECTED_SCC_DECOMPOSE_OUTPUT_BYTES.to_vec()]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_reference::composition_witness::scc_decompose_witness as cpu_ref;

    #[test]
    fn program_uses_packed_source_lane_workgroup() {
        let program = scc_decompose(513, "fwd", "bwd", "comp", 23);
        assert_eq!(program.workgroup_size(), SCC_DECOMPOSE_WORKGROUP_SIZE);
    }

    #[test]
    fn dispatch_grid_packs_node_lanes_into_blocks() {
        assert_eq!(scc_decompose_dispatch_grid(0), [1, 1, 1]);
        assert_eq!(scc_decompose_dispatch_grid(1), [1, 1, 1]);
        assert_eq!(scc_decompose_dispatch_grid(256), [1, 1, 1]);
        assert_eq!(scc_decompose_dispatch_grid(257), [2, 1, 1]);
        assert_eq!(scc_decompose_dispatch_grid(513), [3, 1, 1]);
    }

    #[test]
    fn intersection_stamps_pivot() {
        let out = cpu_ref(4, &[0b0011], &[0b0011], &[u32::MAX; 4], 0);
        assert_eq!(&out[0..2], &[0, 0]);
        assert_eq!(&out[2..4], &[u32::MAX, u32::MAX]);
    }

    #[test]
    fn disjoint_forward_backward_yields_no_change() {
        let comp_in = vec![u32::MAX; 4];
        let out = cpu_ref(4, &[0b0001], &[0b1000], &comp_in, 0);
        assert_eq!(out, comp_in);
    }

    // ------------------------------------------------------------------
    // Adversarial fixtures  -  empty/single/self-loop/disconnected/multi-word.
    // ------------------------------------------------------------------

    #[test]
    fn empty_graph_returns_empty() {
        let out = cpu_ref(0, &[], &[], &[], 0);
        assert!(out.is_empty());
    }

    #[test]
    fn single_node_not_in_intersection_stays_unassigned() {
        let out = cpu_ref(1, &[0], &[0], &[u32::MAX; 1], 0);
        assert_eq!(out, vec![u32::MAX]);
    }

    #[test]
    fn single_node_in_intersection_gets_stamped() {
        let out = cpu_ref(1, &[0b0001], &[0b0001], &[u32::MAX; 1], 7);
        assert_eq!(out, vec![7]);
    }

    #[test]
    fn self_loop_scc() {
        // Node 0 can reach itself forward and backward.
        let out = cpu_ref(1, &[0b0001], &[0b0001], &[u32::MAX; 1], 0);
        assert_eq!(out, vec![0]);
    }

    #[test]
    fn disconnected_components_only_stamp_reachable() {
        // Nodes 0 and 2 are in their own SCCs; nodes 1 and 3 are isolated.
        let forward = vec![0b0101];
        let backward = vec![0b0101];
        let comp_in = vec![u32::MAX; 4];
        let out = cpu_ref(4, &forward, &backward, &comp_in, 0);
        assert_eq!(out[0], 0);
        assert_eq!(out[1], u32::MAX);
        assert_eq!(out[2], 0);
        assert_eq!(out[3], u32::MAX);
    }

    #[test]
    fn all_nodes_pre_assigned_skips_all() {
        let comp_in = vec![5, 5, 5, 5];
        let out = cpu_ref(4, &[0b1111], &[0b1111], &comp_in, 9);
        assert_eq!(
            out,
            vec![5, 5, 5, 5],
            "pre-assigned nodes must not be overwritten"
        );
    }

    #[test]
    fn multi_word_bitset_cross_boundary() {
        // 65 nodes: node 32 (word 1 bit 0) and node 64 (word 2 bit 0) in intersection.
        let mut forward = vec![0u32; 3];
        let mut backward = vec![0u32; 3];
        forward[1] = 1; // node 32
        forward[2] = 1; // node 64
        backward[1] = 1; // node 32
        backward[2] = 1; // node 64
        let comp_in = vec![u32::MAX; 65];
        let out = cpu_ref(65, &forward, &backward, &comp_in, 42);
        assert_eq!(out[32], 42);
        assert_eq!(out[64], 42);
        assert_eq!(out[0], u32::MAX);
        assert_eq!(out[31], u32::MAX);
        assert_eq!(out[33], u32::MAX);
        assert_eq!(out[63], u32::MAX);
    }

    #[test]
    fn try_dense_reachability_bitsets_validates_dimensions() {
        let ok = try_dense_reachability_bitsets(4, 16, 0, "fc", "bc", "fb", "bb");
        assert!(ok.is_ok());

        let err_zero = try_dense_reachability_bitsets(0, 0, 0, "fc", "bc", "fb", "bb");
        assert!(err_zero.unwrap_err().contains("requires node_count > 0"));

        let err_pivot = try_dense_reachability_bitsets(4, 16, 4, "fc", "bc", "fb", "bb");
        assert!(err_pivot
            .unwrap_err()
            .contains("requires pivot < node_count"));

        let err_dense = try_dense_reachability_bitsets(4, 15, 0, "fc", "bc", "fb", "bb");
        assert!(err_dense
            .unwrap_err()
            .contains("requires dense_count == node_count*node_count == 16"));
    }

    #[test]
    fn dense_reachability_bitsets_invalid_emits_trap() {
        let trapped = dense_reachability_bitsets(0, 0, 0, "fc", "bc", "fb", "bb");
        assert!(trapped.stats().trap());
    }
}
