//! Sum-product circuit (probabilistic circuit) evaluator.
//!
//! Sum-product networks (Poon-Domingos 2011, Vergari-Choi 2024) are
//! topologically-ordered weighted DAGs where every marginal is
//! computable in linear time. They sit between graphical models
//! (intractable) and neural networks (no semantics)  -  tractable
//! probability with calibrated uncertainty.
//!
//! Each node is one of:
//! - **Leaf**: a value `v[i]` (observed evidence, probability 1 if
//!   value matches, 0 otherwise; or a marginal probability).
//! - **Sum**: `out = Σ_c w_c · child_out[c]` over its child set.
//! - **Product**: `out = Π_c child_out[c]` over its child set.
//!
//! Forward evaluation is one bottom-up pass  -  exactly what
//! [`level_wave_program`](crate::graph::level_wave) was built for.
//!
//! Two entry points share the per-node body (`sum_product_pass_body`):
//! - `sum_product_evaluate`  -  a SINGLE-PASS fast path, correct only for
//!   DEPTH-1 circuits (every internal node reads leaves). A deeper circuit
//!   races across topo levels (an internal node reads another internal node's
//!   `out` before it commits).
//! - `sum_product_evaluate_leveled`  -  drives the SAME body through the
//!   depth-wave harness with a per-level barrier, so it is correct at ANY
//!   depth. Prefer it whenever a circuit has an internal node feeding another.
//!
//! # Why this primitive is dual-use
//!
//! | Consumer | Use |
//! |---|---|
//! | `vyre-libs::ml::probabilistic` | tractable Bayesian inference |
//! | `vyre-libs::security::risk_score` | calibrated uncertainty on findings |
//! | `vyre-libs::ml::density` | density estimation / anomaly detection |
//! | `vyre-driver/src/cost_model/probabilistic.rs` | **vyre's dispatch cost model** as probabilistic circuit over Program features → calibrated runtime + uncertainty (paired with conformal intervals) → feed megakernel scheduler as soft constraints |
//!
//! # Encoding
//!
//! Each node carries:
//! - `kind`  -  0 = leaf, 1 = sum, 2 = product.
//! - `child_offset`, `child_count`  -  slice into the child-list buffer.
//! - For sum nodes, an aligned weights slice into the weights buffer.
//!
//! u32 fixed-point 16.16 throughout for outputs and weights.

use vyre_foundation::composition::{trap_program, wrap_anonymous_region};

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Op id.
pub const OP_ID: &str = "vyre-libs::graph::sum_product_evaluate";

/// Op id for the depth-leveled evaluator ([`sum_product_evaluate_leveled`]).
pub const OP_ID_LEVELED: &str = "vyre-libs::graph::sum_product_evaluate_leveled";

/// Node-kind tag: leaf node (carries an evidence/marginal value).
pub const KIND_LEAF: u32 = 0;
/// Node-kind tag: sum node (weighted sum over children, mixture).
pub const KIND_SUM: u32 = 1;
/// Node-kind tag: product node (independence factor over children).
pub const KIND_PRODUCT: u32 = 2;

/// Emit one bottom-up sum-product evaluation step. Caller composes
/// this with [`crate::graph::level_wave::level_wave_program`] to drive
/// the wave from leaves up to the root.
///
/// Buffers:
/// - `kinds`: u32 per node  -  0/1/2.
/// - `child_offsets`: u32 per node  -  start index in `children`.
/// - `child_counts`: u32 per node  -  number of children.
/// - `children`: u32 list  -  child node indices (concatenated per node).
/// - `weights`: u32 list  -  sum-node child weights, indexed parallel
///   to `children` (unused for leaf/product slots).
/// - `leaf_values`: u32 per node  -  leaf evidence/marginal values
///   (read only when kind == LEAF).
/// - `out`: u32 per node  -  evaluation output (one per node).
///
/// The dispatch is `n_nodes` lanes; each lane evaluates one node.
/// Children must already be evaluated by the time their parent's lane
/// runs  -  this primitive does NOT enforce ordering on its own.
/// Callers wrap with `level_wave_program` for the wave harness.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn sum_product_evaluate(
    kinds: &str,
    child_offsets: &str,
    child_counts: &str,
    children: &str,
    weights: &str,
    leaf_values: &str,
    out: &str,
    n_nodes: u32,
    n_edges: u32,
) -> Program {
    match try_sum_product_evaluate(
        kinds,
        child_offsets,
        child_counts,
        children,
        weights,
        leaf_values,
        out,
        n_nodes,
        n_edges,
    ) {
        Ok(program) => program,
        Err(error) => trap_program(OP_ID, Some((out, DataType::U32)), error),
    }
}

/// Emit one bottom-up sum-product evaluation step with checked node shape.
///
/// `n_edges == 0` is valid for a leaf-only circuit; children and weight buffers
/// still receive one declared word because several GPU backends reject true
/// zero-sized storage bindings.
#[allow(clippy::too_many_arguments)]
pub fn try_sum_product_evaluate(
    kinds: &str,
    child_offsets: &str,
    child_counts: &str,
    children: &str,
    weights: &str,
    leaf_values: &str,
    out: &str,
    n_nodes: u32,
    n_edges: u32,
) -> Result<Program, String> {
    if n_nodes == 0 {
        return Err(format!(
            "Fix: sum_product_evaluate requires n_nodes > 0, got {n_nodes}."
        ));
    }
    let edge_buffer_count = n_edges.max(1);

    let t = Expr::LogicalIndex { axis: 0 };
    let body = vec![Node::if_then(
        Expr::lt(t.clone(), Expr::u32(n_nodes)),
        sum_product_pass_body(
            kinds,
            child_offsets,
            child_counts,
            children,
            weights,
            leaf_values,
            out,
            // The single-pass form has no depth array to check against.
            None,
        ),
    )];

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(kinds, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(n_nodes),
            BufferDecl::storage(child_offsets, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(n_nodes),
            BufferDecl::storage(child_counts, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(n_nodes),
            BufferDecl::storage(children, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(edge_buffer_count),
            BufferDecl::storage(weights, 4, BufferAccess::ReadOnly, DataType::U32)
                .with_count(edge_buffer_count),
            BufferDecl::storage(leaf_values, 5, BufferAccess::ReadOnly, DataType::U32)
                .with_count(n_nodes),
            BufferDecl::storage(out, 6, BufferAccess::ReadWrite, DataType::U32).with_count(n_nodes),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    ))
}

/// The per-node evaluation body, shared by the single-pass
/// [`sum_product_evaluate`] and the depth-leveled
/// [`sum_product_evaluate_leveled`]. The logical point index (`LogicalIndex`) is
/// the node index; the caller gates the node-in-range (and, for the leveled
/// form, the depth-in-wave) predicate before running this body.
fn sum_product_pass_body(
    kinds: &str,
    child_offsets: &str,
    child_counts: &str,
    children: &str,
    weights: &str,
    leaf_values: &str,
    out: &str,
    depths: Option<&str>,
) -> Vec<Node> {
    let t = Expr::LogicalIndex { axis: 0 };
    let mut body = vec![
        Node::let_bind("kind", Expr::load(kinds, t.clone())),
        Node::let_bind("co", Expr::load(child_offsets, t.clone())),
        Node::let_bind("cc", Expr::load(child_counts, t.clone())),
    ];
    // A wave is only a topological order while every child sits at a strictly
    // smaller depth. Fed a `depths` array that does not describe the circuit,
    // an internal node reads a child the same wave is still writing, and the
    // two arms answer differently rather than wrongly: the reference walks
    // lanes in order and sees the child's new value, a device runs them at once
    // and sees the old one. Composition made that reachable from a registered
    // op, `quest_zero_fill -> sum_product_evaluate_leveled` piping an all-zero
    // buffer into `depths`, where the reference produced 5.0 for a node the
    // device left at 0. Trap instead, so an out-of-contract depth array is
    // refused by both arms rather than answered by neither.
    if let Some(depths) = depths {
        body.push(Node::if_then(
            Expr::ne(Expr::var("kind"), Expr::u32(KIND_LEAF)),
            vec![
                Node::let_bind("spc_depth", Expr::load(depths, t.clone())),
                Node::loop_for(
                    "spc_child_k",
                    Expr::u32(0),
                    Expr::var("cc"),
                    vec![
                        Node::let_bind(
                            "spc_child",
                            Expr::load(
                                children,
                                Expr::add(Expr::var("co"), Expr::var("spc_child_k")),
                            ),
                        ),
                        Node::if_then(
                            Expr::ge(
                                Expr::load(depths, Expr::var("spc_child")),
                                Expr::var("spc_depth"),
                            ),
                            vec![Node::trap(
                                Expr::var("spc_child"),
                                "sum-product-depth-not-topological",
                            )],
                        ),
                    ],
                ),
            ],
        ));
    }
    body.extend([
        // Leaf: out = leaf_values[t]
        Node::if_then(
            Expr::eq(Expr::var("kind"), Expr::u32(KIND_LEAF)),
            vec![Node::store(
                out,
                t.clone(),
                Expr::load(leaf_values, t.clone()),
            )],
        ),
        // Sum: out = Σ fixed_mul_16_16(children[child_idx], weight).
        Node::if_then(
            Expr::eq(Expr::var("kind"), Expr::u32(KIND_SUM)),
            vec![
                Node::let_bind("acc_sum", Expr::u32(0)),
                Node::loop_for(
                    "k",
                    Expr::u32(0),
                    Expr::var("cc"),
                    vec![
                        Node::let_bind(
                            "child_node",
                            Expr::load(children, Expr::add(Expr::var("co"), Expr::var("k"))),
                        ),
                        Node::let_bind(
                            "w",
                            Expr::load(weights, Expr::add(Expr::var("co"), Expr::var("k"))),
                        ),
                        Node::assign(
                            "acc_sum",
                            Expr::add(
                                Expr::var("acc_sum"),
                                crate::math::fixed::fixed_mul_16_16_expr(
                                    Expr::load(out, Expr::var("child_node")),
                                    Expr::var("w"),
                                ),
                            ),
                        ),
                    ],
                ),
                Node::store(out, t.clone(), Expr::var("acc_sum")),
            ],
        ),
        // Product: out = Π children, keeping each fixed-point multiply widened
        // before the 16-bit rescale.
        Node::if_then(
            Expr::eq(Expr::var("kind"), Expr::u32(KIND_PRODUCT)),
            vec![
                Node::let_bind("acc_prod", Expr::u32(1 << 16)), // 1.0 in 16.16
                Node::loop_for(
                    "kk",
                    Expr::u32(0),
                    Expr::var("cc"),
                    vec![
                        Node::let_bind(
                            "cn",
                            Expr::load(children, Expr::add(Expr::var("co"), Expr::var("kk"))),
                        ),
                        Node::assign(
                            "acc_prod",
                            crate::math::fixed::fixed_mul_16_16_expr(
                                Expr::var("acc_prod"),
                                Expr::load(out, Expr::var("cn")),
                            ),
                        ),
                    ],
                ),
                Node::store(out, t.clone(), Expr::var("acc_prod")),
            ],
        ),
    ]);
    body
}

/// Depth-ordered sum-product evaluation that is correct at ANY DAG depth.
///
/// The single-pass [`sum_product_evaluate`] has no barrier between topological
/// levels, so an internal node that reads ANOTHER internal node's `out` races
/// it (correct only for depth-1 circuits, see BACKLOG
/// `BUG-sum-product-multilevel-dag-no-topo-barrier`). This variant drives the
/// SAME per-node body through the shared depth-wave harness
/// [`crate::graph::level_wave::level_wave_program_with_buffers`]: at wave
/// `d = 0..max_depth`, every node whose `depths[node] == d` evaluates, and a
/// `GridSync`/`SeqCst` barrier between waves makes level-`d` writes globally
/// visible before level-`d+1` reads them. So a depth-`d` node always reads its
/// children's committed values.
///
/// `depths` is the per-node topological depth (leaves = 0, an internal node =
/// `1 + max(child depth)`); `max_depth` is one past the deepest node.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn sum_product_evaluate_leveled(
    depths: &str,
    kinds: &str,
    child_offsets: &str,
    child_counts: &str,
    children: &str,
    weights: &str,
    leaf_values: &str,
    out: &str,
    n_nodes: u32,
    n_edges: u32,
    max_depth: u32,
) -> Program {
    match try_sum_product_evaluate_leveled(
        depths,
        kinds,
        child_offsets,
        child_counts,
        children,
        weights,
        leaf_values,
        out,
        n_nodes,
        n_edges,
        max_depth,
    ) {
        Ok(program) => program,
        Err(error) => trap_program(OP_ID, Some((out, DataType::U32)), error),
    }
}

/// Fallible builder for [`sum_product_evaluate_leveled`].
#[allow(clippy::too_many_arguments)]
pub fn try_sum_product_evaluate_leveled(
    depths: &str,
    kinds: &str,
    child_offsets: &str,
    child_counts: &str,
    children: &str,
    weights: &str,
    leaf_values: &str,
    out: &str,
    n_nodes: u32,
    n_edges: u32,
    max_depth: u32,
) -> Result<Program, String> {
    if n_nodes == 0 {
        return Err(format!(
            "Fix: sum_product_evaluate_leveled requires n_nodes > 0, got {n_nodes}."
        ));
    }
    if max_depth == 0 {
        return Err(format!(
            "Fix: sum_product_evaluate_leveled requires max_depth > 0, got {max_depth}."
        ));
    }
    let edge_buffer_count = n_edges.max(1);

    // The depth-wave harness (binding 0 = `depths`) gates `lane < n_nodes` AND
    // `depths[lane] == current_wave`, so the per-node body needs no in-range /
    // depth re-check. The circuit's own buffers are declared at bindings 1..=7.
    let step_body = sum_product_pass_body(
        kinds,
        child_offsets,
        child_counts,
        children,
        weights,
        leaf_values,
        out,
        Some(depths),
    );
    let extra_buffers = vec![
        BufferDecl::storage(kinds, 1, BufferAccess::ReadOnly, DataType::U32).with_count(n_nodes),
        BufferDecl::storage(child_offsets, 2, BufferAccess::ReadOnly, DataType::U32)
            .with_count(n_nodes),
        BufferDecl::storage(child_counts, 3, BufferAccess::ReadOnly, DataType::U32)
            .with_count(n_nodes),
        BufferDecl::storage(children, 4, BufferAccess::ReadOnly, DataType::U32)
            .with_count(edge_buffer_count),
        BufferDecl::storage(weights, 5, BufferAccess::ReadOnly, DataType::U32)
            .with_count(edge_buffer_count),
        BufferDecl::storage(leaf_values, 6, BufferAccess::ReadOnly, DataType::U32)
            .with_count(n_nodes),
        BufferDecl::storage(out, 7, BufferAccess::ReadWrite, DataType::U32).with_count(n_nodes),
    ];

    Ok(
        crate::graph::level_wave::level_wave_program_with_buffers_and_op_id(
            OP_ID_LEVELED,
            step_body,
            depths,
            extra_buffers,
            max_depth,
            n_nodes,
        ),
    )
}

/// Host-side topological depth assignment for driving [`sum_product_evaluate_leveled`].
///
/// Returns `(depths, max_depth)` where `depths[i] == 0` for a childless node (a
/// leaf) and `1 + max(depths[child])` for an internal node, and `max_depth ==
/// max(depths) + 1`, the wave count [`crate::graph::level_wave`] must run so that
/// the deepest node fires (waves `0..max_depth`). A leaf-only circuit yields
/// all-zero depths and `max_depth == 1`.
///
/// The assignment guarantees `depths[parent] > depths[child]` for every edge, so
/// the depth-wave harness commits every child's `out` (at an earlier wave, behind a
/// barrier) before its parent reads it, the property that makes the leveled
/// evaluator correct at ANY depth.
///
/// Computed purely from `child_offsets`/`child_counts`/`children` (node kind is not
/// needed: an internal node always has `child_count >= 1`, a leaf has `0`). The
/// child indices are validated up front; a monotone relaxation reaches the fixed
/// point in at most `n_nodes` passes for a DAG, so failure to converge means the
/// child graph contains a cycle (not a valid sum-product circuit).
///
/// # Errors
///
/// Returns `Err` if a buffer is shorter than `n_nodes`, a child range overflows or
/// exceeds `children.len()`, a child index is `>= n_nodes`, or the child graph has
/// a cycle.
pub fn sum_product_depths(
    child_offsets: &[u32],
    child_counts: &[u32],
    children: &[u32],
    n_nodes: u32,
) -> Result<(Vec<u32>, u32), String> {
    let n = n_nodes as usize;
    if child_offsets.len() < n || child_counts.len() < n {
        return Err(format!(
            "Fix: sum_product_depths needs child_offsets/child_counts of length >= n_nodes={n_nodes}, got offsets={} counts={}.",
            child_offsets.len(),
            child_counts.len()
        ));
    }
    // Validate every child range + index up front so the relaxation below can index
    // `children`/`depths` without bounds risk.
    for i in 0..n {
        let co = child_offsets[i] as usize;
        let cc = child_counts[i] as usize;
        let end = co.checked_add(cc).ok_or_else(|| {
            format!("Fix: sum_product_depths child range overflows usize at node {i}.")
        })?;
        if end > children.len() {
            return Err(format!(
                "Fix: sum_product_depths node {i} child range {co}..{end} exceeds children len {}.",
                children.len()
            ));
        }
        for &c in &children[co..end] {
            if c as usize >= n {
                return Err(format!(
                    "Fix: sum_product_depths node {i} references child {c} outside n_nodes={n_nodes}."
                ));
            }
        }
    }
    let mut depths = vec![0u32; n];
    // Monotone relaxation `depth[i] = 1 + max(depth[child])`. Each pass propagates
    // correct depth one level further, so a DAG (longest path < n_nodes edges)
    // reaches a fixed point in at most `n_nodes` passes; the extra pass confirms no
    // change. If the (n_nodes+1)-th pass still changes something, the child graph
    // has a cycle.
    for _pass in 0..=n {
        let mut changed = false;
        for i in 0..n {
            let cc = child_counts[i] as usize;
            if cc == 0 {
                continue; // leaf / childless → depth 0
            }
            let co = child_offsets[i] as usize;
            let mut max_child = 0u32;
            for &c in &children[co..co + cc] {
                max_child = max_child.max(depths[c as usize]);
            }
            let candidate = max_child + 1;
            if candidate > depths[i] {
                depths[i] = candidate;
                changed = true;
            }
        }
        if !changed {
            let max_depth = depths.iter().copied().max().unwrap_or(0) + 1;
            return Ok((depths, max_depth));
        }
    }
    Err(
        "Fix: sum_product_depths did not converge in n_nodes passes (the circuit child graph has a cycle (a sum-product circuit must be a DAG))."
            .to_string(),
    )
}

const EXPECTED_SUM_PRODUCT_EVALUATE_OUTPUT_BYTES: [u8; 4] = [0, 0, 4, 0];
const EXPECTED_SUM_PRODUCT_ALL_NODES_OUTPUT_BYTES: [u8; 16] =
    [0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 10, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || sum_product_evaluate(
            "kinds",
            "child_offsets",
            "child_counts",
            "children",
            "weights",
            "leaf_values",
            "out",
            1,
            2,
        ),
        Some(|| {
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&[KIND_SUM]),
                vyre_primitives::wire::pack_u32_slice(&[0]),
                vyre_primitives::wire::pack_u32_slice(&[2]),
                vyre_primitives::wire::pack_u32_slice(&[0, 0]),
                vyre_primitives::wire::pack_u32_slice(&[1u32 << 15, 1u32 << 15]),
                vyre_primitives::wire::pack_u32_slice(&[0]),
                vyre_primitives::wire::pack_u32_slice(&[4u32 << 16]),
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_SUM_PRODUCT_EVALUATE_OUTPUT_BYTES.to_vec()]]
        }),
    )
}

// Cross-backend parity fixture for the DEPTH-LEVELED evaluator. The single-pass
// `sum_product_evaluate` above is registered; the leveled variant is now a production
// path (vyre-pass-engine's cost model dispatches it), so it must be walked by the
// conformance matrix too. A genuine DEPTH-2 circuit (a PRODUCT reading an internal SUM)
// exercises the barrier the single-pass form lacks:
//   n0=LEAF 2.0, n1=LEAF 3.0        (depth 0)
//   n2=SUM(n0,n1) unit weights = 5.0 (depth 1)
//   n3=PRODUCT(n2,n0) = 5.0*2.0 = 10.0 (depth 2, the root/point estimate)
// depths=[0,0,1,2], max_depth=3. Binding order: depths, kinds, child_offsets,
// child_counts, children, weights, leaf_values, out (seeded zero). reference_eval
// returns the sole RW buffer `out` = [2.0, 3.0, 5.0, 10.0] in 16.16.
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID_LEVELED,
        || sum_product_evaluate_leveled(
            "depths",
            "kinds",
            "child_offsets",
            "child_counts",
            "children",
            "weights",
            "leaf_values",
            "out",
            4,
            4,
            3,
        ),
        Some(|| {
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&[0, 0, 1, 2]),                 // depths
                vyre_primitives::wire::pack_u32_slice(&[KIND_LEAF, KIND_LEAF, KIND_SUM, KIND_PRODUCT]),
                vyre_primitives::wire::pack_u32_slice(&[0, 0, 0, 2]),                 // child_offsets
                vyre_primitives::wire::pack_u32_slice(&[0, 0, 2, 2]),                 // child_counts
                vyre_primitives::wire::pack_u32_slice(&[0, 1, 2, 0]),                 // children
                vyre_primitives::wire::pack_u32_slice(&[1u32 << 16, 1u32 << 16, 0, 0]), // weights (1.0,1.0)
                vyre_primitives::wire::pack_u32_slice(&[2u32 << 16, 3u32 << 16, 0, 0]), // leaf_values (2.0,3.0)
                vyre_primitives::wire::pack_u32_slice(&[0, 0, 0, 0]),                 // out (seed)
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_SUM_PRODUCT_ALL_NODES_OUTPUT_BYTES.to_vec()]]
        }),
    )
}

#[cfg(test)]
#[path = "sum_product_circuit_tests.rs"]
mod tests;
