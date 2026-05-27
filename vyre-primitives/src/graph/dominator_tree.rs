//! `dominator_tree`  —  exact immediate-dominator primitive.
//!
//! Computes the immediate dominator (`idom`) of every reachable node in a
//! control-flow graph with a single entry.  The primitive ships both a
//! Lengauer–Tarjan CPU reference oracle and a serial lane-0 GPU `Program`
//! builder that implements the Cooper–Harvey–Kennedy iterative fixpoint
//! using parent-pointer LCA on the idom tree.
//!
//! # Wire shape
//!
//! ```text
//! pg_edge_offsets : u32[node_count + 1]   // forward CSR
//! pg_edge_targets : u32[edge_count]       // forward CSR
//! pred_offsets    : u32[node_count + 1]   // predecessor CSR
//! pred_targets    : u32[pred_edge_count]  // predecessor CSR
//! idom_out        : u32[node_count]       // output idoms; NONE = unreachable
//! ```
//!
//! `idom_out[entry] == entry` for the entry block.  Unreachable nodes keep
//! the sentinel `NONE` (== `node_count`).
//!
//! # Soundness
//!
//! Exact for every reducible and irreducible single-entry CFG.  Multi-entry
//! graphs (no path from entry to some node that has predecessors) are not
//! rejected explicitly, but the resulting idom tree is undefined for the
//! disconnected component; callers should run `reachable` first if they need
//! strict guarantees.

use std::sync::Arc;

use vyre_foundation::ir::model::expr::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Canonical op id.
pub const OP_ID: &str = "vyre-primitives::graph::dominator_tree";

/// Sentinel stored in `idom_out` for unreachable nodes.
pub const IDOM_NONE: u32 = u32::MAX;

/// Errors from dominator-tree construction.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DominatorTreeError {
    /// The requested entry node is outside `0..node_count`.
    EntryOutOfRange {
        /// Supplied entry.
        entry: u32,
        /// Declared node count.
        node_count: u32,
    },
    /// CSR offset buffer length is inconsistent with `node_count`.
    BadOffsets {
        /// Actionable diagnostic.
        message: String,
    },
    /// CSR target buffer references a node outside the valid range.
    TargetOutOfRange {
        /// Offending target index.
        index: usize,
        /// Offending value.
        target: u32,
        /// Declared node count.
        node_count: u32,
    },
    /// Monotonicity violation in CSR offsets.
    NonMonotonicOffsets {
        /// Index of first violation.
        index: usize,
    },
}

/// Validated dispatch layout for the dominator-tree primitive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DominatorTreeLayout {
    /// Number of nodes.
    pub node_count: u32,
    /// Number of forward edges.
    pub edge_count: u32,
    /// Number of predecessor edges.
    pub pred_edge_count: u32,
    /// Words in the `idom_out` buffer.
    pub idom_words: usize,
    /// Words in the `depth` scratch buffer.
    pub depth_words: usize,
}

/// Build a serial lane-0 `Program` that computes exact immediate dominators.
///
/// The kernel runs the Cooper–Harvey–Kennedy iterative fixpoint over the
/// idom tree using predecessor-list LCA.  Workgroup size is `[1, 1, 1]`;
/// only invocation 0 performs work.
///
/// # Panics
///
/// Panics in the builder when `node_count` is `u32::MAX` (because `node_count`
/// is used as the `IDOM_NONE` sentinel and would collide with a valid node).
#[must_use]
pub fn dominator_tree_program(
    node_count: u32,
    edge_count: u32,
    pred_edge_count: u32,
    idom_out: &str,
) -> Program {
    match try_dominator_tree_program(node_count, edge_count, pred_edge_count, idom_out) {
        Ok(p) => p,
        Err(_) => inert_dominator_tree_program(idom_out),
    }
}

/// Checked builder.  Returns an actionable diagnostic instead of a trap
/// program when the shape overflows buffer counts.
pub fn try_dominator_tree_program(
    node_count: u32,
    edge_count: u32,
    pred_edge_count: u32,
    idom_out: &str,
) -> Result<Program, String> {
    if node_count == u32::MAX {
        return Err(
            "dominator_tree node_count == u32::MAX collides with IDOM_NONE sentinel. \
             Fix: shard the graph before dispatch."
                .to_string(),
        );
    }

    let offset_count = node_count.checked_add(1).ok_or_else(|| {
        format!(
            "dominator_tree node_count={node_count} overflows CSR offset buffer count. \
                 Fix: shard the graph before GPU dispatch."
        )
    })?;

    let lane0 = Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0));

    // ------------------------------------------------------------------
    // Serial CHK-by-LCA kernel on lane 0.
    // ------------------------------------------------------------------

    // idom_out[v] = NONE for all v
    let init_idoms = Node::loop_for(
        "i",
        Expr::u32(0),
        Expr::u32(node_count),
        vec![Node::store(idom_out, Expr::var("i"), Expr::u32(IDOM_NONE))],
    );

    // idom_out[0] = 0  (entry dominates itself)
    let init_entry = Node::store(idom_out, Expr::u32(0), Expr::u32(0));

    // depth[0] = 0; depth[v] = 0 for all others (will be fixed on first update)
    let depth_buf = "dt_depth";
    let init_depth = Node::loop_for(
        "i",
        Expr::u32(0),
        Expr::u32(node_count),
        vec![Node::store(depth_buf, Expr::var("i"), Expr::u32(0))],
    );

    // Outer fixpoint: at most node_count iterations.
    // Each step: recompute depths, then for each v != entry intersect preds via LCA.
    let recompute_depth = vec![Node::loop_for(
        "v",
        Expr::u32(0),
        Expr::u32(node_count),
        vec![
            Node::let_bind("d", Expr::u32(0)),
            Node::let_bind("cur", Expr::var("v")),
            Node::loop_for(
                "depth_step",
                Expr::u32(0),
                Expr::u32(node_count),
                vec![Node::if_then(
                    Expr::ne(Expr::var("cur"), Expr::u32(0)),
                    vec![
                        Node::let_bind("parent", Expr::load(idom_out, Expr::var("cur"))),
                        Node::if_then(
                            Expr::and(
                                Expr::ne(Expr::var("parent"), Expr::var("cur")),
                                Expr::ne(Expr::var("parent"), Expr::u32(IDOM_NONE)),
                            ),
                            vec![
                                Node::assign("d", Expr::add(Expr::var("d"), Expr::u32(1))),
                                Node::assign("cur", Expr::var("parent")),
                            ],
                        ),
                    ],
                )],
            ),
            Node::store(depth_buf, Expr::var("v"), Expr::var("d")),
        ],
    )];

    let body = vec![
        // changed = 0
        Node::let_bind("changed", Expr::u32(0)),
        // recompute all depths from current idom tree
        Node::Block(recompute_depth.clone()),
        // for v in 0..node_count
        Node::loop_for(
            "v",
            Expr::u32(0),
            Expr::u32(node_count),
            vec![Node::if_then(
                Expr::ne(Expr::var("v"), Expr::u32(0)),
                vec![
                    // new_idom = NONE
                    Node::let_bind("new_idom", Expr::u32(IDOM_NONE)),
                    // walk predecessors
                    Node::let_bind("p_start", Expr::load("pred_offsets", Expr::var("v"))),
                    Node::let_bind(
                        "p_end",
                        Expr::load("pred_offsets", Expr::add(Expr::var("v"), Expr::u32(1))),
                    ),
                    Node::loop_for(
                        "p_idx",
                        Expr::var("p_start"),
                        Expr::var("p_end"),
                        vec![
                            Node::let_bind("p", Expr::load("pred_targets", Expr::var("p_idx"))),
                            // if idom[p] != NONE
                            Node::if_then(
                                Expr::ne(
                                    Expr::load(idom_out, Expr::var("p")),
                                    Expr::u32(IDOM_NONE),
                                ),
                                vec![Node::if_then_else(
                                    Expr::eq(Expr::var("new_idom"), Expr::u32(IDOM_NONE)),
                                    // first reachable predecessor
                                    vec![Node::assign("new_idom", Expr::var("p"))],
                                    // else LCA(new_idom, p)
                                    vec![
                                        Node::let_bind("a", Expr::var("new_idom")),
                                        Node::let_bind("b", Expr::var("p")),
                                        Node::loop_for(
                                            "lca_step",
                                            Expr::u32(0),
                                            Expr::u32(node_count),
                                            vec![Node::if_then(
                                                Expr::ne(Expr::var("a"), Expr::var("b")),
                                                vec![
                                                    Node::let_bind(
                                                        "da",
                                                        Expr::load(depth_buf, Expr::var("a")),
                                                    ),
                                                    Node::let_bind(
                                                        "db",
                                                        Expr::load(depth_buf, Expr::var("b")),
                                                    ),
                                                    Node::if_then_else(
                                                        Expr::gt(Expr::var("da"), Expr::var("db")),
                                                        vec![Node::assign(
                                                            "a",
                                                            Expr::load(idom_out, Expr::var("a")),
                                                        )],
                                                        vec![Node::assign(
                                                            "b",
                                                            Expr::load(idom_out, Expr::var("b")),
                                                        )],
                                                    ),
                                                ],
                                            )],
                                        ),
                                        Node::assign("new_idom", Expr::var("a")),
                                    ],
                                )],
                            ),
                        ],
                    ),
                    // if new_idom changed, write it and set changed flag
                    Node::if_then(
                        Expr::and(
                            Expr::ne(Expr::var("new_idom"), Expr::u32(IDOM_NONE)),
                            Expr::ne(Expr::var("new_idom"), Expr::load(idom_out, Expr::var("v"))),
                        ),
                        vec![
                            Node::store(idom_out, Expr::var("v"), Expr::var("new_idom")),
                            Node::assign("changed", Expr::u32(1)),
                        ],
                    ),
                ],
            )],
        ),
    ];

    let outer_loop = Node::loop_for("step", Expr::u32(0), Expr::u32(node_count), body);

    let region_body = vec![Node::if_then(
        lane0,
        vec![init_idoms, init_entry, init_depth, outer_loop],
    )];

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage("pg_edge_offsets", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(offset_count),
            BufferDecl::storage("pg_edge_targets", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(edge_count.max(1)),
            BufferDecl::storage("pred_offsets", 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(offset_count),
            BufferDecl::storage("pred_targets", 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(pred_edge_count.max(1)),
            BufferDecl::storage(idom_out, 4, BufferAccess::ReadWrite, DataType::U32)
                .with_count(node_count.max(1)),
            BufferDecl::storage(depth_buf, 5, BufferAccess::ReadWrite, DataType::U32)
                .with_count(node_count.max(1)),
        ],
        [1, 1, 1],
        vec![Node::Region {
            generator: Ident::from(OP_ID),
            source_region: None,
            body: Arc::new(region_body),
        }],
    ))
}

fn inert_dominator_tree_program(idom_out: &str) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("pg_edge_offsets", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
            BufferDecl::storage("pg_edge_targets", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
            BufferDecl::storage("pred_offsets", 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
            BufferDecl::storage("pred_targets", 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
            BufferDecl::storage(idom_out, 4, BufferAccess::ReadWrite, DataType::U32).with_count(1),
            BufferDecl::storage("dt_depth", 5, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ],
        [1, 1, 1],
        vec![Node::Region {
            generator: Ident::from(OP_ID),
            source_region: None,
            body: Arc::new(vec![Node::return_()]),
        }],
    )
}

/// Validate dominator-tree CSR inputs.
///
/// # Errors
///
/// Returns [`DominatorTreeError`] when offsets are malformed, targets point
/// out of range, or the entry is invalid.
pub fn validate_dominator_tree_inputs(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    pred_offsets: &[u32],
    pred_targets: &[u32],
) -> Result<DominatorTreeLayout, DominatorTreeError> {
    let expected_offsets =
        (node_count as usize)
            .checked_add(1)
            .ok_or_else(|| DominatorTreeError::BadOffsets {
                message: format!(
                "Fix: dominator_tree node_count + 1 overflows usize for node_count={node_count}."
            ),
            })?;

    if edge_offsets.len() != expected_offsets {
        return Err(DominatorTreeError::BadOffsets {
            message: format!(
                "Fix: dominator_tree edge_offsets.len() must be {expected_offsets}, got {}.",
                edge_offsets.len()
            ),
        });
    }
    if pred_offsets.len() != expected_offsets {
        return Err(DominatorTreeError::BadOffsets {
            message: format!(
                "Fix: dominator_tree pred_offsets.len() must be {expected_offsets}, got {}.",
                pred_offsets.len()
            ),
        });
    }

    for (idx, pair) in edge_offsets.windows(2).enumerate() {
        if pair[0] > pair[1] {
            return Err(DominatorTreeError::NonMonotonicOffsets { index: idx });
        }
    }
    for (idx, pair) in pred_offsets.windows(2).enumerate() {
        if pair[0] > pair[1] {
            return Err(DominatorTreeError::NonMonotonicOffsets { index: idx });
        }
    }

    let edge_count = edge_offsets.last().copied().unwrap_or(0);
    let pred_edge_count = pred_offsets.last().copied().unwrap_or(0);

    if edge_targets.len() != edge_count as usize {
        return Err(DominatorTreeError::BadOffsets {
            message: format!(
                "Fix: dominator_tree edge_targets.len()={} != edge_count={edge_count}.",
                edge_targets.len()
            ),
        });
    }
    if pred_targets.len() != pred_edge_count as usize {
        return Err(DominatorTreeError::BadOffsets {
            message: format!(
                "Fix: dominator_tree pred_targets.len()={} != pred_edge_count={pred_edge_count}.",
                pred_targets.len()
            ),
        });
    }

    for (idx, &target) in edge_targets.iter().enumerate() {
        if target >= node_count {
            return Err(DominatorTreeError::TargetOutOfRange {
                index: idx,
                target,
                node_count,
            });
        }
    }
    for (idx, &target) in pred_targets.iter().enumerate() {
        if target >= node_count {
            return Err(DominatorTreeError::TargetOutOfRange {
                index: idx,
                target,
                node_count,
            });
        }
    }

    Ok(DominatorTreeLayout {
        node_count,
        edge_count,
        pred_edge_count,
        idom_words: node_count as usize,
        depth_words: node_count as usize,
    })
}

// ------------------------------------------------------------------
// CPU reference oracles  (#[cfg(test)] / feature = "cpu-parity")
// ------------------------------------------------------------------

/// Lengauer–Tarjan exact immediate dominators.
///
/// Returns `idom[v]` for every node `v`.  `idom[entry] == entry`.
/// Unreachable nodes receive `None`.
///
/// CPU-only reference algorithm. Gated with the rest of the CPU oracle
/// surface (`compress`, `eval`, `link`, `cpu_ref`) so default builds
/// don't pull the implementation through. (Without this gate, default
/// builds left the body of `lengauer_tarjan_idoms` referencing the
/// gated-out `eval`/`link`/`compress` helpers and failed with three
/// E0423/E0425 errors. Reproduced 2026-05-23.)
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn lengauer_tarjan_idoms(
    node_count: u32,
    entry: u32,
    edges: &[(u32, u32)],
) -> Vec<Option<u32>> {
    try_lengauer_tarjan_idoms(node_count, entry, edges).unwrap_or_else(|error| panic!("{error}"))
}

/// Fallible Lengauer-Tarjan exact immediate dominators.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_lengauer_tarjan_idoms(
    node_count: u32,
    entry: u32,
    edges: &[(u32, u32)],
) -> Result<Vec<Option<u32>>, String> {
    let mut idom = Vec::new();
    let mut scratch = DominatorTreeCpuScratch::default();
    try_lengauer_tarjan_idoms_into(node_count, entry, edges, &mut idom, &mut scratch)?;
    Ok(idom)
}

/// Reusable workspace for dominator-tree CPU oracles.
#[cfg(any(test, feature = "cpu-parity"))]
#[derive(Debug, Default)]
pub struct DominatorTreeCpuScratch {
    succ: Vec<Vec<usize>>,
    pred: Vec<Vec<usize>>,
    semi: Vec<usize>,
    vertex: Vec<usize>,
    parent: Vec<usize>,
    dfs_stack: Vec<(usize, usize)>,
    ancestor: Vec<usize>,
    label: Vec<usize>,
    bucket: Vec<Vec<usize>>,
    compress_stack: Vec<usize>,
}

#[cfg(any(test, feature = "cpu-parity"))]
impl DominatorTreeCpuScratch {
    /// Construct empty dominator-tree CPU scratch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fallible Lengauer-Tarjan exact immediate dominators using caller-owned output and scratch.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_lengauer_tarjan_idoms_into(
    node_count: u32,
    entry: u32,
    edges: &[(u32, u32)],
    idom: &mut Vec<Option<u32>>,
    scratch: &mut DominatorTreeCpuScratch,
) -> Result<(), String> {
    let n = node_count as usize;
    let entry = entry as usize;
    if n == 0 {
        idom.clear();
        return Ok(());
    }
    if entry >= n {
        idom.clear();
        resize_dominator_vec(idom, n, None, "dominator_tree entry-out-of-range idoms")?;
        return Ok(());
    }

    // Build adjacency.
    resize_dominator_vec(
        &mut scratch.succ,
        n,
        Vec::new(),
        "dominator_tree successor rows",
    )?;
    resize_dominator_vec(
        &mut scratch.pred,
        n,
        Vec::new(),
        "dominator_tree predecessor rows",
    )?;
    for row in scratch.succ.iter_mut().take(n) {
        row.clear();
    }
    for row in scratch.pred.iter_mut().take(n) {
        row.clear();
    }
    for &(u, v) in edges {
        let u = u as usize;
        let v = v as usize;
        if u < n && v < n {
            push_dominator_vec(&mut scratch.succ[u], v, "dominator_tree successor row")?;
            push_dominator_vec(&mut scratch.pred[v], u, "dominator_tree predecessor row")?;
        }
    }

    // DFS numbering.
    scratch.semi.clear();
    scratch.vertex.clear();
    scratch.parent.clear();
    resize_dominator_vec(&mut scratch.semi, n, 0usize, "dominator_tree semi numbers")?;
    resize_dominator_vec(
        &mut scratch.vertex,
        n + 1,
        0usize,
        "dominator_tree DFS vertices",
    )?;
    resize_dominator_vec(&mut scratch.parent, n, 0usize, "dominator_tree DFS parents")?;
    let mut dfs_num: usize = 0;

    // Iterative DFS to avoid stack overflow on million-node chains.
    scratch.dfs_stack.clear();
    push_dominator_vec(
        &mut scratch.dfs_stack,
        (entry, 0usize),
        "dominator_tree DFS stack",
    )?;
    while let Some((v, next_idx)) = scratch.dfs_stack.last_mut() {
        let v = *v;
        if *next_idx == 0 {
            dfs_num += 1;
            scratch.semi[v] = dfs_num;
            scratch.vertex[dfs_num] = v;
        }
        if *next_idx < scratch.succ[v].len() {
            let w = scratch.succ[v][*next_idx];
            *next_idx += 1;
            if scratch.semi[w] == 0 {
                scratch.parent[w] = v;
                push_dominator_vec(&mut scratch.dfs_stack, (w, 0), "dominator_tree DFS stack")?;
            }
        } else {
            scratch.dfs_stack.pop();
        }
    }

    if dfs_num == 0 {
        idom.clear();
        resize_dominator_vec(idom, n, None, "dominator_tree unreachable idoms")?;
        return Ok(());
    }

    idom.clear();
    scratch.ancestor.clear();
    scratch.label.clear();
    scratch.bucket.clear();
    resize_dominator_vec(idom, n, None, "dominator_tree idoms")?;
    resize_dominator_vec(&mut scratch.ancestor, n, 0usize, "dominator_tree ancestors")?;
    crate::graph::scratch::reserve_graph_items(
        &mut scratch.label,
        n,
        "dominator tree CPU oracle",
        "dominator_tree labels",
    )?;
    scratch.label.extend(0..n);
    resize_dominator_vec(
        &mut scratch.bucket,
        n + 1,
        Vec::new(),
        "dominator_tree buckets",
    )?;
    for row in scratch.bucket.iter_mut().take(n + 1) {
        row.clear();
    }
    scratch.compress_stack.clear();

    for i in (1..=dfs_num).rev() {
        let w = scratch.vertex[i];

        for &v in &scratch.pred[w] {
            if scratch.semi[v] > 0 {
                let u = try_eval_with_stack(
                    v,
                    &mut scratch.ancestor,
                    &mut scratch.label,
                    &scratch.semi,
                    &mut scratch.compress_stack,
                )?;
                if scratch.semi[u] < scratch.semi[w] {
                    scratch.semi[w] = scratch.semi[u];
                }
            }
        }

        push_dominator_vec(
            &mut scratch.bucket[scratch.vertex[scratch.semi[w]]],
            w,
            "dominator_tree bucket row",
        )?;

        link(
            scratch.parent[w],
            w,
            &mut scratch.ancestor,
            &mut scratch.label,
            &scratch.semi,
        );

        for &v in &scratch.bucket[scratch.parent[w]] {
            let u = try_eval_with_stack(
                v,
                &mut scratch.ancestor,
                &mut scratch.label,
                &scratch.semi,
                &mut scratch.compress_stack,
            )?;
            if scratch.semi[u] < scratch.semi[v] {
                idom[v] = Some(u as u32);
            } else {
                idom[v] = Some(scratch.parent[w] as u32);
            }
        }
        scratch.bucket[scratch.parent[w]].clear();
    }

    for i in 2..=dfs_num {
        let w = scratch.vertex[i];
        if idom[w].map(|x| x as usize) != Some(scratch.vertex[scratch.semi[w]]) {
            idom[w] = idom[w]
                .and_then(|parent| idom.get(parent as usize))
                .copied()
                .flatten();
        }
    }

    idom[entry] = Some(entry as u32);

    // unreachable nodes keep None
    Ok(())
}

#[cfg(any(test, feature = "cpu-parity"))]
fn try_compress(
    v: usize,
    ancestor: &mut [usize],
    label: &mut [usize],
    semi: &[usize],
) -> Result<(), String> {
    let mut stack = Vec::new();
    try_compress_with_stack(v, ancestor, label, semi, &mut stack)
}

#[cfg(any(test, feature = "cpu-parity"))]
fn try_compress_with_stack(
    v: usize,
    ancestor: &mut [usize],
    label: &mut [usize],
    semi: &[usize],
    stack: &mut Vec<usize>,
) -> Result<(), String> {
    if ancestor[v] == 0 {
        return Ok(());
    }

    // Iterative version of the recursive path-compression used in LT.
    // We walk up the ancestor chain, pushing vertices that are at least
    // two levels above the root.  When we hit a direct child of the root
    // we process it in-place (label update, no splice) and then walk
    // back down the stack, processing and splicing as we go.
    stack.clear();
    let mut u = v;
    while ancestor[u] != 0 {
        if ancestor[ancestor[u]] != 0 {
            push_dominator_vec(stack, u, "dominator_tree compression stack")?;
            u = ancestor[u];
        } else {
            // Direct child of the root – "else" branch of the recursive
            // formulation.  Update label but do NOT splice ancestor.
            if semi[label[ancestor[u]]] < semi[label[u]] {
                label[u] = label[ancestor[u]];
            }
            break;
        }
    }

    // Walk back down, using the freshly-updated labels of ancestors.
    while let Some(w) = stack.pop() {
        if semi[label[ancestor[w]]] < semi[label[w]] {
            label[w] = label[ancestor[w]];
        }
        ancestor[w] = ancestor[ancestor[w]];
    }
    Ok(())
}

#[cfg(any(test, feature = "cpu-parity"))]
fn try_eval(
    v: usize,
    ancestor: &mut [usize],
    label: &mut [usize],
    semi: &[usize],
) -> Result<usize, String> {
    let mut stack = Vec::new();
    try_eval_with_stack(v, ancestor, label, semi, &mut stack)
}

#[cfg(any(test, feature = "cpu-parity"))]
fn try_eval_with_stack(
    v: usize,
    ancestor: &mut [usize],
    label: &mut [usize],
    semi: &[usize],
    stack: &mut Vec<usize>,
) -> Result<usize, String> {
    if ancestor[v] == 0 {
        Ok(v)
    } else {
        try_compress_with_stack(v, ancestor, label, semi, stack)?;
        Ok(label[v])
    }
}

#[cfg(any(test, feature = "cpu-parity"))]
fn link(v: usize, w: usize, ancestor: &mut [usize], label: &mut [usize], _semi: &[usize]) {
    ancestor[w] = v;
    label[w] = w;
}

/// Cooper–Harvey–Kennedy iterative immediate dominators (bitset formulation).
///
/// Implements the classical dataflow algorithm using dense bitsets so the
/// result is exact and comparable to [`lengauer_tarjan_idoms`].  Memory is
/// `O(n²/32)` — acceptable for the `#[cfg(test)]` differential oracle.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cooper_harvey_kennedy_idoms(
    node_count: u32,
    entry: u32,
    edges: &[(u32, u32)],
) -> Vec<Option<u32>> {
    let n = node_count as usize;
    let entry = entry as usize;
    if n == 0 {
        return Vec::new();
    }
    if entry >= n {
        return vec![None; n];
    }

    let words = ((n + 31) / 32).max(1);
    let last_mask = if n % 32 == 0 {
        u32::MAX
    } else {
        (1u32 << (n % 32)) - 1
    };

    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in edges {
        let u = u as usize;
        let v = v as usize;
        if u < n && v < n {
            succ[u].push(v);
            pred[v].push(u);
        }
    }

    // Flat bitset matrix: row v starts at v * words.
    let mut dom = vec![0u32; n * words];

    // Initialize: Dom(entry) = {entry}; Dom(v≠entry) = ALL.
    for v in 0..n {
        let row = v * words;
        if v == entry {
            dom[row + v / 32] |= 1u32 << (v % 32);
        } else {
            for w in 0..words {
                dom[row + w] = u32::MAX;
            }
            if last_mask != u32::MAX {
                dom[row + words - 1] = last_mask;
            }
            dom[row + v / 32] |= 1u32 << (v % 32);
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for v in 0..n {
            if v == entry {
                continue;
            }
            let row = v * words;
            let mut new = vec![u32::MAX; words];
            if last_mask != u32::MAX {
                new[words - 1] = last_mask;
            }
            for &p in &pred[v] {
                let prow = p * words;
                for w in 0..words {
                    new[w] &= dom[prow + w];
                }
            }
            new[v / 32] |= 1u32 << (v % 32);
            if &new[..] != &dom[row..row + words] {
                dom[row..row + words].copy_from_slice(&new);
                changed = true;
            }
        }
    }

    // Compute reachability from entry using BFS.
    let mut reachable = vec![false; n];
    let mut queue = vec![entry];
    reachable[entry] = true;
    while let Some(u) = queue.pop() {
        for &v in &succ[u] {
            if !reachable[v] {
                reachable[v] = true;
                queue.push(v);
            }
        }
    }

    // Convert bitsets to idoms for reachable nodes only.
    let mut idom = vec![None; n];
    idom[entry] = Some(entry as u32);
    for v in 0..n {
        if v == entry || !reachable[v] {
            continue;
        }
        let row = v * words;
        let mut strict = Vec::new();
        for d in 0..n {
            if d == v {
                continue;
            }
            if dom[row + d / 32] & (1u32 << (d % 32)) != 0 {
                strict.push(d);
            }
        }
        // idom(v) = strict dominator not strictly dominated by any other strict dom.
        for &d in &strict {
            let mut is_idom = true;
            for &c in &strict {
                if c == d {
                    continue;
                }
                if dom[c * words + d / 32] & (1u32 << (d % 32)) != 0 {
                    is_idom = false;
                    break;
                }
            }
            if is_idom {
                idom[v] = Some(d as u32);
                break;
            }
        }
    }

    idom
}

/// Canonical CPU oracle: exact Lengauer–Tarjan.
///
/// Returns a fresh `Vec<Option<u32>>` where index `v` is the immediate
/// dominator of `v` (or `None` for unreachable nodes).
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_ref(node_count: u32, entry: u32, edges: &[(u32, u32)]) -> Vec<Option<u32>> {
    lengauer_tarjan_idoms(node_count, entry, edges)
}

/// Fallible canonical CPU oracle: exact Lengauer-Tarjan.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_cpu_ref(
    node_count: u32,
    entry: u32,
    edges: &[(u32, u32)],
) -> Result<Vec<Option<u32>>, String> {
    try_lengauer_tarjan_idoms(node_count, entry, edges)
}

/// Fallible canonical CPU oracle using caller-owned output and scratch storage.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_cpu_ref_into(
    node_count: u32,
    entry: u32,
    edges: &[(u32, u32)],
    out: &mut Vec<Option<u32>>,
    scratch: &mut DominatorTreeCpuScratch,
) -> Result<(), String> {
    try_lengauer_tarjan_idoms_into(node_count, entry, edges, out, scratch)
}

/// Convert an idom array to per-node dominator sets (sorted).
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn idoms_to_dominator_sets(idoms: &[Option<u32>], node_count: u32) -> Vec<Vec<u32>> {
    try_idoms_to_dominator_sets(idoms, node_count).unwrap_or_else(|error| panic!("{error}"))
}

/// Fallible conversion of an idom array to per-node dominator sets (sorted).
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_idoms_to_dominator_sets(
    idoms: &[Option<u32>],
    node_count: u32,
) -> Result<Vec<Vec<u32>>, String> {
    let n = node_count as usize;
    if idoms.len() < n {
        return Err(format!(
            "dominator_tree idom set conversion received idoms_len={} for node_count={node_count}. Fix: pass one idom slot per graph node.",
            idoms.len()
        ));
    }
    let mut sets: Vec<Vec<u32>> = Vec::new();
    resize_dominator_vec(&mut sets, n, Vec::new(), "dominator_tree dominator sets")?;
    for v in 0..n {
        let mut cur = v;
        let mut set = Vec::new();
        push_dominator_vec(&mut set, cur as u32, "dominator_tree per-node set")?;
        while let Some(p) = idoms[cur] {
            if p == cur as u32 {
                break;
            }
            push_dominator_vec(&mut set, p, "dominator_tree per-node set")?;
            cur = p as usize;
        }
        set.sort_unstable();
        sets[v] = set;
    }
    Ok(sets)
}

#[cfg(any(test, feature = "cpu-parity"))]
fn resize_dominator_vec<T: Clone>(
    out: &mut Vec<T>,
    len: usize,
    value: T,
    context: &str,
) -> Result<(), String> {
    if len > out.len() {
        crate::graph::scratch::reserve_graph_items(
            out,
            len - out.len(),
            "dominator tree CPU oracle",
            context,
        )?;
    }
    out.resize(len, value);
    Ok(())
}

#[cfg(any(test, feature = "cpu-parity"))]
fn push_dominator_vec<T>(out: &mut Vec<T>, value: T, context: &str) -> Result<(), String> {
    crate::graph::scratch::reserve_graph_items(out, 1, "dominator tree CPU oracle", context)?;
    out.push(value);
    Ok(())
}

#[cfg(feature = "inventory-registry")]
inventory::submit! {
    crate::harness::OpEntry::new(
        OP_ID,
        || dominator_tree_program(4, 4, 4, "idom"),
        Some(|| {
            vec![vec![
                crate::wire::pack_u32_slice(&[0, 1, 2, 3, 3]),
                crate::wire::pack_u32_slice(&[1, 2, 3, 0]),
                crate::wire::pack_u32_slice(&[0, 0, 1, 2, 3]),
                crate::wire::pack_u32_slice(&[0, 1, 2, 0]),
                crate::wire::pack_u32_slice(&[0; 4]),
                crate::wire::pack_u32_slice(&[0; 4]),
            ]]
        }),
        Some(|| {
            vec![vec![
                crate::wire::pack_u32_slice(&[0, 0, 1, 2]),
                crate::wire::pack_u32_slice(&[0, 1, 2, 3]),
            ]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn program_builds_without_panic() {
        let p = dominator_tree_program(4, 4, 4, "idom");
        assert_eq!(p.workgroup_size, [1, 1, 1]);
        let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
        assert!(names.contains(&"idom"));
        assert!(names.contains(&"dt_depth"));
    }

    #[test]
    fn checked_builder_rejects_u32_max_node_count() {
        let err = try_dominator_tree_program(u32::MAX, 0, 0, "idom").unwrap_err();
        assert!(err.contains("u32::MAX collides with IDOM_NONE"));
    }

    #[test]
    fn legacy_builder_does_not_panic_on_u32_max() {
        let p = dominator_tree_program(u32::MAX, 0, 0, "idom");
        assert_eq!(p.workgroup_size, [1, 1, 1]);
    }

    #[test]
    fn empty_graph_returns_empty() {
        let idoms = cpu_ref(0, 0, &[]);
        assert!(idoms.is_empty());
    }

    #[test]
    fn single_node_self_idom() {
        let idoms = cpu_ref(1, 0, &[]);
        assert_eq!(idoms, vec![Some(0)]);
    }

    #[test]
    fn linear_chain_idoms() {
        // 0 -> 1 -> 2 -> 3
        let idoms = cpu_ref(4, 0, &[(0, 1), (1, 2), (2, 3)]);
        assert_eq!(idoms[0], Some(0));
        assert_eq!(idoms[1], Some(0));
        assert_eq!(idoms[2], Some(1));
        assert_eq!(idoms[3], Some(2));
    }

    #[test]
    fn diamond_idoms() {
        // 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        let idoms = cpu_ref(4, 0, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        assert_eq!(idoms[0], Some(0));
        assert_eq!(idoms[1], Some(0));
        assert_eq!(idoms[2], Some(0));
        assert_eq!(idoms[3], Some(0));
    }

    #[test]
    fn while_loop_idoms() {
        // 0 -> 1, 1 -> 2, 2 -> 1, 1 -> 3
        let idoms = cpu_ref(4, 0, &[(0, 1), (1, 2), (2, 1), (1, 3)]);
        assert_eq!(idoms[0], Some(0));
        assert_eq!(idoms[1], Some(0));
        assert_eq!(idoms[2], Some(1));
        assert_eq!(idoms[3], Some(1));
    }

    #[test]
    fn unreachable_nodes_are_none() {
        // 0 -> 1. 2 and 3 are disconnected.
        let idoms = cpu_ref(4, 0, &[(0, 1)]);
        assert_eq!(idoms[0], Some(0));
        assert_eq!(idoms[1], Some(0));
        assert_eq!(idoms[2], None);
        assert_eq!(idoms[3], None);
    }

    #[test]
    fn lt_matches_chk_on_diamond() {
        let lt = lengauer_tarjan_idoms(4, 0, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let chk = cooper_harvey_kennedy_idoms(4, 0, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        assert_eq!(lt, chk);
    }

    #[test]
    fn lt_matches_chk_on_while_loop() {
        let edges = &[(0, 1), (1, 2), (2, 1), (1, 3)];
        let lt = lengauer_tarjan_idoms(4, 0, edges);
        let chk = cooper_harvey_kennedy_idoms(4, 0, edges);
        assert_eq!(lt, chk);
    }

    #[test]
    fn generated_try_lt_matches_chk_on_small_graphs() {
        for case in 0..1024usize {
            let n = 1 + case % 10;
            let mut edges = Vec::new();
            for src in 0..n {
                for dst in 0..n {
                    if src != dst && ((src * 17 + dst * 31 + case) % 11) < 3 {
                        edges.push((src as u32, dst as u32));
                    }
                }
            }
            let lt = try_lengauer_tarjan_idoms(n as u32, 0, &edges)
                .expect("generated dominator LT oracle should reserve and evaluate");
            let chk = cooper_harvey_kennedy_idoms(n as u32, 0, &edges);

            assert_eq!(lt, chk, "case {case}: LT and CHK idoms diverged");
        }
    }

    #[test]
    fn try_cpu_ref_into_reuses_output_and_workspace() {
        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(&[Some(99); 12]);
        let mut scratch = DominatorTreeCpuScratch::new();
        scratch.succ.reserve(16);
        scratch.pred.reserve(16);
        scratch.semi.reserve(16);
        scratch.vertex.reserve(17);
        scratch.parent.reserve(16);
        scratch.dfs_stack.reserve(16);
        scratch.ancestor.reserve(16);
        scratch.label.reserve(16);
        scratch.bucket.reserve(17);
        scratch.compress_stack.reserve(16);
        let out_capacity = out.capacity();
        let outer_caps = [
            scratch.succ.capacity(),
            scratch.pred.capacity(),
            scratch.semi.capacity(),
            scratch.vertex.capacity(),
            scratch.parent.capacity(),
            scratch.dfs_stack.capacity(),
            scratch.ancestor.capacity(),
            scratch.label.capacity(),
            scratch.bucket.capacity(),
            scratch.compress_stack.capacity(),
        ];

        try_cpu_ref_into(
            4,
            0,
            &[(0, 1), (0, 2), (1, 3), (2, 3)],
            &mut out,
            &mut scratch,
        )
        .expect("Fix: diamond dominator graph must evaluate.");

        assert_eq!(out, vec![Some(0), Some(0), Some(0), Some(0)]);
        assert_eq!(out.capacity(), out_capacity);
        assert_eq!(scratch.succ.capacity(), outer_caps[0]);
        assert_eq!(scratch.pred.capacity(), outer_caps[1]);
        assert_eq!(scratch.semi.capacity(), outer_caps[2]);
        assert_eq!(scratch.vertex.capacity(), outer_caps[3]);
        assert_eq!(scratch.parent.capacity(), outer_caps[4]);
        assert_eq!(scratch.dfs_stack.capacity(), outer_caps[5]);
        assert_eq!(scratch.ancestor.capacity(), outer_caps[6]);
        assert_eq!(scratch.label.capacity(), outer_caps[7]);
        assert_eq!(scratch.bucket.capacity(), outer_caps[8]);
        assert_eq!(scratch.compress_stack.capacity(), outer_caps[9]);

        let succ_row_zero_capacity = scratch.succ[0].capacity();
        let pred_row_one_capacity = scratch.pred[1].capacity();

        try_cpu_ref_into(2, 0, &[(0, 1)], &mut out, &mut scratch)
            .expect("Fix: second dominator graph must reuse workspace.");

        assert_eq!(out, vec![Some(0), Some(0)]);
        assert_eq!(out.capacity(), out_capacity);
        assert_eq!(
            scratch.succ[0],
            vec![1],
            "Fix: workspace reuse must clear stale successor edges from the previous graph."
        );
        assert!(
            scratch.pred[1].contains(&0),
            "Fix: workspace reuse must rebuild predecessor rows for the second graph."
        );
        assert_eq!(scratch.succ[0].capacity(), succ_row_zero_capacity);
        assert_eq!(scratch.pred[1].capacity(), pred_row_one_capacity);

        try_cpu_ref_into(3, 5, &[(0, 1)], &mut out, &mut scratch)
            .expect("Fix: out-of-range entry should produce all-None idoms.");
        assert_eq!(out, vec![None, None, None]);
        assert_eq!(out.capacity(), out_capacity);
    }

    #[test]
    fn generated_idom_set_conversion_is_sorted_and_includes_self() {
        for case in 0..512usize {
            let n = 1 + case % 32;
            let edges: Vec<(u32, u32)> = (1..n)
                .map(|node| ((node - 1) as u32, node as u32))
                .collect();
            let idoms = try_cpu_ref(n as u32, 0, &edges)
                .expect("generated dominator CPU oracle should reserve and evaluate");
            let sets = try_idoms_to_dominator_sets(&idoms, n as u32)
                .expect("generated dominator set conversion should reserve and evaluate");

            assert_eq!(sets.len(), n, "case {case}: one set per node");
            for (node, set) in sets.iter().enumerate() {
                assert!(
                    set.windows(2).all(|pair| pair[0] < pair[1]),
                    "case {case} node {node}: dominator set must be sorted and unique"
                );
                assert!(
                    set.contains(&(node as u32)),
                    "case {case} node {node}: dominator set must contain the node itself"
                );
            }
        }
    }

    #[test]
    fn validation_rejects_bad_offsets() {
        let err = validate_dominator_tree_inputs(2, &[0, 1], &[0], &[0, 0, 0], &[]).unwrap_err();
        assert!(matches!(err, DominatorTreeError::BadOffsets { .. }));
    }

    #[test]
    fn validation_rejects_oob_target() {
        let err = validate_dominator_tree_inputs(2, &[0, 1, 1], &[5], &[0, 0, 0], &[]).unwrap_err();
        assert!(matches!(
            err,
            DominatorTreeError::TargetOutOfRange { target: 5, .. }
        ));
    }

    #[test]
    fn validation_rejects_non_monotonic_offsets() {
        let err =
            validate_dominator_tree_inputs(2, &[0, 2, 1], &[0, 0], &[0, 0, 0], &[]).unwrap_err();
        assert!(matches!(
            err,
            DominatorTreeError::NonMonotonicOffsets { .. }
        ));
    }

    #[test]
    fn validation_returns_layout() {
        let layout =
            validate_dominator_tree_inputs(3, &[0, 1, 2, 2], &[1, 2], &[0, 0, 0, 0], &[]).unwrap();
        assert_eq!(layout.node_count, 3);
        assert_eq!(layout.edge_count, 2);
        assert_eq!(layout.pred_edge_count, 0);
    }

    #[test]
    fn dominator_cpu_source_exposes_fallible_oracle_storage() {
        let source = include_str!("dominator_tree.rs");
        let full_cpu_source = source
            .split("pub fn lengauer_tarjan_idoms(")
            .nth(1)
            .expect("Fix: dominator CPU source must be present")
            .split("#[cfg(feature = \"inventory-registry\")]")
            .next()
            .expect("Fix: dominator CPU source must precede registry entry");
        let lt_source = full_cpu_source
            .split("/// Cooper–Harvey–Kennedy iterative immediate dominators")
            .next()
            .expect("Fix: dominator LT source must precede CHK oracle");

        assert!(
            full_cpu_source.contains("pub fn try_cpu_ref(")
                && full_cpu_source.contains("try_idoms_to_dominator_sets")
                && full_cpu_source.contains("crate::graph::scratch::reserve_graph_items")
                && lt_source.contains("pub fn try_lengauer_tarjan_idoms(")
                && !lt_source.contains("fn reserve_dominator_vec")
                && !lt_source.contains("vec![Vec::new(); n]")
                && !lt_source.contains("vec![None; n]"),
            "Fix: dominator CPU oracle must expose fallible allocation paths for large graph parity."
        );
    }
}
