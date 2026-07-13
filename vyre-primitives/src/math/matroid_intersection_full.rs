//! Matroid intersection full Edmonds algorithm (#P-PRIM-10).
//!
//! Grows a common independent set of two matroids by finding an
//! augmenting path in the exchange graph (a level-synchronous BFS from
//! the source set to the nearest sink) and toggling `set_x` along it.
//!
//! The exchange graph passed here is STATIC (read-only), so every
//! augmentation re-finds the identical path and the toggle has period
//! <= 2. The builder therefore emits a SINGLE augmentation and, for
//! `max_augmentations >= 2`, reproduces the sequential reference's
//! seen-state / max-cardinality termination directly (keep the toggled
//! set unless it strictly shrank, else revert) instead of unrolling an
//! oscillating loop. True multi-round intersection (rebuilding the
//! exchange graph each round) needs a matroid independence oracle this
//! primitive does not take, so it is out of scope by contract.
//!
//! Composes `matroid_exchange_bfs_step` and `path_reconstruct`.

use crate::graph::path_reconstruct::path_reconstruct;
use std::sync::Arc;
use vyre_foundation::ir::model::expr::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Op id.
pub const OP_ID: &str = "vyre-primitives::math::matroid_intersection_full";

/// Build a full matroid intersection Program.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn matroid_intersection_full(
    exchange_adj: &str,
    sources: &str,
    sinks: &str,
    set_x: &str,
    parent: &str,
    frontier: &str,
    next_frontier: &str,
    visited: &str,
    any_change: &str,
    path_out: &str,
    path_len: &str,
    n: u32,
    max_augmentations: u32,
) -> Program {
    if n == 0 {
        return crate::invalid_output_program(
            OP_ID,
            set_x,
            DataType::U32,
            "Fix: matroid_intersection_full requires n > 0, got 0.".to_string(),
        );
    }
    let Some(adj_count) = n.checked_mul(n) else {
        return crate::invalid_output_program(
            OP_ID,
            set_x,
            DataType::U32,
            format!("Fix: matroid_intersection_full exchange adjacency cells overflow u32: n={n}."),
        );
    };
    let mut nodes = Vec::new();

    // A STATIC exchange graph (read-only `exchange_adj`/`sources`/`sinks`) admits at most ONE
    // distinct augmentation: every augmentation re-initialises the frontier from the same `sources`
    // and runs the identical BFS, so it re-finds the IDENTICAL augmenting path P. The orbit of
    // `set_x` under "toggle P" therefore has period <= 2 ({s0, s0^P}); unrolling `max_augmentations`
    // copies would just OSCILLATE (BUG-matroid-megakernel-static-graph-oscillates-multi-augmentation).
    // We emit a SINGLE augmentation and encode the reference's seen-state / max-cardinality
    // termination directly in the toggle block below (keep-or-revert), so the result CONVERGES to the
    // sequential Edmonds reference for any `max_augmentations`. The augmentation body is wrapped in one
    // `Node::Block` so its `let` bindings (`found_sink`/`sink_node`, reconstruction locals) get a fresh
    // scope and do not violate IR rule V032.
    if max_augmentations >= 1 {
        let mut augmentation = Vec::new();
        // 1. Find augmenting path via BFS
        augmentation.push(Node::loop_for(
            "__i",
            Expr::u32(0),
            Expr::u32(n),
            vec![
                Node::store(
                    frontier,
                    Expr::var("__i"),
                    Expr::load(sources, Expr::var("__i")),
                ),
                Node::store(
                    visited,
                    Expr::var("__i"),
                    Expr::load(sources, Expr::var("__i")),
                ),
            ],
        ));
        augmentation.push(Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(n),
            vec![Node::if_then(
                Expr::ne(Expr::load(sources, Expr::var("i")), Expr::u32(0)),
                vec![Node::store(parent, Expr::var("i"), Expr::var("i"))],
            )],
        ));

        augmentation.push(Node::let_bind("found_sink", Expr::u32(0)));
        augmentation.push(Node::let_bind("sink_node", Expr::u32(0)));

        // Level 0: a node that is BOTH a source and a sink is a length-0 augmenting path, it can
        // be toggled directly with no BFS. cpu_ref detects this when it dequeues the source and
        // finds it is a sink. Scan ascending and take the MINIMUM-id such node so the choice is
        // deterministic and matches the level-synchronous CPU reference (min-id sink at the
        // earliest level). Guarded by `found_sink == 0` so only the first (lowest-id) one is kept.
        augmentation.push(Node::loop_for(
            "src_sink",
            Expr::u32(0),
            Expr::u32(n),
            vec![Node::if_then(
                Expr::and(
                    Expr::eq(Expr::var("found_sink"), Expr::u32(0)),
                    Expr::and(
                        Expr::ne(Expr::load(sources, Expr::var("src_sink")), Expr::u32(0)),
                        Expr::ne(Expr::load(sinks, Expr::var("src_sink")), Expr::u32(0)),
                    ),
                ),
                vec![
                    Node::assign("found_sink", Expr::u32(1)),
                    Node::assign("sink_node", Expr::var("src_sink")),
                ],
            )],
        ));

        augmentation.push(Node::loop_for(
            "step",
            Expr::u32(0),
            Expr::u32(n),
            vec![Node::if_then(
                Expr::eq(Expr::var("found_sink"), Expr::u32(0)),
                vec![
                    Node::loop_for(
                        "u",
                        Expr::u32(0),
                        Expr::u32(n),
                        vec![Node::if_then(
                            Expr::ne(Expr::load(frontier, Expr::var("u")), Expr::u32(0)),
                            vec![Node::loop_for(
                                "v",
                                Expr::u32(0),
                                Expr::u32(n),
                                vec![Node::if_then(
                                    Expr::and(
                                        Expr::eq(Expr::load(visited, Expr::var("v")), Expr::u32(0)),
                                        Expr::ne(
                                            Expr::load(
                                                exchange_adj,
                                                Expr::add(
                                                    Expr::mul(Expr::var("u"), Expr::u32(n)),
                                                    Expr::var("v"),
                                                ),
                                            ),
                                            Expr::u32(0),
                                        ),
                                    ),
                                    vec![
                                        // Discover v: mark visited, add to the next level, and set
                                        // its parent to THIS u. Because `u` is scanned ascending
                                        // and `visited` gates re-discovery, parent[v] resolves to
                                        // the LOWEST-id previous-level predecessor, deterministic
                                        // and identical to the CPU reference. Sink detection is NOT
                                        // done here (a later, higher `u` could discover a
                                        // lower-id sink `v`); it happens in a dedicated ascending
                                        // scan of the completed next_frontier below so the chosen
                                        // sink is the minimum-id one at this level.
                                        Node::store(visited, Expr::var("v"), Expr::u32(1)),
                                        Node::store(next_frontier, Expr::var("v"), Expr::u32(1)),
                                        Node::store(parent, Expr::var("v"), Expr::var("u")),
                                    ],
                                )],
                            )],
                        )],
                    ),
                    // This level's discoveries are complete. Select the MINIMUM-id sink among the
                    // newly-discovered nodes (ascending scan, guarded by `found_sink == 0` so the
                    // first, lowest-id, sink wins). Because the enclosing `step` loop is guarded
                    // by `found_sink == 0`, this fixes on the earliest BFS level that contains a
                    // sink and stops advancing, matching the CPU reference exactly.
                    Node::loop_for(
                        "sink_scan",
                        Expr::u32(0),
                        Expr::u32(n),
                        vec![Node::if_then(
                            Expr::and(
                                Expr::eq(Expr::var("found_sink"), Expr::u32(0)),
                                Expr::and(
                                    Expr::ne(
                                        Expr::load(next_frontier, Expr::var("sink_scan")),
                                        Expr::u32(0),
                                    ),
                                    Expr::ne(
                                        Expr::load(sinks, Expr::var("sink_scan")),
                                        Expr::u32(0),
                                    ),
                                ),
                            ),
                            vec![
                                Node::assign("found_sink", Expr::u32(1)),
                                Node::assign("sink_node", Expr::var("sink_scan")),
                            ],
                        )],
                    ),
                    Node::loop_for(
                        "i",
                        Expr::u32(0),
                        Expr::u32(n),
                        vec![Node::store(
                            frontier,
                            Expr::var("i"),
                            Expr::load(next_frontier, Expr::var("i")),
                        )],
                    ),
                    Node::loop_for(
                        "i",
                        Expr::u32(0),
                        Expr::u32(n),
                        vec![Node::store(next_frontier, Expr::var("i"), Expr::u32(0))],
                    ),
                ],
            )],
        ));

        let recon = path_reconstruct(parent, "target_node_buf", path_out, path_len, n);
        let mut on_sink = vec![
            Node::store("target_node_buf", Expr::u32(0), Expr::var("sink_node")),
            Node::Region {
                generator: Ident::from(OP_ID),
                source_region: None,
                body: Arc::new(recon.entry().to_vec()),
            },
            Node::let_bind("p_len", Expr::load(path_len, Expr::u32(0))),
            // Cardinality bookkeeping for the static-graph termination: as we toggle the path P,
            // count nodes this toggle ADDS (0->1 = `gained`) vs REMOVES (1->0 = `lost`), so the
            // net change |s0^P| - |s0| = gained - lost is known without a second augmentation.
            Node::let_bind("gained", Expr::u32(0)),
            Node::let_bind("lost", Expr::u32(0)),
            Node::loop_for(
                "idx",
                Expr::u32(0),
                Expr::var("p_len"),
                vec![
                    Node::let_bind("node", Expr::load(path_out, Expr::var("idx"))),
                    Node::let_bind("current_x", Expr::load(set_x, Expr::var("node"))),
                    // current_x in {0,1}: gained += 1 - current_x, lost += current_x (branchless).
                    Node::assign(
                        "gained",
                        Expr::add(
                            Expr::var("gained"),
                            Expr::sub(Expr::u32(1), Expr::var("current_x")),
                        ),
                    ),
                    Node::assign("lost", Expr::add(Expr::var("lost"), Expr::var("current_x"))),
                    Node::store(
                        set_x,
                        Expr::var("node"),
                        Expr::sub(Expr::u32(1), Expr::var("current_x")),
                    ),
                ],
            ),
        ];
        // Reference-faithful termination for a STATIC exchange graph. The BFS re-finds the SAME path
        // P every augmentation, so `set_x`'s orbit under "toggle P" is the 2-cycle {s0, s0^P}. The
        // sequential reference runs its seen-state loop and, on the FIRST repeat (the second
        // augmentation returns to a seen state), keeps the HIGHER-cardinality endpoint of that cycle
        // (ties -> the toggled state s0^P). For max_augmentations >= 2 we reproduce that EXACTLY by
        // reverting this single toggle iff it strictly SHRANK the set (lost > gained, i.e.
        // |s0^P| < |s0|); otherwise s0^P is already the max-cardinality endpoint and stands. For
        // max_augmentations == 1 the reference toggles unconditionally, so no revert is emitted.
        if max_augmentations >= 2 {
            on_sink.push(Node::if_then(
                Expr::lt(Expr::var("gained"), Expr::var("lost")),
                vec![Node::loop_for(
                    "revert_idx",
                    Expr::u32(0),
                    Expr::var("p_len"),
                    vec![
                        Node::let_bind(
                            "revert_node",
                            Expr::load(path_out, Expr::var("revert_idx")),
                        ),
                        Node::let_bind("revert_x", Expr::load(set_x, Expr::var("revert_node"))),
                        Node::store(
                            set_x,
                            Expr::var("revert_node"),
                            Expr::sub(Expr::u32(1), Expr::var("revert_x")),
                        ),
                    ],
                )],
            ));
        }
        augmentation.push(Node::if_then(
            Expr::ne(Expr::var("found_sink"), Expr::u32(0)),
            on_sink,
        ));
        nodes.push(Node::Block(augmentation));
    }

    Program::wrapped(
        vec![
            BufferDecl::storage(exchange_adj, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(adj_count),
            BufferDecl::storage(sources, 1, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::storage(sinks, 2, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::storage(set_x, 3, BufferAccess::ReadWrite, DataType::U32).with_count(n),
            BufferDecl::storage(parent, 4, BufferAccess::ReadWrite, DataType::U32).with_count(n),
            BufferDecl::storage(frontier, 5, BufferAccess::ReadWrite, DataType::U32).with_count(n),
            BufferDecl::storage(next_frontier, 6, BufferAccess::ReadWrite, DataType::U32)
                .with_count(n),
            BufferDecl::storage(visited, 7, BufferAccess::ReadWrite, DataType::U32).with_count(n),
            BufferDecl::storage(any_change, 8, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
            BufferDecl::storage(path_out, 9, BufferAccess::ReadWrite, DataType::U32).with_count(n),
            BufferDecl::storage(path_len, 10, BufferAccess::ReadWrite, DataType::U32).with_count(1),
            BufferDecl::storage(
                "target_node_buf",
                11,
                BufferAccess::ReadWrite,
                DataType::U32,
            )
            .with_count(1),
        ],
        [1, 1, 1],
        vec![Node::Region {
            generator: Ident::from(OP_ID),
            source_region: None,
            // This is a single-threaded sequential Edmonds algorithm: a serial BFS
            // with a shared queue, path reconstruction, and a NON-IDEMPOTENT toggle
            // `set_x[node] = 1 - set_x[node]`. If dispatched with more than one
            // invocation (e.g. buffer-shape grid inference spawns one lane per set
            // element), every lane redundantly runs the whole program and the toggle
            // RACES, two lanes each flip the same slot, so the result diverges from
            // the sequential CPU reference (an idempotent kernel like toposort would
            // survive this, but a toggle cannot). Guard the entire body to lane 0 so
            // exactly one invocation executes it, making the kernel correct under ANY
            // dispatch grid (the canonical GPU idiom for a serial region).
            body: Arc::new(vec![Node::if_then(
                Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
                nodes,
            )]),
        }],
    )
}

/// CPU reference: One full Edmonds augmentation.
#[cfg(any(test, feature = "cpu-parity"))]
#[must_use]
pub fn cpu_ref(
    exchange_adj: &[u32],
    sources: &[u32],
    sinks: &[u32],
    set_x: &[u32],
    n: usize,
) -> Vec<u32> {
    let mut out = Vec::new();
    let mut parent = Vec::new();
    let mut visited = Vec::new();
    let mut queue = Vec::new();
    try_cpu_ref_into(
        exchange_adj,
        sources,
        sinks,
        set_x,
        n,
        &mut out,
        &mut parent,
        &mut visited,
        &mut queue,
    )
    .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - matroid_intersection_full cpu_ref failed: invalid exchange-graph buffers");
    out
}

/// Fallible CPU reference: One full Edmonds augmentation.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_cpu_ref(
    exchange_adj: &[u32],
    sources: &[u32],
    sinks: &[u32],
    set_x: &[u32],
    n: usize,
) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    let mut parent = Vec::new();
    let mut visited = Vec::new();
    let mut queue = Vec::new();
    try_cpu_ref_into(
        exchange_adj,
        sources,
        sinks,
        set_x,
        n,
        &mut out,
        &mut parent,
        &mut visited,
        &mut queue,
    )?;
    Ok(out)
}

/// CPU reference using caller-owned BFS scratch.
#[cfg(any(test, feature = "cpu-parity"))]
#[allow(clippy::too_many_arguments)]
pub fn cpu_ref_into(
    exchange_adj: &[u32],
    sources: &[u32],
    sinks: &[u32],
    set_x: &[u32],
    n: usize,
    out: &mut Vec<u32>,
    parent: &mut Vec<u32>,
    visited: &mut Vec<u32>,
    queue: &mut Vec<usize>,
) {
    try_cpu_ref_into(
        exchange_adj,
        sources,
        sinks,
        set_x,
        n,
        out,
        parent,
        visited,
        queue,
    )
    .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - matroid_intersection_full cpu_ref_into failed: invalid exchange-graph buffers");
}

/// Fallible CPU reference using caller-owned BFS scratch.
#[cfg(any(test, feature = "cpu-parity"))]
#[allow(clippy::too_many_arguments)]
pub fn try_cpu_ref_into(
    exchange_adj: &[u32],
    sources: &[u32],
    sinks: &[u32],
    set_x: &[u32],
    n: usize,
    out: &mut Vec<u32>,
    parent: &mut Vec<u32>,
    visited: &mut Vec<u32>,
    queue: &mut Vec<usize>,
) -> Result<(), String> {
    let adj_cells = n.checked_mul(n).ok_or_else(|| {
        format!("matroid_intersection_full CPU oracle n*n overflows usize: n={n}.")
    })?;
    require_len("exchange_adj", exchange_adj.len(), adj_cells)?;
    require_len("sources", sources.len(), n)?;
    require_len("sinks", sinks.len(), n)?;
    require_len("set_x", set_x.len(), n)?;
    reserve_u32(out, n, "set output")?;
    reserve_u32(parent, n, "parent scratch")?;
    reserve_u32(visited, n, "visited scratch")?;
    reserve_usize(queue, n, "queue scratch")?;

    out.clear();
    out.extend_from_slice(&set_x[..n]);
    parent.clear();
    parent.resize(n, 0);
    visited.clear();
    visited.resize(n, 0);
    // LEVEL-SYNCHRONOUS BFS, matches the IR bitmap kernel exactly. A FIFO queue is a
    // sequential-only construct the GPU cannot realize with a plain bitmap frontier, so the shared
    // spec is: parent[v] = the LOWEST-id previous-level predecessor with an edge to v, and the
    // augmenting path ends at the MINIMUM-id sink on the EARLIEST BFS level (level 0 = the source
    // set). The `queue` scratch is repurposed as the current-frontier flag buffer (0/1 per node);
    // `next_frontier` holds the level being discovered.
    queue.clear();
    queue.resize(n, 0);
    let mut next_frontier = vec![0u32; n];

    for i in 0..n {
        if sources[i] != 0 {
            queue[i] = 1;
            visited[i] = 1;
            parent[i] = i as u32;
        }
    }

    let mut found_sink = None;
    // Level 0: a source that is also a sink is a length-0 augmenting path. Min-id wins.
    for v in 0..n {
        if sources[v] != 0 && sinks[v] != 0 {
            found_sink = Some(v);
            break;
        }
    }

    // Discover level by level (bounded by n (the longest simple BFS chain) until a sink appears).
    let mut step = 0;
    while found_sink.is_none() && step < n {
        for v in next_frontier.iter_mut().take(n) {
            *v = 0;
        }
        // Scan frontier u ascending, neighbors v ascending: the first (lowest) u to reach an
        // unvisited v owns parent[v] (deterministic and identical to the IR).
        for u in 0..n {
            if queue[u] != 0 {
                for v in 0..n {
                    if visited[v] == 0 && exchange_adj[u * n + v] != 0 {
                        visited[v] = 1;
                        next_frontier[v] = 1;
                        parent[v] = u as u32;
                    }
                }
            }
        }
        // Min-id sink discovered at this level.
        for v in 0..n {
            if next_frontier[v] != 0 && sinks[v] != 0 {
                found_sink = Some(v);
                break;
            }
        }
        // Advance the frontier to the level just discovered.
        for i in 0..n {
            queue[i] = next_frontier[i] as usize;
        }
        step += 1;
    }

    if let Some(sink) = found_sink {
        let mut curr = sink;
        loop {
            out[curr] = 1 - out[curr];
            let next = parent[curr] as usize;
            if next == curr {
                break;
            }
            curr = next;
        }
    }
    Ok(())
}

#[cfg(any(test, feature = "cpu-parity"))]
fn require_len(name: &str, got: usize, need: usize) -> Result<(), String> {
    if got < need {
        Err(format!(
            "matroid_intersection_full CPU oracle buffer `{name}` is too short: got {got}, need {need}."
        ))
    } else {
        Ok(())
    }
}

#[cfg(any(test, feature = "cpu-parity"))]
fn reserve_u32(out: &mut Vec<u32>, len: usize, name: &str) -> Result<(), String> {
    if len > out.capacity() {
        crate::graph::scratch::reserve_graph_items(
            out,
            len - out.len(),
            "matroid intersection CPU oracle",
            name,
        )?;
    }
    Ok(())
}

#[cfg(any(test, feature = "cpu-parity"))]
fn reserve_usize(out: &mut Vec<usize>, len: usize, name: &str) -> Result<(), String> {
    if len > out.capacity() {
        crate::graph::scratch::reserve_graph_items(
            out,
            len - out.len(),
            "matroid intersection CPU oracle",
            name,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_ref_single_augmentation() {
        let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
        let src = vec![1, 0, 0];
        let snk = vec![0, 0, 1];
        let x = vec![0, 0, 0];
        let x_new = cpu_ref(&adj, &src, &snk, &x, 3);
        assert_eq!(x_new, vec![1, 1, 1]);
    }

    #[test]
    fn cpu_ref_into_reuses_bfs_storage() {
        let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
        let src = vec![1, 0, 0];
        let snk = vec![0, 0, 1];
        let x = vec![0, 0, 0];
        let mut out = Vec::new();
        let mut parent = Vec::new();
        let mut visited = Vec::new();
        let mut queue = Vec::new();

        cpu_ref_into(
            &adj,
            &src,
            &snk,
            &x,
            3,
            &mut out,
            &mut parent,
            &mut visited,
            &mut queue,
        );
        let out_ptr = out.as_ptr();
        let queue_ptr = queue.as_ptr();
        cpu_ref_into(
            &adj,
            &src,
            &snk,
            &x,
            3,
            &mut out,
            &mut parent,
            &mut visited,
            &mut queue,
        );

        assert_eq!(out, vec![1, 1, 1]);
        assert_eq!(out.as_ptr(), out_ptr);
        assert_eq!(queue.as_ptr(), queue_ptr);
    }

    #[test]
    fn cpu_ref_into_truncates_stale_scratch_without_reallocating() {
        let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
        let src = vec![1, 0, 0];
        let snk = vec![0, 0, 1];
        let x = vec![0, 0, 0];
        let mut out = Vec::with_capacity(8);
        let mut parent = Vec::with_capacity(8);
        let mut visited = Vec::with_capacity(8);
        let mut queue = Vec::with_capacity(8);
        out.extend([99u32; 8]);
        parent.extend([99u32; 8]);
        visited.extend([99u32; 8]);
        queue.extend([99usize; 8]);
        let out_ptr = out.as_ptr();
        let parent_ptr = parent.as_ptr();
        let visited_ptr = visited.as_ptr();
        let queue_ptr = queue.as_ptr();

        try_cpu_ref_into(
            &adj,
            &src,
            &snk,
            &x,
            3,
            &mut out,
            &mut parent,
            &mut visited,
            &mut queue,
        )
        .unwrap();

        assert_eq!(out, vec![1, 1, 1]);
        assert_eq!(out.as_ptr(), out_ptr);
        assert_eq!(parent.as_ptr(), parent_ptr);
        assert_eq!(visited.as_ptr(), visited_ptr);
        assert_eq!(queue.as_ptr(), queue_ptr);
    }

    #[test]
    fn try_cpu_ref_rejects_short_buffers() {
        let err = try_cpu_ref(&[0], &[1, 0], &[0, 1], &[0, 0], 2).unwrap_err();
        assert!(err.contains("exchange_adj"), "{err}");
    }

    #[test]
    fn generated_cpu_ref_into_matches_independent_three_node_bfs_matrix() {
        let n = 3;
        let mut out = Vec::new();
        let mut parent = Vec::new();
        let mut visited = Vec::new();
        let mut queue = Vec::new();

        for edge_mask in 0u32..512 {
            let mut adj = vec![0u32; n * n];
            for bit in 0..(n * n) {
                adj[bit] = (edge_mask >> bit) & 1;
            }
            for source_mask in 0u32..8 {
                let sources = mask_to_vec(source_mask, n);
                for sink_mask in 0u32..8 {
                    let sinks = mask_to_vec(sink_mask, n);
                    for seed_mask in 0u32..8 {
                        let seed = mask_to_vec(seed_mask, n);
                        cpu_ref_into(
                            &adj,
                            &sources,
                            &sinks,
                            &seed,
                            n,
                            &mut out,
                            &mut parent,
                            &mut visited,
                            &mut queue,
                        );
                        assert_eq!(
                            out,
                            independent_one_augmentation(&adj, &sources, &sinks, &seed, n),
                            "edge_mask={edge_mask:#011b} source_mask={source_mask:#05b} sink_mask={sink_mask:#05b} seed_mask={seed_mask:#05b}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn program_buffer_layout() {
        let p = matroid_intersection_full(
            "adj", "src", "snk", "x", "p", "f", "nf", "v", "ch", "po", "pl", 4, 1,
        );
        assert_eq!(p.buffers().len(), 12);
    }

    fn mask_to_vec(mask: u32, n: usize) -> Vec<u32> {
        (0..n).map(|idx| (mask >> idx) & 1).collect()
    }

    // Independent oracle for the SAME level-synchronous augmenting-path rule as `cpu_ref`, but
    // built with a deliberately different structure. Bellman-Ford-style level relaxation followed
    // by explicit min-id-predecessor reconstruction, so it is a genuine cross-check rather than a
    // copy of the reference's frontier-scan loop. Rule: level 0 = the source set; the path ends at
    // the minimum-id sink on the earliest level that has one; parent[v] = the lowest-id node one
    // level up with an edge into v.
    fn independent_one_augmentation(
        exchange_adj: &[u32],
        sources: &[u32],
        sinks: &[u32],
        set_x: &[u32],
        n: usize,
    ) -> Vec<u32> {
        let mut result = set_x.to_vec();
        const INF: u32 = u32::MAX;
        let mut level = vec![INF; n];
        for v in 0..n {
            if sources[v] != 0 {
                level[v] = 0;
            }
        }
        // Relax up to n rounds (a chain of n nodes has depth n-1); stop early at the fixpoint.
        for _ in 0..n {
            let mut changed = false;
            for u in 0..n {
                if level[u] != INF {
                    for v in 0..n {
                        if exchange_adj[u * n + v] != 0 && level[u] + 1 < level[v] {
                            level[v] = level[u] + 1;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        // Earliest level (L*) that contains a reachable sink, then the min-id sink at L*.
        let mut lstar = INF;
        for v in 0..n {
            if sinks[v] != 0 && level[v] < lstar {
                lstar = level[v];
            }
        }
        let sink = if lstar == INF {
            None
        } else {
            (0..n).find(|&v| sinks[v] != 0 && level[v] == lstar)
        };

        if let Some(sink_node) = sink {
            let mut node = sink_node;
            loop {
                result[node] = 1 - result[node];
                if level[node] == 0 {
                    break; // reached a source
                }
                let target = level[node] - 1;
                let pred = (0..n)
                    .find(|&u| level[u] == target && exchange_adj[u * n + node] != 0)
                    .expect("a node at level L has a level-(L-1) predecessor by construction");
                node = pred;
            }
        }
        result
    }
}
