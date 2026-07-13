//! The ONE canonical CSR neighbor-expansion edge-scan, shared by every
//! `csr_forward_or_changed` variant and the persistent-BFS batch step.
//!
//! It lives at `graph/` level, a peer of both `csr_forward_or_changed` and
//! `persistent_bfs`: because it is the common parent of every consumer; parking
//! it inside one variant would force the others to reach across a sibling module.
//!
//! The inner loop: "for a source node whose frontier bit is set, walk its CSR
//! edge range, and for each edge that passes the kind-mask, atomic-OR the target
//! bit into the frontier, marking the run changed on a newly-set bit", was
//! hand-written five times across the graph module (single-serial, grid-sync
//! parallel, batch, batch-global, and persistent-BFS batch step). Byte-identical
//! copies drift (the persistent_bfs seed bug and the `{n,m}` lowering bugs both
//! hid in near-duplicate paths), so it lives here ONCE and every caller supplies
//! only what genuinely differs: how a word index maps into its frontier buffer,
//! and what to write when a new bit is discovered.
//!
//! Two entry points at different levels:
//! - [`csr_edge_scan_nodes`] reads the source bit INLINE then expands, for callers
//!   that check activity where they expand (the serial and batch-global paths).
//! - [`csr_edge_expand_nodes`] is the edge-walk ALONE, for callers that read the
//!   source bit into a pre-barrier snapshot and guard the expansion themselves (the
//!   parallel snapshot path and the persistent-BFS batch step). `csr_edge_scan_nodes`
//!   is itself the inline source-bit read wrapped around this.

use vyre_foundation::ir::{Expr, Node};

use crate::graph::program_graph::{
    ProgramGraphShape, NAME_EDGE_KIND_MASK, NAME_EDGE_OFFSETS, NAME_EDGE_TARGETS,
};

fn local_name(prefix: &str, n: &str) -> String {
    if prefix.is_empty() {
        n.to_string()
    } else {
        format!("{prefix}_{n}")
    }
}

/// Emit ONLY the CSR edge walk for source node `src` (no source-activity guard):
/// load the `[edge_start, edge_end)` range, and for every edge passing
/// `edge_kind_mask`, atomic-OR the target bit into `frontier_out` at
/// `frontier_index(dst_word)`, running `on_new_bit()` when a bit flips 0→1.
///
/// Callers that snapshot source activity before a barrier (the one-hop-per-iteration
/// guarantee) wrap this themselves; callers that read activity inline use
/// [`csr_edge_scan_nodes`].
#[must_use]
pub(in crate::graph) fn csr_edge_expand_nodes(
    shape: ProgramGraphShape,
    frontier_out: &str,
    src: Expr,
    frontier_index: impl Fn(Expr) -> Expr,
    on_new_bit: impl Fn() -> Vec<Node>,
    edge_kind_mask: u32,
    prefix: &str,
) -> Vec<Node> {
    let name = |n: &str| local_name(prefix, n);
    let edge_start = name("edge_start");
    let edge_end = name("edge_end");
    let edge_iter = name("e");
    let kind_mask = name("kind_mask");
    let dst = name("dst");
    let dst_word_idx = name("dst_word_idx");
    let dst_bit = name("dst_bit");
    let old = name("old");

    vec![
        Node::let_bind(
            edge_start.as_str(),
            Expr::load(NAME_EDGE_OFFSETS, src.clone()),
        ),
        Node::let_bind(
            edge_end.as_str(),
            Expr::load(NAME_EDGE_OFFSETS, Expr::add(src.clone(), Expr::u32(1))),
        ),
        Node::loop_for(
            edge_iter.as_str(),
            Expr::var(edge_start.as_str()),
            Expr::var(edge_end.as_str()),
            vec![
                Node::let_bind(
                    kind_mask.as_str(),
                    Expr::load(NAME_EDGE_KIND_MASK, Expr::var(edge_iter.as_str())),
                ),
                Node::if_then(
                    Expr::ne(
                        Expr::bitand(Expr::var(kind_mask.as_str()), Expr::u32(edge_kind_mask)),
                        Expr::u32(0),
                    ),
                    vec![
                        Node::let_bind(
                            dst.as_str(),
                            Expr::load(NAME_EDGE_TARGETS, Expr::var(edge_iter.as_str())),
                        ),
                        Node::if_then(
                            Expr::lt(Expr::var(dst.as_str()), Expr::u32(shape.node_count)),
                            vec![
                                Node::let_bind(
                                    dst_word_idx.as_str(),
                                    frontier_index(Expr::shr(
                                        Expr::var(dst.as_str()),
                                        Expr::u32(5),
                                    )),
                                ),
                                Node::let_bind(
                                    dst_bit.as_str(),
                                    Expr::shl(
                                        Expr::u32(1),
                                        Expr::bitand(Expr::var(dst.as_str()), Expr::u32(31)),
                                    ),
                                ),
                                Node::let_bind(
                                    old.as_str(),
                                    Expr::atomic_or(
                                        frontier_out,
                                        Expr::var(dst_word_idx.as_str()),
                                        Expr::var(dst_bit.as_str()),
                                    ),
                                ),
                                Node::if_then(
                                    Expr::eq(
                                        Expr::bitand(
                                            Expr::var(old.as_str()),
                                            Expr::var(dst_bit.as_str()),
                                        ),
                                        Expr::u32(0),
                                    ),
                                    on_new_bit(),
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ]
}

/// Emit the CSR neighbor expansion for one source node `src`, reading its frontier
/// bit INLINE and expanding only when set.
///
/// The two axes of variation across callers:
/// - `frontier_index(word) -> storage_index`: maps a bitset WORD index to the
///   position in `frontier_out` (identity for a single bitset; `query_word_base +
///   word` for a flat per-query batch).
/// - `on_new_bit() -> Vec<Node>`: what to run when a target bit flips from 0 to 1
///   (a local `assign(changed, 1)`, or `atomic_or(changed, <index>, 1)`).
///
/// `prefix` disambiguates the local bindings when this body is inlined more than
/// once into a larger kernel (the no-shadowing validator forbids reused names);
/// pass `""` for a standalone program to keep the canonical unprefixed names.
///
/// This is exactly the inline source-bit read wrapped around
/// [`csr_edge_expand_nodes`], so a caller's emitted IR is unchanged (locked by the
/// graph oracle/fixpoint matrices + the csr module's per-variant parity tests).
#[must_use]
pub(in crate::graph) fn csr_edge_scan_nodes(
    shape: ProgramGraphShape,
    frontier_out: &str,
    src: Expr,
    frontier_index: impl Fn(Expr) -> Expr,
    on_new_bit: impl Fn() -> Vec<Node>,
    edge_kind_mask: u32,
    prefix: &str,
) -> Vec<Node> {
    let name = |n: &str| local_name(prefix, n);
    let word_idx = name("word_idx");
    let bit_mask = name("bit_mask");
    let src_word = name("src_word");

    vec![
        Node::let_bind(
            word_idx.as_str(),
            frontier_index(Expr::shr(src.clone(), Expr::u32(5))),
        ),
        Node::let_bind(
            bit_mask.as_str(),
            Expr::shl(Expr::u32(1), Expr::bitand(src.clone(), Expr::u32(31))),
        ),
        Node::let_bind(
            src_word.as_str(),
            Expr::load(frontier_out, Expr::var(word_idx.as_str())),
        ),
        Node::if_then(
            Expr::ne(
                Expr::bitand(Expr::var(src_word.as_str()), Expr::var(bit_mask.as_str())),
                Expr::u32(0),
            ),
            csr_edge_expand_nodes(
                shape,
                frontier_out,
                src,
                frontier_index,
                on_new_bit,
                edge_kind_mask,
                prefix,
            ),
        ),
    ]
}
