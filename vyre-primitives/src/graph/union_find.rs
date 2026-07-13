//! Lock-free union-find (disjoint-set) alias tracking as Vyre IR.
//!
//! This module deliberately emits `Program` / `Node` IR, not target shader
//! text. Concrete drivers own target spelling; primitives own the backend-
//! neutral algorithm.

use std::sync::Arc;

use vyre_foundation::ir::model::expr::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Canonical operation id for one union-find merge pass.
pub const OP_ID: &str = "vyre-primitives::graph::union_find";
/// One lane per union edge in a batch.
pub const UNION_FIND_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

/// Dispatch grid that covers every union edge lane.
#[must_use]
pub const fn union_find_dispatch_grid(edge_count: u32) -> [u32; 3] {
    let lanes_per_block = UNION_FIND_WORKGROUP_SIZE[0];
    let full_blocks = edge_count / lanes_per_block;
    let tail_block = if edge_count % lanes_per_block == 0 {
        0
    } else {
        1
    };
    let blocks = full_blocks + tail_block;
    [if blocks == 0 { 1 } else { blocks }, 1, 1]
}

/// Build the path-halving body used by [`union_roots_body`].
///
/// `id_var` is read at entry. On exit `root_var` contains the discovered root
/// and `scratch_parent_var` contains the last parent read. The loop is bounded
/// by `node_count` so malformed parent arrays cannot create an infinite kernel.
#[must_use]
pub fn find_root_body(
    parent: &str,
    id_var: &str,
    root_var: &str,
    scratch_parent_var: &str,
    node_count: u32,
) -> Vec<Node> {
    vec![
        // Loop invariant: `root` is the current node and `scratch` is `parent[root]`, so the
        // guard `root != scratch` == "root is not yet its own parent (not a root)". `scratch`
        // MUST be seeded with `parent[id]`, NOT `id`: seeding it with `id` makes the guard
        // false on iteration 0 (root == scratch == id) and, since nothing else mutates them,
        // the loop is a permanent no-op that returns `id` unwalked. That silently made every
        // multi-hop union operate on raw endpoints instead of roots (no connectivity closure);
        // the 1-hop registration fixture could not catch it because there the endpoint IS the
        // root and the fixture only checks the CAS-written parent array, never `find()`.
        Node::let_bind(root_var, Expr::var(id_var)),
        Node::let_bind(
            scratch_parent_var,
            Expr::atomic_or(parent, Expr::var(id_var), Expr::u32(0)),
        ),
        Node::loop_for(
            "uf_find_iter",
            Expr::u32(0),
            Expr::u32(node_count.max(1)),
            vec![Node::if_then(
                Expr::ne(Expr::var(root_var), Expr::var(scratch_parent_var)),
                vec![
                    Node::assign(root_var, Expr::var(scratch_parent_var)),
                    Node::if_then(
                        Expr::ge(Expr::var(root_var), Expr::u32(node_count)),
                        vec![Node::trap(Expr::var(root_var), "union-find-parent-oob")],
                    ),
                    Node::assign(
                        scratch_parent_var,
                        Expr::atomic_or(parent, Expr::var(root_var), Expr::u32(0)),
                    ),
                    // Bind uf_grandparent and the atomic_min that consumes
                    // it in the SAME if_then so the binding scope covers
                    // the use. Splitting them into two sibling if_then
                    // blocks ends uf_grandparent's binding lifetime
                    // before atomic_min needs it (CUDA backend reports
                    // "uf_grandparent referenced before binding").
                    Node::if_then(
                        Expr::lt(Expr::var(scratch_parent_var), Expr::u32(node_count)),
                        vec![
                            Node::let_bind(
                                "uf_grandparent",
                                Expr::atomic_or(
                                    parent,
                                    Expr::var(scratch_parent_var),
                                    Expr::u32(0),
                                ),
                            ),
                            Node::let_bind(
                                "uf_path_old",
                                Expr::atomic_min(
                                    parent,
                                    Expr::var(root_var),
                                    Expr::var("uf_grandparent"),
                                ),
                            ),
                        ],
                    ),
                ],
            )],
        ),
    ]
}

/// Build one deterministic lock-free union pass for edge `edge_index_var`.
///
/// `edge_a[edge_index]` and `edge_b[edge_index]` are merged into the shared
/// `parent` array using ordered root selection (the lower-index root always
/// wins) and compare-exchange.
///
/// The retry loop is the canonical lock-free union: every iteration RE-FINDS
/// both roots from the *original* endpoints, then points the higher-index root
/// at the lower via a single `CAS(parent[high], high, low)`. Re-finding both
/// each pass, rather than caching one root and patching it after a lost CAS 
/// is what makes it converge: a lost CAS (another lane moved `parent[high]`)
/// simply retries against freshly observed roots, and because ordered selection
/// only ever lowers a root, the pair reaches its shared minimum within
/// `node_count` iterations. Once the roots coincide the `ne` guard turns every
/// remaining iteration into a no-op, so running the full bound is harmless.
///
/// The previous formulation cached `uf_root_a`/`uf_root_b` before the loop and,
/// on a *successful* CAS, updated only `uf_root_b`. When `uf_root_a` was the
/// higher root that left the loop condition permanently true (spinning to the
/// bound) and, worse, dropped merges under the interpreter's lane ordering, the
/// `union_find_program` connectivity defect. All working vars here are bound
/// INSIDE the loop body, shadowing nothing in the enclosing scope (V008-clean;
/// two sequential `find_root_body` calls are already proven shadow-free).
#[must_use]
pub fn union_roots_body(
    parent: &str,
    edge_a: &str,
    edge_b: &str,
    edge_index_var: &str,
    node_count: u32,
) -> Vec<Node> {
    let mut body = vec![
        Node::let_bind("uf_a", Expr::load(edge_a, Expr::var(edge_index_var))),
        Node::let_bind("uf_b", Expr::load(edge_b, Expr::var(edge_index_var))),
        Node::if_then(
            Expr::or(
                Expr::ge(Expr::var("uf_a"), Expr::u32(node_count)),
                Expr::ge(Expr::var("uf_b"), Expr::u32(node_count)),
            ),
            vec![Node::trap(Expr::var(edge_index_var), "union-find-edge-oob")],
        ),
    ];
    body.push(Node::loop_for(
        "uf_union_iter",
        Expr::u32(0),
        Expr::u32(node_count.max(1)),
        {
            // Re-find BOTH roots from the immutable endpoints every iteration. These
            // let-binds live only inside the loop body (no enclosing binding of the same
            // name), so re-binding them per iteration is not a shadow, the same way
            // `find_root_body`'s own inner-loop `uf_grandparent`/`uf_path_old` re-bind.
            let mut iter_body =
                find_root_body(parent, "uf_a", "uf_root_a", "uf_parent_a", node_count);
            iter_body.extend(find_root_body(
                parent,
                "uf_b",
                "uf_root_b",
                "uf_parent_b",
                node_count,
            ));
            iter_body.push(Node::if_then(
                Expr::ne(Expr::var("uf_root_a"), Expr::var("uf_root_b")),
                vec![
                    Node::let_bind(
                        "uf_low",
                        Expr::select(
                            Expr::lt(Expr::var("uf_root_a"), Expr::var("uf_root_b")),
                            Expr::var("uf_root_a"),
                            Expr::var("uf_root_b"),
                        ),
                    ),
                    Node::let_bind(
                        "uf_high",
                        Expr::select(
                            Expr::lt(Expr::var("uf_root_a"), Expr::var("uf_root_b")),
                            Expr::var("uf_root_b"),
                            Expr::var("uf_root_a"),
                        ),
                    ),
                    // Point the higher-index root at the lower. The result is bound but
                    // intentionally unread (like `find_root_body`'s `uf_path_old`): on
                    // success `parent[high]=low`; on a lost CAS the next iteration re-finds
                    // fresh roots and retries. Binding it keeps the atomic as a statement.
                    Node::let_bind(
                        "uf_observed",
                        Expr::atomic_compare_exchange(
                            parent,
                            Expr::var("uf_high"),
                            Expr::var("uf_high"),
                            Expr::var("uf_low"),
                        ),
                    ),
                ],
            ));
            iter_body
        },
    ));
    body
}

/// Build a Program that applies a batch of union operations.
#[must_use]
pub fn union_find_program(
    parent: &str,
    edge_a: &str,
    edge_b: &str,
    node_count: u32,
    edge_count: u32,
) -> Program {
    let lane = Expr::gid_x();
    let body = vec![Node::if_then(
        Expr::lt(lane.clone(), Expr::u32(edge_count)),
        union_roots_body(parent, edge_a, edge_b, "uf_edge", node_count),
    )];
    Program::wrapped(
        vec![
            BufferDecl::storage(parent, 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(node_count.max(1)),
            BufferDecl::storage(edge_a, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(edge_count.max(1)),
            BufferDecl::storage(edge_b, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(edge_count.max(1)),
        ],
        UNION_FIND_WORKGROUP_SIZE,
        vec![Node::Region {
            generator: Ident::from(OP_ID),
            source_region: None,
            body: Arc::new({
                let mut entry = vec![Node::let_bind("uf_edge", lane)];
                entry.extend(body);
                entry
            }),
        }],
    )
}

/// Validated dispatch layout for the union-find primitive.
///
/// The primitive owns these derived counts so dispatch wrappers do not fork
/// parent output sizing or padded edge-buffer policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnionFindLayout {
    /// Number of parent nodes accepted by the primitive.
    pub node_count: u32,
    /// Number of union edges accepted by the primitive.
    pub edge_count: u32,
    /// Number of parent words expected in the backend output.
    pub node_words: usize,
    /// Number of edge words to upload for each edge endpoint buffer.
    pub edge_storage_words: usize,
}

/// Validate the parent/edge arrays consumed by the union-find primitive.
///
/// Returns the full primitive-compatible dispatch layout so dispatch wrappers
/// can build the IR program without duplicating boundary checks or padding
/// rules.
///
/// # Errors
///
/// Returns an actionable diagnostic when edge arrays differ in length, counts
/// exceed the primitive's u32 index space, parent links are malformed, or edge
/// endpoints reference nodes outside the parent set.
pub fn validate_union_find_inputs(
    parent_init: &[u32],
    edge_a: &[u32],
    edge_b: &[u32],
) -> Result<UnionFindLayout, String> {
    if edge_a.len() != edge_b.len() {
        return Err(format!(
            "Fix: union_find requires edge_a.len() == edge_b.len(), got {} vs {}.",
            edge_a.len(),
            edge_b.len()
        ));
    }
    let node_count = u32::try_from(parent_init.len()).map_err(|_| {
        format!(
            "Fix: union_find parent length {} exceeds u32 index space.",
            parent_init.len()
        )
    })?;
    let edge_count = u32::try_from(edge_a.len()).map_err(|_| {
        format!(
            "Fix: union_find edge count {} exceeds u32 index space.",
            edge_a.len()
        )
    })?;
    if node_count == 0 {
        if edge_count == 0 {
            return Ok(UnionFindLayout {
                node_count: 0,
                edge_count: 0,
                node_words: 0,
                edge_storage_words: 1,
            });
        }
        return Err("Fix: union_find cannot union edges against an empty parent set.".to_string());
    }
    for (idx, &parent) in parent_init.iter().enumerate() {
        if parent >= node_count {
            return Err(format!(
                "Fix: union_find parent_init[{idx}]={parent} is outside node_count {node_count}."
            ));
        }
    }
    for (idx, (&a, &b)) in edge_a.iter().zip(edge_b.iter()).enumerate() {
        if a >= node_count || b >= node_count {
            return Err(format!(
                "Fix: union_find edge {idx} endpoint ({a}, {b}) is outside node_count {node_count}."
            ));
        }
    }
    Ok(UnionFindLayout {
        node_count,
        edge_count,
        node_words: parent_init.len(),
        edge_storage_words: edge_a.len().max(1),
    })
}

#[cfg(feature = "inventory-registry")]
inventory::submit! {
    crate::harness::OpEntry::new(
        OP_ID,
        || union_find_program("parent", "edge_a", "edge_b", 4, 2),
        Some(|| {
            // 4 singleton nodes seeded with the identity parent [0,1,2,3]; two DISJOINT
            // union edges 0–1 and 2–3. Ordered root selection keeps the smaller root, so
            // parent[1]→0 and parent[3]→2. The two edges touch disjoint parent slots (1 and
            // 3), so the pass is race-clean under lane reversal while still exercising the
            // full find-root path walk + the compare-exchange union scatter.
            let to_bytes = |w: &[u32]| crate::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[0, 1, 2, 3]), // parent seed (identity: each node its own root)
                to_bytes(&[0, 2]),       // edge_a
                to_bytes(&[1, 3]),       // edge_b
            ]]
        }),
        Some(|| {
            let to_bytes = |w: &[u32]| crate::wire::pack_u32_slice(w);
            // {0,1} merges under root 0, {2,3} under root 2.
            vec![vec![to_bytes(&[0, 0, 2, 2])]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_find_program_uses_atomic_ir_not_target_text() {
        let program = union_find_program("parent", "edge_a", "edge_b", 8, 4);
        let dump = format!("{program:#?}");
        assert!(dump.contains("CompareExchange"));
        assert!(dump.contains("Min"));
        assert!(!dump.contains("atomicCAS"));
        assert!(!dump.contains("ptr<storage"));
    }

    #[test]
    fn union_find_program_declares_batch_buffers() {
        let program = union_find_program("parent", "edge_a", "edge_b", 8, 4);
        assert_eq!(program.buffers().len(), 3);
        assert_eq!(program.workgroup_size(), UNION_FIND_WORKGROUP_SIZE);
    }

    #[test]
    fn dispatch_grid_packs_union_edges_into_workgroups() {
        assert_eq!(union_find_dispatch_grid(0), [1, 1, 1]);
        assert_eq!(union_find_dispatch_grid(1), [1, 1, 1]);
        assert_eq!(union_find_dispatch_grid(256), [1, 1, 1]);
        assert_eq!(union_find_dispatch_grid(257), [2, 1, 1]);
        assert_eq!(union_find_dispatch_grid(1025), [5, 1, 1]);
    }

    #[test]
    fn validate_union_find_inputs_accepts_empty_and_canonical_inputs() {
        assert_eq!(
            validate_union_find_inputs(&[], &[], &[]).unwrap(),
            UnionFindLayout {
                node_count: 0,
                edge_count: 0,
                node_words: 0,
                edge_storage_words: 1,
            }
        );
        assert_eq!(
            validate_union_find_inputs(&[0, 1, 2, 3], &[0, 2], &[1, 3]).unwrap(),
            UnionFindLayout {
                node_count: 4,
                edge_count: 2,
                node_words: 4,
                edge_storage_words: 2,
            }
        );
    }

    #[test]
    fn validate_union_find_inputs_rejects_malformed_inputs() {
        let err = validate_union_find_inputs(&[0, 1], &[0], &[1, 0]).unwrap_err();
        assert!(err.contains("edge_a.len() == edge_b.len()"));

        let err = validate_union_find_inputs(&[], &[0], &[0]).unwrap_err();
        assert!(err.contains("empty parent set"));

        let err = validate_union_find_inputs(&[0, 9], &[0], &[1]).unwrap_err();
        assert!(err.contains("parent_init[1]=9"));

        let err = validate_union_find_inputs(&[0, 1], &[0], &[2]).unwrap_err();
        assert!(err.contains("outside node_count"));
    }
}
