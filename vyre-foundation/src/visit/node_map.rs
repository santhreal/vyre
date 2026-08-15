//! Owning child-recursive `Node` map: takes each body slot of a node out,
//! hands it to the caller's transform, and puts the result back.
//!
//! The cleanup catalog (`empty_block_collapse`,
//! `region_promote_singleton_block`, `loop_trip_zero_eliminate`,
//! `if_constant_branch_eliminate`, `noop_assign_eliminate`,
//! `loop_redundant_bound_check_elide`) used to carry near-identical four-arm
//! `If`/`Loop`/`Block`/`Region` walkers; all of them now compose this helper
//! plus a per-pass rewrite predicate.
//!
//! ## Why an owning map (not a borrowed rebuild)
//!
//! `transform::rewrite_walk::rewrite_node` rebuilds a node from BORROWED
//! bodies and reports "unchanged" without allocating, which is what a pass
//! driver wants. It cannot MOVE a body out, so a transform that consumes its
//! input body (filtering a statement list, or recursing into each child by
//! value) would have to clone every body at every level to use it. This map
//! takes the slots through `transform::visit::child_bodies_mut` instead, so a
//! body is moved, not copied.
//!
//! ## Recursion contract
//!
//! `map_children(node, &mut f)` calls `f` once per immediate child of `node`
//! (not on `node` itself; the caller decides whether to apply its rewrite at
//! the current level). The node comes back with each child replaced by `f`'s
//! output. A node with no body slot is returned untouched and `f` is never
//! called.

use crate::ir::Node;
use crate::transform::visit::child_bodies_mut;

/// Recurse one level into `node`'s child sequences and apply `f` to each
/// immediate child node. Returns the rebuilt node.
///
/// The closure may itself call `map_children` to recurse further; the
/// helper does not do deep recursion on its own.
#[must_use]
pub fn map_children<F>(node: Node, f: &mut F) -> Node
where
    F: FnMut(Node) -> Node,
{
    map_body(node, &mut |body| body.into_iter().map(&mut *f).collect())
}

/// Rewrite every body sequence of `node` through `f`, then hand the node back.
///
/// `f` is called once per body slot the variant really has, in source order:
/// twice for `Node::If`, once for `Node::Loop`, `Node::Block`, and
/// `Node::Region`, and never for a variant that nests nothing.
///
/// Which slots those are is [`child_bodies_mut`]'s decision, the one exhaustive
/// owner in the move direction. This function used to state the list itself and
/// end it in `other => other`, so a body-bearing variant the list had not been
/// told about was handed back unchanged: every pass composed on it, including
/// `rematerialize_cheap_let` and the pass engine's constant propagation, was a
/// silent no-op inside that variant rather than an error, and no test could see
/// the difference because the pass reported success.
#[must_use]
pub fn map_body<F>(mut node: Node, f: &mut F) -> Node
where
    F: FnMut(Vec<Node>) -> Vec<Node>,
{
    for slot in child_bodies_mut(&mut node) {
        *slot = f(std::mem::take(slot));
    }
    node
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::model::expr::Ident;
    use crate::ir::Expr;
    use std::sync::Arc;

    fn store_at(idx: u32, value: u32) -> Node {
        Node::store("buf", Expr::u32(idx), Expr::u32(value))
    }

    #[test]
    fn map_children_recurses_into_if_branches() {
        // `if true { store(0,1) } else { store(0,2) }` → callback applied
        // to both inner stores; callback rewrites Store → Block(empty)
        // for proof.
        let input =
            Node::if_then_else(Expr::bool(true), vec![store_at(0, 1)], vec![store_at(0, 2)]);
        let mut count = 0;
        let mapped = map_children(input, &mut |n| {
            count += 1;
            match n {
                Node::Store { .. } => Node::Block(Vec::new()),
                other => other,
            }
        });
        assert_eq!(count, 2, "callback must fire once per branch's store");
        match mapped {
            Node::If {
                then, otherwise, ..
            } => {
                assert!(matches!(then[0], Node::Block(_)));
                assert!(matches!(otherwise[0], Node::Block(_)));
            }
            other => panic!("expected Node::If; got {other:?}"),
        }
    }

    #[test]
    fn map_children_recurses_into_loop_body() {
        let input = Node::Loop {
            var: Ident::from("i"),
            from: Expr::u32(0),
            to: Expr::u32(4),
            body: vec![store_at(0, 7), store_at(1, 8)],
        };
        let mut count = 0;
        let _mapped = map_children(input, &mut |n| {
            count += 1;
            n
        });
        assert_eq!(count, 2);
    }

    #[test]
    fn map_children_recurses_into_block() {
        let input = Node::Block(vec![store_at(0, 1), store_at(1, 2), store_at(2, 3)]);
        let mut count = 0;
        let _mapped = map_children(input, &mut |n| {
            count += 1;
            n
        });
        assert_eq!(count, 3);
    }

    #[test]
    fn map_children_recurses_into_region_body() {
        let input = Node::Region {
            generator: Ident::from("test_op"),
            source_region: None,
            body: Arc::new(vec![store_at(0, 1)]),
        };
        let mut count = 0;
        let mapped = map_children(input, &mut |n| {
            count += 1;
            n
        });
        assert_eq!(count, 1);
        assert!(matches!(mapped, Node::Region { .. }));
    }

    #[test]
    fn map_children_preserves_op_id_through_region_shared_body_clone_path() {
        // Two strong refs to the same Arc<Vec<Node>> body force the
        // `Arc::make_mut` clone branch inside `child_bodies_mut`.
        let body = Arc::new(vec![store_at(0, 1)]);
        let _keepalive = Arc::clone(&body);
        let input = Node::Region {
            generator: Ident::from("test_op_with_clone"),
            source_region: None,
            body,
        };
        let mapped = map_children(input, &mut |n| n);
        match mapped {
            Node::Region { generator, .. } => {
                assert_eq!(generator.as_str(), "test_op_with_clone");
            }
            other => panic!("expected Region; got {other:?}"),
        }
    }

    #[test]
    fn map_children_preserves_loop_metadata() {
        let input = Node::Loop {
            var: Ident::from("ix"),
            from: Expr::u32(2),
            to: Expr::u32(9),
            body: Vec::new(),
        };
        let mapped = map_children(input, &mut |n| n);
        match mapped {
            Node::Loop { var, from, to, .. } => {
                assert_eq!(var.as_str(), "ix");
                assert!(matches!(from, Expr::LitU32(2)));
                assert!(matches!(to, Expr::LitU32(9)));
            }
            other => panic!("expected Loop; got {other:?}"),
        }
    }

    #[test]
    fn map_children_returns_non_container_nodes_unchanged() {
        let input = store_at(0, 7);
        let mut fired = false;
        let mapped = map_children(input, &mut |_n| {
            fired = true;
            unreachable!("non-container nodes must not invoke the callback")
        });
        assert!(!fired, "no children = no callback invocations");
        assert!(matches!(mapped, Node::Store { .. }));
    }
}
