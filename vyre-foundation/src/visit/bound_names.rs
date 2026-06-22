//! Canonical collector of locally-bound variable names across a node tree.
//!
//! A *bound name* is a name a node introduces into local scope: a `Let`
//! binding or a `Loop` variable. Several scope-aware optimizer passes need
//! this set or its per-name counts — any transform that MOVES, FLATTENS, or
//! EXTENDS a binding's scope (`region_inline`, `tail_duplication`,
//! `read_only_load_hoist`) must reason about which names are bound where to
//! avoid producing duplicate / shadowing bindings that the block-scoped IR
//! validator rejects (V008 / V032).
//!
//! Traversal descends into `If`/`Loop`/`Block`/`Region` bodies. Names that
//! appear only inside expressions (e.g. `Expr::Var`) are *uses*, not bindings,
//! and are intentionally skipped.

use crate::ir::{Ident, Node};
use rustc_hash::{FxHashMap, FxHashSet};

/// Invoke `visit` once for every binding name introduced by `nodes`
/// (recursively): each `Let` name and each `Loop` variable.
pub(crate) fn for_each_bound_name(nodes: &[Node], visit: &mut impl FnMut(&Ident)) {
    for node in nodes {
        match node {
            Node::Let { name, .. } => visit(name),
            Node::Loop { var, body, .. } => {
                visit(var);
                for_each_bound_name(body, visit);
            }
            Node::If {
                then, otherwise, ..
            } => {
                for_each_bound_name(then, visit);
                for_each_bound_name(otherwise, visit);
            }
            Node::Block(body) => for_each_bound_name(body, visit),
            Node::Region { body, .. } => for_each_bound_name(body, visit),
            _ => {}
        }
    }
}

/// Insert every name bound in `nodes` into `out`.
pub(crate) fn collect_bound_names(nodes: &[Node], out: &mut FxHashSet<Ident>) {
    for_each_bound_name(nodes, &mut |name| {
        out.insert(name.clone());
    });
}

/// Tally how many times each name is bound in `nodes` (a name bound in both
/// arms of an `If` counts twice — once per arm — which is exactly what
/// scope-extension passes check against).
pub(crate) fn count_bound_names(nodes: &[Node], counts: &mut FxHashMap<Ident, usize>) {
    for_each_bound_name(nodes, &mut |name| {
        *counts.entry(name.clone()).or_insert(0) += 1;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Expr;

    #[test]
    fn counts_let_and_loop_var_bindings_recursively() {
        // let a; loop b { let c }; if cond { let a } else { let d }
        let nodes = vec![
            Node::let_bind("a", Expr::u32(0)),
            Node::loop_for("b", Expr::u32(0), Expr::u32(2), vec![Node::let_bind("c", Expr::u32(1))]),
            Node::if_then_else(
                Expr::bool(true),
                vec![Node::let_bind("a", Expr::u32(2))],
                vec![Node::let_bind("d", Expr::u32(3))],
            ),
        ];
        let mut counts = FxHashMap::default();
        count_bound_names(&nodes, &mut counts);
        assert_eq!(counts.get("a").copied(), Some(2), "`a` bound at top level and in the then-arm");
        assert_eq!(counts.get("b").copied(), Some(1), "loop variable counts as a binding");
        assert_eq!(counts.get("c").copied(), Some(1), "binding inside the loop body");
        assert_eq!(counts.get("d").copied(), Some(1));
        assert_eq!(counts.get("missing").copied(), None);

        let mut set = FxHashSet::default();
        collect_bound_names(&nodes, &mut set);
        assert!(["a", "b", "c", "d"].iter().all(|n| set.contains(&Ident::from(*n))));
        assert_eq!(set.len(), 4, "collect dedups the two `a` bindings");
    }
}
