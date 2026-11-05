//! Canonical collector of locally-bound variable names across a node tree.
//!
//! A *bound name* is a name a node introduces into local scope: a `Let`
//! binding or a `Loop` variable. Several scope-aware optimizer passes need
//! this set or its per-name counts, any transform that MOVES, FLATTENS, or
//! EXTENDS a binding's scope (`region_inline`, `tail_duplication`,
//! `read_only_load_hoist`) must reason about which names are bound where to
//! avoid producing duplicate / shadowing bindings that the block-scoped IR
//! validator rejects (V008 / V032).
//!
//! Traversal descends through `child_bodies`, the exhaustive owner of which
//! variants nest statements, and the per-node answer comes from `node_scalars`,
//! the exhaustive owner of the scalar namespace, so a new nesting variant
//! cannot hide a binding and a new binding form cannot be classified as "binds
//! nothing" by a catch-all arm.
//! Names that appear only inside expressions (e.g. `Expr::Var`) are *uses*, not bindings,
//! and are intentionally skipped.

use crate::ir::{Ident, Node};
use crate::visit::{child_bodies, node_scalars, NameBinding};
use rustc_hash::{FxHashMap, FxHashSet};

/// Invoke `visit` once for every binding name introduced by `nodes`
/// (recursively): each `Let` name and each `Loop` variable.
///
/// A `Node::Assign` writes a name the enclosing scope already declares, so it
/// reports [`NameBinding::Reassign`] and is not a binding here. Counting it as
/// one would show a scope-extension pass a duplicate declaration and make it
/// refuse a legal rewrite.
pub(crate) fn for_each_bound_name(nodes: &[Node], visit: &mut impl FnMut(&Ident)) {
    for node in nodes {
        match node_scalars(node).binding {
            Some((NameBinding::Declare | NameBinding::Induction, name)) => visit(name),
            Some((NameBinding::Reassign, _)) | None => {}
        }
        for body in child_bodies(node) {
            for_each_bound_name(body, visit);
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
/// arms of an `If` counts twice, once per arm, which is exactly what
/// scope-extension passes check against).
pub(crate) fn count_bound_names(nodes: &[Node], counts: &mut FxHashMap<Ident, usize>) {
    for_each_bound_name(nodes, &mut |name| {
        *counts.entry(name.clone()).or_insert(0) += 1;
    });
}
