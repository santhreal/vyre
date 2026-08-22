//! Hoisting of `Let` bindings a split segment references but does not define,
//! so every segment is a self-contained program.

use std::collections::{BTreeMap, BTreeSet};

use crate::ir::{Expr, Node};
use crate::visit::{for_each_expr, for_each_node};

/// Every `Let` in `nodes` and in every nested body, keyed by bound name.
///
/// Descent comes from `visit::for_each_node`, the one owner of which
/// node variants nest. The hand-written match this replaces ended in `_ => {}`,
/// so a binding inside a fifth body-bearing variant would not have been
/// available to hoist and the segment that reads it would not have been
/// self-contained.
fn collect_global_let_bindings(nodes: &[Node], map: &mut BTreeMap<String, Node>) {
    for_each_node(nodes, |node| {
        if let Node::Let { name, .. } = node {
            map.insert(name.as_str().to_string(), node.clone());
        }
    });
}

/// Every name a statement in `nodes` binds: `Let` names and `Loop` induction
/// variables, at any nesting depth.
fn collect_locally_defined_vars(nodes: &[Node], vars: &mut BTreeSet<String>) {
    for_each_node(nodes, |node| {
        if let Node::Let { name, .. } = node {
            vars.insert(name.as_str().to_string());
        } else if let Node::Loop { var, .. } = node {
            vars.insert(var.as_str().to_string());
        }
    });
}

/// Every variable `nodes` reads, at any nesting depth and in any operand
/// position.
///
/// Node descent, operand positions and expression children all come from
/// `visit::for_each_expr`. This replaces three hand-written matches,
/// each ending in a wildcard: one over node bodies, one over the operand
/// positions of a statement, and one over the operands of an expression. A free
/// variable missed by any of the three is a segment emitted without the binding
/// it reads.
fn collect_referenced_vars(nodes: &[Node], vars: &mut BTreeSet<String>) {
    for_each_expr(nodes, |expr| {
        if let Expr::Var(name) = expr {
            vars.insert(name.as_str().to_string());
        }
    });
}

fn resolve_dependencies(
    name: &str,
    global_lets: &BTreeMap<String, Node>,
    resolved_names: &mut BTreeSet<String>,
    resolved_lets: &mut Vec<Node>,
) {
    if resolved_names.contains(name) {
        return;
    }
    if let Some(let_node) = global_lets.get(name) {
        resolved_names.insert(name.to_string());
        let mut deps = BTreeSet::new();
        collect_referenced_vars(std::slice::from_ref(let_node), &mut deps);
        for dep in &deps {
            resolve_dependencies(dep, global_lets, resolved_names, resolved_lets);
        }
        resolved_lets.push(let_node.clone());
    }
}

/// Prepend to each segment the `Let` bindings it reads but does not define.
///
/// The collections are ordered, not hashed. The hoisted prefix becomes part of
/// the segment's entry sequence, so its order is part of the program's canonical
/// wire bytes and therefore part of the compiled artifact digest. Iterating a
/// `HashSet` of free variable names made that order depend on the hash seed,
/// which is a nondeterministic artifact for one unchanged graph.
pub(super) fn propagate_let_bindings(segments: &mut [Vec<Node>], hoisted_inner: &[Node]) {
    let mut global_lets = BTreeMap::new();
    collect_global_let_bindings(hoisted_inner, &mut global_lets);

    for segment_nodes in segments {
        let mut locally_defined = BTreeSet::new();
        collect_locally_defined_vars(segment_nodes, &mut locally_defined);

        let mut referenced = BTreeSet::new();
        collect_referenced_vars(segment_nodes, &mut referenced);

        let mut resolved_lets = Vec::new();
        let mut resolved_names = BTreeSet::new();
        for name in referenced.difference(&locally_defined) {
            resolve_dependencies(name, &global_lets, &mut resolved_names, &mut resolved_lets);
        }

        if !resolved_lets.is_empty() {
            resolved_lets.extend(std::mem::take(segment_nodes));
            *segment_nodes = resolved_lets;
        }
    }
}
