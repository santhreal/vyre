//! Support passes for the scheduler tests.
//!
//! Every walk here goes through `visit`: the read-only scans through
//! [`for_each_node`] and the in-place rewrites through [`walk_nodes_mut`], both
//! of which take their nesting from `child_bodies`. Each of the six helpers used
//! to carry its own `match node` naming `If`, `Loop`, `Block` and `Region` and
//! ending in a wildcard, and the rewrite side is the half that matters: a
//! rewrite that classifies a nesting variant as a leaf descends into nothing and
//! reports `changed: false`, so the scheduler test it feeds asserts on an
//! unrewritten program and passes.
//!
//! The rewrites take `&mut Program` rather than `&mut [Node]` because
//! `walk_nodes_mut` is the owner of mutable descent and `Program::entry_mut`
//! invalidates the cached stats the passes downstream read.

use super::*;
use crate::visit::{for_each_node, walk_nodes_mut};

pub(super) fn rewrite_first_store_value(program: &mut Program) -> bool {
    let mut done = false;
    walk_nodes_mut(program, |node| {
        if done {
            return;
        }
        if let Node::Store { value, .. } = node {
            *value = Expr::u32(43);
            done = true;
        }
    });
    done
}

pub(super) fn rewrite_matching_stores(
    program: Program,
    batch: Option<&RewriteBatch>,
) -> PassResult {
    let mut rewritten = Clone::clone(&program);
    let changed = match batch {
        Some(batch) => {
            let selected = batch
                .items()
                .iter()
                .map(|item| item.col as usize)
                .collect::<Vec<_>>();
            rewrite_selected_store_ordinals(&mut rewritten, &selected)
        }
        None => rewrite_all_matching_stores(&mut rewritten),
    };
    if changed {
        PassResult {
            program: rewritten,
            changed: true,
        }
    } else {
        PassResult::unchanged(program)
    }
}

pub(super) fn rewrite_store_value_if_matches(node: &mut Node, old: u32, new: u32) -> bool {
    match node {
        Node::Store { value, .. } if *value == Expr::u32(old) => {
            *value = Expr::u32(new);
            true
        }
        _ => false,
    }
}

pub(super) fn rewrite_store_values(program: &mut Program, old: u32, new: u32) -> bool {
    let mut changed = false;
    walk_nodes_mut(program, |node| {
        changed |= rewrite_store_value_if_matches(node, old, new);
    });
    changed
}

pub(super) fn store_value_is(node: &Node, expected: u32) -> bool {
    matches!(node, Node::Store { value, .. } if *value == Expr::u32(expected))
}

pub(super) fn all_stores_have_value(nodes: &[Node], expected: u32) -> bool {
    let mut all = true;
    for_each_node(nodes, |node| {
        if matches!(node, Node::Store { .. }) && !store_value_is(node, expected) {
            all = false;
        }
    });
    all
}

pub(super) fn collect_store_candidates(nodes: &[Node], candidates: &mut Vec<RewriteCandidate>) {
    for_each_node(nodes, |node| {
        if matches!(node, Node::Store { value, .. } if *value == Expr::u32(42)) {
            candidates.push(RewriteCandidate::new(0, candidates.len() as u32));
        }
    });
}

pub(super) fn rewrite_all_matching_stores(program: &mut Program) -> bool {
    let mut changed = false;
    walk_nodes_mut(program, |node| {
        changed |= rewrite_store_value_if_matches(node, 42, 43);
    });
    changed
}

/// Rewrite only the stores whose document-order ordinal is in `selected`.
///
/// The ordinal advances on every `Store`, matched or not, which is the same
/// numbering [`collect_store_candidates`] hands the batch planner. Both walks
/// now read their order from `visit`, so the two cannot disagree
/// about which store a column refers to.
pub(super) fn rewrite_selected_store_ordinals(program: &mut Program, selected: &[usize]) -> bool {
    let mut changed = false;
    let mut ordinal = 0usize;
    walk_nodes_mut(program, |node| {
        if !matches!(node, Node::Store { .. }) {
            return;
        }
        let current = ordinal;
        ordinal += 1;
        if selected.contains(&current) {
            changed |= rewrite_store_value_if_matches(node, 42, 43);
        }
    });
    changed
}

pub(super) fn repeated_store_program(count: usize) -> Program {
    Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(count as u32)],
        [1, 1, 1],
        (0..count)
            .map(|index| Node::store("out", Expr::u32(index as u32), Expr::u32(42)))
            .collect::<Vec<_>>(),
    )
}
