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
use crate::visit::walk_nodes_mut;

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
