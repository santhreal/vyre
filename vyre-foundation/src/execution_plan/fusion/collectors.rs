//! Buffer-target collectors for load/store/atomic walks.
//!
//! `VYRE_IR_HOTSPOTS` HIGH: `fuse_programs_multi` previously called three
//! independent walks (`collect_atomic_targets_from_node`,
//! `collect_load_targets_from_node`, `collect_store_targets_from_node`)
//! per arm  -  three full traversals of the same IR tree. The unified
//! [`collect_buffer_targets`] helper does it in one walk with three
//! mutable target sets. This is the canonical collector API for fusion:
//! adding single-target wrappers would reintroduce duplicate IR walks and
//! make the fusion boundary ambiguous for contributors.

use rustc_hash::FxHashSet;

use crate::ir::{Expr, Ident, Node};

/// One-pass collector: walk `node` once and fan out load / store /
/// atomic buffer targets into the three caller-supplied sets.
pub(super) fn collect_buffer_targets(
    node: &Node,
    loads: &mut FxHashSet<Ident>,
    stores: &mut FxHashSet<Ident>,
    atomics: &mut FxHashSet<Ident>,
) {
    match node {
        Node::Let { value, .. } | Node::Assign { value, .. } => {
            collect_buffer_targets_from_expr(value, loads, atomics);
        }
        Node::Store {
            buffer,
            index,
            value,
        } => {
            stores.insert(Ident::from(buffer));
            collect_buffer_targets_from_expr(index, loads, atomics);
            collect_buffer_targets_from_expr(value, loads, atomics);
        }
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            collect_buffer_targets_from_expr(cond, loads, atomics);
            for n in then.iter().chain(otherwise.iter()) {
                collect_buffer_targets(n, loads, stores, atomics);
            }
        }
        Node::Loop { from, to, body, .. } => {
            collect_buffer_targets_from_expr(from, loads, atomics);
            collect_buffer_targets_from_expr(to, loads, atomics);
            for n in body {
                collect_buffer_targets(n, loads, stores, atomics);
            }
        }
        Node::Block(body) => {
            for n in body {
                collect_buffer_targets(n, loads, stores, atomics);
            }
        }
        Node::Region { body, .. } => {
            for n in body.iter() {
                collect_buffer_targets(n, loads, stores, atomics);
            }
        }
        Node::AllReduce { buffer, .. } | Node::Broadcast { buffer, .. } => {
            loads.insert(buffer.clone());
            stores.insert(buffer.clone());
        }
        Node::AllGather { input, output, .. } | Node::ReduceScatter { input, output, .. } => {
            loads.insert(input.clone());
            stores.insert(output.clone());
        }
        // Async copies are genuine buffer accesses: `source` is read and
        // `destination` is written (vyre-reference `eval_async_load` /
        // `eval_async_store` both read_bytes(source) then write destination).
        // Dropping them hid a cross-arm RAW/WAR hazard from the barrier-
        // insertion pass, letting a later arm read a buffer an earlier arm
        // async-wrote before the write was made visible, the stale-read
        // miscompile this collector exists to prevent. offset/size may Load.
        Node::AsyncLoad {
            source,
            destination,
            offset,
            size,
            ..
        }
        | Node::AsyncStore {
            source,
            destination,
            offset,
            size,
            ..
        } => {
            loads.insert(source.clone());
            stores.insert(destination.clone());
            collect_buffer_targets_from_expr(offset, loads, atomics);
            collect_buffer_targets_from_expr(size, loads, atomics);
        }
        // IndirectDispatch reads `count_buffer` to derive the launch geometry;
        // an earlier arm writing that buffer is a RAW hazard for the dispatch.
        Node::IndirectDispatch { count_buffer, .. } => {
            loads.insert(count_buffer.clone());
        }
        // A trap address expression may Load from a buffer.
        Node::Trap { address, .. } => {
            collect_buffer_targets_from_expr(address, loads, atomics);
        }
        Node::TileLoad { buffer, origin, .. } => {
            loads.insert(buffer.clone());
            for expr in origin {
                collect_buffer_targets_from_expr(expr, loads, atomics);
            }
        }
        Node::TileStore { buffer, origin, .. } => {
            stores.insert(buffer.clone());
            for expr in origin {
                collect_buffer_targets_from_expr(expr, loads, atomics);
            }
        }
        Node::TileElementwise { body, .. } => {
            for n in body {
                collect_buffer_targets(n, loads, stores, atomics);
            }
        }
        Node::TileMatmul { .. } | Node::TileReduce { .. } | Node::TileDecl { .. } => {}
        Node::Return
        | Node::Barrier { .. }
        | Node::AsyncWait { .. }
        | Node::Resume { .. }
        | Node::Opaque(_) => {}
    }
}

fn collect_buffer_targets_from_expr(
    expr: &Expr,
    loads: &mut FxHashSet<Ident>,
    atomics: &mut FxHashSet<Ident>,
) {
    crate::visit::visit_expr_buffer_accesses(expr, |access, buffer| match access {
        crate::visit::ExprBufferAccess::Load => {
            loads.insert(buffer.clone());
        }
        crate::visit::ExprBufferAccess::Atomic => {
            atomics.insert(buffer.clone());
        }
    });
}
