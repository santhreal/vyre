//! Divergence + launch-geometry analysis used by the fusion safety check.

use rustc_hash::FxHashSet;

use crate::ir::{Expr, Ident, Node};
use crate::optimizer::rewrite::expr_contains_atomic;

pub(super) fn has_divergent_invocation_gated_store(
    node: &Node,
    inside_invocation_gate: bool,
) -> bool {
    match node {
        Node::Store { .. } => inside_invocation_gate,
        Node::Let { value, .. } | Node::Assign { value, .. } => {
            inside_invocation_gate && expr_contains_atomic(value)
        }
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            let new_gate = inside_invocation_gate
                || expr_depends_on_launch_geometry(cond, &FxHashSet::default());
            then.iter()
                .chain(otherwise.iter())
                .any(|n| has_divergent_invocation_gated_store(n, new_gate))
        }
        Node::Loop { body, .. } => body
            .iter()
            .any(|n| has_divergent_invocation_gated_store(n, inside_invocation_gate)),
        Node::Block(body) => body
            .iter()
            .any(|n| has_divergent_invocation_gated_store(n, inside_invocation_gate)),
        Node::Region { body, .. } => body
            .iter()
            .any(|n| has_divergent_invocation_gated_store(n, inside_invocation_gate)),
        Node::IndirectDispatch { .. }
        | Node::AllReduce { .. }
        | Node::AllGather { .. }
        | Node::ReduceScatter { .. }
        | Node::Broadcast { .. }
        | Node::Return
        | Node::Barrier { .. }
        | Node::LogicalBarrier { .. }
        | Node::AsyncLoad { .. }
        | Node::AsyncStore { .. }
        | Node::AsyncWait { .. }
        | Node::Trap { .. }
        | Node::Resume { .. }
        | Node::TileLoad { .. }
        | Node::TileStore { .. }
        | Node::TileMatmul { .. }
        | Node::TileReduce { .. }
        | Node::TileDecl { .. } => false,
        Node::Opaque(ext) => ext.is_divergent(),
        Node::TileElementwise { body, .. } => body
            .iter()
            .any(|n| has_divergent_invocation_gated_store(n, inside_invocation_gate)),
    }
}

pub(super) fn has_launch_geometry_dependent_write(nodes: &[Node]) -> bool {
    let mut launch_vars = FxHashSet::default();
    nodes_have_launch_geometry_dependent_write(nodes, &mut launch_vars, false)
}

fn nodes_have_launch_geometry_dependent_write(
    nodes: &[Node],
    launch_vars: &mut FxHashSet<Ident>,
    inside_launch_gate: bool,
) -> bool {
    nodes
        .iter()
        .any(|node| node_has_launch_geometry_dependent_write(node, launch_vars, inside_launch_gate))
}

fn node_has_launch_geometry_dependent_write(
    node: &Node,
    launch_vars: &mut FxHashSet<Ident>,
    inside_launch_gate: bool,
) -> bool {
    match node {
        Node::Let { name, value } | Node::Assign { name, value } => {
            let writes_atomic = expr_contains_atomic(value);
            let depends_on_launch = expr_depends_on_launch_geometry(value, launch_vars);
            if depends_on_launch {
                launch_vars.insert(name.clone());
            } else {
                launch_vars.remove(name);
            }
            writes_atomic && (inside_launch_gate || depends_on_launch)
        }
        Node::Store { index, value, .. } => {
            inside_launch_gate
                || expr_depends_on_launch_geometry(index, launch_vars)
                || expr_depends_on_launch_geometry(value, launch_vars)
        }
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            let new_gate = inside_launch_gate || expr_depends_on_launch_geometry(cond, launch_vars);
            let mut then_vars = launch_vars.clone();
            let mut otherwise_vars = launch_vars.clone();
            let then_writes =
                nodes_have_launch_geometry_dependent_write(then, &mut then_vars, new_gate);
            let otherwise_writes = nodes_have_launch_geometry_dependent_write(
                otherwise,
                &mut otherwise_vars,
                new_gate,
            );
            launch_vars.extend(then_vars);
            launch_vars.extend(otherwise_vars);
            then_writes || otherwise_writes
        }
        Node::Loop {
            var,
            from,
            to,
            body,
        } => {
            let new_gate = inside_launch_gate
                || expr_depends_on_launch_geometry(from, launch_vars)
                || expr_depends_on_launch_geometry(to, launch_vars);
            let mut body_vars = launch_vars.clone();
            if new_gate {
                body_vars.insert(var.clone());
            }
            nodes_have_launch_geometry_dependent_write(body, &mut body_vars, new_gate)
        }
        Node::Block(body) => {
            let mut body_vars = launch_vars.clone();
            nodes_have_launch_geometry_dependent_write(body, &mut body_vars, inside_launch_gate)
        }
        Node::Region { body, .. } => {
            let mut body_vars = launch_vars.clone();
            nodes_have_launch_geometry_dependent_write(body, &mut body_vars, inside_launch_gate)
        }
        Node::AsyncStore { offset, size, .. } => {
            inside_launch_gate
                || expr_depends_on_launch_geometry(offset, launch_vars)
                || expr_depends_on_launch_geometry(size, launch_vars)
        }
        Node::IndirectDispatch { .. }
        | Node::AllReduce { .. }
        | Node::AllGather { .. }
        | Node::ReduceScatter { .. }
        | Node::Broadcast { .. }
        | Node::Return
        | Node::Barrier { .. }
        | Node::LogicalBarrier { .. }
        | Node::AsyncLoad { .. }
        | Node::AsyncWait { .. }
        | Node::Trap { .. }
        | Node::Resume { .. }
        | Node::TileLoad { .. }
        | Node::TileStore { .. }
        | Node::TileMatmul { .. }
        | Node::TileReduce { .. }
        | Node::TileDecl { .. } => false,
        Node::Opaque(ext) => ext.is_divergent() || !ext.is_pure(),
        Node::TileElementwise { body, .. } => {
            nodes_have_launch_geometry_dependent_write(body, launch_vars, inside_launch_gate)
        }
    }
}

/// True when `expr` names launch geometry itself, before any operand of it is
/// considered.
///
/// Exhaustive with no catch-all arm: a new `Expr` variant fails to compile here
/// rather than reading as geometry-independent and letting a divergent write
/// fuse with a uniform one.
fn expr_names_launch_geometry(expr: &Expr) -> bool {
    match expr {
        Expr::InvocationId { .. }
        | Expr::WorkgroupId { .. }
        | Expr::LocalId { .. }
        | Expr::LogicalIndex { .. }
        | Expr::LogicalTileId { .. }
        | Expr::LogicalWithinTileId { .. }
        | Expr::SubgroupLocalId
        | Expr::SubgroupSize => true,
        Expr::LitU32(_)
        | Expr::LitI32(_)
        | Expr::LitF32(_)
        | Expr::LitBool(_)
        | Expr::Var(_)
        | Expr::BufferRef { .. }
        | Expr::BufLen { .. }
        | Expr::Load { .. }
        | Expr::BinOp { .. }
        | Expr::UnOp { .. }
        | Expr::Select { .. }
        | Expr::Cast { .. }
        | Expr::Fma { .. }
        | Expr::Atomic { .. }
        | Expr::Call { .. }
        | Expr::SubgroupBallot { .. }
        | Expr::SubgroupShuffle { .. }
        | Expr::SubgroupReduce { .. } => false,
        Expr::Opaque(ext) => !ext.cse_safe(),
    }
}

/// True when `expr` reads launch geometry, directly or through a name bound to
/// it.
///
/// The two questions this answers stood as two recursive functions with the
/// same twenty arms, differing only in whether a `Var` bound to geometry
/// counted. `launch_vars` is empty for the caller that asks only about the
/// intrinsics. Descent belongs to [`crate::visit::any_subexpr`], so a new
/// operand-carrying variant is covered without editing this file.
fn expr_depends_on_launch_geometry(expr: &Expr, launch_vars: &FxHashSet<Ident>) -> bool {
    crate::visit::any_subexpr(expr, &mut |current| {
        expr_names_launch_geometry(current)
            || matches!(current, Expr::Var(name) if launch_vars.contains(name))
    })
}
