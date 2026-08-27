//! Logical launch span read out of a program's own guards.
//!
//! A launch span derived from declared buffers takes the largest one. That is
//! correct for a gather, whose output is both the guarded domain and the widest
//! buffer, and wrong for a scatter, which guards on a small source and declares
//! a much larger destination. A paged cache append is the case that separates
//! them: it moves one decoded chunk into a cache sized for the whole sequence,
//! so a buffer-derived span fires one lane per cache element and the guard
//! discards all but the chunk.
//!
//! The guard is already in the IR. This analysis reads it, so the span is a
//! compiler-owned fact derived from the program rather than a number an
//! operation publishes for a caller to pass back down.
//!
//! The answer is `None` unless every effect in the program is dominated by a
//! constant upper bound on axis-0 logical index. An unbounded effect means high
//! lanes are observable and the buffer-derived span stands.
//!
//! An effect is not always a statement. An atomic read-modify-write reaches
//! memory from an expression position, and a program whose every write is an
//! atomic OR carries no effect-shaped statement at all, so operand expressions
//! are scanned for one. `admitted_logical_span` still takes the full-span path
//! for such a program: an atomic, a subgroup collective and a workgroup-scoped
//! buffer make the result depend on how many invocations ran rather than only
//! on which elements each one touched.
//!
//! A guard is not always the branch condition either. The predicated-tail form
//! binds the comparison to a local, selects a value that is zero outside it,
//! and branches on that value being nonzero, so the bound reaches the effect
//! through two locals. The walk proves that chain rather than reporting the
//! program unbounded.

use std::collections::{HashMap, HashSet};

use crate::ir::{BufferAccess, Expr, Node, Program};
use crate::ir_inner::model::expr::Ident;
use crate::ir_inner::model::op_signature::BinOp;
use crate::visit::{any_subexpr, node_operands, node_variadic_operands};

/// Largest axis-0 logical index a program can affect, when every effect it
/// performs is dominated by a constant bound on that index.
///
/// Returns `None` when the program performs an effect no such guard dominates,
/// because an unbounded effect leaves high lanes observable. A program that
/// performs no effect at all affects no index, so the answer is `Some(0)`: the
/// launch minimum belongs to whoever sizes the launch, not to this analysis.
#[must_use]
pub fn guarded_logical_span(program: &Program) -> Option<u32> {
    let mut walk = Walk {
        span: None,
        bounded: true,
    };
    let mut facts = Facts::default();
    walk.nodes(&program.entry, None, &mut facts);
    if walk.bounded {
        Some(walk.span.unwrap_or(0))
    } else {
        None
    }
}

/// Whether a launch must cover the whole input span whatever the guards admit.
///
/// Three constructs make the result depend on how many invocations ran rather
/// than only on which elements each one touched, so a narrower launch changes
/// the value instead of skipping idle lanes: an atomic, a subgroup collective,
/// and a workgroup-scoped buffer. The last one is the shared-memory reduction:
/// every lane of a group contributes a partial, and a launch narrowed to the
/// one-element output leaves the rest of the input unreduced.
#[must_use]
pub fn launch_covers_full_input_span(program: &Program) -> bool {
    program.stats().atomic_op_count > 0
        || program
            .buffers()
            .iter()
            .any(|buffer| buffer.access() == BufferAccess::Workgroup)
        || crate::program_caps::scan(program).subgroup_ops
}

/// Narrow a resource-derived launch span to the domain the program admits.
///
/// A resource-derived span takes the widest declared buffer, which a scatter
/// makes far larger than the domain its guard admits. Where every effect is
/// dominated by a constant bound on axis-0 logical index, that bound is the
/// authoritative domain and caps the launch. A full-span program keeps the
/// resource span. The result is at least one, because a launch of zero
/// workgroups records no work at all.
#[must_use]
pub fn admitted_logical_span(program: &Program, resource_span: u32) -> u32 {
    if launch_covers_full_input_span(program) {
        return resource_span.max(1);
    }
    match guarded_logical_span(program) {
        Some(guarded) => resource_span.min(guarded).max(1),
        None => resource_span.max(1),
    }
}

/// Facts the walk proves about locals in scope.
///
/// A guard is not always written against the index expression. The production
/// form binds the predicate to a local, selects a value that is zero outside
/// it, and branches on that value being nonzero, so the bound reaches the
/// effect through two locals rather than through the branch condition. Three
/// sets carry that chain: locals equal to axis-0 logical index, locals holding
/// a predicate that bounds the index, and locals whose value is zero once the
/// index passes a bound.
#[derive(Clone, Default)]
struct Facts {
    index: HashSet<Ident>,
    guards: HashMap<Ident, u32>,
    zeroed: HashMap<Ident, u32>,
}

impl Facts {
    /// Drop every fact about `name`.
    fn forget(&mut self, name: &Ident) {
        self.index.remove(name);
        self.guards.remove(name);
        self.zeroed.remove(name);
    }

    /// Record what `value` proves about the local it is bound to.
    fn learn(&mut self, name: &Ident, value: &Expr) {
        let index = is_axis_zero_index(value, &self.index);
        let guard = axis_zero_upper_bound(value, self);
        let zeroed = zero_outside_bound(value, self);
        self.forget(name);
        if index {
            self.index.insert(name.clone());
        } else if let Some(limit) = guard {
            self.guards.insert(name.clone(), limit);
        } else if let Some(limit) = zeroed {
            self.zeroed.insert(name.clone(), limit);
        }
    }
}

/// Accumulated span, and whether every effect seen so far was bounded.
struct Walk {
    span: Option<u32>,
    bounded: bool,
}

impl Walk {
    /// Record an effect observed under `bound`.
    fn effect(&mut self, bound: Option<u32>) {
        match bound {
            Some(limit) => {
                self.span = Some(self.span.map_or(limit, |seen: u32| seen.max(limit)));
            }
            None => self.bounded = false,
        }
    }

    /// Walk a statement list under an active bound.
    fn nodes(&mut self, nodes: &[Node], bound: Option<u32>, facts: &mut Facts) {
        for node in nodes {
            self.node(node, bound, facts);
        }
    }

    /// Record an atomic this node performs in one of its own operands.
    ///
    /// An atomic reaches memory from an expression position, so a walk that
    /// only counts effect-shaped statements reports "affects no index" for a
    /// program whose every write is an atomic read-modify-write. Operand
    /// positions come from `node_operands` and `node_variadic_operands`, so a
    /// new node variant carries its operands here without naming them again.
    fn atomic_operands(&mut self, node: &Node, bound: Option<u32>) {
        let mut is_atomic = |expr: &Expr| matches!(expr, Expr::Atomic { .. });
        let scalar = node_operands(node)
            .into_iter()
            .flatten()
            .any(|operand| any_subexpr(operand, &mut is_atomic));
        let variadic = node_variadic_operands(node)
            .iter()
            .any(|operand| any_subexpr(operand, &mut is_atomic));
        if scalar || variadic {
            self.effect(bound);
        }
    }

    fn node(&mut self, node: &Node, bound: Option<u32>, facts: &mut Facts) {
        self.atomic_operands(node, bound);
        match node {
            Node::Let { name, value } => facts.learn(name, value),
            Node::Assign { name, .. } => facts.forget(name),
            Node::If {
                cond,
                then,
                otherwise,
            } => {
                let taken = match axis_zero_upper_bound(cond, facts) {
                    Some(limit) => Some(bound.map_or(limit, |outer| outer.min(limit))),
                    None => bound,
                };
                let mut inner = facts.clone();
                self.nodes(then, taken, &mut inner);
                let mut alternate = facts.clone();
                self.nodes(otherwise, bound, &mut alternate);
            }
            Node::Loop { var, body, .. } => {
                let mut inner = facts.clone();
                inner.forget(var);
                let mut rebound = HashSet::new();
                rebound_names(body, &mut rebound);
                for name in &rebound {
                    inner.forget(name);
                }
                self.nodes(body, bound, &mut inner);
            }
            Node::Block(body) => {
                let mut inner = facts.clone();
                self.nodes(body, bound, &mut inner);
            }
            Node::Region { body, .. } => {
                let mut inner = facts.clone();
                self.nodes(body, bound, &mut inner);
            }
            Node::TileElementwise { body, .. } => {
                self.effect(bound);
                let mut inner = facts.clone();
                self.nodes(body, bound, &mut inner);
            }
            Node::Store { .. }
            | Node::AsyncStore { .. }
            | Node::TileStore { .. }
            | Node::TileMatmul { .. }
            | Node::TileReduce { .. }
            | Node::IndirectDispatch { .. }
            | Node::Trap { .. }
            | Node::AllReduce { .. }
            | Node::AllGather { .. }
            | Node::ReduceScatter { .. }
            | Node::Broadcast { .. }
            | Node::Opaque(_) => self.effect(bound),
            Node::TileLoad { tile: name, .. } | Node::TileDecl { name, .. } => facts.forget(name),
            Node::AsyncLoad { .. }
            | Node::AsyncWait { .. }
            | Node::Resume { .. }
            | Node::Return
            | Node::Barrier { .. }
            | Node::LogicalBarrier { .. } => {}
        }
    }
}

/// Names `nodes` rebinds anywhere inside itself.
///
/// A loop body runs more than once, and the walk reads it once. A local proven
/// equal to axis-0 logical index before the loop, and reassigned inside it, no
/// longer holds the index on the second iteration, so a guard written against
/// that local bounds nothing. Dropping every rebound name at the loop boundary
/// keeps the analysis conservative there.
fn rebound_names(nodes: &[Node], out: &mut HashSet<Ident>) {
    for node in nodes {
        match node {
            Node::Let { name, .. } | Node::Assign { name, .. } => {
                out.insert(name.clone());
            }
            Node::If {
                then, otherwise, ..
            } => {
                rebound_names(then, out);
                rebound_names(otherwise, out);
            }
            Node::Loop { var, body, .. } => {
                out.insert(var.clone());
                rebound_names(body, out);
            }
            Node::Block(body) | Node::TileElementwise { body, .. } => rebound_names(body, out),
            Node::Region { body, .. } => rebound_names(body, out),
            Node::Store { .. }
            | Node::AsyncStore { .. }
            | Node::TileStore { .. }
            | Node::TileMatmul { .. }
            | Node::TileReduce { .. }
            | Node::IndirectDispatch { .. }
            | Node::Trap { .. }
            | Node::AllReduce { .. }
            | Node::AllGather { .. }
            | Node::ReduceScatter { .. }
            | Node::Broadcast { .. }
            | Node::Opaque(_)
            | Node::AsyncLoad { .. }
            | Node::AsyncWait { .. }
            | Node::Resume { .. }
            | Node::Return
            | Node::Barrier { .. }
            | Node::LogicalBarrier { .. } => {}
            Node::TileLoad { tile: name, .. } | Node::TileDecl { name, .. } => {
                out.insert(name.clone());
            }
        }
    }
}

/// Whether `expr` is axis-0 logical index, directly or through a proven local.
fn is_axis_zero_index(expr: &Expr, names: &HashSet<Ident>) -> bool {
    match expr {
        Expr::LogicalIndex { axis } | Expr::InvocationId { axis } => *axis == 0,
        Expr::Var(name) => names.contains(name),
        _ => false,
    }
}

/// Bound past which `expr` evaluates to zero for every axis-0 logical index.
///
/// A predicated tail is written as `select(index < k, value, 0)`, so the value
/// carries the guard even where no branch does. The fact composes: masking with
/// a zeroed value keeps the tighter bound, and adding two of them keeps the
/// wider one.
fn zero_outside_bound(expr: &Expr, facts: &Facts) -> Option<u32> {
    match expr {
        Expr::LitU32(0) => Some(0),
        Expr::Var(name) => facts.zeroed.get(name).copied(),
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            if literal_u32(false_val) != Some(0) {
                return None;
            }
            let guard = axis_zero_upper_bound(cond, facts);
            match (guard, zero_outside_bound(true_val, facts)) {
                (Some(bound), Some(taken)) => Some(bound.min(taken)),
                (bound, None) | (None, bound) => bound,
            }
        }
        Expr::BinOp { op, left, right } => match op {
            BinOp::BitAnd | BinOp::Mul => {
                let left_bound = zero_outside_bound(left, facts);
                let right_bound = zero_outside_bound(right, facts);
                match (left_bound, right_bound) {
                    (Some(left_limit), Some(right_limit)) => Some(left_limit.min(right_limit)),
                    (bound, None) | (None, bound) => bound,
                }
            }
            BinOp::BitOr | BinOp::Add => {
                Some(zero_outside_bound(left, facts)?.max(zero_outside_bound(right, facts)?))
            }
            _ => None,
        },
        _ => None,
    }
}

/// Constant upper bound `cond` places on axis-0 logical index, if any.
///
/// `index < k` and `index <= k` bound the taken branch. `index == k` bounds it
/// too, and is the form a serial region takes: one lane does the whole walk, so
/// a launch sized from its output buffer would fire one full walk per element.
/// A conjunction bounds the branch through whichever side bounds it more
/// tightly, because both hold there; a disjunction needs both sides bounded and
/// keeps the wider one. A local holding a predicate carries that predicate's
/// bound, and a test that a zeroed value is nonzero carries the bound past
/// which the value is zero.
fn axis_zero_upper_bound(cond: &Expr, facts: &Facts) -> Option<u32> {
    if let Expr::Var(name) = cond {
        return facts.guards.get(name).copied();
    }
    let Expr::BinOp { op, left, right } = cond else {
        return None;
    };
    let names = &facts.index;
    match op {
        BinOp::Lt if is_axis_zero_index(left, names) => literal_u32(right),
        BinOp::Le if is_axis_zero_index(left, names) => literal_u32(right)?.checked_add(1),
        BinOp::Gt if is_axis_zero_index(right, names) => literal_u32(left),
        BinOp::Ge if is_axis_zero_index(right, names) => literal_u32(left)?.checked_add(1),
        BinOp::Eq if is_axis_zero_index(left, names) => literal_u32(right)?.checked_add(1),
        BinOp::Eq if is_axis_zero_index(right, names) => literal_u32(left)?.checked_add(1),
        BinOp::Ne if literal_u32(right) == Some(0) => zero_outside_bound(left, facts),
        BinOp::Ne if literal_u32(left) == Some(0) => zero_outside_bound(right, facts),
        BinOp::Gt if literal_u32(right) == Some(0) => zero_outside_bound(left, facts),
        BinOp::Lt if literal_u32(left) == Some(0) => zero_outside_bound(right, facts),
        BinOp::And => {
            let left_bound = axis_zero_upper_bound(left, facts);
            let right_bound = axis_zero_upper_bound(right, facts);
            match (left_bound, right_bound) {
                (Some(left_limit), Some(right_limit)) => Some(left_limit.min(right_limit)),
                (bound, None) | (None, bound) => bound,
            }
        }
        BinOp::Or => {
            Some(axis_zero_upper_bound(left, facts)?.max(axis_zero_upper_bound(right, facts)?))
        }
        _ => None,
    }
}

/// Literal `u32` value of `expr`, if it is one.
fn literal_u32(expr: &Expr) -> Option<u32> {
    match expr {
        Expr::LitU32(value) => Some(*value),
        _ => None,
    }
}
