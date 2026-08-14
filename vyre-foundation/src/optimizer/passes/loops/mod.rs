//! Loop / reduction catalog (Phase 4B).
//!
//! Loop-shape rewrites: trip-count zero elimination + bounded
//! compile-time unroll + redundant-bound-check guard elimination. These are
//! IR-level transformations; target-specific loop emission remains inside the
//! driver crates.

/// Shared legality / dependence analysis for the loop restructuring passes.
mod legality;
/// Tighten a `Node::Loop` upper bound when its body is a single
/// `If(Lt(Var(loop_var), Lit(n)), ...)` with `n < to` (ROADMAP A19).
pub mod loop_bound_tighten;
/// Fission a single `Node::Loop` body into two sibling loops sharing
/// the same iteration space when the body partitions cleanly into
/// buffer-disjoint, name-flow-isolated halves (ROADMAP A27).
pub mod loop_fission;
/// Fuse adjacent `Node::Loop` siblings whose bounds match and whose
/// bodies touch disjoint buffer sets (ROADMAP A26).
pub mod loop_fusion;
/// Hoist loop-invariant `Node::Let` bindings out of `Node::Loop`
/// bodies (ROADMAP A17).
pub mod loop_licm;
/// Polyhedral lower-bound normalization: rewrite `Loop(i, lo, hi, body)`
/// with `lo > 0` to `Loop(i', 0, hi-lo, body[i := i'+lo])` so
/// downstream tile/strip-mine/fusion passes see canonical
/// `from=0` bounds (ROADMAP A30).
pub mod loop_lower_bound_normalize;
/// Peel the first iteration of `Node::Loop` when guarded by
/// `If(Eq(Var(loop_var), Lit(0)), ...)` (ROADMAP A28).
pub mod loop_peel;
/// Drop redundant `if loop_var < to { ... }` guards inside matching loops.
pub mod loop_redundant_bound_check_elide;
/// 2-stage Load-then-Store software pipelining: rewrite a loop
/// body whose Load + dependent Store touch distinct buffers into
/// prologue + steady-state-with-prefetch + epilogue (ROADMAP A31).
pub mod loop_software_pipeline;
/// Strip-mine large literal loops into tiled outer and fixed-size
/// inner loops (ROADMAP A29).
pub mod loop_strip_mine;
/// Drop `Node::Loop` whose compile-time-known trip count is zero.
pub mod loop_trip_zero_eliminate;
/// Compile-time-known bounded loop expansion.
pub mod loop_unroll;
/// Loop-induction range facts that fold known-true / known-false
/// `If(Cmp(Var(i), LitU32(n)), then, else)` conditions inside
/// `Loop(i, lo, hi, body)` (ROADMAP A16  -  range facts into branch
/// elision via the structural loop range).
pub mod loop_var_range_fold;
mod substitution;

fn collect_vars_in_expr(expr: &crate::ir::Expr, out: &mut rustc_hash::FxHashSet<crate::ir::Ident>) {
    let mut stack = smallvec::SmallVec::<[&crate::ir::Expr; 16]>::new();
    stack.push(expr);
    while let Some(candidate) = stack.pop() {
        if let crate::ir::Expr::Var(name) = candidate {
            out.insert(name.clone());
        }
        crate::optimizer::rewrite::push_expr_children(candidate, &mut stack);
    }
}

fn collect_var_reads(nodes: &[crate::ir::Node], out: &mut rustc_hash::FxHashSet<crate::ir::Ident>) {
    for node in nodes {
        match node {
            crate::ir::Node::Let { value, .. } | crate::ir::Node::Assign { value, .. } => {
                collect_vars_in_expr(value, out);
            }
            crate::ir::Node::Store { index, value, .. } => {
                collect_vars_in_expr(index, out);
                collect_vars_in_expr(value, out);
            }
            crate::ir::Node::If {
                cond,
                then,
                otherwise,
            } => {
                collect_vars_in_expr(cond, out);
                collect_var_reads(then, out);
                collect_var_reads(otherwise, out);
            }
            crate::ir::Node::Loop { from, to, body, .. } => {
                collect_vars_in_expr(from, out);
                collect_vars_in_expr(to, out);
                collect_var_reads(body, out);
            }
            crate::ir::Node::Block(body) => collect_var_reads(body, out),
            crate::ir::Node::Region { body, .. } => collect_var_reads(body, out),
            _ => {}
        }
    }
}

fn collect_buffers_in_expr(
    expr: &crate::ir::Expr,
    out: &mut rustc_hash::FxHashSet<crate::ir::Ident>,
) {
    let mut stack = smallvec::SmallVec::<[&crate::ir::Expr; 16]>::new();
    stack.push(expr);
    while let Some(candidate) = stack.pop() {
        match candidate {
            crate::ir::Expr::Load { buffer, .. }
            | crate::ir::Expr::BufLen { buffer }
            | crate::ir::Expr::BufferRef { buffer }
            | crate::ir::Expr::Atomic { buffer, .. } => {
                out.insert(buffer.clone());
            }
            _ => {}
        }
        crate::optimizer::rewrite::push_expr_children(candidate, &mut stack);
    }
}

fn collect_touched_buffers(
    nodes: &[crate::ir::Node],
    out: &mut rustc_hash::FxHashSet<crate::ir::Ident>,
) {
    for node in nodes {
        match node {
            crate::ir::Node::Store {
                buffer,
                index,
                value,
            } => {
                out.insert(buffer.clone());
                collect_buffers_in_expr(index, out);
                collect_buffers_in_expr(value, out);
            }
            crate::ir::Node::Let { value, .. } | crate::ir::Node::Assign { value, .. } => {
                collect_buffers_in_expr(value, out);
            }
            crate::ir::Node::If {
                cond,
                then,
                otherwise,
            } => {
                collect_buffers_in_expr(cond, out);
                collect_touched_buffers(then, out);
                collect_touched_buffers(otherwise, out);
            }
            crate::ir::Node::Loop { from, to, body, .. } => {
                collect_buffers_in_expr(from, out);
                collect_buffers_in_expr(to, out);
                collect_touched_buffers(body, out);
            }
            crate::ir::Node::Block(body) => collect_touched_buffers(body, out),
            crate::ir::Node::Region { body, .. } => collect_touched_buffers(body, out),
            crate::ir::Node::AsyncLoad {
                source,
                destination,
                offset,
                size,
                ..
            }
            | crate::ir::Node::AsyncStore {
                source,
                destination,
                offset,
                size,
                ..
            } => {
                out.insert(source.clone());
                out.insert(destination.clone());
                collect_buffers_in_expr(offset, out);
                collect_buffers_in_expr(size, out);
            }
            crate::ir::Node::IndirectDispatch { count_buffer, .. } => {
                out.insert(count_buffer.clone());
            }
            crate::ir::Node::Trap { address, .. } => collect_buffers_in_expr(address, out),
            crate::ir::Node::AllReduce { buffer, .. }
            | crate::ir::Node::Broadcast { buffer, .. } => {
                out.insert(buffer.clone());
            }
            crate::ir::Node::AllGather { input, output, .. }
            | crate::ir::Node::ReduceScatter { input, output, .. } => {
                out.insert(input.clone());
                out.insert(output.clone());
            }
            crate::ir::Node::Barrier { .. }
            | crate::ir::Node::Return
            | crate::ir::Node::AsyncWait { .. }
            | crate::ir::Node::Resume { .. }
            | crate::ir::Node::Opaque(_) => {}
        }
    }
}

fn buffers_disjoint_with(
    first: &[crate::ir::Node],
    second: &[crate::ir::Node],
    collect: fn(&[crate::ir::Node], &mut rustc_hash::FxHashSet<crate::ir::Ident>),
) -> bool {
    let mut first_buffers = rustc_hash::FxHashSet::default();
    let mut second_buffers = rustc_hash::FxHashSet::default();
    collect(first, &mut first_buffers);
    collect(second, &mut second_buffers);
    first_buffers.is_disjoint(&second_buffers)
}

fn rename_var_in_expr(
    expr: crate::ir::Expr,
    from: &crate::ir::Ident,
    to: &crate::ir::Ident,
) -> crate::ir::Expr {
    crate::optimizer::rewrite::rewrite_expr(&expr, &mut |candidate| match candidate {
        crate::ir::Expr::Var(name) if name == from => Some(crate::ir::Expr::Var(to.clone())),
        _ => None,
    })
    .into_owned()
}
