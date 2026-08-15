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
/// Shared IR fixtures for the loop restructuring passes' own tests.
#[cfg(test)]
mod test_fixtures;

/// Every scalar name an expression anywhere in `nodes` reads, name-sorted.
///
/// This is the read set every loop restructuring pass asks about before it
/// reorders statements. It allocates; the accumulating form inside this module
/// is what the passes use when they already own a set.
#[must_use]
pub fn var_reads(nodes: &[crate::ir::Node]) -> Vec<crate::ir::Ident> {
    let mut set = rustc_hash::FxHashSet::default();
    collect_var_reads(nodes, &mut set);
    sorted_names(set)
}

/// Every buffer any statement in `nodes` touches, name-sorted.
///
/// Reads and writes are collapsed because the disjointness question the loop
/// passes ask cares only about overlap. It allocates; the accumulating form
/// inside this module is what the passes use when they already own a set.
#[must_use]
pub fn touched_buffers(nodes: &[crate::ir::Node]) -> Vec<crate::ir::Ident> {
    let mut set = rustc_hash::FxHashSet::default();
    collect_touched_buffers(nodes, &mut set);
    sorted_names(set)
}

/// Every name any statement in `nodes` binds, nested scopes included,
/// name-sorted.
///
/// This is the set the loop passes intersect with a read set to decide whether a
/// binding is live across a restructuring boundary.
#[must_use]
pub fn bound_names(nodes: &[crate::ir::Node]) -> Vec<crate::ir::Ident> {
    let mut set = rustc_hash::FxHashSet::default();
    legality::collect_bound_names(nodes, &mut set);
    sorted_names(set)
}

fn sorted_names(set: rustc_hash::FxHashSet<crate::ir::Ident>) -> Vec<crate::ir::Ident> {
    let mut out: Vec<crate::ir::Ident> = set.into_iter().collect();
    out.sort_by(|left, right| left.as_str().cmp(right.as_str()));
    out
}

/// Every name read by an expression anywhere in `nodes`, nested scopes included.
///
/// Node nesting, operand positions, and sub-expressions all come from
/// [`for_each_expr`](crate::visit::for_each_expr), whose three
/// enumerations are this crate's exhaustive owners.
///
/// The hand-written descent this replaces ended in `_ => {}`, so a `Var` read
/// inside `Node::Trap.address` or an async copy's `offset` / `size` read as
/// ABSENT. `loop_fusion::fusion_has_scalar_dependency` then saw no cross-loop
/// dependency where one existed and fused two loops across a scalar that one of
/// them assigns, which silently changes the observed values; the same reads were
/// invisible to `legality::bindings_flow_across`, weakening the capture guard
/// for both fusion and fission.
fn collect_var_reads(nodes: &[crate::ir::Node], out: &mut rustc_hash::FxHashSet<crate::ir::Ident>) {
    crate::visit::for_each_expr(nodes, |expr| {
        if let crate::ir::Expr::Var(name) = expr {
            out.insert(name.clone());
        }
    });
}

/// Every buffer named by `expr` or any sub-expression.
///
/// The per-variant decision is
/// [`expr_buffer_ref`](crate::visit::expr_buffer_ref) and the descent
/// is [`for_each_subexpr`](crate::visit::for_each_subexpr), both
/// exhaustive. The match this replaces ended in `_ => {}` over `Expr`, so an
/// expression variant that gains a buffer position would report the two loop
/// bodies as touching disjoint memory and let fusion or fission reorder a real
/// memory dependence.
///
/// `Expr::Opaque` names no buffer here even though its real effect is unknown;
/// [`legality::unsummarisable_effect`] is the guard that refuses it, and it runs
/// before this answer is used.
fn collect_buffers_in_expr(
    expr: &crate::ir::Expr,
    out: &mut rustc_hash::FxHashSet<crate::ir::Ident>,
) {
    crate::visit::for_each_subexpr(expr, &mut |candidate| match crate::visit::expr_buffer_ref(
        candidate,
    ) {
        crate::visit::ExprBufferRef::Read(buffer)
        | crate::visit::ExprBufferRef::ReadWrite(buffer) => {
            out.insert(buffer.clone());
        }
        crate::visit::ExprBufferRef::None | crate::visit::ExprBufferRef::Unknown => {}
    });
}

/// Every buffer any statement in `nodes` touches, by name or through an operand.
///
/// The two halves of the answer come from their owners:
/// [`node_buffer_refs`](crate::visit::node_buffer_refs) for the
/// buffers a statement names directly and [`collect_buffers_in_expr`] for the
/// ones an operand expression reaches. Direction is collapsed because the
/// disjointness test the loop passes run cares only about overlap.
fn collect_touched_buffers(
    nodes: &[crate::ir::Node],
    out: &mut rustc_hash::FxHashSet<crate::ir::Ident>,
) {
    crate::visit::for_each_node(nodes, |node| {
        let refs = crate::visit::node_buffer_refs(node);
        for buffer in refs.reads.into_iter().chain(refs.writes).flatten() {
            out.insert(buffer.clone());
        }
        for operand in crate::visit::node_operands(node).into_iter().flatten() {
            collect_buffers_in_expr(operand, out);
        }
    });
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
