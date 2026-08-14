//! Dead-branch elimination over the post-prop Program.
//!
//! Walks the IR looking for `Node::If { cond, then, otherwise }`
//! where `cond` is a constant literal:
//!
//! - `LitU32(0)` / `LitI32(0)` / `LitBool(false)` → drop the entire
//!   `then` branch; splice the `otherwise` body into the parent
//!   scope.
//! - `LitU32(non-zero)` / `LitBool(true)` → drop the `otherwise`;
//!   splice the `then` body into the parent scope.
//!
//! Splicing means the surviving branch's nodes replace the `If` in
//! place  -  they become siblings of the surrounding lets/stores,
//! which is the natural behaviour for an unconditional code path.
//!
//! The rewrite is recursive: surviving branch bodies are
//! themselves walked so nested If's with literal conds collapse all
//! the way down. After this pass, downstream DCE sees a flatter
//! Program with strictly fewer Nodes.
//!
//! Pure CPU; runs after const-prop so propagated literals (e.g.
//! `let cond = false; if cond { … }` → `if false { … }` →
//! eliminated) trigger eliminations they wouldn't otherwise see.

use std::sync::Arc;

use super::expr_no_atomic;
use vyre_foundation::ir::{Expr, Node, Program};

/// Apply dead-branch elimination. Returns a new Program with every
/// constant-cond `Node::If` collapsed to its surviving branch.
pub fn apply_dead_branch(program: &Program) -> Program {
    super::rewrite_program_entry(program, rewrite_scope)
}

fn rewrite_scope(body: &[Node]) -> Vec<Node> {
    let prefix_len = super::encode::reachable_prefix_len(body);
    let mut out: Vec<Node> = Vec::with_capacity(prefix_len);
    for node in &body[..prefix_len] {
        match node {
            Node::If {
                cond,
                then,
                otherwise,
            } => {
                if let Some(taken_branch) = const_branch(cond, then, otherwise) {
                    // Constant cond → splice the surviving branch's
                    // (recursively rewritten) nodes into the output.
                    out.extend(rewrite_scope(taken_branch));
                } else {
                    let new_then = rewrite_scope(then);
                    let new_otherwise = rewrite_scope(otherwise);
                    // Both branches empty + cond has no observable
                    // side effects → drop the entire If. `cond` Loads
                    // are read-only in vyre, so the only side-effect
                    // gate is `Atomic`.
                    if new_then.is_empty() && new_otherwise.is_empty() && expr_no_atomic(cond) {
                        continue;
                    }
                    // Both arms structurally equal + cond has no
                    // observable side effects → splice one copy of
                    // the body into the parent scope. Catches
                    // `if c { do_X(); } else { do_X(); }`.
                    if new_then == new_otherwise && expr_no_atomic(cond) {
                        out.extend(new_then);
                        continue;
                    }
                    out.push(Node::if_then_else(cond.clone(), new_then, new_otherwise));
                }
            }
            Node::Loop {
                var,
                from,
                to,
                body,
            } => {
                // Empty-loop elimination: when both `from` and `to`
                // are literal u32s and `from >= to`, the loop body
                // never executes and the entire `Node::Loop` drops.
                // The IR uses half-open `[from, to)` ranges so equal
                // bounds are also empty.
                if let (Expr::LitU32(f), Expr::LitU32(t)) = (from, to) {
                    if f >= t {
                        // Drop the loop entirely; don't even recurse
                        // into the body.
                        continue;
                    }
                }
                if let (Expr::LitI32(f), Expr::LitI32(t)) = (from, to) {
                    if f >= t {
                        continue;
                    }
                }
                // Empty range via structural equality of pure Exprs:
                // both `from` and `to` evaluate to the same runtime
                // value with no observable side effect. The
                // half-open range [from, to) is then empty.
                if from_to_structurally_equal(from, to) {
                    continue;
                }
                let inner = rewrite_scope(body);
                // No-op-body loop elimination: if the body is empty
                // after rewriting AND both bounds are atomic-free
                // (so evaluating them once vs. zero times has no
                // observable effect), drop the entire loop. Loads
                // are read-only and are considered safe to drop.
                if inner.is_empty() && expr_no_atomic(from) && expr_no_atomic(to) {
                    continue;
                }
                out.push(Node::loop_for(var.clone(), from.clone(), to.clone(), inner));
            }
            Node::Block(body) => {
                let inner = rewrite_scope(body);
                if inner.is_empty() {
                    // Empty block contributes no behaviour; drop it.
                    continue;
                }
                // Block is a transparent scope wrapper  -  splice the
                // inner Nodes into the parent scope instead of
                // wrapping them again. Avoids gratuitous nesting that
                // accumulates after iterative rewrites.
                out.extend(inner);
            }
            Node::Region {
                generator,
                source_region,
                body,
            } => {
                out.push(Node::Region {
                    generator: generator.clone(),
                    source_region: source_region.clone(),
                    body: Arc::new(rewrite_scope(body.as_slice())),
                });
            }
            // Leaves and value-only nodes pass through. We don't
            // touch their inner Exprs  -  const-prop already did.
            other => out.push(other.clone()),
        }
    }
    out
}

/// Structural equality on pure-Expr loop bounds: whether evaluating
/// `from` and `to` at the same program point would produce the same
/// runtime value with no observable side effect. Conservative  -
/// returns false on anything not in the small whitelist below.
fn from_to_structurally_equal(from: &Expr, to: &Expr) -> bool {
    match (from, to) {
        (Expr::LitU32(a), Expr::LitU32(b)) => a == b,
        (Expr::LitI32(a), Expr::LitI32(b)) => a == b,
        (Expr::Var(a), Expr::Var(b)) => a == b,
        (Expr::BufLen { buffer: a }, Expr::BufLen { buffer: b }) => a == b,
        (Expr::InvocationId { axis: a }, Expr::InvocationId { axis: b }) => a == b,
        (Expr::WorkgroupId { axis: a }, Expr::WorkgroupId { axis: b }) => a == b,
        (Expr::LocalId { axis: a }, Expr::LocalId { axis: b }) => a == b,
        _ => false,
    }
}

/// Decide whether `cond` is a constant literal that selects exactly
/// one branch. Returns `Some(taken)` if so. Non-literal conds and
/// literal types we don't reason about (`LitF32`, `LitI32 != 0`)
/// fall back to `None` so the If survives.
fn const_branch<'a>(cond: &Expr, then: &'a [Node], otherwise: &'a [Node]) -> Option<&'a [Node]> {
    match cond {
        Expr::LitBool(true) => Some(then),
        Expr::LitBool(false) => Some(otherwise),
        Expr::LitU32(0) => Some(otherwise),
        Expr::LitU32(_) => Some(then),
        Expr::LitI32(0) => Some(otherwise),
        Expr::LitI32(_) => Some(then),
        _ => None,
    }
}
