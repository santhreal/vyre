//! polyhedral loop-bound normalization.
//!
//! Shipped variant: lower-bound normalization. Every literal-bounded
//! `Loop(i, lo, hi, body)` with `lo > 0` and `hi >= lo` rewrites to
//! `Loop(i', 0, hi - lo, body[i := i' + lo])`. The iteration space is
//! preserved exactly, the trip count is unchanged, and every body
//! expression that read the original loop variable now reads
//! `Var(i') + LitU32(lo)`. This is the polyhedral library's
//! `Affine::Translate(-lo)` rewrite  -  the simplest piece of a real
//! polyhedral substrate, and the prerequisite for the iteration-
//! space normalisation that A29 strip-mine, A26 fusion, and A28 peel
//! all assume.
//!
//! Op id: `vyre-foundation::optimizer::passes::loop_lower_bound_normalize`.
//! Soundness: `Exact`. The rewrite is a pure variable substitution
//! over an integer interval; the body sees `i' + lo` at every site
//! that previously read `i`, so every observable side effect (Store,
//! Atomic, Async, Trap) is keyed on the same value as before.
//! Cost direction: monotone-down on the canonical-form metric used
//! by downstream passes (A29 strip-mine refuses non-zero lower
//! bounds; A26 fusion's bounds-match check is symmetric over
//! normalised loops). Per-iteration cost rises by one Add at each
//! `Var(i)` read site; downstream const-fold + strength-reduce
//! collapses `i' + lo` back into a single offset register before
//! emit, so the net IR size after the next algebraic round is
//! unchanged.
//!
//! ## Pattern
//!
//! ```text
//! Loop(i, LitU32(lo), LitU32(hi), body)
//!     where lo > 0 AND hi >= lo
//! → Loop(i', LitU32(0), LitU32(hi - lo),
//!         body with every Var(i) replaced by (Var(i') + LitU32(lo)))
//! ```
//! The induction variable keeps its name. `is_normalizable_loop` refuses a body
//! that rebinds it, so after the shift every `Var(i)` in the body is the one
//! the loop header binds and reads `i + lo`, which is the value the original
//! header would have produced at the same iteration.
//!
//! ## Conservatism
//!
//! - Both bounds must be `Expr::LitU32` literals. Non-literal lower
//!   bounds need symbolic interval arithmetic (the proper polyhedral
//!   substrate); literal bounds are the structural slice we can
//!   prove sound today.
//! - `lo == 0` is already canonical; the pass skips so we don't
//!   busy-loop the scheduler.
//! - `lo > hi` produces a zero-trip loop and is left for
//!   `loop_trip_zero_eliminate` to drop on its next pass.
//! - The loop variable must not be reassigned anywhere inside the
//!   body (no `Node::Assign { name: i, .. }`), and the loop must not
//!   contain a nested Loop that re-binds the same name. Both
//!   collisions block the rewrite.

use super::substitution::body_rebinds_var;
use crate::ir::{BinOp, Expr, Node, Program};
use crate::optimizer::passes::driver;
use crate::optimizer::{vyre_pass, PassAnalysis, PassResult};
use crate::transform::subst::substitute_nodes;

/// Polyhedral lower-bound normalization pass.
#[derive(Debug, Default)]
#[vyre_pass(
    name = "loop_lower_bound_normalize",
    requires = ["const_fold"],
    invalidates = ["loop_unroll", "loop_strip_mine"]
)]
pub struct LoopLowerBoundNormalize;

impl LoopLowerBoundNormalize {
    /// Skip programs that have no normalizable Loop.
    #[must_use]
    fn analyze_impl(program: &Program) -> PassAnalysis {
        driver::analyze_candidates(
            program,
            &[crate::ir::stats::NODE_KIND_LOOP],
            &mut is_normalizable_loop,
        )
    }

    /// Walk the entry tree and normalize every eligible Loop.
    #[must_use]
    pub fn transform(program: Program) -> PassResult {
        driver::rewrite_entry_nodes(program, &mut normalize_loop)
    }
}

/// `node` with its induction range shifted to start at zero, when the shift is
/// legal.
///
/// Eligibility is [`is_normalizable_loop`]'s decision, so the analysis and the
/// rewrite cannot disagree about which loop the pass fires on.
fn normalize_loop(node: &Node) -> Option<Vec<Node>> {
    if !is_normalizable_loop(node) {
        return None;
    }
    let Node::Loop {
        var,
        from,
        to,
        body,
    } = node
    else {
        return None;
    };
    let (Expr::LitU32(lo), Expr::LitU32(hi)) = (from, to) else {
        return None;
    };
    Some(vec![Node::Loop {
        var: var.clone(),
        from: Expr::u32(0),
        to: Expr::u32(hi - lo),
        body: substitute_nodes(
            body,
            var,
            &Expr::BinOp {
                op: BinOp::Add,
                left: Box::new(Expr::Var(var.clone())),
                right: Box::new(Expr::u32(*lo)),
            },
        ),
    }])
}

fn is_normalizable_loop(node: &Node) -> bool {
    if let Node::Loop {
        var,
        from,
        to,
        body,
    } = node
    {
        match (from, to) {
            (Expr::LitU32(lo), Expr::LitU32(hi)) if *lo > 0 && *hi >= *lo => {}
            _ => return false,
        }
        !body_rebinds_var(body, var)
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node};

    fn buf() -> BufferDecl {
        BufferDecl::storage("buf", 0, BufferAccess::ReadWrite, DataType::U32).with_count(16)
    }

    fn program(entry: Vec<Node>) -> Program {
        Program::wrapped(vec![buf()], [1, 1, 1], entry)
    }

    fn find_loop(nodes: &[Node]) -> Option<&Node> {
        for n in nodes {
            if matches!(n, Node::Loop { .. }) {
                return Some(n);
            }
            match n {
                Node::Block(body) => {
                    if let Some(found) = find_loop(body) {
                        return Some(found);
                    }
                }
                Node::Region { body, .. } => {
                    if let Some(found) = find_loop(body.as_ref()) {
                        return Some(found);
                    }
                }
                Node::If {
                    then, otherwise, ..
                } => {
                    if let Some(found) = find_loop(then) {
                        return Some(found);
                    }
                    if let Some(found) = find_loop(otherwise) {
                        return Some(found);
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Positive: `Loop(i, 4, 12, store(buf, i, ...))` rewrites to
    /// `Loop(i', 0, 8, store(buf, i' + 4, ...))`.
    #[test]
    fn rewrites_positive_lower_bound_to_zero() {
        let entry = vec![Node::Loop {
            var: Ident::from("i"),
            from: Expr::u32(4),
            to: Expr::u32(12),
            body: vec![Node::store("buf", Expr::var("i"), Expr::u32(1))],
        }];
        let result = LoopLowerBoundNormalize::transform(program(entry));
        assert!(result.changed, "loop with from=4 must normalize");
        let loop_node = find_loop(result.program.entry()).expect("Fix: loop present");
        match loop_node {
            Node::Loop {
                var,
                from,
                to,
                body,
            } => {
                assert_eq!(var.as_str(), "i", "var is not freshened after #2734");
                assert_eq!(*from, Expr::LitU32(0), "from must be 0");
                assert_eq!(*to, Expr::LitU32(8), "to must be original (12) - lower (4)");

                match &body[0] {
                    Node::Store { index, .. } => match index {
                        Expr::BinOp { op, left, right } => {
                            assert_eq!(*op, BinOp::Add);
                            assert!(
                                matches!(left.as_ref(), Expr::Var(name) if name.as_str() == var.as_str())
                            );
                            assert_eq!(*right.as_ref(), Expr::LitU32(4));
                        }
                        other => panic!("expected Var(i') + 4, got {other:?}"),
                    },
                    other => panic!("expected Store, got {other:?}"),
                }
            }
            other => panic!("expected Loop, got {other:?}"),
        }
    }

    /// Negative: `Loop(i, 0, N, ...)` is already canonical and
    /// must not be touched.
    #[test]
    fn keeps_loop_with_zero_lower_bound() {
        let entry = vec![Node::Loop {
            var: Ident::from("i"),
            from: Expr::u32(0),
            to: Expr::u32(8),
            body: vec![Node::store("buf", Expr::var("i"), Expr::u32(1))],
        }];
        let result = LoopLowerBoundNormalize::transform(program(entry));
        assert!(!result.changed, "from=0 is already canonical");
    }

    /// Negative: non-literal `from` skips (needs symbolic substrate).
    #[test]
    fn keeps_loop_with_runtime_lower_bound() {
        let entry = vec![Node::Loop {
            var: Ident::from("i"),
            from: Expr::var("k"),
            to: Expr::u32(10),
            body: vec![Node::store("buf", Expr::var("i"), Expr::u32(1))],
        }];
        let result = LoopLowerBoundNormalize::transform(program(entry));
        assert!(!result.changed, "runtime from must skip");
    }

    /// Negative: non-literal `to` skips.
    #[test]
    fn keeps_loop_with_runtime_upper_bound() {
        let entry = vec![Node::Loop {
            var: Ident::from("i"),
            from: Expr::u32(2),
            to: Expr::var("n"),
            body: vec![Node::store("buf", Expr::var("i"), Expr::u32(1))],
        }];
        let result = LoopLowerBoundNormalize::transform(program(entry));
        assert!(!result.changed, "runtime to must skip");
    }

    /// Negative: `lo > hi` is a zero-trip loop; left for
    /// `loop_trip_zero_eliminate` to drop.
    #[test]
    fn keeps_loop_with_inverted_bounds() {
        let entry = vec![Node::Loop {
            var: Ident::from("i"),
            from: Expr::u32(10),
            to: Expr::u32(4),
            body: vec![Node::store("buf", Expr::var("i"), Expr::u32(1))],
        }];
        let result = LoopLowerBoundNormalize::transform(program(entry));
        assert!(
            !result.changed,
            "inverted bounds must be left for trip-zero pass"
        );
    }

    /// Negative: a loop body that reassigns the loop var blocks the
    /// rewrite  -  substitution would not preserve the in-body
    /// reassignment semantics.
    #[test]
    fn keeps_loop_when_body_assigns_loop_var() {
        let entry = vec![Node::Loop {
            var: Ident::from("i"),
            from: Expr::u32(2),
            to: Expr::u32(10),
            body: vec![
                Node::Assign {
                    name: Ident::from("i"),
                    value: Expr::u32(99),
                },
                Node::store("buf", Expr::var("i"), Expr::u32(1)),
            ],
        }];
        let result = LoopLowerBoundNormalize::transform(program(entry));
        assert!(!result.changed, "Assign to loop var must block rewrite");
    }

    /// Negative: a nested Loop that re-binds the same var name
    /// blocks the rewrite (would shadow the substituted name).
    #[test]
    fn keeps_loop_when_nested_loop_shadows_var() {
        let entry = vec![Node::Loop {
            var: Ident::from("i"),
            from: Expr::u32(2),
            to: Expr::u32(10),
            body: vec![Node::Loop {
                var: Ident::from("i"),
                from: Expr::u32(0),
                to: Expr::u32(4),
                body: vec![],
            }],
        }];
        let result = LoopLowerBoundNormalize::transform(program(entry));
        assert!(!result.changed, "shadowing nested Loop must block rewrite");
    }

    /// Positive: nested loop nests normalize bottom-up. Inner
    /// `Loop(j, 5, 10, ...)` normalizes; outer `Loop(i, 0, 4, ...)`
    /// stays canonical.
    #[test]
    fn normalizes_nested_loop_independently() {
        let entry = vec![Node::Loop {
            var: Ident::from("i"),
            from: Expr::u32(0),
            to: Expr::u32(4),
            body: vec![Node::Loop {
                var: Ident::from("j"),
                from: Expr::u32(5),
                to: Expr::u32(10),
                body: vec![Node::store("buf", Expr::var("j"), Expr::u32(1))],
            }],
        }];
        let result = LoopLowerBoundNormalize::transform(program(entry));
        assert!(result.changed, "inner loop must normalize");
    }

    /// `analyze` short-circuits when no eligible Loop is present.
    #[test]
    fn analyze_skips_program_with_only_canonical_loops() {
        let entry = vec![Node::Loop {
            var: Ident::from("i"),
            from: Expr::u32(0),
            to: Expr::u32(8),
            body: vec![],
        }];
        match crate::optimizer::ProgramPass::analyze(&LoopLowerBoundNormalize, &program(entry)) {
            PassAnalysis::SKIP => {}
            other => panic!("expected SKIP, got {other:?}"),
        }
    }
}
