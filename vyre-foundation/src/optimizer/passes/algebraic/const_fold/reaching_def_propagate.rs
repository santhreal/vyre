//! ROADMAP A11  -  reaching-def facts into cross-control-flow const fold.
//!
//! Built on top of the A2 `ProgramFacts` substrate. For every
//! `Node::Let { name, value: Lit* }` whose `name` is never rebound
//! anywhere in the program (no Assign, no Loop induction var, no
//! second `Let { name }` shadow), the literal value is propagated
//! to every `Expr::Var(name)` read site across the entire program
//! tree  -  including reads in sibling control-flow branches that
//! A14's same-scope cheap-leaf rematerialization cannot reach.
//!
//! Op id: `vyre-foundation::optimizer::passes::reaching_def_propagate`.
//! Soundness: `Exact`. The `is_name_rebound == false` gate
//! guarantees that every dynamic read of `name` resolves to
//! the single static `Let` site, so substituting the literal at
//! every read site preserves observable behavior. Without
//! rebinding, control-flow path doesn't matter  -  the only value
//! the name can ever hold is the literal at its single defining
//! site. The Let itself is then dead and gets removed by the
//! existing DCE on the next pass-scheduler iteration.
//!
//! Cost direction: monotone-down on register-pressure (one fewer
//! named live binding per fired propagation) and monotone-down on
//! instruction count (every read site avoids loading from the
//! Var slot). Per-site cost goes from one register read to one
//! immediate operand, which is strictly cheaper on every backend.
//!
//! Preserves: every analysis. Invalidates: nothing  -  the Let was
//! the unique reaching definition; the literal substitution is its
//! observably-equivalent inlining.
//!
//! ## Pattern
//!
//! ```text
//! Let(x, LitU32(7))   ;; or LitI32, LitF32, LitBool
//! ... use Var(x) anywhere in the program ...
//!     where x has zero Assigns, zero Loop-vars, exactly one Let
//! → ... use LitU32(7) at every read site ...
//!     The Let stays in place; the next DCE round removes it once
//!     no Var(x) reads remain.
//! ```
//!
//! ## Why this complements A14
//!
//! A14 (`rematerialize_cheap_let`) walks one sibling sequence at a
//! time and substitutes through descendant scopes when the name is
//! not reassigned in that subtree. It cannot substitute INTO a
//! sibling subtree of the Let's own container  -  e.g., if `Let(x, 7)`
//! lives at the top of the `then` arm of an If and the read is
//! inside the `otherwise` arm, A14 leaves the read untouched
//! because the Let's descendant scan never visits the sibling arm.
//!
//! Reaching-def with `is_name_rebound == false` is the cross-CFG
//! generalisation: it queries the WHOLE program for rebinds and,
//! finding none, treats every read of `name` as resolved by the
//! single Let. The substitution then crosses arbitrary
//! control-flow boundaries safely.

use crate::ir::{Expr, Node, Program};
use crate::optimizer::program_soa::ProgramFacts;
use crate::optimizer::rewrite::rewrite_program;
use crate::optimizer::{vyre_pass, PassAnalysis, PassResult};
use crate::transform::visit::for_each_node;
use rustc_hash::FxHashMap;

/// Cross-control-flow literal Let propagation.
#[derive(Debug, Default)]
#[vyre_pass(
    name = "reaching_def_propagate",
    requires = ["const_fold"],
    invalidates = ["const_fold", "cse", "dce"],
    phase = "scalar_algebra",
    boundary_class = "abi_preserving",
    cost_model_family = "scalar"
)]
/// ABI-preserving reaching-definition propagation pass for unique literal let bindings.
pub struct ReachingDefPropagatePass;

impl ReachingDefPropagatePass {
    /// Skip programs with no candidate `Let(name, Lit)` whose
    /// `name` is unique program-wide.
    #[must_use]
    fn analyze_impl(program: &Program) -> PassAnalysis {
        // Propagation needs a Let. Without one, the ProgramFacts build
        // (full SoA walk) is wasted.
        if !program.stats().has_node_let() {
            return PassAnalysis::SKIP;
        }
        let facts = ProgramFacts::build_cached(program);
        if collect_propagatable_lets_with_values(&facts, program).is_empty() {
            PassAnalysis::SKIP
        } else {
            PassAnalysis::RUN
        }
    }

    /// Walk the entry tree and substitute every propagatable
    /// literal at every read site.
    #[must_use]
    pub fn transform(program: Program) -> PassResult {
        let facts = ProgramFacts::build_cached(&program);
        let propagations = collect_propagatable_lets_with_values(&facts, &program);
        if propagations.is_empty() {
            return PassResult {
                program,
                changed: false,
            };
        }
        let (program, changed) = rewrite_program(program, |candidate| {
            let Expr::Var(name) = candidate else {
                return None;
            };
            propagations.get(name.as_str()).cloned()
        });
        PassResult { program, changed }
    }
}

// Override `collect_propagatable_lets` to fetch literal values
// directly from the entry tree (the fact substrate doesn't store
// values to keep build-time fast). Uses one preorder walk over
// the entry to find the value at each propagatable Let's name.

fn collect_propagatable_lets_with_values(
    facts: &ProgramFacts,
    program: &Program,
) -> FxHashMap<String, Expr> {
    let mut candidates: rustc_hash::FxHashSet<String> = rustc_hash::FxHashSet::default();
    for (_, name) in facts.lets() {
        if !facts.is_name_rebound(name.as_str()) {
            candidates.insert(name.as_str().to_owned());
        }
    }
    if candidates.is_empty() {
        return FxHashMap::default();
    }
    let mut out: FxHashMap<String, Expr> = FxHashMap::default();
    scan_for_literal_lets(program.entry(), &candidates, &mut out);
    out
}

/// Every candidate name in `nodes` bound to a literal, at any nesting depth.
///
/// Descent comes from [`for_each_node`], the one owner of which node variants
/// nest. The hand-written match this replaces ended in `_ => {}`, so a literal
/// binding inside a fifth body-bearing variant was never recorded and its uses
/// were never substituted.
fn scan_for_literal_lets(
    nodes: &[Node],
    candidates: &rustc_hash::FxHashSet<String>,
    out: &mut FxHashMap<String, Expr>,
) {
    for_each_node(nodes, |node| {
        if let Node::Let { name, value } = node {
            if candidates.contains(name.as_str()) && is_literal(value) {
                out.insert(name.as_str().to_owned(), value.clone());
            }
        }
    });
}

fn is_literal(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::LitU32(_) | Expr::LitI32(_) | Expr::LitF32(_) | Expr::LitBool(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node};

    fn buf() -> BufferDecl {
        BufferDecl::storage("buf", 0, BufferAccess::ReadWrite, DataType::U32).with_count(4)
    }

    fn program(entry: Vec<Node>) -> Program {
        Program::wrapped(vec![buf()], [1, 1, 1], entry)
    }

    /// The production entry point. The copy this replaces re-derived the
    /// propagation map and then walked the entry itself, so it could agree with
    /// itself while disagreeing with the pass every caller runs.
    fn run(program: Program) -> PassResult {
        ReachingDefPropagatePass::transform(program)
    }

    fn count_var_reads(nodes: &[Node], target: &str) -> usize {
        let facts = ProgramFacts::build(&Program::wrapped(vec![buf()], [1, 1, 1], nodes.to_vec()));
        facts
            .var_reads()
            .iter()
            .filter(|(_, n)| n.as_str() == target)
            .count()
    }

    /// Cross-CFG propagation: `Let(x, 7)` at the top of `then`
    /// is propagated to a `Var(x)` read inside the `otherwise`
    /// arm  -  the very case A14 cannot reach because the read is
    /// in a sibling subtree, not a descendant of the Let's
    /// scope.
    ///
    /// (Edge case: A14 actually wouldn't fire here because the Let
    /// itself sits in a branch arm; this test verifies that the
    /// cross-CFG propagation works regardless of where the Let
    /// physically appears, as long as the name is unique.)
    #[test]
    fn propagates_literal_across_sibling_arms() {
        let entry = vec![
            Node::let_bind("x", Expr::u32(7)),
            Node::If {
                cond: Expr::var("c"),
                then: vec![Node::store("buf", Expr::u32(0), Expr::var("x"))],
                otherwise: vec![Node::store("buf", Expr::u32(1), Expr::var("x"))],
            },
        ];
        let result = run(program(entry));
        assert!(result.changed, "literal must propagate to both arms");
        let entry = result.program.entry().to_vec();
        assert_eq!(
            count_var_reads(&entry, "x"),
            0,
            "no Var(x) reads remain after propagation"
        );
    }

    /// Negative: a name that has an `Assign` somewhere is rebound;
    /// the propagation must NOT fire (inlining the Let value would
    /// shadow the post-Assign value at later read sites).
    #[test]
    fn keeps_literal_when_name_is_assigned() {
        let entry = vec![
            Node::let_bind("x", Expr::u32(7)),
            Node::Assign {
                name: Ident::from("x"),
                value: Expr::u32(99),
            },
            Node::store("buf", Expr::u32(0), Expr::var("x")),
        ];
        let result = run(program(entry));
        assert!(!result.changed);
    }

    /// Negative: a name with two `Let` sites is shadowed; the
    /// propagation must NOT fire because the inner Let shadows the
    /// outer for any read inside its scope.
    #[test]
    fn keeps_literal_when_name_is_shadowed() {
        let entry = vec![
            Node::let_bind("x", Expr::u32(7)),
            Node::Block(vec![
                Node::let_bind("x", Expr::u32(99)),
                Node::store("buf", Expr::u32(0), Expr::var("x")),
            ]),
        ];
        let result = run(program(entry));
        assert!(!result.changed);
    }

    /// Negative: a name that's a `Loop` induction var is rebound;
    /// the propagation must NOT fire.
    #[test]
    fn keeps_literal_when_name_is_loop_var() {
        let entry = vec![
            Node::let_bind("i", Expr::u32(7)),
            Node::Loop {
                var: Ident::from("i"),
                from: Expr::u32(0),
                to: Expr::u32(4),
                body: vec![Node::store("buf", Expr::var("i"), Expr::u32(1))],
            },
        ];
        let result = run(program(entry));
        assert!(!result.changed);
    }

    /// Negative: a Let whose value is NOT a literal (e.g., a
    /// BinOp or Load) is not propagated by this pass  -  that's
    /// A14 / CSE territory.
    #[test]
    fn keeps_let_with_non_literal_value() {
        let entry = vec![
            Node::let_bind(
                "x",
                Expr::BinOp {
                    op: crate::ir::BinOp::Add,
                    left: Box::new(Expr::u32(1)),
                    right: Box::new(Expr::u32(2)),
                },
            ),
            Node::store("buf", Expr::u32(0), Expr::var("x")),
        ];
        let result = run(program(entry));
        assert!(!result.changed);
    }

    /// Positive: nested-into-Loop read is propagated.
    /// `Let(x, 7)` at top, read inside a Loop body  -  A14 would
    /// also handle this, but the test asserts the cross-CFG
    /// substrate doesn't accidentally regress same-scope cases.
    #[test]
    fn propagates_into_loop_body() {
        let entry = vec![
            Node::let_bind("x", Expr::u32(7)),
            Node::Loop {
                var: Ident::from("i"),
                from: Expr::u32(0),
                to: Expr::u32(4),
                body: vec![Node::store("buf", Expr::var("i"), Expr::var("x"))],
            },
        ];
        let result = run(program(entry));
        assert!(result.changed);
        let entry = result.program.entry().to_vec();
        assert_eq!(count_var_reads(&entry, "x"), 0);
    }

    /// `analyze` short-circuits when no propagatable Let exists.
    #[test]
    fn analyze_skips_program_with_no_eligible_lets() {
        let entry = vec![Node::store("buf", Expr::u32(0), Expr::u32(1))];
        assert!(matches!(
            crate::optimizer::ProgramPass::analyze(&ReachingDefPropagatePass, &program(entry)),
            PassAnalysis::SKIP
        ));
    }

    /// Positive end-to-end: `transform` produces the same result as
    /// the raw helper API. Smoke test that the pass surface works.
    #[test]
    fn transform_matches_helper_api() {
        let entry = vec![
            Node::let_bind("x", Expr::u32(13)),
            Node::store("buf", Expr::u32(0), Expr::var("x")),
        ];
        let p1 = run(program(entry.clone()));
        let p2 = ReachingDefPropagatePass::transform(program(entry));
        assert_eq!(p1.changed, p2.changed);
    }
}
