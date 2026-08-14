//! Minimize identity-op atomics under Relaxed ordering to plain loads.
//! Non-identity read-modify-write atomics remain atomic because one syntactic
//! expression may execute concurrently in every invocation and loop iteration.
//!
//! Op id: `vyre-foundation::optimizer::passes::atomic_minimize`.

use crate::ir::{AtomicOp, Expr, Program};
use crate::memory_model::MemoryOrdering;
use crate::optimizer::rewrite::rewrite_program;
use crate::optimizer::{vyre_pass, PassAnalysis, PassResult};
use crate::transform::visit::any_expr_in;

/// Replace identity-op Relaxed atomics with plain loads.
#[derive(Debug, Default)]
#[vyre_pass(
    name = "atomic_minimize",
    requires = [],
    invalidates = [],
    phase = "sync",
    boundary_class = "abi_preserving",
    cost_model_family = "sync"
)]
pub struct AtomicMinimizePass;

impl AtomicMinimizePass {
    /// Skip programs that contain no identity atomic candidate.
    #[must_use]
    fn analyze_impl(program: &Program) -> PassAnalysis {
        if program.stats().atomic_op_count == 0 {
            return PassAnalysis::SKIP;
        }
        // The scan enumerates node nesting, node operands, and expression
        // operands through the three owners in `transform::visit`. The
        // hand-written scan this replaces ended in a catch-all node arm, so an
        // identity atomic reachable only through `Trap::address` or an async
        // copy offset made this report SKIP and the pass never ran.
        if any_expr_in(program.entry(), &mut is_identity_relaxed_atomic) {
            PassAnalysis::RUN
        } else {
            PassAnalysis::SKIP
        }
    }

    /// Walk the program and collapse identity atomics.
    #[must_use]
    pub fn transform(program: Program) -> PassResult {
        // `rewrite_program` owns the borrow-preserving structural walk: an
        // entry with no candidate comes back as the same allocation, which the
        // owned rebuild this replaces could not do. Its descent is exhaustive,
        // where the owned rewriter passed an unrecognised operand-carrying
        // variant through untouched.
        let (program, changed) = rewrite_program(program, |expr| match expr {
            Expr::Atomic { buffer, index, .. } if is_identity_relaxed_atomic(expr) => {
                Some(Expr::Load {
                    buffer: buffer.clone(),
                    index: index.clone(),
                })
            }
            _ => None,
        });
        PassResult { program, changed }
    }
}

/// True when `expr` is an atomic whose read-modify-write leaves memory as it
/// found it, so the atomic carries no more meaning than a plain load.
///
/// Both side conditions are load-bearing. `Relaxed` is the only ordering that
/// carries no fence, so collapsing any other ordering would drop the
/// synchronization the program asked for. An `expected` operand makes the
/// expression a compare-exchange, whose result reports whether the comparison
/// succeeded rather than the value that was read.
fn is_identity_relaxed_atomic(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Atomic {
            op,
            value,
            expected: None,
            ordering: MemoryOrdering::Relaxed,
            ..
        } if is_identity_atomic(*op, value)
    )
}

fn is_identity_atomic(op: AtomicOp, value: &Expr) -> bool {
    matches!(
        (op, value),
        (
            AtomicOp::Add | AtomicOp::Or | AtomicOp::Xor,
            Expr::LitU32(0) | Expr::LitI32(0)
        ) | (AtomicOp::And, Expr::LitU32(u32::MAX) | Expr::LitI32(-1))
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node};
    use crate::transform::visit::for_each_node;

    fn buf() -> BufferDecl {
        BufferDecl::storage("buf", 0, BufferAccess::ReadWrite, DataType::U32).with_count(4)
    }

    fn program(entry: Vec<Node>) -> Program {
        Program::wrapped(vec![buf()], [1, 1, 1], entry)
    }

    fn relaxed_atomic(op: AtomicOp, value: Expr) -> Expr {
        Expr::Atomic {
            op,
            buffer: Ident::from("buf"),
            index: Box::new(Expr::u32(0)),
            expected: None,
            value: Box::new(value),
            ordering: MemoryOrdering::Relaxed,
        }
    }

    /// Node nesting comes from the `child_bodies` owner behind `walk_nodes`,
    /// so a `Let` in a body shape this helper never named is still found.
    fn extract_let_value(p: &Program, name: &str) -> Expr {
        let mut found = None;
        crate::transform::visit::walk_nodes(p, |node| {
            if let Node::Let { name: bound, value } = node {
                if bound.as_str() == name {
                    found = Some(value.clone());
                }
            }
        });
        found.unwrap_or_else(|| panic!("expected Let `{name}` in entry tree"))
    }

    #[test]
    fn add_zero_relaxed_collapses_to_load() {
        let entry = vec![Node::let_bind(
            "x",
            relaxed_atomic(AtomicOp::Add, Expr::u32(0)),
        )];
        let result = AtomicMinimizePass::transform(program(entry));
        assert!(result.changed);
        assert_eq!(
            extract_let_value(&result.program, "x"),
            Expr::Load {
                buffer: Ident::from("buf"),
                index: Box::new(Expr::u32(0)),
            }
        );
    }

    #[test]
    fn or_zero_relaxed_collapses_to_load() {
        let entry = vec![Node::let_bind(
            "x",
            relaxed_atomic(AtomicOp::Or, Expr::u32(0)),
        )];
        let result = AtomicMinimizePass::transform(program(entry));
        assert!(result.changed);
        // Must collapse to a Load on exactly the same buffer and index, not
        // just any Expr::Load (guards against wrong-buffer/index regressions).
        assert_eq!(
            extract_let_value(&result.program, "x"),
            Expr::Load {
                buffer: Ident::from("buf"),
                index: Box::new(Expr::u32(0)),
            },
            "Or(x, 0) Relaxed must collapse to Load {{ buffer: \"buf\", index: 0 }}"
        );
    }

    #[test]
    fn xor_zero_relaxed_collapses_to_load() {
        let entry = vec![Node::let_bind(
            "x",
            relaxed_atomic(AtomicOp::Xor, Expr::u32(0)),
        )];
        let result = AtomicMinimizePass::transform(program(entry));
        assert!(result.changed);
        assert_eq!(
            extract_let_value(&result.program, "x"),
            Expr::Load {
                buffer: Ident::from("buf"),
                index: Box::new(Expr::u32(0)),
            },
            "Xor(x, 0) Relaxed must collapse to Load {{ buffer: \"buf\", index: 0 }}"
        );
    }

    #[test]
    fn and_max_relaxed_collapses_to_load() {
        let entry = vec![Node::let_bind(
            "x",
            relaxed_atomic(AtomicOp::And, Expr::u32(u32::MAX)),
        )];
        let result = AtomicMinimizePass::transform(program(entry));
        assert!(result.changed);
        assert_eq!(
            extract_let_value(&result.program, "x"),
            Expr::Load {
                buffer: Ident::from("buf"),
                index: Box::new(Expr::u32(0)),
            },
            "And(x, u32::MAX) Relaxed must collapse to Load {{ buffer: \"buf\", index: 0 }}"
        );
    }

    #[test]
    fn syntactically_single_atomic_add_remains_atomic() {
        let entry = vec![Node::let_bind(
            "x",
            relaxed_atomic(AtomicOp::Add, Expr::u32(42)),
        )];
        let result = AtomicMinimizePass::transform(program(entry));
        assert!(
            !result.changed,
            "one atomic expression may execute concurrently in every invocation"
        );
        assert!(matches!(
            extract_let_value(&result.program, "x"),
            Expr::Atomic {
                op: AtomicOp::Add,
                ordering: MemoryOrdering::Relaxed,
                ..
            }
        ));
    }

    #[test]
    fn two_atomic_adds_keep_atomic() {
        let entry = vec![
            Node::let_bind("x", relaxed_atomic(AtomicOp::Add, Expr::u32(42))),
            Node::let_bind("y", relaxed_atomic(AtomicOp::Add, Expr::u32(43))),
        ];
        let result = AtomicMinimizePass::transform(program(entry));
        assert!(!result.changed);
        assert!(matches!(
            extract_let_value(&result.program, "x"),
            Expr::Atomic { .. }
        ));
    }

    #[test]
    fn atomic_with_load_keeps_atomic() {
        let entry = vec![
            Node::let_bind("x", relaxed_atomic(AtomicOp::Add, Expr::u32(42))),
            Node::let_bind(
                "y",
                Expr::Load {
                    buffer: Ident::from("buf"),
                    index: Box::new(Expr::u32(0)),
                },
            ),
        ];
        let result = AtomicMinimizePass::transform(program(entry));
        assert!(!result.changed);
        assert!(matches!(
            extract_let_value(&result.program, "x"),
            Expr::Atomic { .. }
        ));
    }

    #[test]
    fn atomic_with_store_keeps_atomic() {
        let entry = vec![
            Node::let_bind("x", relaxed_atomic(AtomicOp::Add, Expr::u32(42))),
            Node::store("buf", Expr::u32(1), Expr::u32(99)),
        ];
        let result = AtomicMinimizePass::transform(program(entry));
        assert!(!result.changed);
        assert!(matches!(
            extract_let_value(&result.program, "x"),
            Expr::Atomic { .. }
        ));
    }

    #[test]
    fn compare_exchange_not_eligible() {
        let entry = vec![Node::let_bind(
            "x",
            Expr::Atomic {
                op: AtomicOp::CompareExchange,
                buffer: Ident::from("buf"),
                index: Box::new(Expr::u32(0)),
                expected: Some(Box::new(Expr::u32(1))),
                value: Box::new(Expr::u32(42)),
                ordering: MemoryOrdering::Relaxed,
            },
        )];
        let result = AtomicMinimizePass::transform(program(entry));
        assert!(!result.changed);
    }

    #[test]
    fn generated_deep_identity_atomic_expression_rewrites_without_recursive_expr_walk() {
        for depth in [1usize, 8, 64, 512, 4096] {
            let mut value = relaxed_atomic(AtomicOp::Add, Expr::u32(0));
            for _ in 0..depth {
                value = Expr::add(value, Expr::u32(0));
            }

            let PassResult {
                program: rewritten_program,
                changed,
            } = AtomicMinimizePass::transform(program(vec![Node::let_bind("x", value)]));
            assert!(
                changed,
                "Fix: atomic_minimize must rewrite nested identity atomic at generated depth {depth}."
            );
            let rewritten = find_let_value_ref(&rewritten_program, "x")
                .expect("generated deep atomic program must retain let x");
            assert!(
                !expr_contains_atomic(rewritten),
                "Fix: atomic_minimize left an atomic inside generated depth {depth}: {rewritten:?}"
            );
            assert!(
                expr_contains_load(rewritten),
                "Fix: atomic_minimize must replace the identity atomic with a load at generated depth {depth}."
            );
        }
    }

    /// The value bound to `target` by the first matching `Let`, at any nesting
    /// depth.
    ///
    /// Descent comes from `transform::visit::for_each_node`, the one owner of
    /// which node variants nest. The hand-written worklist this replaces ended
    /// in `_ => {}`, so a binding inside a fifth body-bearing variant read as
    /// absent and the assertion below would have failed for the wrong reason.
    fn find_let_value_ref<'a>(program: &'a Program, target: &str) -> Option<&'a Expr> {
        let mut found: Option<&'a Expr> = None;
        for_each_node(program.entry(), |node| {
            if found.is_none() {
                if let Node::Let { name, value } = node {
                    if name.as_str() == target {
                        found = Some(value);
                    }
                }
            }
        });
        found
    }

    #[test]
    fn non_identity_seq_cst_atomic_remains_atomic() {
        let entry = vec![Node::let_bind(
            "x",
            Expr::Atomic {
                op: AtomicOp::Add,
                buffer: Ident::from("buf"),
                index: Box::new(Expr::u32(0)),
                expected: None,
                value: Box::new(Expr::u32(42)),
                ordering: MemoryOrdering::SeqCst,
            },
        )];
        let result = AtomicMinimizePass::transform(program(entry));
        assert!(!result.changed);
        assert!(matches!(
            extract_let_value(&result.program, "x"),
            Expr::Atomic {
                ordering: MemoryOrdering::SeqCst,
                ..
            }
        ));
    }

    #[test]
    fn analyze_skips_program_with_no_candidate() {
        let entry = vec![Node::let_bind("x", Expr::u32(7))];
        match crate::optimizer::ProgramPass::analyze(&AtomicMinimizePass, &program(entry)) {
            PassAnalysis::SKIP => {}
            other => panic!("expected SKIP, got {other:?}"),
        }
    }

    fn expr_contains_atomic(expr: &Expr) -> bool {
        expr_contains(expr, |expr| matches!(expr, Expr::Atomic { .. }))
    }

    fn expr_contains_load(expr: &Expr) -> bool {
        expr_contains(expr, |expr| matches!(expr, Expr::Load { .. }))
    }

    fn expr_contains(expr: &Expr, mut predicate: impl FnMut(&Expr) -> bool) -> bool {
        crate::transform::visit::any_subexpr(expr, &mut predicate)
    }
}
