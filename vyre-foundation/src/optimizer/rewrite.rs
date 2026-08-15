#![allow(clippy::expect_used)]
use crate::ir::{BinOp, Expr, Node, Program};
use crate::transform::rewrite_walk::{self, NodeRewrite};
use crate::visit::{any_subexpr, expr_children};
use smallvec::SmallVec;
use std::borrow::Cow;

/// Push every operand of `expr` onto an order-insensitive worklist.
///
/// Positions come from [`expr_children`], the one exhaustive owner, so a new
/// operand-carrying variant reaches every scan built on this without an edit
/// here.
pub(crate) fn push_expr_children<'a>(expr: &'a Expr, stack: &mut SmallVec<[&'a Expr; 16]>) {
    stack.extend(expr_children(expr).iter());
}

pub(crate) fn expr_contains_atomic(expr: &Expr) -> bool {
    any_subexpr(expr, &mut |candidate| {
        matches!(candidate, Expr::Atomic { .. })
    })
}

/// Run an expression-rewrite closure over every node in \`program\`.
pub(crate) fn rewrite_program(
    program: Program,
    mut expr: impl FnMut(&Expr) -> Option<Expr>,
) -> (Program, bool) {
    match rewrite_nodes_cow(program.entry(), &mut expr) {
        Cow::Borrowed(_) => (program, false),
        Cow::Owned(entry) => (program.with_rewritten_entry(entry), true),
    }
}

pub(crate) fn rewrite_node_slices<'a>(
    nodes: &'a [Node],
    mut rewrite: impl FnMut(&'a Node) -> Cow<'a, [Node]>,
) -> Cow<'a, [Node]> {
    let mut rewritten: Option<Vec<Node>> = None;
    for (index, node) in nodes.iter().enumerate() {
        match rewrite(node) {
            Cow::Borrowed(_) if rewritten.is_none() => {}
            Cow::Borrowed(borrowed) => {
                if let Some(out) = rewritten.as_mut() {
                    out.extend_from_slice(borrowed);
                }
            }
            Cow::Owned(owned) => {
                let out = rewritten.get_or_insert_with(|| nodes[..index].to_vec());
                out.extend(owned);
            }
        }
    }
    rewritten.map_or(Cow::Borrowed(nodes), Cow::Owned)
}

/// Offers every operand expression of a node to one closure, and nothing else.
///
/// The value namespace is left alone: `ident` keeps its default answer, so a
/// `Let` target, a `Loop` induction variable and an async copy tag survive an
/// expression rewrite unrenamed. That is what separates an expression rewrite
/// from the substitutions in [`crate::transform::subst`], which do rename.
struct ExprOperands<'f, F>(&'f mut F);

impl<F: FnMut(&Expr) -> Option<Expr>> NodeRewrite for ExprOperands<'_, F> {
    fn operand(&mut self, expr: &Expr) -> Option<Expr> {
        rewrite_operand(expr, self.0)
    }
}

/// `expr` rewritten under `transform`, or `None` when nothing in it changed.
///
/// The answer shape [`NodeRewrite::operand`] wants, from the `Cow` shape
/// [`rewrite_expr`] returns. Every `NodeRewrite` policy whose operand hook is a
/// whole-expression rewrite goes through here rather than converting the `Cow`
/// itself, because a policy that converts it by hand can get the direction
/// wrong and report a rewrite it did not make, which re-dirties every pass in
/// the scheduler's next fixpoint iteration.
pub(crate) fn rewrite_operand(
    expr: &Expr,
    transform: &mut impl FnMut(&Expr) -> Option<Expr>,
) -> Option<Expr> {
    match rewrite_expr(expr, transform) {
        Cow::Borrowed(_) => None,
        Cow::Owned(rewritten) => Some(rewritten),
    }
}

/// Run an expression-rewrite closure over every node of `nodes`.
///
/// Which positions exist is [`rewrite_walk::rewrite_node`]'s decision, the one
/// rewriting enumeration of `Node`. This module used to carry a second one, an
/// exhaustive `match` per variant that had to be edited in lockstep with the
/// owner; the pair had already diverged once, when the owner descended into an
/// async copy's `offset` and `size` and the copy here did not, so every pass
/// routed through [`rewrite_program`] left those two expression positions
/// unrewritten.
pub(crate) fn rewrite_nodes_cow<'a>(
    nodes: &'a [Node],
    expr: &mut impl FnMut(&Expr) -> Option<Expr>,
) -> Cow<'a, [Node]> {
    rewrite_walk::rewrite_body(nodes, &mut ExprOperands(expr))
        .map_or(Cow::Borrowed(nodes), Cow::Owned)
}

#[expect(
    clippy::too_many_lines,
    reason = "iterative expression rewrite keeps traversal and reassembly order in one stack machine"
)]
pub(crate) fn rewrite_expr<'a>(
    expr: &'a Expr,
    transform: &mut impl FnMut(&Expr) -> Option<Expr>,
) -> Cow<'a, Expr> {
    enum Frame<'a> {
        Expr(&'a Expr),
        Assemble(&'a Expr),
    }

    let mut stack: SmallVec<[Frame<'_>; 32]> = SmallVec::new();
    stack.push(Frame::Expr(expr));
    let mut results: SmallVec<[Cow<'a, Expr>; 32]> = SmallVec::new();

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Expr(e) => {
                stack.push(Frame::Assemble(e));
                // Operand positions and their order come from the one
                // exhaustive owner. Pushed in reverse so the children assemble
                // in source order, which is the order `Frame::Assemble` pops
                // its results in.
                stack.extend(expr_children(e).iter().rev().map(Frame::Expr));
            }
            Frame::Assemble(e) => {
                let rewritten = match e {
                    Expr::LitU32(_)
                    | Expr::LitI32(_)
                    | Expr::LitF32(_)
                    | Expr::LitBool(_)
                    | Expr::Var(_)
                    | Expr::BufferRef { .. }
                    | Expr::BufLen { .. }
                    | Expr::InvocationId { .. }
                    | Expr::WorkgroupId { .. }
                    | Expr::LocalId { .. }
                    | Expr::SubgroupLocalId
                    | Expr::SubgroupSize
                    | Expr::Opaque(_) => Cow::Borrowed(e),
                    Expr::Load { buffer, .. } => {
                        let index = pop_rewrite_result(&mut results, "load index");
                        match index {
                            Cow::Borrowed(_) => Cow::Borrowed(e),
                            Cow::Owned(index) => Cow::Owned(Expr::Load {
                                buffer: buffer.clone(),
                                index: Box::new(index),
                            }),
                        }
                    }
                    Expr::BinOp { op, .. } => {
                        let right = pop_rewrite_result(&mut results, "binary rhs");
                        let left = pop_rewrite_result(&mut results, "binary lhs");
                        rewrite_binary(e, *op, left, right)
                    }
                    Expr::UnOp { op, .. } => {
                        let operand = pop_rewrite_result(&mut results, "unary operand");
                        match operand {
                            Cow::Borrowed(_) => Cow::Borrowed(e),
                            Cow::Owned(operand) => Cow::Owned(Expr::UnOp {
                                op: op.clone(),
                                operand: Box::new(operand),
                            }),
                        }
                    }
                    Expr::Call { op_id, args } => {
                        let start_idx = results.len().checked_sub(args.len()).unwrap_or_else(|| {
                            unreachable!(
                                "Fix: iterative expression rewrite lost call arguments; child/result stack is internally inconsistent."
                            )
                        });
                        let arg_results: Vec<_> = results.drain(start_idx..).collect();
                        let changed = arg_results
                            .iter()
                            .any(|arg_res| matches!(arg_res, Cow::Owned(_)));
                        if changed {
                            Cow::Owned(Expr::Call {
                                op_id: op_id.clone(),
                                args: arg_results.into_iter().map(Cow::into_owned).collect(),
                            })
                        } else {
                            Cow::Borrowed(e)
                        }
                    }
                    Expr::Select { .. } => {
                        let false_val = pop_rewrite_result(&mut results, "select false value");
                        let true_val = pop_rewrite_result(&mut results, "select true value");
                        let cond = pop_rewrite_result(&mut results, "select condition");
                        rewrite_select(e, cond, true_val, false_val)
                    }
                    Expr::Cast { target, .. } => {
                        let value = pop_rewrite_result(&mut results, "cast value");
                        match value {
                            Cow::Borrowed(_) => Cow::Borrowed(e),
                            Cow::Owned(value) => Cow::Owned(Expr::Cast {
                                target: target.clone(),
                                value: Box::new(value),
                            }),
                        }
                    }
                    Expr::Fma { .. } => {
                        let c = pop_rewrite_result(&mut results, "fma c");
                        let b = pop_rewrite_result(&mut results, "fma b");
                        let a = pop_rewrite_result(&mut results, "fma a");
                        rewrite_fma(e, a, b, c)
                    }
                    Expr::Atomic {
                        op,
                        buffer,
                        ordering,
                        expected,
                        ..
                    } => {
                        let value = pop_rewrite_result(&mut results, "atomic value");
                        let new_expected = if expected.is_some() {
                            Some(pop_rewrite_result(&mut results, "atomic expected"))
                        } else {
                            None
                        };
                        let index = pop_rewrite_result(&mut results, "atomic index");
                        if matches!(index, Cow::Borrowed(_))
                            && new_expected
                                .as_ref()
                                .is_none_or(|ex| matches!(ex, Cow::Borrowed(_)))
                            && matches!(value, Cow::Borrowed(_))
                        {
                            Cow::Borrowed(e)
                        } else {
                            Cow::Owned(Expr::Atomic {
                                op: *op,
                                buffer: buffer.clone(),
                                index: Box::new(index.into_owned()),
                                expected: new_expected.map(|ex| Box::new(ex.into_owned())),
                                value: Box::new(value.into_owned()),
                                ordering: *ordering,
                            })
                        }
                    }
                    Expr::SubgroupBallot { .. } => {
                        let cond = pop_rewrite_result(&mut results, "subgroup ballot condition");
                        match cond {
                            Cow::Borrowed(_) => Cow::Borrowed(e),
                            Cow::Owned(cond) => Cow::Owned(Expr::SubgroupBallot {
                                cond: Box::new(cond),
                            }),
                        }
                    }
                    Expr::SubgroupShuffle { .. } => {
                        let lane = pop_rewrite_result(&mut results, "subgroup shuffle lane");
                        let value = pop_rewrite_result(&mut results, "subgroup shuffle value");
                        match (value, lane) {
                            (Cow::Borrowed(_), Cow::Borrowed(_)) => Cow::Borrowed(e),
                            (v, l) => Cow::Owned(Expr::SubgroupShuffle {
                                value: Box::new(v.into_owned()),
                                lane: Box::new(l.into_owned()),
                            }),
                        }
                    }
                    Expr::SubgroupReduce { op, .. } => {
                        let value = pop_rewrite_result(&mut results, "subgroup reduce value");
                        match value {
                            Cow::Borrowed(_) => Cow::Borrowed(e),
                            Cow::Owned(value) => Cow::Owned(Expr::SubgroupReduce {
                                op: *op,
                                value: Box::new(value),
                            }),
                        }
                    }
                };

                let transformed = if let Some(t) = transform(rewritten.as_ref()) {
                    Cow::Owned(t)
                } else {
                    rewritten
                };
                results.push(transformed);
            }
        }
    }
    match results.pop() {
        Some(result) => result,
        None => unreachable!(
            "Fix: iterative expression rewrite produced no result; child/result stack is internally inconsistent."
        ),
    }
}

#[inline]
fn pop_rewrite_result<'a>(
    results: &mut SmallVec<[Cow<'a, Expr>; 32]>,
    context: &'static str,
) -> Cow<'a, Expr> {
    match results.pop() {
        Some(result) => result,
        None => unreachable!(
            "Fix: iterative expression rewrite lost {context}; child/result stack is internally inconsistent."
        ),
    }
}

#[inline]
pub(super) fn rewrite_binary<'a>(
    original: &'a Expr,
    op: BinOp,
    left: Cow<'a, Expr>,
    right: Cow<'a, Expr>,
) -> Cow<'a, Expr> {
    if matches!((&left, &right), (Cow::Borrowed(_), Cow::Borrowed(_))) {
        return Cow::Borrowed(original);
    }
    Cow::Owned(Expr::BinOp {
        op,
        left: Box::new(left.into_owned()),
        right: Box::new(right.into_owned()),
    })
}

#[inline]
fn rewrite_ternary<'a>(
    original: &'a Expr,
    first: Cow<'a, Expr>,
    second: Cow<'a, Expr>,
    third: Cow<'a, Expr>,
    build: impl FnOnce(Expr, Expr, Expr) -> Expr,
) -> Cow<'a, Expr> {
    if matches!(
        (&first, &second, &third),
        (Cow::Borrowed(_), Cow::Borrowed(_), Cow::Borrowed(_))
    ) {
        return Cow::Borrowed(original);
    }
    Cow::Owned(build(
        first.into_owned(),
        second.into_owned(),
        third.into_owned(),
    ))
}

#[inline]
pub(super) fn rewrite_fma<'a>(
    original: &'a Expr,
    a: Cow<'a, Expr>,
    b: Cow<'a, Expr>,
    c: Cow<'a, Expr>,
) -> Cow<'a, Expr> {
    rewrite_ternary(original, a, b, c, |a, b, c| Expr::Fma {
        a: Box::new(a),
        b: Box::new(b),
        c: Box::new(c),
    })
}

#[inline]
pub(super) fn rewrite_select<'a>(
    original: &'a Expr,
    cond: Cow<'a, Expr>,
    true_val: Cow<'a, Expr>,
    false_val: Cow<'a, Expr>,
) -> Cow<'a, Expr> {
    rewrite_ternary(
        original,
        cond,
        true_val,
        false_val,
        |cond, true_val, false_val| Expr::Select {
            cond: Box::new(cond),
            true_val: Box::new(true_val),
            false_val: Box::new(false_val),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufferDecl, DataType};

    fn simple_program() -> Program {
        Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
        )
    }

    #[test]
    fn identity_rewrite_unchanged() {
        let p = simple_program();
        let (result, changed) = rewrite_program(p, |_| None);
        assert!(!changed);
        assert_eq!(result.entry().len(), 1);
    }

    #[test]
    fn rewrite_replaces_constant() {
        let p = simple_program();
        let (result, changed) = rewrite_program(p, |expr| match expr {
            Expr::LitU32(42) => Some(Expr::u32(99)),
            _ => None,
        });
        assert!(changed);
        let entry = result.entry();
        fn find_99(nodes: &[Node]) -> bool {
            nodes.iter().any(|n| match n {
                Node::Store { value, .. } => matches!(value, Expr::LitU32(99)),
                Node::Region { body, .. } => find_99(body),
                _ => false,
            })
        }
        assert!(find_99(entry));
    }

    #[test]
    fn rewrite_into_if_branch() {
        let p = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![Node::if_then(
                Expr::bool(true),
                vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
            )],
        );
        let (_result, changed) = rewrite_program(p, |expr| match expr {
            Expr::LitU32(7) => Some(Expr::u32(8)),
            _ => None,
        });
        assert!(changed);
    }

    #[test]
    fn rewrite_rebuild_preserves_child_order() {
        let expr = Expr::sub(Expr::u32(10), Expr::u32(3));
        let rewritten = rewrite_expr(&expr, &mut |expr| match expr {
            Expr::LitU32(3) => Some(Expr::u32(4)),
            _ => None,
        });
        assert_eq!(
            rewritten.into_owned(),
            Expr::sub(Expr::u32(10), Expr::u32(4))
        );

        let expr = Expr::Select {
            cond: Box::new(Expr::bool(false)),
            true_val: Box::new(Expr::u32(1)),
            false_val: Box::new(Expr::u32(2)),
        };
        let rewritten = rewrite_expr(&expr, &mut |expr| match expr {
            Expr::LitU32(2) => Some(Expr::u32(3)),
            _ => None,
        });
        assert_eq!(
            rewritten.into_owned(),
            Expr::Select {
                cond: Box::new(Expr::bool(false)),
                true_val: Box::new(Expr::u32(1)),
                false_val: Box::new(Expr::u32(3)),
            }
        );
    }
}
