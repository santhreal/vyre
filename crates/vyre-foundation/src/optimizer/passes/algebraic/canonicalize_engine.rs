//! P4.2  -  Canonical-form pass.
//!
//! Rewrites `Program` IR into a canonical shape so semantically-equal
//! Programs have byte-equal wire output. This is the foundation the
//! content-addressed pipeline cache (P4.3) pins against: two Cat-A
//! authors who write the same computation via different spellings
//! get the same fingerprint after `canonicalize(program)`.
//!
//! Rules applied:
//!
//! 1. **Hoist literal operands to the right** on commutative BinOps
//!    so `1 + a` canonicalizes to `a + 1`. Non-literal operand order
//!    is preserved because IEEE-754 NaN payload propagation makes
//!    `a + b` and `b + a` observably different for float operations.
//!    Commutativity is looked up in `vyre-spec::BinOp` (Add/Mul,
//!    And/Or/Xor, Eq/Ne). Min/Max are intentionally excluded.
//! 2. **Hoist literal-left**: when only one operand of a commutative
//!    op is a literal, order is literal-on-right. Canonical form
//!    for constant folding.
//!
//! Rules owned by CSE / DCE / rewrite passes:
//!
//! - Expression-CSE (lives in `cse.rs`).
//! - Identity-fold (`x + 0 → x`)  -  requires `AlgebraicLaw::Identity`
//!   registration lookup; next pass.
//! - Associativity-rearranging  -  left-fold stays left, no right-
//!   rotation.
//! - **Reflexive-comparison value folding (`x == x → true`,
//!   `x != x → false`) is deliberately NOT done here.** canonicalize is
//!   type-blind (it has no `DataType` for a bare `Var`), and reflexive
//!   comparison is *unsound* for float operands: a float `x` that is
//!   `NaN` at runtime makes `x == x` false and `x != x` true under
//!   IEEE-754, which the reference oracle (`vyre-reference`
//!   `binop_f32`) and every target emitter honor.
//!   Folding it to a bool literal here would (a) miscompile NaN checks
//!   and (b) corrupt the content-addressed cache by giving the
//!   genuinely-distinct programs `x == x` and `true` the same
//!   fingerprint. Type-aware value folding lives in `const_fold`, where
//!   the operand type can be proven before the fold fires.
//!
//! The pass is idempotent: `canonicalize(canonicalize(p)) ==
//! canonicalize(p)` on every valid `Program`.

use crate::ir_inner::model::expr::Expr;
use crate::ir_inner::model::node::Node;
use crate::ir_inner::model::program::Program;
use crate::optimizer::PassResult;
use std::borrow::Cow;

/// Run the canonical-form pass on `program`, returning the
/// canonicalized form with `validated=false` so downstream passes
/// re-check the rewritten shape.
#[must_use]
pub fn run(program: Program) -> Program {
    run_pass(program).program
}

/// Run the canonical-form pass and report whether any expression changed
/// without cloning or rebuilding an already-canonical `Program`.
#[must_use]
pub fn run_pass(program: Program) -> PassResult {
    match canonicalize_nodes_cow(program.entry()) {
        Cow::Borrowed(_) => PassResult::unchanged(program),
        Cow::Owned(rewritten) => PassResult::changed(program.with_rewritten_entry(rewritten)),
    }
}

/// Run the canonical-form pass from a borrowed `Program`.
///
/// This is for hot paths such as pipeline fingerprinting where the caller owns
/// a shared `Program` and only needs a temporary canonical serialization view.
/// It preserves the program scaffold by reference and clones only the entry
/// nodes that are actually rewritten.
#[must_use]
pub fn run_borrowed(program: &Program) -> Program {
    match canonicalize_nodes_cow(program.entry()) {
        Cow::Borrowed(_) => program.clone(),
        Cow::Owned(rewritten) => program.with_rewritten_entry(rewritten),
    }
}

fn canonicalize_nodes_cow(nodes: &[Node]) -> Cow<'_, [Node]> {
    for (idx, node) in nodes.iter().enumerate() {
        if let Cow::Owned(first) = canonicalize_node_cow(node) {
            let mut out = Vec::with_capacity(nodes.len());
            out.extend_from_slice(&nodes[..idx]);
            out.push(first);
            for rest in &nodes[idx + 1..] {
                out.push(canonicalize_node_cow(rest).into_owned());
            }
            return Cow::Owned(out);
        }
    }
    Cow::Borrowed(nodes)
}

fn canonicalize_exprs_cow(exprs: &[Expr]) -> Cow<'_, [Expr]> {
    for (idx, expr) in exprs.iter().enumerate() {
        if let Cow::Owned(first) = canonicalize_expr_cow(expr) {
            let mut out = Vec::with_capacity(exprs.len());
            out.extend_from_slice(&exprs[..idx]);
            out.push(first);
            for rest in &exprs[idx + 1..] {
                out.push(canonicalize_expr_cow(rest).into_owned());
            }
            return Cow::Owned(out);
        }
    }
    Cow::Borrowed(exprs)
}

fn canonicalize_node_cow(node: &Node) -> Cow<'_, Node> {
    match node {
        Node::Let { name, value } => match canonicalize_expr_cow(value) {
            Cow::Borrowed(_) => Cow::Borrowed(node),
            Cow::Owned(value) => Cow::Owned(Node::Let {
                name: name.clone(),
                value,
            }),
        },
        Node::Assign { name, value } => match canonicalize_expr_cow(value) {
            Cow::Borrowed(_) => Cow::Borrowed(node),
            Cow::Owned(value) => Cow::Owned(Node::Assign {
                name: name.clone(),
                value,
            }),
        },
        Node::Store {
            buffer,
            index,
            value,
        } => match (canonicalize_expr_cow(index), canonicalize_expr_cow(value)) {
            (Cow::Borrowed(_), Cow::Borrowed(_)) => Cow::Borrowed(node),
            (index, value) => Cow::Owned(Node::Store {
                buffer: buffer.clone(),
                index: index.into_owned(),
                value: value.into_owned(),
            }),
        },
        Node::If {
            cond,
            then,
            otherwise,
        } => match (
            canonicalize_expr_cow(cond),
            canonicalize_nodes_cow(then),
            canonicalize_nodes_cow(otherwise),
        ) {
            (Cow::Borrowed(_), Cow::Borrowed(_), Cow::Borrowed(_)) => Cow::Borrowed(node),
            (cond, then, otherwise) => Cow::Owned(Node::If {
                cond: cond.into_owned(),
                then: then.into_owned(),
                otherwise: otherwise.into_owned(),
            }),
        },
        Node::Loop {
            var,
            from,
            to,
            body,
        } => match (
            canonicalize_expr_cow(from),
            canonicalize_expr_cow(to),
            canonicalize_nodes_cow(body),
        ) {
            (Cow::Borrowed(_), Cow::Borrowed(_), Cow::Borrowed(_)) => Cow::Borrowed(node),
            (from, to, body) => Cow::Owned(Node::Loop {
                var: var.clone(),
                from: from.into_owned(),
                to: to.into_owned(),
                body: body.into_owned(),
            }),
        },
        Node::Block(children) => match canonicalize_nodes_cow(children) {
            Cow::Borrowed(_) => Cow::Borrowed(node),
            Cow::Owned(children) => Cow::Owned(Node::Block(children)),
        },
        Node::Region {
            generator,
            source_region,
            body,
        } => match canonicalize_nodes_cow(body) {
            Cow::Borrowed(_) => Cow::Borrowed(node),
            Cow::Owned(new_body) => Cow::Owned(Node::Region {
                generator: generator.clone(),
                source_region: source_region.clone(),
                body: std::sync::Arc::new(new_body),
            }),
        },
        _ => Cow::Borrowed(node),
    }
}

fn canonicalize_expr_cow(expr: &Expr) -> Cow<'_, Expr> {
    match expr {
        Expr::BinOp { op, left, right } => {
            let l = canonicalize_expr_cow(left);
            let r = canonicalize_expr_cow(right);
            if crate::ir_inner::model::program::canonical::commutative_operands_out_of_order(
                *op,
                l.as_ref(),
                r.as_ref(),
            ) {
                return Cow::Owned(Expr::BinOp {
                    op: *op,
                    left: Box::new(r.into_owned()),
                    right: Box::new(l.into_owned()),
                });
            }
            match (l, r) {
                (Cow::Borrowed(_), Cow::Borrowed(_)) => Cow::Borrowed(expr),
                (l, r) => Cow::Owned(Expr::BinOp {
                    op: *op,
                    left: Box::new(l.into_owned()),
                    right: Box::new(r.into_owned()),
                }),
            }
        }
        Expr::UnOp { op, operand } => match canonicalize_expr_cow(operand) {
            Cow::Borrowed(_) => Cow::Borrowed(expr),
            Cow::Owned(operand) => Cow::Owned(Expr::UnOp {
                op: op.clone(),
                operand: Box::new(operand),
            }),
        },
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => match (
            canonicalize_expr_cow(cond),
            canonicalize_expr_cow(true_val),
            canonicalize_expr_cow(false_val),
        ) {
            (Cow::Borrowed(_), Cow::Borrowed(_), Cow::Borrowed(_)) => Cow::Borrowed(expr),
            (cond, true_val, false_val) => Cow::Owned(Expr::Select {
                cond: Box::new(cond.into_owned()),
                true_val: Box::new(true_val.into_owned()),
                false_val: Box::new(false_val.into_owned()),
            }),
        },
        Expr::Fma { a, b, c } => match (
            canonicalize_expr_cow(a),
            canonicalize_expr_cow(b),
            canonicalize_expr_cow(c),
        ) {
            (Cow::Borrowed(_), Cow::Borrowed(_), Cow::Borrowed(_)) => Cow::Borrowed(expr),
            (a, b, c) => Cow::Owned(Expr::Fma {
                a: Box::new(a.into_owned()),
                b: Box::new(b.into_owned()),
                c: Box::new(c.into_owned()),
            }),
        },
        Expr::Cast { target, value } => match canonicalize_expr_cow(value) {
            Cow::Borrowed(_) => Cow::Borrowed(expr),
            Cow::Owned(value) => Cow::Owned(Expr::Cast {
                target: target.clone(),
                value: Box::new(value),
            }),
        },
        Expr::Load { buffer, index } => match canonicalize_expr_cow(index) {
            Cow::Borrowed(_) => Cow::Borrowed(expr),
            Cow::Owned(index) => Cow::Owned(Expr::Load {
                buffer: buffer.clone(),
                index: Box::new(index),
            }),
        },
        Expr::Atomic {
            op,
            buffer,
            index,
            expected,
            value,
            ordering,
        } => {
            let idx_cow = canonicalize_expr_cow(index);
            let exp_cow = expected.as_deref().map(canonicalize_expr_cow);
            let val_cow = canonicalize_expr_cow(value);
            if matches!(idx_cow, Cow::Borrowed(_))
                && exp_cow
                    .as_ref()
                    .is_none_or(|e| matches!(e, Cow::Borrowed(_)))
                && matches!(val_cow, Cow::Borrowed(_))
            {
                Cow::Borrowed(expr)
            } else {
                Cow::Owned(Expr::Atomic {
                    op: *op,
                    buffer: buffer.clone(),
                    index: Box::new(idx_cow.into_owned()),
                    expected: exp_cow.map(|e| Box::new(e.into_owned())),
                    value: Box::new(val_cow.into_owned()),
                    ordering: *ordering,
                })
            }
        }
        Expr::Call { op_id, args } => match canonicalize_exprs_cow(args) {
            Cow::Borrowed(_) => Cow::Borrowed(expr),
            Cow::Owned(args) => Cow::Owned(Expr::Call {
                op_id: op_id.clone(),
                args,
            }),
        },
        Expr::SubgroupBallot { cond } => match canonicalize_expr_cow(cond) {
            Cow::Borrowed(_) => Cow::Borrowed(expr),
            Cow::Owned(cond) => Cow::Owned(Expr::SubgroupBallot {
                cond: Box::new(cond),
            }),
        },
        Expr::SubgroupShuffle { value, lane } => {
            match (canonicalize_expr_cow(value), canonicalize_expr_cow(lane)) {
                (Cow::Borrowed(_), Cow::Borrowed(_)) => Cow::Borrowed(expr),
                (value, lane) => Cow::Owned(Expr::SubgroupShuffle {
                    value: Box::new(value.into_owned()),
                    lane: Box::new(lane.into_owned()),
                }),
            }
        }
        Expr::SubgroupReduce { op, value } => match canonicalize_expr_cow(value) {
            Cow::Borrowed(_) => Cow::Borrowed(expr),
            Cow::Owned(value) => Cow::Owned(Expr::SubgroupReduce {
                op: *op,
                value: Box::new(value),
            }),
        },
        _ => Cow::Borrowed(expr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinOp, BufferDecl, DataType, Expr as E, Program};

    /// Which operand side a literal must land on is what these assertions
    /// name. The canonical order itself is decided in
    /// `ir_inner::model::program::canonical`.
    fn is_literal(expr: &Expr) -> bool {
        matches!(
            expr,
            Expr::LitU32(_) | Expr::LitI32(_) | Expr::LitF32(_) | Expr::LitBool(_)
        )
    }

    fn scalar_out_prog(body: Vec<Node>) -> Program {
        Program::wrapped(
            vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            body,
        )
    }

    #[test]
    fn commutative_add_operand_order_canonicalized() {
        // `a + 1` and `1 + a` must canonicalize to the same IR.
        let p1 = scalar_out_prog(vec![Node::store(
            "out",
            E::u32(0),
            E::add(E::var("a"), E::u32(1)),
        )]);
        let p2 = scalar_out_prog(vec![Node::store(
            "out",
            E::u32(0),
            E::add(E::u32(1), E::var("a")),
        )]);
        let c1 = run(p1).to_wire().unwrap();
        let c2 = run(p2).to_wire().unwrap();
        assert_eq!(c1, c2, "canonicalize(a+1) == canonicalize(1+a)");
    }

    #[test]
    fn noncommutative_sub_order_preserved() {
        // `a - b` must NOT canonicalize to `b - a`; sub is non-commutative.
        let p1 = scalar_out_prog(vec![Node::store(
            "out",
            E::u32(0),
            E::sub(E::var("a"), E::var("b")),
        )]);
        let p2 = scalar_out_prog(vec![Node::store(
            "out",
            E::u32(0),
            E::sub(E::var("b"), E::var("a")),
        )]);
        let c1 = run(p1).to_wire().unwrap();
        let c2 = run(p2).to_wire().unwrap();
        assert_ne!(c1, c2, "sub must preserve operand order");
    }

    #[test]
    fn canonicalize_is_idempotent() {
        let body = vec![Node::store(
            "out",
            E::u32(0),
            E::add(E::u32(5), E::mul(E::var("b"), E::var("a"))),
        )];
        let p = scalar_out_prog(body);
        let once = run(p).to_wire().unwrap();
        let twice = run(Program::from_wire(&once).unwrap()).to_wire().unwrap();
        assert_eq!(once, twice, "canonicalize must be idempotent");
    }

    #[test]
    fn literal_hoists_to_right_on_commutative_ops() {
        // `3 + a` canonicalizes to `a + 3`.
        let p = scalar_out_prog(vec![Node::store(
            "out",
            E::u32(0),
            E::add(E::u32(3), E::var("a")),
        )]);
        let canonical = run(p);
        let body = crate::test_ir_inspect::region_body(&canonical);
        match &body[0] {
            Node::Store { value, .. } => match value {
                Expr::BinOp { left, right, .. } => {
                    assert!(
                        !is_literal(left),
                        "literal must not be on the left after canonicalize"
                    );
                    assert!(
                        is_literal(right),
                        "literal must be on the right after canonicalize"
                    );
                }
                _ => panic!("expected BinOp"),
            },
            other => panic!("expected Store, got {other:?}"),
        }
    }

    #[test]
    fn inner_binop_also_canonicalized() {
        // Nested commutative BinOps canonicalize recursively. The
        // inner mul's literal (2) must end up on the right of the
        // mul regardless of the outer add's layout. We find the
        // mul wherever canonicalize places it and assert that.
        let body = vec![Node::store(
            "out",
            E::u32(0),
            E::add(E::var("outer"), E::mul(E::u32(2), E::var("inner"))),
        )];
        let canonical = run(scalar_out_prog(body));

        // Walk the value tree and find the Mul; assert its literal
        // operand is on the right.
        fn find_mul_and_check(e: &Expr) {
            match e {
                Expr::BinOp {
                    op: BinOp::Mul,
                    left,
                    right,
                } => {
                    // Exactly one side is the literal; it must be
                    // the right after canonicalize.
                    let l_lit = is_literal(left);
                    let r_lit = is_literal(right);
                    assert!(l_lit ^ r_lit, "test expects exactly one literal operand");
                    assert!(
                        r_lit && !l_lit,
                        "literal must canonicalize to the right of Mul"
                    );
                }
                Expr::BinOp { left, right, .. } => {
                    find_mul_and_check(left);
                    find_mul_and_check(right);
                }
                _ => {}
            }
        }
        let entry_body = crate::test_ir_inspect::region_body(&canonical);
        match &entry_body[0] {
            Node::Store { value, .. } => find_mul_and_check(value),
            other => panic!("expected Store, got {other:?}"),
        }
    }

    #[test]
    fn eq_same_var_not_folded_by_canonicalize() {
        // canonicalize must NOT fold `a == a` to `true`. The pass is
        // type-blind, and for a float `a` that is NaN at runtime
        // `a == a` is *false* under IEEE-754 (the reference oracle's
        // `binop_f32` and every target emitter agree).
        // Folding it to a bool literal would miscompile NaN checks and
        // collide distinct programs in the content-addressed cache.
        // The Eq node is preserved verbatim (operand order unchanged
        // because both operands are the same Var).
        let p = scalar_out_prog(vec![Node::let_bind("t", E::eq(E::var("a"), E::var("a")))]);
        let canonical = run(p);
        let entry_body = crate::test_ir_inspect::region_body(&canonical);
        match &entry_body[0] {
            Node::Let { value, .. } => match value {
                Expr::BinOp {
                    op: BinOp::Eq,
                    left,
                    right,
                } => {
                    assert_eq!(left.as_ref(), &Expr::var("a"), "lhs preserved as Var(a)");
                    assert_eq!(right.as_ref(), &Expr::var("a"), "rhs preserved as Var(a)");
                }
                other => panic!("a == a must be preserved as an Eq BinOp, got {other:?}"),
            },
            other => panic!("expected Let, got {other:?}"),
        }
    }

    #[test]
    fn ne_same_var_not_folded_by_canonicalize() {
        // Symmetric to `eq_same_var_not_folded_by_canonicalize`: for a
        // float NaN, `a != a` is *true*, not false, so canonicalize must
        // not fold it to a bool literal. The Ne node is preserved.
        let p = scalar_out_prog(vec![Node::let_bind("t", E::ne(E::var("a"), E::var("a")))]);
        let canonical = run(p);
        let entry_body = crate::test_ir_inspect::region_body(&canonical);
        match &entry_body[0] {
            Node::Let { value, .. } => match value {
                Expr::BinOp {
                    op: BinOp::Ne,
                    left,
                    right,
                } => {
                    assert_eq!(left.as_ref(), &Expr::var("a"), "lhs preserved as Var(a)");
                    assert_eq!(right.as_ref(), &Expr::var("a"), "rhs preserved as Var(a)");
                }
                other => panic!("a != a must be preserved as an Ne BinOp, got {other:?}"),
            },
            other => panic!("expected Let, got {other:?}"),
        }
    }

    #[test]
    fn canonicalize_descends_into_subgroup_reduce_operand() {
        // The pass's contract is "semantically-equal Programs have byte-equal
        // wire output" (the content-addressed pipeline-cache fingerprint).
        // A commutative BinOp nested inside a subgroup operand must therefore
        // canonicalize just like a top-level one: `subgroup_add(1 + a)` and
        // `subgroup_add(a + 1)` are the same computation and must produce the
        // same wire bytes. If canonicalize_expr's catch-all swallows
        // SubgroupReduce, the inner `1 + a` stays un-normalized and the two
        // fingerprints diverge (cache fragmentation).
        let p1 = scalar_out_prog(vec![Node::store(
            "out",
            E::u32(0),
            E::subgroup_add(E::add(E::u32(1), E::var("a"))),
        )]);
        let p2 = scalar_out_prog(vec![Node::store(
            "out",
            E::u32(0),
            E::subgroup_add(E::add(E::var("a"), E::u32(1))),
        )]);
        let c1 = run(p1).to_wire().unwrap();
        let c2 = run(p2).to_wire().unwrap();
        assert_eq!(
            c1, c2,
            "canonicalize(subgroup_add(1 + a)) must equal canonicalize(subgroup_add(a + 1))"
        );
    }

    #[test]
    fn canonicalize_descends_into_call_args() {
        // Same contract for Call arguments: `f(1 + a)` and `f(a + 1)` are the
        // same computation and must canonicalize to the same wire bytes. A
        // catch-all that skips Call leaves the arg's commutative BinOp
        // un-normalized.
        let p1 = scalar_out_prog(vec![Node::store(
            "out",
            E::u32(0),
            E::call("f", vec![E::add(E::u32(1), E::var("a"))]),
        )]);
        let p2 = scalar_out_prog(vec![Node::store(
            "out",
            E::u32(0),
            E::call("f", vec![E::add(E::var("a"), E::u32(1))]),
        )]);
        let c1 = run(p1).to_wire().unwrap();
        let c2 = run(p2).to_wire().unwrap();
        assert_eq!(
            c1, c2,
            "canonicalize(f(1 + a)) must equal canonicalize(f(a + 1))"
        );
    }

    #[test]
    fn eq_different_vars_unchanged() {
        let p = scalar_out_prog(vec![Node::let_bind("t", E::eq(E::var("a"), E::var("b")))]);
        let canonical = run(p);
        let entry_body = crate::test_ir_inspect::region_body(&canonical);
        match &entry_body[0] {
            Node::Let { value, .. } => match value {
                Expr::BinOp {
                    op: BinOp::Eq,
                    left,
                    right,
                } => {
                    let l = left.as_ref();
                    let r = right.as_ref();
                    assert!(
                        (l == &Expr::var("a") && r == &Expr::var("b"))
                            || (l == &Expr::var("b") && r == &Expr::var("a"))
                    );
                }
                other => panic!("expected Eq BinOp, got {other:?}"),
            },
            other => panic!("expected Let, got {other:?}"),
        }
    }
}
