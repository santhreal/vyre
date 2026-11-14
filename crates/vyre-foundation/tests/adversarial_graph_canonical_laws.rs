//! Adversarial invariants for canonicalize and algebraic_law_registry.
//!
//! The programs come from `contract_cases::optimizer_program_corpus`, which
//! also owns the fixed-point-and-semantics scaffold.

use vyre_foundation::algebraic_law_registry::{
    has_law, is_commutative, laws_for_op, AlgebraicLaw, AlgebraicLawRegistration,
};
use vyre_foundation::ir::{BinOp, Expr, Node};
use vyre_foundation::optimizer::passes::algebraic::canonicalize_engine as canonicalize;

use crate::corpus;

use corpus::output_only_store;

inventory::submit! {
    AlgebraicLawRegistration::new("test::binop::add", AlgebraicLaw::Commutative)
}
inventory::submit! {
    AlgebraicLawRegistration::new("test::duplicate::commutative", AlgebraicLaw::Commutative)
}
inventory::submit! {
    AlgebraicLawRegistration::new("test::duplicate::commutative", AlgebraicLaw::Commutative)
}

fn canonicalized_store_value(expr: Expr) -> Expr {
    let canonical = canonicalize::run(output_only_store(expr));
    let first = canonical
        .entry()
        .first()
        .expect("Fix: store_program always produces one root region");
    let store = match first {
        Node::Region { body, .. } => body
            .first()
            .expect("Fix: store_program root region must contain one store node"),
        other => other,
    };
    match store {
        Node::Store { value, .. } => value.clone(),
        other => panic!("Fix: expected canonicalized store node, got {other:?}"),
    }
}

/// BinOps where canonicalize may sort non-literal operands without
/// changing semantics. Add/Mul are excluded because IEEE-754 float
/// NaN payload propagation makes them non-commutative at the bit level.
fn safe_to_sort_nonliterals_binops() -> [BinOp; 7] {
    [
        BinOp::BitAnd,
        BinOp::BitOr,
        BinOp::BitXor,
        BinOp::Eq,
        BinOp::Ne,
        BinOp::And,
        BinOp::Or,
    ]
}

fn operand_ordered_binops() -> [BinOp; 4] {
    [BinOp::Sub, BinOp::Div, BinOp::Shl, BinOp::Shr]
}

#[test]
fn commutative_binops_canonicalize_to_the_same_operand_order() {
    for op in safe_to_sort_nonliterals_binops() {
        let lhs = Expr::var("z");
        let rhs = Expr::var("a");
        let forward = canonicalized_store_value(Expr::BinOp {
            op,
            left: Box::new(lhs.clone()),
            right: Box::new(rhs.clone()),
        });
        let reversed = canonicalized_store_value(Expr::BinOp {
            op,
            left: Box::new(rhs),
            right: Box::new(lhs),
        });
        assert_eq!(
            forward, reversed,
            "Fix: canonicalize must sort operands for bitwise/boolean commutative BinOps; failed for {op:?}"
        );
    }
}

#[test]
fn add_mul_preserve_nonliteral_operand_order() {
    // IEEE-754 NaN payload propagation is not commutative for float
    // Add/Mul, so canonicalize must not reorder non-literal operands.
    for op in [BinOp::Add, BinOp::Mul] {
        let lhs = Expr::var("z");
        let rhs = Expr::var("a");
        let forward = canonicalized_store_value(Expr::BinOp {
            op,
            left: Box::new(lhs.clone()),
            right: Box::new(rhs.clone()),
        });
        let reversed = canonicalized_store_value(Expr::BinOp {
            op,
            left: Box::new(rhs),
            right: Box::new(lhs),
        });
        assert_ne!(
            forward, reversed,
            "Fix: canonicalize must NOT sort non-literal operands for {op:?} because IEEE-754 NaN payloads are not commutative"
        );
    }
}

#[test]
fn literals_are_hoisted_right_for_commutative_add() {
    let canonical = canonicalized_store_value(Expr::add(Expr::u32(1), Expr::var("x")));
    match canonical {
        Expr::BinOp {
            op: BinOp::Add,
            left,
            right,
        } => {
            assert!(
                !matches!(&*left, Expr::LitU32(_)),
                "Fix: canonicalize must hoist literal Add operands to the right"
            );
            assert!(
                matches!(&*right, Expr::LitU32(1)),
                "Fix: canonicalize must preserve the literal payload when hoisting it right"
            );
        }
        other => panic!("Fix: canonicalize(Add) must remain a BinOp, got {other:?}"),
    }
}

#[test]
fn non_commutative_binops_preserve_operand_order() {
    for op in operand_ordered_binops() {
        let canonical = canonicalized_store_value(Expr::BinOp {
            op,
            left: Box::new(Expr::var("lhs")),
            right: Box::new(Expr::var("rhs")),
        });
        match canonical {
            Expr::BinOp {
                op: actual,
                left,
                right,
            } => {
                assert_eq!(actual, op);
                assert_eq!(&*left, &Expr::var("lhs"));
                assert_eq!(&*right, &Expr::var("rhs"));
            }
            other => panic!("Fix: canonicalize({op:?}) must remain a BinOp, got {other:?}"),
        }
    }
}

#[test]
fn laws_for_unknown_op_returns_empty_vec() {
    assert!(
        laws_for_op("test::missing::law").is_empty(),
        "Fix: querying an unknown op id must return an empty law set"
    );
}

#[test]
fn has_law_is_idempotent_under_duplicate_registration() {
    let once = has_law("test::duplicate::commutative", |law| {
        matches!(law, AlgebraicLaw::Commutative)
    });
    let twice = has_law("test::duplicate::commutative", |law| {
        matches!(law, AlgebraicLaw::Commutative)
    });
    assert!(
        once,
        "Fix: duplicate registration must still satisfy has_law"
    );
    assert_eq!(
        once, twice,
        "Fix: duplicate registration must not make has_law nondeterministic"
    );
}

#[test]
fn is_commutative_distinguishes_add_from_sub() {
    assert!(
        is_commutative("test::binop::add"),
        "Fix: Add-style op ids registered as commutative must query true"
    );
    assert!(
        !is_commutative("test::binop::sub"),
        "Fix: unregistered Sub-style op ids must not be reported commutative"
    );
}
