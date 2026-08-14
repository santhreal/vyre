//! End-to-end test: vyre's canonicalize pass running as a vyre Program
//! on the GPU. The kernel marks each commutative `BinOp` whose
//! operands are (literal, non-literal) for swap; the decoder applies.
//!
//! V1 covers the load-bearing rewrite (literal-on-right). The
//! non-literal sort tie-break and `x == x` self-fold migrate as
//! follow-up kernels.

#![cfg(test)]

mod common;
use common::acquire_live_backend as live_backend;
use common::self_optimizer::{first_let_value, wrapped, WgpuProgramDispatcher};

use vyre::ir::{BinOp, Expr, Node};
use vyre_pass_engine::optimizer::canonicalize_via_encoded::gpu_canonicalize;

#[test]
fn canonicalize_lit_plus_var_swaps_to_var_plus_lit_on_real_gpu() {
    let backend = live_backend();
    let dispatcher = WgpuProgramDispatcher::new(&backend);

    // let x = 1 + a   →   let x = a + 1   (literal on right)
    let p = wrapped(vec![Node::let_bind(
        "x",
        Expr::add(Expr::u32(1), Expr::var("a")),
    )]);
    let canon = gpu_canonicalize(p, &dispatcher).expect("dispatches");
    let got = first_let_value(&canon);
    match got {
        Expr::BinOp { op, left, right } => {
            assert!(matches!(op, BinOp::Add));
            assert!(
                matches!(*left, Expr::Var(ref n) if n.as_str() == "a"),
                "left must be Var(a) after canonicalize, got {left:?}"
            );
            assert!(
                matches!(*right, Expr::LitU32(1)),
                "right must be LitU32(1), got {right:?}"
            );
        }
        other => panic!("expected BinOp Add, got {other:?}"),
    }
}

#[test]
fn canonicalize_var_plus_lit_unchanged_on_real_gpu() {
    let backend = live_backend();
    let dispatcher = WgpuProgramDispatcher::new(&backend);

    // let x = a + 1   →   unchanged (already canonical)
    let p = wrapped(vec![Node::let_bind(
        "x",
        Expr::add(Expr::var("a"), Expr::u32(1)),
    )]);
    let canon = gpu_canonicalize(p, &dispatcher).expect("dispatches");
    let got = first_let_value(&canon);
    match got {
        Expr::BinOp { left, right, .. } => {
            assert!(matches!(*left, Expr::Var(ref n) if n.as_str() == "a"));
            assert!(matches!(*right, Expr::LitU32(1)));
        }
        other => panic!("expected unchanged BinOp Add, got {other:?}"),
    }
}

#[test]
fn canonicalize_two_lits_unchanged_on_real_gpu() {
    let backend = live_backend();
    let dispatcher = WgpuProgramDispatcher::new(&backend);

    // Both literals  -  no swap (CPU canonicalize also leaves these
    // alone for non-tie-breaking ops).
    let p = wrapped(vec![Node::let_bind(
        "x",
        Expr::add(Expr::u32(2), Expr::u32(3)),
    )]);
    let canon = gpu_canonicalize(p, &dispatcher).expect("dispatches");
    let got = first_let_value(&canon);
    match got {
        Expr::BinOp { left, right, .. } => {
            assert!(matches!(*left, Expr::LitU32(2)));
            assert!(matches!(*right, Expr::LitU32(3)));
        }
        other => panic!("expected unchanged BinOp, got {other:?}"),
    }
}

#[test]
fn canonicalize_two_vars_unchanged_on_real_gpu() {
    let backend = live_backend();
    let dispatcher = WgpuProgramDispatcher::new(&backend);

    // V1 doesn't tie-break non-literals → unchanged.
    let p = wrapped(vec![Node::let_bind(
        "x",
        Expr::add(Expr::var("a"), Expr::var("b")),
    )]);
    let canon = gpu_canonicalize(p, &dispatcher).expect("dispatches");
    let got = first_let_value(&canon);
    match got {
        Expr::BinOp { left, right, .. } => {
            assert!(matches!(*left, Expr::Var(ref n) if n.as_str() == "a"));
            assert!(matches!(*right, Expr::Var(ref n) if n.as_str() == "b"));
        }
        other => panic!("expected unchanged BinOp, got {other:?}"),
    }
}

#[test]
fn canonicalize_lit_times_var_swaps_on_real_gpu() {
    let backend = live_backend();
    let dispatcher = WgpuProgramDispatcher::new(&backend);

    // let x = 5 * a   →   let x = a * 5
    let p = wrapped(vec![Node::let_bind(
        "x",
        Expr::mul(Expr::u32(5), Expr::var("a")),
    )]);
    let canon = gpu_canonicalize(p, &dispatcher).expect("dispatches");
    let got = first_let_value(&canon);
    match got {
        Expr::BinOp { op, left, right } => {
            assert!(matches!(op, BinOp::Mul));
            assert!(matches!(*left, Expr::Var(ref n) if n.as_str() == "a"));
            assert!(matches!(*right, Expr::LitU32(5)));
        }
        other => panic!("expected BinOp Mul, got {other:?}"),
    }
}

#[test]
fn canonicalize_non_commutative_div_unchanged_on_real_gpu() {
    let backend = live_backend();
    let dispatcher = WgpuProgramDispatcher::new(&backend);

    // Div is NOT commutative  -  must NEVER swap regardless of operands.
    let p = wrapped(vec![Node::let_bind(
        "x",
        Expr::div(Expr::u32(10), Expr::var("a")),
    )]);
    let canon = gpu_canonicalize(p, &dispatcher).expect("dispatches");
    let got = first_let_value(&canon);
    match got {
        Expr::BinOp { op, left, right } => {
            assert!(matches!(op, BinOp::Div));
            // Left stays literal (not swapped  -  Div is non-commutative).
            assert!(matches!(*left, Expr::LitU32(10)));
            assert!(matches!(*right, Expr::Var(ref n) if n.as_str() == "a"));
        }
        other => panic!("expected BinOp Div unchanged, got {other:?}"),
    }
}
