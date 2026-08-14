//! End-to-end test: vyre's const-fold pass running as a vyre Program
//! (compute kernel) on the GPU through `WgpuBackend::dispatch`.
//!
//! Unlike DCE which composes existing graph primitives, const-fold
//! ships a brand-new analysis Program written in vyre IR. The kernel
//! does a sequential bottom-up scan over the encoded Expr arena,
//! marking foldable Exprs and computing their u32 values. The decoder
//! rewrites the IR. This is the architectural proof that compute
//! kernels  -  not just graph-primitive reductions  -  run as vyre
//! Programs on real hardware.
//!
//! V1 op coverage: literals + BinOp::{Add, Sub, Mul, BitAnd, BitOr,
//! BitXor} on u32. Other arithmetic / typed ops extend mechanically.

#![cfg(test)]

mod common;
use common::acquire_live_backend as live_backend;
use common::self_optimizer::{WgpuOptimizerDispatcher, first_let_value, wrapped};

use vyre::ir::{Expr, Node};
use vyre_self_substrate::optimizer::const_fold_via_encoded::gpu_const_fold;




#[test]
fn const_fold_two_plus_three_yields_lit_five_on_real_gpu() {
    let backend = live_backend();
    let dispatcher = WgpuOptimizerDispatcher::new(&backend);

    // let x = 2 + 3
    let p = wrapped(vec![Node::let_bind(
        "x",
        Expr::add(Expr::u32(2), Expr::u32(3)),
    )]);
    let folded = gpu_const_fold(p, &dispatcher).expect("const-fold dispatches cleanly");
    let got = first_let_value(&folded);
    assert!(
        matches!(got, Expr::LitU32(5)),
        "GPU const-fold must compute 2 + 3 = 5; got {got:?}"
    );
}

#[test]
fn const_fold_chained_arithmetic_on_real_gpu() {
    let backend = live_backend();
    let dispatcher = WgpuOptimizerDispatcher::new(&backend);

    // let x = (2 + 3) * 4   →   20
    let p = wrapped(vec![Node::let_bind(
        "x",
        Expr::mul(Expr::add(Expr::u32(2), Expr::u32(3)), Expr::u32(4)),
    )]);
    let folded = gpu_const_fold(p, &dispatcher).expect("dispatches cleanly");
    let got = first_let_value(&folded);
    assert!(
        matches!(got, Expr::LitU32(20)),
        "GPU const-fold must compute (2+3)*4 = 20; got {got:?}"
    );
}

#[test]
fn const_fold_subtraction_on_real_gpu() {
    let backend = live_backend();
    let dispatcher = WgpuOptimizerDispatcher::new(&backend);

    // let x = 10 - 7   →   3
    let p = wrapped(vec![Node::let_bind(
        "x",
        Expr::sub(Expr::u32(10), Expr::u32(7)),
    )]);
    let folded = gpu_const_fold(p, &dispatcher).expect("dispatches cleanly");
    let got = first_let_value(&folded);
    assert!(matches!(got, Expr::LitU32(3)));
}

#[test]
fn const_fold_bitwise_ops_on_real_gpu() {
    let backend = live_backend();
    let dispatcher = WgpuOptimizerDispatcher::new(&backend);

    // (0xFF | 0x100) & 0x1FF   →   0x1FF
    let p = wrapped(vec![Node::let_bind(
        "x",
        Expr::bitand(
            Expr::bitor(Expr::u32(0xFF), Expr::u32(0x100)),
            Expr::u32(0x1FF),
        ),
    )]);
    let folded = gpu_const_fold(p, &dispatcher).expect("dispatches cleanly");
    let got = first_let_value(&folded);
    assert!(matches!(got, Expr::LitU32(0x1FF)));
}

#[test]
fn const_fold_unfoldable_var_passes_through_on_real_gpu() {
    let backend = live_backend();
    let dispatcher = WgpuOptimizerDispatcher::new(&backend);

    // let x = a + 2  →  unchanged (a is not foldable).
    let p = wrapped(vec![Node::let_bind(
        "x",
        Expr::add(Expr::var("a"), Expr::u32(2)),
    )]);
    let folded = gpu_const_fold(p, &dispatcher).expect("dispatches cleanly");
    let got = first_let_value(&folded);
    // Top-level Expr is still the Add (not foldable because Var(a) isn't).
    match got {
        Expr::BinOp { op, .. } => {
            assert!(matches!(op, vyre::ir::BinOp::Add));
        }
        other => panic!("expected unchanged Add; got {other:?}"),
    }
}
