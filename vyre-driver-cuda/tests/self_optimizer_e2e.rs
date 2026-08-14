//! End-to-end tests: vyre's self-hosted optimizer passes running as
//! vyre Programs on real CUDA hardware via `CudaBackend::dispatch`.
//!
//! Mirrors the wgpu E2E suites (DCE, const-fold, canonicalize,
//! pipeline). Confirms the ProgramDispatcher abstraction is
//! backend-agnostic  -  the same encoder + analysis Programs run
//! unchanged on both CUDA and wgpu paths.
//!
//! The canonicalize and pipeline inputs, and the shape each pass owes for them,
//! are `vyre_test_support::pass_programs`, shared with the wgpu suites so the two
//! backends cannot assert different rewrites of the same program. The dispatcher
//! and the DCE differential against the foundation CPU oracle stay here.

#![cfg(test)]

mod common;

use common::{live_backend, CudaProgramDispatcher};
use vyre::ir::{BinOp, Expr, Node};
use vyre_foundation::optimizer::fingerprint_program;
use vyre_foundation::optimizer::passes::fusion_cse::dce::engine::dce as cpu_dce_oracle;
use vyre_pass_engine::optimizer::canonicalize_via_encoded::gpu_canonicalize;
use vyre_pass_engine::optimizer::const_fold_via_encoded::gpu_const_fold;
use vyre_pass_engine::optimizer::dce_via_encoded::gpu_dce;
use vyre_test_support::pass_programs::{
    assert_canonicalized, assert_pipeline_body, canonicalize_case, first_let_value, pipeline_case,
    wrapped,
};

// ---- DCE on CUDA -----------------------------------------------------------

fn assert_dce_matches_cpu_oracle_cuda(entry: Vec<Node>) {
    let backend = live_backend();
    let dispatcher = CudaProgramDispatcher { backend: &backend };

    let oracle_in = wrapped(entry.clone());
    let test_in = wrapped(entry);

    let oracle_out = cpu_dce_oracle(oracle_in);
    let gpu_out = gpu_dce(test_in, &dispatcher).expect("gpu_dce dispatches through cuda cleanly");
    assert_eq!(
        fingerprint_program(&oracle_out),
        fingerprint_program(&gpu_out),
        "CUDA-dispatched DCE must match the foundation CPU oracle. \
         oracle entry={:?} gpu entry={:?}",
        oracle_out.entry(),
        gpu_out.entry()
    );
}

#[test]
fn cuda_dce_dead_let_dropped() {
    assert_dce_matches_cpu_oracle_cuda(vec![Node::let_bind("dead", Expr::u32(7))]);
}

#[test]
fn cuda_dce_live_let_kept() {
    assert_dce_matches_cpu_oracle_cuda(vec![
        Node::let_bind("x", Expr::u32(7)),
        Node::store("buf", Expr::u32(0), Expr::var("x")),
    ]);
}

#[test]
fn cuda_dce_chained_lets_propagate() {
    assert_dce_matches_cpu_oracle_cuda(vec![
        Node::let_bind("a", Expr::u32(1)),
        Node::let_bind("b", Expr::var("a")),
        Node::store("buf", Expr::u32(0), Expr::var("b")),
    ]);
}

#[test]
fn cuda_dce_loop_with_induction_var() {
    assert_dce_matches_cpu_oracle_cuda(vec![Node::loop_for(
        "i",
        Expr::u32(0),
        Expr::u32(10),
        vec![Node::store("buf", Expr::var("i"), Expr::u32(0))],
    )]);
}

// ---- Const-fold on CUDA ----------------------------------------------------

#[test]
fn cuda_const_fold_two_plus_three_yields_lit_five() {
    let backend = live_backend();
    let dispatcher = CudaProgramDispatcher { backend: &backend };

    let p = wrapped(vec![Node::let_bind(
        "x",
        Expr::add(Expr::u32(2), Expr::u32(3)),
    )]);
    let folded = gpu_const_fold(p, &dispatcher).expect("dispatches");
    let got = first_let_value(&folded);
    assert!(
        matches!(got, Expr::LitU32(5)),
        "CUDA const-fold must compute 2 + 3 = 5; got {got:?}"
    );
}

#[test]
fn cuda_const_fold_chained_arithmetic() {
    let backend = live_backend();
    let dispatcher = CudaProgramDispatcher { backend: &backend };

    let p = wrapped(vec![Node::let_bind(
        "x",
        Expr::mul(Expr::add(Expr::u32(2), Expr::u32(3)), Expr::u32(4)),
    )]);
    let folded = gpu_const_fold(p, &dispatcher).expect("dispatches");
    let got = first_let_value(&folded);
    assert!(matches!(got, Expr::LitU32(20)));
}

#[test]
fn cuda_const_fold_bitwise_ops() {
    let backend = live_backend();
    let dispatcher = CudaProgramDispatcher { backend: &backend };

    let p = wrapped(vec![Node::let_bind(
        "x",
        Expr::bitand(
            Expr::bitor(Expr::u32(0xFF), Expr::u32(0x100)),
            Expr::u32(0x1FF),
        ),
    )]);
    let folded = gpu_const_fold(p, &dispatcher).expect("dispatches");
    let got = first_let_value(&folded);
    assert!(matches!(got, Expr::LitU32(0x1FF)));
}

#[test]
fn cuda_const_fold_unfoldable_var_passes_through() {
    let backend = live_backend();
    let dispatcher = CudaProgramDispatcher { backend: &backend };

    let p = wrapped(vec![Node::let_bind(
        "x",
        Expr::add(Expr::var("a"), Expr::u32(2)),
    )]);
    let folded = gpu_const_fold(p, &dispatcher).expect("dispatches");
    let got = first_let_value(&folded);
    match got {
        Expr::BinOp { op, .. } => assert!(matches!(op, BinOp::Add)),
        other => panic!("expected unchanged Add; got {other:?}"),
    }
}

// ---- Canonicalize on CUDA --------------------------------------------------

/// Dispatch canonicalize for the named case on the live device and assert the
/// rewrite the case owes.
fn assert_canonicalize_case(label: &str) {
    let backend = live_backend();
    let dispatcher = CudaProgramDispatcher { backend: &backend };
    let case = canonicalize_case(label);
    let canon = gpu_canonicalize(case.input(), &dispatcher).expect("dispatches");
    assert_canonicalized("cuda", case, &canon);
}

#[test]
fn cuda_canonicalize_lit_plus_var_swaps_to_var_plus_lit() {
    assert_canonicalize_case("lit_plus_var");
}

#[test]
fn cuda_canonicalize_var_plus_lit_unchanged() {
    assert_canonicalize_case("var_plus_lit");
}

#[test]
fn cuda_canonicalize_non_commutative_div_unchanged() {
    assert_canonicalize_case("non_commutative_div");
}

// ---- Multi-pass pipeline on CUDA ------------------------------------------

/// Run all three passes for the named case on the live device and assert the
/// body the case owes.
fn assert_pipeline_case(label: &str) {
    let backend = live_backend();
    let dispatcher = CudaProgramDispatcher { backend: &backend };
    let case = pipeline_case(label);

    let p = gpu_canonicalize(case.input(), &dispatcher).expect("canonicalize dispatches");
    let p = gpu_const_fold(p, &dispatcher).expect("const-fold dispatches");
    let p = gpu_dce(p, &dispatcher).expect("dce dispatches");

    assert_pipeline_body("cuda", case, &p);
}

#[test]
fn cuda_full_pipeline_canonicalize_then_const_fold_then_dce() {
    assert_pipeline_case("dead_let_and_unfoldable_store");
}

#[test]
fn cuda_pipeline_collapses_unused_compute_chain() {
    assert_pipeline_case("unused_compute_chain");
}
