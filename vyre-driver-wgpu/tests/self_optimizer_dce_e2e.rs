//! End-to-end test: vyre's DCE pass running as a vyre Program on the
//! GPU through the canonical `WgpuBackend::dispatch` API.
//!
//! No CPU fallback. The test wires a `WgpuProgramDispatcher` that
//! satisfies the `vyre_foundation::program_dispatch::ProgramDispatcher`
//! trait and calls `gpu_dce`. Result is asserted fingerprint-equal to
//! the foundation CPU `dce` pass on the same input  -  proving the
//! self-hosted GPU pass is semantically identical, with the substrate
//! actually running on hardware.

#![cfg(feature = "device-tests")]
#![cfg(test)]

mod harness;
use harness::acquire_live_backend as live_backend;
use harness::self_optimizer::{wrapped, WgpuProgramDispatcher};

use vyre::ir::{Expr, Node};
use vyre_foundation::optimizer::fingerprint_program;
use vyre_foundation::optimizer::passes::fusion_cse::dce::dce as cpu_dce_oracle;
use vyre_pass_engine::optimizer::dce_via_encoded::gpu_dce;

fn assert_gpu_dce_matches_cpu_oracle(entry: Vec<Node>) {
    let backend = live_backend();
    let dispatcher = WgpuProgramDispatcher::new(&backend);

    let oracle_in = wrapped(entry.clone());
    let test_in = wrapped(entry);

    let oracle_out = cpu_dce_oracle(oracle_in);
    let gpu_out = gpu_dce(test_in, &dispatcher).expect("gpu_dce dispatches through wgpu cleanly");
    assert_eq!(
        fingerprint_program(&oracle_out),
        fingerprint_program(&gpu_out),
        "GPU-dispatched DCE must match the foundation CPU oracle. \
         oracle entry={:?} gpu entry={:?}",
        oracle_out.entry(),
        gpu_out.entry()
    );
}

#[test]
fn dce_dead_let_dropped_on_real_gpu() {
    assert_gpu_dce_matches_cpu_oracle(vec![Node::let_bind("dead", Expr::u32(7))]);
}

#[test]
fn dce_live_let_kept_on_real_gpu() {
    assert_gpu_dce_matches_cpu_oracle(vec![
        Node::let_bind("x", Expr::u32(7)),
        Node::store("buf", Expr::u32(0), Expr::var("x")),
    ]);
}

#[test]
fn dce_chained_lets_propagate_on_real_gpu() {
    assert_gpu_dce_matches_cpu_oracle(vec![
        Node::let_bind("a", Expr::u32(1)),
        Node::let_bind("b", Expr::var("a")),
        Node::store("buf", Expr::u32(0), Expr::var("b")),
    ]);
}

#[test]
fn dce_unused_chain_dropped_on_real_gpu() {
    assert_gpu_dce_matches_cpu_oracle(vec![
        Node::let_bind("a", Expr::u32(1)),
        Node::let_bind("b", Expr::var("a")),
        Node::let_bind("c", Expr::u32(2)),
        Node::store("buf", Expr::u32(0), Expr::var("c")),
    ]);
}

#[test]
fn dce_if_branch_with_dead_lets_on_real_gpu() {
    assert_gpu_dce_matches_cpu_oracle(vec![Node::If {
        cond: Expr::var("c"),
        then: vec![Node::let_bind("dead_then", Expr::u32(0))],
        otherwise: vec![Node::let_bind("dead_else", Expr::u32(0))],
    }]);
}

#[test]
fn dce_loop_with_induction_var_on_real_gpu() {
    assert_gpu_dce_matches_cpu_oracle(vec![Node::loop_for(
        "i",
        Expr::u32(0),
        Expr::u32(10),
        vec![Node::store("buf", Expr::var("i"), Expr::u32(0))],
    )]);
}
