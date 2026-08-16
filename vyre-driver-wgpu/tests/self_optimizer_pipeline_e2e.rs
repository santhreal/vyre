//! Multi-pass self-hosted optimizer pipeline running entirely on GPU.
//!
//! Composes `gpu_canonicalize → gpu_const_fold → gpu_dce` against the
//! same input Program through `WgpuBackend::dispatch`. Each pass
//! re-encodes its input and dispatches its own analysis Program. No
//! CPU optimizer pass runs at any point.
//!
//! The input programs and the surviving body each one owes are
//! `vyre_test_support::pass_programs::PIPELINE_CASES`, shared with the CUDA
//! suite. The three dispatches stay here, because that composition is what this
//! backend owes proof of.

#![cfg(test)]

mod harness;
use harness::acquire_live_backend as live_backend;
use harness::self_optimizer::WgpuProgramDispatcher;

use vyre_pass_engine::optimizer::canonicalize_via_encoded::gpu_canonicalize;
use vyre_pass_engine::optimizer::const_fold_via_encoded::gpu_const_fold;
use vyre_pass_engine::optimizer::dce_via_encoded::gpu_dce;
use vyre_test_support::pass_programs::{assert_pipeline_body, pipeline_case};

/// Run all three passes for the named case on the live GPU and assert the body
/// the case owes.
fn assert_case_on_real_gpu(label: &str) {
    let backend = live_backend();
    let dispatcher = WgpuProgramDispatcher::new(&backend);
    let case = pipeline_case(label);

    let p = gpu_canonicalize(case.input(), &dispatcher).expect("canonicalize dispatches");
    let p = gpu_const_fold(p, &dispatcher).expect("const-fold dispatches");
    let p = gpu_dce(p, &dispatcher).expect("dce dispatches");

    assert_pipeline_body("wgpu", case, &p);
}

#[test]
fn full_pipeline_canonicalize_then_const_fold_then_dce_on_real_gpu() {
    assert_case_on_real_gpu("dead_let_and_unfoldable_store");
}

#[test]
fn pipeline_collapses_unused_compute_chain_on_real_gpu() {
    assert_case_on_real_gpu("unused_compute_chain");
}
