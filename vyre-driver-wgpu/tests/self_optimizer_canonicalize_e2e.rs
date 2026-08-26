//! End-to-end test: vyre's canonicalize pass running as a vyre Program
//! on the GPU. The kernel marks each commutative `BinOp` whose
//! operands are (literal, non-literal) for swap; the decoder applies.
//!
//! V1 covers the load-bearing rewrite (literal-on-right). The
//! non-literal sort tie-break and `x == x` self-fold migrate as
//! follow-up kernels.
//!
//! The input programs and the value the pass owes for each are
//! `vyre_test_support::pass_programs::CANONICALIZE_CASES`, shared with the CUDA
//! suite so both backends assert the same rewrite. The dispatch stays here: a
//! pass proven through naga's WGSL is not proven through PTX.

#![cfg(all(test, feature = "device-tests"))]

mod harness;
use harness::acquire_live_backend as live_backend;
use harness::self_optimizer::semantic_execution;

use vyre_pass_engine::optimizer::canonicalize_via_encoded::gpu_canonicalize;
use vyre_test_support::pass_programs::{assert_canonicalized, canonicalize_case};

/// Dispatch canonicalize for the named case on the live GPU and assert the
/// rewrite the case owes.
fn assert_case_on_real_gpu(label: &str) {
    let backend = live_backend();
    let (executor, policy) = semantic_execution(&backend);
    let case = canonicalize_case(label);
    let canon = gpu_canonicalize(case.input(), &executor, &policy).expect("dispatches");
    assert_canonicalized("wgpu", case, &canon);
}

#[test]
fn canonicalize_lit_plus_var_swaps_to_var_plus_lit_on_real_gpu() {
    assert_case_on_real_gpu("lit_plus_var");
}

#[test]
fn canonicalize_var_plus_lit_unchanged_on_real_gpu() {
    assert_case_on_real_gpu("var_plus_lit");
}

#[test]
fn canonicalize_two_lits_unchanged_on_real_gpu() {
    // Both literals: no swap. The CPU canonicalize leaves these alone too for
    // ops without a tie-break.
    assert_case_on_real_gpu("two_lits");
}

#[test]
fn canonicalize_two_vars_unchanged_on_real_gpu() {
    // V1 does not tie-break non-literals, so this must stay as written.
    assert_case_on_real_gpu("two_vars");
}

#[test]
fn canonicalize_lit_times_var_swaps_on_real_gpu() {
    assert_case_on_real_gpu("lit_times_var");
}

#[test]
fn canonicalize_non_commutative_div_unchanged_on_real_gpu() {
    // Div is NOT commutative, so the literal must stay on the left whatever the
    // operands are.
    assert_case_on_real_gpu("non_commutative_div");
}
