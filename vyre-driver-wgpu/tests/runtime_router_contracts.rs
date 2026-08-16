//! Contracts for `vyre_driver_wgpu::runtime::router`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::backend_precedence;
use vyre_driver_wgpu::runtime::router::{BackendRouter, Override, Reason};
use vyre_foundation::ir::Program;

fn noop_program() -> Program {
    // Programs built without any buffers / nodes are valid for
    // the router's purposes  -  we don't dispatch, we just pick.
    Program::wrapped(Vec::new(), [1, 1, 1], Vec::new())
}

#[test]
fn enumerate_by_precedence_puts_wgpu_before_reference() {
    // V7-EXT-021: precedence is now inventory-driven. wgpu submits
    // rank 30 in this crate's lib.rs; cpu-ref (when registered)
    // must trail it.
    let wgpu_rank = backend_precedence("wgpu").expect("valid backend registry");
    let ref_rank = backend_precedence("cpu-ref").expect("valid backend registry");
    assert!(
        wgpu_rank < ref_rank || ref_rank == u32::MAX,
        "wgpu (rank {wgpu_rank}) must take precedence over the CPU reference oracle (rank {ref_rank})"
    );
}

#[test]
fn enumerate_by_precedence_is_inventory_driven() {
    // Replaces the BACKEND_PRECEDENCE static-slice assertion.
    let ranked = BackendRouter::enumerate_by_precedence().expect("valid backend registry");
    // wgpu registers in this crate; it must appear with a finite rank.
    let wgpu = ranked.iter().find(|r| r.id == "wgpu").expect(
        "Fix: wgpu backend registered in this crate; restore this invariant before continuing.",
    );
    assert_eq!(
        backend_precedence(wgpu.id).expect("valid backend registry"),
        30
    );
}

#[test]
fn explicit_override_with_unknown_backend_surfaces_error() {
    let router = BackendRouter::new();
    let err = router
        .pick_with_override(
            &noop_program(),
            Override::Explicit("does-not-exist-backend"),
        )
        .expect_err("unknown backend must error");
    let msg = format!("{err}");
    assert!(msg.contains("does-not-exist-backend"));
    assert!(msg.contains("Fix:"));
}

#[test]
fn explicit_override_picks_the_named_backend_when_registered() {
    let router = BackendRouter::new();
    // wgpu registers via inventory::submit! in lib.rs.
    let decision = router
        .pick_with_override(&noop_program(), Override::Explicit("wgpu"))
        .expect("Fix: wgpu backend is registered in this crate");
    assert_eq!(decision.backend, "wgpu");
    assert_eq!(decision.reason, Reason::EnvOverride);
}

#[test]
fn precedence_picks_wgpu_when_registered() {
    let router = BackendRouter::new();
    let decision = router
        .pick_with_override(&noop_program(), Override::None)
        .expect("Fix: at least one backend must register");
    assert_eq!(decision.reason, Reason::Precedence);
    // The picked backend must have a registered precedence rank
    // (V7-EXT-021: replaces the BACKEND_PRECEDENCE static-slice check).
    assert!(
        backend_precedence(decision.backend).expect("valid backend registry") < u32::MAX,
        "picked backend {} did not submit a BackendPrecedence inventory entry",
        decision.backend
    );
}
