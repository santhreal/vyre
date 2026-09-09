//! Contracts for `vyre_driver_wgpu::runtime::router`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

#![cfg(feature = "device-tests")]

use vyre_driver::backend_precedence;
use vyre_driver_wgpu::runtime::router::{BackendRouter, Override, Reason};
use vyre_foundation::ir::Program;

fn noop_program() -> Program {
    // Programs built without any buffers / nodes are valid for
    // the router's purposes  -  we don't dispatch, we just pick.
    Program::wrapped(Vec::new(), [1, 1, 1], Vec::new())
}

// `enumerate_by_precedence_puts_wgpu_before_reference` was here. It compared
// wgpu's rank against `backend_precedence("cpu-ref")` and passed when that
// call returned `u32::MAX`, which is what an unregistered id returns. The
// interpreter submits no registration, so the assertion could no longer fail
// on the thing its name claimed. What it meant to prove is now proven
// stronger and elsewhere: `vyre-driver-reference/tests/production_registry_execution_domain.rs`
// requires that no entry in the registry, and therefore no entry in the
// precedence order, executes on the host at all.

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
