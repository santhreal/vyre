//! Negative control for the shared driver contracts.
//!
//! A shared assertion helper is only worth having if it can fail. Both contracts
//! this crate includes take their expectations as arguments, so the way to prove
//! they discriminate is to hand them an expectation that does not describe this
//! backend and require the assertion to reject it. Without this, a helper that
//! silently accepted every argument would pass for all four backends and prove
//! nothing about any of them.
//!
//! SPIR-V is the backend used here because its target compiler is pure: these
//! cases reach the payload format and the registry without acquiring a device.

use std::panic::{catch_unwind, AssertUnwindSafe};

use vyre_driver_spirv::SpirvBackend;

mod target_artifacts;
use target_artifacts::spirv as truthful;
use target_artifacts::target_compiler_contract::{
    assert_target_compiler_emits_bundle, TargetExpectation,
};

#[path = "../../tests/support/preferred_dispatch_backend_contract.rs"]
mod preferred_dispatch_contract;
use preferred_dispatch_contract::assert_backend_registry_metadata;

/// Require a contract call to fail, and name what the falsified expectation was.
fn must_reject(falsified: &str, case: impl FnOnce()) {
    let outcome = catch_unwind(AssertUnwindSafe(case));
    assert!(
        outcome.is_err(),
        "Fix: the shared contract accepted a {falsified} it should have rejected, so it cannot \
         detect that failure for any backend that calls it"
    );
}

/// WHY: the payload format identity is what admission compares a payload against.
/// A contract that ignored it would let a backend advertise one dialect and ship
/// another.
#[test]
fn shared_target_contract_rejects_a_wrong_payload_format_identity() {
    must_reject("payload format identity", || {
        assert_target_compiler_emits_bundle(
            &TargetExpectation {
                format_identity: "not-spv",
                ..truthful()
            },
            |_| {},
        );
    });
}

/// WHY: the format version is how a payload built by an older compiler is kept
/// out of a newer materializer.
#[test]
fn shared_target_contract_rejects_a_wrong_payload_format_version() {
    must_reject("payload format version", || {
        assert_target_compiler_emits_bundle(
            &TargetExpectation {
                format_version: 99,
                ..truthful()
            },
            |_| {},
        );
    });
}

/// WHY: entry point agreement between the emitted module and the payload entry is
/// what makes the dispatch grid describe the kernel that runs.
#[test]
fn shared_target_contract_rejects_a_wrong_entry_point() {
    must_reject("entry point", || {
        assert_target_compiler_emits_bundle(
            &TargetExpectation {
                entry_point: "not_main",
                ..truthful()
            },
            |_| {},
        );
    });
}

/// WHY: the contract must not pass for a backend that is not linked at all,
/// which is the failure mode a registry lookup silently swallows when the
/// assertion only inspects what it found.
#[test]
fn shared_target_contract_rejects_an_unlinked_backend() {
    must_reject("unlinked backend id", || {
        assert_target_compiler_emits_bundle(
            &TargetExpectation {
                backend_id: "backend-that-is-not-linked",
                ..truthful()
            },
            |_| {},
        );
    });
}

/// WHY: precedence decides which linked backend a release dispatch selects, so a
/// registry-metadata contract that accepted any rank would let a reordering ship.
#[test]
fn shared_registry_metadata_contract_rejects_a_wrong_precedence() {
    must_reject("precedence rank", || {
        assert_backend_registry_metadata::<SpirvBackend>(
            vyre_driver_spirv::SPIRV_BACKEND_ID,
            u32::MAX,
            "dispatch capability",
            "precedence rank",
        );
    });
}

/// WHY: the truthful expectation must still pass, or the cases above would be
/// satisfied by a contract that rejects everything.
#[test]
fn shared_contracts_accept_the_truthful_expectation() {
    assert_target_compiler_emits_bundle(&truthful(), |_| {});
    assert_backend_registry_metadata::<SpirvBackend>(
        vyre_driver_spirv::SPIRV_BACKEND_ID,
        30,
        "Fix: SPIR-V must register dispatches=true",
        "Fix: SPIR-V must keep precedence rank 30",
    );
}
