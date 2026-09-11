//! No entry in the production backend registry executes on the host.
//!
//! WHY: the reference interpreter used to submit a `BackendRegistration` with
//! id `cpu-ref`, which made a CPU execution route dispatchable, selectable and
//! indistinguishable from a device at the seam. Deleting that registration is
//! only half the fix. The other half is a rule that fails when someone adds it
//! back, and the rules that claimed to be that rule could not:
//!
//! - The registry loop in `backend_registration.rs` ran inside
//!   `if let Ok(registrations)`, so a registry that failed to freeze skipped
//!   every assertion, and it rejected ids by substring, so a host path
//!   registered as `host-interp` passed.
//! - The same substring check in `vyre-registry-link` read the real registry
//!   and admitted the same ids.
//! - Neither could observe a registration at all from a binary that links no
//!   driver crate, which is what `vyre-driver-reference` and `vyre-driver` are.
//!
//! So this suite links `vyre-registry-link` with every declared driver and
//! reads the registry those crates submit into. `live_backend_registry` asserts
//! its own per-source floor first, so the set enumerated below is the whole
//! production registry rather than whatever survived a dropped object file, and
//! `vyre_test_support::backend_execution_domain` decides each entry through an
//! exhaustive match with no catch-all arm.
//!
//! The reference oracle is still reachable from this crate, through
//! `CpuRefEvaluator` and `ReferenceSemanticExecutor`. That is the point: the
//! oracle seam is a named API, not a registry id, so nothing that enumerates
//! backends can return it.

use std::collections::BTreeSet;

use vyre_driver::{acquire, registered_backends, registered_backends_by_precedence_slice};
use vyre_registry_link::backend::{live_backend_registry, DECLARED_SOURCES};
use vyre_test_support::backend_execution_domain::{
    assert_dispatch_leaves_the_host, decision_for, ExecutionDomain, ProductionBackend,
};

/// Names a caller can spell when it wants the interpreter. None of them may
/// resolve to a backend.
///
/// The oracle's own id, plus the legacy `cpu-ref` label recorded evidence
/// still carries and the bare `cpu` nothing routes but a caller reaches for.
/// Deriving the first from the owning constant is what keeps this judging the
/// whole set: a change to the oracle id becomes a name this suite requires the
/// registry to refuse, without an edit here.
fn oracle_names() -> Vec<&'static str> {
    vec![vyre_driver_reference::ORACLE_EXECUTOR_ID, "cpu-ref", "cpu"]
}

fn registry_ids() -> BTreeSet<&'static str> {
    live_backend_registry()
        .expect("Fix: the production backend registry must freeze cleanly")
        .iter()
        .map(|registration| registration.id)
        .collect()
}

/// The structural half: the ledger's crate set is the declared driver set.
///
/// `DECLARED_SOURCES` is judged against the tree at run time by
/// `vyre-registry-link`'s own rules, so a workspace member that starts
/// submitting a `BackendRegistration` has to appear there, and then it has to
/// appear here with a recorded domain.
#[test]
fn every_declared_driver_crate_carries_an_execution_domain_decision() {
    let declared: BTreeSet<&str> = DECLARED_SOURCES.iter().copied().collect();
    let decided: BTreeSet<&str> = ProductionBackend::driver_crates().into_iter().collect();

    assert_eq!(
        declared, decided,
        "Fix: `vyre_test_support::backend_execution_domain::ProductionBackend` and \
         `vyre_registry_link::backend::DECLARED_SOURCES` name different driver crates. Add the \
         missing variant and record its execution domain, or drop the variant whose crate no \
         longer submits a registration."
    );
}

/// The behavioral half: every entry the production registry actually reports.
#[test]
fn no_backend_in_the_production_registry_executes_on_the_host() {
    let registry =
        live_backend_registry().expect("Fix: the production backend registry must freeze cleanly");
    assert!(
        !registry.is_empty(),
        "Fix: this binary links `vyre-registry-link` with every declared driver, so the registry \
         it reads cannot be empty. An empty registry makes every assertion below vacuous."
    );

    for registration in registry {
        assert_dispatch_leaves_the_host(registration.id);
        assert!(
            !registration.reference_oracle,
            "Fix: backend `{}` sets `reference_oracle`, so a conformance oracle is in the \
             production registry. Reach the oracle through `CpuRefEvaluator` or \
             `ReferenceSemanticExecutor` instead of registering it.",
            registration.id
        );
    }
}

/// A registry read taken through `vyre-driver` rather than the linkage owner
/// reports the same set, so the closure is not a property of one accessor.
#[test]
fn the_precedence_order_and_the_driver_accessor_report_the_decided_set() {
    let expected = registry_ids();
    let flat: BTreeSet<&'static str> = registered_backends()
        .expect("Fix: the production backend registry must freeze cleanly")
        .iter()
        .map(|registration| registration.id)
        .collect();
    let ranked: BTreeSet<&'static str> = registered_backends_by_precedence_slice()
        .expect("Fix: the production backend registry must freeze cleanly")
        .iter()
        .map(|registration| registration.id)
        .collect();

    assert_eq!(
        expected, flat,
        "Fix: `registered_backends` and the linkage owner's registry read disagree about which \
         backends are linked, so one of them is judging a partial set."
    );
    assert_eq!(
        expected, ranked,
        "Fix: precedence ordering added or dropped a backend. A selection route that ranks a \
         different set than the registry holds can return a backend nothing judged."
    );
}

/// No decision may record a host domain, whatever the registry happens to hold
/// on this build target.
///
/// `metal` compiles its registration out on a non-Apple host, so the loop above
/// never judges it there. This one does, because the decision is a fact about
/// the crate rather than about this target.
#[test]
fn no_recorded_decision_admits_host_execution() {
    for backend in ProductionBackend::ALL {
        assert_eq!(
            backend.execution_domain(),
            ExecutionDomain::Device,
            "Fix: `{}` is recorded as executing on the host while still submitting a \
             `BackendRegistration`. A host evaluator reachable by backend id is a CPU execution \
             route through the production dispatch seam. Delete the registration.",
            backend.driver_crate()
        );
    }
}

/// The name a caller reaches for when it wants the interpreter resolves to
/// nothing, in the registry and in acquisition.
#[test]
fn no_oracle_name_resolves_to_a_backend() {
    let present = registry_ids();
    for name in oracle_names() {
        assert!(
            !present.contains(name),
            "Fix: `{name}` is in the production backend registry. The interpreter is an oracle \
             reached by a named API, not a dispatch target."
        );
        assert!(
            decision_for(name).is_err(),
            "Fix: `{name}` carries an execution-domain decision, which records the interpreter as \
             a production backend."
        );
        let message = match acquire(name) {
            Ok(backend) => panic!(
                "Fix: `acquire(\"{name}\")` returned backend `{}`, so a caller can construct the \
                 interpreter through the production dispatch seam.",
                backend.id()
            ),
            Err(error) => error.to_string(),
        };
        assert!(
            message.contains("Fix:"),
            "Fix: the refusal must state the corrective action, got: {message}"
        );
    }
}
