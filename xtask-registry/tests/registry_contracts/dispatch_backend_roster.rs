//! Every dispatch backend a release requires has a producer that can report it.
//!
//! WHY: a release requires three dispatch backends and only two of them are a
//! `BackendRegistration`. The reference route executes on the host through the
//! interpreter, which is exactly why it submits no registration and why
//! `vyre-driver-reference` owns an execution-domain closure forbidding one. The
//! conformance roster read the backend registry alone, so `cpu-ref` was a name
//! the roster demanded and no reader could ever produce, and the gate reported
//! it missing on every run regardless of what the tree could dispatch.
//!
//! Nothing here spells a backend id. `REQUIRED_DISPATCH_BACKENDS` is the release
//! decision, the registry reports what this build links, and the reference
//! crate reports its own route, so adding a required name with no producer,
//! or dropping a producer a required name depends on, turns this red.
//!
//! What it does not catch: whether a device backend that reports `dispatches`
//! can reach a device. That is the conformance run's judgement, not a roster's.

use std::collections::BTreeSet;

use vyre_driver_reference::ORACLE_EXECUTOR_ID;
use xtask::release::conformance_evidence_semantics::{
    ORACLE_RECORD_ID, REQUIRED_DISPATCH_BACKENDS,
};
use xtask_registry::release::conformance_matrix::{dispatch_backend_roster, RECORDED_BACKENDS};

/// The roster as the gate composes it.
fn roster() -> BTreeSet<String> {
    dispatch_backend_roster()
        .expect("Fix: the backend registry must start before a roster can be composed.")
        .into_iter()
        .collect()
}

/// Every id the registry reports as submitting `dispatches: true`.
fn registered_dispatching() -> BTreeSet<String> {
    vyre_registry_link::backend::live_backend_registry_by_precedence()
        .expect("Fix: the backend registry must start.")
        .iter()
        .filter(|registration| {
            vyre_driver::backend_dispatches(registration.id)
                .expect("Fix: the backend registry must start.")
        })
        .map(|registration| registration.id.to_string())
        .collect()
}

/// The requirement and the producers agree, name by name.
#[test]
fn every_required_dispatch_backend_has_a_producer() {
    let roster = roster();

    let unproducible = REQUIRED_DISPATCH_BACKENDS
        .iter()
        .filter(|required| !roster.contains(**required))
        .collect::<Vec<_>>();

    assert_eq!(
        unproducible,
        Vec::<&&str>::new(),
        "Fix: a required dispatch backend no producer can report is a requirement about nothing. \
         Supply the producer or record why the requirement is retired; roster={roster:?}"
    );
}

/// The roster is exactly what the backend registry reports.
///
/// WHY: the requirement above is satisfiable by appending any string. This
/// pins the roster to the live registry, so a name cannot be added to quiet a
/// gate without a registered backend behind it.
#[test]
fn the_roster_is_exactly_the_registry() {
    assert_eq!(
        roster(),
        registered_dispatching(),
        "Fix: the roster reports what the backend registry reports and nothing else."
    );
}

/// Satisfying the roster did not put a host executor into the registry.
///
/// WHY: the cheap way to close the missing-`cpu-ref` finding is to submit a
/// `BackendRegistration` for the interpreter, which restores a CPU execution
/// route to backend discovery, precedence and autoroute. Every spelling that
/// reaches the oracle must still resolve to nothing there.
#[test]
fn no_oracle_spelling_resolves_to_a_registered_backend() {
    let registrations = vyre_registry_link::backend::live_backend_registry()
        .expect("Fix: the backend registry must start.");

    for id in [ORACLE_EXECUTOR_ID, "cpu-ref", "cpu"] {
        assert!(
            !registrations
                .iter()
                .any(|registration| registration.id == id),
            "Fix: `{id}` reaches the interpreter and must not be a registered backend."
        );
        assert!(
            !vyre_driver::backend_dispatches(id).expect("Fix: the backend registry must start."),
            "Fix: `{id}` must not report as a dispatching registration."
        );
    }
    for registration in registrations {
        assert!(
            !registration.reference_oracle,
            "Fix: backend `{}` claims to be the oracle and must not be in the registry.",
            registration.id
        );
    }
}

/// The oracle is not a required dispatch backend.
///
/// WHY: the release required `cpu-ref` as a dispatch backend while the
/// registry is built to refuse it, so the requirement could only ever report
/// missing. The oracle is a recorded executor, not a dispatched device, and
/// the required set names devices only.
#[test]
fn the_oracle_is_not_a_required_dispatch_backend() {
    let oracle_spellings = REQUIRED_DISPATCH_BACKENDS
        .iter()
        .filter(|required| {
            **required == ORACLE_EXECUTOR_ID || **required == "cpu-ref" || **required == "cpu"
        })
        .copied()
        .collect::<Vec<_>>();

    assert!(
        oracle_spellings.is_empty(),
        "Fix: the required dispatch set names devices the registry can produce. The oracle is \
         recorded under `ORACLE_RECORD_ID`, not required as a backend; found {oracle_spellings:?}"
    );
}

/// The recorded oracle label and the oracle's own id have one owner between
/// them.
///
/// WHY: `xtask` links no drivers, so its `ORACLE_RECORD_ID` is a copy of a
/// name `vyre-driver-reference` owns. This crate links both, which is the only
/// place the two can be compared. Without this they drift and a release reads
/// an oracle record under a label nothing writes.
#[test]
fn the_recorded_oracle_label_is_the_oracle_id() {
    assert_eq!(
        ORACLE_RECORD_ID, ORACLE_EXECUTOR_ID,
        "Fix: the label a release reads for the oracle record is the id the oracle writes."
    );
}

/// Every required dispatch backend has a recorded run to be judged against.
///
/// WHY: the roster and the evidence table are two lists that must agree. A
/// required backend with no row in the recorded table is dispatched, measured
/// and then never compared to a single OP_MATRIX cell, which is a whole
/// backend's worth of claims certified by nothing.
#[test]
fn every_required_dispatch_backend_has_a_recorded_run() {
    let recorded = RECORDED_BACKENDS
        .iter()
        .map(|(_column, _artifact, recorded_id)| *recorded_id)
        .collect::<BTreeSet<_>>();

    let unjudged = REQUIRED_DISPATCH_BACKENDS
        .iter()
        .filter(|required| !recorded.contains(*required))
        .collect::<Vec<_>>();

    assert_eq!(
        unjudged,
        Vec::<&&str>::new(),
        "Fix: a required dispatch backend with no recorded run has every claim about it \
         unchecked; recorded={recorded:?}"
    );
}

/// Every backend column OP_MATRIX declares is one the recorded table can judge.
///
/// WHY: the other direction, and the one that reaches the op registry. The
/// matrix is generated per registered operation and per backend column, so a
/// new column arrives with a status for hundreds of operations at once. A
/// column with no recorded artifact behind it makes every one of those cells a
/// claim nothing observes, which is the exact state the agreement rule exists
/// to make impossible.
#[test]
fn every_op_matrix_backend_column_has_a_recorded_run() {
    let catalog = xtask::release::conformance_op_matrix::read_conformance_required_op_matrix(
        &xtask::checkout::checkout_root(),
    );
    assert!(
        !catalog.release_backend_specs.is_empty(),
        "Fix: OP_MATRIX must declare backend rows, or this contract judges nothing."
    );
    let recorded = RECORDED_BACKENDS
        .iter()
        .map(|(column, _artifact, _recorded_id)| *column)
        .collect::<BTreeSet<_>>();

    let unrecorded = catalog
        .release_backend_specs
        .iter()
        .map(|spec| spec.backend.as_str())
        .filter(|backend| !recorded.contains(backend))
        .collect::<BTreeSet<_>>();

    assert_eq!(
        unrecorded,
        BTreeSet::new(),
        "Fix: give the column a recorded conformance run, or stop generating the column; \
         recorded={recorded:?}"
    );
}
