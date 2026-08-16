//! Every generator `release-evidence` takes a census of is a registered gate.
//!
//! WHY: `release-evidence` used to run thirteen other subcommands as child
//! processes, resolving the child from `std::env::current_exe`. That was
//! correct only while it lived in `xtask`; once it moved to this crate, twelve
//! of the thirteen re-entered this binary, hit this crate's own short table,
//! and exited 1 with `is not implemented in xtask-evidence`. The parent still
//! labelled each failure `xtask <name>`, so the whole release evidence surface
//! read as twelve gates failing rather than as one wrong binary.
//!
//! Nothing is spawned now. Every generator in the census is a registered gate
//! the sweep runs directly, which is what removed the defect rather than fixing
//! it. What still has to hold is that the census names gates that exist: a
//! generator named here and registered nowhere owes an artifact nobody
//! produces, and the census would report it missing forever with no gate to
//! blame. Both sides are derived at run time, so a gate renamed on either side
//! is red here rather than during a release.

#![forbid(unsafe_code)]

use xtask::subcommands::{find, owned_by, registry, subset};
use xtask_evidence::release::release_evidence::covered_subcommands;

/// WHY: a covered name that no gate answers to is an artifact with no owner.
/// The roster is read from the command table, so an entry added tomorrow is
/// judged tomorrow.
#[test]
fn every_covered_generator_is_a_registered_gate() {
    let covered = covered_subcommands();
    assert!(
        covered.len() >= 2,
        "Fix: the release evidence census covers {} generators; the command \
         table is wrong, and an empty roster makes every assertion here vacuous",
        covered.len()
    );
    let unregistered: Vec<&str> = covered
        .iter()
        .copied()
        .filter(|name| find(name).is_none() && subset(name).is_none())
        .collect();
    assert!(
        unregistered.is_empty(),
        "Fix: the release evidence census covers {unregistered:?}, which the \
         registry does not hold. Register the gate, or stop naming it in the \
         census."
    );
}

/// WHY: this is the shape of the original defect. Most of what the census
/// covers is implemented in another binary, so resolving any of it from this
/// process would be wrong for every one of those names. Asserting that the
/// census reaches past this crate is what keeps the contract non-vacuous:
/// a census that only ever named this crate's own gates would pass while
/// proving nothing about cross-package coverage.
#[test]
fn the_census_reaches_past_this_crate() {
    let mine = owned_by("xtask-evidence");
    let elsewhere: Vec<&str> = covered_subcommands()
        .into_iter()
        .filter(|name| !mine.contains(name))
        .collect();
    assert!(
        !elsewhere.is_empty(),
        "Fix: the release evidence census must cover the whole evidence surface, \
         not this crate alone. Every generator it names is now owned here, which \
         removes the only run-time evidence that the census spans packages."
    );
}

/// WHY: the sweep routes a delegated gate by building the owning package on
/// demand, so a covered gate homed in a package must name one the sweep can
/// build. A package that owns no gates would fail the child build after the
/// release started.
#[test]
fn every_delegated_generator_has_a_buildable_home() {
    let gates = registry();
    for name in covered_subcommands() {
        if let Some(gate) = gates.iter().find(|gate| gate.name() == name) {
            let package = gate.package();
            if package == "xtask" {
                continue;
            }
            assert!(
                !owned_by(package).is_empty(),
                "Fix: `{name}` is homed in `{package}`, which owns no gates. The \
                 home and descriptor registry disagree."
            );
        } else if let Some(sub) = subset(name) {
            for gate_name in sub.gates {
                let gate = gates
                    .iter()
                    .find(|gate| gate.name() == gate_name)
                    .expect("subset gate must be in registry");
                let package = gate.package();
                if package != "xtask" {
                    assert!(
                        !owned_by(package).is_empty(),
                        "Fix: `{gate_name}` is homed in `{package}`, which owns no gates. The \
                         home and descriptor registry disagree."
                    );
                }
            }
        } else {
            panic!("Fix: `{name}` is neither a registered gate nor a subset");
        }
    }
}
