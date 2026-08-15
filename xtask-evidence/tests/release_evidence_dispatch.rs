//! Every subcommand `release-evidence` spawns is one the dispatcher can route.
//!
//! WHY: `release-evidence` runs thirteen other subcommands as child processes.
//! It used to resolve that child from `std::env::current_exe`, which was correct
//! only while it lived in `xtask`; once it moved to this crate, twelve of the
//! thirteen re-entered this binary, hit this crate's own six-entry table, and
//! exited 1 with `is not implemented in xtask-evidence`. The parent still
//! labelled each failure `xtask <name>`, so the whole release evidence surface
//! read as twelve gates failing rather than as one wrong binary.
//!
//! Both sides are derived at run time: the spawned roster from the command
//! table, and the routing from the dispatcher's own `SUBCOMMANDS`. A subcommand
//! renamed on either side is red here rather than during a release.

#![forbid(unsafe_code)]

use xtask::subcommands::{Home, SUBCOMMANDS};
use xtask_evidence::release::release_evidence::spawned_subcommands;
use xtask_evidence::IMPLEMENTED;

/// The row the dispatcher registers for `name`, if it registers one.
fn registered(name: &str) -> Option<&'static xtask::subcommands::Subcommand> {
    SUBCOMMANDS.iter().find(|entry| entry.name == name)
}

/// WHY: a spawned name the dispatcher does not register runs nothing and reports
/// an exit code, so a renamed or deleted subcommand would surface as its own
/// gate failing. The roster is read from the command table, so an entry added
/// tomorrow is judged tomorrow.
#[test]
fn every_spawned_subcommand_is_registered_with_the_dispatcher() {
    let spawned = spawned_subcommands();
    assert!(
        spawned.len() >= 2,
        "Fix: release-evidence spawns {} subcommands; the roster parse or the \
         command table is wrong, and an empty roster makes every assertion here \
         vacuous",
        spawned.len()
    );
    let unregistered: Vec<&str> = spawned
        .iter()
        .copied()
        .filter(|name| registered(name).is_none())
        .collect();
    assert!(
        unregistered.is_empty(),
        "Fix: release-evidence spawns {unregistered:?}, which the dispatcher \
         does not register. Add the row to xtask::subcommands::SUBCOMMANDS or \
         stop spawning the name."
    );
}

/// WHY: this is the defect itself. Most of what `release-evidence` spawns is
/// implemented in another binary, so resolving the child from this process is
/// wrong for every one of those names. Asserting that at least one spawned
/// subcommand is owned elsewhere is what makes the routing contract non-vacuous:
/// re-introducing `current_exe` here passes only if this crate implements
/// everything it spawns, which it must never do.
#[test]
fn most_spawned_subcommands_are_owned_by_another_binary() {
    let mine: Vec<&str> = IMPLEMENTED.iter().map(|(name, _)| *name).collect();
    let elsewhere: Vec<&str> = spawned_subcommands()
        .into_iter()
        .filter(|name| !mine.contains(name))
        .collect();
    assert!(
        !elsewhere.is_empty(),
        "Fix: release-evidence must spawn the dispatcher, not itself. Every \
         subcommand it spawns is now implemented here, which removes the only \
         run-time evidence that the child binary is resolved by owner."
    );
}

/// WHY: the dispatcher routes a delegated subcommand by building the owning
/// package on demand, so a spawned subcommand whose home names a package must
/// name one the dispatcher can build. A home pointing at a package that is not a
/// delegate would make the child build fail after the release started.
#[test]
fn every_spawned_subcommand_has_a_buildable_home() {
    for name in spawned_subcommands() {
        let entry = registered(name).expect("checked by the registration contract");
        match entry.home {
            Home::Local(_) => {}
            Home::Registry | Home::Evidence => {
                let package = entry
                    .home
                    .package()
                    .expect("a delegated home names its package");
                assert!(
                    !xtask::subcommands::owned_by(package).is_empty(),
                    "Fix: `{name}` is homed in `{package}`, which owns no \
                     subcommands. The home and the delegate table disagree."
                );
            }
        }
    }
}
