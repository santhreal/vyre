//! Structural contracts for the live workspace.
//!
//! WHY: the duplication class these tests close is *identity duplication*. One
//! kernel gets registered a second time one tier up so the higher crate can own
//! an id, and the workspace ends with two op ids, two catalog rows, two fixture
//! sets, and two test suites over one implementation. Nothing failed when that
//! happened, so it happened everywhere.
//!
//! Each test judges the real tree, not a fixture, because a fixture proves the
//! rule parses and proves nothing about the repository. The rule logic itself is
//! unit-tested in `structure_gate`; these tests are the standing gate.
//!
//! What these do NOT catch: two implementations of one algorithm that never
//! register an operation, and duplication inside a single crate. `lego-audit`
//! and `dedup-report` own the IR-fingerprint side of that.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use structure_gate::{
    category_home_failures, frontend_owner_failures, operation_identity_failures,
    registration_owner_failures, registry_link_failures, roster_failures, scan,
    substrate_home_failures, workspace_root, Workspace,
};

fn workspace() -> Workspace {
    scan(&workspace_root())
}

fn report(kind: &str, failures: &[String]) -> String {
    let mut message = format!("{} {kind} violation(s):\n", failures.len());
    for failure in failures {
        message.push_str("  - ");
        message.push_str(failure);
        message.push('\n');
    }
    message
}

/// Only `vyre-libs` (Category A) and `vyre-primitives` (Category C) own operations.
#[test]
fn only_the_two_category_crates_register_operations() {
    let failures = registration_owner_failures(&workspace().registrations);

    assert!(
        failures.is_empty(),
        "{}",
        report("registration-owner", &failures)
    );
}

/// One kernel carries exactly one operation identity.
#[test]
fn no_operation_is_registered_under_two_identities() {
    let failures = operation_identity_failures(&workspace().registrations);

    assert!(
        failures.is_empty(),
        "{}",
        report("operation-identity", &failures)
    );
}

/// Category A stays in the composition crate; Category C stays in the hardware crate.
#[test]
fn every_operation_sits_in_its_category_home() {
    let failures = category_home_failures(&workspace().registrations);

    assert!(
        failures.is_empty(),
        "{}",
        report("category-home", &failures)
    );
}

/// The substrate concept has one home.
#[test]
fn the_substrate_concept_has_one_home() {
    let failures = substrate_home_failures(&workspace().substrate_paths);

    assert!(
        failures.is_empty(),
        "{}",
        report("substrate-home", &failures)
    );
}

/// One source language, one frontend crate.
#[test]
fn each_source_language_has_one_frontend() {
    let failures = frontend_owner_failures(&workspace().frontend_paths);

    assert!(
        failures.is_empty(),
        "{}",
        report("frontend-owner", &failures)
    );
}

/// The workspace roster is a reviewed, closed list.
#[test]
fn the_workspace_roster_matches_the_reviewed_list() {
    let failures = roster_failures(&workspace().members);

    assert!(failures.is_empty(), "{}", report("roster", &failures));
}

/// A member that ships a product rather than a compiler layer is not a member.
///
/// A product consumes the platform. Keeping one inside the workspace makes the
/// facade depend on it, which is how `vyre` came to pull a corpus scanner into
/// every consumer of the compiler.
#[test]
fn no_product_crate_ships_inside_the_platform_workspace() {
    let workspace = workspace();
    let members: BTreeSet<&str> = workspace.members.iter().map(String::as_str).collect();

    for product in ["vyre-scan"] {
        assert!(
            !members.contains(product),
            "`{product}` is a product built on Vyre, not a layer of it; it belongs outside this workspace"
        );
    }
}

/// The scan finds registrations at all.
///
/// Guards the other tests: a parser that silently matches nothing would make
/// every registration rule pass by finding no registrations to judge.
#[test]
fn the_registration_scan_is_not_vacuous() {
    let registrations = workspace().registrations;

    assert!(
        registrations.len() > 100,
        "expected the workspace to register far more than {} operations; the source scan is broken, \
         and every registration rule above is passing vacuously",
        registrations.len()
    );
}

/// No crate that submits inventory registrations is linked by name alone.
///
/// WHY: registrations live in the declaring crate's object file, and the linker
/// keeps that object only when a symbol inside it is referenced. `use vyre_libs
/// as _;` names the crate and references nothing, so the registrations were
/// dropped from every binary that did not otherwise call into that crate: three
/// registry rules iterated an empty registry and passed while the production
/// binary saw all of it. The rule is derived from the tree, so a new submitting
/// crate is judged the moment it submits.
#[test]
fn no_registry_source_is_linked_by_a_discarding_import() {
    let workspace = workspace();
    let failures = registry_link_failures(
        &workspace.registry_submitters,
        &workspace.discarding_imports,
    );

    assert!(
        failures.is_empty(),
        "{}",
        report("registry-link", &failures)
    );
}

/// The submitting-crate scan finds the crates that submit.
///
/// Guards the rule above: a scan that matched nothing would accept every
/// discarding import in the tree.
#[test]
fn the_registry_submitter_scan_is_not_vacuous() {
    let submitters = workspace().registry_submitters;

    for expected in [
        "vyre-libs",
        "vyre-primitives",
        "vyre-driver-cuda",
        "vyre-driver-metal",
        "vyre-driver-reference",
        "vyre-driver-spirv",
        "vyre-driver-wgpu",
    ] {
        assert!(
            submitters.iter().any(|found| found == expected),
            "`{expected}` submits inventory registrations but the scan did not find it; every \
             discarding import naming it would be accepted. Found: {submitters:?}"
        );
    }
}
