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

use structure_gate::module_layout::{
    directory_stutter_failures, generic_module_name_failures, numbered_sibling_failures,
    sibling_module_failures, source_test_directory_failures,
};
use structure_gate::{
    category_home_failures, frontend_owner_failures, geometry_constant_failures,
    operation_identity_failures, registration_owner_failures, registry_link_failures,
    roster_failures, scan, substrate_home_failures, workspace_root, Workspace,
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
    let workspace = workspace();
    let failures = operation_identity_failures(&workspace.registrations, &workspace.members);

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
///
/// This is also what keeps a product out of the platform: a product consumes the
/// compiler, so a member that ships one makes the facade depend on it, and the
/// only way such a crate becomes a member is by being added to the reviewed
/// roster. The direct edge is held separately by the `layering` gate, which
/// names the facade as substrate-neutral.
#[test]
fn the_workspace_roster_matches_the_reviewed_list() {
    let failures = roster_failures(&workspace().members);

    assert!(failures.is_empty(), "{}", report("roster", &failures));
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

/// No source file sits beside a directory of its own name.
///
/// WHY: `foo.rs` next to `foo/` is one module written in two places, so a
/// reader who opens either half sees a module that appears to be missing its
/// other half, and a new child gets added to whichever half the author found.
/// The workspace carried 110 such pairs at once in `src/` and 32 in `tests/`.
/// All source trees are held to this invariant.
#[test]
fn no_module_file_sits_beside_its_own_directory() {
    let failures = sibling_module_failures(&workspace().source_files);

    assert!(
        failures.is_empty(),
        "{}",
        report("sibling-module", &failures)
    );
}

/// No crate keeps tests inside `src/**/tests/`.
///
/// WHY: behavioral tests live in top-level `tests/` directories per the workspace
/// external-test contract.
#[test]
fn no_source_test_directory_exists_under_src() {
    let failures = source_test_directory_failures(&workspace().module_files);

    assert!(
        failures.is_empty(),
        "{}",
        report("source-test-directory", &failures)
    );
}

/// Every file name states what the file holds.
///
/// WHY: `helpers`, `common`, `core`, `types`, `misc`, `support`, `utils` and the
/// same words as a suffix answer no question a reader has, so the file becomes
/// wherever an item went when nobody decided where it belonged, and it grows
/// without limit. Judged over `tests/`, `benches/` and `examples/` as well as
/// `src/`, because the prohibition was enforced against modules only and the
/// population moved into test-adjacent files: 15 of the last 16 offenders were
/// `tests/common/mod.rs` or `tests/support/mod.rs`. A module the committed
/// public-API snapshot publishes is exempt while it stays published, because
/// renaming it renames a path consumers import.
///
/// Red on this branch by 8 files: the C frontend's `sparse_impl`,
/// `build_declaration_kind_inner` and the `c_ast_gpu_parity_support` and
/// `c_token_support` test trees close by deletion with the frontend rather than
/// by rename, and teaching the rule to skip them would name paths that stop
/// existing.
#[test]
fn no_file_name_states_no_contract() {
    let workspace = workspace();
    let failures = generic_module_name_failures(
        &workspace.source_files,
        &workspace.crate_roots,
        &workspace.published_modules,
    );

    assert!(
        failures.is_empty(),
        "{}",
        report("generic-module-name", &failures)
    );
}

/// No sibling files are told apart by a number.
///
/// WHY: ten files called `nodes_00` through `nodes_09` convey nothing about
/// which one classifies a given node, so answering that question means opening
/// all ten and a new case lands in whichever file was already open. A number
/// inside a name that means something, `crc32` or `flash_attention_2`, has no
/// numbered sibling, which is exactly what this rule keys on.
///
/// Red on this branch by the ten `vyre-libs/src/parsing/c/parse/vast/classify`
/// files, which the C frontend removal deletes.
#[test]
fn no_siblings_are_told_apart_by_a_number() {
    let failures = numbered_sibling_failures(&workspace().source_files);

    assert!(
        failures.is_empty(),
        "{}",
        report("numbered-sibling", &failures)
    );
}

/// No file repeats the directory that holds it.
///
/// WHY: `fma_f32/fma_f32.rs` states its contents once and its location twice,
/// and a reader who opens the directory cannot tell whether the file is the
/// module or one part of it. The module is the directory, so that file is
/// `mod.rs`.
#[test]
fn no_file_repeats_the_directory_holding_it() {
    let failures = directory_stutter_failures(&workspace().source_files);

    assert!(
        failures.is_empty(),
        "{}",
        report("directory-stutter", &failures)
    );
}

/// Operations in `vyre-libs` and `vyre-primitives` declare `GeometryRequirements`
/// and lower through `GeometryStrategy`, never declaring hardcoded geometry constants.
#[test]
fn no_operation_declares_hardcoded_geometry_constants() {
    let failures = geometry_constant_failures(&workspace_root());

    assert!(
        failures.is_empty(),
        "{}",
        report("geometry-constants", &failures)
    );
}

/// Every crate in the checkout is judged, and the published-module scan reads.
///
/// Guards the two rules above from both directions: an empty file list accepts
/// every pair in the tree, and an empty published list instead reports every
/// published module that carries a banned name. The crate roster comes from the
/// scan itself rather than a list written here, so a crate added anywhere in
/// the checkout has to appear in the judged file list or this fails.
#[test]
fn every_crate_in_the_checkout_is_judged() {
    let workspace = workspace();

    assert!(
        workspace.crate_roots.len() > 20,
        "expected far more than {} crate root(s) in the checkout; the layout rules above are \
         passing vacuously. Found: {:?}",
        workspace.crate_roots.len(),
        workspace.crate_roots
    );
    for crate_root in &workspace.crate_roots {
        let prefix = format!("{}/src/", crate_root.directory);
        assert!(
            workspace
                .module_files
                .iter()
                .any(|file| file.starts_with(&prefix)),
            "`{}` declares a package and holds a src/ directory, but the module-file scan read \
             nothing under it, so its layout is unjudged",
            crate_root.directory
        );
        assert!(
            !crate_root.ident.contains('-'),
            "`{}` resolved to crate identifier `{}`, which no consumer can write; the \
             public-API exemption would never match it",
            crate_root.directory,
            crate_root.ident
        );
    }
    assert!(
        workspace.source_files.len() > workspace.module_files.len(),
        "the name rules read the same {} file(s) as the src-only scan, so every fixture module in \
         the checkout is unjudged",
        workspace.source_files.len()
    );
    for tree in ["tests", "benches", "examples"] {
        assert!(
            workspace
                .source_files
                .iter()
                .any(|file| file.contains(&format!("/{tree}/"))),
            "no file under any crate's {tree}/ tree reached the name rules, so that whole tree is \
             unjudged"
        );
    }
    assert!(
        workspace
            .crate_roots
            .iter()
            .any(|crate_root| !workspace.members.contains(&crate_root.directory)),
        "every judged crate root is a workspace member, so a crate outside the workspace would \
         grow pairs unjudged. Roots: {:?}",
        workspace.crate_roots
    );
    assert!(
        workspace
            .published_modules
            .iter()
            .any(|module| module == "vyre_libs::parsing"),
        "expected docs/public-api to publish vyre_libs::parsing; the snapshot scan read nothing, \
         so every published module would be reported as a banned name. Found {} module(s)",
        workspace.published_modules.len()
    );
}
