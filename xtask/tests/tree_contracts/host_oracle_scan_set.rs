//! The source roster the host-oracle gate scans.
//!
//! The gate answers a question about the whole workspace, and it answers it by
//! reading a set of directories. When that set was three literal paths the gate
//! reported a clean verdict over sixteen crates while reading three, and the
//! narrowing left no trace in the output: an empty scan and a clean scan print
//! the same line.
//!
//! Both halves are derived from the tree here. The roster comes from the gate,
//! and the expectation comes from an independent walk of the workspace members,
//! so a crate that starts registering operations, a crate that stops, and a
//! move of the registration type each turn this red instead of shrinking the
//! scan in silence.

use std::collections::BTreeSet;
use std::path::Path;

use xtask::gate::Report;
use xtask::gates::host_oracle_closure::shipped_source_roots;
use xtask::gates::host_oracle_elimination::operation_bearing_roots;
use xtask::gates::scan::Tree;

use super::workspace_sources::workspace_root;

/// The `src` directory of every workspace member whose sources match `predicate`.
fn members_whose_sources(root: &Path, predicate: impl Fn(&str) -> bool) -> BTreeSet<String> {
    let tree = Tree::open(root).expect("the workspace manifest is readable");
    let mut matched = BTreeSet::new();
    for member in tree.members().expect("the member list is readable") {
        let src = format!("{member}/src");
        if !tree.exists(&src) {
            continue;
        }
        let files = tree
            .rust(&[src.as_str()])
            .expect("a member source tree is walkable");
        for file in &files {
            let text = tree.read(file).expect("a source file is readable");
            if predicate(&text) {
                matched.insert(src.clone());
                break;
            }
        }
    }
    matched
}

/// The scanned roster is every shipped crate that registers an operation, and nothing else.
#[test]
fn the_host_oracle_scan_set_is_the_operation_registering_roster() {
    let root = workspace_root();
    let tree = Tree::open(&root).expect("the workspace manifest is readable");
    let mut report = Report::clean();

    let scanned: BTreeSet<String> = operation_bearing_roots(&tree, &mut report)
        .expect("the roster derivation reads the tree")
        .into_iter()
        .collect();

    let shipped: BTreeSet<String> = shipped_source_roots(&tree, &mut report)
        .expect("the shipped roster reads the tree")
        .into_iter()
        .collect();

    let registers = members_whose_sources(&root, |text| text.contains("OperationRegistration::"));
    let declares = members_whose_sources(&root, |text| text.contains("impl OperationRegistration"));

    let expected: BTreeSet<String> = registers
        .difference(&declares)
        .filter(|src| shipped.contains(*src))
        .cloned()
        .collect();

    assert_eq!(
        scanned, expected,
        "the gate scans exactly the shipped crates that register an operation without declaring \
         the trait; a crate appearing on one side only means the roster derivation and the \
         workspace disagree"
    );
}

/// The roster covers more than the three directories that were once written out by hand.
#[test]
fn the_host_oracle_scan_set_is_not_a_vestigial_handful() {
    let root = workspace_root();
    let tree = Tree::open(&root).expect("the workspace manifest is readable");
    let mut report = Report::clean();

    let scanned =
        operation_bearing_roots(&tree, &mut report).expect("the roster derivation reads the tree");
    let files = tree
        .rust(&scanned.iter().map(String::as_str).collect::<Vec<_>>())
        .expect("the scanned roster is walkable");

    assert!(
        scanned.len() > 3,
        "three literal directories once stood here and covered a fraction of the workspace while \
         reporting a clean verdict; the derived roster is {scanned:?}"
    );
    assert!(
        files.len() > 100,
        "a roster of {} directories holding {} source files cannot support a workspace-wide \
         verdict",
        scanned.len(),
        files.len()
    );
}

/// The crate that declares the registration trait is not scanned as a registrant.
#[test]
fn the_crate_declaring_the_registration_trait_is_out_of_scope() {
    let root = workspace_root();
    let tree = Tree::open(&root).expect("the workspace manifest is readable");
    let mut report = Report::clean();

    let scanned: BTreeSet<String> = operation_bearing_roots(&tree, &mut report)
        .expect("the roster derivation reads the tree")
        .into_iter()
        .collect();
    let declares = members_whose_sources(&root, |text| text.contains("impl OperationRegistration"));

    assert!(
        !declares.is_empty(),
        "the registration trait is implemented somewhere in this workspace, and a derivation that \
         finds no declaring crate is reading nothing"
    );
    for src in &declares {
        assert!(
            !scanned.contains(src),
            "{src} declares the registration trait, so its own constructors are not the \
             registrations the gate looks for"
        );
    }
}
