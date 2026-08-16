//! Which files the tree reaches only under `#[cfg(test)]`.
//!
//! A rule that reads one file at a time cannot see the attribute that gates it,
//! because the attribute sits on the declaration in the parent. Getting this
//! wrong holds test doubles to a production contract, which is how a panic in a
//! dispatcher double came to be counted against a shipped crate's panic budget.

use structure_gate::cfg_test::test_gated_module_files;
use structure_gate::workspace_root;

/// A module gated in its parent is reported, whichever spelling it uses.
#[test]
fn a_module_gated_in_its_parent_is_reported() {
    let gated = test_gated_module_files(&workspace_root());
    assert!(
        gated.contains("vyre-libs/src/test_parity_oracles.rs"),
        "Fix: `#[cfg(test)] mod test_parity_oracles;` in vyre-libs/src/lib.rs makes that file \
         test-only, and it is not in the set"
    );
}

/// A production module is not reported, so the set is not everything.
#[test]
fn a_production_module_is_not_reported() {
    let gated = test_gated_module_files(&workspace_root());
    for file in [
        "vyre-libs/src/lib.rs",
        "structure-gate/src/cfg_test.rs",
        "vyre-runtime/src/resident_work_queue/policy/mod.rs",
    ] {
        assert!(
            !gated.contains(file),
            "Fix: {file} is reached by a production build and must not read as test-gated"
        );
    }
}

/// The set covers the whole subtree of a gated directory module, and no more.
///
/// A gated `mod fixtures;` whose body is `fixtures/mod.rs` reaches every file
/// under `fixtures/`, and stopping at the declaration would hold each of those
/// to a production contract. A file whose name merely starts with the module's
/// is a different module and stays out.
#[test]
fn every_file_under_a_gated_directory_is_reported() {
    let tree = tempfile::tempdir().expect("Fix: the fixture root must be creatable.");
    let source = tree.path().join("demo/src");
    std::fs::create_dir_all(source.join("fixtures/deep"))
        .expect("Fix: the fixture source tree must be creatable.");
    std::fs::write(
        source.join("lib.rs"),
        "#[cfg(test)]\nmod fixtures;\npub mod fixtures_registry;\n",
    )
    .expect("Fix: the fixture crate root must be writable.");
    for file in [
        "fixtures/mod.rs",
        "fixtures/helper.rs",
        "fixtures/deep/inner.rs",
        "fixtures_registry.rs",
    ] {
        std::fs::write(source.join(file), "").expect("Fix: a fixture module must be writable.");
    }

    let gated = test_gated_module_files(tree.path());

    assert_eq!(
        gated.into_iter().collect::<Vec<_>>(),
        vec![
            "demo/src/fixtures/deep/inner.rs".to_string(),
            "demo/src/fixtures/helper.rs".to_string(),
            "demo/src/fixtures/mod.rs".to_string(),
        ],
        "Fix: a gated directory module reaches its whole subtree and nothing beside it"
    );
}

/// Every path the set names is a file the tree holds.
///
/// A caller opens what the set names. Reporting `<name>.rs` beside a directory
/// module that has no such file hands out a path that cannot be read.
#[test]
fn every_reported_path_is_a_file() {
    let root = workspace_root();
    for file in test_gated_module_files(&root) {
        assert!(
            root.join(&file).is_file(),
            "Fix: {file} is in the set and the tree has no such file"
        );
    }
}

/// A declaration inside an inline `mod` block resolves under that block.
///
/// The compiler looks for the child of `pub mod outer { .. }` in `outer/`, so
/// reading the file name alone records a path the tree does not hold and the
/// real file keeps counting as production code with nothing to say so.
#[test]
fn a_declaration_inside_an_inline_module_resolves_under_it() {
    let tree = tempfile::tempdir().expect("Fix: the fixture root must be creatable.");
    let source = tree.path().join("demo/src");
    std::fs::create_dir_all(source.join("outer")).expect("Fix: the tree must be creatable.");
    std::fs::write(
        source.join("lib.rs"),
        "pub mod outer {\n    #[cfg(test)]\n    mod checks;\n}\n",
    )
    .expect("Fix: the fixture crate root must be writable.");
    std::fs::write(source.join("outer/checks.rs"), "")
        .expect("Fix: the fixture module must be writable.");
    std::fs::write(source.join("checks.rs"), "")
        .expect("Fix: the decoy module must be writable.");

    let gated = test_gated_module_files(tree.path());

    assert_eq!(
        gated.into_iter().collect::<Vec<_>>(),
        vec!["demo/src/outer/checks.rs".to_string()],
        "Fix: the gated module is the child of the inline block, not of the file"
    );
}

/// Which gate spellings the set accepts, on a tree written for the case.
///
/// `#[cfg(test)]` and `#[cfg(all(test, unix))]` name a module no build without
/// `test` compiles. `#[cfg(any(test, feature = "test-fixtures"))]` names one
/// that compiles whenever the feature is on, so it is production code a
/// consumer reaches; reading it as test-only exempted nine vyre-driver panics
/// from that crate's panic budget.
#[test]
fn only_a_gate_no_build_satisfies_without_test_counts() {
    let tree = tempfile::tempdir().expect("Fix: the fixture root must be creatable.");
    let source = tree.path().join("demo/src");
    std::fs::create_dir_all(&source).expect("Fix: the fixture source tree must be creatable.");
    std::fs::write(
        source.join("lib.rs"),
        "#[cfg(test)]\nmod gated;\n#[cfg(all(test, unix))]\nmod unix_gated;\n#[cfg(any(test, \
         feature = \"test-fixtures\"))]\npub mod fixtures;\npub mod shipped;\n",
    )
    .expect("Fix: the fixture crate root must be writable.");
    for name in ["gated", "unix_gated", "fixtures", "shipped"] {
        std::fs::write(source.join(format!("{name}.rs")), "")
            .expect("Fix: a fixture module must be writable.");
    }

    let gated = test_gated_module_files(tree.path());

    assert_eq!(
        gated.into_iter().collect::<Vec<_>>(),
        vec![
            "demo/src/gated.rs".to_string(),
            "demo/src/unix_gated.rs".to_string()
        ],
        "Fix: only a module whose gate every configuration satisfies with `test` on is test-only"
    );
}
