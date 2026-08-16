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

/// The set covers the whole subtree of a gated directory module.
///
/// A gated `mod tests;` whose body is `tests/mod.rs` reaches every file under
/// `tests/`, and stopping at the declaration would hold each of those to a
/// production contract.
#[test]
fn every_file_under_a_gated_directory_is_reported() {
    let gated = test_gated_module_files(&workspace_root());
    let root = workspace_root();
    let directories: Vec<String> = gated
        .iter()
        .filter(|file| root.join(file.trim_end_matches(".rs")).is_dir())
        .cloned()
        .collect();
    assert!(
        !directories.is_empty(),
        "Fix: the tree declares no gated directory module, so this case proves nothing"
    );
    for declaration in directories {
        let home = declaration.trim_end_matches(".rs");
        let child = format!("{home}/mod.rs");
        if root.join(&child).is_file() {
            assert!(
                gated.contains(&child),
                "Fix: {child} sits under the gated module {declaration} and is not in the set"
            );
        }
    }
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
