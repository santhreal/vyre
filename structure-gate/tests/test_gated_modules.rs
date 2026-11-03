//! Which files the tree reaches only under `#[cfg(test)]`.
//!
//! A rule that reads one file at a time cannot see the attribute that gates it,
//! because the attribute sits on the declaration in the parent. Getting this
//! wrong holds test doubles to a production contract, which is how a panic in a
//! dispatcher double came to be counted against a shipped crate's panic budget.

use structure_gate::cfg_test::{test_gated_module_files, TestSources};
use structure_gate::workspace_root;

/// A module gated in its parent is reported, whichever spelling it uses.
#[test]
fn a_module_gated_in_its_parent_is_reported() {
    let root = workspace_root();
    let gated = test_gated_module_files(&root);
    assert!(
        !gated.is_empty(),
        "Fix: workspace must contain parent-gated test modules."
    );
    for file in &gated {
        let full = root.join(file);
        assert!(
            full.is_file(),
            "Fix: reported path {file} must exist as a file on disk."
        );
    }
}

/// Test tooling is not declared in a library crate's published surface.
///
/// Shared parity oracles and dispatcher doubles belong in `vyre-test-support`.
/// An ungated `pub mod` of test tooling in a library crate exposes test doubles
/// in that crate's published API.
#[test]
fn test_tooling_is_not_in_a_published_library_surface() {
    let root = workspace_root();
    for source in structure_gate::source_scan::rust_sources_with_text(&root) {
        let structure_gate::source_scan::SourceText::Read { path, text } = source else {
            continue;
        };
        if !path.contains("/src/") || path.starts_with("vyre-test-support/") {
            continue;
        }
        let production = structure_gate::cfg_test::strip_cfg_test_items(&text);
        for line in production.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("pub mod test_parity_oracles;")
                || trimmed.starts_with("pub mod fixture_bytes;")
            {
                panic!(
                    "Fix: {path} declares ungated `{trimmed}`. \
                     Test tooling must be owned by `vyre-test-support` and not exposed in \
                     a library crate's published surface."
                );
            }
        }
    }
}

/// No domain crate declares a relative source include into a `tests/` directory.
///
/// A crate's test module belongs in that crate, and reaching across crate boundaries
/// with relative `#[path = "...tests/..."]` attributes couples packages to monorepo layout.
#[test]
fn no_domain_crate_declares_relative_tests_include() {
    let root = workspace_root();
    let members = structure_gate::workspace_members(&root);
    let mut failures = Vec::new();

    for member in members {
        if !member.starts_with("vyre-libs") {
            continue;
        }
        let crate_dir = root.join(&member);
        for source in structure_gate::source_scan::rust_sources_with_text(&crate_dir) {
            let structure_gate::source_scan::SourceText::Read { path, text } = source else {
                continue;
            };
            for (line_no, line) in text.lines().enumerate() {
                let trimmed = line.trim();
                if trimmed.contains("#[path")
                    && trimmed.contains("tests/")
                    && trimmed.contains("../")
                {
                    failures.push(format!("{path}:{}: {trimmed}", line_no + 1));
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "Fix: domain crates must not contain relative source includes into a `tests/` directory:\n{}",
        failures.join("\n")
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
    std::fs::write(source.join("checks.rs"), "").expect("Fix: the decoy module must be writable.");

    let gated = test_gated_module_files(tree.path());

    assert_eq!(
        gated.into_iter().collect::<Vec<_>>(),
        vec!["demo/src/outer/checks.rs".to_string()],
        "Fix: the gated module is the child of the inline block, not of the file"
    );
}

/// An inline block written in a comment or a literal is text, not a module.
///
/// A commented `mod outer {` has no closing brace of its own, so reading it as
/// real code borrows the closer of the block around it and spans every
/// declaration between the two. The derived path then carries a module the tree
/// does not hold, the real file matches nothing, and it keeps counting as
/// production code with nothing reporting it.
#[test]
fn a_quoted_inline_module_does_not_move_a_declaration() {
    let tree = tempfile::tempdir().expect("Fix: the fixture root must be creatable.");
    let source = tree.path().join("demo/src");
    std::fs::create_dir_all(source.join("real")).expect("Fix: the tree must be creatable.");
    std::fs::write(
        source.join("lib.rs"),
        "pub mod real {\n    // pub mod outer {\n    #[cfg(test)]\n    mod checks;\n}\n",
    )
    .expect("Fix: the fixture crate root must be writable.");
    std::fs::write(source.join("real/checks.rs"), "")
        .expect("Fix: the fixture module must be writable.");

    let gated = test_gated_module_files(tree.path());

    assert_eq!(
        gated.into_iter().collect::<Vec<_>>(),
        vec!["demo/src/real/checks.rs".to_string()],
        "Fix: a quoted or commented block must not move the declaration under it"
    );
}

/// A block whose keyword, name and brace are separated by a newline or a
/// comment is the same block to the compiler.
///
/// A reader that demands the literal `mod ` followed by one identifier and then
/// only whitespace records no block for `mod\nouter {`, so the declaration
/// inside it resolves against the file's own directory. The path it names holds
/// no file, the real one matches nothing, and the module keeps counting as
/// production code.
#[test]
fn a_block_spelled_across_a_line_or_a_comment_still_holds_its_declaration() {
    for (label, header) in [
        ("newline", "pub mod\nouter {\n"),
        ("comment", "pub mod outer /* named here */ {\n"),
        ("two spaces", "pub mod  outer {\n"),
    ] {
        let tree = tempfile::tempdir().expect("Fix: the fixture root must be creatable.");
        let source = tree.path().join("demo/src");
        std::fs::create_dir_all(source.join("outer")).expect("Fix: the tree must be creatable.");
        std::fs::write(
            source.join("lib.rs"),
            format!("{header}    #[cfg(test)]\n    mod checks;\n}}\n"),
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
            "Fix: the {label} spelling of an inline module must still hold its declaration"
        );
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

/// Write `files` under a fresh fixture root, each `(relative path, text)`.
fn fixture_tree(files: &[(&str, &str)]) -> tempfile::TempDir {
    let tree = tempfile::tempdir().expect("Fix: the fixture root must be creatable.");
    for (relative, text) in files {
        let path = tree.path().join(relative);
        std::fs::create_dir_all(path.parent().expect("Fix: a fixture file has a parent."))
            .expect("Fix: the fixture directory must be creatable.");
        std::fs::write(&path, text).expect("Fix: a fixture file must be writable.");
    }
    tree
}

/// A gated declaration carrying `#[path]` gates the file the attribute names.
///
/// The relocated spelling is how a crate keeps a large test module beside the
/// file it tests: `#[cfg(test)] #[path = "pool_tests.rs"] mod tests;`. Reading
/// only the module name looked for `pool/tests.rs`, which does not exist, and
/// the real file counted as shipped code. Every placement is covered: the
/// attribute before or after the gate, inside an inline block, and climbing out
/// of the declaring directory. The decoys sit where the unrelocated rule looks.
#[test]
fn a_relocated_gated_module_is_the_file_its_path_attribute_names() {
    let tree = fixture_tree(&[
        (
            "demo/src/lib.rs",
            "mod pool;\npub mod outer {\n    #[cfg(test)]\n    #[path = \"deep.rs\"]\n    \
             mod deep;\n}\n#[cfg(test)]\n#[path = \"../fixtures/shared.rs\"]\nmod shared;\n",
        ),
        (
            "demo/src/pool.rs",
            "#[cfg(test)]\n#[path = \"pool_tests.rs\"]\nmod tests;\n\
             #[path = \"other_tests.rs\"]\n#[cfg(test)]\nmod other;\n",
        ),
        ("demo/src/pool_tests.rs", ""),
        ("demo/src/other_tests.rs", ""),
        ("demo/src/outer/deep.rs", ""),
        ("demo/fixtures/shared.rs", ""),
        ("demo/src/pool/tests.rs", ""),
        ("demo/src/pool/other.rs", ""),
        ("demo/src/deep.rs", ""),
        ("demo/src/shared.rs", ""),
    ]);

    let gated = test_gated_module_files(tree.path());

    assert_eq!(
        gated.into_iter().collect::<Vec<_>>(),
        vec![
            "demo/fixtures/shared.rs".to_string(),
            "demo/src/other_tests.rs".to_string(),
            "demo/src/outer/deep.rs".to_string(),
            "demo/src/pool_tests.rs".to_string(),
        ],
        "Fix: a relocated gated module is the file its `#[path]` names, resolved \
         against the declaring directory, or the inline block's directory inside one"
    );
}

/// Test source is a target directory or a gated declaration, never a name.
///
/// Seven gates once held their own file-name rules, and they disagreed: one
/// excused every `test_*.rs`, which skipped the shipped `bitset/test_bit.rs`
/// operation, and another missed `benches/`. A name that reads like a test and
/// is compiled into the library ships; a file under `tests/`, `benches/`,
/// `examples/` or `fuzz/` never does, and neither does a module behind a
/// `#[cfg(test)]` declaration, whatever it is called.
#[test]
fn test_sources_are_judged_by_target_directory_and_declaration_not_by_name() {
    let tree = fixture_tree(&[
        (
            "demo/src/lib.rs",
            "pub mod test_bit;\nmod latest_tests;\nmod geom;\n#[cfg(test)]\nmod checks;\n",
        ),
        ("demo/src/geom.rs", "mod tests;\n"),
        ("demo/src/geom/tests/mod.rs", ""),
        ("demo/src/test_bit.rs", ""),
        ("demo/src/latest_tests.rs", ""),
        ("demo/src/checks.rs", ""),
        ("demo/tests/integration.rs", ""),
        ("demo/benches/throughput.rs", ""),
        ("demo/examples/walkthrough.rs", ""),
        ("fuzz/fuzz_targets/decode.rs", ""),
    ]);

    let tests = TestSources::of(tree.path());

    for test_source in [
        "demo/src/checks.rs",
        "demo\\src\\checks.rs",
        "demo/tests/integration.rs",
        "demo/benches/throughput.rs",
        "demo/examples/walkthrough.rs",
        "fuzz/fuzz_targets/decode.rs",
    ] {
        assert!(
            tests.contains(test_source),
            "Fix: `{test_source}` builds no shipped artifact and must read as test source"
        );
    }
    for shipped in [
        "demo/src/lib.rs",
        "demo/src/test_bit.rs",
        "demo/src/latest_tests.rs",
        "demo/src/tests.rs",
        "demo/src/geom/tests/mod.rs",
    ] {
        assert!(
            !tests.contains(shipped),
            "Fix: `{shipped}` is compiled into the library and must read as shipped source"
        );
    }
    for harness in [
        "demo/src/checks.rs",
        "demo/tests/integration.rs",
        "demo/benches/throughput.rs",
    ] {
        assert!(
            tests.contains_harness(harness),
            "Fix: `{harness}` builds a test or bench harness"
        );
    }
    for program in [
        "demo/examples/walkthrough.rs",
        "fuzz/fuzz_targets/decode.rs",
    ] {
        assert!(
            !tests.contains_harness(program),
            "Fix: `{program}` builds a standalone program, not a test harness"
        );
    }
}
