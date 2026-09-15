//! Contract tests for the scanner that decides whether a test reads this
//! checkout's own source.
//!
//! A test that inspects the tree it runs in is held to a different rule than
//! one that authors its own fixture tree, so every way the inspection can be
//! written has to be recognised: a hoisted read, a delegated parse, a walk
//! stated only inside a macro, and a transitive or aliased one.

use super::*;
use std::path::Path;

#[test]
fn source_inspection_test_scanner_is_syntax_aware_and_fail_closed() {
    let forbidden = r#"
            #[cfg(test)]
            mod tests {
                #[test]
                fn freezes_helper_spelling() {
                    let source = include_str!("owner.rs");
                    assert!(source.contains("fn helper"));
                }
            }
        "#;
    let allowed = r###"
            #[cfg(test)]
            mod tests {
                #[test]
                fn verifies_product_text() {
                    let template = include_str!("launcher.rs.tmpl");
                    assert!(template.contains("pub fn launch"));
                }

                #[test]
                fn verifies_behavior() {
                    let summary = ResultSummary { source: "derived_pair_envelope" };
                    assert!(summary.source.contains("derived_pair_envelope"));
                }

                #[test]
                fn scanner_fixture_is_data() {
                    let forbidden = r##"include_str!("owner.rs").contains("fn helper")"##;
                    assert!(forbidden.contains("owner.rs"));
                }
            }
        "###;
    let mut findings = Vec::new();
    scan_source_inspection_tests(Path::new("driver/src/lib.rs"), forbidden, &mut findings);
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].pattern, "source_inspection_test");
    assert!(findings[0].text.contains("freezes_helper_spelling"));

    findings.clear();
    scan_source_inspection_tests(Path::new("driver/src/lib.rs"), allowed, &mut findings);
    assert!(findings.is_empty());
}

/// WHY: the walk followed calls between functions only, so hoisting a source
/// read into a `LazyLock` static took it out of the graph. A test reading the
/// static then reached no function that read anything, the file scanned as
/// inspecting no source, and every declared row naming it read as stale. That
/// is the silent direction of the failure: the gate stops covering the file and
/// reports the exemptions as the thing to delete.
///
/// It does not catch a read reached only through a trait object or a function
/// pointer stored at run time, which no name in the file resolves.
#[test]
fn source_inspection_test_scanner_follows_a_read_hoisted_into_a_static() {
    let hoisted = r#"
            static SOURCE_CORPUS: LazyLock<Vec<String>> = LazyLock::new(read_source_corpus);

            fn read_source_corpus() -> Vec<String> {
                let text = std::fs::read_to_string("owner.rs").unwrap();
                vec![text]
            }

            #[cfg(test)]
            mod tests {
                #[test]
                fn freezes_helper_spelling() {
                    assert!(SOURCE_CORPUS[0].contains("fn helper"));
                }
            }
        "#;
    let mut findings = Vec::new();
    scan_source_inspection_tests(Path::new("driver/src/lib.rs"), hoisted, &mut findings);
    assert_eq!(
        findings.len(),
        1,
        "a source read behind a static must still reach the test that reads it: {findings:?}"
    );
    assert_eq!(findings[0].pattern, "source_inspection_test");
    assert!(findings[0].text.contains("freezes_helper_spelling"));
}

/// WHY: text inspection was detected only through five string methods, so a
/// test that read a `.rs` file and handed the text to a parser was classified as
/// inspecting nothing. Three declared rows in `STRUCTURAL_GATES.toml` were
/// reported stale on that account while the tests they name still read source.
/// Delegating the parse is the same inspection as searching the text.
#[test]
fn source_inspection_is_detected_when_the_parse_is_delegated() {
    let delegated = r#"
            #[test]
            fn every_declared_variant_is_listed() {
                let source = std::fs::read_to_string(root().join("objective/metric.rs")).unwrap();
                let body = vyre_test_support::braced_body(&source, DECLARATION).unwrap();
                assert_eq!(vyre_test_support::top_level_variant_names(body).len(), 9);
            }
        "#;
    let mut findings = Vec::new();
    scan_source_inspection_tests(
        Path::new("driver/tests/objective.rs"),
        delegated,
        &mut findings,
    );
    assert_eq!(findings.len(), 1, "a delegated parse is source inspection");
    assert!(findings[0]
        .text
        .contains("every_declared_variant_is_listed"));
}

/// WHY: a typed visitor never enters a macro body, so a test that states its
/// whole inspection inside `assert_eq!` named its `.rs` file nowhere the
/// scanner looked. The walk over this checkout was invisible while the same
/// test written with a `let` binding was a blocker, which made the rule depend
/// on assertion style rather than on what the test reads.
#[test]
fn source_inspection_stated_only_inside_a_macro_is_detected() {
    let inside_macro = r#"
            fn variants(relative: &str) -> Vec<String> {
                let source = std::fs::read_to_string(root().join(relative)).unwrap();
                vyre_test_support::top_level_variant_names(&source)
            }

            #[test]
            fn every_declared_policy_variant_is_listed() {
                assert_eq!(variants("objective/workload.rs"), listed(Policy::ALL));
            }
        "#;
    let mut findings = Vec::new();
    scan_source_inspection_tests(
        Path::new("driver/tests/policy.rs"),
        inside_macro,
        &mut findings,
    );
    assert_eq!(
        findings.len(),
        1,
        "an inspection inside an assertion macro is still an inspection"
    );
    assert!(findings[0]
        .text
        .contains("every_declared_policy_variant_is_listed"));

    let whole_read_inside_macro = r#"
            #[test]
            fn every_declared_metric_is_listed() {
                assert!(std::fs::read_to_string("objective/metric.rs")
                    .unwrap()
                    .contains("Latency"));
            }
        "#;
    findings.clear();
    scan_source_inspection_tests(
        Path::new("driver/tests/metric.rs"),
        whole_read_inside_macro,
        &mut findings,
    );
    assert_eq!(
        findings.len(),
        1,
        "a read stated inside an assertion macro is still a read"
    );
    assert!(findings[0].text.contains("every_declared_metric_is_listed"));
}

/// WHY: the rule condemns a test that reads this checkout's source instead of
/// asserting behavior. A test that WRITES a source tree in a temporary
/// directory and runs the analyzer over it is asserting behavior, and the
/// analyzer's subject happens to be source text. Every unit test of every
/// source-reading gate was reported as a release blocker on that account: 25 of
/// them, including the tests that prove this scanner itself. A test that writes
/// a fixture and ALSO resolves the checkout keeps the finding, so the fixture
/// cannot be used to cover a walk over the real tree.
#[test]
fn a_test_that_authors_its_own_source_tree_is_not_inspecting_this_checkout() {
    let fixture_only = r#"
            #[cfg(test)]
            mod tests {
                #[test]
                fn a_copied_block_is_measured() {
                    let root = std::env::temp_dir().join("scan-fixture");
                    std::fs::create_dir_all(root.join("crate-a/src")).unwrap();
                    std::fs::write(root.join("crate-a/src/lib.rs"), "let a = 1;\n").unwrap();
                    let text = std::fs::read_to_string(root.join("crate-a/src/lib.rs")).unwrap();
                    assert!(text.contains("let a"));
                }
            }
        "#;
    let fixture_and_checkout = r#"
            #[cfg(test)]
            mod tests {
                #[test]
                fn a_copied_block_is_measured_in_the_tree_too() {
                    let root = std::env::temp_dir().join("scan-fixture");
                    std::fs::create_dir_all(root.join("crate-a/src")).unwrap();
                    std::fs::write(root.join("crate-a/src/lib.rs"), "let a = 1;\n").unwrap();
                    let live = std::fs::read_to_string(
                        crate::checkout::checkout_root().join("crate-a/src/lib.rs"),
                    )
                    .unwrap();
                    assert!(live.contains("let a"));
                }
            }
        "#;

    let mut findings = Vec::new();
    scan_source_inspection_tests(
        Path::new("xtask/src/gates/scan.rs"),
        fixture_only,
        &mut findings,
    );
    assert!(
        findings.is_empty(),
        "a fixture the test wrote is not this checkout: {:?}",
        findings
            .iter()
            .map(|finding| finding.text.clone())
            .collect::<Vec<_>>()
    );

    let root_and_writes_in_separate_helpers = r#"
            fn enforced_schema_shape() -> String {
                std::fs::read_to_string(workspace_root().join("docs/generated/OP_SCHEMA.json"))
                    .unwrap()
            }

            fn write_fixture(root: &std::path::Path) {
                let schema = enforced_schema_shape();
                std::fs::write(root.join("crate-a/src/lib.rs"), schema).unwrap();
            }

            fn fixture() -> tempfile::TempDir {
                let temp = tempfile::tempdir().unwrap();
                write_fixture(temp.path());
                temp
            }

            #[cfg(test)]
            mod tests {
                #[test]
                fn a_stale_claim_fails_closed() {
                    let temp = fixture();
                    let text =
                        std::fs::read_to_string(temp.path().join("crate-a/src/lib.rs")).unwrap();
                    assert!(text.contains("let a"));
                }
            }
        "#;
    scan_source_inspection_tests(
        Path::new("xtask/tests/tree_contracts/architecture_docs.rs"),
        root_and_writes_in_separate_helpers,
        &mut findings,
    );
    assert!(
        findings.is_empty(),
        "a fixture builder that reads a generated artifact from the checkout is not reading this tree's source: {:?}",
        findings
            .iter()
            .map(|finding| finding.text.clone())
            .collect::<Vec<_>>()
    );

    let fixture_borrows_one_tool = r#"
            #[cfg(test)]
            mod tests {
                #[test]
                fn the_extraction_covers_a_gated_module() {
                    let temp = tempfile::tempdir().unwrap();
                    let root = temp.path();
                    std::fs::create_dir_all(root.join("fixture/src")).unwrap();
                    std::fs::write(root.join("fixture/src/lib.rs"), "pub mod public;\n").unwrap();
                    std::fs::copy(workspace_root().join("cargo_full"), root.join("cargo_full"))
                        .unwrap();
                    let snapshot =
                        std::fs::read_to_string(root.join("docs/public-api/fixture.txt")).unwrap();
                    assert!(snapshot.contains("pub mod fixture::public"));
                }
            }
        "#;
    scan_source_inspection_tests(
        Path::new("xtask/tests/tree_contracts/public_api_snapshot_inventory.rs"),
        fixture_borrows_one_tool,
        &mut findings,
    );
    assert!(
        findings.is_empty(),
        "the Rust paths this test names are the fixture's; the one file it takes from the checkout is a tool: {:?}",
        findings
            .iter()
            .map(|finding| finding.text.clone())
            .collect::<Vec<_>>()
    );

    scan_source_inspection_tests(
        Path::new("xtask/src/gates/scan.rs"),
        fixture_and_checkout,
        &mut findings,
    );
    assert_eq!(findings.len(), 1);
    assert!(findings[0]
        .text
        .contains("a_copied_block_is_measured_in_the_tree_too"));
}

/// A macro body reaches the callee graph through raw tokens, and the
/// scanner used to render those tokens to a string and split on every
/// non-identifier character. That split the CONTENTS of string literals,
/// so `assert!(failures.iter().any(|f| f.contains("vyre-scan")))` claimed
/// a call to a local `fn scan`, whose real body reads Rust source, and the
/// pure test that owns that assertion was reported as a release blocker.
/// Punctuation inside a literal is not a call.
#[test]
fn a_string_literal_inside_a_macro_is_not_a_call() {
    let source = r#"
            fn scan(root: &str) -> Vec<String> {
                let text = std::fs::read_to_string("owner.rs").unwrap();
                text.split('\n').map(ToOwned::to_owned).collect()
            }

            fn roster_failures(members: &[String]) -> Vec<String> {
                members.iter().filter(|m| m.starts_with("vyre")).cloned().collect()
            }

            #[cfg(test)]
            mod tests {
                #[test]
                fn a_product_crate_on_the_roster_is_rejected() {
                    let failures = roster_failures(&["vyre-scan".to_string()]);
                    assert!(failures.iter().any(|f| f.contains("vyre-scan")));
                }

                #[test]
                fn a_real_call_inside_a_macro_is_still_seen() {
                    assert!(scan("root").iter().any(|f| f.contains("owner")));
                }
            }
        "#;
    let mut findings = Vec::new();
    scan_source_inspection_tests(Path::new("gate/src/lib.rs"), source, &mut findings);
    let names = findings
        .iter()
        .map(|finding| finding.text.clone())
        .collect::<Vec<_>>();
    assert!(
        !names
            .iter()
            .any(|text| text.contains("a_product_crate_on_the_roster_is_rejected")),
        "a literal naming `vyre-scan` must not resolve to `fn scan`: {names:?}"
    );
    assert!(
        names
            .iter()
            .any(|text| text.contains("a_real_call_inside_a_macro_is_still_seen")),
        "a genuine call written inside a macro must still be followed: {names:?}"
    );
}

#[test]
fn source_inspection_test_scanner_covers_integration_files_and_inline_test_modules() {
    let root = tempfile::tempdir().expect("Fix: scanner fixture root must be creatable.");
    let inline = root.path().join("driver/src/lib.rs");
    let integration = root.path().join("driver/tests/source_contract.rs");
    std::fs::create_dir_all(
        inline
            .parent()
            .expect("Fix: inline scanner fixture must have a parent."),
    )
    .expect("Fix: inline scanner fixture directory must be creatable.");
    std::fs::create_dir_all(
        integration
            .parent()
            .expect("Fix: integration scanner fixture must have a parent."),
    )
    .expect("Fix: integration scanner fixture directory must be creatable.");
    std::fs::write(
        &inline,
        r#"
                #[cfg(test)]
                mod tests {
                    #[test]
                    fn inline_contract() {
                        let source = include_str!("owner.rs");
                        assert!(source.contains("fn helper"));
                    }
                }
            "#,
    )
    .expect("Fix: inline scanner fixture must be writable.");
    std::fs::write(
        &integration,
        r#"
                #[test]
                fn integration_contract() {
                    let source = include_str!("../src/lib.rs");
                    assert!(source.contains("fn helper"));
                }
            "#,
    )
    .expect("Fix: integration scanner fixture must be writable.");

    let mut findings = Vec::new();
    let mut scanned_files = 0;
    scan_root(root.path(), &mut scanned_files, &mut findings);
    scan_source_inspection_test_files(root.path(), &mut scanned_files, &mut findings);

    let source_findings = findings
        .iter()
        .filter(|finding| finding.pattern == "source_inspection_test")
        .collect::<Vec<_>>();
    assert_eq!(
        source_findings.len(),
        2,
        "Fix: the repository scanner must reject source-shape tests in both inline modules and integration-test files."
    );
    // The scanner records a `/`-separated path on every host, so the fixture
    // path is spelled the same way before it is compared.
    let recorded = |path: &std::path::Path| path.to_string_lossy().replace('\\', "/");
    assert!(source_findings
        .iter()
        .any(|finding| finding.path == recorded(&inline)));
    assert!(source_findings
        .iter()
        .any(|finding| finding.path == recorded(&integration)));
}

#[test]
fn source_inspection_test_scanner_rejects_transitive_nested_and_aliased_walks() {
    let forbidden = r#"
            use std::path::{Path, PathBuf};

            #[test]
            fn freezes_architecture_spelling() {
                assert!(collect_sources(Path::new("src")).is_empty());
            }

            struct Helpers;

            impl Helpers {
                fn rust_files(root: &Path) -> Vec<PathBuf> {
                    collect_sources(root)
                }
            }

            fn collect_sources(root: &Path) -> Vec<PathBuf> {
                let mut files = Vec::new();
                for entry in std::fs::read_dir(root).unwrap() {
                    let path = entry.unwrap().path();
                    if path.extension().is_some_and(|extension| extension == "rs") {
                        let source = std::fs::read_to_string(&path).unwrap();
                        if source.contains("fn helper") {
                            files.push(path);
                        }
                    }
                }
                files
            }

            #[test]
            fn unrelated_behavior_remains_allowed() {
                assert_eq!(2 + 2, 4);
            }
        "#;
    let mut findings = Vec::new();
    scan_source_inspection_tests(
        Path::new("driver/tests/source_contract.rs"),
        forbidden,
        &mut findings,
    );

    assert_eq!(findings.len(), 1);
    assert!(findings[0].text.contains("freezes_architecture_spelling"));
}

/// WHY: reading and the `.rs` decision can live in different helpers. The
/// transitive walk must combine both facts or a source-inspection test can hide
/// the forbidden contract behind two individually harmless functions.
#[test]
fn source_inspection_test_scanner_combines_split_read_and_path_facts() {
    let forbidden = r#"
            use std::path::Path;

            #[test]
            fn freezes_architecture_spelling() {
                assert!(inspect(Path::new("src/lib.rs")));
            }

            fn inspect(path: &Path) -> bool {
                let source = load(path);
                is_rust(path) && source.contains("fn helper")
            }

            fn load(path: &Path) -> String {
                std::fs::read_to_string(path).unwrap()
            }

            fn is_rust(path: &Path) -> bool {
                path.extension().is_some_and(|extension| extension == "rs")
            }
        "#;
    let mut findings = Vec::new();
    scan_source_inspection_tests(
        Path::new("driver/tests/source_contract.rs"),
        forbidden,
        &mut findings,
    );

    assert_eq!(findings.len(), 1);
    assert!(findings[0].text.contains("freezes_architecture_spelling"));
}

/// WHY: every fact the scanner collects keyed off a literal ending in `.rs`, so
/// a test that resolves its file from a declaration marker named no path and
/// scanned as inspecting nothing. Two rows declared for that family in the
/// structural-gate registry were reported stale while the tests they name still
/// walk this checkout, which is the silent direction: the gate stops covering
/// the file and asks for the exemption to be deleted.
#[test]
fn source_inspection_resolved_from_a_declaration_marker_is_detected() {
    let resolved = r#"
            #[test]
            fn every_declared_law_has_a_recorded_derivation() {
                let path = vyre_test_support::monorepo::declaring_source_file("pub enum AlgebraicLaw {");
                let source = std::fs::read_to_string(&path).unwrap();
                assert!(source.contains("Commutative"));
            }
        "#;
    let mut findings = Vec::new();
    scan_source_inspection_tests(
        Path::new("vyre-foundation/tests/law_derived_region_alternatives.rs"),
        resolved,
        &mut findings,
    );

    assert_eq!(
        findings.len(),
        1,
        "a path resolved from a marker is still a path into this checkout"
    );
    assert!(findings[0]
        .text
        .contains("every_declared_law_has_a_recorded_derivation"));
}

/// The fixture exemption still applies to a marker-resolved read: a test that
/// authors its own tree and resolves nothing from the checkout is exercising an
/// analyzer, not inspecting this source.
#[test]
fn a_marker_resolver_over_an_authored_tree_is_not_inspecting_this_checkout() {
    let authored = r#"
            #[test]
            fn the_analyzer_finds_the_declaration() {
                let root = tempfile::tempdir().unwrap();
                std::fs::create_dir_all(root.path().join("src")).unwrap();
                std::fs::write(root.path().join("src/lib.rs"), "pub enum Law {}").unwrap();
                let found = analyze(root.path());
                assert!(found.contains("Law"));
            }
        "#;
    let mut findings = Vec::new();
    scan_source_inspection_tests(
        Path::new("xtask/tests/analyzer.rs"),
        authored,
        &mut findings,
    );

    assert!(
        findings.is_empty(),
        "a test that writes the tree it reads asserts behavior: {findings:?}"
    );
}

/// WHY: text inspection was recognised from a hardcoded list of five string
/// methods, written out twice. `match_indices` was in neither copy, so a test
/// that read two emitter dispatch modules and searched each for every declared
/// variant scanned as inspecting nothing, and the registry row declared for it
/// reported as stale.
#[test]
fn a_whole_identifier_search_is_text_inspection() {
    let searching = r#"
            #[test]
            fn emitter_dispatch_names_every_variant() {
                let source = std::fs::read_to_string("vyre-emit-ptx/src/emitter/dispatch.rs").unwrap();
                assert!(names_variant(&source, "LoopCarrierEnd"));
            }

            fn names_variant(content: &str, variant: &str) -> bool {
                content.match_indices(variant).count() > 0
            }
        "#;
    let mut findings = Vec::new();
    scan_source_inspection_tests(
        Path::new("vyre-lower/tests/emitter_decisions.rs"),
        searching,
        &mut findings,
    );

    assert_eq!(
        findings.len(),
        1,
        "searching source for an identifier is inspecting it"
    );
    assert!(findings[0]
        .text
        .contains("emitter_dispatch_names_every_variant"));
}
