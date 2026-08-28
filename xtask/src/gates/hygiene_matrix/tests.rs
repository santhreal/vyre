use super::*;
use std::collections::BTreeSet;
use std::path::Path;

use crate::tree_walk::{self, BUILD_OUTPUT_AND_VCS};

/// WHY: `unbounded_read` matched `fs::read(` as a bare substring, so any
/// type whose name ends in `fs` produced a permanent release blocker on
/// correct code. `BufferRefs::read(count_buffer)` reads a GPU buffer
/// reference: there is no file, no length, and nothing to bound, and the
/// only way to clear the finding would have been to rename the type.
#[test]
fn a_method_on_a_type_ending_in_fs_is_not_a_filesystem_read() {
    assert!(!line_contains_read_call(
        "Node::IndirectDispatch { count_buffer, .. } => BufferRefs::read(count_buffer),"
    ));
    assert!(!line_contains_read_call("let refs = Refs::read(buffer);"));
}

/// The narrowing above must not stop the rule catching a real read.
#[test]
fn every_filesystem_read_spelling_is_still_a_read_call() {
    for line in [
        "let text = fs::read_to_string(path)?;",
        "let text = std::fs::read_to_string(path)?;",
        "let bytes = fs::read(path)?;",
        "let bytes = std::fs::read(path)?;",
        "file.read_to_end(&mut bytes)?;",
        "handle.read_to_string(&mut text)?;",
    ] {
        assert!(line_contains_read_call(line), "missed `{line}`");
    }
}

/// WHY: three spellings of one nanosecond narrowing coexisted in `vyre-bench`,
/// and the crate-local gate that closed them there could not see the same
/// truncating cast in `vyre-foundation` or `vyre-runtime`, where two survived.
/// The rule is whole-tree, so the cast is a release blocker wherever it is
/// written. The dense match is what catches the spelling rustfmt splits across
/// two lines, which a per-line scan reads as a reader with no cast after it.
///
/// What it does not catch: a narrowing routed through a variable, and
/// `as_secs()`, which already answers a `u64`.
#[test]
fn a_truncating_duration_cast_is_found_however_it_is_spelled() {
    let path = Path::new("vyre-foundation/src/perf.rs");
    for source in [
        "let elapsed_ns = self.start.elapsed().as_nanos() as u64;",
        "let millis = span.as_millis() as u32;",
        "let micros = span.as_micros()\n    as u64;",
        "let elapsed_ns = self\n    .start\n    .elapsed()\n    .as_nanos() as u64;",
    ] {
        assert!(
            !truncating_duration_cast_lines(path, source).is_empty(),
            "missed `{source}`"
        );
    }
    for source in [
        "let elapsed_ns = u64::try_from(span.as_nanos()).unwrap_or(u64::MAX);",
        "let clamped = span.as_nanos().min(u128::from(u64::MAX)) as u64;",
        "let seconds = span.as_secs() as u64;",
        "let ratio = span.as_secs_f64() as f32;",
        "// span.as_nanos() as u64 is the spelling this rule forbids",
    ] {
        assert!(
            truncating_duration_cast_lines(path, source).is_empty(),
            "false positive on `{source}`"
        );
    }
}

/// The reported line is the line the cast is written on, not the line the
/// duration reader is written on: a reader whose cast rustfmt moved down reads
/// correct at the reader and wrong at the cast.
#[test]
fn the_reported_line_is_where_the_cast_is_written() {
    let source = "fn wall() -> u64 {\n    let span = start.elapsed();\n    span\n        .as_nanos() as u64\n}\n";
    assert_eq!(
        truncating_duration_cast_lines(Path::new("vyre-runtime/src/clock.rs"), source),
        vec![4]
    );
}

/// WHY: xtask owns the rule and its fixtures, so it spells what the rule
/// forbids. Every other crate is read.
#[test]
fn the_rule_owner_is_exempt_and_nothing_else_is() {
    let source = "let elapsed_ns = start.elapsed().as_nanos() as u64;";
    assert!(truncating_duration_cast_lines(
        Path::new("/w/xtask/src/gates/hygiene_matrix/rules.rs"),
        source
    )
    .is_empty());
    assert_eq!(
        truncating_duration_cast_lines(Path::new("/w/vyre-bench/src/api/metric.rs"), source),
        vec![1]
    );
}

/// WHY: the wrapper rule read every line that spelled a cargo command, so a
/// sentence describing what a build does was a finding a reader could only
/// clear by describing the build less precisely. An instruction is what the
/// rule is about, and the verb that makes it one comes before the command.
/// A sentence the code emits is judged by the same question: three gates
/// that spawn through the one resolver were reported for the diagnostic
/// naming the command that failed to start, while a message printed for a
/// reader to type still names the wrapper.
#[test]
fn a_cargo_command_is_a_finding_when_a_comment_tells_a_reader_to_run_it() {
    for instruction in [
        "//! Run it with `cargo run -p structure-gate`.",
        "/// Regenerate the table with `cargo test -p vyre-driver`.",
        "// rebuild it with `cargo build -p xtask`",
        "let usage = \"cargo xtask gate1\";",
        "println!(\"  cargo run -p {package} -- <subcommand>\");",
        "panic!(\"rerun with `cargo test -p xtask --lib`\");",
        "Command::new(\"bash\").arg(\"-c\").arg(\"cargo test -p xtask\");",
    ] {
        assert!(
            line_contains_raw_workspace_cargo(instruction),
            "missed the instruction `{instruction}`"
        );
    }
    for description in [
        "//! The gates that run a full cargo build of the workspace.",
        "//! `cargo check -p <member>` is what the plain default build gets.",
        "// A cargo test target that does not exist fails before it runs.",
        "//! `cargo check -p <member>` is what the plain default build gets.",
        "format!(\"`cargo test --test {suite}` could not be started: {error}\"),",
        "format!(\"cannot run cargo test for `{package}`: {error}\"),",
        "GateError::new(format!(\"`cargo build` produced no binary\"), advice),",
    ] {
        assert!(
            !line_contains_raw_workspace_cargo(description),
            "read the description `{description}` as an instruction"
        );
    }
}

/// WHY: `cargo +<toolchain> <command>` was matched by a blanket `cargo +`
/// fallback that ran before any exemption, so installing a gate's own
/// dependency on a pinned nightly was a release blocker while the identical
/// `cargo install` line was exempt. The selector chooses a compiler, not a
/// command, and the rule now reads the command underneath it. This does not
/// catch a selector spelled through a shell variable, which no workflow uses.
#[test]
fn a_toolchain_selector_does_not_change_which_cargo_command_a_line_runs() {
    for exempt in [
        "cargo +nightly-2026-08-07 install --locked cargo-public-api --version 0.51.0",
        "cargo +stable install cargo-deny",
        "./cargo_full +nightly build -p xtask",
    ] {
        assert!(
            !line_contains_raw_workspace_cargo(exempt),
            "reported the exempt line `{exempt}`"
        );
    }
    for raw in [
        "cargo +stable build -p xtask",
        "cargo +nightly-2026-08-07 test -p vyre-libs",
        "cargo +nightly public-api",
    ] {
        assert!(
            line_contains_raw_workspace_cargo(raw),
            "missed the raw workspace command `{raw}`"
        );
    }
}

/// WHY: widening the release scan to every xtask source made each gate that
/// detects a stub report itself: the pattern table row `text: "todo!(",` is
/// how the hot-path scan spells the thing it looks for. A string literal
/// names a call and does not make one, which is the same reason a doc
/// comment was already exempt. The call itself must still block.
#[test]
fn a_code_call_named_in_a_literal_is_not_a_call() {
    let rule_row = "        text: \"todo!(\",";
    assert!(
        !line_contains_blocked_pattern(
            Path::new("/w/xtask/src/gates/hot_path_scan.rs"),
            "todo_macro",
            "todo!(",
            rule_row,
            &rule_row.to_ascii_lowercase()
        ),
        "Fix: a pattern table row is a rule definition, not a stub."
    );
    let call = "        todo!(\"finish the lowering\");";
    assert!(
        line_contains_blocked_pattern(
            Path::new("/w/vyre-driver/src/backend/dispatch.rs"),
            "todo_macro",
            "todo!(",
            call,
            &call.to_ascii_lowercase()
        ),
        "Fix: a real todo call must still block the release."
    );
}

/// WHY: two path lists exempt the files that own a rule from the rule. A row
/// naming a file that no longer exists exempts nothing while reading as a
/// decision, which is how an exemption list rots into a lie.
#[test]
fn every_exempted_rule_source_exists() {
    let root = crate::checkout::checkout_root();
    for candidate in HYGIENE_RULE_SOURCES
        .iter()
        .chain(HIDDEN_FALLBACK_GUARD_SOURCES.iter())
    {
        let path = root.join(candidate);
        assert!(
            path.is_file(),
            "Fix: exempted rule source `{candidate}` does not exist; delete the row."
        );
    }
}

/// WHY: `CHANGELOG.md` is generated from `release/changes`, and a released
/// entry records what a version did instead of telling a reader what to
/// run. Scanning it recorded eleven line numbers that every added fragment
/// moved, so the evidence artifact went red for a document nobody edited,
/// and the only place the text could be edited is a fragment that no longer
/// exists. An authored document is still scanned.
#[test]
fn a_generated_release_history_is_recorded_not_instructed() {
    let tree = tempfile::TempDir::new().expect("Fix: create a fixture tree.");
    let bare = "Run `cargo test --workspace` to reproduce it.\n";
    for name in ["README.md", "CHANGELOG.md"] {
        fs::write(tree.path().join(name), bare).expect("Fix: write the fixture document.");
    }
    for (relative, body) in [
        ("CONTRIBUTING.md", "See the README.\n"),
        ("docs/testing/TESTING.toml", "suite = \"none\"\n"),
        ("conform/README.md", "See the README.\n"),
        ("vyre-bench/README.md", "See the README.\n"),
    ] {
        let path = tree.path().join(relative);
        fs::create_dir_all(path.parent().expect("Fix: a fixture path has a parent."))
            .expect("Fix: create the fixture directory.");
        fs::write(path, body).expect("Fix: write the fixture document.");
    }

    let mut scanned = 0usize;
    let mut findings = Vec::new();
    scan_release_docs(tree.path(), &mut scanned, &mut findings);

    let flagged = findings
        .iter()
        .map(|finding| finding.path.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        scanned, 5,
        "Fix: the five authored documents are the scanned set, got {scanned}."
    );
    assert!(
        flagged.iter().any(|path| path.ends_with("README.md")),
        "Fix: an authored document that spells a bare cargo command is still a finding, got {flagged:?}."
    );
    assert!(
        !flagged.iter().any(|path| path.ends_with("CHANGELOG.md")),
        "Fix: the generated release history is not scanned, got {flagged:?}."
    );
}

/// WHY: the release-tooling scan read `.sh`, `.yml` and `.yaml` only, so a
/// rule that blocks a shell heredoc could be satisfied by moving the body
/// into a `.py` beside it, where nothing looked. Seven gate scripts were
/// rewritten that way, which would have moved 1100 lines of release tooling
/// out of scan range in the same change that cleared the findings.
#[test]
fn python_release_tooling_is_scanned_like_shell_release_tooling() {
    let tree = tempfile::TempDir::new().expect("Fix: create a fixture tree.");
    let scripts = tree.path().join("scripts/lib");
    fs::create_dir_all(&scripts).expect("Fix: create the fixture scripts directory.");
    for (name, body) in [
        ("gate.sh", "#!/usr/bin/env bash\ncargo build --workspace\n"),
        (
            "gate.py",
            "import sys\nrun([\"x\"])  # cargo build --workspace\n",
        ),
    ] {
        fs::write(scripts.join(name), body).expect("Fix: write the fixture script.");
    }

    let mut scanned = 0usize;
    let mut findings = Vec::new();
    scan_release_tooling(tree.path(), &mut scanned, &mut findings);

    let scanned_extensions = findings
        .iter()
        .filter(|finding| finding.pattern == "raw_workspace_cargo")
        .filter_map(|finding| {
            Path::new(&finding.path)
                .extension()
                .and_then(|extension| extension.to_str())
                .map(ToString::to_string)
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        scanned_extensions,
        BTreeSet::from(["py".to_string(), "sh".to_string()]),
        "Fix: release tooling written in Python must be scanned like release tooling written in shell; findings={findings:?}"
    );
}

/// WHY: surface classification matched `/docs/` anywhere in the path, so an
/// xtask module grouped under `xtask/src/docs/` was filed as documentation
/// and lost its release-tooling thresholds. What decides the surface is
/// which tree the file lives in, not which subdirectory of that tree.
#[test]
fn xtask_sources_are_release_tooling_whatever_group_holds_them() {
    for path in [
        "/w/xtask/src/main.rs",
        "/w/xtask/src/docs/catalog.rs",
        "/w/xtask/src/release/version_matrix.rs",
        "/w/xtask/src/bench/bench_release.rs",
        "/w/xtask/src/gates/gate1.rs",
    ] {
        assert_eq!(
            hygiene_surface_for_path(Path::new("/w"), path, &BTreeSet::new()),
            "release_tooling",
            "Fix: {path} is xtask source and must carry release-tooling thresholds."
        );
    }
    assert_eq!(
        hygiene_surface_for_path(
            Path::new("/w"),
            "/w/docs/optimization/PASSES.md",
            &BTreeSet::new()
        ),
        "docs",
        "Fix: real documentation must still classify as docs."
    );
    assert_eq!(
        hygiene_surface_for_path(
            Path::new("/w"),
            "/w/vyre-libs/src/docs/loader.rs",
            &BTreeSet::new()
        ),
        "docs",
        "Fix: only the xtask tree is reclassified; other trees keep the docs rule."
    );
}

/// WHY: the release hygiene scan named thirteen xtask command modules by
/// hand and resolved each to a file. A command added beside them was never
/// scanned, and a renamed module kept its row while resolving to nothing,
/// which reads as coverage. The scan walks every xtask crate instead, so
/// the contract is that a command module the tree holds is scanned, and a
/// test source beside it is not.
#[test]
fn every_xtask_command_module_is_scanned_and_no_test_source_is() {
    let root = crate::checkout::checkout_root();
    let mut expected = 0usize;
    for source_root in xtask_source_roots(&root) {
        for entry in tree_walk::pruned(&source_root, BUILD_OUTPUT_AND_VCS).flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
                continue;
            }
            if is_test_source_path(path)
                || path.file_name().and_then(|name| name.to_str()) == Some("hygiene_matrix.rs")
            {
                continue;
            }
            expected += 1;
        }
    }
    let mut scanned = 0usize;
    let mut findings = Vec::new();
    scan_release_xtask(&root, &mut scanned, &mut findings);
    assert!(
        expected > 0,
        "Fix: the xtask crates hold production source; the enumeration found none."
    );
    assert_eq!(
        scanned, expected,
        "Fix: the walk scanned {scanned} of the {expected} xtask production source file(s)."
    );
    for finding in &findings {
        assert!(
            !is_test_source_path(Path::new(&finding.path)),
            "Fix: `{}` is test source and must stay out of the production scan.",
            finding.path
        );
    }
    assert!(
        findings
            .iter()
            .all(|finding| finding.pattern != "unreadable_source_file"),
        "Fix: every file the walk reached must be readable: {:?}",
        findings
            .iter()
            .filter(|finding| finding.pattern == "unreadable_source_file")
            .map(|finding| finding.path.clone())
            .collect::<Vec<_>>()
    );
}

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
    assert!(source_findings
        .iter()
        .any(|finding| finding.path == inline.display().to_string()));
    assert!(source_findings
        .iter()
        .any(|finding| finding.path == integration.display().to_string()));
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

#[test]
fn hidden_fallback_scan_ignores_guard_implementation_text() {
    let guard = Path::new("vyre-lints/src/production_cpu_fallbacks.rs");

    assert!(
        !line_contains_blocked_pattern(
            guard,
            "cpu_fallback",
            "cpu fallback",
            "//! Production CPU fallback guard.",
            "//! production cpu fallback guard.",
        ),
        "Fix: hygiene evidence must not count the guard's own forbidden-token description as shipped fallback behavior."
    );
}

#[test]
fn hidden_fallback_scan_ignores_negated_product_status() {
    let source = Path::new("tools/example-consumer/src/lib.rs");

    assert!(
        !line_contains_blocked_pattern(
            source,
            "cpu_fallback",
            "cpu fallback",
            "status: beta compile-evidence driver; no CPU fallback",
            "status: beta compile-evidence driver; no cpu fallback",
        ),
        "Fix: explicit no-fallback product status text must not be reported as hidden fallback behavior."
    );
}

#[test]
fn hidden_fallback_scan_still_flags_positive_product_fallback() {
    let source = Path::new("surge/surgec/src/scan/pipeline/parse_driver.rs");

    assert!(
        line_contains_blocked_pattern(
            source,
            "cpu_fallback",
            "cpu fallback",
            "CpuRayonParseDriver is a temporary CPU fallback.",
            "cpurayonparsedriver is a temporary cpu fallback.",
        ),
        "Fix: real positive fallback claims must remain visible in release hygiene evidence."
    );
}

#[test]
fn cfg_not_gpu_attr_is_not_a_hidden_fallback_by_itself() {
    let source = Path::new("surge/surgec/src/cmd_scan.rs");

    assert!(
        !line_contains_blocked_pattern(
            source,
            "cfg_not_gpu",
            "cfg(not(feature = \"gpu\"))",
            "#[cfg(not(feature = \"gpu\"))]",
            "#[cfg(not(feature = \"gpu\"))]",
        ),
        "Fix: a fail-closed compile-time GPU feature guard must not be treated as a runtime hidden fallback without fallback behavior."
    );
}

/// A registry with the given `(file, test)` rows, each with a stated reason.
fn structural_gates(rows: &[(&str, &str)]) -> StructuralGateArtifact {
    StructuralGateArtifact {
        schema_version: STRUCTURAL_GATE_SCHEMA_VERSION,
        source: STRUCTURAL_GATE_SOURCE,
        declarations: rows
            .iter()
            .map(|(file, test)| StructuralGateDeclaration {
                file: (*file).to_string(),
                test: (*test).to_string(),
                reason: "no run-time witness".to_string(),
            })
            .collect(),
        blockers: Vec::new(),
    }
}

#[test]
fn hygiene_classifier_separates_test_from_release_blocker() {
    let hot_paths = std::collections::BTreeSet::new();
    let findings = vec![
        HygieneFinding {
            path: "vyre-driver/src/pipeline/mod.rs".to_string(),
            line: 10,
            pattern: "unbounded_read",
            text: "std::fs::read(path)?".to_string(),
            test: None,
        },
        HygieneFinding {
            path: "vyre-driver/tests/pipeline_contracts.rs".to_string(),
            line: 20,
            pattern: "test_ignored",
            text: "#[ignore]".to_string(),
            test: None,
        },
    ];

    let classes = classify_findings(
        Path::new("."),
        &findings,
        &hot_paths,
        &structural_gates(&[]),
        &BTreeSet::new(),
    );

    assert_eq!(classes[0].surface, "production");
    assert_eq!(classes[0].risk, "release_blocker");
    assert!(classes[0].release_blocker);
    assert_eq!(classes[1].surface, "test");
    assert_eq!(classes[1].risk, "test_hygiene");
    assert!(!classes[1].release_blocker);
}

#[test]
fn undeclared_source_inspection_tests_are_release_blockers() {
    let findings = vec![HygieneFinding {
        path: "driver/tests/source_contracts.rs".to_string(),
        line: 7,
        pattern: "source_inspection_test",
        text: "test inspects Rust source text".to_string(),
        test: Some("every_module_is_reachable".to_string()),
    }];

    let classes = classify_findings(
        Path::new("."),
        &findings,
        &std::collections::BTreeSet::new(),
        &structural_gates(&[]),
        &BTreeSet::new(),
    );

    assert_eq!(classes[0].surface, "test");
    assert_eq!(classes[0].risk, "release_blocker");
    assert!(classes[0].release_blocker);
}

/// A declared gate is informational; its neighbour in the same file is not.
///
/// Keying on the file alone would let one reviewed declaration exempt every
/// later source-inspecting test added beside it, which is the cost the
/// declaration exists to charge.
#[test]
fn only_the_declared_source_inspection_test_is_informational() {
    let findings = vec![
        HygieneFinding {
            path: "/repo/driver/tests/source_contracts.rs".to_string(),
            line: 7,
            pattern: "source_inspection_test",
            text: "declared".to_string(),
            test: Some("no_other_file_calls_the_owner".to_string()),
        },
        HygieneFinding {
            path: "/repo/driver/tests/source_contracts.rs".to_string(),
            line: 40,
            pattern: "source_inspection_test",
            text: "undeclared".to_string(),
            test: Some("added_later_without_a_row".to_string()),
        },
    ];

    let classes = classify_findings(
        Path::new("/repo"),
        &findings,
        &std::collections::BTreeSet::new(),
        &structural_gates(&[(
            "driver/tests/source_contracts.rs",
            "no_other_file_calls_the_owner",
        )]),
        &BTreeSet::new(),
    );

    assert_eq!(
        classes[0].risk, "informational",
        "Fix: a reviewed row in {STRUCTURAL_GATE_SOURCE} must exempt the test it names"
    );
    assert!(!classes[0].release_blocker);
    assert_eq!(
        classes[1].risk, "release_blocker",
        "Fix: a source-inspecting test with no reviewed row must block the release"
    );
    assert!(classes[1].release_blocker);
}

/// A row the tree no longer backs is a blocker, not a silent no-op.
#[test]
fn stale_structural_gate_rows_block_the_release() {
    let findings = vec![HygieneFinding {
        path: "/repo/driver/tests/source_contracts.rs".to_string(),
        line: 7,
        pattern: "source_inspection_test",
        text: "declared".to_string(),
        test: Some("still_here".to_string()),
    }];
    let declarations = structural_gates(&[
        ("driver/tests/source_contracts.rs", "still_here"),
        ("driver/tests/source_contracts.rs", "renamed_away"),
        ("driver/tests/deleted_contracts.rs", "gone_with_the_file"),
    ])
    .declarations;

    let blockers = stale_declaration_blockers(Path::new("/repo"), &declarations, &findings);

    assert_eq!(
        blockers.len(),
        2,
        "Fix: a row whose test or file the tree no longer has must block the release; blockers={blockers:?}"
    );
    assert!(
        blockers[0].contains("renamed_away")
            && blockers[0].contains("no longer has a test by that name"),
        "{blockers:?}"
    );
    assert!(
        blockers[1].contains("deleted_contracts.rs")
            && blockers[1].contains("contains no source-inspecting test"),
        "{blockers:?}"
    );
}

/// A module a crate declares under `#[cfg(test)]` compiles only for tests, so its
/// panics are test hygiene even though the file sits under `src`. The classifier
/// learns that from the set of gated files the tree itself declares, so a module
/// that loses its gate is counted against the crate's panic budget again.
#[test]
fn a_cfg_test_gated_source_is_test_hygiene_and_the_same_file_ungated_is_not() {
    let hot_paths = std::collections::BTreeSet::new();
    let findings = vec![HygieneFinding {
        path: "/repo/vyre-libs/src/test_parity_oracles.rs".to_string(),
        line: 44,
        pattern: "panic_macro",
        text: "panic!(\"unsupported oracle width\")".to_string(),
        test: None,
    }];

    let gated = BTreeSet::from(["vyre-libs/src/test_parity_oracles.rs".to_string()]);
    let classes = classify_findings(
        Path::new("/repo"),
        &findings,
        &hot_paths,
        &structural_gates(&[]),
        &gated,
    );
    assert_eq!(classes[0].surface, "test");
    assert_eq!(classes[0].risk, "test_hygiene");
    assert!(!classes[0].release_blocker);

    let classes = classify_findings(
        Path::new("/repo"),
        &findings,
        &hot_paths,
        &structural_gates(&[]),
        &BTreeSet::new(),
    );
    assert_eq!(classes[0].surface, "production");
    assert_ne!(classes[0].risk, "test_hygiene");
}

/// The dedicated test-support crate is test infrastructure even though its code lives under `src`.
#[test]
fn test_support_crate_findings_are_test_hygiene() {
    let hot_paths = std::collections::BTreeSet::new();
    let findings = vec![
        HygieneFinding {
            path: "vyre-test-support/src/consumer_boundary.rs".to_string(),
            line: 161,
            pattern: "panic_macro",
            text: "panic!(\"fixture contract failed\")".to_string(),
            test: None,
        },
        HygieneFinding {
            path: "/repo/vyre-test-support/src/monorepo.rs".to_string(),
            line: 66,
            pattern: "expect_call",
            text: ".expect(\"workspace root\")".to_string(),
            test: None,
        },
    ];

    let classes = classify_findings(
        Path::new("/repo"),
        &findings,
        &hot_paths,
        &structural_gates(&[]),
        &BTreeSet::new(),
    );

    assert!(classes.iter().all(|class| class.surface == "test"));
    assert!(classes.iter().all(|class| class.risk == "test_hygiene"));
    assert!(classes.iter().all(|class| !class.release_blocker));
}

#[test]
fn rust_doc_comment_call_examples_do_not_count_as_production_blockers() {
    assert!(!line_contains_blocked_pattern(
        Path::new("vyre-libs/src/lib.rs"),
        "unwrap_call",
        ".unwrap()",
        "//! let value = fallible().unwrap();",
        "//! let value = fallible().unwrap();",
    ));
}

/// Feature-gated test harness modules remain test infrastructure even when Cargo places them under `src`.
#[test]
fn feature_gated_test_harness_sources_are_test_hygiene() {
    assert_eq!(
        hygiene_surface_for_path(
            Path::new("/repo"),
            "/repo/vyre-driver-cuda/src/test_harness/fake_backend.rs",
            &BTreeSet::new(),
        ),
        "test"
    );
}

#[test]
fn fuzz_targets_are_test_surface_not_release_production() {
    assert_eq!(
        hygiene_surface_for_path(
            Path::new("."),
            "vyre-foundation/fuzz/fuzz_targets/reachability.rs",
            &BTreeSet::new(),
        ),
        "test"
    );
}

#[test]
fn cfg_cpu_parity_attr_is_classified_as_non_release_item() {
    assert!(is_non_release_cfg_attr(
        "#[cfg(any(test, feature = \"cpu-parity\"))]"
    ));
    assert!(is_non_release_cfg_attr(
        "#[cfg(any(test, feature = \"legacy-infallible\"))]"
    ));
    assert!(!is_non_release_cfg_attr("#[cfg(feature = \"serde\")]"));
}

#[test]
fn stacked_cfg_after_test_attr_still_counts_as_test_body() {
    let mut findings = Vec::new();
    let mut scanned_files = 0;
    let dir =
        std::env::temp_dir().join(format!("vyre-hygiene-stacked-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("test temp dir");
    let path = dir.join("stacked_test.rs");
    std::fs::write(
        &path,
        "#[test]\n#[cfg(feature = \"gpu\")]\nfn generated_e2e() {\n    fallible().expect(\"test-only assertion\");\n}\n",
    )
    .expect("write stacked test fixture");
    scan_file(&path, &mut scanned_files, &mut findings);
    std::fs::remove_file(&path).ok();
    std::fs::remove_dir(&dir).ok();
    assert_eq!(scanned_files, 1);
    assert!(
        findings.is_empty(),
        "stacked #[test] + #[cfg] function body must not be release hygiene"
    );
}

#[test]
fn hygiene_classifier_marks_hot_path_debt_as_release_blocker() {
    let hot_paths = std::collections::BTreeSet::from([
        "vyre-runtime/src/resident_work_queue/ring.rs".to_string(),
    ]);
    let findings = vec![HygieneFinding {
        path: "vyre-runtime/src/resident_work_queue/ring.rs".to_string(),
        line: 12,
        pattern: "TODO",
        text: "// TODO: remove allocation".to_string(),
        test: None,
    }];

    let classes = classify_findings(
        Path::new("."),
        &findings,
        &hot_paths,
        &structural_gates(&[]),
        &BTreeSet::new(),
    );

    assert!(classes[0].hot_path);
    assert_eq!(classes[0].risk, "release_blocker");
}

#[test]
fn hidden_fallback_guard_source_is_identified_for_gpu_skip_phrases() {
    assert!(is_hidden_fallback_guard_source(Path::new(
        "vyre-lints/src/gpu_skip_guards.rs"
    )));
}

#[test]
fn required_cargo_wrapper_is_tool_owned() {
    let workspace = tempfile::TempDir::new()
        .expect("Fix: create temp workspace for cargo wrapper hygiene test.");
    let vyre_root = workspace.path().join("vyre");
    fs::create_dir_all(&vyre_root)
        .expect("Fix: create temp vyre root for cargo wrapper hygiene test.");
    fs::write(vyre_root.join("cargo_full"), b"#!/usr/bin/env bash\n")
        .expect("Fix: write temp cargo_full wrapper for hygiene test.");

    let mut findings = Vec::new();
    check_required_cargo_wrappers(&vyre_root, &mut findings);

    assert!(
        findings.is_empty(),
        "Fix: Vyre release hygiene must require the tool-owned bounded cargo wrapper; findings={findings:?}"
    );
}

/// A production `panic!` whose enclosing function documents `# Panics` is a declared
/// contract, not release debt.
///
/// Vyre ships infallible wrappers over `try_*` twins because the quiet alternative
/// (an empty match set, an empty table) reports a dirty input as clean (Law 10). The
/// gate reads Rust's own `# Panics` section so there is no second allowlist to rot.
#[test]
fn documented_panic_contract_is_recognized() {
    let source = "\
/// Pack a haystack.
///
/// # Panics
/// Panics when the haystack exceeds the u32 ABI.
pub fn pack(haystack: &[u8]) -> Vec<u8> {
    panic!(\"nope\")
}
";
    let panic_line = source
        .lines()
        .position(|line| line.contains("panic!("))
        .expect("Fix: keep the panic site in the documented-contract fixture.");

    assert!(
        has_documented_panic_contract(source, panic_line),
        "Fix: a panic inside a function documenting `# Panics` must not be a release blocker."
    );
}

/// Restricted visibility may contain spaces inside `pub(in path)`.
///
/// The panic-contract walk must parse that visibility without widening the
/// function to `pub(crate)` merely to make its documented contract visible.
#[test]
fn documented_panic_contract_recognizes_restricted_visibility() {
    let source = "\
/// Build a pattern program.
///
/// # Panics
/// Panics when the fallible builder rejects its input.
pub(in crate::pattern) fn build() -> Program {
    inner().expect(\"pattern program must build\")
}
";
    let site = source
        .lines()
        .position(|line| line.contains(".expect("))
        .expect("Fix: keep the expect site in the restricted-visibility fixture.");

    assert!(
        has_documented_panic_contract(source, site),
        "Fix: `pub(in path)` must remain visible to the panic-contract scanner."
    );
}

/// An undocumented production panic stays a release blocker.
///
/// This is the whole point of reading the docs: if the contract is not written down,
/// a caller cannot know the call can abort, and the panic is debt.
#[test]
fn undocumented_panic_is_not_a_contract() {
    let source = "\
/// Pack a haystack.
pub fn pack(haystack: &[u8]) -> Vec<u8> {
    panic!(\"nope\")
}
";
    let panic_line = source
        .lines()
        .position(|line| line.contains("panic!("))
        .expect("Fix: keep the panic site in the undocumented fixture.");

    assert!(
        !has_documented_panic_contract(source, panic_line),
        "Fix: an undocumented panic must remain a release blocker."
    );
}

/// Attributes and plain `//` notes between the doc block and the signature must not
/// hide the contract.
///
/// `// INTENTIONAL: ...` above `#[allow(clippy::expect_used)]` is the house style for
/// a deliberate panic; a walk that stopped at the first non-doc line reported both
/// `vyre-grammar-gen` DFA builders as blockers even though each documents `# Panics`.
#[test]
fn documented_contract_survives_attributes_and_plain_comments() {
    let source = "\
/// Build the lexer DFA.
///
/// # Panics
/// Panics when a compile-time pattern is invalid.
// INTENTIONAL: the pattern table is a constant; a failure is a broken build.
#[must_use]
#[allow(clippy::expect_used)]
pub fn build() -> Dfa {
    inner().expect(\"constant patterns must compile\")
}
";
    let site = source
        .lines()
        .position(|line| line.contains(".expect("))
        .expect("Fix: keep the expect site in the attribute fixture.");

    assert!(
        has_documented_panic_contract(source, site),
        "Fix: attributes and plain comments between docs and signature must not hide a `# Panics` contract."
    );
}

/// A `# Panics` section on a neighbouring function must not exempt an undocumented one.
///
/// The walk back looks for the ENCLOSING signature. If it drifted past the function
/// it started in, one documented panic anywhere in a file would silence the rest.
#[test]
fn documented_contract_does_not_leak_to_the_next_function() {
    let source = "\
/// Documented.
///
/// # Panics
/// Panics on bad input.
pub fn documented() {
    unreachable!()
}

pub fn undocumented() {
    panic!(\"nope\")
}
";
    let site = source
        .lines()
        .position(|line| line.contains("panic!("))
        .expect("Fix: keep the panic site in the leak fixture.");

    assert!(
        !has_documented_panic_contract(source, site),
        "Fix: a `# Panics` section on an earlier function must not exempt a later one."
    );
}

/// Braces inside string and character literals must not terminate a cfg(test) module early.
///
/// The hygiene scan previously treated `split("}\n}")` in an inline test as two closing
/// module braces, then reported the remaining test assertions as production panic blockers.
#[test]
fn brace_depth_ignores_literal_and_comment_delimiters() {
    assert_eq!(
        update_brace_depth(1, r#"let _ = source.split("}\n}").next();"#),
        1
    );
    assert_eq!(update_brace_depth(1, "let brace = '}';"), 1);
    assert_eq!(update_brace_depth(1, "call(); // }"), 1);
    assert_eq!(update_brace_depth(1, "if ready {"), 2);
    let mut raw = BraceDepthState::with_depth(1);
    raw.update("let artifact = br#\"{");
    raw.update("  \"nested\": {");
    raw.update("}\"#;");
    assert_eq!(raw.depth, 1);
}

/// Every spelling of a test cfg gates the item out of the production scan.
///
/// The scan used to list four exact predicate spellings, so
/// `#[cfg(all(test, feature = \"...\"))]` (how the regex scan suites gate themselves)
/// was treated as production source and four `mod tests` blocks had their helpers
/// reported as release blockers.
#[test]
fn every_test_cfg_spelling_is_non_release() {
    for attribute in [
        "#[cfg(test)]",
        "#[cfg(any(test, feature = \"cpu-parity\"))]",
        "#[cfg(all(test, feature = \"pattern-regex\", feature = \"pattern-dfa\"))]",
        "#[cfg(all(feature = \"pattern-regex\", test))]",
    ] {
        assert!(
            is_non_release_cfg_attr(attribute),
            "Fix: `{attribute}` gates the item to test builds and must be excluded from the production hygiene scan."
        );
    }
}

/// `not(test)` and feature-only gates stay in the production scan.
///
/// `#[cfg(not(test))]` is the OPPOSITE gate: that code ships. Treating it as test-only
/// would blind the scan to exactly the production paths it exists to check.
#[test]
fn production_cfg_attributes_stay_in_scope() {
    for attribute in [
        "#[cfg(not(test))]",
        "#[cfg(feature = \"cuda\")]",
        "#[cfg(target_os = \"linux\")]",
        "#[derive(Debug)]",
    ] {
        assert!(
            !is_non_release_cfg_attr(attribute),
            "Fix: `{attribute}` does not gate the item to test builds and must stay in the production hygiene scan."
        );
    }
}

mod threshold_policy_contracts {
    use super::*;

    fn valid_row() -> ThresholdPolicyTomlRow {
        ThresholdPolicyTomlRow {
            id: "fixture".to_string(),
            path: "src/fixture.rs".to_string(),
            name: "FIXTURE_THRESHOLD".to_string(),
            unit: "items".to_string(),
            provenance: "measured fixture".to_string(),
            config_tier: "tier_a".to_string(),
            override_path: "compiled default -> tool.toml -> CLI override".to_string(),
            evidence_link: THRESHOLD_POLICY_ARTIFACT.to_string(),
            release_rule: "VX-475".to_string(),
        }
    }

    /// A blank required field must remain a release blocker so malformed policy data cannot pass through the rules-as-data gate.
    #[test]
    fn malformed_threshold_policy_rows_are_rejected() {
        let mut row = valid_row();
        row.unit.clear();
        let mut blockers = Vec::new();

        validate_threshold_policy_row(&row, &mut blockers);

        assert_eq!(
            blockers,
            vec![
                "docs/optimization/THRESHOLD_POLICY.toml row `fixture` has blank unit. Fix: every threshold policy row must carry unit, provenance, tier, override, evidence, and VX ownership."
            ]
        );
    }

    /// A valid Tier A row must stay accepted so the malformed-fixture proof does not reject correctly governed operator thresholds.
    #[test]
    fn valid_threshold_policy_rows_are_accepted() {
        let mut blockers = Vec::new();

        validate_threshold_policy_row(&valid_row(), &mut blockers);

        assert_eq!(blockers, Vec::<String>::new());
    }

    /// An unknown tier must fail even when every descriptive field is present, because an unclassified threshold has no override contract.
    #[test]
    fn unknown_threshold_policy_tiers_are_rejected() {
        let mut row = valid_row();
        row.config_tier = "runtime".to_string();
        let mut blockers = Vec::new();

        validate_threshold_policy_row(&row, &mut blockers);

        assert_eq!(
            blockers,
            vec![
                "docs/optimization/THRESHOLD_POLICY.toml row `fixture` uses config_tier `runtime`. Fix: use `tier_a`, `tier_b`, or `structural`."
            ]
        );
    }

    /// A structural threshold must reject operator overrides because changing a wire or ABI bound requires compatibility review.
    #[test]
    fn structural_threshold_policy_rejects_operator_overrides() {
        let mut row = valid_row();
        row.config_tier = "structural".to_string();
        let mut blockers = Vec::new();

        validate_threshold_policy_row(&row, &mut blockers);

        assert_eq!(
            blockers,
            vec![
                "docs/optimization/THRESHOLD_POLICY.toml row `fixture` is structural but override_path does not say `not operator configurable`. Fix: separate wire/ABI bounds from runtime knobs."
            ]
        );
    }
}

/// WHY: the xtask tooling is split across `xtask` and the `xtask-*` crates
/// that link vyre, and three separate rules key off that: the surface a file
/// is classified under, the owner lane it is attributed to, and whether the
/// generic source walk skips it. Each rule used to match the literal string
/// `xtask`, so moving a module into a sibling crate reclassified it as
/// production source under production thresholds. The crate list is read out
/// of the workspace manifest at run time, so a fourth xtask crate turns this
/// red instead of quietly inheriting the wrong rules.
#[test]
fn every_xtask_crate_carries_the_release_tooling_rules() {
    let manifest = fs::read_to_string(crate::checkout::checkout_root().join("Cargo.toml"))
        .expect("Fix: the workspace manifest must be readable");
    let crates: Vec<String> = manifest
        .lines()
        .filter_map(|line| {
            line.trim()
                .strip_prefix('"')?
                .strip_suffix("\",")
                .map(str::to_string)
        })
        .filter(|member| member == "xtask" || member.starts_with("xtask-"))
        .collect();
    assert!(
        crates.len() >= 3,
        "expected the xtask family in the workspace roster, found {crates:?}"
    );
    for member in &crates {
        let source = format!("/w/{member}/src/gates/some_gate.rs");
        assert_eq!(
            hygiene_surface_for_path(Path::new("/w"), &source, &BTreeSet::new()),
            "release_tooling",
            "Fix: {source} is xtask source and must carry release-tooling thresholds."
        );
        assert_eq!(
            hygiene_owner_lane_for_path(&source),
            "testing_evidence",
            "Fix: {source} is xtask source and must be owned by testing_evidence."
        );
        assert!(
            is_xtask_tree_directory(member),
            "Fix: the generic source walk must skip `{member}`, which the \
             release xtask scan already reads."
        );
    }
}

/// WHY: `is_xtask_source_path` gates the unbounded-read exemption, so a match
/// that is too loose exempts production source from the read cap. A crate
/// merely named `xtask-...` outside its `src` tree, and an unrelated crate
/// whose path happens to contain the word, must both stay unexempt.
#[test]
fn the_xtask_source_match_does_not_leak_past_the_src_tree() {
    for exempt in [
        "/w/xtask/src/gates/a.rs",
        "/w/xtask-registry/src/gates/a.rs",
        "/w/xtask-evidence/src/release/a.rs",
    ] {
        assert!(is_xtask_source_path(exempt), "`{exempt}` must be exempt");
    }
    for not_exempt in [
        "/w/xtask-registry/tests/a.rs",
        "/w/xtask-registry/build.rs",
        "/w/vyre-libs/src/xtask-notes/a.rs",
        "/w/vyre-libs/src/a.rs",
    ] {
        assert!(
            !is_xtask_source_path(not_exempt),
            "`{not_exempt}` is not xtask source and must keep the read cap"
        );
    }
}

/// WHY: a panic that is neither documented nor on a hot path was bounded by
/// nothing, and the answer has to fail in three directions or it is an
/// allowlist. Over the ceiling is new debt, a crate with no row at all is a
/// crate nobody decided about, and a ceiling left above a crate that reached
/// zero is what covers the next panic added there. Improvement short of zero
/// is a note, because a gate that fails on the improvement it asks for is a
/// gate somebody switches off.
#[test]
fn the_panic_ceiling_fails_over_unrecorded_and_stale_and_only_notes_slack() {
    let (_directory, root) = crate::gates::fixture_checkout::checkout(&[(
            "docs/testing/PANIC_BUDGET.toml",
            "schema = 1\n\n[[crate_budget]]\nname = \"over\"\nceiling = 1\n\n[[crate_budget]]\nname = \"slack\"\nceiling = 3\n\n[[crate_budget]]\nname = \"stale\"\nceiling = 2\n",
        )]);
    let class = |path: &str, pattern: &'static str, surface: &'static str, blocker: bool| {
        HygieneFindingClass {
            path: root.join(path).display().to_string(),
            line: 1,
            pattern,
            owner_lane: "testing_evidence",
            surface,
            risk: if blocker {
                "release_blocker"
            } else {
                "informational"
            },
            hot_path: blocker,
            release_blocker: blocker,
        }
    };
    let classes = vec![
        class("over/src/a.rs", "panic_macro", "production", false),
        class("over/src/b.rs", "unwrap_call", "production", false),
        class("slack/src/a.rs", "expect_call", "release_tooling", false),
        class("unrecorded/src/a.rs", "expect_call", "production", false),
        // Neither of these is this ratchet's population: one is documented,
        // the other is already a release blocker and counted as one.
        class(
            "over/src/c.rs",
            "documented_panic_contract",
            "production",
            false,
        ),
        class("over/src/d.rs", "panic_macro", "production", true),
    ];

    let budget = collect_panic_budget(&root, &classes);
    let blockers = budget.blockers.join("\n");
    assert!(
        blockers.contains("over carries 2 undocumented panic(s)")
            && blockers.contains("ceiling of 1"),
        "over the ceiling has to block: {blockers}"
    );
    assert!(
        blockers.contains("unrecorded carries 1") && blockers.contains("records no ceiling for it"),
        "a crate with no row has to block: {blockers}"
    );
    assert!(
        blockers.contains("ceiling of 2 for stale, which now carries none"),
        "a ceiling above a crate that reached zero has to block: {blockers}"
    );
    assert_eq!(
        budget.notes.len(),
        1,
        "slack is one note, not a blocker: {:?}",
        budget.notes
    );
    assert!(
        budget.notes[0].contains("slack carries 1") && budget.notes[0].contains("to 1"),
        "the note carries the number to write: {:?}",
        budget.notes
    );
    assert_eq!(
        budget
            .rows
            .iter()
            .map(|row| (row.crate_name.as_str(), row.ceiling, row.measured))
            .collect::<Vec<_>>(),
        [("over", 1, 2), ("slack", 3, 1), ("stale", 2, 0)],
        "every recorded row carries what the tree measured against it"
    );
}
