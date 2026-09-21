//! Which integration-test files may not share a harness binary, decided once.
//!
//! Cargo links one binary per integration-test file, and this workspace declares
//! over thirteen hundred of them. Grouping a crate's files into one harness per
//! required-feature set is what bounds that, and it is only safe for a test that
//! does not mutate process-global state: an environment write, the working
//! directory, a panic hook, or a one-time initialization another case observes.
//!
//! The candidate set is derived from source on every run rather than listed in
//! the decision file, so a test added tomorrow that touches one of those APIs
//! turns the sweep red until a row records the decision for it. A hardcoded list
//! of members goes stale in silence, which is the same failure as having no rule.
//!
//! A match is not a defect. Most of them are benign: a `OnceLock` holding a
//! file's own fixture, an errno constant read from a syscall result. What the
//! rule requires is that somebody said which, in writing, beside the reason.
//!
//! # What it does not catch
//!
//! The scan reads source text, so a process-global mutation reached through a
//! helper outside the package's `tests` subtree is invisible here. It answers
//! whether a decision exists for every file that names one of these APIs, not
//! whether the decision is right.
//!
//! Whether a `grouped` row is true is a question only the suite answers. Two of
//! them claimed a duplicate-registration fixture was benign to share a binary
//! with, and twelve sibling cases failed on the poisoned registry before anyone
//! read the rows again. A textual rule for that case was tried and withdrawn: a
//! duplicate registration is poison in a registry that rejects one and harmless
//! in a registry that folds it, and the file does not say which.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::{self, Code, CodeCursor, Tree};

/// The decision record every matching test file needs a row in.
const DECISIONS: &str = "xtask/test-harness-isolation.toml";

/// Source text that makes a test a grouping candidate, with what it reaches.
///
/// Each entry is the API itself rather than a category name, so a finding says
/// what the file touched. `Once` is spelled with its path because the bare word
/// appears in prose constantly.
///
/// Backend enumeration and acquisition are here because their answer is the set
/// of registrations the binary linked, so a case asserting on that answer is
/// coupled to every sibling that registers a fixture. Looking one operation up
/// by id is not: the registry is append-only and a lookup reads its own key.
const PROCESS_GLOBAL_APIS: &[(&str, &str)] = &[
    ("panic::set_hook", "installs a process-wide panic hook"),
    ("take_hook", "takes the process-wide panic hook"),
    ("set_var", "writes the process environment"),
    ("remove_var", "clears a process environment variable"),
    ("set_current_dir", "changes the process working directory"),
    ("process::abort", "aborts the process"),
    ("atexit", "registers a process exit handler"),
    ("OnceLock", "holds one-time initialization"),
    ("OnceCell", "holds one-time initialization"),
    ("std::sync::Once", "holds one-time initialization"),
    (
        "inventory::submit",
        "submits a registration into a process-global inventory",
    ),
    (
        "registered_backends",
        "reads the process-global backend registry",
    ),
    (
        "acquire_preferred_dispatch_backend",
        "acquires from the process-global backend registry",
    ),
    ("libc::", "calls libc directly"),
];

/// The two decisions a row may record.
const GROUPINGS: &[&str] = &["isolated", "grouped"];

/// Every integration test that names a process-global API has a recorded decision.
pub struct TestHarnessIsolation;

impl crate::gate::GateBehavior for TestHarnessIsolation {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let tests: Vec<PathBuf> = tree
            .all_rust()
            .into_iter()
            .filter(|path| is_integration_test(path))
            .collect();
        report.cover_complete("integration test source files", tests.len());
        if let Some(note) = tree.absence_note() {
            report.note(note);
        }
        if tests.is_empty() {
            return Err(GateError::new(
                "no integration test file found",
                "run this gate inside the workspace checkout; a scan over an empty test set \
                 reports success forever",
            ));
        }

        let mut matches: BTreeMap<String, Vec<&'static str>> = BTreeMap::new();
        for path in &tests {
            let text = tree.read(path)?;
            let reached: Vec<&'static str> = PROCESS_GLOBAL_APIS
                .iter()
                .filter(|(api, _)| text.contains(api))
                .map(|(_, what)| *what)
                .collect();
            if !reached.is_empty() {
                matches.insert(as_key(path), reached);
            }
        }
        report.note(format!(
            "{} test file(s) name a process-global API",
            matches.len()
        ));

        let declared = decisions(&tree)?;
        for (path, reached) in &matches {
            let Some(grouping) = declared.get(path) else {
                let mut unique: Vec<&str> = reached.clone();
                unique.sort_unstable();
                unique.dedup();
                report.find(Finding::at(
                    PathBuf::from(path),
                    1,
                    format!(
                        "no grouping decision recorded; this test {}",
                        unique.join(", ")
                    ),
                    "add a row to xtask/test-harness-isolation.toml stating `isolated` with the \
                     state it mutates, or `grouped` with why the match is benign",
                ));
                continue;
            };
            if !GROUPINGS.contains(&grouping.as_str()) {
                report.find(Finding::at(
                    PathBuf::from(DECISIONS),
                    1,
                    format!("`{path}` records grouping `{grouping}`"),
                    "record `isolated` or `grouped`; no other decision has a meaning here",
                ));
            }
        }
        for path in declared.keys() {
            if matches.contains_key(path) {
                continue;
            }
            let corrective = if tree.has(path) {
                "delete the row: the file no longer names a process-global API, so the decision \
                 reviews nothing"
            } else {
                "delete the row, or repoint it at the path the test moved to"
            };
            report.find(Finding::at(
                PathBuf::from(DECISIONS),
                1,
                format!("`{path}` has a grouping decision and is not a candidate"),
                corrective,
            ));
        }
        report_duplicate_includes(&tree, &tests, &mut report)?;
        Ok(report)
    }
}

/// A fixture that registers into an inventory is linked once per test binary.
///
/// `#[path]` does not import a module, it compiles the file again at a second
/// place in the tree. For a fixture holding `inventory::submit!` that is a
/// second registration of the same id in the same process, and a registry that
/// rejects one answers every later reader in the binary with the rejection
/// instead of the table. The decision record cannot see this: each including
/// file names the API through the include and gets its own row, and every row
/// can be individually true while the set of them is not.
///
/// A binary is a test file no other file includes, and its contents are what a
/// walk from it reaches, which is what Cargo links. Two binaries in one crate
/// may each include the same fixture; that is two processes and two tables.
fn report_duplicate_includes(
    tree: &Tree,
    tests: &[PathBuf],
    report: &mut Report,
) -> Result<(), GateError> {
    let mut includes: BTreeMap<PathBuf, Vec<PathBuf>> = BTreeMap::new();
    for path in tests {
        includes.insert(path.clone(), included_paths(&tree.read(path)?, path));
    }
    let included: BTreeSet<&PathBuf> = includes.values().flatten().collect();
    // A cargo test target is a file directly under `tests/`. A deeper file that
    // nothing includes is unreachable code, not a second binary, and walking it
    // would attribute a duplicate to a binary that does not exist.
    let roots: Vec<&PathBuf> = tests
        .iter()
        .filter(|path| !included.contains(path) && is_target_root(path))
        .collect();

    for root in roots {
        let mut visits: BTreeMap<&PathBuf, usize> = BTreeMap::new();
        let mut stack = vec![root];
        while let Some(current) = stack.pop() {
            let Some((_, targets)) = includes.get_key_value(current) else {
                continue;
            };
            for target in targets {
                let count = visits.entry(target).or_default();
                *count += 1;
                // Walk a target once: a second visit is the finding, and
                // following it again would multiply its own subtree.
                if *count == 1 {
                    stack.push(target);
                }
            }
        }
        for (target, count) in visits {
            if count < 2 || !tree.read(target)?.contains("inventory::submit!") {
                continue;
            }
            report.find(Finding::at(
                target.clone(),
                1,
                format!(
                    "submits an inventory registration and is included {count} times by the \
                     `{}` binary",
                    as_key(root)
                ),
                "declare the fixture once in the binary's root file and have the suites reach \
                 it through `crate::`, so the registration is submitted once",
            ));
        }
    }
    Ok(())
}

/// Every `#[path = "..."]` target in one file, resolved against its directory.
///
/// The scan is over code, not text. A file that documents the attribute, or
/// builds one as a format template, carries the token in a comment or a string
/// literal; taking the next quote after it read `{}` out of a diagnostic and
/// asked the tree for `tests/{}`. A mention with no literal after it at all is
/// worse than a crash, because the next quote anywhere later in the file
/// resolves to an unrelated path the gate then attributes as an include.
fn included_paths(text: &str, from: &Path) -> Vec<PathBuf> {
    /// How far after `#[path` its literal may start. Enough for whitespace and
    /// `=`, short enough that an unrelated later literal is not the value.
    const ATTRIBUTE_SPAN: usize = 64;

    let directory = from.parent().unwrap_or(Path::new(""));
    let mut targets = Vec::new();
    let mut pending: Option<usize> = None;
    let mut cursor = CodeCursor::new(text);
    while let Some((at, span)) = cursor.step() {
        match span {
            Code::Opaque(piece) => {
                if scan::is_comment_span(piece) {
                    continue;
                }
                let Some(start) = pending.take() else {
                    continue;
                };
                let literal = piece
                    .strip_prefix('"')
                    .and_then(|rest| rest.strip_suffix('"'));
                if let Some(literal) = literal {
                    if at.saturating_sub(start) <= ATTRIBUTE_SPAN {
                        targets.push(resolved_module_path(directory, literal));
                    }
                }
            }
            Code::Byte(_) => {
                if text[at..].starts_with("#[path") {
                    pending = Some(at);
                }
                cursor.seek(at + text[at..].chars().next().map_or(1, char::len_utf8));
            }
        }
    }
    targets
}

/// A `#[path]` value against the including file's own directory.
///
/// `../` collapses rather than staying in the key, so two spellings of one file
/// do not read as two files.
fn resolved_module_path(directory: &Path, literal: &str) -> PathBuf {
    let mut resolved = directory.to_path_buf();
    for part in Path::new(literal).components() {
        match part {
            std::path::Component::ParentDir => {
                resolved.pop();
            }
            std::path::Component::CurDir => {}
            other => resolved.push(other),
        }
    }
    resolved
}

/// The declared grouping of every path the decision record names.
///
/// A record carrying no `[[test]]` row is zero decisions, not a malformed file:
/// every candidate is then reported as undecided, which is the state a checkout
/// is in before anyone reviews it. A row that is present but malformed is still
/// an error, because a half-written decision states nothing.
fn decisions(tree: &Tree) -> Result<BTreeMap<String, String>, GateError> {
    let table = tree.read_toml(DECISIONS)?;
    let empty = toml::value::Array::new();
    let rows = match table.get("test") {
        Some(value) => value.as_array().ok_or_else(|| {
            GateError::new(
                format!("{DECISIONS} declares `test` as something other than an array"),
                "declare each decision as a `[[test]]` row with `path`, `grouping` and `reason`",
            )
        })?,
        None => &empty,
    };
    let mut declared = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for row in rows {
        let path = string_field(row, "path")?;
        let grouping = string_field(row, "grouping")?;
        let reason = string_field(row, "reason")?;
        if reason.trim().is_empty() {
            return Err(GateError::new(
                format!("{DECISIONS} row `{path}` states an empty reason"),
                "state why the test is isolated, or why its match is benign",
            ));
        }
        if !seen.insert(path.clone()) {
            return Err(GateError::new(
                format!("{DECISIONS} names `{path}` twice"),
                "keep one row per test file; two rows can disagree",
            ));
        }
        declared.insert(path, grouping);
    }
    Ok(declared)
}

/// One required string field of a decision row.
fn string_field(row: &toml::Value, field: &str) -> Result<String, GateError> {
    row.get(field)
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .ok_or_else(|| {
            GateError::new(
                format!("{DECISIONS} has a row without a string `{field}`"),
                format!("give every `[[test]]` row a string `{field}`"),
            )
        })
}

/// Whether a path is compiled into an integration-test binary.
///
/// A cargo integration target is a `.rs` file directly under a package's `tests`
/// directory, but the code it links is the whole subtree: a module a target root
/// includes runs in that binary and mutates the same process. Keying only the
/// roots hid the exact case this gate exists for, an `inventory::submit!` two
/// directories down that every sibling test in the binary then observed. Every
/// `.rs` file under a package's `tests` directory is a candidate, at any depth.
fn is_integration_test(path: &Path) -> bool {
    if path.extension().and_then(|value| value.to_str()) != Some("rs") {
        return false;
    }
    let components: Vec<&str> = path
        .iter()
        .filter_map(|component| component.to_str())
        .collect();
    let Some(index) = components
        .iter()
        .position(|component| *component == "tests")
    else {
        return false;
    };
    !components[..index].contains(&"src")
}

/// Whether a path is a cargo test target root: a file directly under `tests`.
fn is_target_root(path: &Path) -> bool {
    path.parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        == Some("tests")
}

/// The decision-record spelling of a path.
fn as_key(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::GateBehavior;
    use crate::gates::fixture_checkout::{checkout, files, messages};

    /// The decision record a fixture checkout carries, with the rows given.
    fn record(rows: &str) -> String {
        format!("schema_version = 1\n{rows}")
    }

    /// WHY: the include scan took the next quote after the token `#[path`
    /// wherever it appeared, so a file documenting the attribute or building
    /// one as a format template contributed a path nobody wrote. On this tree
    /// that read `{}` out of a diagnostic and failed the gate on
    /// `vyre-test-support/tests/{}`; the quieter form is a mention with no
    /// literal near it, which picks up an unrelated literal later in the file
    /// and attributes a real path as an include of a binary that never had it.
    #[test]
    fn a_path_attribute_named_in_prose_or_a_template_is_not_an_include() {
        let source = "//! Rust resolves a `#[path]` inside the including file.\n\
                      #[path = \"real.rs\"]\n\
                      mod real;\n\
                      fn render(value: &str) -> String {\n    \
                      format!(\"#[path = \\\"{}\\\"]\", value)\n}\n\
                      const LATER: &str = \"unrelated.rs\";\n";

        assert_eq!(
            included_paths(source, Path::new("pkg/tests/root.rs")),
            [PathBuf::from("pkg/tests/real.rs")],
            "only the attribute written as code contributes an include"
        );
    }

    /// WHY: this is the mutation the gate exists for. A test that writes the
    /// process environment cannot share a binary, and adding one without a
    /// decision is exactly how the grouping in row 74 would break a suite that
    /// stays green. The neighbour with a decision must stay unreported, or the
    /// gate rejects the record it just read.
    #[test]
    fn a_candidate_without_a_decision_is_reported_and_a_decided_one_is_not() {
        let (_directory, root) = checkout(&[
            (
                "pkg/tests/decided.rs",
                "#[test]\nfn it_runs() {\n    std::env::set_var(\"FLAG\", \"1\");\n}\n",
            ),
            (
                "pkg/tests/undecided.rs",
                "#[test]\nfn it_runs() {\n    std::env::set_current_dir(\"/\").unwrap();\n}\n",
            ),
            ("pkg/tests/plain.rs", "#[test]\nfn it_runs() {}\n"),
            (
                DECISIONS,
                &record(
                    "[[test]]\npath = \"pkg/tests/decided.rs\"\ngrouping = \"isolated\"\n\
                     reason = \"writes FLAG into the process environment\"\n",
                ),
            ),
        ]);

        let report = TestHarnessIsolation
            .run(&GateCtx::new(root, Vec::new()))
            .expect("Fix: the gate must read the fixture tree; check the fixture git step");
        let reported = files(&report);
        assert_eq!(
            reported,
            ["pkg/tests/undecided.rs"],
            "only the candidate with no decision is reported: {:?}",
            messages(&report)
        );
    }

    /// WHY: a decision that outlives its subject overstates what was reviewed,
    /// which is how the unsafe budget came to name three crates that no longer
    /// existed. A row for a file that stopped being a candidate must fail.
    #[test]
    fn a_decision_for_something_that_is_not_a_candidate_is_reported() {
        let (_directory, root) = checkout(&[
            ("pkg/tests/plain.rs", "#[test]\nfn it_runs() {}\n"),
            (
                DECISIONS,
                &record(
                    "[[test]]\npath = \"pkg/tests/plain.rs\"\ngrouping = \"grouped\"\n\
                     reason = \"the OnceLock it used to hold is gone\"\n\
                     [[test]]\npath = \"pkg/tests/deleted.rs\"\ngrouping = \"isolated\"\n\
                     reason = \"installed a panic hook\"\n",
                ),
            ),
        ]);

        let report = TestHarnessIsolation
            .run(&GateCtx::new(root, Vec::new()))
            .expect("Fix: the gate must read the fixture tree; check the fixture git step");
        let messages = messages(&report);
        assert_eq!(
            messages.len(),
            2,
            "both stale rows are reported: {messages:?}"
        );
        assert!(
            messages.iter().any(|message| message.contains("plain.rs"))
                && messages
                    .iter()
                    .any(|message| message.contains("deleted.rs")),
            "the surviving file and the missing one are both named: {messages:?}"
        );
    }

    /// WHY: the defect this gate shipped with. A target root is a linked binary,
    /// but a module under it runs in that binary and mutates the same process,
    /// and keying only the roots let an `inventory::submit!` in
    /// `cert_regression_pin/test_operation.rs` register a fixture op that a
    /// sibling test in the same harness then read as a shipped op. Every depth
    /// under a package `tests` directory is a candidate; nothing under `src` is.
    #[test]
    fn a_module_under_tests_is_a_candidate_at_any_depth() {
        assert!(is_integration_test(&PathBuf::from("pkg/tests/case.rs")));
        assert!(is_integration_test(&PathBuf::from(
            "pkg/tests/support/case.rs"
        )));
        assert!(is_integration_test(&PathBuf::from(
            "pkg/tests/a/b/c/case.rs"
        )));
        assert!(!is_integration_test(&PathBuf::from("pkg/src/lib.rs")));
        assert!(!is_integration_test(&PathBuf::from(
            "pkg/src/thing/tests/case.rs"
        )));
        assert!(!is_integration_test(&PathBuf::from(
            "pkg/tests/fixture.toml"
        )));
    }

    /// WHY: a decision keyed to a target root cannot review a mutation that a
    /// module two directories down performs, and the gate must report the file
    /// that names the API rather than the binary it happens to link into.
    #[test]
    fn a_deep_module_needs_its_own_decision() {
        let (_directory, root) = checkout(&[
            (
                "pkg/tests/all_tests.rs",
                "#[path = \"fixtures/op.rs\"]\npub mod op;\n",
            ),
            (
                "pkg/tests/fixtures/op.rs",
                "inventory::submit! { Op::new() }\n",
            ),
            (DECISIONS, &record("")),
        ]);

        let report = TestHarnessIsolation
            .run(&GateCtx::new(root, Vec::new()))
            .expect("Fix: the gate must read the fixture tree; check the fixture git step");
        assert_eq!(
            files(&report),
            ["pkg/tests/fixtures/op.rs"],
            "the module that submits the registration is the one reported: {:?}",
            messages(&report)
        );
    }
}

#[cfg(test)]
mod include_graph_tests {
    use super::*;
    use crate::gate::GateBehavior;
    use crate::gates::fixture_checkout::{checkout, files, messages};

    /// The fixture every case here includes, and the row that decides it.
    const FIXTURE: &str = "inventory::submit! { Resolver::new(\"echo\") }\n";

    /// A decision record whose rows mark every path given as `grouped`.
    fn grouped(paths: &[&str]) -> String {
        let rows: String = paths
            .iter()
            .map(|path| {
                format!(
                    "[[test]]\npath = \"{path}\"\ngrouping = \"grouped\"\n\
                     reason = \"reaches the fixture resolver\"\n"
                )
            })
            .collect();
        format!("schema_version = 1\n{rows}")
    }

    /// WHY: this is the defect the rule was written from. Four suites in
    /// `vyre-foundation` each carried `#[path]` to one fixture holding an
    /// `inventory::submit!`, all four linked into one `all_tests` binary, and
    /// `OpaqueExprResolver` rejected the second registration. Eleven wire tests
    /// then failed on the poisoned table rather than on anything they asserted.
    /// Every including file had a truthful `grouped` row, so the decision record
    /// could not express it: the rule is over the set, not any one row.
    #[test]
    fn a_registering_fixture_included_twice_by_one_binary_is_reported() {
        let (_directory, root) = checkout(&[
            (
                "pkg/tests/all_tests.rs",
                "#[path = \"first.rs\"]\nmod first;\n#[path = \"deep/second.rs\"]\nmod second;\n",
            ),
            (
                "pkg/tests/first.rs",
                "#[path = \"support/echo.rs\"]\nmod echo;\n",
            ),
            (
                "pkg/tests/deep/second.rs",
                "#[path = \"../support/echo.rs\"]\nmod echo;\n",
            ),
            ("pkg/tests/support/echo.rs", FIXTURE),
            (DECISIONS, &grouped(&["pkg/tests/support/echo.rs"])),
        ]);

        let report = TestHarnessIsolation
            .run(&GateCtx::new(root, Vec::new()))
            .expect("Fix: the gate must read the fixture tree; check the fixture git step");
        assert_eq!(
            files(&report),
            ["pkg/tests/support/echo.rs"],
            "the fixture is reached twice from one root, once through `../`, and both \
             spellings must resolve to the one file: {:?}",
            messages(&report)
        );
    }

    /// WHY: the repair must satisfy the rule, and the shape it asks for is the
    /// one the fix used: the root owns the include and the suites reach it
    /// through `crate::`. A gate that still reports the tree it recommends
    /// cannot be satisfied and would be worked around instead of obeyed.
    #[test]
    fn one_include_owned_by_the_root_is_not_reported() {
        let (_directory, root) = checkout(&[
            (
                "pkg/tests/all_tests.rs",
                "#[path = \"support/echo.rs\"]\npub mod echo;\n\
                 #[path = \"first.rs\"]\nmod first;\n#[path = \"deep/second.rs\"]\nmod second;\n",
            ),
            ("pkg/tests/first.rs", "use crate::echo::Resolver;\n"),
            ("pkg/tests/deep/second.rs", "use crate::echo::Resolver;\n"),
            ("pkg/tests/support/echo.rs", FIXTURE),
            (DECISIONS, &grouped(&["pkg/tests/support/echo.rs"])),
        ]);

        let report = TestHarnessIsolation
            .run(&GateCtx::new(root, Vec::new()))
            .expect("Fix: the gate must read the fixture tree; check the fixture git step");
        assert!(
            report.findings.is_empty(),
            "the recommended shape must pass: {:?}",
            messages(&report)
        );
    }

    /// WHY: two test targets are two processes and two registries, which is the
    /// arrangement the isolated driver fixtures rely on. Counting includes per
    /// crate instead of per binary would report it and push the repair toward
    /// sharing state across binaries that must not share it.
    #[test]
    fn two_binaries_each_including_the_fixture_once_are_not_reported() {
        let (_directory, root) = checkout(&[
            (
                "pkg/tests/alpha.rs",
                "#[path = \"support/echo.rs\"]\nmod echo;\n",
            ),
            (
                "pkg/tests/beta.rs",
                "#[path = \"support/echo.rs\"]\nmod echo;\n",
            ),
            ("pkg/tests/support/echo.rs", FIXTURE),
            (DECISIONS, &grouped(&["pkg/tests/support/echo.rs"])),
        ]);

        let report = TestHarnessIsolation
            .run(&GateCtx::new(root, Vec::new()))
            .expect("Fix: the gate must read the fixture tree; check the fixture git step");
        assert!(
            report.findings.is_empty(),
            "each binary registers once, which is the per-process contract: {:?}",
            messages(&report)
        );
    }

    /// WHY: the rule is about a registration surviving into a shared process,
    /// not about `#[path]`. Reporting a twice-included helper that submits
    /// nothing would make the common case noisy, and a gate whose findings are
    /// mostly noise stops being read.
    #[test]
    fn a_twice_included_file_that_registers_nothing_is_not_reported() {
        let (_directory, root) = checkout(&[
            (
                "pkg/tests/all_tests.rs",
                "#[path = \"first.rs\"]\nmod first;\n#[path = \"second.rs\"]\nmod second;\n",
            ),
            (
                "pkg/tests/first.rs",
                "#[path = \"support/helper.rs\"]\nmod helper;\n",
            ),
            (
                "pkg/tests/second.rs",
                "#[path = \"support/helper.rs\"]\nmod helper;\n",
            ),
            (
                "pkg/tests/support/helper.rs",
                "pub(crate) fn program() -> u32 {\n    7\n}\n",
            ),
            (DECISIONS, &grouped(&[])),
        ]);

        let report = TestHarnessIsolation
            .run(&GateCtx::new(root, Vec::new()))
            .expect("Fix: the gate must read the fixture tree; check the fixture git step");
        assert!(
            report.findings.is_empty(),
            "a duplicated helper with no registration leaves nothing in the process: {:?}",
            messages(&report)
        );
    }
}
