//! Published documentation reference contract tests.
//!
//! WHY: a backticked path and a copyable command are claims a reader acts on, and
//! neither is checked by Markdown link validation. This target drives the
//! `docs-references` gate over fixture trees, one fixture per way a claim can
//! resolve to nothing, plus the real checkout so the gate cannot pass fixtures
//! while covering no published document.

#![forbid(unsafe_code)]

mod workspace_sources;

use std::fs;
use std::path::Path;
use std::process::Command;

use tempfile::TempDir;

use workspace_sources::workspace_root;
use xtask::gate::{Gate, GateCtx, Report};
use xtask::gates::docs_references::DocsReferences;

fn judge(root: &Path) -> Report {
    DocsReferences
        .run(&GateCtx::new(root.to_path_buf(), Vec::new()))
        .expect("Fix: the gate must be able to read the fixture tree")
}

/// Every finding rendered as one line, for substring assertions.
fn rendered(report: &Report) -> String {
    report
        .findings
        .iter()
        .map(|finding| {
            format!(
                "{} {}",
                finding
                    .file
                    .as_deref()
                    .map(Path::to_string_lossy)
                    .unwrap_or_default(),
                finding.message
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn fixture() -> TempDir {
    let root = tempfile::tempdir().expect("Fix: temporary fixture directory must be writable");
    fs::create_dir_all(root.path().join("docs")).unwrap();
    fs::write(
        root.path().join("Cargo.toml"),
        "[workspace]\nresolver = \"2\"\nmembers = []\n",
    )
    .unwrap();
    fs::write(root.path().join("README.md"), "# Fixture\n").unwrap();
    let status = Command::new("git")
        .args(["init", "-q"])
        .current_dir(root.path())
        .status()
        .expect("Fix: git must launch for ignored-path fixture semantics");
    assert!(status.success());
    root
}

/// Locks out regressions where the gate passes fixtures but no longer checks the real published corpus.
#[test]
fn current_workspace_references_resolve_to_published_inputs() {
    let report = judge(&workspace_root());
    assert_eq!(report.count(), 0, "{}", rendered(&report));
    assert!(
        report
            .notes
            .iter()
            .any(|note| note.contains("path-like code span")),
        "the gate must report how much it read: {:?}",
        report.notes
    );
}

/// Prevents a backticked nonexistent repository path from evading ordinary Markdown link validation.
#[test]
fn missing_path_like_code_span_fails_closed() {
    let root = fixture();
    fs::write(
        root.path().join("docs/guide.md"),
        "Read `docs/missing-input.md` before running the command.\n",
    )
    .unwrap();

    let report = judge(root.path());
    let text = rendered(&report);
    assert_eq!(report.count(), 1, "{text}");
    assert!(
        text.contains("docs/guide.md code span `docs/missing-input.md` resolves to `docs/missing-input.md`, which the checkout does not carry"),
        "{text}"
    );
}

/// Prevents runnable examples from naming an input file that a clean checkout cannot provide.
#[test]
fn missing_command_input_fails_closed() {
    let root = fixture();
    fs::write(
        root.path().join("docs/guide.md"),
        "```bash\ncat docs/missing.json\n```\n",
    )
    .unwrap();

    let report = judge(root.path());
    let text = rendered(&report);
    assert_eq!(report.count(), 1, "{text}");
    assert!(
        text.contains("docs/guide.md command `docs/missing.json` resolves to `docs/missing.json`"),
        "{text}"
    );
}

/// Prevents a locally present but unpublished input from making public instructions appear reproducible.
#[test]
fn gitignored_input_fails_closed() {
    let root = fixture();
    fs::write(root.path().join(".gitignore"), "docs/private.json\n").unwrap();
    fs::write(root.path().join("docs/private.json"), "{}\n").unwrap();
    fs::write(
        root.path().join("docs/guide.md"),
        "Load `docs/private.json`.\n",
    )
    .unwrap();

    let report = judge(root.path());
    let text = rendered(&report);
    assert_eq!(report.count(), 1, "{text}");
    assert!(
        text.contains("an ignore rule excludes from the checkout"),
        "{text}"
    );
}

/// Keeps generated destinations usable in examples while still checking the published input beside them.
#[test]
fn output_destinations_do_not_need_to_preexist() {
    let root = fixture();
    fs::create_dir_all(root.path().join("inputs")).unwrap();
    fs::write(root.path().join("inputs/table.json"), "{}\n").unwrap();
    fs::write(
        root.path().join("docs/guide.md"),
        "```bash\ntool emit --out-dir ./generated/ --lr-json inputs/table.json\ntool emit --output docs/generated.json --lr-json inputs/table.json\n```\n",
    )
    .unwrap();

    let report = judge(root.path());
    assert_eq!(report.count(), 0, "{}", rendered(&report));
}

/// Ensures wildcard evidence references prove that at least one published artifact exists.
#[test]
fn wildcard_input_requires_a_published_match() {
    let root = fixture();
    fs::write(
        root.path().join("docs/guide.md"),
        "Inspect `docs/evidence-*.json`.\n",
    )
    .unwrap();

    let missing = judge(root.path());
    let text = rendered(&missing);
    assert_eq!(missing.count(), 1, "{text}");
    assert!(text.contains("matches nothing published"), "{text}");

    fs::write(root.path().join("docs/evidence-01.json"), "{}\n").unwrap();
    let present = judge(root.path());
    assert_eq!(present.count(), 0, "{}", rendered(&present));
}

/// Keeps archived evidence from becoming an executable public-input contract after it is superseded.
#[test]
fn archived_documents_are_not_scanned_as_current_instructions() {
    let root = fixture();
    fs::create_dir_all(root.path().join("docs/archive")).unwrap();
    fs::write(
        root.path().join("docs/archive/old.md"),
        "Historical input: `docs/removed.json`.\n",
    )
    .unwrap();

    let report = judge(root.path());
    assert_eq!(report.count(), 0, "{}", rendered(&report));
}

/// Prevents generated per-crate guides from bypassing the same clean-checkout input contract as root docs.
#[test]
fn workspace_crate_readmes_are_scanned() {
    let root = fixture();
    fs::create_dir_all(root.path().join("crate-a")).unwrap();
    fs::write(
        root.path().join("Cargo.toml"),
        "[workspace]\nresolver = \"2\"\nmembers = [\"crate-a\"]\n",
    )
    .unwrap();
    fs::write(
        root.path().join("crate-a/Cargo.toml"),
        "[package]\nname = \"crate-a\"\nversion = \"0.0.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    fs::write(
        root.path().join("crate-a/README.md"),
        "Run with `scripts/missing-driver.sh`.\n",
    )
    .unwrap();

    let report = judge(root.path());
    let text = rendered(&report);
    assert_eq!(report.count(), 1, "{text}");
    assert!(
        text.contains("crate-a/README.md code span `scripts/missing-driver.sh`"),
        "{text}"
    );
}

/// Preserves source line anchors while validating the published file that carries the evidence.
#[test]
fn source_line_selectors_resolve_the_underlying_file() {
    let root = fixture();
    fs::create_dir_all(root.path().join("src")).unwrap();
    fs::write(root.path().join("src/model.rs"), "pub struct Model;\n").unwrap();
    fs::write(
        root.path().join("docs/guide.md"),
        "Inspect `src/model.rs:7-9` for the bounded source claim.\n",
    )
    .unwrap();

    let report = judge(root.path());
    assert_eq!(report.count(), 0, "{}", rendered(&report));
}

/// Prevents superseded prose from becoming an executable input contract.
#[test]
fn superseded_manifest_documents_are_not_scanned() {
    let root = fixture();
    fs::write(
        root.path().join("docs/old.md"),
        "# Old\n\nUse `docs/removed.json`.\n",
    )
    .unwrap();
    fs::write(
        root.path().join("docs/DOCS.toml"),
        "[[page]]\npath = \"old.md\"\nstatus = \"superseded\"\n",
    )
    .unwrap();

    let report = judge(root.path());
    assert_eq!(report.count(), 0, "{}", rendered(&report));
}

/// Keeps crate-local source and test paths relative to that crate even when the workspace has the same top-level directory name.
#[test]
fn crate_readme_relative_paths_resolve_inside_the_crate() {
    let root = fixture();
    fs::create_dir_all(root.path().join("crate-a/tests")).unwrap();
    fs::create_dir_all(root.path().join("tests")).unwrap();
    fs::write(
        root.path().join("Cargo.toml"),
        "[workspace]\nresolver = \"2\"\nmembers = [\"crate-a\"]\n",
    )
    .unwrap();
    fs::write(
        root.path().join("crate-a/Cargo.toml"),
        "[package]\nname = \"crate-a\"\nversion = \"0.0.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    fs::write(
        root.path().join("crate-a/tests/behavior.rs"),
        "#[test] fn behavior() {}\n",
    )
    .unwrap();
    fs::write(
        root.path().join("crate-a/README.md"),
        "The exact target is `tests/behavior.rs`.\n",
    )
    .unwrap();

    let report = judge(root.path());
    assert_eq!(report.count(), 0, "{}", rendered(&report));
}

/// A device or system path is not something this repository can publish, and
/// reporting it as a missing repository input names a defect the document does
/// not have.
#[test]
fn absolute_paths_outside_the_checkout_are_not_repository_claims() {
    let root = fixture();
    fs::write(
        root.path().join("docs/guide.md"),
        "The old script discarded stderr to `/dev/null`, and `/etc/hostname` is not ours.\n",
    )
    .unwrap();

    let report = judge(root.path());
    assert_eq!(report.count(), 0, "{}", rendered(&report));
}

/// An absolute path that lands inside the checkout is still a claim about a
/// published input, so it must not become an escape hatch from the check.
#[test]
fn absolute_paths_inside_the_checkout_are_still_checked() {
    let root = fixture();
    let absolute = root
        .path()
        .canonicalize()
        .expect("Fix: the fixture root must resolve")
        .join("docs/absent.md");
    fs::write(
        root.path().join("docs/guide.md"),
        format!("The input is `{}`.\n", absolute.display()),
    )
    .unwrap();

    let report = judge(
        &root
            .path()
            .canonicalize()
            .expect("Fix: the fixture root must resolve"),
    );
    let text = rendered(&report);
    assert_eq!(report.count(), 1, "{text}");
    assert!(text.contains("docs/absent.md"), "{text}");
}

/// A reference that climbs above the checkout is a claim this repository cannot
/// satisfy, and it is reported as leaving rather than as merely absent, because
/// the two need different corrections. One `..` from `docs/` still lands inside
/// the checkout, so the escape needs two.
#[test]
fn a_reference_above_the_checkout_is_reported_as_leaving_it() {
    let root = fixture();
    fs::write(
        root.path().join("docs/guide.md"),
        "The sibling tree is at `../../elsewhere/table.json`.\n",
    )
    .unwrap();

    let report = judge(root.path());
    let text = rendered(&report);
    assert_eq!(report.count(), 1, "{text}");
    assert!(text.contains("outside this repository"), "{text}");
}

/// A crate document names a module the way the crate's own source tree does,
/// relative to `src/`, so `builder/range_ordering.rs` in a member's
/// `ARCHITECTURE.md` is that member's `src/builder/range_ordering.rs`. Reading
/// it against the member directory alone reported every module heading of every
/// crate architecture document as a path the checkout does not carry.
#[test]
fn a_crate_document_resolves_a_module_against_its_own_src() {
    let root = fixture();
    fs::create_dir_all(root.path().join("vyre-thing/src/builder")).unwrap();
    fs::write(
        root.path().join("vyre-thing/Cargo.toml"),
        "[package]\nname = \"vyre-thing\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    fs::write(
        root.path().join("vyre-thing/src/builder/range_ordering.rs"),
        "pub fn ordered() {}\n",
    )
    .unwrap();
    fs::write(
        root.path().join("vyre-thing/ARCHITECTURE.md"),
        "### `builder/range_ordering.rs`\nSorted-range helpers.\n",
    )
    .unwrap();

    let report = judge(root.path());
    assert_eq!(report.count(), 0, "{}", rendered(&report));
}

/// The `src/` reading is a reading, not an exemption: a module heading naming a
/// file no crate carries is still reported, and the report names the path the
/// crate's own layout makes of it.
#[test]
fn a_crate_document_naming_an_absent_module_is_reported() {
    let root = fixture();
    fs::create_dir_all(root.path().join("vyre-thing/src")).unwrap();
    fs::write(
        root.path().join("vyre-thing/Cargo.toml"),
        "[package]\nname = \"vyre-thing\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    fs::write(
        root.path().join("vyre-thing/ARCHITECTURE.md"),
        "### `builder/absent.rs`\nSorted-range helpers.\n",
    )
    .unwrap();

    let report = judge(root.path());
    let text = rendered(&report);
    assert_eq!(report.count(), 1, "{text}");
    assert!(
        text.contains("resolves to `vyre-thing/src/builder/absent.rs`"),
        "{text}"
    );
}

/// A crate document naming a workspace path that no longer exists is reported
/// at the path it wrote. Reading a crate document against the member's `src/`
/// found modules named by module path, and it also turned every deleted
/// workspace path in a crate document into a finding at a path no document
/// carries, which sends the reader looking for a file nobody ever wrote.
#[test]
fn a_crate_document_naming_an_absent_root_path_is_reported_where_it_wrote_it() {
    let root = fixture();
    fs::create_dir_all(root.path().join("vyre-thing/src")).unwrap();
    fs::write(
        root.path().join("vyre-thing/Cargo.toml"),
        "[package]\nname = \"vyre-thing\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    fs::write(
        root.path().join("vyre-thing/ARCHITECTURE.md"),
        "The table lives in `docs/absent-table.json`.\n",
    )
    .unwrap();

    let report = judge(root.path());
    let text = rendered(&report);
    assert_eq!(report.count(), 1, "{text}");
    assert!(
        text.contains("resolves to `docs/absent-table.json`"),
        "{text}"
    );
}
