use std::fs;
use std::process::Command;

#[test]
fn flags_consumer_name_in_platform_rust_doc_comment() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("vyre-libs/src/security");
    fs::create_dir_all(&src).expect("create src");
    fs::write(
        src.join("mod.rs"),
        "//! surgec-facing op surface.\n\npub fn taint_flow() {}\n",
    )
    .expect("write fixture");

    let violations =
        vyre_lints::run_consumer_coupling(&[src.as_path()]).expect("consumer coupling scan");
    assert_eq!(violations.len(), 1);
    assert_eq!(
        violations[0].kind,
        vyre_lints::ViolationKind::ConsumerCoupling
    );
    assert!(violations[0].message.contains("surgec"));
    assert!(violations[0].message.contains("Fix:"));
}

#[test]
fn flags_consumer_name_in_current_markdown() {
    let dir = tempfile::tempdir().expect("tempdir");
    let docs = dir.path().join("docs");
    fs::create_dir_all(&docs).expect("create docs");
    // The fixture must literally contain a banned consumer name: this test proves the
    // lint FLAGS it. A rename sweep once rewrote this string to "external-dataflow",
    // which left nothing for the lint to find, so the test failed while the lint was
    // working correctly. Do not neutralize a negative fixture.
    fs::write(
        docs.join("ARCHITECTURE.md"),
        "The foundation scheduler owns weir-dataflow phases.\n",
    )
    .expect("write fixture");

    let violations =
        vyre_lints::run_consumer_coupling(&[docs.as_path()]).expect("consumer coupling scan");
    assert_eq!(violations.len(), 1);
    assert!(violations[0].message.contains("weir"));
}

#[test]
fn ignores_archived_markdown_consumer_history() {
    let dir = tempfile::tempdir().expect("tempdir");
    let archive = dir.path().join("docs/archive");
    fs::create_dir_all(&archive).expect("create archive");
    fs::write(
        archive.join("OLD_PLAN.md"),
        "Historical notes can mention keyhog, gossan, surgec, and weir.\n",
    )
    .expect("write fixture");

    let violations =
        vyre_lints::run_consumer_coupling(&[dir.path()]).expect("consumer coupling scan");
    assert!(violations.is_empty());
}

/// Release runbooks may name the products in the combined release train.
///
/// A release tag such as `vyre-0.4.1-weir-0.0.1` is a literal string the operator
/// types into `git tag`, not a description of a capability. A blanket substitution
/// over these files previously rewrote tags, paths, and the `vyre-release-gate`
/// subcommand into phrases containing spaces, producing instructions that cannot be
/// run. This test locks the exemption so that repair cannot be undone by the next
/// well-meaning rename sweep.
#[test]
fn allows_release_train_product_names_in_release_runbooks() {
    let dir = tempfile::tempdir().expect("tempdir");
    let docs = dir.path().join("docs");
    let release = docs.join("release");
    fs::create_dir_all(&release).expect("create docs/release");

    for (path, body) in [
        (
            docs.join("RELEASE.md"),
            "git tag vyre-0.4.1-weir-0.0.1\ngit push origin vyre-v0.4.1 vyre-0.4.1-weir-0.0.1\n",
        ),
        (
            docs.join("RELEASE_CHECKLIST.md"),
            "- [ ] Create `weir-v0.0.1`.\n",
        ),
        (
            docs.join("RELEASE_ENGINEERING.md"),
            "Weir tag: `weir-v0.0.1`.\n",
        ),
        (
            docs.join("PUBLISH_GATE.md"),
            "| weir | 0.0.1 | crates.io | Weir integration evidence |\n",
        ),
        (
            release.join("v0.4.1.md"),
            "Final tags: vyre-v0.4.1, weir-v0.0.1, vyre-0.4.1-weir-0.0.1.\n",
        ),
    ] {
        fs::write(path, body).expect("write fixture");
    }

    let violations =
        vyre_lints::run_consumer_coupling(&[dir.path()]).expect("consumer coupling scan");
    assert!(
        violations.is_empty(),
        "release runbooks must not be flagged, got: {:?}",
        violations.iter().map(|v| &v.message).collect::<Vec<_>>()
    );
}

/// The release-runbook exemption must not leak into ordinary documentation.
///
/// `docs/RELEASE.md` is exempt; `docs/RELEASING_GUIDE.md`, `docs/ARCHITECTURE.md`, and
/// a nested `docs/guide/release-notes.md` are not. Without this test, a suffix match
/// like `ends_with("RELEASE.md")` or a substring match on "release" would silently
/// exempt half the documentation tree.
#[test]
fn release_runbook_exemption_does_not_leak_to_neighbouring_docs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let docs = dir.path().join("docs");
    let guide = docs.join("guide");
    fs::create_dir_all(&guide).expect("create docs/guide");

    for name in ["ARCHITECTURE.md", "RELEASING_GUIDE.md", "PRE_RELEASE.md"] {
        fs::write(docs.join(name), "The scheduler owns weir-dataflow phases.\n")
            .expect("write fixture");
    }
    fs::write(
        guide.join("release-notes.md"),
        "The scheduler owns weir-dataflow phases.\n",
    )
    .expect("write fixture");

    let violations =
        vyre_lints::run_consumer_coupling(&[dir.path()]).expect("consumer coupling scan");
    let flagged: Vec<&str> = violations.iter().map(|v| v.file.as_str()).collect();

    assert_eq!(
        violations.len(),
        4,
        "every non-runbook doc must still be flagged, got {flagged:?}"
    );
    for expected in [
        "ARCHITECTURE.md",
        "RELEASING_GUIDE.md",
        "PRE_RELEASE.md",
        "release-notes.md",
    ] {
        assert!(
            flagged.iter().any(|file| file.ends_with(expected)),
            "{expected} must be flagged, got {flagged:?}"
        );
    }
}

/// Rust source is never exempt, including under a release-shaped module path.
///
/// The exemption is about operator runbooks. `vyre-self-substrate/src/integration/
/// release/*.rs` is platform code that happens to sit under a `release/` directory,
/// and naming a consumer there is exactly the API coupling the guard exists to stop.
#[test]
fn release_runbook_exemption_never_covers_rust_source() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("vyre-self-substrate/src/integration/release");
    fs::create_dir_all(&src).expect("create src");
    fs::write(
        src.join("launch.rs"),
        "//! Launch sequence for weir publish receipts.\n\npub fn launch() {}\n",
    )
    .expect("write fixture");

    let violations =
        vyre_lints::run_consumer_coupling(&[dir.path()]).expect("consumer coupling scan");
    assert_eq!(violations.len(), 1);
    assert!(violations[0].message.contains("weir"));
    assert!(violations[0].message.contains("comment"));
}

#[test]
fn ignores_consumer_name_in_non_comment_rust_code() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("vyre-libs/src");
    fs::create_dir_all(&src).expect("create src");
    fs::write(
        src.join("scan.rs"),
        "pub fn keyhog_counter() -> usize { 1 }\n",
    )
    .expect("write fixture");

    let violations =
        vyre_lints::run_consumer_coupling(&[src.as_path()]).expect("consumer coupling scan");
    assert!(violations.is_empty());
}

#[test]
fn flags_consumer_name_in_platform_rust_string_literal() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("vyre-libs/src/security");
    fs::create_dir_all(&src).expect("create src");
    fs::write(
        src.join("diagnostic.rs"),
        "pub fn diagnostic() -> &'static str { \"keyhog scanner path\" }\n",
    )
    .expect("write fixture");

    let violations =
        vyre_lints::run_consumer_coupling(&[src.as_path()]).expect("consumer coupling scan");
    assert_eq!(violations.len(), 1);
    assert_eq!(
        violations[0].kind,
        vyre_lints::ViolationKind::ConsumerCoupling
    );
    assert!(violations[0].message.contains("string literal"));
    assert!(violations[0].message.contains("keyhog"));
}

#[test]
fn flags_consumer_name_in_platform_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("vyre-libs/src/security/surgec_bridge");
    fs::create_dir_all(&src).expect("create src");
    fs::write(src.join("mod.rs"), "pub fn neutral_code() {}\n").expect("write fixture");

    let violations =
        vyre_lints::run_consumer_coupling(&[dir.path()]).expect("consumer coupling scan");
    assert_eq!(violations.len(), 1);
    assert_eq!(
        violations[0].kind,
        vyre_lints::ViolationKind::ConsumerCoupling
    );
    assert!(violations[0].message.contains("path"));
    assert!(violations[0].message.contains("surgec"));
}

#[test]
fn cli_rejects_missing_consumer_coupling_root() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = dir.path().join("missing-docs");
    let output = Command::new(env!("CARGO_BIN_EXE_vyre-lints"))
        .arg("--check-consumer-coupling")
        .arg("--consumer-root")
        .arg(&missing)
        .output()
        .expect("run vyre-lints");

    assert!(
        !output.status.success(),
        "missing consumer coupling roots must fail, not shrink scan coverage"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("consumer coupling root not found"),
        "missing-root diagnostic must be actionable, got: {stderr}"
    );
}
