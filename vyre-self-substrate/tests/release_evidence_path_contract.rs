//! Cargo-visible entry point for the release-evidence path contract.
//!
//! The rule is implemented once, in `scripts/check_evidence_paths.sh`, because
//! `scripts/nightly_ci.sh` runs the same gate. This test is how it reaches
//! `cargo test`.

use std::path::Path;
use std::process::Command;

/// Every filesystem path cited inside `release/evidence` must resolve on disk.
///
/// This exists because, before it, NOTHING validated that. Exactly two
/// consumers read the findings array workspace-wide,
/// `release_completion_audit/semantics/part5.rs` and
/// `vyre_weir_release_gate/semantic/release_hygiene.rs`, and both only compare
/// `findings.len()` against the summed `finding_summary` counts. Neither reads
/// `findings[].path`, and nothing stats it. `release_evidence/artifact_status.rs`
/// does stat files, but only the artifact files themselves from a hardcoded
/// expected list, never paths parsed out of their contents.
///
/// So the only semantic check on an artifact was INTERNAL SELF-CONSISTENCY, and
/// a stale artifact passes that trivially: deleting a source file changes
/// neither the array nor the summary, so the counts still agree. The artifact
/// stays perfectly self-validating while citing code that no longer exists.
/// Release evidence is the worst place for that failure, because its entire
/// purpose is to be trusted at release time.
///
/// Scope is every object array carrying a `path` field, not just `findings`.
/// When this gate was written the tree carried 185 stale citations across 16
/// artifacts and only 8 were in a findings array. The largest block, 124 of
/// them, was a stale path PREFIX: the dataflow component was renamed from
/// `dataflow-consumer` to `weir`, and two artifacts kept citing
/// `libs/dataflow/dataflow-consumer/src`. Those analyses do exist, at
/// `libs/dataflow/weir/src`.
///
/// Breaks if it regresses: release evidence keeps citing paths no reader can
/// open, and a rename silently invalidates the evidence for a whole component
/// while every self-consistency check on it stays green.
#[test]
fn release_evidence_cites_only_paths_that_exist() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest
        .parent()
        .expect("vyre-self-substrate should live directly under the workspace root");
    let script = workspace.join("scripts/check_evidence_paths.sh");

    let output = Command::new("bash")
        .arg(&script)
        .current_dir(workspace)
        .output()
        .expect("evidence path contract script should execute");

    assert!(
        output.status.success(),
        "release evidence path contract failed.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stderr).as_ref(),
        "",
        "evidence path contract must be quiet on success"
    );
}

/// The gate must be able to FAIL, proven against a known-bad input.
///
/// This exists because tonight produced four separate checks that ran, went
/// green, and carried no information about the property they appeared to
/// establish: a docs index that asked git what was on disk, a public-API gate
/// that was byte-stable across the deletion of an entire crate, a benchmark
/// harness that discarded the host-load probe it had already measured, and a
/// `cargo check --lib` all-clear that could not compile the failing cfg(test)
/// code. A gate that cannot fail is indistinguishable from no gate, and worse
/// than none, because it consumes the trust that would fund a real check.
///
/// So this pins the failing direction directly: a fixture citing one absent
/// path must exit non-zero and must name the artifact, the array, the index and
/// the path. Without it, a refactor that silently stopped discovering
/// citations, mis-parsed the JSON, or resolved every path to some fallback
/// would leave the contract above passing and meaningless.
#[test]
fn evidence_path_gate_fails_on_a_citation_that_does_not_resolve() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest
        .parent()
        .expect("vyre-self-substrate should live directly under the workspace root");
    let script = workspace.join("scripts/check_evidence_paths.sh");

    let fixture = std::env::temp_dir().join("vyre_evidence_path_gate_red_fixture");
    let evidence = fixture.join("evidence");
    let _ = std::fs::remove_dir_all(&fixture);
    std::fs::create_dir_all(&evidence).expect("fixture evidence directory should be creatable");
    std::fs::write(
        evidence.join("artifact.json"),
        r#"{"findings":[{"path":"definitely/not/on/disk/anywhere.rs"}]}"#,
    )
    .expect("fixture artifact should be writable");

    let output = Command::new("bash")
        .arg(&script)
        .current_dir(workspace)
        .env("EVIDENCE_DIR", &evidence)
        .output()
        .expect("evidence path contract script should execute");

    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let _ = std::fs::remove_dir_all(&fixture);

    assert!(
        !output.status.success(),
        "gate must exit non-zero when a cited path is absent; it exited {:?}.\nstderr:\n{stderr}",
        output.status.code()
    );
    assert!(
        stderr.contains("artifact.json"),
        "failure must name the artifact.\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("findings[0]"),
        "failure must name the array and index so the citation is locatable.\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("definitely/not/on/disk/anywhere.rs"),
        "failure must name the missing path.\nstderr:\n{stderr}"
    );
}

/// A cited path that exists but is gitignored must be reported, separately from
/// a path that is simply absent.
///
/// This exists because existence and publishability are different questions and
/// the gate answers them with different oracles: `stat(2)` for existence, and
/// `git check-ignore` for whether a path will ever reach another reader. Using
/// git for the first question is the exact defect that broke the docs index
/// gate, where `git ls-files` called an untracked-but-present file missing and
/// demanded a deleted-but-still-tracked file stay indexed.
///
/// This branch never fires on the current tree, where zero cited paths are
/// gitignored, so without this test it would be dead code that has never been
/// observed to work and would be trusted anyway. The fixture also pins the
/// index-aware behaviour: a tracked file matching an ignore pattern is already
/// in public history and must NOT be reported.
#[test]
fn evidence_path_gate_reports_a_cited_path_that_is_gitignored() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest
        .parent()
        .expect("vyre-self-substrate should live directly under the workspace root");
    let script = workspace.join("scripts/check_evidence_paths.sh");

    let fixture = std::env::temp_dir().join("vyre_evidence_path_gate_ignored_fixture");
    let repo = fixture.join("repo");
    let evidence = fixture.join("evidence");
    let _ = std::fs::remove_dir_all(&fixture);
    std::fs::create_dir_all(&repo).expect("fixture repo should be creatable");
    std::fs::create_dir_all(&evidence).expect("fixture evidence directory should be creatable");

    // `generated.rs` is ignored; `tracked.rs` is committed and must stay clean.
    std::fs::write(repo.join(".gitignore"), "generated.rs\n").expect("gitignore writable");
    std::fs::write(repo.join("generated.rs"), "fn generated() {}\n").expect("generated writable");
    std::fs::write(repo.join("tracked.rs"), "fn tracked() {}\n").expect("tracked writable");
    for args in [
        vec!["init", "-q", "."],
        vec!["add", ".gitignore", "tracked.rs"],
        vec![
            "-c",
            "user.email=gate@vyre.test",
            "-c",
            "user.name=gate",
            "commit",
            "-qm",
            "fixture",
        ],
    ] {
        let status = Command::new("git")
            .args(&args)
            .current_dir(&repo)
            .status()
            .expect("git should be available for the ignore fixture");
        assert!(status.success(), "fixture git step failed: {args:?}");
    }

    std::fs::write(
        evidence.join("artifact.json"),
        format!(
            r#"{{"files":[{{"path":"{}"}},{{"path":"{}"}}]}}"#,
            repo.join("tracked.rs").display(),
            repo.join("generated.rs").display()
        ),
    )
    .expect("fixture artifact should be writable");

    let output = Command::new("bash")
        .arg(&script)
        .current_dir(workspace)
        .env("EVIDENCE_DIR", &evidence)
        .output()
        .expect("evidence path contract script should execute");

    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let _ = std::fs::remove_dir_all(&fixture);

    assert!(
        !output.status.success(),
        "gate must exit non-zero when a cited path is gitignored.\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("gitignored"),
        "failure must say the path is gitignored, not that it is missing.\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("generated.rs"),
        "failure must name the gitignored path.\nstderr:\n{stderr}"
    );
    assert!(
        !stderr.contains("tracked.rs"),
        "a tracked file must not be reported; check-ignore must stay index-aware.\nstderr:\n{stderr}"
    );
}
