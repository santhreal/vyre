//! Cargo-visible entry point for `scripts/check_evidence_paths.sh`.
//!
//! Every cited release-evidence path must resolve to an owned artifact.

use std::process::Command;

/// Every filesystem path cited inside `release/evidence` must resolve on disk.
///
/// The check covers every string leaf whose value is shaped like a filesystem
/// path, under any key and at any depth. This prevents source renames and
/// artifact migrations from leaving internally consistent evidence that points
/// to files no reader can open.
#[test]
fn release_evidence_cites_only_paths_that_exist() {
    let workspace = vyre_test_support::monorepo::vyre_workspace_root();
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
    let workspace = vyre_test_support::monorepo::vyre_workspace_root();
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
    let workspace = vyre_test_support::monorepo::vyre_workspace_root();
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

/// A citation the gate would previously never read must still be reported.
///
/// Discovery has been narrowed twice and both narrowings hid live defects. It
/// first walked one shape only, a top-level key holding an array of objects with
/// a `path` field, which hid 81 of 634 citations; the one dead citation among
/// them sat on an artifact's own root object, an unexpanded `${SANTH_ROOT}`
/// template naming a README in a repository this one does not contain. It then
/// read the key `path` and nothing else, which hid 2775 further citations under
/// `manifest`, `artifact`, `evidence_link`, `source_artifact`, `workflow` and
/// bare array members; nine of those were dead, all naming one crate's
/// `Cargo.toml` after that crate had been folded into another.
///
/// So the fixture places one absent path at each shape a narrower filter missed:
/// the root object, an object nested under another object, an object inside a
/// nested array, a key that is not called `path`, and a bare string inside an
/// array. Each must be reported at its full route, including the key that
/// carries it, so that two citations on one object stay distinguishable and a
/// schema that moves a citation deeper or renames its key cannot silently take
/// it out of the gate's reach.
#[test]
fn evidence_path_gate_reads_citations_outside_a_top_level_array() {
    let workspace = vyre_test_support::monorepo::vyre_workspace_root();
    let script = workspace.join("scripts/check_evidence_paths.sh");

    let fixture = std::env::temp_dir().join("vyre_evidence_path_gate_nested_fixture");
    let evidence = fixture.join("evidence");
    let _ = std::fs::remove_dir_all(&fixture);
    std::fs::create_dir_all(&evidence).expect("fixture evidence directory should be creatable");
    std::fs::write(
        evidence.join("artifact.json"),
        concat!(
            r#"{"path":"absent/on/the/root/object.rs","#,
            r#""subject":{"path":"absent/under/an/object.rs"},"#,
            r#""groups":[{"rows":[{"path":"absent/in/a/nested/array.rs"}]}],"#,
            r#""crate":{"manifest":"absent/crate/Cargo.toml"},"#,
            r#""sources":["absent/array/member.rs"]}"#
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
        "gate must exit non-zero for citations outside a top-level array.\nstderr:\n{stderr}"
    );
    for (route, path) in [
        ("path", "absent/on/the/root/object.rs"),
        ("subject.path", "absent/under/an/object.rs"),
        ("groups[0].rows[0].path", "absent/in/a/nested/array.rs"),
        ("crate.manifest", "absent/crate/Cargo.toml"),
        ("sources[0]", "absent/array/member.rs"),
    ] {
        assert!(
            stderr.contains(&format!("{route} cites a path that does not exist: {path}")),
            "failure must locate the citation at `{route}`.\nstderr:\n{stderr}"
        );
    }
    assert!(
        stderr.contains("5 citation(s)"),
        "every unreadable placement must be counted, not just the first.\nstderr:\n{stderr}"
    );
}

/// A string that is not a path must not be read as a citation.
///
/// The widened discovery rule is a shape rule, so it needs a floor on the other
/// side: a gate with false positives gets muted, and muting this one restores
/// the state it was built to end. Version strings, operation identifiers, schema
/// identifiers, fingerprints and recorded shell commands all live beside real
/// citations in these artifacts, and none of them names a file.
///
/// The extension vocabulary is what separates them, and it is derived from the
/// tree at run time rather than listed here, so this fixture pins the property
/// instead of the list: `0` and `v1` are not extensions the workspace uses, a
/// command contains whitespace, and an operation id has no `/`. A future
/// loosening that starts matching any dotted token turns this red.
#[test]
fn evidence_path_gate_does_not_read_non_path_strings_as_citations() {
    let workspace = vyre_test_support::monorepo::vyre_workspace_root();
    let script = workspace.join("scripts/check_evidence_paths.sh");

    let fixture = std::env::temp_dir().join("vyre_evidence_path_gate_shape_fixture");
    let evidence = fixture.join("evidence");
    let _ = std::fs::remove_dir_all(&fixture);
    std::fs::create_dir_all(&evidence).expect("fixture evidence directory should be creatable");
    std::fs::write(
        evidence.join("artifact.json"),
        concat!(
            r#"{"version":"1.2.0","#,
            r#""schema_id":"vyre-conform-input-envelope-v1","#,
            r#""op":"vyre-primitives::hardware::subgroup_shuffle","#,
            r#""fingerprint":"source-tree-v1:f42685f0","#,
            r#""command":"git grep -nE \"trait CpuOp\" -- absent/crate/src","#,
            r#""ratio":"0.0/1.0"}"#
        ),
    )
    .expect("fixture artifact should be writable");

    let output = Command::new("bash")
        .arg(&script)
        .current_dir(workspace)
        .env("EVIDENCE_DIR", &evidence)
        .output()
        .expect("evidence path contract script should execute");

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let _ = std::fs::remove_dir_all(&fixture);

    assert!(
        output.status.success(),
        "no string in the fixture names a file, so the gate must pass.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("all 0 cited path(s)"),
        "the gate must read zero citations here; anything else is a false positive.\nstdout:\n{stdout}"
    );
}
