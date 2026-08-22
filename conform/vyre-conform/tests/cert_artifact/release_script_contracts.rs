#[test]
fn release_scripts_make_sharded_conformance_certificate_load_bearing() {
    let repo = structure_gate::workspace_root();
    let prove = std::fs::read_to_string(repo.join("scripts/prove-release-shards.sh"))
        .expect("Fix: sharded release proof helper must be readable");
    assert!(
        prove.contains("vyre_select_cargo_runner"),
        "Fix: sharded release proof must use the shared OOM-safe cargo runner selector."
    );
    assert!(
        prove.contains("metadata --no-deps --format-version 1")
            && prove.contains("target_directory"),
        "Fix: release proof must discover Cargo's configured target directory instead of assuming ./target."
    );
    assert!(
        prove.contains("VYRE_RELEASE_SHARD_WORKERS"),
        "Fix: release proof shards must run through a bounded parallel worker pool."
    );
    assert!(
        prove.contains("\"$RUNNER_BIN\" \"${prove_args[@]}\"")
            && prove.contains("\"$RUNNER_BIN\" \"${merge_args[@]}\""),
        "Fix: release proof must build vyre-conform once, then use the binary for prove and merge."
    );

    // The signed-conformance wrapper is gone. It exported the four defaults
    // `prove-release-shards.sh` already carries at lines 5 to 8 and then checked
    // that the merged certificate was non-empty, which the merge step and the
    // certificate suite in this crate both assert.
    let final_launch = std::fs::read_to_string(repo.join("scripts/final-launch.sh"))
        .expect("Fix: final launch script must be readable");
    assert!(
        final_launch.contains("scripts/prove-release-shards.sh")
            && final_launch.contains("release/evidence/conformance/release-all-backends-certificate.json")
            && final_launch.contains("prove sharded all-backend conformance certificate"),
        "Fix: final launch must make the merged sharded certificate load-bearing release evidence before publish."
    );
}

/// The shard pool counts the shards that failed, and only those.
///
/// WHY: the pool reaped every worker with a bare `wait`, which reports the
/// status of the last job and leaves the counter claiming a worker is still
/// running. A clean run of four shards then exited 1 saying one worker failed,
/// and a genuinely failed shard was counted as a pass. Reading the script for
/// the string `wait -n` could not see either, so the pool is run here with a
/// stub conform binary instead.
#[test]
fn the_shard_pool_counts_exactly_the_shards_that_failed() {
    let repo = structure_gate::workspace_root();
    let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
    let cargo = stub(
        dir.path(),
        "cargo",
        "case \"$1\" in\n  metadata) printf '{\"target_directory\":\"%s\"}\\n' \"$PWD/target\";;\nesac\nexit 0\n",
    );
    let conform = stub(
        dir.path(),
        "vyre-conform",
        "mode=\"$1\"; shift\nout=\"\"; shard=\"\"\nwhile [ $# -gt 0 ]; do\n  case \"$1\" in\n    --out) out=\"$2\"; shift 2;;\n    --shard) shard=\"$2\"; shift 2;;\n    *) shift;;\n  esac\ndone\nindex=\"${shard%%/*}\"\nif [ \"$mode\" = prove ]; then\n  case \" $FAILING_SHARDS \" in *\" $index \"*) exit 3;; esac\nfi\nprintf '{}' > \"$out\"\n",
    );

    let clean = run_pool(&repo, dir.path(), &cargo, &conform, "");
    assert!(
        clean.status.success(),
        "a run whose shards all pass must exit zero: {}",
        String::from_utf8_lossy(&clean.stderr)
    );

    let failed = run_pool(&repo, dir.path(), &cargo, &conform, "1 2");
    let stderr = String::from_utf8_lossy(&failed.stderr).into_owned();
    assert_eq!(failed.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.contains("2 release conformance shard worker(s) failed"),
        "the pool must count both failed shards: {stderr}"
    );
}

/// Write one executable stub and return its path.
fn stub(dir: &std::path::Path, name: &str, body: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, format!("#!/usr/bin/env bash\nset -u\n{body}"))
        .expect("Fix: stub must be writable");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
            .expect("Fix: stub must be executable");
    }
    path
}

/// Run the shard script over four shards with two workers.
fn run_pool(
    repo: &std::path::Path,
    out: &std::path::Path,
    cargo: &std::path::Path,
    conform: &std::path::Path,
    failing: &str,
) -> std::process::Output {
    std::process::Command::new("bash")
        .arg(repo.join("scripts/prove-release-shards.sh"))
        .current_dir(repo)
        .env("VYRE_CARGO_RUNNER", cargo)
        .env("VYRE_CONFORM_RUNNER_BIN", conform)
        .env("VYRE_RELEASE_CERT_DIR", out.join("certs"))
        .env("VYRE_RELEASE_SHARDS", "4")
        .env("VYRE_RELEASE_SHARD_WORKERS", "2")
        .env("FAILING_SHARDS", failing)
        .output()
        .expect("Fix: bash must be runnable to exercise the shard pool")
}
