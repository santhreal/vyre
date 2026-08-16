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
        prove.contains("VYRE_RELEASE_SHARD_WORKERS") && prove.contains("wait -n"),
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
