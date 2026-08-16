//! The toolchain a workflow installs is the only thing `--print-toolchain`
//! writes.
//!
//! WHY: the gates workflow installs the MSRV toolchain from the first stdout
//! line of `xtask feature-msrv --print-toolchain` and hands that line to
//! `rustup toolchain install`. The coupling is a contract with no declaration:
//! a banner printed by the dispatcher, a note rendered before the early return,
//! or a second line of any kind is fed to rustup, which then installs a
//! toolchain named after a sentence, or installs nothing and leaves the sweep
//! running on the default. This drives the real binary so the whole path is
//! covered, dispatcher included, and reads the expected version out of the root
//! manifest rather than restating it.

#![forbid(unsafe_code)]

use std::process::Command;

use crate::workspace_sources::workspace_root;

#[test]
fn print_toolchain_writes_the_advertised_version_and_nothing_else() {
    let root = workspace_root();
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["feature-msrv", "--print-toolchain"])
        .current_dir(&root)
        .output()
        .expect("Fix: the runner must launch");
    assert!(
        output.status.success(),
        "Fix: the gate must exit zero when it is only asked for the version: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().filter(|line| !line.is_empty()).collect();
    assert_eq!(
        lines.len(),
        1,
        "Fix: the workflow installs the first stdout line, so this mode writes one line: {stdout:?}"
    );

    let manifest = std::fs::read_to_string(root.join("Cargo.toml"))
        .expect("Fix: the root manifest must be readable");
    let table: toml::Table = toml::from_str(&manifest).expect("Fix: the root manifest must parse");
    let advertised = table
        .get("workspace")
        .and_then(|workspace| workspace.get("package"))
        .and_then(|package| package.get("rust-version"))
        .and_then(toml::Value::as_str)
        .expect("Fix: [workspace.package] must declare rust-version");
    assert_eq!(
        lines[0], advertised,
        "Fix: the printed toolchain is the advertised minimum"
    );
    assert!(
        lines[0]
            .split('.')
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit())),
        "Fix: rustup installs a version, and `{}` is not one",
        lines[0]
    );
}

/// WHY: `cargo-fuzz` installs on stable, but the libFuzzer runner passes `-Z`
/// sanitizer flags and therefore requires nightly. The smoke workflow installed
/// and selected stable, so every fuzz target failed before reading one byte.
/// The job set is derived from the workflow so a new fuzz job is covered.
#[test]
fn every_cargo_fuzz_job_selects_nightly() {
    let root = workspace_root();
    let workflow = std::fs::read_to_string(root.join(".github/workflows/fuzz.yml"))
        .expect("Fix: the fuzz workflow must be readable");
    let mut job = "<none>";
    let mut toolchain = None;
    let mut installs = 0usize;

    for line in workflow.lines() {
        if line.starts_with("  ")
            && !line.starts_with("    ")
            && line.trim_end().ends_with(':')
        {
            job = line.trim().trim_end_matches(':');
            toolchain = None;
        }

        let trimmed = line.trim();
        if let Some(selected) = trimmed.strip_prefix("- uses: dtolnay/rust-toolchain@") {
            toolchain = Some(selected);
        }
        if !trimmed.contains("install --locked cargo-fuzz") {
            continue;
        }

        installs += 1;
        assert_eq!(
            toolchain,
            Some("nightly"),
            "Fix: fuzz job `{job}` must select nightly before installing cargo-fuzz"
        );
        assert_eq!(
            trimmed,
            "run: cargo +nightly install --locked cargo-fuzz",
            "Fix: fuzz job `{job}` must install cargo-fuzz with the selected nightly toolchain"
        );
    }

    assert!(
        installs > 0,
        "Fix: the fuzz workflow contains no cargo-fuzz install, so this contract guards nothing"
    );
}
