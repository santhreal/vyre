//! Integration test that exercises the `wire_harness_smoke` example as
//! a real subprocess - the same way an agent harness would invoke it.
//!
//! Locks the user-visible CLI contract (stdin/stdout shape, exit code,
//! determinism) so the harness can build against a frozen interface.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::LazyLock;

/// The example binary, built by this test in every lane.
///
/// Cargo only builds examples as plain binaries under its default target
/// selection. `--tests`, `--examples` and `--all-targets` all compile an
/// example as a *test harness* instead, named `wire_harness_smoke-<hash>`,
/// which runs zero tests and never reaches the CLI `main` this test exercises.
/// The coverage lane passes `--tests`, so relying on the default selection made
/// this test fail on a flag rather than on a defect.
///
/// Building it here inherits the invoking cargo's environment, including the
/// instrumentation a coverage run exports, so the binary lands beside this
/// test in every lane. The build runs even when an example binary already
/// exists: a `--tests` run never refreshes that binary, so a leftover from an
/// earlier commit would otherwise stand in for the code under test, and cargo's
/// own freshness check makes the build a no-op when it is current.
/// Built once per test process; five tests share one binary.
static EXAMPLE: LazyLock<PathBuf> = LazyLock::new(build_example);

fn build_example() -> PathBuf {
    // The target directory is resolved from the running test binary, not
    // inherited: cargo-llvm-cov passes --target-dir on its own command line
    // rather than exporting it, so a child cargo that reads only the
    // environment writes the example into the default directory and the
    // coverage run never sees it. RUSTFLAGS is exported, so the example is
    // built with the same instrumentation as everything beside it.
    //
    // The test binary's depth under the target directory depends on the cargo
    // release (`<profile>/deps/` before the build-dir layout,
    // `<profile>/build/<package>/<hash>/out/` after it), so it is located by
    // the target directory's cache marker, and the profile is the first path
    // component below it. The example is always written to
    // `<target>/<profile>/examples/`.
    let target = vyre_test_support::monorepo::cargo_target_directory();
    let exe = std::env::current_exe().expect("current_exe");
    let profile_dir = exe
        .strip_prefix(&target)
        .ok()
        .and_then(|relative| relative.components().next())
        .expect("a test binary lives under <target>/<profile>/")
        .as_os_str()
        .to_owned();
    // The suffix is part of the name cargo writes: on Windows the example is
    // `wire_harness_smoke.exe`, and a bare stem never exists there.
    let path = target.join(&profile_dir).join("examples").join(format!(
        "wire_harness_smoke{}",
        std::env::consts::EXE_SUFFIX
    ));
    // Cargo writes the `dev` profile into `debug/`; every other profile
    // writes into a directory of its own name.
    let profile = if profile_dir == "debug" {
        "dev".into()
    } else {
        profile_dir
    };
    let built = Command::new(env!("CARGO"))
        .args([
            "build",
            "--example",
            "wire_harness_smoke",
            "-p",
            "vyre-primitives",
            "--profile",
        ])
        .arg(&profile)
        .arg("--target-dir")
        .arg(&target)
        .status()
        .expect("cargo must be invocable to build the example under test");
    assert!(built.success(), "building wire_harness_smoke failed");
    assert!(
        path.exists(),
        "wire_harness_smoke was built but is not at {}",
        path.display()
    );
    path
}

fn run_harness(stdin_input: &str) -> (String, String, Option<i32>) {
    let path = &*EXAMPLE;
    let mut child = Command::new(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn harness");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(stdin_input.as_bytes())
        .unwrap();
    drop(child.stdin.take());
    let out = child.wait_with_output().expect("wait");
    (
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
        out.status.code(),
    )
}

#[test]
fn pack_u32_round_trip_via_subprocess() {
    let (stdout, stderr, code) = run_harness("pack-u32 1,2,3\n");
    assert!(stderr.is_empty(), "unexpected stderr: {stderr}");
    assert_eq!(code, Some(0));
    assert_eq!(stdout.trim(), "010000000200000003000000");
}

#[test]
fn unpack_u32_decodes_to_original_values() {
    let (stdout, stderr, code) = run_harness("unpack-u32 010000000200000003000000 3\n");
    assert!(stderr.is_empty(), "unexpected stderr: {stderr}");
    assert_eq!(code, Some(0));
    assert_eq!(stdout.trim(), "1,2,3");
}

#[test]
fn pack_f32_uses_le_byte_order() {
    let (stdout, stderr, code) = run_harness("pack-f32 1.0,-0.0\n");
    assert!(stderr.is_empty(), "unexpected stderr: {stderr}");
    assert_eq!(code, Some(0));
    // 1.0_f32 = 0x3f800000 (LE: 00 00 80 3f); -0.0_f32 = 0x80000000 (LE: 00 00 00 80).
    assert_eq!(stdout.trim(), "0000803f00000080");
}

#[test]
fn unknown_command_writes_err_and_nonzero_exit() {
    let (stdout, stderr, code) = run_harness("rotate 7\n");
    assert_eq!(code, Some(1));
    assert!(stderr.contains("unknown command"), "stderr: {stderr}");
    assert_eq!(stdout.trim(), "ERR");
}

#[test]
fn deterministic_across_repeated_runs() {
    let input = "pack-u32 7,11,13\npack-f32 3.14,2.718\npack-u32 0\n";
    let (a_out, _, a_code) = run_harness(input);
    let (b_out, _, b_code) = run_harness(input);
    assert_eq!(a_code, Some(0));
    assert_eq!(b_code, Some(0));
    assert_eq!(a_out, b_out, "harness output must be deterministic");
}
