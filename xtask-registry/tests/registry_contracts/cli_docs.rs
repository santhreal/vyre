//! Executable command-line documentation contract tests for the registry binaries.

use std::process::Command;

/// Prevents `vyre_new_op --help` from scaffolding an operation while answering a
/// reader who only asked what the arguments are.
#[test]
fn the_new_op_scaffolder_help_route_exits_zero() {
    let output = Command::new(env!("CARGO_BIN_EXE_vyre_new_op"))
        .arg("--help")
        .output()
        .expect("Fix: documented CLI executable must launch");
    assert!(
        output.status.success(),
        "vyre_new_op --help returned {:?}: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("Usage:"));
}

/// Prevents a delegated crate from answering a subcommand it does not implement.
/// `xtask` decides who owns a name; a binary that ran an unowned name anyway
/// would make that decision unobservable.
#[test]
fn an_unowned_subcommand_is_refused_with_a_fix_message() {
    let output = Command::new(env!("CARGO_BIN_EXE_xtask-registry"))
        .arg("dep-drift")
        .output()
        .expect("Fix: the registry binary must launch");
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Fix:"), "stderr was: {stderr}");
    assert!(stderr.contains("dep-drift"), "stderr was: {stderr}");
}
