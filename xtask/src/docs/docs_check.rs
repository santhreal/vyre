//! Thin release-tool adapter for the canonical documentation manifest gate.

use std::process::Command;

pub(crate) fn run(args: &[String]) {
    if args.len() > 2 {
        eprintln!("Fix: docs-check accepts no arguments.");
        std::process::exit(2);
    }
    let workspace_root = crate::checkout::checkout_root();
    let status = Command::new("python3")
        .arg("scripts/docs_manifest.py")
        .arg("--check")
        .current_dir(workspace_root)
        .status()
        .unwrap_or_else(|error| {
            eprintln!("docs-check: failed to run documentation manifest gate: {error}");
            std::process::exit(1);
        });
    std::process::exit(status.code().unwrap_or(1));
}
