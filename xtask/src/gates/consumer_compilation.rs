//! Every out-of-workspace consumer package compiles cleanly.
//!
//! Consumer packages live outside the compiler workspace (under `consumers/`)
//! and declare their own `[workspace]` table so cargo stops looking upward.
//! Because they are not workspace members, `cargo check --workspace` never
//! checks them.
//!
//! This gate derives the consumer set from the tree at run time and compiles
//! each package with `--all-targets`.

use std::path::{Path, PathBuf};

use crate::gate::{Finding, GateBehavior, GateCtx, GateError, Report};

/// Compile every out-of-workspace consumer package.
pub struct ConsumerCompilation;

impl GateBehavior for ConsumerCompilation {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let manifests = consumer_manifests(&ctx.root)?;

        report.cover_complete("consumer packages", manifests.len());
        report.note(format!("{} consumer package(s) found", manifests.len()));

        for manifest in &manifests {
            let findings = check_consumer(&ctx.root, manifest)?;
            for finding in findings {
                report.find(finding);
            }
        }

        Ok(report)
    }
}

/// Enumerate all out-of-workspace consumer package manifests at runtime.
pub fn consumer_manifests(root: &Path) -> Result<Vec<PathBuf>, GateError> {
    let consumers_dir = root.join("consumers");
    if !consumers_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut manifests = Vec::new();
    let entries = std::fs::read_dir(&consumers_dir).map_err(|error| {
        GateError::new(
            format!(
                "cannot read `{}` directory: {error}",
                consumers_dir.display()
            ),
            "ensure the consumers directory is readable",
        )
    })?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            GateError::new(
                format!(
                    "cannot read directory entry in `{}`: {error}",
                    consumers_dir.display()
                ),
                "ensure the consumers directory is readable",
            )
        })?;
        let path = entry.path();
        if path.is_dir() {
            let manifest = path.join("Cargo.toml");
            if manifest.is_file() {
                let relative = manifest
                    .strip_prefix(root)
                    .map_or_else(|_| manifest.clone(), Path::to_path_buf);
                manifests.push(relative);
            }
        }
    }
    manifests.sort();
    Ok(manifests)
}

/// Run cargo check on one consumer manifest and collect findings.
fn check_consumer(root: &Path, manifest_path: &Path) -> Result<Vec<Finding>, GateError> {
    let full_manifest = root.join(manifest_path);
    let Some(manifest_argument) = full_manifest.to_str() else {
        return Err(GateError::new(
            format!(
                "consumer manifest `{}` is not valid UTF-8, so it cannot be named on a command line",
                full_manifest.display()
            ),
            "rename the consumer directory to a UTF-8 path",
        ));
    };
    let run = crate::cargo_runner::diagnostics(
        root,
        &[
            "check",
            "--manifest-path",
            manifest_argument,
            "--all-targets",
        ],
        false,
    )?;

    if let Some(missing) = run.unmeasured {
        return Ok(vec![Finding::new(
            format!(
                "`cargo check --manifest-path {}` measured nothing: the build named `{missing}`, which the build directory does not carry",
                manifest_path.display()
            ),
            "run the gate again against an intact build directory; a compile whose own inputs were deleted under it reports the state of the disk",
        )]);
    }

    if run.failed_silently() {
        return Ok(vec![Finding::in_file(
            manifest_path,
            format!(
                "consumer `{}` exited {} and emitted no compiler diagnostic: {}",
                manifest_path.display(),
                run.code(),
                run.stderr.trim()
            ),
            format!(
                "repair compilation of consumer `{}`",
                manifest_path.display()
            ),
        )]);
    }

    // One consumer that does not build is one finding. Every later diagnostic
    // is a consequence of the first and names the same repair.
    let Some(diagnostic) = run.found.first() else {
        return Ok(Vec::new());
    };
    let message = format!(
        "consumer `{}` compile error: {}",
        manifest_path.display(),
        diagnostic.message
    );
    let fix = format!("fix compiler error in `{}`", manifest_path.display());
    Ok(vec![diagnostic
        .place(root, &message, &fix)
        .unwrap_or_else(|| Finding::in_file(manifest_path, message.clone(), fix.clone()))])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: nothing else compiles these packages. `cargo check --workspace`
    /// skips them because they declare their own workspace table, so a consumer
    /// this enumeration misses is never built by anything, and the gate reports
    /// a clean tree while it is broken. The roster is derived from the
    /// directory rather than declared, so what the derivation admits and
    /// rejects is the contract.
    #[test]
    fn every_consumer_directory_holding_a_manifest_is_enumerated() {
        let temp = tempfile::tempdir().expect("tempdir");
        let consumers = temp.path().join("consumers");
        for package in ["pkg-b", "pkg-a"] {
            std::fs::create_dir_all(consumers.join(package)).unwrap();
            std::fs::write(
                consumers.join(package).join("Cargo.toml"),
                format!("[package]\nname = \"{package}\"\n"),
            )
            .unwrap();
        }
        std::fs::create_dir_all(consumers.join("not-a-package")).unwrap();
        std::fs::write(consumers.join("README.md"), "consumers live here\n").unwrap();

        let manifests = consumer_manifests(temp.path()).expect("enumerate");
        assert_eq!(
            manifests,
            vec![
                PathBuf::from("consumers/pkg-a/Cargo.toml"),
                PathBuf::from("consumers/pkg-b/Cargo.toml"),
            ],
            "every manifest is found, stated relative to the checkout, in a stable order"
        );
    }

    /// WHY: a checkout with no consumers directory is not a checkout whose
    /// consumers all build. An error here would fail every gate run in a tree
    /// that legitimately has none.
    #[test]
    fn a_checkout_with_no_consumers_directory_enumerates_nothing() {
        let temp = tempfile::tempdir().expect("tempdir");
        assert!(consumer_manifests(temp.path()).expect("enumerate").is_empty());
    }
}
