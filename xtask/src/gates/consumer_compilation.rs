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
use std::process::Command;

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

/// One compiler diagnostic parsed from `--message-format=json`.
#[derive(Debug)]
struct Diagnostic {
    file: Option<String>,
    line: Option<u32>,
    message: String,
}

/// Run cargo check on one consumer manifest and collect findings.
fn check_consumer(root: &Path, manifest_path: &Path) -> Result<Vec<Finding>, GateError> {
    let cargo = crate::cargo_runner::binary(root);
    let full_manifest = root.join(manifest_path);

    let output = Command::new(&cargo)
        .arg("check")
        .arg("--manifest-path")
        .arg(&full_manifest)
        .arg("--all-targets")
        .arg("--message-format=json")
        .current_dir(root)
        .output()
        .map_err(|error| {
            GateError::new(
                format!(
                    "cannot run `cargo check --manifest-path {} --all-targets`: {error}",
                    manifest_path.display()
                ),
                "restore the cargo_full wrapper at the workspace root",
            )
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let diagnostics = parse_compiler_diagnostics(&stdout);

    let stderr = String::from_utf8_lossy(&output.stderr);
    if let Some(missing) = crate::cargo_runner::unmeasured(&stderr) {
        return Ok(vec![Finding::new(
            format!(
                "`cargo check --manifest-path {}` measured nothing: the build named `{missing}`, which the build directory does not carry",
                manifest_path.display()
            ),
            "run the gate again against an intact build directory; a compile whose own inputs were deleted under it reports the state of the disk",
        )]);
    }

    if !output.status.success() && diagnostics.is_empty() {
        return Ok(vec![Finding::in_file(
            manifest_path,
            format!(
                "consumer `{}` exited {} and emitted no compiler diagnostic: {}",
                manifest_path.display(),
                output.status.code().unwrap_or(-1),
                stderr.trim()
            ),
            format!(
                "repair compilation of consumer `{}`",
                manifest_path.display()
            ),
        )]);
    }

    let mut findings = Vec::new();
    if let Some(diag) = diagnostics.into_iter().next() {
        let msg = format!(
            "consumer `{}` compile error: {}",
            manifest_path.display(),
            diag.message
        );
        let fix = format!("fix compiler error in `{}`", manifest_path.display());
        let finding = match (diag.file, diag.line) {
            (Some(file), Some(line)) => {
                let file_path = PathBuf::from(file);
                let relative = file_path.strip_prefix(root).unwrap_or(&file_path);
                Finding::at(relative, line, msg, fix)
            }
            (Some(file), None) => {
                let file_path = PathBuf::from(file);
                let relative = file_path.strip_prefix(root).unwrap_or(&file_path);
                Finding::in_file(relative, msg, fix)
            }
            (None, _) => Finding::in_file(manifest_path, msg, fix),
        };
        findings.push(finding);
    }

    Ok(findings)
}

/// Parse compiler error diagnostics from `--message-format=json` lines.
fn parse_compiler_diagnostics(stdout: &str) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for line in stdout.lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if value.get("reason").and_then(serde_json::Value::as_str) != Some("compiler-message") {
            continue;
        }
        let Some(message) = value.get("message") else {
            continue;
        };
        if message.get("level").and_then(serde_json::Value::as_str) != Some("error") {
            continue;
        }
        let text = message
            .get("message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("compiler reported an error")
            .to_string();
        let primary = message
            .get("spans")
            .and_then(serde_json::Value::as_array)
            .and_then(|spans| {
                spans.iter().find(|span| {
                    span.get("is_primary")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false)
                })
            });
        diagnostics.push(Diagnostic {
            file: primary
                .and_then(|span| span.get("file_name"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            line: primary
                .and_then(|span| span.get("line_start"))
                .and_then(serde_json::Value::as_u64)
                .and_then(|l| u32::try_from(l).ok()),
            message: text,
        });
    }
    diagnostics
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: proves consumer enumeration finds manifests under the consumers directory.
    #[test]
    fn enumerates_consumers_from_directory() {
        let temp = tempfile::tempdir().expect("tempdir");
        let consumers = temp.path().join("consumers");
        std::fs::create_dir_all(consumers.join("pkg-a")).unwrap();
        std::fs::create_dir_all(consumers.join("pkg-b")).unwrap();
        std::fs::write(
            consumers.join("pkg-a/Cargo.toml"),
            "[package]\nname = \"pkg-a\"\n",
        )
        .unwrap();
        std::fs::write(
            consumers.join("pkg-b/Cargo.toml"),
            "[package]\nname = \"pkg-b\"\n",
        )
        .unwrap();

        let manifests = consumer_manifests(temp.path()).expect("enumerate");
        assert_eq!(manifests.len(), 2);
        assert_eq!(manifests[0], PathBuf::from("consumers/pkg-a/Cargo.toml"));
        assert_eq!(manifests[1], PathBuf::from("consumers/pkg-b/Cargo.toml"));
    }

    /// WHY: proves compiler error diagnostics are extracted from JSON output.
    #[test]
    fn finds_compile_failure_in_consumer() {
        let json = r#"{"reason":"compiler-message","package_id":"foo","message":{"rendered":"...","level":"error","message":"cannot find value `x` in this scope","spans":[{"file_name":"src/lib.rs","line_start":42,"is_primary":true}]}}"#;
        let diags = parse_compiler_diagnostics(json);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].message, "cannot find value `x` in this scope");
        assert_eq!(diags[0].file.as_deref(), Some("src/lib.rs"));
        assert_eq!(diags[0].line, Some(42));
    }
}
