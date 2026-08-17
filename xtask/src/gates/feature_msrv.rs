//! Every feature selection this workspace judges compiles on the version it
//! advertises.
//!
//! `[workspace.package].rust-version` is a published claim: a consumer whose
//! toolchain is that version reads it as a promise that the crate builds. The
//! CI toolchain matrix is stable and nightly, so nothing else compiles this
//! workspace on the version it names, and the claim held only because nobody
//! tried it.
//!
//! The axis is [`crate::gates::feature_isolation::derive_pairs`], the same one the isolation
//! gate judges, so a new member or feature joins this sweep on the commit that
//! declares it. Two rosters over the same manifests would disagree the first
//! time one of them was edited.
//!
//! Two modes, because the costs are three orders of magnitude apart:
//!
//!   - Default: the declaration is readable, names an installable version, and
//!     yields a non-empty axis. No cargo, so the sweep runs it on every change.
//!   - `--sweep`: compiles every selection through the workspace cargo with a
//!     leading `+<version>`, and reports each selection that fails. The
//!     toolchain is named rather than defaulted, so a host whose default
//!     compiler is newer cannot report a pass for a version this workspace
//!     never built on.

use std::path::Path;
use std::process::Command;

use crate::cargo_runner;
use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::feature_isolation::{derive_pairs, Pair};
use crate::gates::scan::Tree;

/// The manifest that owns the advertised version.
const ROOT_MANIFEST: &str = "Cargo.toml";

/// Compiles every declared feature selection on the advertised MSRV.
pub struct FeatureMsrv;

impl crate::gate::GateBehavior for FeatureMsrv {
    fn usage(&self) -> &'static [&'static str] {
        &[
            "--sweep compiles each declared selection on the advertised MSRV",
            "--print-toolchain writes the validated MSRV version for workflow installation",
        ]
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let manifest = tree.read_toml(ROOT_MANIFEST)?;
        let mut report = Report::clean();
        report.cover_complete("workspace manifests", tree.members()?.len());

        let declared = declared_version(&manifest);
        let Some(version) = declared else {
            report.find(Finding::in_file(
                ROOT_MANIFEST,
                "[workspace.package] declares no rust-version, so no toolchain can be pinned to the advertised minimum",
                "declare rust-version, which is the version every published crate promises a consumer",
            ));
            return Ok(report);
        };
        if !names_a_toolchain(&version) {
            report.find(Finding::in_file(
                ROOT_MANIFEST,
                format!(
                    "rust-version `{version}` is not a version rustup can install, so the sweep would compile on whatever toolchain is default"
                ),
                "write the minimum as `major.minor` or `major.minor.patch`",
            ));
            return Ok(report);
        }
        if ctx.has("--print-toolchain") {
            println!("{version}");
            return Ok(report);
        }

        let axis = derive_pairs(&ctx.root).map_err(|reason| {
            GateError::new(
                reason,
                "repair the manifests so the feature axis can be derived; a sweep over an axis it cannot read proves nothing",
            )
        })?;
        if axis.is_empty() {
            report.find(Finding::in_file(
                ROOT_MANIFEST,
                "no workspace member declares a feature, so the sweep would compile nothing",
                "keep the axis derived from the manifests, and delete this gate only when the workspace declares no features at all",
            ));
            return Ok(report);
        }

        if !ctx.has("--sweep") {
            report.note(format!(
                "advertised rust-version {version}, {} selection(s) on the axis, compiled by --sweep",
                axis.len()
            ));
            return Ok(report);
        }

        installed(&version)?;
        let mut compiled = 0usize;
        for pair in &axis {
            if let Some(failure) = compile_failure(&ctx.root, &version, pair)? {
                report.find(failure);
            }
            compiled += 1;
        }
        report.note(format!("compiled {compiled} selection(s) on {version}"));
        Ok(report)
    }
}

/// The version `[workspace.package].rust-version` advertises.
fn declared_version(manifest: &toml::Table) -> Option<String> {
    let value = manifest
        .get("workspace")?
        .get("package")?
        .get("rust-version")?
        .as_str()?
        .trim();
    (!value.is_empty()).then(|| value.to_string())
}

/// Whether the declared minimum names a toolchain rather than a channel.
///
/// `stable` and `beta` move, so a sweep against one of them reports whatever
/// the host had installed on the day it ran.
fn names_a_toolchain(version: &str) -> bool {
    let mut parts = version.split('.');
    let major = parts.next().unwrap_or_default();
    let minor = parts.next().unwrap_or_default();
    let patch = parts.next();
    if parts.next().is_some() {
        return false;
    }
    let numeric =
        |value: &str| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit());
    numeric(major) && numeric(minor) && patch.is_none_or(numeric)
}

/// Fail unless the named toolchain is installed.
///
/// Falling back to the default toolchain is the failure this gate exists to
/// prevent: it reports a pass for a compiler the workspace never claimed.
fn installed(version: &str) -> Result<(), GateError> {
    let listed = Command::new("rustup")
        .args(["toolchain", "list"])
        .output()
        .map_err(|error| {
            GateError::new(
                format!("cannot run `rustup toolchain list`: {error}"),
                "install rustup; the sweep pins a toolchain by name, and without rustup there is no way to compile on the advertised minimum",
            )
        })?;
    if !listed.status.success() {
        return Err(GateError::new(
            format!(
                "`rustup toolchain list` failed: {}",
                String::from_utf8_lossy(&listed.stderr).trim()
            ),
            "repair the rustup installation; a sweep that cannot confirm the toolchain would report some other compiler as the minimum",
        ));
    }
    let stdout = String::from_utf8_lossy(&listed.stdout);
    if stdout.lines().any(|line| line.starts_with(version)) {
        return Ok(());
    }
    Err(GateError::new(
        format!("the advertised toolchain {version} is not installed"),
        format!("rustup toolchain install {version}"),
    ))
}

/// Compile one selection on the named toolchain, reporting the failure.
///
/// The compile goes through the workspace cargo with a leading `+<version>`
/// rather than through `rustup run`, because the wrapper is what decides which
/// target directory a checkout builds into. A bare cargo compiled this
/// workspace into a directory another checkout owns, where the same member path
/// hashes to the same unit and one checkout's artifact answers another's
/// request. `installed` above still holds the toolchain to a named version.
fn compile_failure(root: &Path, version: &str, pair: &Pair) -> Result<Option<Finding>, GateError> {
    let output = cargo_runner::command(root)
        .arg(format!("+{version}"))
        .args(["check", "--locked", "-p"])
        .arg(&pair.member)
        .args(pair.cargo_flags())
        .output()
        .map_err(|error| {
            GateError::new(
                format!(
                    "cannot run `cargo +{version} check` for `{}`: {error}",
                    pair.label()
                ),
                "install the advertised toolchain so the sweep can compile against it",
            )
        })?;
    if output.status.success() {
        return Ok(None);
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    if let Some(missing) = cargo_runner::unmeasured(&stderr) {
        return Ok(Some(Finding::new(
            format!(
                "{} was not measured on {version}: the build named `{missing}`, which the build directory does not carry",
                pair.label()
            ),
            "run the sweep again against an intact build directory; a compile whose own inputs were deleted under it says nothing about the selection it was pointed at",
        )));
    }
    Ok(Some(Finding::new(
        format!(
            "{} does not compile on {version}: {}",
            pair.label(),
            first_error(&stderr)
        ),
        "fix the selection, or raise rust-version to a compiler that builds it and say so in the release notes",
    )))
}

/// The first compiler error in a failed build, for the finding message.
fn first_error(stderr: &str) -> String {
    stderr
        .lines()
        .find(|line| line.trim_start().starts_with("error"))
        .unwrap_or("the compiler reported no error line")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: a channel name installs whatever the host has today, so a sweep
    /// against one reports a pass for a compiler the workspace never claimed.
    /// Only the crate-private version predicate can be reached for this.
    #[test]
    fn a_channel_is_not_an_advertised_minimum() {
        assert!(names_a_toolchain("1.90"));
        assert!(names_a_toolchain("1.90.1"));
        assert!(!names_a_toolchain("stable"));
        assert!(!names_a_toolchain("1"));
        assert!(!names_a_toolchain("1.90.1.2"));
        assert!(!names_a_toolchain("1.x"));
    }

    /// WHY: an empty or whitespace declaration reads as present to a naive
    /// lookup, and the sweep would then pin an empty toolchain name.
    #[test]
    fn a_blank_declaration_is_no_declaration() {
        let blank: toml::Table =
            toml::from_str("[workspace.package]\nrust-version = \"  \"\n").expect("table");
        assert_eq!(declared_version(&blank), None);
        let declared: toml::Table =
            toml::from_str("[workspace.package]\nrust-version = \"1.90\"\n").expect("table");
        assert_eq!(declared_version(&declared), Some("1.90".to_string()));
    }

    /// WHY: the finding has to name the compiler error, because a sweep whose
    /// report says only that a selection failed sends the reader back to cargo.
    #[test]
    fn the_finding_carries_the_compiler_error() {
        let stderr = "   Compiling vyre-libs v0.1.0\nerror[E0432]: unresolved import `crate::vfs`\n  --> src/lib.rs:1:5\n";
        assert_eq!(
            first_error(stderr),
            "error[E0432]: unresolved import `crate::vfs`"
        );
        assert_eq!(
            first_error("warning: nothing\n"),
            "the compiler reported no error line"
        );
    }
}
