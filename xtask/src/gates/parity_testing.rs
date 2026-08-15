//! The gate that keeps the WGSL parity oracle out of production builds.
//!
//! This was `scripts/check_parity_testing_not_leaked.sh`, wired into CI only
//! through the `check-cat-a` composite.
//!
//! `parity-testing` enables a probe path that emits raw shader text, bypassing
//! the IR, validation and the conformance gate. It exists for the f32
//! transcendental parity oracle. A manifest that enables it outside a
//! development dependency links that path into a shipped binary.
//!
//! The scan is line-oriented over the manifest text rather than a parsed table,
//! because the rule is about the section a mention appears in: a feature name
//! reaches a dependency through `features = [...]`, a `[dependencies.x]` table,
//! a target-specific table or a `[features]` entry, and the section is what
//! decides whether that is legal.

use std::path::{Path, PathBuf};

use walkdir::WalkDir;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};

/// The feature that must never be enabled outside a development dependency.
const FEATURE: &str = "parity-testing";

/// The crate allowed to declare the feature in its own `[features]` table.
const DECLARING_CRATE: &str = "vyre-driver-wgpu";

/// The corrective action for every leak this gate reports.
const FIX: &str = "move the activation into the crate's [dev-dependencies] or [target.'cfg(...)'.dev-dependencies] block, because a production build must never link the raw-shader probe path";

/// Largest manifest this gate will read.
const MAX_MANIFEST_BYTES: u64 = 1_048_576;

/// Every manifest in the checkout, ignoring build output.
fn manifests(root: &Path) -> Result<Vec<PathBuf>, GateError> {
    let mut found = Vec::new();
    for entry in WalkDir::new(root).into_iter().filter_entry(|entry| {
        let name = entry.file_name().to_string_lossy();
        name != "target" && name != ".git"
    }) {
        let entry = entry.map_err(|error| {
            GateError::new(
                format!("cannot walk {}: {error}", root.display()),
                "make every directory in the checkout readable",
            )
        })?;
        if entry.file_type().is_file() && entry.file_name() == "Cargo.toml" {
            found.push(entry.path().to_path_buf());
        }
    }
    found.sort();
    Ok(found)
}

/// Whether a mention inside `section` of `crate_directory` is allowed.
fn is_allowed(crate_directory: &str, section: &str) -> bool {
    if crate_directory == DECLARING_CRATE && section == "[features]" {
        return true;
    }
    section.contains("dev-dependencies")
}

/// Keeps the raw-shader parity oracle out of every non-development dependency.
pub struct ParityTestingIsolated;

impl Gate for ParityTestingIsolated {
    fn name(&self) -> &'static str {
        "parity-testing-isolated"
    }

    fn help(&self) -> &'static str {
        "Fail when a manifest enables the parity-testing feature outside a development dependency"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let paths = manifests(&ctx.root)?;
        for path in &paths {
            let text =
                crate::output_arg::read_text_bounded(path, MAX_MANIFEST_BYTES, "parity-testing scan")
                    .map_err(|error| {
                        GateError::new(
                            format!("cannot read {}: {error}", path.display()),
                            "make every manifest in the checkout readable",
                        )
                    })?;
            let crate_directory = path
                .parent()
                .and_then(Path::file_name)
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            let mut section = String::new();
            for (index, line) in text.lines().enumerate() {
                let trimmed = line.split('#').next().unwrap_or("").trim();
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed.starts_with('[') && trimmed.ends_with(']') {
                    section = trimmed.to_string();
                    continue;
                }
                if !trimmed.contains(FEATURE) || is_allowed(&crate_directory, &section) {
                    continue;
                }
                report.find(Finding::at(
                    path.strip_prefix(&ctx.root).unwrap_or(path),
                    u32::try_from(index + 1).unwrap_or(u32::MAX),
                    format!(
                        "section `{}` enables `{FEATURE}`",
                        if section.is_empty() {
                            "[package]"
                        } else {
                            section.as_str()
                        }
                    ),
                    FIX,
                ));
            }
        }
        report.note(format!("read {} manifest(s)", paths.len()));
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the two legal cases are the whole rule, and a gate that allowed a
    /// third would read as coverage while the probe path shipped.
    #[test]
    fn allows_only_the_declaration_and_a_development_dependency() {
        assert!(is_allowed(DECLARING_CRATE, "[features]"));
        assert!(is_allowed("vyre-reference", "[dev-dependencies]"));
        assert!(is_allowed(
            "vyre-reference",
            "[target.'cfg(unix)'.dev-dependencies]"
        ));
        assert!(!is_allowed("vyre-reference", "[features]"));
        assert!(!is_allowed("vyre-reference", "[dependencies]"));
        assert!(!is_allowed(DECLARING_CRATE, "[dependencies]"));
    }
}
