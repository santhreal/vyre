//! The gate that keeps the WGSL parity oracle out of production builds.
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

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// The feature that must never be enabled outside a development dependency.
const FEATURE: &str = "parity-testing";

/// The crate allowed to declare the feature in its own `[features]` table.
const DECLARING_CRATE: &str = "vyre-driver-wgpu";

/// The corrective action for every leak this gate reports.
const FIX: &str = "move the activation into the crate's [dev-dependencies] or [target.'cfg(...)'.dev-dependencies] block, because a production build must never link the raw-shader probe path";

/// Every manifest the checkout lists, nested and non-member ones included,
/// because a fuzz or example manifest links the probe path as surely as a
/// member does.
fn manifests(tree: &Tree) -> Vec<PathBuf> {
    tree.paths()
        .iter()
        .filter(|path| path.file_name().is_some_and(|name| name == "Cargo.toml"))
        .cloned()
        .collect()
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

impl crate::gate::GateBehavior for ParityTestingIsolated {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let tree = Tree::open(&ctx.root)?;
        let paths = manifests(&tree);
        report.cover_complete("parity test manifests", paths.len());
        for path in &paths {
            let text = tree.read(path)?;
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
                    path,
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
