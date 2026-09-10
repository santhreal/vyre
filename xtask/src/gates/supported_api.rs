//! The supported-API classification gate and the manifest generated from it.
//!
//! Every publishable workspace package declares an explicit publication class in
//! its own `[package.metadata.vyre]` table. This gate reads the committed
//! public-API snapshots under `docs/public-api/` and classifies every exported
//! item by seam, stability, feature, and wire compatibility.
//!
//! [`SUPPORTED_API_MANIFEST`] (`docs/SUPPORTED_API.toml`) records the resulting
//! classification. This gate verifies that every exported item is classified, that
//! no unclassified `pub` item or newly publishable package is introduced without
//! policy, and that the committed manifest matches the tree.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::crate_registry::{
    load_registry, CrateRecord, REGISTRY, VALID_PUBLICATION_CLASSES,
};
use crate::gates::public_api::{roster, SNAPSHOT_DIR};
use crate::gates::scan::Tree;

/// The rendered supported-API manifest path.
pub const SUPPORTED_API_MANIFEST: &str = "docs/SUPPORTED_API.toml";
/// The command that rewrites the supported-API manifest.
pub const WRITE_COMMAND: &str = "xtask supported-api --write";
/// Manifest schema version.
pub const SCHEMA_VERSION: u32 = 1;

/// One classified exported public API item.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct ClassifiedItem {
    /// Canonical item path or signature.
    pub path: String,
    /// Classified item kind (`mod`, `use`, `struct`, `enum`, `fn`, `trait`, `type`, `macro`, `const`, `static`).
    pub kind: String,
    /// Stability classification (`stable`, `extension`, `backend`, `internal`, `tooling`, `private`).
    pub stability: String,
    /// Cargo feature required or default.
    pub feature: String,
    /// Wire compatibility classification (`wire-bound`, `api-only`).
    pub wire_compatibility: String,
}

/// One classified publishable package.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackageClassification {
    /// Package name.
    pub name: String,
    /// Repository-relative directory.
    pub path: String,
    /// Name of the seam the package owns, as the architecture manifest states it.
    pub seam: String,
    /// Publication class from crate registry.
    pub publication_class: String,
    /// Stability level.
    pub stability: String,
    /// Classified exported items in stable order.
    pub items: Vec<ClassifiedItem>,
}

/// Classify a single public API snapshot line.
#[must_use]
pub fn classify_line(line: &str, publication_class: &str) -> ClassifiedItem {
    let trimmed = line.trim();
    let stability = match publication_class {
        "stable-consumer-sdk" => "stable",
        "extension-sdk" => "extension",
        "concrete-backend" => "backend",
        "internal-engine" => "internal",
        "conformance-tooling" => "tooling",
        "private-test-support" => "private",
        _ => "internal",
    }
    .to_string();

    let (kind, path) = if let Some(rest) = trimmed.strip_prefix("pub proc macro ") {
        ("macro", rest.trim())
    } else if let Some(rest) = trimmed.strip_prefix("pub mod ") {
        ("mod", rest.trim())
    } else if let Some(rest) = trimmed.strip_prefix("pub use ") {
        ("use", rest.trim())
    } else if let Some(rest) = trimmed.strip_prefix("pub struct ") {
        ("struct", rest.trim())
    } else if let Some(rest) = trimmed.strip_prefix("pub enum ") {
        ("enum", rest.trim())
    } else if let Some(rest) = trimmed.strip_prefix("pub fn ") {
        ("fn", rest.trim())
    } else if let Some(rest) = trimmed.strip_prefix("pub trait ") {
        ("trait", rest.trim())
    } else if let Some(rest) = trimmed.strip_prefix("pub type ") {
        ("type", rest.trim())
    } else if let Some(rest) = trimmed.strip_prefix("pub const ") {
        ("const", rest.trim())
    } else if let Some(rest) = trimmed.strip_prefix("pub static ") {
        ("static", rest.trim())
    } else {
        ("item", trimmed)
    };

    let wire_compatibility = if path.contains("wire")
        || path.contains("delta")
        || path.contains("wire_format")
        || path.contains("schema")
        || path.contains("Wire")
        || path.contains("Delta")
    {
        "wire-bound".to_string()
    } else {
        "api-only".to_string()
    };

    ClassifiedItem {
        path: path.to_string(),
        kind: kind.to_string(),
        stability,
        feature: "default".to_string(),
        wire_compatibility,
    }
}

/// Render the complete supported-API manifest as valid TOML.
#[must_use]
pub fn render_manifest(packages: &[PackageClassification]) -> String {
    let mut out = String::new();
    out.push_str("# Generated from committed public-API snapshots and docs/CRATE_OWNERSHIP.toml\n");
    out.push_str(&format!("# by `{WRITE_COMMAND}`.\n\n"));
    out.push_str(&format!("schema_version = {SCHEMA_VERSION}\n\n"));

    for pkg in packages {
        out.push_str("[[package]]\n");
        out.push_str(&format!("name = \"{}\"\n", pkg.name));
        out.push_str(&format!("path = \"{}\"\n", pkg.path));
        out.push_str(&format!("seam = \"{}\"\n", pkg.seam));
        out.push_str(&format!(
            "publication_class = \"{}\"\n",
            pkg.publication_class
        ));
        out.push_str(&format!("stability = \"{}\"\n", pkg.stability));
        out.push_str(&format!("exported_item_count = {}\n", pkg.items.len()));
        out.push_str("\n");
        for item in &pkg.items {
            out.push_str("  [[package.item]]\n");
            out.push_str(&format!(
                "  path = {}\n",
                crate::toml_text::quote(&item.path)
            ));
            out.push_str(&format!("  kind = \"{}\"\n", item.kind));
            out.push_str(&format!("  stability = \"{}\"\n", item.stability));
            out.push_str(&format!("  feature = \"{}\"\n", item.feature));
            out.push_str(&format!(
                "  wire_compatibility = \"{}\"\n\n",
                item.wire_compatibility
            ));
        }
    }
    out
}

/// Classify all publishable packages given the workspace tree and registry records.
pub fn classify_workspace(
    tree: &Tree,
    records: &[CrateRecord],
    report: &mut Report,
) -> Result<Vec<PackageClassification>, GateError> {
    let snapshotted = roster(tree)?;
    let by_package: BTreeMap<&str, &CrateRecord> = records
        .iter()
        .map(|rec| (rec.package.as_str(), rec))
        .collect();

    let mut classifications = Vec::new();

    for pkg in &snapshotted {
        let Some(record) = by_package.get(pkg.package.as_str()) else {
            report.find(Finding::in_file(
                REGISTRY,
                format!(
                    "publishable package `{}` has no row in {REGISTRY}",
                    pkg.package
                ),
                format!(
                    "add a [[crate]] row for `{}` with an explicit publication_class",
                    pkg.package
                ),
            ));
            continue;
        };

        if record.publication_class.is_empty() {
            report.find(Finding::in_file(
                REGISTRY,
                format!(
                    "package `{}` declares no publication_class in {REGISTRY}",
                    pkg.package
                ),
                format!("declare one of: {}", VALID_PUBLICATION_CLASSES.join(", ")),
            ));
            continue;
        }

        let stability = match record.publication_class.as_str() {
            "stable-consumer-sdk" => "stable",
            "extension-sdk" => "extension",
            "concrete-backend" => "backend",
            "internal-engine" => "internal",
            "conformance-tooling" => "tooling",
            "private-test-support" => "private",
            other => {
                report.find(Finding::in_file(
                    REGISTRY,
                    format!(
                        "package `{}` declares unknown publication_class `{other}`",
                        pkg.package
                    ),
                    format!("declare one of: {}", VALID_PUBLICATION_CLASSES.join(", ")),
                ));
                "unknown"
            }
        }
        .to_string();

        let snapshot_file = PathBuf::from(SNAPSHOT_DIR).join(format!("{}.txt", pkg.package));
        let snapshot_text = tree.read(&snapshot_file).unwrap_or_default();
        let mut items: Vec<ClassifiedItem> = snapshot_text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| classify_line(line, &record.publication_class))
            .collect();
        items.sort();
        items.dedup();

        classifications.push(PackageClassification {
            name: pkg.package.clone(),
            path: record.path.clone(),
            seam: record.seam.clone(),
            publication_class: record.publication_class.clone(),
            stability,
            items,
        });
    }

    classifications.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(classifications)
}

/// The `supported-api` gate behavior.
pub struct SupportedApi;

impl crate::gate::GateBehavior for SupportedApi {
    fn usage(&self) -> &'static [&'static str] {
        &["--write    regenerate docs/SUPPORTED_API.toml from snapshots and crate ownership"]
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        report.produced(SUPPORTED_API_MANIFEST);

        let records = load_registry(&tree, &mut report)?;
        report.cover_complete("workspace manifests", records.len());
        if !report.findings.is_empty() {
            return Ok(report);
        }

        let classifications = classify_workspace(&tree, &records, &mut report)?;
        if !report.findings.is_empty() {
            return Ok(report);
        }

        let rendered = render_manifest(&classifications);
        let manifest_path = ctx.root.join(SUPPORTED_API_MANIFEST);

        if ctx.write {
            fs::write(&manifest_path, &rendered).map_err(|error| {
                GateError::new(
                    format!("cannot write `{SUPPORTED_API_MANIFEST}`: {error}"),
                    "make the documentation directory writable",
                )
            })?;
            report.note(format!(
                "supported-api: wrote {SUPPORTED_API_MANIFEST} for {} publishable package(s)",
                classifications.len()
            ));
            return Ok(report);
        }

        let actual = fs::read_to_string(&manifest_path).unwrap_or_default();
        let actual_normalized: Vec<&str> = actual.lines().map(str::trim_end).collect();
        let rendered_normalized: Vec<&str> = rendered.lines().map(str::trim_end).collect();

        if actual_normalized != rendered_normalized {
            report.find(Finding::in_file(
                SUPPORTED_API_MANIFEST,
                format!("`{SUPPORTED_API_MANIFEST}` is out of date with public API snapshots or publication classes"),
                format!("run `{WRITE_COMMAND}` to regenerate the manifest"),
            ));
        }

        report.note(format!(
            "supported-api: verified {} publishable package(s)",
            classifications.len()
        ));
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_line_identifies_kinds_and_stability() {
        let item = classify_line(
            "pub struct vyre::compiler::CompileRequest",
            "stable-consumer-sdk",
        );
        assert_eq!(item.kind, "struct");
        assert_eq!(item.stability, "stable");
        assert_eq!(item.wire_compatibility, "api-only");

        let wire_item = classify_line(
            "pub fn vyre_foundation::ir::to_wire_bytes()",
            "internal-engine",
        );
        assert_eq!(wire_item.kind, "fn");
        assert_eq!(wire_item.stability, "internal");
        assert_eq!(wire_item.wire_compatibility, "wire-bound");
    }

    #[test]
    fn render_manifest_produces_valid_schema() {
        let pkgs = vec![PackageClassification {
            name: "vyre".to_string(),
            path: "vyre".to_string(),
            seam: "compiler-facade".to_string(),
            publication_class: "stable-consumer-sdk".to_string(),
            stability: "stable".to_string(),
            items: vec![ClassifiedItem {
                path: "vyre::compiler::CompileRequest".to_string(),
                kind: "struct".to_string(),
                stability: "stable".to_string(),
                feature: "default".to_string(),
                wire_compatibility: "api-only".to_string(),
            }],
        }];
        let rendered = render_manifest(&pkgs);
        assert!(rendered.contains("schema_version = 1"));
        assert!(rendered.contains("name = \"vyre\""));
        assert!(rendered.contains("publication_class = \"stable-consumer-sdk\""));
    }

    #[test]
    fn classify_line_covers_all_kinds() {
        assert_eq!(classify_line("pub mod foo", "extension-sdk").kind, "mod");
        assert_eq!(
            classify_line("pub use foo::bar", "extension-sdk").kind,
            "use"
        );
        assert_eq!(
            classify_line("pub enum foo::Bar", "extension-sdk").kind,
            "enum"
        );
        assert_eq!(
            classify_line("pub trait foo::Baz", "extension-sdk").kind,
            "trait"
        );
        assert_eq!(
            classify_line("pub type foo::T = u32", "extension-sdk").kind,
            "type"
        );
        assert_eq!(
            classify_line("pub const foo::C: u32 = 1", "extension-sdk").kind,
            "const"
        );
        assert_eq!(
            classify_line("pub static foo::S: u32 = 1", "extension-sdk").kind,
            "static"
        );
        assert_eq!(
            classify_line("pub proc macro foo::bar!()", "extension-sdk").kind,
            "macro"
        );
    }
}
