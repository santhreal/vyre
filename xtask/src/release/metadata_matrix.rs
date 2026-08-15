//! Hold the crate metadata evidence to the package tables the manifests declare.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::artifact_gate::{self, Inspection};
use crate::gate::{Gate, GateCtx, GateError, Report};
use crate::manifest_walk::{
    self, workspace_package as load_workspace_package, PackageManifest, MAX_MANIFEST_BYTES,
};
use crate::release::release_train;

/// The artifact this gate owns, relative to the workspace root.
const ARTIFACT: &str = "release/evidence/metadata/metadata-matrix.json";

#[derive(Debug, Serialize)]
struct MetadataMatrix {
    schema_version: u32,
    publishable_package_count: usize,
    vyre_package_count: usize,
    internal_tooling_count: usize,
    root_patch_section_count: usize,
    required_release_surfaces: Vec<&'static str>,
    missing_required_release_surfaces: Vec<String>,
    packages: Vec<PackageMetadata>,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct PackageMetadata {
    name: String,
    manifest: String,
    version: Option<String>,
    description: Option<String>,
    license: Option<String>,
    readme: Option<String>,
    repository: Option<String>,
    publish: Option<bool>,
    release_kind: &'static str,
    release_group: &'static str,
    release_surface: &'static str,
    expected_version: Option<&'static str>,
    publish_policy: &'static str,
    blockers: Vec<String>,
}

/// A crate the release train must find in the metadata matrix.
///
/// Every required surface is a publishable crate. The workspace used to keep
/// one release surface that was deliberately not published; that crate left
/// the workspace, so the category has no members and no spelling here.
#[derive(Debug, Clone, Copy)]
struct RequiredReleaseSurface {
    name: &'static str,
    expected_version: &'static str,
    release_surface: &'static str,
}

const MAX_README_BYTES: u64 = 2_097_152;

fn required_release_surfaces() -> Vec<RequiredReleaseSurface> {
    vec![
        RequiredReleaseSurface {
            name: "vyre",
            expected_version: release_train::vyre_version(),
            release_surface: "vyre-engine",
        },
        RequiredReleaseSurface {
            name: "vyre-driver-cuda",
            expected_version: release_train::vyre_version(),
            release_surface: "cuda-backend",
        },
        RequiredReleaseSurface {
            name: "vyre-driver-wgpu",
            expected_version: release_train::vyre_version(),
            release_surface: "wgpu-backend",
        },
    ]
}

/// Holds the metadata matrix to the package tables in the manifests.
pub struct MetadataMatrixGate;

impl Gate for MetadataMatrixGate {
    fn name(&self) -> &'static str {
        "metadata-matrix"
    }

    fn help(&self) -> &'static str {
        "Regenerate release/evidence/metadata/metadata-matrix.json from every workspace manifest \
         and report each line the committed artifact disagrees on. Proves every publishable crate \
         declares version, description, license, an https repository and a readme that exists and \
         is not empty, that its version is the one the release train names, that no manifest \
         carries a [patch.crates-io] section, and that the three required release surfaces are \
         present. Proves nothing about whether a crate packages or publishes: that is \
         package-readiness."
    }

    fn generates(&self) -> bool {
        true
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        Ok(artifact_gate::settle_inspection(
            ctx,
            self.name(),
            inspect(&ctx.root),
        ))
    }
}

/// What the manifests declare about metadata, and the artifact recording it.
fn inspect(vyre_root: &Path) -> Inspection {
    let mut inspection = Inspection::new();
    let mut packages = Vec::new();
    let mut metadata_blockers = Vec::new();
    let workspace_package =
        load_workspace_package(vyre_root, "release metadata", &mut metadata_blockers);
    collect_packages(
        vyre_root,
        workspace_package.as_ref(),
        &mut packages,
        &mut metadata_blockers,
    );
    let (root_patch_section_count, patch_blockers) =
        root_patch_section_count(&[vyre_root.join("Cargo.toml")]);
    packages.sort_by(|left, right| left.name.cmp(&right.name));
    let required_release_surfaces = required_release_surfaces()
        .into_iter()
        .map(|surface| surface.name)
        .collect::<Vec<_>>();
    let missing_required_release_surfaces = missing_required_release_surfaces(&packages);
    let mut blockers: Vec<String> = packages
        .iter()
        .flat_map(|package| {
            package
                .blockers
                .iter()
                .map(|blocker| format!("{}: {blocker}", package.name))
        })
        .collect();
    for package in &packages {
        for blocker in &package.blockers {
            inspection.blocked(
                ARTIFACT,
                format!("{}: {blocker}", package.name),
                format!("Correct the [package] table in {}.", package.manifest),
            );
        }
    }
    for blocker in &metadata_blockers {
        inspection.blocked(
            ARTIFACT,
            blocker.clone(),
            "Repair the manifest the sentence names. A metadata matrix built from a tree it could \
             not finish reading describes a smaller workspace than the one that ships.",
        );
    }
    for blocker in &patch_blockers {
        inspection.blocked(
            ARTIFACT,
            blocker.clone(),
            "Make the root manifest readable so the patch scan can run. A scan that cannot read \
             its input reports zero patches, which is indistinguishable from a clean tree.",
        );
    }
    blockers.extend(metadata_blockers);
    blockers.extend(patch_blockers);
    for surface in &missing_required_release_surfaces {
        let blocker = format!("missing required release surface `{surface}`");
        inspection.blocked(
            ARTIFACT,
            blocker.clone(),
            format!(
                "The release train names `{surface}` at the declared version with a README and \
                 publishable-crate kind. Add the crate, or drop the surface from the release train."
            ),
        );
        blockers.push(blocker);
    }
    if root_patch_section_count > 0 {
        let blocker = format!(
            "release manifests contain {root_patch_section_count} [patch.crates-io] section(s); remove root patches before publishing"
        );
        inspection.blocked(
            ARTIFACT,
            blocker.clone(),
            "Delete the [patch.crates-io] section from the root manifest. A published crate built \
             against a patched dependency is not the crate anyone else will build.",
        );
        blockers.push(blocker);
    }
    let publishable_package_count = packages
        .iter()
        .filter(|package| package.release_kind == "publishable-crate")
        .count();
    let vyre_package_count = packages
        .iter()
        .filter(|package| package.release_group == "vyre")
        .count();
    let internal_tooling_count = packages
        .iter()
        .filter(|package| package.release_kind == "internal-tooling")
        .count();
    let matrix = MetadataMatrix {
        schema_version: 3,
        publishable_package_count,
        vyre_package_count,
        internal_tooling_count,
        root_patch_section_count,
        required_release_surfaces,
        missing_required_release_surfaces,
        packages,
        blockers,
    };
    inspection.generates(ARTIFACT, &matrix);
    inspection
}

fn root_patch_section_count(manifests: &[PathBuf]) -> (usize, Vec<String>) {
    let mut blockers = Vec::new();
    let count = manifests
        .iter()
        .map(|manifest| {
            match crate::output_arg::read_text_bounded(
                manifest,
                MAX_MANIFEST_BYTES,
                "release metadata",
            ) {
                Ok(text) => text
                    .lines()
                    .filter(|line| line.trim() == "[patch.crates-io]")
                    .count(),
                Err(error) => {
                    blockers.push(format!(
                        "failed to read manifest for patch scan `{}`: {error}",
                        manifest.display()
                    ));
                    0
                }
            }
        })
        .sum();
    (count, blockers)
}

fn collect_packages(
    root: &Path,
    workspace_package: Option<&toml::value::Table>,
    packages: &mut Vec<PackageMetadata>,
    blockers: &mut Vec<String>,
) {
    manifest_walk::collect_manifests(root, "metadata", packages, blockers, |path| {
        parse_package(path, workspace_package)
    });
}

fn parse_package(
    path: &Path,
    workspace_package: Option<&toml::value::Table>,
) -> Result<Option<PackageMetadata>, String> {
    let Some(PackageManifest { document, name }) =
        manifest_walk::parse_package_manifest(path, "release metadata")?
    else {
        return Ok(None);
    };
    let Some(table) = document.get("package").and_then(toml::Value::as_table) else {
        return Ok(None);
    };
    let version = inherited_string(table, workspace_package, "version");
    let description = inherited_string(table, workspace_package, "description");
    let license = inherited_string(table, workspace_package, "license");
    let readme = inherited_string(table, workspace_package, "readme");
    let repository = inherited_string(table, workspace_package, "repository");
    let publish = table.get("publish").and_then(toml::Value::as_bool);
    let release_kind = release_kind(&name, publish);
    let release_group = release_group(path, release_kind);
    let release_surface = release_surface(&name, release_group, release_kind);
    let expected_version = expected_version(&name, release_group, release_kind);
    let publish_policy = if release_kind == "internal-tooling" {
        "publish=false allowed for release tooling that is not a crates.io artifact"
    } else {
        "publishable release crate"
    };
    let mut blockers = Vec::new();
    let internal_tooling = release_kind == "internal-tooling";
    if !internal_tooling && version.as_ref().is_none_or(|value| value.trim().is_empty()) {
        blockers.push("missing package.version".to_string());
    }
    if !internal_tooling
        && description
            .as_ref()
            .is_none_or(|value| value.trim().is_empty())
    {
        blockers.push("missing package.description".to_string());
    }
    if !internal_tooling && license.as_ref().is_none_or(|value| value.trim().is_empty()) {
        blockers.push("missing package.license".to_string());
    }
    if !internal_tooling
        && repository
            .as_ref()
            .is_none_or(|value| value.trim().is_empty())
    {
        blockers.push("missing package.repository".to_string());
    } else if repository
        .as_ref()
        .is_some_and(|value| !value.starts_with("https://"))
    {
        blockers.push("package.repository must be an https URL".to_string());
    }
    if !internal_tooling && readme.as_ref().is_none_or(|value| value.trim().is_empty()) {
        blockers.push("missing package.readme".to_string());
    }
    if release_kind == "publishable-crate" && publish == Some(false) {
        blockers.push("package.publish=false blocks release packaging".to_string());
    }
    if let Some(expected) = expected_version {
        if version.as_deref() != Some(expected) {
            blockers.push(format!(
                "package.version must be `{expected}` for {release_group} release"
            ));
        }
    }
    if let Some(readme) = readme.as_ref() {
        let readme_path = path.parent().unwrap_or_else(|| Path::new(".")).join(readme);
        if !readme_path.exists() {
            blockers.push(format!("readme `{readme}` does not exist"));
        } else {
            match crate::output_arg::read_text_bounded(
                &readme_path,
                MAX_README_BYTES,
                "release metadata",
            ) {
                Ok(text) if text.trim().is_empty() => {
                    blockers.push(format!("readme `{readme}` is empty"));
                }
                Ok(_) => {}
                Err(error) => blockers.push(format!("readme `{readme}` is unreadable: {error}")),
            }
        }
    }
    Ok(Some(PackageMetadata {
        name: name.to_string(),
        manifest: path.display().to_string(),
        version,
        description,
        license,
        readme,
        repository,
        publish,
        release_kind,
        release_group,
        release_surface,
        expected_version,
        publish_policy,
        blockers,
    }))
}

fn missing_required_release_surfaces(packages: &[PackageMetadata]) -> Vec<String> {
    required_release_surfaces()
        .into_iter()
        .filter_map(|required| {
            let present = packages.iter().any(|package| {
                package.name == required.name
                    && package.version.as_deref() == Some(required.expected_version)
                    && package.release_surface == required.release_surface
                    && package.readme.as_deref() == Some("README.md")
                    && package.release_kind == "publishable-crate"
            });
            (!present).then(|| {
                format!(
                    "{}@{}:{}",
                    required.name, required.expected_version, required.release_surface
                )
            })
        })
        .collect()
}

fn inherited_string(
    table: &toml::value::Table,
    workspace_package: Option<&toml::value::Table>,
    key: &str,
) -> Option<String> {
    match table.get(key) {
        Some(value) => value
            .as_str()
            .map(str::to_string)
            .or_else(|| workspace_string_if_requested(value, workspace_package, key)),
        None => None,
    }
}

fn workspace_string_if_requested(
    value: &toml::Value,
    workspace_package: Option<&toml::value::Table>,
    key: &str,
) -> Option<String> {
    if value.get("workspace").and_then(toml::Value::as_bool) != Some(true) {
        return None;
    }
    workspace_package?
        .get(key)
        .and_then(toml::Value::as_str)
        .map(str::to_string)
}

fn release_kind(name: &str, publish: Option<bool>) -> &'static str {
    if required_release_surface(name).is_some() {
        return "publishable-crate";
    }
    if matches!(
        name,
        "xtask" | "vyre-bench" | "vyre-bench-competitors" | "vyre-conform" | "vyre-foundation-fuzz"
    ) {
        "internal-tooling"
    } else if publish == Some(false) {
        "internal-tooling"
    } else {
        "publishable-crate"
    }
}

fn required_release_surface(name: &str) -> Option<RequiredReleaseSurface> {
    required_release_surfaces()
        .into_iter()
        .find(|surface| surface.name == name)
}

fn release_group(_path: &Path, release_kind: &str) -> &'static str {
    if release_kind == "internal-tooling" {
        "internal-tooling"
    } else {
        "vyre"
    }
}

fn release_surface(name: &str, _release_group: &str, release_kind: &str) -> &'static str {
    match name {
        "vyre" => "vyre-engine",
        "vyre-driver-cuda" => "cuda-backend",
        "vyre-driver-wgpu" => "wgpu-backend",
        _ if release_kind == "internal-tooling" => "internal-tooling",
        _ => "vyre-crate",
    }
}

fn expected_version(name: &str, release_group: &str, release_kind: &str) -> Option<&'static str> {
    if release_kind == "internal-tooling" {
        return None;
    }
    if let Some(required) = required_release_surface(name) {
        return Some(required.expected_version);
    }
    release_train::release_group_version(release_group)
}

#[cfg(test)]
mod tests {
    use super::{expected_version, release_group, release_surface};
    use std::path::Path;

    #[test]
    fn release_classification_is_vyre_owned() {
        assert_eq!(
            release_group(Path::new("vyre/Cargo.toml"), "publishable-crate"),
            "vyre"
        );
        assert_eq!(
            release_surface("vyre-driver-cuda", "vyre", "publishable-crate"),
            "cuda-backend"
        );
        assert_eq!(
            expected_version("vyre-primitives", "vyre", "publishable-crate"),
            Some(crate::release::release_train::vyre_version())
        );
    }
}
