//! Hold the crate feature evidence to the feature tables the manifests declare.

use std::path::Path;

use serde::Serialize;

use crate::artifact_gate::Inspection;
use crate::manifest_walk::{self, PackageManifest};

/// The artifact this gate owns, relative to the workspace root.
const ARTIFACT: &str = "release/evidence/metadata/feature-matrix.json";
#[derive(Debug, Serialize)]
struct FeatureMatrix {
    schema_version: u32,
    required_release_packages: Vec<&'static str>,
    missing_required_release_packages: Vec<&'static str>,
    packages: Vec<PackageFeatures>,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct PackageFeatures {
    name: String,
    manifest: String,
    feature_count: usize,
    has_default_feature: bool,
    default_feature_members: Vec<String>,
    features: Vec<String>,
    malformed_features: Vec<String>,
    unresolved_feature_members: Vec<String>,
    release_policy: &'static str,
}

const REQUIRED_RELEASE_PACKAGES: &[&str] = &["vyre", "vyre-driver-cuda", "vyre-driver-wgpu"];

crate::artifact_gate! {
    /// Holds the feature matrix to the feature tables in the manifests.
    FeatureMatrixGate,
    name: "feature-matrix",
    help: "Regenerate release/evidence/metadata/feature-matrix.json from every workspace manifest \
           and report each line the committed artifact disagrees on. Proves every feature table \
           parses, every feature member resolves to a local feature, an optional dependency or a \
           dependency feature, every package that declares features declares a default policy, the \
           three release packages exist with empty defaults, and that vyre, vyre-driver-cuda and \
           vyre-driver-wgpu declare their release features. Proves nothing about whether any \
           feature selection compiles: that is feature-isolation.",
    inspect: |ctx| inspect(&ctx.root),
}

/// What the manifests declare about features, and the artifact recording it.
fn inspect(root: &Path) -> Inspection {
    let mut inspection = Inspection::new();
    let mut packages = Vec::new();
    let mut blockers = Vec::new();
    manifest_walk::collect_manifests(
        root,
        "feature matrix",
        &mut packages,
        &mut blockers,
        parse_features,
    );
    for blocker in &blockers {
        inspection.blocked(
            ARTIFACT,
            blocker.clone(),
            "Repair the manifest the sentence names. A feature matrix built from a tree it could \
             not finish reading describes a smaller workspace than the one that ships.",
        );
    }
    packages.sort_by(|left, right| left.name.cmp(&right.name));
    let missing_required_release_packages = REQUIRED_RELEASE_PACKAGES
        .iter()
        .copied()
        .filter(|required| !packages.iter().any(|package| package.name == *required))
        .collect::<Vec<_>>();
    if packages.is_empty() {
        let blocker = "feature matrix found zero packages".to_string();
        inspection.blocked(
            ARTIFACT,
            blocker.clone(),
            "The walk reached no manifest at all, so every assertion below is vacuous. Check that \
             the gate is running against the checkout root.",
        );
        blockers.push(blocker);
    }
    for package in &missing_required_release_packages {
        let blocker = format!("feature matrix is missing required release package `{package}`");
        inspection.blocked(
            ARTIFACT,
            blocker.clone(),
            format!("Add `{package}` to the workspace, or drop it from the required release set."),
        );
        blockers.push(blocker);
    }
    for package in &packages {
        collect_package_blockers(package, &mut inspection, &mut blockers);
    }
    let matrix = FeatureMatrix {
        schema_version: 1,
        required_release_packages: REQUIRED_RELEASE_PACKAGES.to_vec(),
        missing_required_release_packages,
        packages,
        blockers,
    };
    inspection.generates(ARTIFACT, &matrix);
    inspection
}

/// Every release feature rule one package breaks.
fn collect_package_blockers(
    package: &PackageFeatures,
    inspection: &mut Inspection,
    blockers: &mut Vec<String>,
) {
    let mut record = |message: String, fix: String| {
        inspection.blocked(ARTIFACT, message.clone(), fix);
        blockers.push(message);
    };
    if !package.malformed_features.is_empty() {
        record(
            format!(
                "{} has malformed feature definitions: {}",
                package.name,
                package.malformed_features.join(", ")
            ),
            format!(
                "Every entry under [features] in {} must be an array of strings.",
                package.manifest
            ),
        );
    }
    if package.feature_count > 0 && !package.has_default_feature {
        record(
            format!(
                "{} defines {} feature(s) but no explicit default feature policy",
                package.name, package.feature_count
            ),
            format!(
                "Add a `default = [...]` entry to [features] in {}. An absent default is a \
                 decision nobody wrote down.",
                package.manifest
            ),
        );
    }
    if !package.unresolved_feature_members.is_empty() {
        record(
            format!(
                "{} has feature members that do not resolve to local features, optional dependencies, or dependency features: {}",
                package.name,
                package.unresolved_feature_members.join(", ")
            ),
            format!(
                "Each `feature:member` pair names a member cargo cannot resolve. Correct the name \
                 in {}, or declare the dependency it refers to.",
                package.manifest
            ),
        );
    }
    if matches!(
        package.name.as_str(),
        "vyre" | "vyre-driver-cuda" | "vyre-driver-wgpu"
    ) && !package.default_feature_members.is_empty()
    {
        record(
            format!(
                "{} default feature set must stay empty; GPU release paths are explicit feature choices",
                package.name
            ),
            format!(
                "Empty the `default` feature in {}. A GPU backend selected by a default is one \
                 nobody chose.",
                package.manifest
            ),
        );
    }
    if package.name == "vyre" {
        for required in ["cuda", "wgpu"] {
            if !package.features.iter().any(|feature| feature == required) {
                record(
                    format!("vyre top-level crate is missing release feature `{required}`"),
                    format!("Declare a `{required}` feature in {}.", package.manifest),
                );
            }
        }
    }
    if package.name == "vyre-driver-cuda"
        && !package.features.iter().any(|feature| feature == "cuda")
    {
        record(
            "vyre-driver-cuda is missing explicit `cuda` release feature".to_string(),
            format!("Declare a `cuda` feature in {}.", package.manifest),
        );
    }
    if package.name == "vyre-driver-wgpu"
        && !package.features.iter().any(|feature| feature == "wgpu")
    {
        record(
            "vyre-driver-wgpu is missing explicit `wgpu` fallback release feature".to_string(),
            format!("Declare a `wgpu` feature in {}.", package.manifest),
        );
    }
}

fn parse_features(path: &Path) -> Result<Option<PackageFeatures>, String> {
    let Some(PackageManifest { document, name }) =
        manifest_walk::parse_package_manifest(path, "release feature")?
    else {
        return Ok(None);
    };
    let features_table = document.get("features").and_then(toml::Value::as_table);
    let mut features: Vec<String> = features_table
        .map(|table| table.keys().cloned().collect())
        .unwrap_or_default();
    features.sort();
    let has_default_feature = features.iter().any(|feature| feature == "default");
    let default_feature_members = features_table
        .and_then(|table| table.get("default"))
        .and_then(toml::Value::as_array)
        .map(|members| {
            members
                .iter()
                .filter_map(toml::Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let malformed_features = features_table
        .map(|table| {
            table
                .iter()
                .filter_map(|(feature, members)| {
                    let Some(members) = members.as_array() else {
                        return Some(format!("{feature}: value is not an array"));
                    };
                    let bad_member = members.iter().any(|member| member.as_str().is_none());
                    bad_member.then(|| format!("{feature}: contains non-string member"))
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let dependency_names = crate::manifest_walk::dependency_names(&document);
    let optional_dependency_names = crate::manifest_walk::optional_dependency_names(&document);
    let unresolved_feature_members = unresolved_feature_members(
        features_table,
        &features,
        &dependency_names,
        &optional_dependency_names,
    );
    let release_policy = release_policy(&name);
    Ok(Some(PackageFeatures {
        name: name.to_string(),
        manifest: path.display().to_string(),
        feature_count: features.len(),
        has_default_feature,
        default_feature_members,
        features,
        malformed_features,
        unresolved_feature_members,
        release_policy,
    }))
}

/// Every `feature:member` pair naming a member cargo cannot resolve.
///
/// `pub(crate)` because the contract test drives it directly: the resolution
/// rules are the part of this gate a fixture can prove without a workspace.
pub(crate) fn unresolved_feature_members(
    features_table: Option<&toml::Table>,
    features: &[String],
    dependencies: &[String],
    optional_dependencies: &[String],
) -> Vec<String> {
    let Some(table) = features_table else {
        return Vec::new();
    };
    let mut unresolved = Vec::new();
    for (feature, members) in table {
        let Some(members) = members.as_array() else {
            continue;
        };
        for member in members.iter().filter_map(toml::Value::as_str) {
            if feature_member_resolves(member, features, dependencies, optional_dependencies) {
                continue;
            }
            unresolved.push(format!("{feature}:{member}"));
        }
    }
    unresolved.sort();
    unresolved
}

fn feature_member_resolves(
    member: &str,
    features: &[String],
    dependencies: &[String],
    optional_dependencies: &[String],
) -> bool {
    if let Some(dependency) = member.strip_prefix("dep:") {
        return optional_dependencies
            .iter()
            .any(|candidate| candidate == dependency);
    }
    if let Some((dependency, _feature)) = member.split_once('/') {
        return dependencies.iter().any(|candidate| candidate == dependency);
    }
    features.iter().any(|feature| feature == member)
        || optional_dependencies
            .iter()
            .any(|dependency| dependency == member)
}

fn release_policy(name: &str) -> &'static str {
    match name {
        "vyre" => {
            "top-level crate exposes explicit cuda and wgpu feature switches with empty default"
        }
        "vyre-driver-cuda" => {
            "CUDA backend crate keeps default empty and is selected explicitly by release tooling"
        }
        "vyre-driver-wgpu" => "WGPU backend crate keeps default empty as fallback path",
        _ => "feature definitions are syntactically valid and have an explicit default policy",
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unresolved_feature_members_and_release_policy_are_enforced() {
        let mut features_table = toml::Table::new();
        features_table.insert(
            "default".to_string(),
            toml::Value::Array(vec![
                toml::Value::String("cuda".to_string()),
                toml::Value::String("dep:optional_dep".to_string()),
                toml::Value::String("unresolved_feature".to_string()),
            ]),
        );

        let features = vec!["cuda".to_string(), "default".to_string()];
        let dependencies = vec!["required_dep".to_string()];
        let optional_dependencies = vec!["optional_dep".to_string()];

        let unresolved = unresolved_feature_members(
            Some(&features_table),
            &features,
            &dependencies,
            &optional_dependencies,
        );

        assert_eq!(unresolved, vec!["default:unresolved_feature".to_string()]);

        let mut inspection = Inspection::new();
        let mut blockers = Vec::new();
        let bad_driver = PackageFeatures {
            name: "vyre-driver-cuda".to_string(),
            manifest: "vyre-driver-cuda/Cargo.toml".to_string(),
            feature_count: 2,
            has_default_feature: true,
            default_feature_members: vec!["cuda".to_string()],
            features: vec!["cuda".to_string(), "default".to_string()],
            malformed_features: Vec::new(),
            unresolved_feature_members: Vec::new(),
            release_policy: release_policy("vyre-driver-cuda"),
        };
        collect_package_blockers(&bad_driver, &mut inspection, &mut blockers);
        assert!(blockers
            .iter()
            .any(|b| b.contains("vyre-driver-cuda default feature set must stay empty")));
    }
}
