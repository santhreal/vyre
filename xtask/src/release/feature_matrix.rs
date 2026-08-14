//! Crate feature release evidence for Vyre.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::manifest_walk::MAX_MANIFEST_BYTES;

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

pub(crate) fn run(args: &[String]) {
    let output = match parse_output(args) {
        Ok(output) => output,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let vyre_root = crate::checkout::checkout_root();
    let roots = [vyre_root];
    let mut packages = Vec::new();
    let mut blockers = Vec::new();
    for root in &roots {
        collect_features(root, &mut packages, &mut blockers);
    }
    packages.sort_by(|left, right| left.name.cmp(&right.name));
    let missing_required_release_packages = REQUIRED_RELEASE_PACKAGES
        .iter()
        .copied()
        .filter(|required| !packages.iter().any(|package| package.name == *required))
        .collect::<Vec<_>>();
    if packages.is_empty() {
        blockers.push("feature matrix found zero packages".to_string());
    }
    blockers.extend(
        missing_required_release_packages.iter().map(|package| {
            format!("feature matrix is missing required release package `{package}`")
        }),
    );
    for package in &packages {
        if !package.malformed_features.is_empty() {
            blockers.push(format!(
                "{} has malformed feature definitions: {}",
                package.name,
                package.malformed_features.join(", ")
            ));
        }
        if package.feature_count > 0 && !package.has_default_feature {
            blockers.push(format!(
                "{} defines {} feature(s) but no explicit default feature policy",
                package.name, package.feature_count
            ));
        }
        if !package.unresolved_feature_members.is_empty() {
            blockers.push(format!(
                "{} has feature members that do not resolve to local features, optional dependencies, or dependency features: {}",
                package.name,
                package.unresolved_feature_members.join(", ")
            ));
        }
        if matches!(
            package.name.as_str(),
            "vyre" | "vyre-driver-cuda" | "vyre-driver-wgpu"
        ) && !package.default_feature_members.is_empty()
        {
            blockers.push(format!(
                "{} default feature set must stay empty; GPU release paths are explicit feature choices",
                package.name
            ));
        }
        if package.name == "vyre" {
            for required in ["cuda", "wgpu"] {
                if !package.features.iter().any(|feature| feature == required) {
                    blockers.push(format!(
                        "vyre top-level crate is missing release feature `{required}`"
                    ));
                }
            }
        }
        if package.name == "vyre-driver-cuda"
            && !package.features.iter().any(|feature| feature == "cuda")
        {
            blockers
                .push("vyre-driver-cuda is missing explicit `cuda` release feature".to_string());
        }
        if package.name == "vyre-driver-wgpu"
            && !package.features.iter().any(|feature| feature == "wgpu")
        {
            blockers.push(
                "vyre-driver-wgpu is missing explicit `wgpu` fallback release feature".to_string(),
            );
        }
    }
    let matrix = FeatureMatrix {
        schema_version: 1,
        required_release_packages: REQUIRED_RELEASE_PACKAGES.to_vec(),
        missing_required_release_packages,
        packages,
        blockers,
    };

    if let Some(parent) = output.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            eprintln!("Fix: failed to create `{}`: {error}", parent.display());
            std::process::exit(1);
        }
    }
    crate::output_arg::write_json(&output, &matrix);
    println!("feature-matrix: wrote {}", output.display());
    if !matrix.blockers.is_empty() {
        std::process::exit(1);
    }
}

fn collect_features(root: &Path, packages: &mut Vec<PackageFeatures>, blockers: &mut Vec<String>) {
    crate::manifest_walk::collect_manifests(
        root,
        "feature matrix",
        packages,
        blockers,
        parse_features,
    );
}

fn parse_features(path: &Path) -> Result<Option<PackageFeatures>, String> {
    let text = crate::output_arg::read_text_bounded(path, MAX_MANIFEST_BYTES, "release feature")
        .map_err(|error| {
            format!(
                "failed to read feature manifest `{}`: {error}",
                path.display()
            )
        })?;
    let value = toml::from_str::<toml::Value>(&text).map_err(|error| {
        format!(
            "failed to parse feature manifest `{}`: {error}",
            path.display()
        )
    })?;
    let Some(package) = value.get("package") else {
        return Ok(None);
    };
    let Some(name) = package
        .get("name")
        .and_then(toml::Value::as_str)
        .map(str::to_string)
    else {
        return Err(format!(
            "package manifest `{}` is missing package.name",
            path.display()
        ));
    };
    let mut features: Vec<String> = value
        .get("features")
        .and_then(toml::Value::as_table)
        .map(|table| table.keys().cloned().collect())
        .unwrap_or_default();
    features.sort();
    let has_default_feature = features.iter().any(|feature| feature == "default");
    let default_feature_members = value
        .get("features")
        .and_then(toml::Value::as_table)
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
    let malformed_features = value
        .get("features")
        .and_then(toml::Value::as_table)
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
    let dependency_names = dependency_names(&value);
    let optional_dependency_names = optional_dependency_names(&value);
    let unresolved_feature_members = unresolved_feature_members(
        value.get("features").and_then(toml::Value::as_table),
        &features,
        &dependency_names,
        &optional_dependency_names,
    );
    let release_policy = release_policy(&name);
    Ok(Some(PackageFeatures {
        name,
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

fn dependency_names(value: &toml::Value) -> Vec<String> {
    let mut names = Vec::new();
    collect_dependency_names_recursive(value, false, &mut names);
    names.sort();
    names.dedup();
    names
}

fn optional_dependency_names(value: &toml::Value) -> Vec<String> {
    let mut names = Vec::new();
    collect_dependency_names_recursive(value, true, &mut names);
    names.sort();
    names.dedup();
    names
}

fn collect_dependency_names_recursive(
    value: &toml::Value,
    optional_only: bool,
    out: &mut Vec<String>,
) {
    let Some(table) = value.as_table() else {
        return;
    };
    for (key, child) in table {
        if matches!(
            key.as_str(),
            "dependencies" | "dev-dependencies" | "build-dependencies"
        ) {
            if let Some(dependencies) = child.as_table() {
                for (name, dependency) in dependencies {
                    if !optional_only
                        || dependency.get("optional").and_then(toml::Value::as_bool) == Some(true)
                    {
                        out.push(name.clone());
                    }
                }
            }
        } else {
            collect_dependency_names_recursive(child, optional_only, out);
        }
    }
}

fn unresolved_feature_members(
    features_table: Option<&toml::value::Table>,
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

fn parse_output(args: &[String]) -> Result<PathBuf, String> {
    crate::output_arg::parse_output_arg(
        args,
        "feature-matrix",
        "Writes Vyre crate feature evidence.",
        default_output,
    )
}

fn default_output() -> PathBuf {
    crate::checkout::checkout_root().join("release/evidence/metadata/feature-matrix.json")
}
