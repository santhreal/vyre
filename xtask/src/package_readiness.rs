//! Pre-publish package graph evidence for the Vyre release.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use serde::Serialize;

use crate::release_train;

const MAX_JSON_BYTES: u64 = 8_388_608;
const MAX_MANIFEST_BYTES: u64 = 1_048_576;

#[derive(Debug, Serialize)]
struct PackageReadiness {
    schema_version: u32,
    release_train: ReleaseTrain,
    publish_order: Vec<PublishStep>,
    package_verify_passed: Vec<&'static str>,
    observed_package_failures: Vec<ObservedPackageFailure>,
    missing_metadata_packages: Vec<String>,
    extra_metadata_packages: Vec<String>,
    dependency_order_edges: Vec<DependencyEdge>,
    package_content_checks: Vec<PackageContentCheck>,
    versioned_local_dependencies: Vec<VersionedLocalDependency>,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ReleaseTrain {
    vyre: &'static str,
    cuda_release_path: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct PublishStep {
    package: &'static str,
    version: &'static str,
    manifest: &'static str,
}

#[derive(Debug, Serialize)]
struct ObservedPackageFailure {
    package: String,
    command: &'static str,
    reason: String,
}

#[derive(Debug, Serialize)]
struct DependencyEdge {
    package: String,
    dependency: String,
    dependency_version: String,
    manifest: String,
}

#[derive(Debug, Serialize)]
struct VersionedLocalDependency {
    package: String,
    dependency: String,
    version: String,
    manifest: String,
    source: &'static str,
}

#[derive(Debug, Serialize)]
struct PackageContentCheck {
    package: String,
    manifest: String,
    cargo_package_list_succeeded: bool,
    file_count: usize,
    file_list_digest: String,
    rust_source_count: usize,
    missing_required_files: Vec<String>,
    forbidden_files: Vec<String>,
    command_error: Option<String>,
    blockers: Vec<String>,
}

fn publish_order() -> Vec<PublishStep> {
    vec![
        // First: no internal vyre dependencies, only third-party ones. It was dropped from
        // the train after 0.6.2 and went stale on crates.io while every sibling advanced,
        // which is why in-workspace consumers had to pin it path-only. A crate that is
        // already public stays current with the train.
        step(
            "vyre-grammar-gen",
            release_train::vyre_version(),
            "vyre-grammar-gen/Cargo.toml",
        ),
        step(
            "vyre-macros",
            release_train::vyre_version(),
            "vyre-macros/Cargo.toml",
        ),
        step(
            "vyre-spec",
            release_train::vyre_version(),
            "vyre-spec/Cargo.toml",
        ),
        step(
            "vyre-lints",
            release_train::vyre_version(),
            "vyre-lints/Cargo.toml",
        ),
        step(
            "vyre-foundation",
            release_train::vyre_version(),
            "vyre-foundation/Cargo.toml",
        ),
        step(
            "vyre-lower",
            release_train::vyre_version(),
            "vyre-lower/Cargo.toml",
        ),
        step(
            "vyre-megakernel",
            release_train::vyre_version(),
            "vyre-megakernel/Cargo.toml",
        ),
        step(
            "vyre-emit-ptx",
            release_train::vyre_version(),
            "vyre-emit-ptx/Cargo.toml",
        ),
        step(
            "vyre-primitives",
            release_train::vyre_version(),
            "vyre-primitives/Cargo.toml",
        ),
        step(
            "vyre-reference",
            release_train::vyre_version(),
            "vyre-reference/Cargo.toml",
        ),
        step(
            "vyre-self-substrate",
            release_train::vyre_version(),
            "vyre-self-substrate/Cargo.toml",
        ),
        step(
            "vyre-driver",
            release_train::vyre_version(),
            "vyre-driver/Cargo.toml",
        ),
        step(
            "vyre-runtime",
            release_train::vyre_version(),
            "vyre-runtime/Cargo.toml",
        ),
        step(
            "vyre-emit-naga",
            release_train::vyre_version(),
            "vyre-emit-naga/Cargo.toml",
        ),
        step(
            "vyre-emit-spirv",
            release_train::vyre_version(),
            "vyre-emit-spirv/Cargo.toml",
        ),
        step(
            "vyre-emit-metal",
            release_train::vyre_version(),
            "vyre-emit-metal/Cargo.toml",
        ),
        step(
            "vyre-driver-cuda",
            release_train::vyre_version(),
            "vyre-driver-cuda/Cargo.toml",
        ),
        step(
            "vyre-driver-wgpu",
            release_train::vyre_version(),
            "vyre-driver-wgpu/Cargo.toml",
        ),
        step(
            "vyre-driver-metal",
            release_train::vyre_version(),
            "vyre-driver-metal/Cargo.toml",
        ),
        step(
            "vyre-driver-spirv",
            release_train::vyre_version(),
            "vyre-driver-spirv/Cargo.toml",
        ),
        step(
            "vyre-driver-reference",
            release_train::vyre_version(),
            "vyre-driver-reference/Cargo.toml",
        ),
        step(
            "vyre-intrinsics",
            release_train::vyre_version(),
            "vyre-intrinsics/Cargo.toml",
        ),
        step(
            "vyre-libs",
            release_train::vyre_version(),
            "vyre-libs/Cargo.toml",
        ),
        step(
            "vyre-safetensors",
            release_train::vyre_version(),
            "vyre-safetensors/Cargo.toml",
        ),
        step("vyre", release_train::vyre_version(), "vyre/Cargo.toml"),
        step(
            "vyre-debug",
            release_train::vyre_version(),
            "vyre-debug/Cargo.toml",
        ),
        step(
            "vyre-aot",
            release_train::vyre_version(),
            "vyre-aot/Cargo.toml",
        ),
    ]
}

const fn step(package: &'static str, version: &'static str, manifest: &'static str) -> PublishStep {
    PublishStep {
        package,
        version,
        manifest,
    }
}

pub(crate) fn run(args: &[String]) {
    let output = match parse_output(args) {
        Ok(output) => output,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let vyre_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let metadata_path = vyre_root.join("release/evidence/metadata/metadata-matrix.json");
    let mut blockers = Vec::new();
    let metadata_packages = metadata_publishable_packages(&metadata_path, &mut blockers);
    let publish_order = publish_order();
    let ordered_packages = publish_order
        .iter()
        .map(|step| step.package.to_string())
        .collect::<BTreeSet<_>>();
    let missing_metadata_packages = metadata_packages
        .difference(&ordered_packages)
        .cloned()
        .collect::<Vec<_>>();
    let extra_metadata_packages = ordered_packages
        .difference(&metadata_packages)
        .cloned()
        .collect::<Vec<_>>();
    for package in &missing_metadata_packages {
        blockers.push(format!(
            "metadata publishable package `{package}` is missing from publish_order"
        ));
    }
    for package in &extra_metadata_packages {
        blockers.push(format!(
            "publish_order package `{package}` is not publishable in metadata matrix"
        ));
    }

    let order_index = publish_order
        .iter()
        .enumerate()
        .map(|(index, step)| (step.package, index))
        .collect::<BTreeMap<_, _>>();
    let mut dependency_order_edges = Vec::new();
    let mut versioned_local_dependencies = Vec::new();
    for (consumer_index, step) in publish_order.iter().enumerate() {
        let manifest = vyre_root.join(step.manifest);
        check_manifest_package(step, &manifest, &mut blockers);
        collect_dependency_edges(
            step,
            consumer_index,
            &manifest,
            &order_index,
            &mut dependency_order_edges,
            &mut versioned_local_dependencies,
            &mut blockers,
        );
    }

    let package_content_checks = publish_order
        .iter()
        .map(|step| audit_package_contents(&vyre_root, step, &publish_order))
        .collect::<Vec<_>>();
    for check in &package_content_checks {
        blockers.extend(
            check
                .blockers
                .iter()
                .map(|blocker| format!("package `{}` archive: {blocker}", check.package)),
        );
    }

    dependency_order_edges.sort_by(|left, right| {
        left.package
            .cmp(&right.package)
            .then(left.dependency.cmp(&right.dependency))
    });
    versioned_local_dependencies.sort_by(|left, right| {
        left.package
            .cmp(&right.package)
            .then(left.dependency.cmp(&right.dependency))
    });

    let readiness = PackageReadiness {
        schema_version: 3,
        release_train: ReleaseTrain {
            vyre: release_train::vyre_version(),
            cuda_release_path: true,
        },
        publish_order,
        package_verify_passed: release_train::package_verify_passed(),
        observed_package_failures: vec![
            // The versions come from the release train: these two failures are inherent to
            // the publish order (a crate cannot be packaged before its dependency is
            // indexed), so they recur at every version and must not be pinned to one.
            ObservedPackageFailure {
                package: format!("vyre-lower@{}", release_train::vyre_version()),
                command: "cargo_full package --allow-dirty --manifest-path vyre-lower/Cargo.toml",
                reason: format!(
                    "crates.io does not yet contain vyre-foundation@{}",
                    release_train::vyre_version()
                ),
            },
        ],
        missing_metadata_packages,
        extra_metadata_packages,
        dependency_order_edges,
        versioned_local_dependencies,
        package_content_checks,
        blockers,
    };
    if let Some(parent) = output.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            eprintln!("Fix: failed to create `{}`: {error}", parent.display());
            std::process::exit(1);
        }
    }
    crate::output_arg::write_json(&output, &readiness);
    println!("package-readiness: wrote {}", output.display());
    if !readiness.blockers.is_empty() {
        std::process::exit(1);
    }
}

fn audit_package_contents(
    root: &Path,
    step: &PublishStep,
    publish_order: &[PublishStep],
) -> PackageContentCheck {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"));
    let patch_args = match cargo_package_patch_args(root, step, publish_order) {
        Ok(args) => args,
        Err(error) => {
            return PackageContentCheck {
                package: step.package.to_string(),
                manifest: step.manifest.to_string(),
                cargo_package_list_succeeded: false,
                file_count: 0,
                file_list_digest: String::new(),
                rust_source_count: 0,
                missing_required_files: Vec::new(),
                forbidden_files: Vec::new(),
                command_error: Some(error.clone()),
                blockers: vec![format!(
                    "could not prepare local release dependency patches: {error}"
                )],
            };
        }
    };
    let output = Command::new(cargo)
        .current_dir(root)
        .args(&patch_args)
        .args([
            "package",
            "--list",
            "--allow-dirty",
            "--manifest-path",
            step.manifest,
        ])
        .output();
    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            inspect_package_file_list(step, &stdout)
        }
        Ok(output) => {
            let error = bounded_command_error(&output.stderr);
            PackageContentCheck {
                package: step.package.to_string(),
                manifest: step.manifest.to_string(),
                cargo_package_list_succeeded: false,
                file_count: 0,
                file_list_digest: String::new(),
                rust_source_count: 0,
                missing_required_files: Vec::new(),
                forbidden_files: Vec::new(),
                command_error: Some(error.clone()),
                blockers: vec![format!("`cargo package --list` failed: {error}")],
            }
        }
        Err(error) => PackageContentCheck {
            package: step.package.to_string(),
            manifest: step.manifest.to_string(),
            cargo_package_list_succeeded: false,
            file_count: 0,
            file_list_digest: String::new(),
            rust_source_count: 0,
            missing_required_files: Vec::new(),
            forbidden_files: Vec::new(),
            command_error: Some(error.to_string()),
            blockers: vec![format!("could not launch `cargo package --list`: {error}")],
        },
    }
}

fn cargo_package_patch_args(
    root: &Path,
    step: &PublishStep,
    publish_order: &[PublishStep],
) -> Result<Vec<OsString>, String> {
    let manifest = root.join(step.manifest);
    let text = crate::output_arg::read_text_bounded(&manifest, MAX_MANIFEST_BYTES, "")
        .map_err(|error| format!("failed to read `{}`: {error}", manifest.display()))?;
    let value = toml::from_str::<toml::Value>(&text)
        .map_err(|error| format!("failed to parse `{}`: {error}", manifest.display()))?;
    let mut workspace_blockers = Vec::new();
    let workspace_dependencies = workspace_dependencies(&manifest, &mut workspace_blockers);
    if !workspace_blockers.is_empty() {
        return Err(workspace_blockers.join("; "));
    }

    let local_manifests = publish_order
        .iter()
        .filter(|candidate| candidate.package != step.package)
        .map(|candidate| (candidate.package, candidate))
        .collect::<BTreeMap<_, _>>();
    let mut patches = BTreeMap::<String, PathBuf>::new();
    for table_name in ["dependencies", "dev-dependencies", "build-dependencies"] {
        let Some(table) = value.get(table_name).and_then(toml::Value::as_table) else {
            continue;
        };
        for (dependency, spec) in table {
            let package = dependency_package_name(spec, &workspace_dependencies, dependency);
            let Some(candidate) = local_manifests.get(package.as_str()) else {
                continue;
            };
            let Some(version) = dependency_version(spec, &workspace_dependencies, dependency)
            else {
                continue;
            };
            if version != candidate.version {
                continue;
            }
            let Some(crate_dir) = root
                .join(candidate.manifest)
                .parent()
                .map(Path::to_path_buf)
            else {
                return Err(format!(
                    "release package `{package}` manifest `{}` has no parent directory",
                    candidate.manifest
                ));
            };
            patches.insert(package, crate_dir);
        }
    }

    let mut args = Vec::with_capacity(patches.len() * 2);
    for (package, path) in patches {
        args.push(OsString::from("--config"));
        args.push(OsString::from(format!(
            "patch.crates-io.{package}.path={:?}",
            path.to_string_lossy()
        )));
    }
    Ok(args)
}

fn dependency_package_name(
    spec: &toml::Value,
    workspace_dependencies: &BTreeMap<String, toml::Value>,
    dependency: &str,
) -> String {
    spec.get("package")
        .and_then(toml::Value::as_str)
        .or_else(|| {
            dependency_uses_workspace(spec).then(|| {
                workspace_dependencies
                    .get(dependency)
                    .and_then(|value| value.get("package"))
                    .and_then(toml::Value::as_str)
            })?
        })
        .unwrap_or(dependency)
        .to_string()
}

fn inspect_package_file_list(step: &PublishStep, stdout: &str) -> PackageContentCheck {
    const REQUIRED_FILES: &[&str] = &[
        "Cargo.toml",
        "Cargo.toml.orig",
        "README.md",
        "LICENSE-APACHE",
        "LICENSE-MIT",
    ];

    let mut files = stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    files.sort();
    files.dedup();

    let file_set = files.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let missing_required_files = REQUIRED_FILES
        .iter()
        .filter(|required| !file_set.contains(**required))
        .map(|required| (*required).to_string())
        .collect::<Vec<_>>();
    let forbidden_files = files
        .iter()
        .filter(|path| package_path_is_forbidden(path))
        .cloned()
        .collect::<Vec<_>>();
    let rust_source_count = files
        .iter()
        .filter(|path| path.starts_with("src/") && path.ends_with(".rs"))
        .count();
    let mut blockers = Vec::new();
    if files.is_empty() {
        blockers.push("`cargo package --list` returned no files".to_string());
    }
    if !missing_required_files.is_empty() {
        blockers.push(format!(
            "missing required package files: {}",
            missing_required_files.join(", ")
        ));
    }
    if rust_source_count == 0 {
        blockers.push("contains no Rust source under `src/`".to_string());
    }
    if !forbidden_files.is_empty() {
        blockers.push(format!(
            "contains internal or unsafe package paths: {}",
            forbidden_files.join(", ")
        ));
    }
    let canonical_file_list = files.join("\n");
    PackageContentCheck {
        package: step.package.to_string(),
        manifest: step.manifest.to_string(),
        cargo_package_list_succeeded: true,
        file_count: files.len(),
        file_list_digest: format!(
            "blake3:{}",
            blake3::hash(canonical_file_list.as_bytes()).to_hex()
        ),
        rust_source_count,
        missing_required_files,
        forbidden_files,
        command_error: None,
        blockers,
    }
}

fn package_path_is_forbidden(path: &str) -> bool {
    const FORBIDDEN_FILE_NAMES: &[&str] = &[
        "AGENTS.md",
        "BACKLOG.md",
        "CLAUDE.md",
        "GEMINI.md",
        "SKILL.md",
    ];

    let path = Path::new(path);
    if path.starts_with("benches/baselines") || path.starts_with("tests/corpus") {
        return true;
    }
    if path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::RootDir))
    {
        return true;
    }
    path.components().any(|component| {
        let Component::Normal(component) = component else {
            return false;
        };
        let component = component.to_string_lossy();
        FORBIDDEN_FILE_NAMES.contains(&component.as_ref())
            || matches!(component.as_ref(), ".git" | "credentials" | "target")
            || component == ".env"
            || component.starts_with(".env.")
    })
}

fn bounded_command_error(stderr: &[u8]) -> String {
    const MAX_ERROR_BYTES: usize = 4_096;
    let stderr = &stderr[..stderr.len().min(MAX_ERROR_BYTES)];
    let message = String::from_utf8_lossy(stderr).trim().to_string();
    if message.is_empty() {
        "command exited unsuccessfully without a diagnostic".to_string()
    } else {
        message
    }
}

pub(crate) fn package_content_evidence_issues(value: &serde_json::Value) -> Vec<String> {
    let Some(publish_order) = value
        .get("publish_order")
        .and_then(serde_json::Value::as_array)
    else {
        return vec![
            "publish_order must be an array before package contents can be proven".to_string(),
        ];
    };
    let Some(content_checks) = value
        .get("package_content_checks")
        .and_then(serde_json::Value::as_array)
    else {
        return vec!["package_content_checks must be an array".to_string()];
    };

    let mut issues = Vec::new();
    let mut expected = BTreeMap::<String, String>::new();
    for entry in publish_order {
        let Some(package) = entry.get("package").and_then(serde_json::Value::as_str) else {
            issues.push("publish_order entry is missing package".to_string());
            continue;
        };
        let Some(manifest) = entry.get("manifest").and_then(serde_json::Value::as_str) else {
            issues.push(format!(
                "publish_order package `{package}` is missing manifest"
            ));
            continue;
        };
        if expected
            .insert(package.to_string(), manifest.to_string())
            .is_some()
        {
            issues.push(format!(
                "publish_order contains duplicate package `{package}`"
            ));
        }
    }

    let mut observed = BTreeMap::<String, &serde_json::Value>::new();
    for check in content_checks {
        let Some(package) = check.get("package").and_then(serde_json::Value::as_str) else {
            issues.push("package_content_checks entry is missing package".to_string());
            continue;
        };
        if observed.insert(package.to_string(), check).is_some() {
            issues.push(format!(
                "package_content_checks contains duplicate package `{package}`"
            ));
        }
    }
    for (package, manifest) in &expected {
        let Some(check) = observed.get(package) else {
            issues.push(format!(
                "package_content_checks is missing publishable package `{package}`"
            ));
            continue;
        };
        if check.get("manifest").and_then(serde_json::Value::as_str) != Some(manifest.as_str()) {
            issues.push(format!(
                "package `{package}` content check manifest does not match `{manifest}`"
            ));
        }
        if check
            .get("cargo_package_list_succeeded")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        {
            issues.push(format!(
                "package `{package}` did not pass `cargo package --list`"
            ));
        }
        for field in ["file_count", "rust_source_count"] {
            if check
                .get(field)
                .and_then(serde_json::Value::as_u64)
                .is_none_or(|count| count == 0)
            {
                issues.push(format!("package `{package}` has non-positive `{field}`"));
            }
        }
        let valid_digest = check
            .get("file_list_digest")
            .and_then(serde_json::Value::as_str)
            .and_then(|digest| digest.strip_prefix("blake3:"))
            .is_some_and(|digest| {
                digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            });
        if !valid_digest {
            issues.push(format!("package `{package}` has invalid file_list_digest"));
        }
        for field in ["missing_required_files", "forbidden_files", "blockers"] {
            if check
                .get(field)
                .and_then(serde_json::Value::as_array)
                .is_none_or(|entries| !entries.is_empty())
            {
                issues.push(format!(
                    "package `{package}` content check field `{field}` must be an empty array"
                ));
            }
        }
        if check
            .get("command_error")
            .is_none_or(|error| !error.is_null())
        {
            issues.push(format!(
                "package `{package}` content check command_error must be null"
            ));
        }
    }
    for package in observed.keys() {
        if !expected.contains_key(package) {
            issues.push(format!(
                "package_content_checks contains non-publish-order package `{package}`"
            ));
        }
    }
    issues
}

fn metadata_publishable_packages(path: &Path, blockers: &mut Vec<String>) -> BTreeSet<String> {
    let text = match crate::output_arg::read_text_bounded(path, MAX_JSON_BYTES, "") {
        Ok(text) => text,
        Err(error) => {
            blockers.push(format!(
                "failed to read metadata matrix `{}`: {error}",
                path.display()
            ));
            return BTreeSet::new();
        }
    };
    let value = match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(value) => value,
        Err(error) => {
            blockers.push(format!(
                "failed to parse metadata matrix `{}`: {error}",
                path.display()
            ));
            return BTreeSet::new();
        }
    };
    value
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter(|package| {
            package
                .get("release_kind")
                .and_then(serde_json::Value::as_str)
                == Some("publishable-crate")
        })
        .filter_map(|package| package.get("name").and_then(serde_json::Value::as_str))
        .map(str::to_string)
        .collect()
}

fn check_manifest_package(step: &PublishStep, manifest: &Path, blockers: &mut Vec<String>) {
    let value = match read_manifest(manifest, blockers) {
        Some(value) => value,
        None => return,
    };
    let Some(package) = value.get("package").and_then(toml::Value::as_table) else {
        blockers.push(format!("{} has no [package] table", manifest.display()));
        return;
    };
    if package.get("name").and_then(toml::Value::as_str) != Some(step.package) {
        blockers.push(format!(
            "{} package.name does not match publish_order `{}`",
            manifest.display(),
            step.package
        ));
    }
    if package_version(package) != Some(step.version) {
        blockers.push(format!(
            "{} package.version does not match publish_order `{}`",
            manifest.display(),
            step.version
        ));
    }
    if package.get("publish").and_then(toml::Value::as_bool) == Some(false) {
        blockers.push(format!(
            "{} is publish=false but appears in publish_order",
            step.package
        ));
    }
}

fn collect_dependency_edges(
    step: &PublishStep,
    consumer_index: usize,
    manifest: &Path,
    order_index: &BTreeMap<&'static str, usize>,
    dependency_order_edges: &mut Vec<DependencyEdge>,
    versioned_local_dependencies: &mut Vec<VersionedLocalDependency>,
    blockers: &mut Vec<String>,
) {
    let value = match read_manifest(manifest, blockers) {
        Some(value) => value,
        None => return,
    };
    let workspace_dependencies = workspace_dependencies(manifest, blockers);
    for table_name in ["dependencies", "dev-dependencies", "build-dependencies"] {
        let Some(table) = value.get(table_name).and_then(toml::Value::as_table) else {
            continue;
        };
        for (dependency, spec) in table {
            let Some(local_path) =
                dependency_has_local_path(spec, &workspace_dependencies, dependency)
            else {
                continue;
            };
            let Some(version) = dependency_version(spec, &workspace_dependencies, dependency)
            else {
                // Path-only `[dev-dependencies]` are NOT a publish blocker: cargo
                // strips dev-dependencies that carry no version from the published
                // manifest (they cannot be resolved from crates.io and never reach
                // downstream consumers), so the crate publishes cleanly. This is how
                // the 0.6.3 train shipped, its publish-readiness recorded zero
                // blockers with these same path-only dev-deps, several of which are
                // deliberately path-only to break the dev-dependency publish cycle
                // (the dependency is published AFTER its consumer). Regular and build
                // dependencies still block: those DO ship in the published manifest.
                if table_name != "dev-dependencies" {
                    blockers.push(format!(
                        "{} dependency `{dependency}` in [{table_name}] has local path `{local_path}` but no crates.io version",
                        manifest.display()
                    ));
                }
                continue;
            };
            versioned_local_dependencies.push(VersionedLocalDependency {
                package: step.package.to_string(),
                dependency: dependency.clone(),
                version: version.clone(),
                manifest: manifest.display().to_string(),
                source: if dependency_uses_workspace(spec) {
                    "workspace"
                } else {
                    "manifest"
                },
            });
            if table_name != "dev-dependencies" {
                if let Some(dependency_index) = order_index.get(dependency.as_str()) {
                    dependency_order_edges.push(DependencyEdge {
                        package: step.package.to_string(),
                        dependency: dependency.clone(),
                        dependency_version: version,
                        manifest: manifest.display().to_string(),
                    });
                    if *dependency_index >= consumer_index {
                        blockers.push(format!(
                            "publish_order puts `{}` before dependency `{dependency}`",
                            step.package
                        ));
                    }
                }
            }
        }
    }
}

fn workspace_dependencies(
    manifest: &Path,
    blockers: &mut Vec<String>,
) -> BTreeMap<String, toml::Value> {
    let Some(root) = workspace_root_for_manifest(manifest) else {
        return BTreeMap::new();
    };
    let value = match read_manifest(&root.join("Cargo.toml"), blockers) {
        Some(value) => value,
        None => return BTreeMap::new(),
    };
    value
        .get("workspace")
        .and_then(|workspace| workspace.get("dependencies"))
        .and_then(toml::Value::as_table)
        .map(|table| {
            table
                .iter()
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default()
}

fn workspace_root_for_manifest(manifest: &Path) -> Option<PathBuf> {
    for ancestor in manifest.ancestors().skip(1) {
        let candidate = ancestor.join("Cargo.toml");
        let Ok(text) = crate::output_arg::read_text_bounded(&candidate, MAX_MANIFEST_BYTES, "")
        else {
            continue;
        };
        let Ok(value) = toml::from_str::<toml::Value>(&text) else {
            continue;
        };
        if value.get("workspace").is_some() {
            return Some(ancestor.to_path_buf());
        }
    }
    None
}

fn dependency_has_local_path(
    spec: &toml::Value,
    workspace_dependencies: &BTreeMap<String, toml::Value>,
    dependency: &str,
) -> Option<String> {
    if let Some(path) = spec.get("path").and_then(toml::Value::as_str) {
        return Some(path.to_string());
    }
    if dependency_uses_workspace(spec) {
        return workspace_dependencies
            .get(dependency)
            .and_then(|value| value.get("path"))
            .and_then(toml::Value::as_str)
            .map(str::to_string);
    }
    None
}

fn dependency_version(
    spec: &toml::Value,
    workspace_dependencies: &BTreeMap<String, toml::Value>,
    dependency: &str,
) -> Option<String> {
    spec.as_str()
        .map(str::to_string)
        .or_else(|| {
            spec.get("version")
                .and_then(toml::Value::as_str)
                .map(str::to_string)
        })
        .or_else(|| {
            dependency_uses_workspace(spec).then(|| {
                workspace_dependencies
                    .get(dependency)
                    .and_then(|value| {
                        value
                            .as_str()
                            .or_else(|| value.get("version").and_then(toml::Value::as_str))
                    })
                    .map(str::to_string)
            })?
        })
}

fn dependency_uses_workspace(spec: &toml::Value) -> bool {
    spec.get("workspace").and_then(toml::Value::as_bool) == Some(true)
}

fn package_version(package: &toml::value::Table) -> Option<&str> {
    package.get("version").and_then(toml::Value::as_str)
}

fn read_manifest(path: &Path, blockers: &mut Vec<String>) -> Option<toml::Value> {
    let text = match crate::output_arg::read_text_bounded(path, MAX_MANIFEST_BYTES, "") {
        Ok(text) => text,
        Err(error) => {
            blockers.push(format!(
                "failed to read manifest `{}`: {error}",
                path.display()
            ));
            return None;
        }
    };
    match toml::from_str::<toml::Value>(&text) {
        Ok(value) => Some(value),
        Err(error) => {
            blockers.push(format!(
                "failed to parse manifest `{}`: {error}",
                path.display()
            ));
            None
        }
    }
}

fn parse_output(args: &[String]) -> Result<PathBuf, String> {
    crate::output_arg::parse_output_arg(
        args,
        "package-readiness",
        "Writes pre-publish package-order evidence.",
        default_output,
    )
}

fn default_output() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|path| path.join("release/evidence/package/publish-readiness.json"))
        .unwrap_or_else(|| PathBuf::from("release/evidence/package/publish-readiness.json"))
}

#[cfg(test)]
#[path = "package_readiness/archive_contracts.rs"]
mod archive_contracts;
