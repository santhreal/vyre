//! Every test target admitted by `device-tests` compiles on any host.
//!
//! Test targets behind the `device-tests` feature are not executed in hosted CI
//! lanes because they require physical GPU hardware. When test code behind that
//! feature contains compile errors, hosted CI remains green and the defect is
//! only observed on a runner that owns a device.
//!
//! This gate derives the set of workspace test targets admitted by
//! `device-tests` from source and compiles each target with the feature enabled
//! using `cargo check`.
//!
//! # What it does not catch
//!
//! This gate proves that the test target compiles without errors. It does not
//! prove that the test passes on hardware, that the backend is available, or
//! that runtime GPU assertions hold.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use toml::Value;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::Tree;
use crate::gates::test_target_membership::{self, TestTarget};

/// The feature that admits a device-acquiring test.
const FEATURE: &str = "device-tests";

/// Directory containing CI workflows.
const WORKFLOWS: &str = ".github/workflows";

/// Compile every `device-tests`-admitted test target without running it.
pub struct DeviceTestCompilation;

impl crate::gate::GateBehavior for DeviceTestCompilation {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();

        let workflow_features = workflow_feature_pairings(&tree)?;
        let admitted = admitted_test_targets(&tree)?;

        report.cover_complete("workspace members declaring device-tests", admitted.len());
        let total_targets: usize = admitted.values().map(|targets| targets.len()).sum();
        report.note(format!(
            "{} package(s) declaring `{FEATURE}`, {} device-admitted test target(s)",
            admitted.len(),
            total_targets
        ));

        for (package_name, targets) in &admitted {
            if targets.is_empty() {
                continue;
            }
            let mut features = workflow_features
                .get(package_name)
                .cloned()
                .unwrap_or_default();
            features.insert(FEATURE.to_string());
            for target in targets {
                features.extend(target.required_features.iter().cloned());
            }

            let findings = check_package_targets(&ctx.root, package_name, targets, &features)?;
            for finding in findings {
                report.find(finding);
            }
        }

        Ok(report)
    }
}

/// Find all workspace packages declaring `device-tests` and their admitted test targets.
pub fn admitted_test_targets(tree: &Tree) -> Result<BTreeMap<String, Vec<TestTarget>>, GateError> {
    let mut admitted = BTreeMap::new();
    for member in tree.member_manifests()? {
        let Some(features) = member.manifest.get("features").and_then(Value::as_table) else {
            continue;
        };
        if !features.contains_key(FEATURE) {
            continue;
        }

        let ownership = test_target_membership::ownership(tree, &member);
        let mut member_targets = Vec::new();

        for target in ownership.targets {
            if is_target_admitted(tree, &target, &ownership.owners)? {
                member_targets.push(target);
            }
        }
        admitted.insert(member.name, member_targets);
    }
    Ok(admitted)
}

/// Whether one test target is admitted by `device-tests`.
fn is_target_admitted(
    tree: &Tree,
    target: &TestTarget,
    owners: &BTreeMap<String, Vec<String>>,
) -> Result<bool, GateError> {
    if target.required_features.contains(FEATURE) {
        return Ok(true);
    }
    if tree.has(&target.root) {
        let text = tree.read(Path::new(&target.root))?;
        if text_admits_device_tests(&text) {
            return Ok(true);
        }
    }
    for (file_path, target_names) in owners {
        if target_names.contains(&target.name) && tree.has(file_path) {
            let text = tree.read(Path::new(file_path))?;
            if text_admits_device_tests(&text) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Whether the source text contains an admission for `device-tests`.
fn text_admits_device_tests(text: &str) -> bool {
    text.contains("feature = \"device-tests\"")
        || text.contains("feature = \\\"device-tests\\\"")
        || text.contains("feature = \"device-tests\"")
}

/// Read workflow steps to find additional features paired with `device-tests` per package.
pub fn workflow_feature_pairings(
    tree: &Tree,
) -> Result<BTreeMap<String, BTreeSet<String>>, GateError> {
    let mut pairings: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for path in tree.paths() {
        if path.starts_with(WORKFLOWS) && path.extension().is_some_and(|e| e == "yml") {
            let text = tree.read(path)?;
            let collapsed = text
                .lines()
                .filter(|line| !line.trim_start().starts_with('#'))
                .flat_map(str::split_whitespace)
                .collect::<Vec<_>>()
                .join(" ");
            for segment in collapsed.split("./cargo_full").skip(1) {
                let command = segment.split("- name:").next().unwrap_or(segment);
                if !command.contains(FEATURE) {
                    continue;
                }
                let mut packages = BTreeSet::new();
                let mut features = BTreeSet::new();
                let mut tokens = command.split(' ');
                while let Some(token) = tokens.next() {
                    match token {
                        "-p" | "--package" => {
                            if let Some(pkg) = tokens.next() {
                                packages.insert(pkg.trim_matches(['\'', '"']).to_string());
                            }
                        }
                        "--features" => {
                            if let Some(fts) = tokens.next() {
                                for ft in fts.trim_matches(['\'', '"']).split(',') {
                                    if !ft.is_empty() {
                                        features.insert(ft.to_string());
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                for pkg in packages {
                    pairings
                        .entry(pkg)
                        .or_default()
                        .extend(features.iter().cloned());
                }
            }
        }
    }
    Ok(pairings)
}

/// One compiler diagnostic parsed from `--message-format=json`.
struct Diagnostic {
    target: Option<String>,
    file: Option<String>,
    line: Option<u32>,
    message: String,
}

/// Run cargo check for one package's admitted test targets and collect any compiler errors.
fn check_package_targets(
    root: &Path,
    package: &str,
    targets: &[TestTarget],
    features: &BTreeSet<String>,
) -> Result<Vec<Finding>, GateError> {
    let cargo = crate::cargo_runner::binary(root);
    let feature_list = features.iter().cloned().collect::<Vec<_>>().join(",");

    let mut cmd = Command::new(&cargo);
    cmd.arg("check")
        .arg("-p")
        .arg(package)
        .arg("--features")
        .arg(&feature_list)
        .arg("--message-format=json");

    for target in targets {
        cmd.arg("--test").arg(&target.name);
    }
    cmd.current_dir(root);

    let output = cmd.output().map_err(|error| {
        GateError::new(
            format!("cannot run `cargo check -p {package} --features {feature_list}`: {error}"),
            "restore the cargo_full wrapper at the workspace root",
        )
    })?;

    let target_names: BTreeSet<String> = targets.iter().map(|t| t.name.clone()).collect();
    let primary_target = targets
        .first()
        .map(|t| t.name.as_str())
        .unwrap_or("all_tests");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let diagnostics = parse_compiler_diagnostics(&stdout);

    let stderr = String::from_utf8_lossy(&output.stderr);
    if let Some(missing) = crate::cargo_runner::unmeasured(&stderr) {
        return Ok(vec![Finding::new(
            format!(
                "`cargo check -p {package}` measured nothing: the build named `{missing}`, which the build directory does not carry"
            ),
            "run the gate again against an intact build directory; a compile whose own inputs were deleted under it reports the state of the disk",
        )]);
    }

    if !output.status.success() && diagnostics.is_empty() {
        return Ok(vec![Finding::new(
            format!(
                "`cargo check -p {package} --features {feature_list}` exited {} and emitted no compiler diagnostic: {}",
                output.status.code().unwrap_or(-1),
                stderr.trim()
            ),
            format!("repair the build of test target `{primary_target}` in `{package}` or remove its `{FEATURE}` admission"),
        )]);
    }

    let mut findings = Vec::new();
    for diag in diagnostics {
        let tgt = match &diag.target {
            Some(t) if target_names.contains(t) => t.as_str(),
            _ => primary_target,
        };
        let msg = format!("{package} (test target `{tgt}`): {}", diag.message);
        let fix = format!(
            "repair test target `{tgt}` in `{package}` or remove its `{FEATURE}` admission"
        );

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
            (None, _) => Finding::new(msg, fix),
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
        let level = message.get("level").and_then(serde_json::Value::as_str);
        if level != Some("error") {
            continue;
        }
        let text = message
            .get("message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("the compiler reported an error with no message")
            .to_string();

        let target_name = value
            .get("target")
            .and_then(|t| t.get("name"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);

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
            target: target_name,
            file: primary
                .and_then(|span| span.get("file_name"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            line: primary
                .and_then(|span| span.get("line_start"))
                .and_then(serde_json::Value::as_u64)
                .and_then(|line| u32::try_from(line).ok()),
            message: text,
        });
    }
    diagnostics
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_tree(files: &[(&str, &str)]) -> (tempfile::TempDir, Tree) {
        let (dir, root) = crate::gates::fixture_checkout::checkout(files);
        let tree = Tree::open(&root).expect("open tree");
        (dir, tree)
    }

    /// WHY: a manifest requiring `device-tests` admits that target directly.
    #[test]
    fn required_features_admits_test_target() {
        let target = TestTarget {
            name: "all_tests_device_tests".to_string(),
            root: "conform/vyre-conform/tests/all_tests_device_tests.rs".to_string(),
            required_features: ["device-tests".to_string()].into_iter().collect(),
        };
        let owners = BTreeMap::new();
        let (_dir, tree) = fixture_tree(&[(
            "conform/vyre-conform/tests/all_tests_device_tests.rs",
            "fn dummy() {}",
        )]);
        assert!(is_target_admitted(&tree, &target, &owners).unwrap());
    }

    /// WHY: a test target with no required-features and no cfg is not admitted.
    #[test]
    fn unadmitted_test_target_is_not_admitted() {
        let target = TestTarget {
            name: "all_tests".to_string(),
            root: "vyre-foo/tests/all_tests.rs".to_string(),
            required_features: BTreeSet::new(),
        };
        let owners = BTreeMap::new();
        let (_dir, tree) = fixture_tree(&[("vyre-foo/tests/all_tests.rs", "fn dummy() {}")]);
        assert!(!is_target_admitted(&tree, &target, &owners).unwrap());
    }

    /// WHY: a test file containing an inner or outer `device-tests` cfg admits its harness target.
    #[test]
    fn file_cfg_admits_test_target() {
        let target = TestTarget {
            name: "all_tests".to_string(),
            root: "vyre-driver-wgpu/tests/all_tests.rs".to_string(),
            required_features: BTreeSet::new(),
        };
        let mut owners = BTreeMap::new();
        owners.insert(
            "vyre-driver-wgpu/tests/connected_graph.rs".to_string(),
            vec!["all_tests".to_string()],
        );
        let (_dir, tree) = fixture_tree(&[
            (
                "vyre-driver-wgpu/tests/all_tests.rs",
                "pub mod connected_graph;",
            ),
            (
                "vyre-driver-wgpu/tests/connected_graph.rs",
                "#![cfg(feature = \"device-tests\")]\nfn live() {}",
            ),
        ]);
        assert!(is_target_admitted(&tree, &target, &owners).unwrap());
    }

    /// WHY: workflow feature pairings extract paired features like `cuda` for `vyre-driver-cuda`.
    #[test]
    fn workflow_feature_pairings_extracts_cuda_pairing() {
        let workflow = r#"
name: GPU Parity
jobs:
  cuda:
    steps:
      - name: CUDA contracts
        run: ./cargo_full test --release -p vyre-driver-cuda --features device-tests,cuda -- --test-threads=1
      - name: WGPU contracts
        run: ./cargo_full test -p vyre-driver-wgpu --features device-tests -- --test-threads=1
"#;
        let (_dir, tree) = fixture_tree(&[(".github/workflows/gpu-parity.yml", workflow)]);
        let pairings = workflow_feature_pairings(&tree).unwrap();
        assert_eq!(
            pairings.get("vyre-driver-cuda").unwrap(),
            &["cuda".to_string(), "device-tests".to_string()]
                .into_iter()
                .collect::<BTreeSet<_>>()
        );
        assert_eq!(
            pairings.get("vyre-driver-wgpu").unwrap(),
            &["device-tests".to_string()]
                .into_iter()
                .collect::<BTreeSet<_>>()
        );
    }

    /// WHY: a compiler error diagnostic is converted into an actionable finding naming package and target.
    #[test]
    fn diagnostic_to_finding_formats_message_and_fix() {
        let diag = Diagnostic {
            target: Some("all_tests".to_string()),
            file: Some("vyre-driver-wgpu/tests/connected_graph.rs".to_string()),
            line: Some(42),
            message: "no method named `target_payload` found".to_string(),
        };
        let finding_msg = format!(
            "vyre-driver-wgpu (test target `all_tests`): {}",
            diag.message
        );
        let fix = "repair test target `all_tests` in `vyre-driver-wgpu` or remove its `device-tests` admission";
        let finding = Finding::at(
            Path::new("vyre-driver-wgpu/tests/connected_graph.rs"),
            42,
            finding_msg,
            fix,
        );
        assert_eq!(
            finding.message,
            "vyre-driver-wgpu (test target `all_tests`): no method named `target_payload` found"
        );
        assert_eq!(
            finding.fix,
            "repair test target `all_tests` in `vyre-driver-wgpu` or remove its `device-tests` admission"
        );
        assert_eq!(finding.line, Some(42));
    }

    /// WHY: a package declaring `device-tests` in `[features]` is discovered and its admitted targets enumerated.
    #[test]
    fn admitted_test_targets_discovers_declaring_members() {
        let manifest = r#"
[workspace]
members = ["crate-a", "crate-b"]

[workspace.dependencies]
"#;
        let crate_a_toml = r#"
[package]
name = "crate-a"
version = "0.1.0"
edition = "2021"

[features]
device-tests = []

[[test]]
name = "all_tests"
path = "tests/all_tests.rs"
"#;
        let crate_b_toml = r#"
[package]
name = "crate-b"
version = "0.1.0"
edition = "2021"

[features]
other-feature = []

[[test]]
name = "all_tests"
path = "tests/all_tests.rs"
"#;
        let (_dir, tree) = fixture_tree(&[
            ("Cargo.toml", manifest),
            ("crate-a/Cargo.toml", crate_a_toml),
            ("crate-a/tests/all_tests.rs", "pub mod device_case;"),
            (
                "crate-a/tests/device_case.rs",
                "#![cfg(feature = \"device-tests\")]\nfn live() {}",
            ),
            ("crate-b/Cargo.toml", crate_b_toml),
            ("crate-b/tests/all_tests.rs", "fn normal_test() {}"),
        ]);
        let admitted = admitted_test_targets(&tree).unwrap();
        assert!(admitted.contains_key("crate-a"));
        assert!(!admitted.contains_key("crate-b"));
        let a_targets = &admitted["crate-a"];
        assert_eq!(a_targets.len(), 1);
        assert_eq!(a_targets[0].name, "all_tests");
    }

    /// WHY: compiler JSON diagnostics parsing extracts error messages and ignores warnings and notes.
    #[test]
    fn compiler_json_diagnostics_extracts_errors_and_ignores_warnings() {
        let json_output = r#"
{"reason":"compiler-message","package_id":"vyre-foo 0.8.0","target":{"name":"all_tests","kind":["test"]},"message":{"level":"warning","message":"unused variable","spans":[]}}
{"reason":"compiler-message","package_id":"vyre-foo 0.8.0","target":{"name":"all_tests","kind":["test"]},"message":{"level":"error","message":"no method named `target_payload` found","spans":[{"file_name":"tests/foo.rs","line_start":12,"is_primary":true}]}}
{"reason":"build-finished","success":false}
"#;
        let diags = parse_compiler_diagnostics(json_output);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].target.as_deref(), Some("all_tests"));
        assert_eq!(diags[0].file.as_deref(), Some("tests/foo.rs"));
        assert_eq!(diags[0].line, Some(12));
        assert_eq!(diags[0].message, "no method named `target_payload` found");
    }
}
