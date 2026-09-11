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
use std::path::Path;

use toml::Value;

use crate::cargo_runner::Diagnostic;
use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::Tree;
use crate::gates::test_target_membership::{self, TestTarget};
use crate::gates::workflow_commands;

/// The feature that admits a device-acquiring test.
const FEATURE: &str = "device-tests";

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
    for command in workflow_commands::enabling_in_tree(tree, FEATURE)? {
        for package in command.packages {
            pairings
                .entry(package)
                .or_default()
                .extend(command.features.iter().cloned());
        }
    }
    Ok(pairings)
}

/// Run cargo check for one package's admitted test targets and collect any compiler errors.
fn check_package_targets(
    root: &Path,
    package: &str,
    targets: &[TestTarget],
    features: &BTreeSet<String>,
) -> Result<Vec<Finding>, GateError> {
    let feature_list = features.iter().cloned().collect::<Vec<_>>().join(",");
    let mut arguments = vec!["check", "-p", package, "--features", &feature_list];
    for target in targets {
        arguments.push("--test");
        arguments.push(&target.name);
    }
    let run = crate::cargo_runner::diagnostics(root, &arguments, false)?;

    if let Some(missing) = run.unmeasured {
        return Ok(vec![Finding::new(
            format!(
                "`cargo check -p {package}` measured nothing: the build named `{missing}`, which the build directory does not carry"
            ),
            "run the gate again against an intact build directory; a compile whose own inputs were deleted under it reports the state of the disk",
        )]);
    }

    let target_names: BTreeSet<&str> = targets.iter().map(|target| target.name.as_str()).collect();
    let primary_target = targets
        .first()
        .map(|target| target.name.as_str())
        .unwrap_or("all_tests");

    if run.failed_silently() {
        return Ok(vec![Finding::new(
            format!(
                "`cargo check -p {package} --features {feature_list}` exited {} and emitted no compiler diagnostic: {}",
                run.code(),
                run.stderr.trim()
            ),
            format!("repair the build of test target `{primary_target}` in `{package}` or remove its `{FEATURE}` admission"),
        )]);
    }

    let findings = run
        .found
        .iter()
        .map(|diagnostic| attribute(root, package, diagnostic, &target_names, primary_target))
        .collect();

    Ok(findings)
}

/// Attribute one diagnostic to the test target whose build it explains.
///
/// Cargo names the target it was building, and that name is trusted only when
/// this invocation asked for it. Anything else is attributed to the first
/// target requested: a diagnostic raised while building a dependency of the
/// test target is still the reason that target does not build, and dropping it
/// would let the gate report a package that does not compile as clean.
fn attribute(
    root: &Path,
    package: &str,
    diagnostic: &Diagnostic,
    target_names: &BTreeSet<&str>,
    primary_target: &str,
) -> Finding {
    let target = match diagnostic.target.as_deref() {
        Some(named) if target_names.contains(named) => named,
        _ => primary_target,
    };
    let message = format!("{package} (test target `{target}`): {}", diagnostic.message);
    let fix =
        format!("repair test target `{target}` in `{package}` or remove its `{FEATURE}` admission");
    diagnostic
        .place(root, &message, &fix)
        .unwrap_or_else(|| Finding::new(message.clone(), fix.clone()))
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

    /// WHY: attribution is what makes a finding actionable, and it is the one
    /// decision this gate makes about a diagnostic. A diagnostic raised while
    /// building something other than a requested test target still explains why
    /// that target does not build, so it is attributed to the first target
    /// asked for rather than dropped or blamed on a target nobody requested.
    #[test]
    fn a_diagnostic_naming_an_unrequested_target_is_attributed_to_the_requested_one() {
        let requested: BTreeSet<&str> = ["all_tests"].into_iter().collect();
        let root = Path::new("/checkout");

        let named = Diagnostic {
            target: Some("all_tests".to_string()),
            file: Some("/checkout/vyre-driver-wgpu/tests/connected_graph.rs".to_string()),
            line: Some(42),
            message: "no method named `target_payload` found".to_string(),
        };
        let finding = attribute(root, "vyre-driver-wgpu", &named, &requested, "all_tests");
        assert_eq!(
            finding.message,
            "vyre-driver-wgpu (test target `all_tests`): no method named `target_payload` found"
        );
        assert_eq!(
            finding.fix,
            "repair test target `all_tests` in `vyre-driver-wgpu` or remove its `device-tests` admission"
        );
        assert_eq!(finding.line, Some(42));
        assert_eq!(
            finding.file.as_deref(),
            Some(Path::new("vyre-driver-wgpu/tests/connected_graph.rs")),
            "an absolute compiler path is stated relative to the checkout"
        );

        let elsewhere = Diagnostic {
            target: Some("build-script-build".to_string()),
            file: None,
            line: None,
            message: "linker `cc` not found".to_string(),
        };
        let finding = attribute(
            root,
            "vyre-driver-wgpu",
            &elsewhere,
            &requested,
            "all_tests",
        );
        assert_eq!(
            finding.message, "vyre-driver-wgpu (test target `all_tests`): linker `cc` not found",
            "a diagnostic from outside the requested targets still explains their build"
        );
        assert!(
            finding.file.is_none(),
            "a diagnostic with no span states no location"
        );
    }
}
