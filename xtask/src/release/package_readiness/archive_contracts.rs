use super::{
    cargo_package_patch_args, inspect_package_file_list, package_path_is_forbidden, PublishStep,
};

fn step(package: &str, version: &'static str, manifest: &str) -> PublishStep {
    PublishStep {
        package: package.to_string(),
        version,
        manifest: manifest.to_string(),
    }
}

fn example_step() -> PublishStep {
    step("vyre-example", "0.7.0", "vyre-example/Cargo.toml")
}

/// A complete crates.io file list must retain metadata and Rust sources without archive blockers.
#[test]
fn complete_package_file_list_passes_archive_contract() {
    let check = inspect_package_file_list(
        &example_step(),
        "Cargo.toml\nCargo.toml.orig\nLICENSE-APACHE\nLICENSE-MIT\nREADME.md\nexamples/basic.rs\nsrc/lib.rs\n",
    );

    assert!(check.cargo_package_list_succeeded);
    assert_eq!(check.file_count, 7);
    assert_eq!(check.rust_source_count, 1);
    assert!(check.missing_required_files.is_empty());
    assert!(check.forbidden_files.is_empty());
    assert!(check.command_error.is_none());
    assert!(check.blockers.is_empty());
    assert!(check.file_list_digest.starts_with("blake3:"));
}

/// Package file-list normalization must make evidence independent of Cargo's output ordering and duplicate lines.
#[test]
fn package_file_list_digest_is_order_and_duplicate_stable() {
    let first = inspect_package_file_list(
        &example_step(),
        "Cargo.toml\nCargo.toml.orig\nLICENSE-APACHE\nLICENSE-MIT\nREADME.md\nexamples/basic.rs\nsrc/lib.rs\n",
    );
    let reordered = inspect_package_file_list(
        &example_step(),
        "src/lib.rs\nREADME.md\nCargo.toml\nexamples/basic.rs\nLICENSE-MIT\nCargo.toml.orig\nLICENSE-APACHE\nsrc/lib.rs\n",
    );

    assert_eq!(first.file_count, reordered.file_count);
    assert_eq!(first.file_list_digest, reordered.file_list_digest);
}

/// Agent instructions, secrets, build output, and escaping paths must never enter a crates.io archive.
#[test]
fn internal_and_unsafe_package_paths_are_rejected() {
    for path in [
        "tests/SKILL.md",
        "src/AGENTS.md",
        ".env",
        ".env.production",
        "credentials/token",
        "target/release/lib.rlib",
        "benches/baselines/scan/report/index.html",
        "tests/corpus/generated/hostile.bin",
        "../outside",
        "/absolute/path",
    ] {
        assert!(
            package_path_is_forbidden(path),
            "Fix: `{path}` must remain outside publishable archives"
        );
    }
    for path in [
        ".cargo_vcs_info.json",
        "tests/behavior.rs",
        "src/targeting.rs",
        "examples/release_surface.rs",
        "benches/baselines.rs",
        "tests/corpus_contract.rs",
    ] {
        assert!(
            !package_path_is_forbidden(path),
            "Fix: valid package path `{path}` must not be rejected"
        );
    }
}

/// Missing license/readme metadata, source, and internal instructions must report every independent blocker together.
#[test]
fn incomplete_package_file_list_reports_all_archive_gaps() {
    let check = inspect_package_file_list(&example_step(), "Cargo.toml\ntests/SKILL.md\n");

    assert_eq!(
        check.missing_required_files,
        vec![
            "Cargo.toml.orig",
            "README.md",
            "LICENSE-APACHE",
            "LICENSE-MIT"
        ]
    );
    assert_eq!(check.forbidden_files, vec!["tests/SKILL.md"]);
    assert_eq!(check.rust_source_count, 0);
    assert_eq!(check.blockers.len(), 3);
    assert!(check
        .blockers
        .iter()
        .any(|blocker| blocker.contains("missing required package files")));
    assert!(check
        .blockers
        .iter()
        .any(|blocker| blocker.contains("no Rust source")));
    assert!(check
        .blockers
        .iter()
        .any(|blocker| blocker.contains("tests/SKILL.md")));
}

fn complete_package_evidence() -> serde_json::Value {
    serde_json::json!({
        "publish_order": [
            {"package": "vyre-example", "manifest": "vyre-example/Cargo.toml"}
        ],
        "package_content_checks": [
            {
                "package": "vyre-example",
                "manifest": "vyre-example/Cargo.toml",
                "cargo_package_list_succeeded": true,
                "file_count": 7,
                "file_list_digest": format!("blake3:{}", "a".repeat(64)),
                "rust_source_count": 1,
                "missing_required_files": [],
                "forbidden_files": [],
                "command_error": null,
                "blockers": []
            }
        ]
    })
}

/// Final release semantics must accept one complete content proof for every publish-order package.
#[test]
fn complete_package_content_evidence_passes_release_semantics() {
    assert!(super::package_content_evidence_issues(&complete_package_evidence()).is_empty());
}

/// The release gate must reject every failed package-content invariant rather than trusting an empty top-level blocker list.
#[test]
fn malformed_package_content_evidence_reports_every_failed_invariant() {
    let mut evidence = complete_package_evidence();
    let check = evidence["package_content_checks"][0]
        .as_object_mut()
        .expect("package content fixture is an object");
    check.insert(
        "cargo_package_list_succeeded".to_string(),
        serde_json::json!(false),
    );
    check.insert("file_count".to_string(), serde_json::json!(0));
    check.insert("rust_source_count".to_string(), serde_json::json!(0));
    check.insert(
        "file_list_digest".to_string(),
        serde_json::json!("sha256:not-the-contract"),
    );
    check.insert(
        "missing_required_files".to_string(),
        serde_json::json!(["README.md"]),
    );
    check.insert(
        "forbidden_files".to_string(),
        serde_json::json!(["tests/SKILL.md"]),
    );
    check.insert("command_error".to_string(), serde_json::json!("failed"));
    check.insert("blockers".to_string(), serde_json::json!(["broken"]));

    let issues = super::package_content_evidence_issues(&evidence);
    for expected in [
        "did not pass `cargo package --list`",
        "non-positive `file_count`",
        "non-positive `rust_source_count`",
        "invalid file_list_digest",
        "`missing_required_files` must be an empty array",
        "`forbidden_files` must be an empty array",
        "`blockers` must be an empty array",
        "command_error must be null",
    ] {
        assert!(
            issues.iter().any(|issue| issue.contains(expected)),
            "Fix: malformed package evidence must report `{expected}`; issues={issues:?}"
        );
    }
    assert_eq!(issues.len(), 8);
}

/// Missing, duplicate, and extra content rows must not satisfy one-to-one publish-order coverage.
#[test]
fn package_content_evidence_requires_exact_publish_order_cardinality() {
    let evidence = serde_json::json!({
        "publish_order": [
            {"package": "vyre-a", "manifest": "a/Cargo.toml"},
            {"package": "vyre-b", "manifest": "b/Cargo.toml"}
        ],
        "package_content_checks": [
            {
                "package": "vyre-a",
                "manifest": "a/Cargo.toml",
                "cargo_package_list_succeeded": true,
                "file_count": 7,
                "file_list_digest": format!("blake3:{}", "b".repeat(64)),
                "rust_source_count": 1,
                "missing_required_files": [],
                "forbidden_files": [],
                "command_error": null,
                "blockers": []
            },
            {
                "package": "vyre-extra",
                "manifest": "extra/Cargo.toml",
                "cargo_package_list_succeeded": true,
                "file_count": 7,
                "file_list_digest": format!("blake3:{}", "c".repeat(64)),
                "rust_source_count": 1,
                "missing_required_files": [],
                "forbidden_files": [],
                "command_error": null,
                "blockers": []
            }
        ]
    });

    let issues = super::package_content_evidence_issues(&evidence);
    assert!(issues
        .iter()
        .any(|issue| issue.contains("missing publishable package `vyre-b`")));
    assert!(issues
        .iter()
        .any(|issue| issue.contains("non-publish-order package `vyre-extra`")));
    assert_eq!(issues.len(), 2);
}

/// Exact-train path dependencies still need registry patches because Cargo removes their paths while packaging.
///
/// A dependency from a different release train must not be redirected to the
/// current source tree because that would validate different code than the
/// packaged manifest requests.
#[test]
fn archive_check_patches_exact_local_dependencies_but_not_mismatched_versions() {
    let temp = tempfile::tempdir().expect("Fix: create package patch fixture directory");
    std::fs::create_dir_all(temp.path().join("consumer"))
        .expect("Fix: create consumer fixture directory");
    std::fs::write(
        temp.path().join("consumer/Cargo.toml"),
        r#"[package]
name = "consumer"
version = "0.1.0"

[dependencies]
vyre.workspace = true
vyre-driver-wgpu = "0.7.1"

[workspace]

[workspace.dependencies]
vyre = { version = "0.7.2", path = "../vyre" }
"#,
    )
    .expect("Fix: write consumer package manifest");

    let order = [
        step("vyre", "0.7.2", "vyre/Cargo.toml"),
        step("vyre-driver-wgpu", "0.7.2", "vyre-driver-wgpu/Cargo.toml"),
        step("consumer", "0.1.0", "consumer/Cargo.toml"),
    ];

    let actual = cargo_package_patch_args(temp.path(), &order[2], &order)
        .expect("Fix: inspect exact and mismatched dependencies")
        .into_iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    assert_eq!(
        actual,
        vec![
            "--config",
            &format!(
                "patch.crates-io.vyre.path={:?}",
                temp.path().join("vyre").to_string_lossy()
            ),
        ],
        "Fix: package validation must use exact local release sources only"
    );
}

/// A path-only dependency has no crates.io contract and must not be converted into a release patch.
///
/// This preserves Cargo's path-only development dependency semantics instead
/// of inventing a registry dependency that the package never declared.
#[test]
fn archive_check_does_not_patch_path_only_dependencies() {
    let temp = tempfile::tempdir().expect("Fix: create package patch fixture directory");
    std::fs::create_dir_all(temp.path().join("consumer"))
        .expect("Fix: create consumer fixture directory");
    std::fs::write(
        temp.path().join("consumer/Cargo.toml"),
        r#"[package]
name = "consumer"
version = "0.1.0"

[dev-dependencies]
vyre = { path = "../vyre" }

[workspace]
"#,
    )
    .expect("Fix: write path-only dependency fixture");
    let order = [
        step("vyre", "0.7.2", "vyre/Cargo.toml"),
        step("consumer", "0.1.0", "consumer/Cargo.toml"),
    ];

    assert!(
        cargo_package_patch_args(temp.path(), &order[1], &order)
            .expect("Fix: inspect path-only dependency")
            .is_empty(),
        "Fix: path-only dependencies must remain outside registry patching"
    );
}

/// A malformed cross-repository manifest must fail before Cargo can produce misleading archive evidence.
#[test]
fn archive_check_rejects_malformed_manifest_before_launch() {
    let temp = tempfile::tempdir().expect("Fix: create package patch fixture directory");
    std::fs::create_dir_all(temp.path().join("consumer"))
        .expect("Fix: create consumer fixture directory");
    std::fs::write(
        temp.path().join("consumer/Cargo.toml"),
        "[package\nname = \"consumer\"\n",
    )
    .expect("Fix: write malformed consumer manifest");
    let order = [step("consumer", "0.1.0", "consumer/Cargo.toml")];

    let error = cargo_package_patch_args(temp.path(), &order[0], &order)
        .expect_err("Fix: malformed package manifests must fail closed");
    assert!(
        error.contains("failed to parse") && error.contains("consumer/Cargo.toml"),
        "Fix: parse failure must name the malformed manifest; error={error}"
    );
}

/// The publish order must come out of the manifests, in dependency order.
///
/// WHY THIS EXISTS. The order used to be a hardcoded `vec![]` of 26 steps. Moving
/// library code into `vyre-libs` gave it new consumers, and because the table
/// still held it near the end, the recorded evidence certified an order that
/// publishes consumers against a version crates.io does not have yet, with
/// `blockers: []` throughout. A member table cannot see a new edge. This fixture
/// is the previous defect in miniature: `consumer` sorts before `lib`
/// alphabetically, and only reading the dependency inverts them. Against the old
/// hardcoded order this assertion could not even be written, because the order
/// did not depend on the tree.
#[test]
fn publish_order_follows_manifest_dependencies_not_crate_names() {
    let fixture = OrderFixture::new("derived-order");
    fixture.crate_manifest("lib", &[]);
    fixture.crate_manifest("consumer", &["lib"]);
    fixture.metadata(&["consumer", "lib"]);

    let (order, domain) = fixture.publish_order();

    assert_eq!(
        order
            .iter()
            .map(|step| step.package.as_str())
            .collect::<Vec<_>>(),
        vec!["lib", "consumer"],
        "Fix: a dependency must publish before its consumer"
    );
    assert_eq!(
        domain,
        ["consumer".to_string(), "lib".to_string()]
            .into_iter()
            .collect(),
        "Fix: the sort domain must be every publishable crate the metadata matrix names"
    );
    assert!(
        fixture.blockers().is_empty(),
        "Fix: an orderable set must produce no blockers; blockers={:?}",
        fixture.blockers()
    );
}

/// A dependency cycle among publishable crates must be reported, never ordered around.
///
/// Fail-closed is the whole point of deriving: picking some order for a set that
/// has none would publish a broken train and record it as ready. Both members
/// must still reach the artifact so a reader can see what the cycle was.
#[test]
fn publish_order_reports_a_cycle_instead_of_choosing_an_order() {
    let fixture = OrderFixture::new("cyclic-order");
    fixture.crate_manifest("left", &["right"]);
    fixture.crate_manifest("right", &["left"]);
    fixture.metadata(&["left", "right"]);

    let (order, _) = fixture.publish_order();
    let blockers = fixture.blockers();

    assert_eq!(
        order.len(),
        2,
        "Fix: a cycle must still record both members so the artifact names them"
    );
    for package in ["left", "right"] {
        assert!(
            blockers.iter().any(|blocker| {
                blocker.contains(&format!("publish order cannot be derived for `{package}`"))
                    && blocker.contains("dependency cycle")
            }),
            "Fix: the cycle must be reported for `{package}`; blockers={blockers:?}"
        );
    }
}

/// A crate the metadata matrix calls publishable but disk does not carry must block.
///
/// Dropping it instead would take it out of the order and out of every check
/// keyed on the order, so a deleted or renamed crate would read as a shorter,
/// still-green release. This is the exact rot that left nine evidence citations
/// pointing at one crate's `Cargo.toml` after it had been folded into another.
#[test]
fn publish_order_blocks_a_publishable_crate_whose_manifest_is_absent() {
    let fixture = OrderFixture::new("absent-manifest");
    fixture.crate_manifest("present", &[]);
    fixture.metadata(&["absent", "present"]);

    let (order, _) = fixture.publish_order();
    let blockers = fixture.blockers();

    assert_eq!(
        order
            .iter()
            .map(|step| step.package.as_str())
            .collect::<Vec<_>>(),
        vec!["present"],
        "Fix: a crate with no manifest cannot be ordered"
    );
    assert!(
        blockers.iter().any(|blocker| {
            blocker.contains("publishable crate `absent`")
                && blocker.contains("absent/Cargo.toml")
                && blocker.contains("not on disk")
        }),
        "Fix: the missing manifest must be named with its path; blockers={blockers:?}"
    );
}

/// A path-only dev-dependency must not constrain publish order.
///
/// Cargo strips a dev-dependency carrying no version from the published
/// manifest, so it never reaches a downstream consumer and cannot fail a
/// publish. Several in this workspace are deliberately path-only to break a
/// cycle, so treating one as an edge would report a cycle that does not exist.
#[test]
fn publish_order_ignores_path_only_dev_dependencies() {
    let fixture = OrderFixture::new("dev-dependency-order");
    fixture.write(
        "left/Cargo.toml",
        "[package]\nname = \"left\"\nversion = \"0.7.0\"\n\n[dev-dependencies]\nright = { path = \"../right\" }\n",
    );
    fixture.write(
        "right/Cargo.toml",
        "[package]\nname = \"right\"\nversion = \"0.7.0\"\n\n[dev-dependencies]\nleft = { path = \"../left\" }\n",
    );
    fixture.metadata(&["left", "right"]);

    let (order, _) = fixture.publish_order();

    assert_eq!(
        order
            .iter()
            .map(|step| step.package.as_str())
            .collect::<Vec<_>>(),
        vec!["left", "right"],
        "Fix: path-only dev-dependencies are not publish-order edges"
    );
    assert!(
        fixture.blockers().is_empty(),
        "Fix: path-only dev-dependencies must not report a cycle; blockers={:?}",
        fixture.blockers()
    );
}

/// A checkout carrying a metadata matrix, used to drive the derivation directly.
struct OrderFixture {
    dir: tempfile::TempDir,
    blockers: std::cell::RefCell<Vec<String>>,
}

impl OrderFixture {
    fn new(name: &str) -> Self {
        let dir = tempfile::Builder::new()
            .prefix(name)
            .tempdir()
            .expect("Fix: create publish order fixture directory");
        Self {
            dir,
            blockers: std::cell::RefCell::new(Vec::new()),
        }
    }

    fn write(&self, relative: &str, contents: &str) {
        let path = self.dir.path().join(relative);
        std::fs::create_dir_all(path.parent().expect("Fix: fixture path needs a parent"))
            .expect("Fix: create fixture parent directory");
        std::fs::write(&path, contents).expect("Fix: write fixture file");
    }

    /// One crate manifest whose in-workspace dependencies carry both a path and a version.
    fn crate_manifest(&self, package: &str, dependencies: &[&str]) {
        let mut manifest = format!("[package]\nname = \"{package}\"\nversion = \"0.7.0\"\n");
        if !dependencies.is_empty() {
            manifest.push_str("\n[dependencies]\n");
            for dependency in dependencies {
                manifest.push_str(&format!(
                    "{dependency} = {{ version = \"0.7.0\", path = \"../{dependency}\" }}\n"
                ));
            }
        }
        self.write(&format!("{package}/Cargo.toml"), &manifest);
    }

    fn metadata(&self, packages: &[&str]) {
        let entries = packages
            .iter()
            .map(|package| {
                serde_json::json!({
                    "name": package,
                    "manifest": format!("{package}/Cargo.toml"),
                    "release_kind": "publishable-crate",
                })
            })
            .collect::<Vec<_>>();
        self.write(
            "release/evidence/metadata/metadata-matrix.json",
            &serde_json::json!({ "packages": entries }).to_string(),
        );
    }

    fn publish_order(&self) -> (Vec<PublishStep>, std::collections::BTreeSet<String>) {
        let metadata_path = self
            .dir
            .path()
            .join("release/evidence/metadata/metadata-matrix.json");
        super::publish_order(
            self.dir.path(),
            &metadata_path,
            &mut self.blockers.borrow_mut(),
        )
    }

    fn blockers(&self) -> Vec<String> {
        self.blockers.borrow().clone()
    }
}

/// The authoritative gate descriptor must declare exactly the canonical package-readiness artifact.
#[test]
fn authoritative_descriptor_declares_exact_package_readiness_artifacts() {
    let descriptor = crate::gate_metadata::descriptor_by_name("package-readiness");
    let mut expected: Vec<&str> = vec![super::ARTIFACT];
    expected.sort_unstable();
    let mut actual: Vec<&str> = descriptor.artifacts.to_vec();
    actual.sort_unstable();
    assert_eq!(
        actual, expected,
        "package-readiness gate descriptor must declare exactly the canonical publish readiness evidence artifact"
    );
}
