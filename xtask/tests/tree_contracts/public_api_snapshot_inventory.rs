//! Publishable-crate and public-API snapshot set contracts.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::process::Command;

use xtask::gate::{GateBehavior, GateCtx};
use xtask::gates::public_api::{roster, PublicApiSnapshot};
use xtask::gates::scan::Tree;

use super::workspace_sources::workspace_root;

/// Turn a directory into something `Tree::open` can list.
fn git_init(root: &Path) {
    let status = Command::new("git")
        .args(["init", "-q", "."])
        .current_dir(root)
        .status()
        .expect("Fix: git must be available to build a fixture checkout");
    assert!(
        status.success(),
        "Fix: git init must succeed in the fixture"
    );
}

/// The roster the gate is taken over, as `(directory, package)` pairs.
fn inventory(root: &Path) -> BTreeSet<(String, String)> {
    let tree = Tree::open(root).expect("Fix: the fixture must be a git checkout");
    roster(&tree)
        .expect("Fix: the workspace must publish at least one package")
        .into_iter()
        .map(|row| (row.directory, row.package))
        .collect()
}

/// The same set derived independently, straight from the manifests.
///
/// Two derivations of one set is the point: the gate's own walk resolves member
/// globs and reads `publish` through one code path, and this reads the raw TOML
/// through another. A single derivation compared against itself would prove only
/// that it equals itself, which is how a required list nothing could satisfy
/// survived here once already.
fn manifest_inventory(root: &Path) -> BTreeSet<(String, String)> {
    let workspace: toml::Value = toml::from_str(
        &fs::read_to_string(root.join("Cargo.toml"))
            .expect("Fix: workspace manifest must be readable"),
    )
    .expect("Fix: workspace manifest must parse");
    workspace["workspace"]["members"]
        .as_array()
        .expect("Fix: workspace.members must be an array")
        .iter()
        .filter_map(|member| {
            let member = member
                .as_str()
                .expect("Fix: workspace member entries must be strings");
            let manifest: toml::Value = toml::from_str(
                &fs::read_to_string(root.join(member).join("Cargo.toml"))
                    .expect("Fix: member manifest must be readable"),
            )
            .expect("Fix: member manifest must parse");
            let package = &manifest["package"];
            let publish = package.get("publish");
            let private = publish.is_some_and(|value| {
                value.as_bool() == Some(false)
                    || value
                        .as_array()
                        .is_some_and(|registries| registries.is_empty())
            });
            (!private).then(|| {
                (
                    member.to_string(),
                    package["name"]
                        .as_str()
                        .expect("Fix: package.name must be a string")
                        .to_string(),
                )
            })
        })
        .collect()
}

/// The gate's roster must exactly match publishability in every workspace manifest.
///
/// This prevents a newly publishable crate from bypassing the stability gate and
/// prevents private tooling from accidentally acquiring a public API promise.
#[test]
fn inventory_matches_publishable_workspace_manifests() {
    let root = workspace_root();
    assert_eq!(inventory(&root), manifest_inventory(&root));
}

/// Snapshot filenames must be identical to the publishable package-name set.
///
/// Missing files leave a released crate unprotected. Extra files preserve a
/// stability contract for a package that no longer exists or no longer publishes.
#[test]
fn snapshot_directory_matches_publishable_package_set() {
    let root = workspace_root();
    let expected: BTreeSet<String> = manifest_inventory(&root)
        .into_iter()
        .map(|(_, package)| package)
        .collect();
    let actual: BTreeSet<String> = fs::read_dir(root.join("docs/public-api"))
        .expect("Fix: docs/public-api must be readable")
        .map(|entry| {
            entry
                .expect("Fix: snapshot directory entry must be readable")
                .path()
        })
        .filter(|path| path.extension().is_some_and(|extension| extension == "txt"))
        .map(|path| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .expect("Fix: snapshot filename must be UTF-8")
                .to_string()
        })
        .collect();

    assert_eq!(actual, expected);
}

/// CUDA must be covered while private benchmark and conformance crates stay excluded.
///
/// These are the concrete positive and negative cases that exposed the old
/// hand-maintained six-crate list as incomplete in both directions.
#[test]
fn inventory_includes_cuda_and_excludes_private_tooling() {
    let inventory = inventory(&workspace_root());
    assert!(inventory.contains(&(
        "vyre-driver-cuda".to_string(),
        "vyre-driver-cuda".to_string()
    )));
    for private in ["xtask", "vyre-bench", "conform/vyre-conform"] {
        assert!(
            inventory.iter().all(|(member, _)| member != private),
            "private workspace member {private} must not receive a public API snapshot"
        );
    }
}

/// Both Cargo spellings for a non-publishable package must be excluded.
///
/// Cargo accepts `publish = false` and `publish = []`. Treating only one form
/// as private would make the inventory depend on stylistic manifest spelling.
#[test]
fn inventory_excludes_false_and_empty_registry_publish_values() {
    let temp = tempfile::tempdir().expect("Fix: temporary workspace must be creatable");
    fs::write(
        temp.path().join("Cargo.toml"),
        "[workspace]\nmembers = [\"public\", \"private_bool\", \"private_list\"]\n",
    )
    .expect("Fix: fixture workspace manifest must be writable");
    for (member, publish) in [
        ("public", ""),
        ("private_bool", "publish = false\n"),
        ("private_list", "publish = []\n"),
    ] {
        let directory = temp.path().join(member);
        fs::create_dir(&directory).expect("Fix: fixture member directory must be writable");
        fs::write(
            directory.join("Cargo.toml"),
            format!("[package]\nname = \"{member}\"\nversion = \"0.1.0\"\n{publish}"),
        )
        .expect("Fix: fixture member manifest must be writable");
    }
    git_init(temp.path());

    assert_eq!(
        inventory(temp.path()),
        BTreeSet::from([("public".to_string(), "public".to_string())])
    );
}

/// Snapshot extraction must model the externally reachable Rust API, including
/// surface that only exists behind a feature.
///
/// A source-line scan loses reexports and incorrectly includes `pub` items
/// nested beneath private modules. The feature-gated module is the case that was
/// missing: an extraction over the default feature set promises stability for
/// the default feature set only, and a whole gated surface sat outside the file
/// that claims to pin the public API.
#[test]
fn snapshot_includes_modules_reexports_and_feature_gated_surface() {
    let temp = tempfile::tempdir().expect("Fix: temporary workspace must be creatable");
    let root = temp.path();
    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"fixture\"]\nresolver = \"2\"\n",
    )
    .expect("Fix: fixture workspace manifest must be writable");

    let crate_dir = root.join("fixture");
    fs::create_dir_all(crate_dir.join("src"))
        .expect("Fix: fixture crate source directory must be creatable");
    fs::write(
        crate_dir.join("Cargo.toml"),
        "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[features]\ndefault = []\ngated = []\n",
    )
    .expect("Fix: fixture crate manifest must be writable");
    fs::write(
        crate_dir.join("src/lib.rs"),
        "mod private_module;\npub mod public_module;\n#[cfg(feature = \"gated\")]\npub mod gated_module;\npub use public_module::PublicType;\nfn private_function() {}\n",
    )
    .expect("Fix: fixture crate root must be writable");
    fs::write(
        crate_dir.join("src/private_module.rs"),
        "pub struct HiddenType;\n",
    )
    .expect("Fix: fixture private module must be writable");
    fs::write(
        crate_dir.join("src/public_module.rs"),
        "pub struct PublicType;\n",
    )
    .expect("Fix: fixture public module must be writable");
    fs::write(
        crate_dir.join("src/gated_module.rs"),
        "pub struct GatedType;\n",
    )
    .expect("Fix: fixture gated module must be writable");
    fs::copy(workspace_root().join("cargo_full"), root.join("cargo_full"))
        .expect("Fix: the bounded cargo wrapper must be copyable into the fixture");
    git_init(root);

    let report = PublicApiSnapshot
        .run(&GateCtx::new(
            root.to_path_buf(),
            vec![
                "--write".to_string(),
                "--crate".to_string(),
                "fixture".to_string(),
            ],
        ))
        .expect("Fix: the snapshot gate must be able to extract the fixture surface");
    assert_eq!(report.count(), 0, "{:?}", report.findings);
    let snapshot = fs::read_to_string(root.join("docs/public-api/fixture.txt"))
        .expect("Fix: fixture public API snapshot must be readable");

    for public_item in [
        "pub mod fixture::public_module",
        "pub struct fixture::PublicType",
        "pub struct fixture::public_module::PublicType",
        "pub mod fixture::gated_module",
        "pub struct fixture::gated_module::GatedType",
    ] {
        assert!(
            snapshot.lines().any(|line| line == public_item),
            "snapshot must include `{public_item}`:\n{snapshot}"
        );
    }
    for private_item in ["private_module", "HiddenType", "private_function"] {
        assert!(
            !snapshot.contains(private_item),
            "snapshot must exclude private item `{private_item}`:\n{snapshot}"
        );
    }
}

/// A refreshed snapshot must be byte-stable on a second read.
///
/// WHY: the snapshot's line order used to come from `sort` under the caller's
/// locale, so the committed file was a function of the environment rather than
/// of the tree, and two hosts disagreed about a surface neither had changed.
#[test]
fn a_refreshed_snapshot_verifies_clean_against_the_same_tree() {
    let temp = tempfile::tempdir().expect("Fix: temporary workspace must be creatable");
    let root = temp.path();
    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"fixture\"]\nresolver = \"2\"\n",
    )
    .expect("Fix: fixture workspace manifest must be writable");
    let crate_dir = root.join("fixture");
    fs::create_dir_all(crate_dir.join("src"))
        .expect("Fix: fixture crate source directory must be creatable");
    fs::write(
        crate_dir.join("Cargo.toml"),
        "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("Fix: fixture crate manifest must be writable");
    fs::write(
        crate_dir.join("src/lib.rs"),
        "pub struct Zed;\npub struct Alpha;\npub fn middle() {}\n",
    )
    .expect("Fix: fixture crate root must be writable");
    fs::copy(workspace_root().join("cargo_full"), root.join("cargo_full"))
        .expect("Fix: the bounded cargo wrapper must be copyable into the fixture");
    git_init(root);

    let write = GateCtx::new(root.to_path_buf(), vec!["--write".to_string()]);
    PublicApiSnapshot
        .run(&write)
        .expect("Fix: the snapshot gate must be able to write the fixture surface");
    let installed = fs::read_to_string(root.join("docs/public-api/fixture.txt"))
        .expect("Fix: fixture public API snapshot must be readable");
    let mut sorted: Vec<&str> = installed.lines().collect();
    sorted.sort_unstable();
    assert_eq!(
        installed.lines().collect::<Vec<&str>>(),
        sorted,
        "the snapshot must be in byte order, not the caller's collation order"
    );

    let verify = PublicApiSnapshot
        .run(&GateCtx::new(root.to_path_buf(), Vec::new()))
        .expect("Fix: the snapshot gate must be able to verify the fixture surface");
    assert_eq!(verify.count(), 0, "{:?}", verify.findings);
}

/// A stale snapshot missing newly removed items or missing newly added items must fail the gate.
///
/// WHY: When a public function or type is deleted from a crate without refreshing the snapshot,
/// the committed snapshot remains stale. The non-mutating gate run must catch this drift and
/// report the exact added and removed items in its finding message.
#[test]
fn stale_public_api_snapshot_is_reported_as_drift_finding() {
    let temp = tempfile::tempdir().expect("Fix: temporary workspace must be creatable");
    let root = temp.path();
    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"fixture\"]\nresolver = \"2\"\n",
    )
    .expect("Fix: fixture workspace manifest must be writable");
    let crate_dir = root.join("fixture");
    fs::create_dir_all(crate_dir.join("src"))
        .expect("Fix: fixture crate source directory must be creatable");
    fs::write(
        crate_dir.join("Cargo.toml"),
        "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("Fix: fixture crate manifest must be writable");
    fs::write(
        crate_dir.join("src/lib.rs"),
        "pub struct Alpha;\npub fn removed_item() {}\n",
    )
    .expect("Fix: fixture crate root must be writable");
    fs::copy(workspace_root().join("cargo_full"), root.join("cargo_full"))
        .expect("Fix: the bounded cargo wrapper must be copyable into the fixture");
    git_init(root);

    let write = GateCtx::new(root.to_path_buf(), vec!["--write".to_string()]);
    PublicApiSnapshot
        .run(&write)
        .expect("Fix: the snapshot gate must be able to write the fixture surface");

    // Mutate source to simulate removing an API without updating the snapshot.
    fs::write(crate_dir.join("src/lib.rs"), "pub struct Alpha;\n")
        .expect("Fix: fixture crate root must be writable");

    let verify = PublicApiSnapshot
        .run(&GateCtx::new(root.to_path_buf(), Vec::new()))
        .expect("Fix: the snapshot gate must run");
    assert_eq!(verify.count(), 1, "stale snapshot must report 1 finding");
    let message = &verify.findings[0].message;
    assert!(
        message.contains("the public API of `fixture` no longer matches its snapshot"),
        "unexpected finding message: {message}"
    );
    assert!(
        message.contains("1 removed"),
        "finding must record 1 removed item: {message}"
    );
    assert!(
        message.contains("-pub fn fixture::removed_item()"),
        "finding must name the removed item: {message}"
    );
}

/// The committed snapshots are the live public surface.
///
/// WHY: every other contract in this file judges which snapshot files exist and
/// how one is extracted. None of them reads the committed bytes against today's
/// rustdoc output, so a public item added without refreshing its snapshot
/// passes all of them. `.github/workflows/public-api.yml` runs this gate, so the
/// drift was caught in CI and not by a local test run.
#[test]
fn committed_snapshots_match_the_live_public_surface() {
    let root = workspace_root();
    let report = PublicApiSnapshot
        .run(&GateCtx::new(root, Vec::new()))
        .expect("Fix: the public API snapshot gate must be able to run");
    assert_eq!(
        report.count(),
        0,
        "Fix: the public API changed without its snapshot. Refresh it with \
         `xtask public-api-snapshot --write --crate <package>`.\n{:#?}",
        report.findings
    );
}
