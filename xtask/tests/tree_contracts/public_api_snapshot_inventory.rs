//! Publishable-crate and public-API snapshot set contracts.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::workspace_sources::workspace_root;

fn inventory_script() -> PathBuf {
    workspace_root().join("scripts/public_api_snapshot_inventory.py")
}

fn snapshot_script() -> PathBuf {
    workspace_root().join("scripts/check_public_api_snapshot.sh")
}

fn run_snapshot_refresh(root: &Path, package: &str) -> Result<(), String> {
    let scripts = root.join("scripts");
    fs::create_dir(&scripts)
        .map_err(|error| format!("could not create fixture scripts directory: {error}"))?;
    for source in [snapshot_script(), inventory_script()] {
        let file_name = source
            .file_name()
            .expect("Fix: public API scripts must have filenames");
        fs::copy(&source, scripts.join(file_name))
            .map_err(|error| format!("could not copy {}: {error}", source.display()))?;
    }
    fs::copy(workspace_root().join("cargo_full"), root.join("cargo_full"))
        .map_err(|error| format!("could not copy bounded cargo wrapper: {error}"))?;

    let output = Command::new("bash")
        .arg(scripts.join("check_public_api_snapshot.sh"))
        .arg("--refresh")
        .arg(package)
        .current_dir(root)
        .output()
        .map_err(|error| format!("could not execute public API snapshot refresh: {error}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

fn run_inventory(root: &Path) -> Result<BTreeSet<(String, String)>, String> {
    let output = Command::new("python3")
        .arg(inventory_script())
        .arg(root)
        .output()
        .map_err(|error| format!("could not execute public API inventory: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("public API inventory was not UTF-8: {error}"))?;
    stdout
        .lines()
        .map(|line| {
            let (member, package) = line
                .split_once(':')
                .ok_or_else(|| format!("inventory row has no separator: {line}"))?;
            Ok((member.to_string(), package.to_string()))
        })
        .collect()
}

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

/// The Python inventory must exactly match publishability in every workspace manifest.
///
/// This prevents a newly publishable crate from bypassing the stability gate and
/// prevents private tooling from accidentally acquiring a public API promise.
#[test]
fn inventory_matches_publishable_workspace_manifests() {
    let root = workspace_root();
    assert_eq!(
        run_inventory(&root).expect("Fix: public API inventory must execute"),
        manifest_inventory(&root)
    );
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
    let inventory = run_inventory(&workspace_root()).expect("Fix: inventory must execute");
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

    assert_eq!(
        run_inventory(temp.path()).expect("Fix: fixture inventory must execute"),
        BTreeSet::from([("public".to_string(), "public".to_string())])
    );
}

/// Snapshot extraction must model the externally reachable Rust API.
///
/// A source-line scan loses reexports and incorrectly includes `pub` items
/// nested beneath private modules. The canonical refresh must preserve the
/// public module and reexport while excluding both kinds of private item.
#[test]
fn snapshot_includes_modules_and_reexports_but_excludes_private_items() {
    let temp = tempfile::tempdir().expect("Fix: temporary workspace must be creatable");
    fs::write(
        temp.path().join("Cargo.toml"),
        "[workspace]\nmembers = [\"fixture\"]\nresolver = \"2\"\n",
    )
    .expect("Fix: fixture workspace manifest must be writable");

    let crate_dir = temp.path().join("fixture");
    fs::create_dir_all(crate_dir.join("src"))
        .expect("Fix: fixture crate source directory must be creatable");
    fs::write(
        crate_dir.join("Cargo.toml"),
        "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("Fix: fixture crate manifest must be writable");
    fs::write(
        crate_dir.join("src/lib.rs"),
        "mod private_module;\npub mod public_module;\npub use public_module::PublicType;\nfn private_function() {}\n",
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

    run_snapshot_refresh(temp.path(), "fixture")
        .expect("Fix: canonical public API snapshot refresh must succeed");
    let snapshot = fs::read_to_string(temp.path().join("docs/public-api/fixture.txt"))
        .expect("Fix: fixture public API snapshot must be readable");

    for public_item in [
        "pub mod fixture::public_module",
        "pub struct fixture::PublicType",
        "pub struct fixture::public_module::PublicType",
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

/// The committed snapshots are the live public surface.
///
/// WHY: every other contract in this file judges which snapshot files exist and
/// how one is extracted. None of them reads the committed bytes against today's
/// rustdoc output, so a public item added without refreshing its snapshot
/// passes all of them. `.github/workflows/public-api.yml` runs this script, so
/// the drift was caught in CI and not by a local test run.
#[test]
fn committed_snapshots_match_the_live_public_surface() {
    let script = snapshot_script();
    let output = Command::new("bash")
        .arg(&script)
        .current_dir(workspace_root())
        .output()
        .expect("Fix: bash must be available to run the public API snapshot gate");

    assert!(
        output.status.success(),
        "Fix: the public API changed without its snapshot. Refresh it with `{} --refresh <package>`.\nstdout:\n{}\nstderr:\n{}",
        script.display(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
