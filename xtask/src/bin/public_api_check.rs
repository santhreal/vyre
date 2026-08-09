//! Run `cargo public-api` against facade crates and diff the result
//! against each facade's committed `PUBLIC_API.md`.
//!
//! Run via `cargo xtaskbin public_api_check`. The binary
//! exits non-zero when any facade's public-API surface drifts from its
//! frozen snapshot, which is the publish-floor invariant.

use std::collections::HashSet;
use std::fs;
use std::io::{self, Read};
use std::path::Path;
use std::process::Command;

const MAX_PUBLIC_API_CHECK_TEXT_BYTES: u64 = 4_194_304;

const FACADE_CRATES: &[&str] = &[
    "vyre",
    "vyre-foundation",
    "vyre-driver",
    "vyre-driver-wgpu",
    "vyre-primitives",
    "vyre-spec",
    "vyre-libs",
];
const PUBLIC_API_SIMPLIFICATION_FLAG: &str = "-sss";
const UPDATE_COMMAND: &str = "./cargo_full run --bin public_api_check -- --update";
const BREAKING_UPDATE_COMMAND: &str =
    "./cargo_full run --bin public_api_check -- --update --allow-breaking";

fn print_help() {
    println!("Check publishable facade APIs against committed snapshots.");
    println!();
    println!("Usage: public_api_check [--update [--allow-breaking]]");
    println!();
    println!("Options:");
    println!("  --update          refresh snapshots after compatibility checks");
    println!("  --allow-breaking  permit removed API items while refreshing");
    println!("  -h, --help        print this help");
    println!();
    println!("Environment:");
    println!("  VYRE_CARGO_RUNNER  Cargo wrapper to invoke (default: cargo_full)");
    println!();
    println!("Exit codes:");
    println!("  0  snapshots match or were refreshed");
    println!("  1  API generation, compatibility, or snapshot checks failed");
    println!("  2  command-line arguments are invalid");
}

fn main() {
    let mut is_update = false;
    let mut allow_breaking = false;
    let mut show_help = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--update" => is_update = true,
            "--allow-breaking" => allow_breaking = true,
            "-h" | "--help" => show_help = true,
            _ => {
                eprintln!(
                    "Fix: unknown public_api_check argument `{arg}`. Use no arguments, `--update`, or `--update --allow-breaking`."
                );
                std::process::exit(2);
            }
        }
    }
    if show_help {
        if is_update || allow_breaking {
            eprintln!("Fix: `--help` cannot be combined with update arguments.");
            std::process::exit(2);
        }
        print_help();
        return;
    }
    if allow_breaking && !is_update {
        eprintln!("Fix: `--allow-breaking` requires `--update`.");
        std::process::exit(2);
    }

    let mut failed = false;

    let root = match workspace_root() {
        Some(root) => root,
        None => {
            eprintln!(
                "Fix: public_api_check must run from an xtask crate with a workspace parent."
            );
            std::process::exit(1);
        }
    };
    for error in facade_snapshot_inventory_failures(&root, FACADE_CRATES) {
        eprintln!("{error}");
        failed = true;
    }

    let cargo_runner = std::env::var("VYRE_CARGO_RUNNER").unwrap_or_else(|_| "cargo_full".into());

    for crate_name in FACADE_CRATES {
        let output = match Command::new(&cargo_runner)
            .arg("public-api")
            .arg(PUBLIC_API_SIMPLIFICATION_FLAG)
            .arg("-p")
            .arg(crate_name)
            .output()
        {
            Ok(output) => output,
            Err(error) => {
                eprintln!(
                    "Fix: failed to execute `{cargo_runner} public-api -p {crate_name}`: {error}"
                );
                failed = true;
                continue;
            }
        };

        if !output.status.success() {
            eprintln!(
                "Failed to generate public API for {}: {}",
                crate_name,
                String::from_utf8_lossy(&output.stderr)
            );
            failed = true;
            continue;
        }

        let new_api = match String::from_utf8(output.stdout) {
            Ok(api) => api,
            Err(error) => {
                eprintln!("Fix: public API output for {crate_name} was not UTF-8: {error}");
                failed = true;
                continue;
            }
        };

        let md_path = match find_crate_dir(crate_name, &root) {
            Ok(Some(p)) => p.join("PUBLIC_API.md"),
            Ok(None) => {
                eprintln!("Could not find dir for crate {}", crate_name);
                failed = true;
                continue;
            }
            Err(error) => {
                eprintln!("Fix: failed while locating crate {crate_name}: {error}");
                failed = true;
                continue;
            }
        };

        if is_update {
            if !allow_breaking {
                match read_text_bounded(&md_path) {
                    Ok(old_api) => {
                        let removed = removed_public_api_items(&old_api, &new_api);
                        if !removed.is_empty() {
                            eprintln!(
                                "Refusing to update {} because it removes or changes {} public API item(s):",
                                md_path.display(),
                                removed.len()
                            );
                            for item in removed.iter().take(20) {
                                eprintln!("  - {item}");
                            }
                            if removed.len() > 20 {
                                eprintln!("  ... and {} more", removed.len() - 20);
                            }
                            eprintln!(
                                "Fix the compatibility regression, or for an intentional breaking release run `{BREAKING_UPDATE_COMMAND}`."
                            );
                            failed = true;
                            continue;
                        }
                    }
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                    Err(error) => {
                        eprintln!(
                            "Fix: failed to read public API snapshot `{}` before update: {error}",
                            md_path.display()
                        );
                        failed = true;
                        continue;
                    }
                }
            }
            if let Err(error) = fs::write(&md_path, new_api) {
                eprintln!("Fix: failed to write `{}`: {error}", md_path.display());
                failed = true;
                continue;
            }
            println!("Updated {}", md_path.display());
        } else {
            let old_api = match read_text_bounded(&md_path) {
                Ok(api) => api,
                Err(error) => {
                    eprintln!(
                        "Fix: failed to read public API snapshot `{}`: {error}",
                        md_path.display()
                    );
                    failed = true;
                    continue;
                }
            };
            if new_api != old_api {
                eprintln!(
                    "Public API drifted for crate {crate_name}. Fix: run `{UPDATE_COMMAND}` to regenerate."
                );
                failed = true;
            } else {
                println!("{} API matches snapshot.", crate_name);
            }
        }
    }

    if failed {
        std::process::exit(1);
    }
}

fn workspace_root() -> Option<std::path::PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
}

fn facade_snapshot_inventory_failures(root: &Path, facade_crates: &[&str]) -> Vec<String> {
    let mut failures = Vec::new();
    let mut gated_snapshots = HashSet::new();
    for crate_name in facade_crates {
        match find_crate_dir(crate_name, root) {
            Ok(Some(crate_dir)) => {
                gated_snapshots.insert(crate_dir.join("PUBLIC_API.md"));
            }
            Ok(None) => failures.push(format!(
                "Fix: public API gate names unknown workspace package `{crate_name}`."
            )),
            Err(error) => failures.push(format!(
                "Fix: failed while resolving public API package `{crate_name}`: {error}"
            )),
        }
    }

    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_entry(|entry| {
            !entry
                .path()
                .components()
                .any(|component| component.as_os_str() == "target")
        })
    {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                failures.push(format!(
                    "Fix: failed while discovering public API snapshots: {error}"
                ));
                continue;
            }
        };
        if entry.file_name() != "PUBLIC_API.md" {
            continue;
        }
        let snapshot = entry.path();
        let Some(crate_dir) = snapshot.parent() else {
            continue;
        };
        if !crate_dir.join("Cargo.toml").is_file() {
            continue;
        }
        if !gated_snapshots.contains(snapshot) {
            failures.push(format!(
                "Fix: `{}` is a crate public API snapshot but its package is absent from the executable facade gate.",
                snapshot.display()
            ));
        }
    }

    for snapshot in gated_snapshots {
        if !snapshot.is_file() {
            failures.push(format!(
                "Fix: gated public API snapshot `{}` does not exist.",
                snapshot.display()
            ));
        }
    }
    failures
}

fn find_crate_dir(name: &str, root: &Path) -> Result<Option<std::path::PathBuf>, String> {
    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_entry(|entry| {
            !entry
                .path()
                .components()
                .any(|component| component.as_os_str() == "target")
        })
    {
        let entry = entry.map_err(|error| error.to_string())?;
        if entry.file_name() != "Cargo.toml" {
            continue;
        }
        if manifest_package_name(entry.path())?.as_deref() == Some(name) {
            return Ok(entry.path().parent().map(Path::to_path_buf));
        }
    }
    Ok(None)
}

fn manifest_package_name(path: &Path) -> Result<Option<String>, String> {
    let content =
        read_text_bounded(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let manifest = toml::from_str::<toml::Value>(&content)
        .map_err(|error| format!("{} is not valid TOML: {error}", path.display()))?;
    Ok(manifest
        .get("package")
        .and_then(|package| package.get("name"))
        .and_then(toml::Value::as_str)
        .map(str::to_owned))
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    let mut reader = fs::File::open(path)?.take(MAX_PUBLIC_API_CHECK_TEXT_BYTES.saturating_add(1));
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    if text.len() as u64 > MAX_PUBLIC_API_CHECK_TEXT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} exceeds {MAX_PUBLIC_API_CHECK_TEXT_BYTES} byte public API check read cap",
                path.display()
            ),
        ));
    }
    Ok(text)
}

fn removed_public_api_items<'a>(old_api: &'a str, new_api: &str) -> Vec<&'a str> {
    let new_items = new_api.lines().collect::<HashSet<_>>();
    old_api
        .lines()
        .filter(|item| !item.is_empty() && !new_items.contains(item))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every crate-level snapshot on disk must participate in the executable API drift gate.
    #[test]
    fn facade_crates_cover_all_committed_public_api_snapshots() {
        let root = workspace_root().expect("xtask must have a workspace parent");

        assert_eq!(
            facade_snapshot_inventory_failures(&root, FACADE_CRATES),
            Vec::<String>::new(),
        );
    }

    /// Adding a crate snapshot without adding its package to the gate must fail closed.
    #[test]
    fn facade_inventory_rejects_ungated_crate_snapshot() {
        let root = tempfile::tempdir().expect("temporary workspace must be available");
        let gated = root.path().join("gated");
        let forgotten = root.path().join("forgotten");
        fs::create_dir_all(&gated).expect("gated crate directory must be created");
        fs::create_dir_all(&forgotten).expect("forgotten crate directory must be created");
        fs::write(
            gated.join("Cargo.toml"),
            "[package]\nname = \"gated\"\nversion = \"0.1.0\"\n",
        )
        .expect("gated manifest must be written");
        fs::write(gated.join("PUBLIC_API.md"), "pub fn gated::stable()\n")
            .expect("gated snapshot must be written");
        fs::write(
            forgotten.join("Cargo.toml"),
            "[package]\nname = \"forgotten\"\nversion = \"0.1.0\"\n",
        )
        .expect("forgotten manifest must be written");
        fs::write(
            forgotten.join("PUBLIC_API.md"),
            "pub fn forgotten::stable()\n",
        )
        .expect("forgotten snapshot must be written");

        let failures = facade_snapshot_inventory_failures(root.path(), &["gated"]);
        let forgotten_snapshot = forgotten.join("PUBLIC_API.md").display().to_string();

        assert_eq!(failures.len(), 1);
        assert!(
            failures[0].contains(&forgotten_snapshot),
            "the diagnostic must identify the exact ungated snapshot: {failures:?}",
        );
    }

    /// Dependency metadata that mentions another package name must not hijack manifest lookup.
    #[test]
    fn crate_lookup_matches_package_name_field_only() {
        let root = tempfile::tempdir().expect("temporary workspace must be available");
        let decoy = root.path().join("a-decoy");
        let target = root.path().join("z-target");
        fs::create_dir_all(&decoy).expect("decoy crate directory must be created");
        fs::create_dir_all(&target).expect("target crate directory must be created");
        fs::write(
            decoy.join("Cargo.toml"),
            "[package]\nname = \"decoy\"\nversion = \"0.1.0\"\n[package.metadata]\nnote = 'name = \"target\"'\n",
        )
        .expect("decoy manifest must be written");
        fs::write(
            target.join("Cargo.toml"),
            "[package]\nname = \"target\"\nversion = \"0.1.0\"\n",
        )
        .expect("target manifest must be written");

        assert_eq!(
            find_crate_dir("target", root.path()).expect("manifest lookup must succeed"),
            Some(target),
        );
    }

    /// Drift diagnostics must name the executable update path that the repository actually ships.
    #[test]
    fn update_command_targets_public_api_check_binary() {
        assert_eq!(
            UPDATE_COMMAND,
            "./cargo_full run --bin public_api_check -- --update"
        );
    }

    /// Snapshot generation must omit dependency-derived blanket and auto implementations.
    #[test]
    fn public_api_generation_uses_stable_simplified_output() {
        assert_eq!(PUBLIC_API_SIMPLIFICATION_FLAG, "-sss");
    }

    /// Additive API growth must remain eligible for a normal snapshot update.
    #[test]
    fn compatibility_check_accepts_additions_without_removing_old_items() {
        let old = "pub fn api::stable()\npub struct api::Record\n";
        let new = "pub fn api::new()\npub fn api::stable()\npub struct api::Record\n";

        assert!(removed_public_api_items(old, new).is_empty());
    }

    /// Removed and signature-changed items must block the ordinary update path.
    #[test]
    fn compatibility_check_reports_removed_and_changed_items() {
        let old =
            "pub fn api::removed()\npub fn api::changed(value: u32)\npub struct api::Stable\n";
        let new = "pub fn api::changed(value: u64)\npub struct api::Stable\n";

        assert_eq!(
            removed_public_api_items(old, new),
            vec!["pub fn api::removed()", "pub fn api::changed(value: u32)"]
        );
    }
}
