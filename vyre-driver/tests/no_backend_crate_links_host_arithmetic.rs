//! No concrete backend crate can reach host arithmetic or a peer backend.
//!
//! WHY: "the GPU path silently fell back to the CPU" is not something a
//! reviewer can rule out by reading a dispatch function, because the fallback
//! would be one `unwrap_or_else` deep in an error path. It is something the
//! dependency graph can rule out for the whole crate at once: a driver crate
//! that does not link `vyre-reference` has no host interpreter to substitute,
//! and one that does not link a peer `vyre-driver-*` crate cannot quietly hand
//! the work to another backend. That property was stated in prose on
//! `acquire_preferred_dispatch_backend` and in `routing/mod.rs`; this is the
//! executable form.
//!
//! The crate set is derived from the workspace manifest and the
//! `reference_oracle` flag in each crate's own source, so a backend crate added
//! tomorrow is covered without editing this file. A new crate that registers a
//! backend and links the reference interpreter fails here on its first build.
//!
//! `[dev-dependencies]` are deliberately not scanned: parity tests compare a
//! backend against the reference oracle on purpose, and that dependency does not
//! exist in a shipped build.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vyre_test_support::monorepo::vyre_workspace_root;

/// Crates whose whole purpose is host arithmetic. Only the reference driver may
/// link one.
const HOST_ARITHMETIC_CRATES: &[&str] = &["vyre-reference"];

/// The one crate allowed to declare itself a reference oracle.
const REFERENCE_DRIVER: &str = "vyre-driver-reference";

/// The shared, backend-neutral driver crate. Every concrete driver depends on
/// it; it registers no backend of its own.
const SHARED_DRIVER: &str = "vyre-driver";

struct BackendCrate {
    name: String,
    /// `true` when the crate's source declares `reference_oracle: true`.
    declares_reference_oracle: bool,
    dependencies: BTreeSet<String>,
}

fn workspace_members(root: &Path) -> Vec<String> {
    let manifest = std::fs::read_to_string(root.join("Cargo.toml"))
        .expect("Fix: the workspace manifest must be readable");
    let parsed: toml::Table = manifest
        .parse()
        .expect("Fix: the workspace manifest must parse as TOML");
    parsed
        .get("workspace")
        .and_then(|workspace| workspace.get("members"))
        .and_then(|members| members.as_array())
        .expect("Fix: the workspace manifest must declare [workspace] members")
        .iter()
        .map(|member| {
            member
                .as_str()
                .expect("Fix: every workspace member must be a string path")
                .to_string()
        })
        .collect()
}

/// Read `dependencies` and `build-dependencies` keys for one crate.
fn declared_dependencies(manifest_path: &Path) -> BTreeSet<String> {
    let text = std::fs::read_to_string(manifest_path)
        .unwrap_or_else(|e| panic!("Fix: {manifest_path:?} must be readable: {e}"));
    let parsed: toml::Table = text
        .parse()
        .unwrap_or_else(|e| panic!("Fix: {manifest_path:?} must parse as TOML: {e}"));
    let mut names = BTreeSet::new();
    for section in ["dependencies", "build-dependencies"] {
        if let Some(table) = parsed.get(section).and_then(|value| value.as_table()) {
            names.extend(table.keys().cloned());
        }
    }
    names
}

/// `Some(declares_reference_oracle)` when the crate's source registers a
/// backend, `None` when it does not.
fn reference_oracle_flag(src: &Path) -> Option<bool> {
    let mut files = Vec::new();
    vyre_test_support::collect_rust_files(src, &mut files);
    let mut declared = None;
    for path in files {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Fix: {path:?} must be readable: {e}"));
        let mut search = 0;
        while let Some(rel) = text[search..].find("reference_oracle:") {
            let start = search + rel + "reference_oracle:".len();
            search = start;
            let value = text[start..].trim_start();
            let flag = if value.starts_with("true") {
                true
            } else if value.starts_with("false") {
                false
            } else {
                // A field declaration or a read, not a struct-literal value.
                continue;
            };
            declared = Some(declared.unwrap_or(false) || flag);
        }
    }
    declared
}

fn backend_crates() -> Vec<BackendCrate> {
    let root = vyre_workspace_root();
    let mut crates = Vec::new();
    for member in workspace_members(&root) {
        let dir = root.join(&member);
        let Some(declares_reference_oracle) = reference_oracle_flag(&dir.join("src")) else {
            continue;
        };
        let name = dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(&member)
            .to_string();
        crates.push(BackendCrate {
            name,
            declares_reference_oracle,
            dependencies: declared_dependencies(&dir.join("Cargo.toml")),
        });
    }
    assert!(
        crates.len() >= 5,
        "Fix: the backend-crate scan found {} crates registering a backend. The workspace has at \
         least five concrete drivers, so a scan this small means the enumeration broke rather than \
         that the tree shrank.",
        crates.len()
    );
    crates
}

#[test]
fn only_the_reference_driver_declares_itself_a_reference_oracle() {
    let oracles: BTreeSet<String> = backend_crates()
        .into_iter()
        .filter(|entry| entry.declares_reference_oracle)
        .map(|entry| entry.name)
        .collect();
    assert_eq!(
        oracles,
        BTreeSet::from([REFERENCE_DRIVER.to_string()]),
        "Fix: exactly one crate may set `reference_oracle: true`, and it is {REFERENCE_DRIVER}. A \
         second one makes host arithmetic look like two independent oracles; zero makes the CPU \
         reference an implicit dispatch target."
    );
}

#[test]
fn no_device_backend_crate_links_host_arithmetic() {
    let mut offenders: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for entry in backend_crates() {
        if entry.name == REFERENCE_DRIVER {
            continue;
        }
        let linked: Vec<String> = HOST_ARITHMETIC_CRATES
            .iter()
            .filter(|host| entry.dependencies.contains(**host))
            .map(|host| (*host).to_string())
            .collect();
        if !linked.is_empty() {
            offenders.insert(entry.name, linked);
        }
    }
    assert!(
        offenders.is_empty(),
        "Fix: these backend crates link a host-arithmetic crate in [dependencies] or \
         [build-dependencies]: {offenders:?}. vyre never runs a user program on the CPU. Move the \
         dependency to [dev-dependencies] if it is there for parity testing, and delete the code \
         path if it is there for a fallback."
    );
}

#[test]
fn no_backend_crate_links_a_peer_backend_crate() {
    let mut offenders: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for entry in backend_crates() {
        let peers: Vec<String> = entry
            .dependencies
            .iter()
            .filter(|dependency| {
                let name = dependency.as_str();
                name.starts_with("vyre-driver") && name != SHARED_DRIVER && name != entry.name
            })
            .cloned()
            .collect();
        if !peers.is_empty() {
            offenders.insert(entry.name.clone(), peers);
        }
    }
    assert!(
        offenders.is_empty(),
        "Fix: these backend crates link a peer backend crate: {offenders:?}. A driver that can \
         construct another driver can substitute it on an error path, which is the silent \
         cross-backend fallback this gate exists to rule out. Backend selection belongs to \
         `acquire_preferred_dispatch_backend`, which reports the failure instead."
    );
}
