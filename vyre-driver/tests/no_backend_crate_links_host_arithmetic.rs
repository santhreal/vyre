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
//! The interpreter is no longer a registered backend, so the set of crates
//! registering one is exactly the concrete driver members other than
//! `vyre-driver-reference`, and no workspace crate sets `reference_oracle`. The
//! flag survives for an out-of-tree registration, which
//! `acquire_preferred_dispatch_backend` skips.
//!
//! Both sets are derived from the workspace manifest and each crate's own
//! source, so a backend crate added tomorrow is covered without editing this
//! file, and a driver crate that registers nothing fails here rather than
//! shrinking the scan in silence.
//!
//! `[dev-dependencies]` are deliberately not scanned: parity tests compare a
//! backend against the reference oracle on purpose, and that dependency does not
//! exist in a shipped build.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vyre_test_support::monorepo::vyre_workspace_root;

/// Crates whose whole purpose is host arithmetic. A crate that registers a
/// backend may not link one outside `[dev-dependencies]`.
const HOST_ARITHMETIC_CRATES: &[&str] = &["vyre-reference"];

/// Owns the reference interpreter's driver-facing surface: the target profile
/// and the semantic executor the conformance oracle calls directly. It
/// registers no backend, so host arithmetic has no dispatch identity at all,
/// and it is the one concrete-driver member excluded from the scan below.
const REFERENCE_DRIVER: &str = "vyre-driver-reference";

/// The shared, backend-neutral driver crate. Every concrete driver depends on
/// it; it registers no backend of its own.
const SHARED_DRIVER: &str = "vyre-driver";

/// Prefix every concrete driver member carries.
const CONCRETE_DRIVER_PREFIX: &str = "vyre-driver-";

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

/// Concrete driver members, by directory name, from the workspace manifest.
fn concrete_driver_members(root: &Path) -> BTreeSet<String> {
    workspace_members(root)
        .into_iter()
        .filter_map(|member| {
            let name = Path::new(&member).file_name()?.to_str()?.to_string();
            name.starts_with(CONCRETE_DRIVER_PREFIX).then_some(name)
        })
        .collect()
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
    // The scan is a source-text search, so an enumeration that breaks reports
    // an empty tree rather than a violation. The expected set is derived from
    // the same manifest: every concrete driver member registers a backend, and
    // the reference driver registers none. Adding a driver crate that forgets
    // to register, or restoring a registration to the reference driver, turns
    // this red before any linkage rule below is consulted.
    let found: BTreeSet<String> = crates.iter().map(|entry| entry.name.clone()).collect();
    let mut expected = concrete_driver_members(&root);
    expected.remove(REFERENCE_DRIVER);
    assert_eq!(
        found, expected,
        "Fix: the crates registering a backend must be exactly the concrete driver members other \
         than {REFERENCE_DRIVER}. A missing one means the scan broke or a driver registers \
         nothing; an extra one means a backend is registered outside a driver crate; \
         {REFERENCE_DRIVER} appearing means host arithmetic regained a dispatch identity."
    );
    crates
}

#[test]
fn no_crate_declares_itself_a_reference_oracle() {
    let oracles: BTreeSet<String> = backend_crates()
        .into_iter()
        .filter(|entry| entry.declares_reference_oracle)
        .map(|entry| entry.name)
        .collect();
    assert!(
        oracles.is_empty(),
        "Fix: {oracles:?} set `reference_oracle: true`. No crate in this workspace registers the \
         interpreter as a backend, so the flag exists for an out-of-tree registration and \
         `acquire_preferred_dispatch_backend` skips it. A workspace crate setting it makes host \
         arithmetic reachable by explicit id, which is the dispatch route the reference backend \
         was deleted to close."
    );
}

#[test]
fn no_device_backend_crate_links_host_arithmetic() {
    let mut offenders: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for entry in backend_crates() {
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
