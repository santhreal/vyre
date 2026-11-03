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
//! Both sets are derived from the workspace manifest and each crate's shipped
//! source, so a backend crate added tomorrow is covered without editing this
//! file, and a driver crate that registers nothing fails here rather than
//! shrinking the scan in silence. Test sources, as `TestSources` reads them,
//! register nothing a build ships: a gate fixture that builds a registration
//! is not a backend.
//!
//! `[dev-dependencies]` are not scanned: parity tests compare a backend
//! against the reference oracle on purpose, and that dependency does not exist
//! in a shipped build.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use structure_gate::cfg_test::TestSources;
use structure_gate::source_scan::{rust_sources_with_text, SourceText};
use structure_gate::workspace_manifest::workspace_members;
use structure_gate::workspace_root;

/// Crates whose whole purpose is host arithmetic. A crate that registers a
/// backend may not link one outside `[dev-dependencies]`.
const HOST_ARITHMETIC_CRATES: &[&str] = &["vyre-reference"];

/// Holds the reference interpreter's driver-facing surface: the target profile
/// and the semantic executor the conformance oracle calls directly. It
/// registers no backend, so host arithmetic has no dispatch identity at all,
/// and it is the one concrete-driver member excluded from the scan below.
const REFERENCE_DRIVER: &str = "vyre-driver-reference";

/// The shared, backend-neutral driver crate. Every concrete driver depends on
/// it; it registers no backend of its own.
const SHARED_DRIVER: &str = "vyre-driver";

/// Prefix every concrete driver member carries.
const CONCRETE_DRIVER_PREFIX: &str = "vyre-driver-";

/// The struct-literal field a backend registration sets.
const REFERENCE_ORACLE_FIELD: &str = "reference_oracle:";

struct BackendCrate {
    name: String,
    /// `true` when the crate's source declares `reference_oracle: true`.
    declares_reference_oracle: bool,
    dependencies: BTreeSet<String>,
}

/// `dependencies` and `build-dependencies` keys of one crate.
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

/// `Some(true)` when `text` sets `reference_oracle: true`, `Some(false)` when
/// it sets only `false`, and `None` when it sets the field nowhere.
fn reference_oracle_flag(text: &str) -> Option<bool> {
    let mut declared = None;
    let mut search = 0;
    while let Some(rel) = text[search..].find(REFERENCE_ORACLE_FIELD) {
        let start = search + rel + REFERENCE_ORACLE_FIELD.len();
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
    declared
}

/// Every member whose shipped source sets `reference_oracle`, with whether any
/// site sets it `true`.
fn registering_members(root: &Path, members: &[String]) -> BTreeMap<String, bool> {
    let tests = TestSources::of(root);
    let mut registering = BTreeMap::new();
    for source in rust_sources_with_text(root) {
        let SourceText::Read { path, text } = source else {
            continue;
        };
        if tests.contains(&path) {
            continue;
        }
        let Some(member) = members
            .iter()
            .find(|member| path.starts_with(&format!("{member}/src/")))
        else {
            continue;
        };
        if let Some(flag) = reference_oracle_flag(&text) {
            let declared = registering.entry(member.clone()).or_insert(false);
            *declared |= flag;
        }
    }
    registering
}

fn member_name(member: &str) -> String {
    Path::new(member)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(member)
        .to_string()
}

fn backend_crates() -> Vec<BackendCrate> {
    let root = workspace_root();
    let members = workspace_members(&root);
    let crates: Vec<BackendCrate> = registering_members(&root, &members)
        .into_iter()
        .map(|(member, declares_reference_oracle)| BackendCrate {
            name: member_name(&member),
            declares_reference_oracle,
            dependencies: declared_dependencies(&root.join(&member).join("Cargo.toml")),
        })
        .collect();
    // The scan is a source-text search, so an enumeration that breaks reports
    // an empty tree rather than a violation. The expected set is derived from
    // the same manifest: every concrete driver member registers a backend, and
    // the reference driver registers none. Adding a driver crate that forgets
    // to register, or restoring a registration to the reference driver, turns
    // this red before any linkage rule below is consulted.
    let found: BTreeSet<String> = crates.iter().map(|entry| entry.name.clone()).collect();
    let mut expected: BTreeSet<String> = members
        .iter()
        .map(|member| member_name(member))
        .filter(|name| name.starts_with(CONCRETE_DRIVER_PREFIX))
        .collect();
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
         cross-backend fallback this contract exists to rule out. Backend selection belongs to \
         `acquire_preferred_dispatch_backend`, which reports the failure instead."
    );
}
