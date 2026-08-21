//! Rules that keep this owner's account of the inventory registries true.
//!
//! WHY: the defect these exist for is invisible by construction. An unlinked
//! source crate contributes nothing, so every count, every document and every
//! rule agrees with itself while describing a partial registry. Two things
//! therefore have to be proven, and neither can be proven from a list somebody
//! maintains by hand:
//!
//! 1. This test binary links nothing but this crate. It names no driver crate and
//!    no operation-owning crate, so the registrations it reads are there only
//!    because the accessors call into their sources. A regression in an anchor
//!    turns these red.
//! 2. The candidate set is read from the tree at run time. A new crate that
//!    publishes an `operation_catalog` module or submits a `BackendRegistration`
//!    turns these red until it is recorded as a source, rather than being
//!    absorbed in silence.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use vyre_registry_link::backend::{
    linked_backend_source_names, linked_backend_sources, live_backend_registry,
    live_backend_registry_by_precedence, DECLARED_SOURCES,
};
use vyre_registry_link::operation::{live_operation_registry, registration_sources};

fn checkout_root() -> PathBuf {
    structure_gate::workspace_root()
}

fn workspace_members(root: &Path) -> Vec<String> {
    let manifest = root.join("Cargo.toml");
    let text = std::fs::read_to_string(&manifest)
        .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", manifest.display()));
    let table: toml::Value = toml::from_str(&text)
        .unwrap_or_else(|error| panic!("Fix: cannot parse {}: {error}", manifest.display()));
    table
        .get("workspace")
        .and_then(|workspace| workspace.get("members"))
        .and_then(toml::Value::as_array)
        .expect("Fix: the workspace manifest must list its members")
        .iter()
        .filter_map(toml::Value::as_str)
        .map(str::to_string)
        .collect()
}

fn crate_name(member: &str) -> String {
    member.rsplit('/').next().unwrap_or(member).to_string()
}

fn rust_sources(dir: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_sources(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
}

/// Whether a crate publishes an `operation_catalog` module, at any depth.
///
/// `vyre-libs` keeps its catalog in `src/plumbing/registration/`, so a check
/// that only looked at `src/operation_catalog.rs` read that crate as owning
/// nothing. The ownership set then lost a counted source in silence, which is
/// the exact shape of defect this file exists to make impossible.
fn owns_an_operation_catalog(src: &Path) -> bool {
    const MODULE: &str = "operation_catalog";
    let mut files = Vec::new();
    rust_sources(src, &mut files);
    files.iter().any(|path| {
        let is_file_module = path.file_stem().is_some_and(|stem| stem == MODULE);
        let is_directory_module = path.file_name().is_some_and(|name| name == "mod.rs")
            && path
                .parent()
                .and_then(Path::file_name)
                .is_some_and(|name| name == MODULE);
        is_file_module || is_directory_module
    })
}

/// A submission is `inventory::submit!` followed by the registration type. The
/// type is matched on the following lines rather than anywhere in the file, so a
/// doc comment naming the type is not read as a registration.
fn submits(text: &str, registration_type: &str) -> bool {
    let lines: Vec<&str> = text.lines().collect();
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") || !trimmed.contains("inventory::submit!") {
            continue;
        }
        let opens = format!("{registration_type} {{");
        if lines[index..lines.len().min(index + 4)]
            .iter()
            .any(|candidate| candidate.contains(&opens))
        {
            return true;
        }
    }
    false
}

fn members_submitting(registration_type: &str) -> Vec<String> {
    let root = checkout_root();
    let mut submitters = Vec::new();
    for member in workspace_members(&root) {
        let mut files = Vec::new();
        rust_sources(&root.join(&member).join("src"), &mut files);
        if files.iter().any(|path| {
            std::fs::read_to_string(path).is_ok_and(|text| submits(&text, registration_type))
        }) {
            submitters.push(crate_name(&member));
        }
    }
    submitters.sort();
    submitters
}

/// A crate declares that it owns operation registrations by publishing an
/// `operation_catalog` module. Every such crate has to be a counted source here.
#[test]
fn every_crate_that_owns_an_operation_catalog_is_a_source() {
    let root = checkout_root();
    let mut owners: Vec<String> = workspace_members(&root)
        .into_iter()
        .filter(|member| owns_an_operation_catalog(&root.join(member).join("src")))
        .map(|member| crate_name(&member))
        .collect();
    owners.sort();
    let mut counted: Vec<String> = registration_sources()
        .iter()
        .map(|(name, _)| (*name).to_string())
        .collect();
    counted.sort();
    assert_eq!(
        counted, owners,
        "Fix: every crate publishing an `operation_catalog` module must be counted in `vyre_registry_link::operation`, so its registrations are linked into whatever binary reads the operation registry"
    );
}

/// Every counted operation source reached the registry in this binary, which
/// names none of them.
#[test]
fn every_operation_source_reaches_the_registry_through_the_accessor() {
    let registry = live_operation_registry();
    for (source, count) in registration_sources() {
        assert!(
            *count > 0,
            "Fix: `{source}` reached no operation registration through the accessor, so the anchor that links it is gone"
        );
    }
    assert!(
        registry.iter().len() > 0,
        "Fix: the operation registry is empty in a binary that reads it through the accessor"
    );
}

/// The registry holds exactly what the counted sources contributed.
///
/// WHY: the per-source floor catches a source that vanished entirely. This
/// catches the other direction: registrations reaching the registry from a crate
/// nobody counted, which means the accessor's account of where registrations come
/// from is out of date even though every rule still passes.
#[test]
fn the_operation_registry_holds_exactly_what_the_counted_sources_contributed() {
    let registry = live_operation_registry();
    let counted: usize = registration_sources().iter().map(|(_, count)| count).sum();
    assert_eq!(
        registry.iter().len(),
        counted,
        "Fix: the live registry carries registrations from a crate the accessor does not name; count it there or stop registering from it"
    );
}

/// Every operation the tree registers reached this binary's registry.
///
/// WHY: the per-source floor above asks whether a source contributed anything,
/// and both other operation rules compare the registry against the same live
/// counts, so a source linked with a narrow feature selection satisfies all
/// three while the registry is missing whole domains: the counts shrink
/// together and agree. `docs/generated/catalog.toml` is generated from a build
/// that links every registering domain and is held to the live registry by its
/// own gate, so it is the one account of the whole set that does not shrink
/// with this crate's feature selection. Reading it here is what turns a
/// narrowed dependency red. It caught exactly that: `vyre-libs` was linked on
/// default features, so 14 domains never registered.
#[test]
fn the_registry_holds_every_operation_the_generated_catalog_names() {
    let catalog = checkout_root().join("docs/generated/catalog.toml");
    let text = std::fs::read_to_string(&catalog)
        .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", catalog.display()));
    let table: toml::Value = toml::from_str(&text)
        .unwrap_or_else(|error| panic!("Fix: cannot parse {}: {error}", catalog.display()));
    let named: BTreeSet<String> = table
        .get("subsystem")
        .and_then(toml::Value::as_array)
        .expect("Fix: the generated catalog must list its subsystems")
        .iter()
        .filter_map(|subsystem| subsystem.get("operations"))
        .filter_map(toml::Value::as_array)
        .flatten()
        .filter_map(toml::Value::as_str)
        .map(str::to_string)
        .collect();
    assert!(
        !named.is_empty(),
        "Fix: the generated catalog named no operation, so this rule proves nothing; regenerate it with `xtask catalog --write`"
    );
    let registry = live_operation_registry();
    let present: BTreeSet<String> = registry
        .iter()
        .map(|operation| operation.id.to_string())
        .collect();
    let absent: Vec<&String> = named.difference(&present).collect();
    assert!(
        absent.is_empty(),
        "Fix: {} of the {} operations the generated catalog names did not reach the registry this crate publishes, so every rule reading it is judging a partial tree. Widen the feature selection on the registration source in vyre-registry-link/Cargo.toml until each one links. Missing: {absent:?}",
        absent.len(),
        named.len()
    );
}

/// A crate declares that it owns a backend registration by submitting one. Every
/// such crate has to be declared here, because this owner is what references it.
#[test]
fn every_crate_that_submits_a_backend_registration_is_declared_here() {
    let submitters = members_submitting("BackendRegistration");
    let mut declared: Vec<String> = DECLARED_SOURCES
        .iter()
        .map(|name| (*name).to_string())
        .collect();
    declared.sort();
    assert_eq!(
        declared, submitters,
        "Fix: every workspace member submitting a `BackendRegistration` must be declared in `vyre_registry_link::backend::DECLARED_SOURCES` with a cargo feature that links it, so its registration reaches whatever binary reads the backend registry"
    );
}

/// The default feature set links every declared driver, so these rules judge the
/// whole set rather than a feature-narrow slice.
#[test]
fn the_default_feature_set_links_every_declared_driver() {
    let mut linked = linked_backend_source_names();
    linked.sort_unstable();
    let mut declared: Vec<&str> = DECLARED_SOURCES.to_vec();
    declared.sort_unstable();
    assert_eq!(
        declared, linked,
        "Fix: each entry in `DECLARED_SOURCES` needs a default-on cargo feature that pushes it into the linked set"
    );
}

/// The backend registry holds exactly what the linked drivers say they registered
/// on this target, in this binary, which names none of them.
#[test]
fn the_backend_registry_holds_exactly_what_the_linked_drivers_registered() {
    let registry = live_backend_registry().expect("Fix: the backend registry must freeze cleanly");
    let present: BTreeSet<&str> = registry
        .iter()
        .map(|registration| registration.id)
        .collect();
    let expected: BTreeSet<&str> = linked_backend_sources()
        .iter()
        .filter_map(|source| source.registered_here)
        .collect();
    assert_eq!(
        expected, present,
        "Fix: the backend registry carries a registration from a crate `DECLARED_SOURCES` does not name; declare it here or stop registering from it"
    );
}

/// The precedence-ordered view is the same set, so a consumer that sorts by
/// precedence is judged against the same floor.
#[test]
fn the_precedence_view_carries_the_same_backends() {
    let flat = live_backend_registry().expect("Fix: the backend registry must freeze cleanly");
    let by_precedence = live_backend_registry_by_precedence()
        .expect("Fix: the backend registry must freeze cleanly");
    let flat_ids: BTreeSet<&str> = flat.iter().map(|registration| registration.id).collect();
    let ordered_ids: BTreeSet<&str> = by_precedence
        .iter()
        .map(|registration| registration.id)
        .collect();
    assert_eq!(
        flat_ids, ordered_ids,
        "Fix: the precedence view and the flat registry disagree about which backends are linked"
    );
}
