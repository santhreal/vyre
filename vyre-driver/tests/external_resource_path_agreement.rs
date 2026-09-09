//! Proves every concrete driver publishes its external-resource surface at one shared path.
//!
//! WHY: the external memory descriptor, its handle enum, the importer and the imported
//! resource record are one contract restated per device API. A caller that learned the
//! path from one driver writes the same path for the next, so a driver publishing the
//! surface under a module of its own, or at two paths at once, turns that path into a
//! per-backend fact and splits one contract into as many shapes as there are backends.
//!
//! The crate list is the `concrete-backend` layer of `docs/CRATE_OWNERSHIP.toml`, and the
//! path of each published item is read from that crate's snapshot under `docs/public-api`,
//! which the `public-api-snapshot` gate holds equal to the surface the crate really
//! publishes. Adding a driver is enough to bring it under this test, and a driver whose
//! source declares the surface but whose snapshot does not publish it is reported by name.
//!
//! What this does not judge: the item set. A handle variant is a hardware fact, so a
//! backend admitting a file descriptor where another admits an IOSurface is not a defect.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use toml::Value as TomlValue;
use vyre_test_support::monorepo::vyre_workspace_root;

/// Type-name fragments that place an item in the external-resource surface.
const SURFACE_MARKERS: [&str; 3] = ["ExternalMemory", "ExternalResource", "ImportedResource"];

/// Fewest concrete drivers that carry the surface today, so a parse that silently matches
/// nothing fails instead of certifying agreement across an empty set.
const MINIMUM_PARTICIPANTS: usize = 3;

/// How a failure names the empty module path, so a red run reads as prose.
const CRATE_ROOT: &str = "the crate root";

/// Packages recorded in the `concrete-backend` layer of `docs/CRATE_OWNERSHIP.toml`.
fn concrete_backends(root: &Path) -> BTreeSet<String> {
    let path = root.join("docs/CRATE_OWNERSHIP.toml");
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("Fix: failed to read {}: {error}", path.display()));
    let document: TomlValue = toml::from_str(&text)
        .unwrap_or_else(|error| panic!("Fix: failed to parse {}: {error}", path.display()));
    let entries = document
        .get("crate")
        .and_then(TomlValue::as_array)
        .unwrap_or_else(|| panic!("Fix: {} declares no `crate` array", path.display()));
    let packages: BTreeSet<String> = entries
        .iter()
        .filter(|entry| entry.get("layer").and_then(TomlValue::as_str) == Some("concrete-backend"))
        .filter_map(|entry| entry.get("package").and_then(TomlValue::as_str))
        .map(str::to_string)
        .collect();
    assert!(
        !packages.is_empty(),
        "Fix: {} records no concrete-backend package, so no driver would be judged",
        path.display()
    );
    packages
}

/// Whether a name belongs to the external-resource surface.
fn is_surface_name(name: &str) -> bool {
    SURFACE_MARKERS.iter().any(|marker| name.contains(marker))
}

/// The crate-rooted path one snapshot line declares, without the crate segment.
///
/// A line names paths from several crates: a field type and a return type are rendered at
/// their own owner's path. The subject is the first path rooted at this crate, which is
/// what the extractor emits first for every line shape it produces.
fn subject_path<'line>(line: &'line str, crate_ident: &str) -> Option<&'line str> {
    let needle = format!("{crate_ident}::");
    let mut search = 0;
    while let Some(offset) = line[search..].find(&needle) {
        let start = search + offset;
        let joined = start > 0
            && line[..start]
                .chars()
                .next_back()
                .is_some_and(|ch| ch.is_alphanumeric() || ch == '_' || ch == ':');
        if joined {
            search = start + needle.len();
            continue;
        }
        let end = start
            + line[start..]
                .find(|ch: char| !(ch.is_alphanumeric() || ch == '_' || ch == ':'))
                .unwrap_or(line.len() - start);
        // A field line ends its subject with the `:` that opens the type, and `:` is a
        // path character, so the scan stops one byte past the path it wanted.
        return line[start..end].trim_end_matches(':').strip_prefix(&needle);
    }
    None
}

/// Split a crate-relative path into the module path it is reachable through and the item
/// it names. A type segment starts uppercase, so everything before the first such segment
/// is the module path and the crate root is the empty string.
fn split_module(path: &str) -> (String, String) {
    let segments: Vec<&str> = path.split("::").filter(|part| !part.is_empty()).collect();
    let boundary = segments
        .iter()
        .position(|segment| segment.starts_with(char::is_uppercase))
        .unwrap_or(segments.len().saturating_sub(1));
    (
        segments[..boundary].join("::"),
        segments[boundary..].join("::"),
    )
}

/// Every external-resource item one snapshot publishes, with the module paths it is
/// reachable through.
fn published_surface(snapshot: &str, crate_ident: &str) -> BTreeMap<String, BTreeSet<String>> {
    let mut surface: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for line in snapshot.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some(path) = subject_path(trimmed, crate_ident) else {
            continue;
        };
        let (module, item) = split_module(path);
        if item.is_empty() || !is_surface_name(&item) {
            continue;
        }
        surface.entry(item).or_default().insert(module);
    }
    surface
}

/// Whether a crate's own sources declare a public external-resource item, which is the
/// source-side answer to whether the crate carries the surface at all.
fn declares_surface(crate_root: &Path) -> bool {
    let mut pending = vec![crate_root.join("src")];
    while let Some(directory) = pending.pop() {
        let Ok(entries) = fs::read_dir(&directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            if path.extension().is_none_or(|extension| extension != "rs") {
                continue;
            }
            let Ok(text) = fs::read_to_string(&path) else {
                continue;
            };
            if text.lines().any(|line| {
                let trimmed = line.trim_start();
                (trimmed.starts_with("pub struct ") || trimmed.starts_with("pub enum "))
                    && is_surface_name(trimmed)
            }) {
                return true;
            }
        }
    }
    false
}

/// The module path each concrete driver publishes its external-resource surface at.
///
/// A driver whose sources declare the surface but whose snapshot publishes it at no path,
/// or at more than one, is returned as an error rather than dropped, because both are the
/// shape this test exists to reject.
fn published_paths(root: &Path) -> Result<BTreeMap<String, String>, Vec<String>> {
    let mut paths = BTreeMap::new();
    let mut wrong = Vec::new();
    for package in concrete_backends(root) {
        let declared = declares_surface(&root.join(&package));
        let snapshot_path = root.join("docs/public-api").join(format!("{package}.txt"));
        let Ok(snapshot) = fs::read_to_string(&snapshot_path) else {
            if declared {
                wrong.push(format!(
                    "`{package}` declares an external-resource surface and has no snapshot at {}, so the path it publishes cannot be read",
                    snapshot_path.display()
                ));
            }
            continue;
        };
        let crate_ident = package.replace('-', "_");
        let surface = published_surface(&snapshot, &crate_ident);
        let modules: BTreeSet<&String> = surface.values().flatten().collect();
        if !declared {
            if !modules.is_empty() {
                wrong.push(format!(
                    "`{package}` publishes an external-resource surface its sources do not declare, at {modules:?}"
                ));
            }
            continue;
        }
        match modules.len() {
            1 => {
                let module = modules
                    .into_iter()
                    .next()
                    .expect("a set of one holds one path");
                let path = if module.is_empty() {
                    CRATE_ROOT.to_string()
                } else {
                    module.clone()
                };
                paths.insert(package, path);
            }
            0 => wrong.push(format!(
                "`{package}` declares an external-resource surface and publishes none of it; publish it at the path its sibling drivers use"
            )),
            _ => wrong.push(format!(
                "`{package}` publishes its external-resource surface at {} paths {:?}; keep one and delete the rest",
                modules.len(),
                modules
            )),
        }
    }
    if wrong.is_empty() {
        Ok(paths)
    } else {
        Err(wrong)
    }
}

#[test]
fn every_concrete_driver_publishes_its_external_resource_surface_at_one_path() {
    let root = vyre_workspace_root();
    match published_paths(&root) {
        Ok(paths) => assert!(
            paths.len() >= MINIMUM_PARTICIPANTS,
            "Fix: only {} concrete drivers were judged {:?}; the surface parse matched fewer crates than carry it",
            paths.len(),
            paths
        ),
        Err(wrong) => panic!("Fix: {}", wrong.join("; ")),
    }
}

#[test]
fn concrete_drivers_agree_on_the_external_resource_path() {
    let root = vyre_workspace_root();
    let paths = published_paths(&root).unwrap_or_else(|wrong| panic!("Fix: {}", wrong.join("; ")));
    let mut by_path: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (package, path) in &paths {
        by_path
            .entry(path.as_str())
            .or_default()
            .push(package.as_str());
    }
    assert_eq!(
        by_path.len(),
        1,
        "Fix: concrete drivers publish their external-resource surface at {} different paths {:?}; one contract addressed several ways is several contracts",
        by_path.len(),
        by_path
    );
}

/// The parse must see a second path where one exists, or every assertion above passes on
/// a snapshot it failed to read.
#[test]
fn a_second_path_in_one_driver_is_visible_to_the_parse() {
    let snapshot = "\
pub struct vyre_driver_probe::ProbeExternalMemoryDescriptor
pub struct vyre_driver_probe::external_resource::ProbeExternalMemoryDescriptor
";
    let surface = published_surface(snapshot, "vyre_driver_probe");
    let modules: BTreeSet<String> = surface.values().flatten().cloned().collect();
    assert_eq!(
        modules,
        BTreeSet::from([String::new(), "external_resource".to_string()]),
        "Fix: the parse must see both paths of a doubly published item"
    );
}

/// A field type is rendered at its own owner's path on the same line as the field, so a
/// parse that read every path on a line would report a second path for every crate.
#[test]
fn a_rendered_field_type_is_not_read_as_a_second_path() {
    let snapshot = "pub vyre_driver_probe::ProbeExternalMemoryDescriptor::handle: vyre_driver_probe::external_resource::ProbeExternalMemoryHandle";
    let surface = published_surface(snapshot, "vyre_driver_probe");
    let modules: BTreeSet<String> = surface.values().flatten().cloned().collect();
    assert_eq!(
        modules,
        BTreeSet::from([String::new()]),
        "Fix: only the subject of a line declares a path"
    );
}
