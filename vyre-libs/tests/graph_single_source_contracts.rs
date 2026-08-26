//! Contracts for graph-domain single-sourcing between `vyre-libs::graph` and the
//! dispatch wrappers in this crate.
//!
//! `vyre-libs::graph` owns graph algorithms and primitive Program builders. A wrapper
//! under `src/graph/dispatch/` may add scratch buffers, batching, a plan cache
//! and backend wiring, and must not fork the algorithm it dispatches.
//!
//! WHY these rules are derived rather than listed: the wrapper set, the primitive
//! each wrapper wraps, and the reference functions each primitive publishes are
//! all read from the tree at run time, so a new wrapper is judged without editing
//! this file.
//!
//! What these rules do not catch: a wrapper that names its primitive and then
//! reimplements the algorithm beside the call. Behavioural parity against the
//! primitive is asserted by each wrapper's own test module, which compares
//! dispatch output with the primitive reference; these rules exist to make sure
//! such a comparison has something to compare against.

use std::fs;
use std::path::{Path, PathBuf};

/// A dispatch wrapper: a directory under `src/graph/dispatch/` that pairs with
/// a canonical primitive module in `vyre-libs/src/graph/`.
#[derive(Debug)]
struct Wrapper {
    name: String,
    primitive_name: String,
    /// Every `.rs` file the wrapper owns (including relocated internal test files), concatenated.
    source: String,
    /// The primitive module's source, file or directory form.
    primitive_source: String,
}

/// The count below which the derivation is presumed broken rather than the tree
/// empty. Nine wrappers exist; a derivation that suddenly finds two would make
/// every rule here vacuous while still passing.
const WRAPPER_FLOOR: usize = 9;

/// Known non-wrapper dispatch infrastructure / pipeline modules under `src/graph/dispatch/`.
const KNOWN_DISPATCH_INFRASTRUCTURE: &[&str] = &[
    "dispatch_bridge",
    "frontier",
    "mod",
    "plan_cache",
    "structural_kernel_pipeline",
    "traversal_dispatch_pipeline",
];

fn is_known_dispatch_infrastructure(name: &str) -> bool {
    KNOWN_DISPATCH_INFRASTRUCTURE.contains(&name)
}
/// Find any duplicate graph owners in the workspace outside `canonical_crate_root`.
fn find_second_graph_owners(workspace_root: &Path, canonical_crate_root: &Path) -> Vec<String> {
    let mut other_owners = Vec::new();
    for entry in read_dir(workspace_root) {
        if entry.is_dir() && entry != canonical_crate_root {
            let candidate_graph = entry.join("src/graph");
            if candidate_graph.exists() {
                other_owners.push(candidate_graph.display().to_string());
            }
        }
    }
    other_owners
}

/// Assert that `vyre-libs` is the single canonical graph owner in the workspace.
fn assert_single_graph_owner() {
    let workspace_root = crate_root()
        .parent()
        .expect("Fix: vyre-libs must live under the workspace root")
        .to_path_buf();
    let other_owners = find_second_graph_owners(&workspace_root, &crate_root());
    assert!(
        other_owners.is_empty(),
        "Fix: detected a second graph owner in the workspace:\n{}\n`vyre-libs/src/graph` is the single canonical owner of the graph domain.",
        other_owners.join("\n")
    );
}

/// Find any uncatalogued or unpaired dispatch entries under `dispatch_dir`.
fn find_uncatalogued_dispatch_entries(dispatch_dir: &Path, graph_dir: &Path) -> Vec<String> {
    let mut uncatalogued = Vec::new();
    for entry in read_dir(dispatch_dir) {
        let name = entry
            .file_name()
            .expect("Fix: entry must have a name")
            .to_string_lossy()
            .into_owned();
        let stem = name.strip_suffix(".rs").unwrap_or(&name);
        if is_known_dispatch_infrastructure(stem) {
            continue;
        }
        if !entry.is_dir() {
            uncatalogued.push(format!("non-directory dispatch entry `{name}`"));
            continue;
        }
        if resolve_primitive_for_wrapper(&entry, stem, graph_dir).is_none() {
            uncatalogued.push(format!("unpaired dispatch wrapper `{name}`"));
        }
    }
    uncatalogued
}

/// Assert that every entry under `src/graph/dispatch/` is either recognized
/// dispatch infrastructure or pairs with a primitive in `src/graph/`.
fn assert_no_uncatalogued_dispatch_entries() {
    let dispatch = crate_root().join("src/graph/dispatch");
    let graph = crate_root().join("src/graph");
    let uncatalogued = find_uncatalogued_dispatch_entries(&dispatch, &graph);
    assert!(
        uncatalogued.is_empty(),
        "Fix: every entry under `src/graph/dispatch/` must be a derived wrapper or registered infrastructure:\n{}",
        uncatalogued.join("\n")
    );
}

#[test]
fn the_wrapper_set_is_derived_and_not_empty() {
    assert_single_graph_owner();
    assert_no_uncatalogued_dispatch_entries();
    let wrappers = wrappers();
    assert!(
        wrappers.len() >= WRAPPER_FLOOR,
        "Fix: only {} graph dispatch wrappers were derived, below the floor of {WRAPPER_FLOOR}; the pairing between `vyre-libs/src/graph/dispatch/<name>` and `vyre-libs/src/graph/<name>` broke, and every rule in this file would otherwise pass by judging nothing",
        wrappers.len()
    );
}

/// WHY: an infrastructure name exempts an entry under `src/graph/dispatch/`
/// from the pairing rule. A name whose module was deleted exempts whatever is
/// added under that name next, which is the pairing rule silently not applying
/// to a wrapper nobody decided about.
#[test]
fn every_registered_infrastructure_entry_still_exists() {
    let dispatch = crate_root().join("src/graph/dispatch");
    let missing: Vec<&str> = KNOWN_DISPATCH_INFRASTRUCTURE
        .iter()
        .copied()
        .filter(|name| {
            !dispatch.join(format!("{name}.rs")).is_file() && !dispatch.join(name).is_dir()
        })
        .collect();
    assert!(
        missing.is_empty(),
        "Fix: remove the infrastructure exemption for a module that no longer exists: {}",
        missing.join(", ")
    );
}

#[test]
fn every_dispatch_wrapper_names_the_graph_primitive_it_wraps() {
    let mut failures = Vec::new();
    for wrapper in wrappers() {
        let crate_path = format!("crate::graph::{}", wrapper.primitive_name);
        let crate_path_direct = format!("crate::graph::{}", wrapper.name);
        let libs_path = format!("vyre_libs::graph::{}", wrapper.primitive_name);
        let libs_path_direct = format!("vyre_libs::graph::{}", wrapper.name);
        let bare_path = format!("graph::{}", wrapper.primitive_name);
        let bare_path_direct = format!("graph::{}", wrapper.name);
        if !wrapper.source.contains(&crate_path)
            && !wrapper.source.contains(&crate_path_direct)
            && !wrapper.source.contains(&libs_path)
            && !wrapper.source.contains(&libs_path_direct)
            && !wrapper.source.contains(&bare_path)
            && !wrapper.source.contains(&bare_path_direct)
        {
            failures.push(format!(
                "{} never names {crate_path}, so it dispatches an algorithm it does not delegate",
                wrapper.name
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "Fix: a graph dispatch wrapper delegates its algorithm to the primitive it wraps:\n{}",
        failures.join("\n")
    );
}

#[test]
fn every_dispatch_wrapper_names_its_primitive_cpu_reference() {
    let mut failures = Vec::new();
    for wrapper in wrappers() {
        let references = cpu_reference_functions(&wrapper.primitive_source);
        if references.is_empty() {
            // The primitive publishes no CPU reference, so there is nothing for
            // the wrapper to compare against and nothing to require here.
            continue;
        }
        if !references
            .iter()
            .any(|reference| wrapper.source.contains(reference))
        {
            failures.push(format!(
                "{} names none of the CPU references its primitive publishes ({})",
                wrapper.name,
                references.join(", ")
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "Fix: a wrapper whose primitive publishes a CPU reference must compare against it rather than trusting the dispatch path:\n{}",
        failures.join("\n")
    );
}

#[test]
fn no_dispatch_wrapper_is_shadowed_by_an_older_module_path() {
    let crate_root = crate_root();
    let mut failures = Vec::new();
    for wrapper in wrappers() {
        let stale = crate_root.join("src").join(format!("{}.rs", wrapper.name));
        if stale.exists() {
            failures.push(format!(
                "{} also exists; a dispatch wrapper lives only under src/graph/dispatch/",
                stale.display()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "Fix: remove the module a wrapper was migrated from:\n{}",
        failures.join("\n")
    );
}

#[test]
fn the_dispatch_module_declares_every_wrapper_once() {
    let source = read(&crate_root().join("src/graph/dispatch/mod.rs"));
    let mut failures = Vec::new();
    for wrapper in wrappers() {
        let declaration = format!("mod {};", wrapper.name);
        let declared = source
            .lines()
            .filter(|line| line.trim_start().ends_with(&declaration))
            .count();
        if declared != 1 {
            failures.push(format!(
                "src/graph/dispatch/mod.rs declares `{}` {declared} times",
                wrapper.name
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "Fix: each wrapper is declared exactly once by the dispatch module:\n{}",
        failures.join("\n")
    );
}

#[test]
fn the_single_graph_owner_check_detects_workspace_duplicates() {
    let temp = std::env::temp_dir().join(format!("vyre_test_single_owner_{}", std::process::id()));
    let fake_vyre_libs = temp.join("vyre-libs");
    let fake_other_crate = temp.join("vyre-primitives");
    let fake_other_graph = fake_other_crate.join("src/graph");
    let _ = fs::create_dir_all(&fake_vyre_libs);
    let _ = fs::create_dir_all(&fake_other_graph);

    let foreign = find_second_graph_owners(&temp, &fake_vyre_libs);
    let _ = fs::remove_dir_all(&temp);
    assert_eq!(foreign.len(), 1);
    assert!(foreign[0].contains("vyre-primitives"));
}

#[test]
fn uncatalogued_wrapper_detection_catches_unpaired_entries() {
    let temp = std::env::temp_dir().join(format!("vyre_test_uncatalogued_{}", std::process::id()));
    let fake_dispatch = temp.join("dispatch");
    let fake_graph = temp.join("graph");
    let fake_unpaired = fake_dispatch.join("unknown_custom_wrapper");
    let _ = fs::create_dir_all(&fake_unpaired);
    let _ = fs::create_dir_all(&fake_graph);
    let _ = fs::write(fake_unpaired.join("mod.rs"), "pub fn foo() {}");

    let uncatalogued = find_uncatalogued_dispatch_entries(&fake_dispatch, &fake_graph);
    let _ = fs::remove_dir_all(&temp);
    assert_eq!(uncatalogued.len(), 1);
    assert!(uncatalogued[0].contains("unknown_custom_wrapper"));
}

/// Resolve the canonical primitive in `graph_dir` that a dispatch wrapper pairs with.
fn resolve_primitive_for_wrapper(
    wrapper_dir: &Path,
    wrapper_name: &str,
    graph_dir: &Path,
) -> Option<(String, String)> {
    // 1. Direct name match: src/graph/<name>.rs or src/graph/<name>/
    let direct_file = graph_dir.join(format!("{wrapper_name}.rs"));
    let direct_dir = graph_dir.join(wrapper_name);
    if direct_file.is_file() {
        return Some((wrapper_name.to_string(), read(&direct_file)));
    } else if direct_dir.is_dir() {
        return Some((wrapper_name.to_string(), concatenate(&direct_dir)));
    }

    // 2. Derive primitive name by inspecting wrapper source imports: crate::graph::<primitive>
    let wrapper_source = concatenate(wrapper_dir);
    let mut candidates = Vec::new();
    for token in ["crate::graph::", "graph::", "vyre_libs::graph::"] {
        let mut rest = wrapper_source.as_str();
        while let Some(pos) = rest.find(token) {
            let after = &rest[pos + token.len()..];
            let ident: String = after
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !ident.is_empty() && ident != "dispatch" && !candidates.contains(&ident) {
                let cand_file = graph_dir.join(format!("{ident}.rs"));
                let cand_dir = graph_dir.join(&ident);
                if cand_file.is_file() || cand_dir.is_dir() {
                    candidates.push(ident);
                }
            }
            rest = after;
        }
    }
    if candidates.len() == 1 {
        let prim_name = candidates.remove(0);
        let cand_file = graph_dir.join(format!("{prim_name}.rs"));
        let cand_dir = graph_dir.join(&prim_name);
        let prim_source = if cand_file.is_file() {
            read(&cand_file)
        } else {
            concatenate(&cand_dir)
        };
        Some((prim_name, prim_source))
    } else if candidates.len() > 1 {
        if let Some(matching) = candidates
            .iter()
            .find(|c| wrapper_name.starts_with(c.as_str()))
        {
            let prim_name = matching.clone();
            let cand_file = graph_dir.join(format!("{prim_name}.rs"));
            let cand_dir = graph_dir.join(&prim_name);
            let prim_source = if cand_file.is_file() {
                read(&cand_file)
            } else {
                concatenate(&cand_dir)
            };
            Some((prim_name, prim_source))
        } else {
            let prim_name = candidates.remove(0);
            let cand_file = graph_dir.join(format!("{prim_name}.rs"));
            let cand_dir = graph_dir.join(&prim_name);
            let prim_source = if cand_file.is_file() {
                read(&cand_file)
            } else {
                concatenate(&cand_dir)
            };
            Some((prim_name, prim_source))
        }
    } else {
        None
    }
}

/// Every dispatch wrapper, paired with the primitive module it wraps.
fn wrappers() -> Vec<Wrapper> {
    let crate_root = crate_root();
    let dispatch = crate_root.join("src/graph/dispatch");

    let mut wrappers = Vec::new();
    for entry in read_dir(&dispatch) {
        if !entry.is_dir() {
            continue;
        }
        let name = entry
            .file_name()
            .expect("Fix: a directory entry has a name")
            .to_string_lossy()
            .into_owned();
        if is_known_dispatch_infrastructure(&name) {
            continue;
        }
        let graph_dir = crate_root.join("src/graph");
        let Some((primitive_name, primitive_source)) =
            resolve_primitive_for_wrapper(&entry, &name, &graph_dir)
        else {
            continue;
        };

        let mut source = concatenate(&entry);
        let internal_test = crate_root.join("tests/internal/graph/dispatch").join(&name);
        if internal_test.is_dir() {
            source.push('\n');
            source.push_str(&concatenate(&internal_test));
        }

        wrappers.push(Wrapper {
            name,
            primitive_name,
            source,
            primitive_source,
        });
    }
    wrappers.sort_by(|left, right| left.name.cmp(&right.name));
    wrappers
}

/// Public functions of a primitive module whose name marks them a CPU
/// reference. The naming is the crate's own convention: `cpu_ref`,
/// `try_cpu_ref`, `cpu_ref_into`, `cpu_sparse_dense_step`, `csr_foc_cpu`.
fn cpu_reference_functions(primitive_source: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in primitive_source.lines() {
        let Some(rest) = line.trim_start().strip_prefix("pub fn ") else {
            continue;
        };
        let name: String = rest
            .chars()
            .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
            .collect();
        if name.contains("cpu") && !names.contains(&name) {
            names.push(name);
        }
    }
    names.sort();
    names
}

/// This crate's directory, resolved from the checkout this run is inside.
fn crate_root() -> PathBuf {
    vyre_test_support::monorepo::vyre_workspace_root().join("vyre-libs")
}

fn read(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("Fix: {} must be readable: {error}", path.display()))
}

fn read_dir(path: &Path) -> Vec<PathBuf> {
    fs::read_dir(path)
        .unwrap_or_else(|error| panic!("Fix: {} must be readable: {error}", path.display()))
        .map(|entry| {
            entry
                .unwrap_or_else(|error| {
                    panic!(
                        "Fix: an entry of {} must be readable: {error}",
                        path.display()
                    )
                })
                .path()
        })
        .collect()
}

/// Every `.rs` file under `dir`, at any depth, concatenated in path order.
fn concatenate(dir: &Path) -> String {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for path in read_dir(&directory) {
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
        .iter()
        .map(|file| read(file))
        .collect::<Vec<_>>()
        .join("\n")
}
