//! Contracts for graph-domain single-sourcing between `vyre-primitives` and the
//! dispatch wrappers in this crate.
//!
//! `vyre-primitives` owns graph algorithms and their CPU references. A wrapper
//! under `src/graph/dispatch/` may add scratch buffers, batching, a plan cache
//! and backend wiring, and must not fork the algorithm it dispatches.
//!
//! WHY these rules are derived rather than listed: this file used to carry a
//! table of eleven wrapper file names, each with a list of identifier and prose
//! fragments its source had to contain and a line-count ceiling. Every rename in
//! `vyre-primitives` and every test reorganisation in this crate broke it while
//! the delegation it was supposed to protect was intact: it failed on
//! `merge_frontier_or_changed` and `plan_dominator_frontier_dispatch`, two names
//! the primitive crate no longer uses, and on the absence of the phrase
//! "primitive output" in a test that had been renamed. A gate that reads for
//! spellings reports refactors as regressions and cannot see a real fork that
//! happens to keep the old words. The wrapper set, the primitive each wrapper
//! wraps, and the reference functions each primitive publishes are all read from
//! the tree at run time, so a new wrapper is judged without editing this file.
//!
//! What these rules do not catch: a wrapper that names its primitive and then
//! reimplements the algorithm beside the call. Behavioural parity against the
//! primitive is asserted by each wrapper's own test module, which compares
//! dispatch output with the primitive reference; these rules exist to make sure
//! such a comparison has something to compare against.

use std::fs;
use std::path::{Path, PathBuf};

/// A dispatch wrapper: a directory under `src/graph/dispatch/` whose name is
/// also a module in `vyre-primitives/src/graph/`. Wrapping a primitive of the
/// same name is what makes it a wrapper rather than dispatch infrastructure such
/// as `dispatch_bridge` or the CSR frontier-queue plumbing.
struct Wrapper {
    name: String,
    /// Every `.rs` file the wrapper owns, concatenated.
    source: String,
    /// The primitive module's source, file or directory form.
    primitive_source: String,
}

/// The count below which the derivation is presumed broken rather than the tree
/// empty. Eleven wrappers exist; a derivation that suddenly finds two would make
/// every rule here vacuous while still passing.
const WRAPPER_FLOOR: usize = 11;

#[test]
fn the_wrapper_set_is_derived_and_not_empty() {
    let wrappers = wrappers();
    assert!(
        wrappers.len() >= WRAPPER_FLOOR,
        "Fix: only {} graph dispatch wrappers were derived, below the floor of {WRAPPER_FLOOR}; the pairing between `vyre-libs/src/graph/dispatch/<name>` and `vyre-primitives/src/graph/<name>` broke, and every rule in this file would otherwise pass by judging nothing",
        wrappers.len()
    );
}

#[test]
fn every_dispatch_wrapper_names_the_graph_primitive_it_wraps() {
    let mut failures = Vec::new();
    for wrapper in wrappers() {
        let path = format!("vyre_libs::graph::{}", wrapper.name);
        if !wrapper.source.contains(&path) {
            failures.push(format!(
                "{} never names {path}, so it dispatches an algorithm it does not delegate",
                wrapper.name
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "Fix: a graph dispatch wrapper delegates its algorithm to the primitive of the same name:\n{}",
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
        for stale in [
            crate_root.join("src").join(format!("{}.rs", wrapper.name)),
            crate_root
                .join("src/graph")
                .join(format!("{}.rs", wrapper.name)),
        ] {
            if stale.exists() {
                failures.push(format!(
                    "{} also exists; a dispatch wrapper lives only under src/graph/dispatch/",
                    stale.display()
                ));
            }
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

/// Every dispatch wrapper, paired with the primitive module it wraps.
fn wrappers() -> Vec<Wrapper> {
    let crate_root = crate_root();
    let primitives = crate_root
        .parent()
        .expect("Fix: vyre-libs must live under the workspace root")
        .join("vyre-primitives/src/graph");
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
        let module_file = primitives.join(format!("{name}.rs"));
        let module_dir = primitives.join(&name);
        let primitive_source = if module_file.is_file() {
            read(&module_file)
        } else if module_dir.is_dir() {
            concatenate(&module_dir)
        } else {
            continue;
        };
        wrappers.push(Wrapper {
            name,
            source: concatenate(&entry),
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
