//! The facade's re-export surface against the partition crates that own it.
//!
//! WHY: `vyre-libs` is a facade. It defines no composition; it re-exports the
//! `vyre-libs-<domain>` crates that do. That makes exactly two things the
//! facade can prove on its own, and neither is provable from a copy of a
//! domain crate's assertions: that every module a partition crate publishes is
//! reachable through `vyre_libs::`, and that every partition crate is
//! referenced by `link_anchor` so the linker retains its operation
//! registrations. A module the facade forgets is a composition no consumer of
//! the facade can call. A crate `link_anchor` forgets registers nothing, and
//! the catalog it should have populated comes back short with nothing red.
//!
//! The roster is derived from `workspace.members` at run time and each crate's
//! published module list is read from its own `src/lib.rs`, so a partition
//! crate added tomorrow, or a module added to one, is judged tomorrow and
//! fails until the facade carries it. A hardcoded list would go stale in
//! silence, which is the same failure as having no test.
//!
//! Resolution is proven by the build rather than by the text: `pub use
//! vyre_libs_math::math;` cannot compile unless it names a real item in that
//! crate. What this file adds is that the re-export exists at all, and that it
//! points at the owning partition crate rather than at something the facade
//! defined itself.
//!
//! What this does not catch: a module that is re-exported and empty, or a
//! partition crate whose `link_anchor` body registers nothing. Reachability is
//! not coverage; `registry_closure` owns that question.

#![forbid(unsafe_code)]

use std::path::Path;

/// Minimum partition crates the roster walk must find.
///
/// A derivation that stops matching finds zero crates, and zero crates are
/// trivially all re-exported. The floor sits within reach of the real
/// population (22) so a broken walk fails instead of reporting a clean sweep
/// of an empty set, and it moves down only when a deliberate removal takes
/// crates with it.
const PARTITION_CRATE_FLOOR: usize = 20;

/// Minimum published modules the roster walk must find, for the same reason.
const PUBLISHED_MODULE_FLOOR: usize = 30;

/// Top-level modules `lib.rs` publishes, in declaration order.
fn published_modules(source: &str) -> Vec<String> {
    source
        .lines()
        .filter_map(|line| {
            line.strip_prefix("pub mod ")
                .and_then(|rest| rest.strip_suffix(';'))
        })
        .filter(|name| !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_'))
        .map(str::to_string)
        .collect()
}

/// Every `vyre-libs-<domain>` member of the workspace, as (crate name, path).
fn partition_crates(workspace_root: &Path) -> Vec<(String, std::path::PathBuf)> {
    let mut crates: Vec<(String, std::path::PathBuf)> =
        structure_gate::workspace_members(workspace_root)
            .into_iter()
            .filter_map(|member| {
                let name = member.rsplit('/').next().unwrap_or(member.as_str());
                name.starts_with("vyre-libs-")
                    .then(|| (name.to_string(), workspace_root.join(&member)))
            })
            .collect();
    crates.sort();
    crates
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|error| panic!("cannot read {path:?}: {error}"))
}

#[test]
fn the_facade_re_exports_every_module_its_partition_crates_publish() {
    let workspace_root = vyre_test_support::monorepo::vyre_workspace_root();
    let facade = read(&workspace_root.join("vyre-libs/src/lib.rs"));
    let crates = partition_crates(&workspace_root);

    let mut modules = 0usize;
    let mut missing = Vec::new();
    for (name, directory) in &crates {
        let owner = name.replace('-', "_");
        for module in published_modules(&read(&directory.join("src/lib.rs"))) {
            modules += 1;
            let expected = format!("pub use {owner}::{module};");
            if !facade.lines().any(|line| line.trim() == expected) {
                missing.push(format!("{name}::{module} (expected `{expected}`)"));
            }
        }
    }

    assert!(
        crates.len() >= PARTITION_CRATE_FLOOR,
        "the roster walk found only {} partition crates, below the {PARTITION_CRATE_FLOOR} floor: \
         the member derivation is broken, not the facade",
        crates.len()
    );
    assert!(
        modules >= PUBLISHED_MODULE_FLOOR,
        "the roster walk found only {modules} published modules, below the \
         {PUBLISHED_MODULE_FLOOR} floor: the module derivation is broken, not the facade"
    );
    assert!(
        missing.is_empty(),
        "vyre-libs/src/lib.rs does not re-export {} module(s) its partition crates publish:\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}

#[test]
fn link_anchor_references_every_partition_crate() {
    let workspace_root = vyre_test_support::monorepo::vyre_workspace_root();
    let facade = read(&workspace_root.join("vyre-libs/src/lib.rs"));
    let crates = partition_crates(&workspace_root);

    let unanchored: Vec<String> = crates
        .iter()
        .map(|(name, _)| (name, format!("{}::link_anchor();", name.replace('-', "_"))))
        .filter(|(_, call)| !facade.contains(call.as_str()))
        .map(|(name, call)| format!("{name} (expected `{call}`)"))
        .collect();

    assert!(
        crates.len() >= PARTITION_CRATE_FLOOR,
        "the roster walk found only {} partition crates, below the {PARTITION_CRATE_FLOOR} floor",
        crates.len()
    );
    assert!(
        unanchored.is_empty(),
        "vyre_libs::link_anchor does not reference {} partition crate(s), so the linker may drop \
         their operation registrations:\n  {}",
        unanchored.len(),
        unanchored.join("\n  ")
    );
}

#[test]
fn the_active_feature_set_registers_operations_through_the_facade() {
    let registered = vyre_libs::link_anchor();
    assert!(
        registered > 0,
        "link_anchor reported {registered} registered library operations, so no partition crate's \
         registrations survived linking through the facade"
    );
}
