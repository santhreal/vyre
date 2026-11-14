//! The oracle is unreachable from production routing, derived from manifests.
//!
//! `vyre-reference` is the only CPU execution path in the workspace. Reaching
//! it from a routing decision, from a compiled artifact, from autoroute, or
//! from the backend registry turns a GPU regression into a green run: the
//! answer is right and no device produced it.
//!
//! The reverse-dependency closure is computed from the workspace manifests at
//! run time. A crate that adds `vyre-reference` to `[dependencies]` or
//! `[build-dependencies]` joins the closure on the next run and fails this
//! contract until it is recorded, so a new production route cannot arrive
//! quietly. A dev-dependency is a parity test and is not a route.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;

/// Crates permitted to link the oracle outside a dev-dependency.
///
/// Every one is an oracle surface: the reference driver the conformance
/// harness drives, the conformance harness itself, the parity fixtures, the
/// benchmark comparator, and the registry inspector. None of them is on a
/// routing, artifact, or autoroute path.
const RECORDED_ORACLE_LINKERS: &[&str] = &[
    "vyre-driver-reference",
    "vyre-test-support",
    "vyre-bench",
];

/// Crates that decide where work runs or what a device is handed.
///
/// The oracle must be absent from the closure above each of these, whatever
/// path a new dependency edge takes to get there.
const PRODUCTION_ROUTING_CRATES: &[&str] = &[
    "vyre",
    "vyre-compiler",
    "vyre-runtime",
    "vyre-driver-metal",
    "vyre-driver-wgpu",
    "vyre-driver-spirv",
    "vyre-driver-cuda",
    "vyre-compiler",
    "vyre-aot",
    "vyre-compiler",
    "vyre-emit",
    "vyre-registry-link",
];

#[test]
fn only_recorded_oracle_surfaces_link_the_oracle() {
    let manifests = workspace_manifests();
    let recorded: BTreeSet<&str> = RECORDED_ORACLE_LINKERS
        .iter()
        .copied()
        .filter(|name| manifests.contains_key(*name))
        .collect();
    let linkers: BTreeSet<&str> = manifests
        .iter()
        .filter(|(_, deps)| deps.contains("vyre-reference"))
        .map(|(name, _)| name.as_str())
        .collect();

    let unrecorded: Vec<&&str> = linkers.difference(&recorded).collect();
    assert!(
        unrecorded.is_empty(),
        "Fix: {unrecorded:?} link the parity oracle outside a dev-dependency. The oracle is the \
         only CPU execution path in the workspace; move the dependency to \
         `[dev-dependencies]`, or record the crate in RECORDED_ORACLE_LINKERS with the reason it \
         is an oracle surface."
    );

    let departed: Vec<&&str> = recorded.difference(&linkers).collect();
    assert!(
        departed.is_empty(),
        "Fix: {departed:?} are recorded as oracle surfaces but no longer link the oracle. Remove \
         them from RECORDED_ORACLE_LINKERS so the recorded set stays the real one."
    );
}

#[test]
fn no_production_routing_crate_reaches_the_oracle() {
    let manifests = workspace_manifests();

    let mut violations = Vec::new();
    let active_routing: Vec<&str> = PRODUCTION_ROUTING_CRATES
        .iter()
        .copied()
        .filter(|name| manifests.contains_key(*name))
        .collect();
    assert!(
        !active_routing.is_empty(),
        "Fix: none of PRODUCTION_ROUTING_CRATES matched a workspace member."
    );
    for crate_name in active_routing {
        if let Some(path) = oracle_path(&manifests, crate_name) {
            violations.push(format!("  {}", path.join(" -> ")));
        }
    }

    assert!(
        violations.is_empty(),
        "Fix: a production routing crate reaches the parity oracle. Break the edge; a routing \
         path that can execute on the CPU reports a correct answer no device produced:\n{}",
        violations.join("\n")
    );
}

#[test]
fn the_oracle_declares_no_backend_and_no_driver() {
    let manifests = workspace_manifests();
    let deps = manifests
        .get("vyre-reference")
        .expect("Fix: `vyre-reference` must be a workspace member");

    let backends: Vec<&String> = deps
        .iter()
        .filter(|dep| dep.starts_with("vyre-runtime") || dep.starts_with("vyre-emit"))
        .collect();

    assert!(
        backends.is_empty(),
        "Fix: the oracle declares {backends:?}. It grades a backend and must not link one; the \
         dependency runs the other way."
    );
}

/// The shortest non-dev dependency path from `start` to the oracle.
fn oracle_path<'a>(
    manifests: &'a BTreeMap<String, BTreeSet<String>>,
    start: &'a str,
) -> Option<Vec<&'a str>> {
    let mut queue = std::collections::VecDeque::from([vec![start]]);
    let mut seen = BTreeSet::from([start]);
    while let Some(path) = queue.pop_front() {
        let tail = *path.last().expect("Fix: a queued path is never empty");
        let Some(deps) = manifests.get(tail) else {
            continue;
        };
        for dep in deps {
            if dep == "vyre-reference" {
                let mut found = path.clone();
                found.push("vyre-reference");
                return Some(found);
            }
            if !seen.insert(dep.as_str()) {
                continue;
            }
            let mut next = path.clone();
            next.push(dep.as_str());
            queue.push_back(next);
        }
    }
    None
}

/// Every workspace member's package name and its non-dev vyre dependencies.
fn workspace_manifests() -> BTreeMap<String, BTreeSet<String>> {
    let mut manifests = BTreeMap::new();
    for (name, directory) in vyre_test_support::monorepo::vyre_member_directories() {
        let Ok(text) = fs::read_to_string(directory.join("Cargo.toml")) else {
            continue;
        };
        manifests.insert(name, runtime_dependencies(&text));
    }

    assert!(
        manifests.len() >= 10,
        "Fix: only {} workspace manifest(s) parsed. The member list moved and this contract is \
         covering nothing.",
        manifests.len()
    );
    manifests
}

/// Every vyre crate a manifest names in `[dependencies]` or `[build-dependencies]`.
fn runtime_dependencies(manifest: &str) -> BTreeSet<String> {
    let mut deps = BTreeSet::new();
    let mut runtime_section = false;
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            let section = trimmed.trim_matches(|c| c == '[' || c == ']');
            runtime_section = matches!(section, "dependencies" | "build-dependencies")
                || (section.ends_with(".dependencies")
                    && !section.contains("dev-dependencies")
                    && section.starts_with("target."));
            continue;
        }
        if !runtime_section {
            continue;
        }
        let Some(name) = dependency_name(trimmed) else {
            continue;
        };
        if name.starts_with("vyre") {
            deps.insert(name.to_owned());
        }
    }
    deps
}

/// The dependency name a manifest line declares.
fn dependency_name(line: &str) -> Option<&str> {
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let key = line.split(['=', '.']).next()?.trim();
    (!key.is_empty()
        && key
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'))
    .then_some(key)
}
