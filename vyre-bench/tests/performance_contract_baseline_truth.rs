//! Every performance contract names the crate its CPU baseline runs from, and that
//! crate must exist in this checkout.
//!
//! WHY: contracts carried competitor names nothing here could run. `bigint.modexp.4096`
//! claimed `rug` (GMP) while timing an in-tree u32-limb loop, `interpreter.bytecode.
//! dispatch.10m` claimed a hand-tuned C threaded interpreter with computed goto while
//! timing a Rust match-dispatch loop, and two more rows spelled prose into the crate
//! field ("std+rayon", "OpenSSL EVP AES-128-CTR"). A reader treated the speedup as a
//! win over the named engine. Nothing compared the label to the timed code, so the
//! claim survived every gate.
//!
//! The roster is the registry and both name sets are read from manifests at run time,
//! so a new case or a renamed dependency is judged rather than assumed. What this
//! cannot catch: a contract that names a real dependency it does not call. That is
//! what the per-case reference implementation and the parity oracles are for.

use std::collections::BTreeSet;

use vyre_bench::api::case::BaselineClass;

fn workspace_root() -> std::path::PathBuf {
    vyre_test_support::monorepo::vyre_workspace_root()
}

fn manifest(path: &str) -> toml::Value {
    let text = std::fs::read_to_string(workspace_root().join(path))
        .unwrap_or_else(|error| panic!("Fix: {path} must be readable: {error}"));
    toml::from_str(&text).unwrap_or_else(|error| panic!("Fix: {path} must parse as TOML: {error}"))
}

/// Crate names a baseline may legitimately claim: every workspace member, plus every
/// dependency the benchmark crate declares. Both are read from manifests so the set
/// tracks the tree instead of a list somebody has to remember to update.
fn resolvable_crate_names() -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let root = manifest("Cargo.toml");
    let members = root
        .get("workspace")
        .and_then(|workspace| workspace.get("members"))
        .and_then(toml::Value::as_array)
        .expect("Fix: the root manifest must declare workspace members.");
    for member in members.iter().filter_map(toml::Value::as_str) {
        let member_manifest = manifest(&format!("{member}/Cargo.toml"));
        if let Some(name) = member_manifest
            .get("package")
            .and_then(|package| package.get("name"))
            .and_then(toml::Value::as_str)
        {
            names.insert(name.to_string());
        }
    }
    let bench = manifest("vyre-bench/Cargo.toml");
    for table in ["dependencies", "dev-dependencies", "build-dependencies"] {
        if let Some(deps) = bench.get(table).and_then(toml::Value::as_table) {
            names.extend(deps.keys().cloned());
        }
    }
    if let Some(targets) = bench.get("target").and_then(toml::Value::as_table) {
        for target in targets.values() {
            for table in ["dependencies", "dev-dependencies"] {
                if let Some(deps) = target.get(table).and_then(toml::Value::as_table) {
                    names.extend(deps.keys().cloned());
                }
            }
        }
    }
    names
}

#[test]
fn every_cpu_baseline_names_a_crate_this_checkout_can_run() {
    let resolvable = resolvable_crate_names();
    assert!(
        resolvable.contains("vyre-bench") && resolvable.contains("pcre2"),
        "Fix: the resolvable crate set must include workspace members and benchmark dependencies; it holds {} name(s).",
        resolvable.len()
    );
    let registry = vyre_bench::registry::collect_all();
    let mut contracts = 0usize;
    let mut failures = Vec::new();
    for case in registry.iter() {
        let Some(contract) = case.performance_contract() else {
            continue;
        };
        for baseline in &contract.baselines {
            if !matches!(baseline.class, BaselineClass::CpuSota) {
                continue;
            }
            contracts += 1;
            if baseline.crate_name.is_empty() {
                failures.push(format!(
                    "Fix: case `{}` declares a CPU baseline with no crate name.",
                    case.id().0
                ));
                continue;
            }
            if !resolvable.contains(&baseline.crate_name) {
                failures.push(format!(
                    "Fix: case `{}` claims CPU baseline crate `{}`, which is neither a workspace member nor a vyre-bench dependency. Name the crate the timed baseline runs from, or add the dependency and call it.",
                    case.id().0,
                    baseline.crate_name
                ));
            }
        }
    }
    assert!(
        contracts >= 20,
        "Fix: the registry exposed only {contracts} CPU baseline contract(s); a filter this narrow cannot judge the class."
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
