//! The matrix rows derived from the live operation registry.

use std::collections::{BTreeMap, BTreeSet};

use vyre_foundation::operation::OperationTier as OpTier;

use super::record::OpRecord;

/// Every registered op that produced a row, recording the ones that could not.
///
/// A registration problem used to abort the whole matrix. It now excludes that
/// one id and is reported, so the remaining rows are still checked.
pub(super) fn registered_records(problems: &mut Vec<String>) -> Vec<OpRecord> {
    let mut ids = BTreeMap::<String, BTreeSet<String>>::new();
    let mut tiers = BTreeMap::<String, OpTier>::new();

    let registry = vyre_registry_link::operation::live_operation_registry();
    for entry in registry.iter() {
        push_registered(&mut ids, entry.id, "vyre-foundation::operation", problems);
        tiers.insert(entry.id.to_string(), entry.tier);
    }

    ids.into_iter()
        .filter_map(|(id, sources)| {
            let tier = tiers.get(&id).copied().unwrap_or(OpTier::Unknown);
            match record_for_registered_id(&id, tier, sources) {
                Ok(record) => Some(record),
                Err(problem) => {
                    problems.push(problem);
                    None
                }
            }
        })
        .collect()
}

fn push_registered(
    ids: &mut BTreeMap<String, BTreeSet<String>>,
    id: &str,
    source: &str,
    problems: &mut Vec<String>,
) {
    let sources = ids.entry(id.to_string()).or_default();
    if !sources.insert(source.to_string()) {
        problems.push(format!(
            "Fix: duplicate op id `{id}` registered more than once by `{source}`. \
             Keep one canonical registration in that registry."
        ));
    }
}

fn record_for_registered_id(
    id: &str,
    tier: OpTier,
    sources: BTreeSet<String>,
) -> Result<OpRecord, String> {
    if tier == OpTier::Unknown {
        return Err(format!(
            "Fix: op id `{id}` from `{sources:?}` declares no tier."
        ));
    }
    if sources.len() > 1 {
        return Err(format!(
            "Fix: op id `{id}` is registered by multiple semantic sources {sources:?}."
        ));
    }

    let record = OpRecord {
        family: id.to_string(),
        tier,
        owners: owner_paths(id, tier),
        ops: vec![id.to_string()],
        duplicate_ok: sources.len() > 1,
        registry_sources: sources.into_iter().collect(),
        reference: "supported",
        foundation_ir: "supported",
        cuda: "supported",
        wgpu: "supported",
        spirv: "experimental",
        release_blocking_notes: release_notes(id, tier),
        tests: test_paths(id, tier),
        bench_targets: Vec::new(),
    };

    Ok(record)
}

/// Directory that carries the definition, for every tier that has one.
///
/// Both tiers read the checkout through [`namespace_source_dir`]. A frozen id
/// names the crate that minted it, so a domain alias keyed on the id namespace
/// answered `vyre-libs/src/unknown` for every composition that kept a
/// `vyre-primitives::` id.
fn owner_paths(id: &str, tier: OpTier) -> Vec<String> {
    match tier {
        OpTier::Intrinsic | OpTier::Library => vec![namespace_source_dir(id)],
        OpTier::External => vec!["docs/optimization/README.md".to_string()],
        OpTier::Foundation | OpTier::Unknown => Vec::new(),
        _ => Vec::new(),
    }
}

/// `vyre-primitives::graph::toposort` becomes the directory that carries the
/// code today, `vyre-libs/src/graph`.
///
/// Operation ids are frozen, so the id names the crate an operation was minted
/// under and not the crate it lives in: the composition move left 154 ids whose
/// namespace crate no longer carries the code for their domain. The owner is
/// therefore read from the checkout, which is the only thing that knows, and the
/// id supplies just the domain. When neither crate carries the domain the
/// id-derived path is kept, so `vyre-conform` `op_matrix_truth` reports the row
/// instead of this generator inventing a plausible directory.
///
/// The question is whether a directory carries Rust source, not whether it
/// exists: deleting every file in a directory leaves the directory behind, since
/// git tracks files and not directories, and the move left one such shell at
/// `vyre-primitives/src/matching`. An existence test would have named it the
/// owner of eleven operations whose code is in `vyre-libs`.
fn namespace_source_dir(id: &str) -> String {
    let Some((crate_name, rest)) = id.split_once("::") else {
        return String::new();
    };
    let domain = rest.split("::").next().unwrap_or("unknown");
    let minted = format!("{crate_name}/src/{domain}");
    if carries_rust_source(&minted) {
        return minted;
    }
    let moved = format!("vyre-libs/src/{domain}");
    if carries_rust_source(&moved) {
        return moved;
    }
    minted
}

/// Whether the directory holds a Rust file, at any depth below it.
fn carries_rust_source(dir: &str) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    let mut nested = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "rs") {
            return true;
        }
        if path.is_dir() {
            nested.push(path);
        }
    }
    nested
        .into_iter()
        .any(|path| path.to_str().is_some_and(carries_rust_source))
}

fn namespace_domain<'a>(id: &'a str, prefix: &str) -> &'a str {
    id.strip_prefix(prefix)
        .and_then(|rest| rest.split("::").next())
        .unwrap_or("unknown")
}

fn test_paths(id: &str, tier: OpTier) -> Vec<String> {
    let mut tests = match tier {
        OpTier::Intrinsic => {
            let crate_name = id.split_once("::").map_or("", |(name, _)| name);
            let domain = id.split("::").nth(1).unwrap_or("");
            if domain == "hardware" {
                vec![format!("{crate_name}/tests/hardware_conform.rs")]
            } else {
                vec![format!("{crate_name}/tests/integration.rs")]
            }
        }
        OpTier::Library | OpTier::External => {
            vec!["vyre-libs/tests/universal_harness.rs".to_string()]
        }
        OpTier::Foundation | OpTier::Unknown => Vec::new(),
        _ => Vec::new(),
    };
    tests.push("conform/vyre-conform/tests/op_matrix_truth.rs".to_string());
    tests
}

fn release_notes(_id: &str, tier: OpTier) -> String {
    match tier {
        OpTier::Intrinsic => {
            "Source-backed row generated from the Category C operation catalog; a hardware-intrinsic id stays in its owning crate's namespace and passes hardware_conform.".to_string()
        }
        OpTier::Library => {
            "Source-backed row generated from vyre-foundation::operation; library ids must stay in the vyre-libs namespace.".to_string()
        }
        OpTier::External => {
            "Source-backed row generated from vyre-foundation::operation for an external consumer crate.".to_string()
        }
        OpTier::Foundation | OpTier::Unknown => String::new(),
        _ => String::new(),
    }
}
