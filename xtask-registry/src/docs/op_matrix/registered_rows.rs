//! The matrix rows derived from the live operation registry.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use structure_gate::source_scan::{carries_rust_source, source_directory_named};
use vyre_foundation::operation::OperationTier as OpTier;

use super::record::OpRecord;

/// Every registered op that produced a row, recording the ones that could not.
///
/// A registration problem used to abort the whole matrix. It now excludes that
/// one id and is reported, so the remaining rows are still checked.
///
/// `root` is the checkout the owner directories are read from. Reading them
/// through relative paths answered from whatever directory the process happened
/// to start in, so the same registry produced different owners for the same
/// tree.
pub(super) fn registered_records(root: &Path, problems: &mut Vec<String>) -> Vec<OpRecord> {
    let mut ids = BTreeMap::<String, BTreeSet<String>>::new();
    let mut tiers = BTreeMap::<String, OpTier>::new();

    let registry = vyre_registry_link::operation::live_operation_registry();
    for entry in registry.iter() {
        push_registered(&mut ids, entry.id, "vyre-foundation::operation", problems);
        tiers.insert(entry.id.to_string(), entry.tier);
    }

    // One directory search per namespace and domain, not per operation. The
    // fallback walks a crate source tree, and 327 registrations share 40-odd
    // domains, so the uncached reader walked the same trees hundreds of times.
    let mut owners = BTreeMap::<String, String>::new();
    ids.into_iter()
        .filter_map(|(id, sources)| {
            let tier = tiers.get(&id).copied().unwrap_or(OpTier::Unknown);
            match record_for_registered_id(root, &mut owners, &id, tier, sources) {
                Ok(record) => Some(record),
                Err(problem) => {
                    problems.push(problem);
                    None
                }
            }
        })
        .collect()
}

/// Every op id the live registry declares.
///
/// The matrix is checked against this rather than against the rows it just
/// built, so a row that names something the registry never registered is a
/// blocker instead of an unremarkable line in a generated document.
pub(super) fn live_operation_ids() -> BTreeSet<&'static str> {
    vyre_registry_link::operation::live_operation_registry()
        .iter()
        .map(|entry| entry.id)
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
    root: &Path,
    owners: &mut BTreeMap<String, String>,
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
        owners: owner_paths(root, owners, id, tier)?,
        ops: vec![id.to_string()],
        duplicate_ok: sources.len() > 1,
        registry_sources: sources.into_iter().collect(),
        reference: "supported",
        foundation_ir: "supported",
        cuda: "supported",
        wgpu: "supported",
        spirv: "experimental",
        release_blocking_notes: release_notes(id, tier)?,
        tests: test_paths(id, tier)?,
    };

    Ok(record)
}

/// Directory that carries the definition, for every tier that has one.
///
/// Both tiers read the checkout through [`namespace_source_dir`]. A frozen id
/// names the crate that minted it, so a domain alias keyed on the id namespace
/// answered `vyre-libs/src/unknown` for every composition that kept a
/// `vyre-primitives::` id.
///
/// An external extension is defined by a crate outside this checkout, so no
/// directory here owns it, and a foundation op is defined by the IR rather than
/// by a domain directory.
///
/// # Errors
///
/// `OperationTier` is `non_exhaustive`, so a tier this rule does not name is
/// reported as the id that carries it rather than taking the empty owner list a
/// foundation op gets. An empty list reads as "defined by the IR", which is the
/// one answer that would let a new tier ship with no owner and no finding.
fn owner_paths(
    root: &Path,
    owners: &mut BTreeMap<String, String>,
    id: &str,
    tier: OpTier,
) -> Result<Vec<String>, String> {
    match tier {
        OpTier::Intrinsic | OpTier::Library => Ok(vec![namespace_source_dir(root, owners, id)]),
        OpTier::Foundation | OpTier::External | OpTier::Unknown => Ok(Vec::new()),
        tier => Err(format!(
            "Fix: op id `{id}` declares tier `{tier:?}`, which reaches no owner rule. Record where a registration of that tier keeps its definition, in `owner_paths` of the op matrix rows."
        )),
    }
}

/// `vyre-libs::graph::toposort` becomes the directory that carries the
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
///
/// A domain is not always a top-level module. The optimizer and quantization
/// ops moved under `nn`, so the search runs over the whole crate source tree
/// through [`source_directory_named`] once the top-level answer holds no code.
fn namespace_source_dir(root: &Path, owners: &mut BTreeMap<String, String>, id: &str) -> String {
    let Some((crate_name, rest)) = id.split_once("::") else {
        return String::new();
    };
    let domain = rest.split("::").next().unwrap_or("unknown");
    let key = format!("{crate_name}/{domain}");
    if let Some(found) = owners.get(&key) {
        return found.clone();
    }
    let found = resolve_source_dir(root, crate_name, domain);
    owners.insert(key, found.clone());
    found
}

/// Read one namespace and domain out of the checkout.
fn resolve_source_dir(root: &Path, crate_name: &str, domain: &str) -> String {
    let minted = format!("{crate_name}/src/{domain}");
    if carries_rust_source(&root.join(&minted)) {
        return minted;
    }
    let moved = format!("vyre-libs/src/{domain}");
    if carries_rust_source(&root.join(&moved)) {
        return moved;
    }
    for relative in [format!("{crate_name}/src"), "vyre-libs/src".to_string()] {
        if let Some(found) = source_directory_named(&root.join(&relative), domain) {
            let found = found.to_string_lossy().replace('\\', "/");
            let root_prefix = format!("{}/", root.to_string_lossy().replace('\\', "/"));
            return found
                .strip_prefix(&root_prefix)
                .unwrap_or(&found)
                .to_string();
        }
    }
    minted
}

fn namespace_domain<'a>(id: &'a str, prefix: &str) -> &'a str {
    id.strip_prefix(prefix)
        .and_then(|rest| rest.split("::").next())
        .unwrap_or("unknown")
}

/// Suites that judge one operation, per tier.
///
/// # Errors
///
/// `OperationTier` is `non_exhaustive`. A tier this rule does not name reaches
/// no suite, and an empty list here reads as an operation the harnesses already
/// cover, so the tier is reported instead. Naming a new tier in one field rule
/// and not the others is exactly how a row ships judged by nothing.
fn test_paths(id: &str, tier: OpTier) -> Result<Vec<String>, String> {
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
        tier => {
            return Err(format!(
                "Fix: op id `{id}` declares tier `{tier:?}`, which reaches no suite rule. Record which suite judges a registration of that tier, in `test_paths` of the op matrix rows."
            ));
        }
    };
    tests.push("conform/vyre-conform/tests/op_matrix_truth/mod.rs".to_string());
    Ok(tests)
}

/// What the row says about the tier it was generated from.
///
/// # Errors
///
/// A tier this rule does not name is reported rather than described by an empty
/// sentence, for the reason [`test_paths`] gives.
fn release_notes(id: &str, tier: OpTier) -> Result<String, String> {
    match tier {
        OpTier::Intrinsic => Ok(
            "Source-backed row generated from the Category C operation catalog; a hardware-intrinsic id stays in its owning crate's namespace and passes hardware_conform.".to_string()
        ),
        OpTier::Library => Ok(
            "Source-backed row generated from vyre-foundation::operation; library ids must stay in the vyre-libs namespace.".to_string()
        ),
        OpTier::External => Ok(
            "Source-backed row generated from vyre-foundation::operation for an external consumer crate.".to_string()
        ),
        OpTier::Foundation | OpTier::Unknown => Ok(String::new()),
        tier => Err(format!(
            "Fix: op id `{id}` declares tier `{tier:?}`, which reaches no release-note rule. Record what a row of that tier states, in `release_notes` of the op matrix rows."
        )),
    }
}
