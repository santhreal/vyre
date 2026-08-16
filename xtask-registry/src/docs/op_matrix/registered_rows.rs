//! The matrix rows derived from the live operation registry.

use std::collections::{BTreeMap, BTreeSet};

use vyre_foundation::operation::{
    classify_operation_id as classify_op_id, OperationTier as OpTier,
};

use super::record::OpRecord;

/// Every registered op that produced a row, recording the ones that could not.
///
/// A registration problem used to abort the whole matrix. It now excludes that
/// one id and is reported, so the remaining rows are still checked.
pub(super) fn registered_records(problems: &mut Vec<String>) -> Vec<OpRecord> {
    let mut ids = BTreeMap::<String, BTreeSet<String>>::new();

    let registry = vyre_registry_link::operation::live_operation_registry();
    for entry in registry.iter() {
        push_registered(&mut ids, entry.id, "vyre-foundation::operation", problems);
    }

    ids.into_iter()
        .filter_map(
            |(id, sources)| match record_for_registered_id(&id, sources) {
                Ok(record) => Some(record),
                Err(problem) => {
                    problems.push(problem);
                    None
                }
            },
        )
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

fn record_for_registered_id(id: &str, sources: BTreeSet<String>) -> Result<OpRecord, String> {
    let tier = classify_op_id(id);
    if tier == OpTier::Unknown {
        return Err(format!(
            "Fix: op id `{id}` from `{sources:?}` has no canonical tier namespace."
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

fn owner_paths(id: &str, tier: OpTier) -> Vec<String> {
    match tier {
        OpTier::Intrinsic => vec![namespace_source_dir(id)],
        OpTier::Library => {
            let domain = namespace_domain(id, "vyre-libs::");
            let owner = match domain {
                "matching" => "vyre-libs/src/scan".to_string(),
                "optim" => "vyre-libs/src/nn/optim".to_string(),
                "quant" => "vyre-libs/src/nn/quant".to_string(),
                "builder" => "vyre-libs/src/builder/registrations.rs".to_string(),
                _ => format!("vyre-libs/src/{domain}"),
            };
            vec![owner]
        }
        OpTier::External => vec!["docs/optimization/README.md".to_string()],
        OpTier::Foundation | OpTier::Unknown => Vec::new(),
        _ => Vec::new(),
    }
}

/// `vyre-primitives::graph::toposort` becomes `vyre-primitives/src/graph`.
///
/// Read from the id so a namespace move carries its owner path with it instead
/// of leaving a second hardcoded crate name in this generator.
fn namespace_source_dir(id: &str) -> String {
    match id.split_once("::") {
        Some((crate_name, rest)) => {
            let domain = rest.split("::").next().unwrap_or("unknown");
            format!("{crate_name}/src/{domain}")
        }
        None => String::new(),
    }
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
