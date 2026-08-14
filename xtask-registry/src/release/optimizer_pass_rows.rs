//! The optimizer catalog row published by the optimization evidence generators.
//!
//! The integration matrix and the corpus pass manifest publish the same row for
//! every catalog entry. The matrix adds the semantic input and output contract
//! around it; the row itself, and the catalog read that produces it, have one
//! owner here so the two artifacts can never drift apart.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use vyre_foundation::optimizer::pass_catalog::{
    optimization_catalog, OptimizationCatalogEntry, OptimizationCatalogEntryKind,
};
use vyre_foundation::optimizer::{registered_pass_registrations, PassMetadata};

#[derive(Debug, Serialize)]
pub(crate) struct OptimizerPassRow {
    pub(crate) id: String,
    kind: &'static str,
    pub(crate) owner: &'static str,
    phase: String,
    boundary: String,
    requires: Vec<&'static str>,
    invalidates: Vec<&'static str>,
    capabilities: Vec<&'static str>,
    preserves_abi: bool,
    pub(crate) invariant: &'static str,
    termination: &'static str,
    pub(crate) proof: &'static str,
    pub(crate) benchmark: &'static str,
}

/// Every catalog entry as a publishable row, paired with the number of passes the
/// registry actually schedules. Exits with the registry or catalog error when
/// either refuses to resolve, because evidence built on a partial catalog is
/// worse than no evidence.
pub(crate) fn collect() -> (usize, Vec<OptimizerPassRow>) {
    let registrations = registered_pass_registrations().unwrap_or_else(|error| {
        eprintln!("Fix: semantic optimizer registry must schedule: {error}");
        std::process::exit(1);
    });
    let metadata = registrations
        .iter()
        .map(|registration| (registration.metadata.name, registration.metadata))
        .collect::<BTreeMap<_, _>>();
    let catalog = optimization_catalog().unwrap_or_else(|error| {
        eprintln!("Fix: semantic optimizer catalog must resolve: {error}");
        std::process::exit(1);
    });
    let rows = catalog
        .iter()
        .map(|entry| row(entry, metadata.get(entry.name).copied()))
        .collect::<Vec<_>>();
    (registrations.len(), rows)
}

/// The blocker for a catalog that names the same pass or rule twice. An id is the
/// only handle the published evidence offers, so a repeated one makes every row
/// under it ambiguous.
pub(crate) fn duplicate_id_blockers(rows: &[OptimizerPassRow]) -> Vec<String> {
    let unique = rows
        .iter()
        .map(|row| row.id.as_str())
        .collect::<BTreeSet<_>>();
    if unique.len() == rows.len() {
        return Vec::new();
    }
    vec!["optimizer catalog contains duplicate pass or rule ids".to_string()]
}

fn row(entry: &OptimizationCatalogEntry, metadata: Option<PassMetadata>) -> OptimizerPassRow {
    OptimizerPassRow {
        id: entry.name.to_string(),
        kind: match entry.kind {
            OptimizationCatalogEntryKind::ExecutablePass => "executable-pass",
            OptimizationCatalogEntryKind::SupplementalRule => "supplemental-rule",
        },
        owner: entry.owner,
        phase: format!("{:?}", entry.phase),
        boundary: format!("{:?}", entry.boundary_class),
        requires: metadata.map_or_else(Vec::new, |row| row.requires.to_vec()),
        invalidates: metadata.map_or_else(Vec::new, |row| row.invalidates.to_vec()),
        capabilities: entry.requires_caps.to_vec(),
        preserves_abi: entry.preserves_abi,
        invariant: entry.invariant,
        termination: match entry.kind {
            OptimizationCatalogEntryKind::ExecutablePass => {
                "bounded by the registered scheduler restart and iteration budgets"
            }
            OptimizationCatalogEntryKind::SupplementalRule => {
                "bounded by its owning registered executable pass"
            }
        },
        proof: match entry.kind {
            OptimizationCatalogEntryKind::ExecutablePass => {
                "optimizer::pass_invariants::audit_registered_passes plus semantic differential fixtures"
            }
            OptimizationCatalogEntryKind::SupplementalRule => {
                "owning pass differential and invariant fixtures"
            }
        },
        benchmark: entry.benchmark,
    }
}
