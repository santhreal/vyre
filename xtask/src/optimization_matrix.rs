//! Generate source-owned optimization integration evidence.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::Serialize;
use vyre_foundation::optimizer::pass_catalog::{
    optimization_catalog, OptimizationCatalogEntryKind,
};
use vyre_foundation::optimizer::{registered_pass_registrations, PassMetadata};

#[derive(Debug, Serialize)]
struct OptimizationMatrix {
    schema_version: u32,
    architecture: &'static str,
    semantic_optimizer_owner: &'static str,
    verified_lowering_owner: &'static str,
    target_strategy_owner: &'static str,
    executable_passes: usize,
    catalog_entries: usize,
    entries: Vec<OptimizationMatrixEntry>,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct OptimizationMatrixEntry {
    id: String,
    kind: &'static str,
    owner: &'static str,
    phase: String,
    boundary: String,
    input: &'static str,
    output: &'static str,
    requires: Vec<&'static str>,
    invalidates: Vec<&'static str>,
    capabilities: Vec<&'static str>,
    preserves_abi: bool,
    invariant: &'static str,
    termination: &'static str,
    proof: &'static str,
    benchmark: &'static str,
}

pub(crate) fn run(args: &[String]) {
    let output = match parse_output(args) {
        Ok(output) => output,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
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
    let entries = catalog
        .iter()
        .map(|entry| matrix_entry(entry, metadata.get(entry.name).copied()))
        .collect::<Vec<_>>();
    let mut blockers = Vec::new();
    let unique = entries
        .iter()
        .map(|entry| entry.id.as_str())
        .collect::<BTreeSet<_>>();
    if unique.len() != entries.len() {
        blockers.push("optimizer catalog contains duplicate pass or rule ids".to_string());
    }
    if entries.iter().any(|entry| {
        entry.owner.is_empty()
            || entry.invariant.is_empty()
            || entry.proof.is_empty()
            || entry.benchmark.is_empty()
    }) {
        blockers.push(
            "optimizer catalog contains an entry without owner, invariant, proof, or benchmark"
                .to_string(),
        );
    }
    let matrix = OptimizationMatrix {
        schema_version: 2,
        architecture: "semantic Program optimizer -> verified representation lowering -> concrete target strategy",
        semantic_optimizer_owner: "vyre-foundation",
        verified_lowering_owner: "vyre-lower",
        target_strategy_owner: "concrete emitters and drivers",
        executable_passes: registrations.len(),
        catalog_entries: entries.len(),
        entries,
        blockers,
    };
    crate::output_arg::write_json(&output, &matrix);
    println!("optimization-matrix: wrote {}", output.display());
    if !matrix.blockers.is_empty() {
        std::process::exit(1);
    }
}

fn matrix_entry(
    entry: &vyre_foundation::optimizer::pass_catalog::OptimizationCatalogEntry,
    metadata: Option<PassMetadata>,
) -> OptimizationMatrixEntry {
    let kind = match entry.kind {
        OptimizationCatalogEntryKind::ExecutablePass => "executable-pass",
        OptimizationCatalogEntryKind::SupplementalRule => "supplemental-rule",
    };
    OptimizationMatrixEntry {
        id: entry.name.to_string(),
        kind,
        owner: entry.owner,
        phase: format!("{:?}", entry.phase),
        boundary: format!("{:?}", entry.boundary_class),
        input: "vyre-foundation Program",
        output: "semantically equivalent vyre-foundation Program",
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

fn parse_output(args: &[String]) -> Result<PathBuf, String> {
    crate::output_arg::parse_output_arg(
        args,
        "optimization-matrix",
        "Writes source-owned semantic optimizer integration evidence.",
        default_output,
    )
}

fn default_output() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|path| path.join("release/evidence/optimization/optimization-integration-matrix.json"))
        .unwrap_or_else(|| {
            PathBuf::from("release/evidence/optimization/optimization-integration-matrix.json")
        })
}
