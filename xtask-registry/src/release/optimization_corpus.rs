//! Generate release evidence for the canonical semantic `Program` optimizer.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use vyre_foundation::optimizer::corpus::{
    generate_release_corpus, manifest_for, OptimizationCorpusCase, OptimizationCorpusManifest,
};
use vyre_foundation::optimizer::pass_catalog::{
    optimization_catalog, OptimizationCatalogEntryKind,
};
use vyre_foundation::optimizer::registered_pass_registrations;

const REQUIRED_FAMILIES: &[&str] = &[
    "scalar-algebra",
    "strength-reduction",
    "fusion-cse",
    "dead-code",
    "memory-dataflow",
    "loop-transform",
    "control-flow",
    "canonicalization",
];
const MIN_CASES_PER_FAMILY: usize = 512;

#[derive(Debug, Serialize)]
struct OptimizationCorpusContracts {
    schema_version: u32,
    required_min_cases: usize,
    generated_cases: usize,
    verified_cases: usize,
    optimized_cases: usize,
    non_converged_cases: usize,
    total_nodes_before: usize,
    total_nodes_after: usize,
    pass_instance_count: usize,
    changed_pass_instances: usize,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct OptimizationFamilyManifest<'a> {
    schema_version: u32,
    required_family_count: usize,
    required_families: &'a [&'a str],
    missing_required_families: Vec<String>,
    families: &'a [vyre_foundation::optimizer::corpus::OptimizationFamilyCount],
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct OptimizationCaseManifest {
    schema_version: u32,
    required_min_cases: usize,
    generated_cases: usize,
    unique_case_ids: usize,
    duplicate_case_ids: Vec<String>,
    entries: Vec<OptimizationCaseManifestEntry>,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct OptimizationCaseManifestEntry {
    id: String,
    family: String,
    node_count: usize,
    instruction_count: u64,
    memory_op_count: u64,
    control_flow_count: u64,
    program_fingerprint: String,
}

#[derive(Debug, Serialize)]
struct OptimizerPassManifest {
    schema_version: u32,
    executable_passes: usize,
    catalog_entries: usize,
    entries: Vec<OptimizerPassManifestEntry>,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct OptimizerPassManifestEntry {
    id: String,
    kind: &'static str,
    owner: &'static str,
    phase: String,
    boundary: String,
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
    let cases = generate_release_corpus();
    let manifest = manifest_for(&cases);
    if manifest.generated_cases < manifest.required_min_cases
        || manifest.verified_cases != manifest.generated_cases
        || manifest.optimized_cases == 0
        || manifest.changed_pass_instances == 0
        || manifest.non_converged_cases != 0
        || !manifest.blockers.is_empty()
    {
        eprintln!(
            "optimization-corpus: semantic optimizer evidence is incomplete: generated={}, verified={}, optimized={}, changed_pass_instances={}, non_converged={}, blockers={}",
            manifest.generated_cases,
            manifest.verified_cases,
            manifest.optimized_cases,
            manifest.changed_pass_instances,
            manifest.non_converged_cases,
            manifest.blockers.len(),
        );
        for blocker in manifest.blockers.iter().take(20) {
            eprintln!("  - {blocker}");
        }
        std::process::exit(1);
    }
    let Some(parent) = output.parent() else {
        eprintln!(
            "Fix: optimization corpus output `{}` has no parent",
            output.display()
        );
        std::process::exit(1);
    };
    if let Err(error) = fs::create_dir_all(parent) {
        eprintln!("Fix: create `{}`: {error}", parent.display());
        std::process::exit(1);
    }
    xtask::output_arg::write_json(&output, &manifest);
    write_contracts(parent, &manifest);
    write_family_manifest(parent, &manifest);
    write_case_manifest(parent, &cases, &manifest);
    write_pass_manifest(parent);
    println!(
        "optimization-corpus: wrote {} semantic Program cases to {}",
        manifest.generated_cases,
        output.display()
    );
}

fn write_contracts(parent: &Path, manifest: &OptimizationCorpusManifest) {
    xtask::output_arg::write_json(
        &parent.join("optimization-corpus-contracts.json"),
        &OptimizationCorpusContracts {
            schema_version: 2,
            required_min_cases: manifest.required_min_cases,
            generated_cases: manifest.generated_cases,
            verified_cases: manifest.verified_cases,
            optimized_cases: manifest.optimized_cases,
            non_converged_cases: manifest.non_converged_cases,
            total_nodes_before: manifest.total_nodes_before,
            total_nodes_after: manifest.total_nodes_after,
            pass_instance_count: manifest.pass_instance_count,
            changed_pass_instances: manifest.changed_pass_instances,
            blockers: manifest.blockers.clone(),
        },
    );
}

fn write_family_manifest(parent: &Path, manifest: &OptimizationCorpusManifest) {
    let missing_required_families = REQUIRED_FAMILIES
        .iter()
        .filter(|required| {
            !manifest
                .families
                .iter()
                .any(|family| family.family == **required && family.cases >= MIN_CASES_PER_FAMILY)
        })
        .map(|required| (*required).to_string())
        .collect::<Vec<_>>();
    let mut blockers = Vec::new();
    if !missing_required_families.is_empty() {
        blockers.push(format!(
            "semantic optimizer corpus is missing required families: {}",
            missing_required_families.join(", ")
        ));
    }
    xtask::output_arg::write_json(
        &parent.join("optimization-family-manifest.json"),
        &OptimizationFamilyManifest {
            schema_version: 2,
            required_family_count: REQUIRED_FAMILIES.len(),
            required_families: REQUIRED_FAMILIES,
            missing_required_families,
            families: &manifest.families,
            blockers,
        },
    );
}

fn write_case_manifest(
    parent: &Path,
    cases: &[OptimizationCorpusCase],
    manifest: &OptimizationCorpusManifest,
) {
    let mut seen = BTreeSet::new();
    let mut duplicate_case_ids = BTreeSet::new();
    let entries = cases
        .iter()
        .map(|case| {
            if !seen.insert(case.id.clone()) {
                duplicate_case_ids.insert(case.id.clone());
            }
            let stats = case.program.stats();
            OptimizationCaseManifestEntry {
                id: case.id.clone(),
                family: case.family.clone(),
                node_count: stats.node_count,
                instruction_count: stats.instruction_count,
                memory_op_count: stats.memory_op_count,
                control_flow_count: stats.control_flow_count,
                program_fingerprint: hex(&case.program.fingerprint()),
            }
        })
        .collect::<Vec<_>>();
    let duplicate_case_ids = duplicate_case_ids.into_iter().collect::<Vec<_>>();
    let mut blockers = Vec::new();
    if !duplicate_case_ids.is_empty() {
        blockers.push(format!(
            "semantic optimizer corpus has {} duplicate case id(s)",
            duplicate_case_ids.len()
        ));
    }
    xtask::output_arg::write_json(
        &parent.join("optimization-case-manifest.json"),
        &OptimizationCaseManifest {
            schema_version: 2,
            required_min_cases: manifest.required_min_cases,
            generated_cases: cases.len(),
            unique_case_ids: seen.len(),
            duplicate_case_ids,
            entries,
            blockers,
        },
    );
}

fn write_pass_manifest(parent: &Path) {
    let registrations = registered_pass_registrations().unwrap_or_else(|error| {
        eprintln!("Fix: semantic optimizer registry must schedule: {error}");
        std::process::exit(1);
    });
    let metadata = registrations
        .iter()
        .map(|registration| (registration.metadata.name, registration.metadata))
        .collect::<std::collections::BTreeMap<_, _>>();
    let catalog = optimization_catalog().unwrap_or_else(|error| {
        eprintln!("Fix: semantic optimizer catalog must resolve: {error}");
        std::process::exit(1);
    });
    let entries = catalog
        .iter()
        .map(|entry| {
            let registered = metadata.get(entry.name);
            OptimizerPassManifestEntry {
                id: entry.name.to_string(),
                kind: match entry.kind {
                    OptimizationCatalogEntryKind::ExecutablePass => "executable-pass",
                    OptimizationCatalogEntryKind::SupplementalRule => "supplemental-rule",
                },
                owner: entry.owner,
                phase: format!("{:?}", entry.phase),
                boundary: format!("{:?}", entry.boundary_class),
                requires: registered.map_or_else(Vec::new, |row| row.requires.to_vec()),
                invalidates: registered.map_or_else(Vec::new, |row| row.invalidates.to_vec()),
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
        })
        .collect::<Vec<_>>();
    let unique = entries
        .iter()
        .map(|entry| entry.id.as_str())
        .collect::<BTreeSet<_>>();
    let mut blockers = Vec::new();
    if unique.len() != entries.len() {
        blockers.push("optimizer catalog contains duplicate pass or rule ids".to_string());
    }
    xtask::output_arg::write_json(
        &parent.join("optimizer-pass-manifest.json"),
        &OptimizerPassManifest {
            schema_version: 1,
            executable_passes: registrations.len(),
            catalog_entries: entries.len(),
            entries,
            blockers,
        },
    );
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

fn parse_output(args: &[String]) -> Result<PathBuf, String> {
    xtask::output_arg::parse_output_arg(
        args,
        "optimization-corpus",
        "Generates semantic Program optimizer release evidence.",
        default_output,
    )
}

fn default_output() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|path| path.join("release/evidence/optimization/optimization-corpus.json"))
        .unwrap_or_else(|| PathBuf::from("release/evidence/optimization/optimization-corpus.json"))
}
