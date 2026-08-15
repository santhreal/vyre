//! Hold the semantic `Program` optimizer corpus evidence to the corpus generator.

use std::collections::BTreeSet;

use serde::Serialize;
use vyre_foundation::optimizer::corpus::{
    generate_release_corpus, manifest_for, OptimizationCorpusCase, OptimizationCorpusManifest,
    RELEASE_OPTIMIZATION_FAMILIES,
};
use xtask::artifact_gate::{self, Inspection};
use xtask::gate::{Gate, GateCtx, GateError, Report};

use crate::release::optimizer_pass_rows::{self, OptimizerPassRow};

/// The five artifacts this gate owns, relative to the workspace root.
const CORPUS: &str = "release/evidence/optimization/optimization-corpus.json";
const CONTRACTS: &str = "release/evidence/optimization/optimization-corpus-contracts.json";
const FAMILIES: &str = "release/evidence/optimization/optimization-family-manifest.json";
const CASES: &str = "release/evidence/optimization/optimization-case-manifest.json";
const PASSES: &str = "release/evidence/optimization/optimizer-pass-manifest.json";

/// The families this generator must produce, read from the generator itself so
/// the manifest cannot demand a set the corpus never emits.
const REQUIRED_FAMILIES: &[&str] = &RELEASE_OPTIMIZATION_FAMILIES;
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
    entries: Vec<OptimizerPassRow>,
    blockers: Vec<String>,
}

/// Holds the five optimizer corpus artifacts to the corpus the generator emits.
pub struct OptimizationCorpusGate;

impl Gate for OptimizationCorpusGate {
    fn name(&self) -> &'static str {
        "optimization-corpus"
    }

    fn help(&self) -> &'static str {
        "Regenerate the five artifacts under release/evidence/optimization from the semantic \
         Program optimizer corpus and report each line the committed copies disagree on. Proves \
         the corpus reaches its case floor, that every case verifies after optimization, that at \
         least one pass instance changed a program, that no case failed to converge, that every \
         required family carries its minimum case count, that no case id repeats, and that no \
         optimizer pass id repeats. Proves nothing about runtime performance: the corpus is \
         optimized and re-verified in process, never executed on a device."
    }

    fn generates(&self) -> bool {
        true
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        Ok(artifact_gate::settle_inspection(
            ctx,
            self.name(),
            inspect(),
        ))
    }
}

/// What the corpus generator produces, and the five artifacts recording it.
///
/// The generator used to exit before writing anything when the corpus came out
/// incomplete, so the run that most needed fresh evidence was the run that left
/// the stalest. Every artifact is now rendered whatever the corpus says, and the
/// incompleteness is reported as findings beside it.
fn inspect() -> Inspection {
    let mut inspection = Inspection::new();
    let cases = generate_release_corpus();
    let manifest = manifest_for(&cases);
    report_corpus_completeness(&manifest, &mut inspection);
    inspection.generates(CORPUS, &manifest);
    contracts(&manifest, &mut inspection);
    family_manifest(&manifest, &mut inspection);
    case_manifest(&cases, &manifest, &mut inspection);
    pass_manifest(&mut inspection);
    inspection
}

/// Every way the generated corpus falls short of the release floor.
fn report_corpus_completeness(manifest: &OptimizationCorpusManifest, inspection: &mut Inspection) {
    if manifest.generated_cases < manifest.required_min_cases {
        inspection.blocked(
            CORPUS,
            format!(
                "semantic optimizer corpus generated {} case(s), below the release floor {}",
                manifest.generated_cases, manifest.required_min_cases
            ),
            "Widen the corpus generator in vyre-foundation until it reaches the floor.",
        );
    }
    if manifest.verified_cases != manifest.generated_cases {
        inspection.blocked(
            CORPUS,
            format!(
                "{} of {} corpus case(s) verified after optimization",
                manifest.verified_cases, manifest.generated_cases
            ),
            "A case that does not verify means a pass changed program semantics. Fix the pass.",
        );
    }
    if manifest.optimized_cases == 0 {
        inspection.blocked(
            CORPUS,
            "no corpus case was optimized at all",
            "Every pass is a no-op on this corpus, so the evidence proves nothing. Check that the \
             pass registry is populated and the corpus holds optimizable programs.",
        );
    }
    if manifest.changed_pass_instances == 0 {
        inspection.blocked(
            CORPUS,
            "no pass instance changed a program",
            "The corpus exercises no pass. Widen it, or register the passes it was written for.",
        );
    }
    if manifest.non_converged_cases != 0 {
        inspection.blocked(
            CORPUS,
            format!(
                "{} corpus case(s) did not converge",
                manifest.non_converged_cases
            ),
            "A pass pipeline that does not reach a fixpoint can loop in production. Find the pass \
             pair that keeps undoing each other.",
        );
    }
    for blocker in &manifest.blockers {
        inspection.blocked(
            CORPUS,
            blocker.clone(),
            "Correct the corpus generator in vyre-foundation to satisfy the sentence.",
        );
    }
}

fn contracts(manifest: &OptimizationCorpusManifest, inspection: &mut Inspection) {
    inspection.generates(
        CONTRACTS,
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

fn family_manifest(manifest: &OptimizationCorpusManifest, inspection: &mut Inspection) {
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
    for blocker in &blockers {
        inspection.blocked(
            FAMILIES,
            blocker.clone(),
            format!(
                "Each required family needs at least {MIN_CASES_PER_FAMILY} case(s). Extend the \
                 generator for the named families."
            ),
        );
    }
    inspection.generates(
        FAMILIES,
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

fn case_manifest(
    cases: &[OptimizationCorpusCase],
    manifest: &OptimizationCorpusManifest,
    inspection: &mut Inspection,
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
    for blocker in &blockers {
        inspection.blocked(
            CASES,
            blocker.clone(),
            format!(
                "Two cases sharing an id collapse into one row and hide a case. Rename: {}",
                duplicate_case_ids.join(", ")
            ),
        );
    }
    inspection.generates(
        CASES,
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

fn pass_manifest(inspection: &mut Inspection) {
    let (executable_passes, entries) = optimizer_pass_rows::collect();
    let blockers = optimizer_pass_rows::duplicate_id_blockers(&entries);
    for blocker in &blockers {
        inspection.blocked(
            PASSES,
            blocker.clone(),
            "Two optimizer passes sharing an id make the catalog ambiguous. Give each pass its \
             own id.",
        );
    }
    inspection.generates(
        PASSES,
        &OptimizerPassManifest {
            schema_version: 1,
            executable_passes,
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
