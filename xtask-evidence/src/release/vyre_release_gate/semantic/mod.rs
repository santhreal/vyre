//! Per-requirement semantic evidence checks.
//!
//! The dispatch is total: a requirement whose id reaches no check is reported.
//! Twelve of the fourteen shipped requirements were routed here and the
//! fourteenth, `docs-evidence-linked`, fell through a catch-all arm that did
//! nothing, so its evidence was judged only for existence and the map inside it
//! went on requiring a page that had left the tree. A new requirement now turns
//! this gate red until someone writes the check that judges it.

mod conformance_hard_gate;
mod cpu_only_100x_proof;
mod crate_metadata;
mod cuda_first_path;
mod docs_evidence_linked;
mod megakernel_default;
mod optimization_corpus_4096;
mod optimization_integration;
mod proof_workloads_12;
mod public_launch;
mod release_hygiene;
mod version_story;
mod wgpu_fallback;

use std::path::Path;

use super::gate_inputs::{GateMode, Requirement};

pub(super) fn run_semantic_requirement_checks(
    requirement: &Requirement,
    base_dir: &Path,
    mode: GateMode,
    failures: &mut Vec<String>,
) {
    match requirement.id.as_str() {
        "semantic-optimizer-registration" => {
            optimization_integration::check(requirement, base_dir, failures)
        }
        "conformance-hard-gate" => conformance_hard_gate::check(requirement, base_dir, failures),
        "cpu-only-100x-proof" => cpu_only_100x_proof::check(requirement, base_dir, failures),
        "crate-metadata" => crate_metadata::check(requirement, base_dir, failures),
        "docs-evidence-linked" => docs_evidence_linked::check(requirement, base_dir, failures),
        "cuda-first-path" => cuda_first_path::check(requirement, base_dir, failures),
        "public-launch" => public_launch::check(requirement, base_dir, mode, failures),
        "megakernel-default" => megakernel_default::check(requirement, base_dir, failures),
        "optimization-benchmark-proof" => {
            optimization_integration::check(requirement, base_dir, failures)
        }
        "optimization-corpus-4096" => {
            optimization_corpus_4096::check(requirement, base_dir, failures)
        }
        "proof-workloads-12" => proof_workloads_12::check(requirement, base_dir, failures),
        "release-hygiene" => release_hygiene::check(requirement, base_dir, failures),
        "version-story" => version_story::check(requirement, base_dir, failures),
        "wgpu-fallback" => wgpu_fallback::check(requirement, base_dir, failures),
        unknown => failures.push(format!(
            "requirement `{unknown}` reaches no semantic evidence check. Fix: add the check that judges its evidence and route the id to it in xtask-evidence, or remove the requirement from the release manifest"
        )),
    }
}
