//! Per-requirement semantic evidence checks.

mod conformance_hard_gate;
mod cpu_only_100x_proof;
mod crate_metadata;
mod cuda_first_path;
mod megakernel_default;
mod optimization_corpus_4096;
mod optimization_integration;
mod proof_workloads_12;
mod public_launch;
mod release_hygiene;
mod version_story;
mod wgpu_fallback;

use std::path::Path;

use super::types::{GateMode, Requirement};

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
        _ => {}
    }
}
