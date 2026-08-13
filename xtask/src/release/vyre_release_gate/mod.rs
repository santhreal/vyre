//! Evidence gate for the Vyre release objective.
//!
//! This gate intentionally checks artifacts, not intent. The release is
//! blocked until every requirement in `release/vyre-release-evidence.toml` is
//! closed and backed by concrete files.

use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

mod checks;
mod paths;
mod semantic;
mod types;

use checks::check_markdown_evidence_path_ready;
use paths::{escapes_repository, options_from_args, read_text_bounded, resolve_manifest_path};
use semantic::run_semantic_requirement_checks;
use types::{EvidenceManifest, GateMode};

pub(crate) fn run(args: &[String]) {
    let options = match options_from_args(args) {
        Ok(options) => options,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let manifest_path = options.manifest_path;

    let manifest_text = match read_text_bounded(&manifest_path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!(
                "Fix: failed to read release evidence manifest `{}`: {error}",
                manifest_path.display()
            );
            std::process::exit(1);
        }
    };

    let manifest: EvidenceManifest = match toml::from_str(&manifest_text) {
        Ok(manifest) => manifest,
        Err(error) => {
            eprintln!(
                "Fix: release evidence manifest `{}` is invalid TOML: {error}",
                manifest_path.display()
            );
            std::process::exit(1);
        }
    };

    let base_dir = manifest_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut failures = Vec::new();

    if manifest.schema_version != 2 {
        failures.push(format!(
            "schema_version must be 2, found {}",
            manifest.schema_version
        ));
    }
    if manifest.release.vyre.trim().is_empty() {
        failures.push("release.vyre is empty".to_string());
    }

    // The release contract is a tracked document in this repository.
    let release_contract_path = resolve_manifest_path(&base_dir, &manifest.release_contract_path);
    if !release_contract_path.is_file() {
        failures.push(format!(
            "release_contract_path `{}` does not resolve to a file",
            release_contract_path.display()
        ));
    } else {
        match read_text_bounded(&release_contract_path) {
            Ok(_) => {}
            Err(error) => failures.push(format!(
                "release contract `{}` could not be read: {error}",
                release_contract_path.display()
            )),
        }
    }

    let mut ids = BTreeSet::new();
    for requirement in &manifest.requirements {
        if !ids.insert(requirement.id.as_str()) {
            failures.push(format!("duplicate requirement id `{}`", requirement.id));
        }
        if requirement.title.trim().is_empty() {
            failures.push(format!(
                "requirement `{}` has an empty title",
                requirement.id
            ));
        }
        if requirement.status != "closed" {
            failures.push(format!(
                "requirement `{}` is `{}`; release requires `closed`",
                requirement.id, requirement.status
            ));
        }
        if requirement.status == "closed" {
            if requirement.evidence.len() < requirement.minimum_evidence {
                failures.push(format!(
                    "requirement `{}` has {} evidence item(s), needs at least {}",
                    requirement.id,
                    requirement.evidence.len(),
                    requirement.minimum_evidence
                ));
            }
            for evidence in &requirement.evidence {
                if is_manifest_command_evidence(evidence) {
                    continue;
                }
                if escapes_repository(evidence) {
                    failures.push(format!(
                        "requirement `{}` evidence path `{evidence}` resolves outside the repository",
                        requirement.id
                    ));
                    continue;
                }
                let evidence_path = resolve_manifest_path(&base_dir, evidence);
                match fs::metadata(&evidence_path) {
                    Ok(metadata) if metadata.is_file() && metadata.len() > 0 => {
                        if evidence.ends_with(".md") {
                            check_markdown_evidence_path_ready(
                                requirement,
                                &evidence_path,
                                evidence,
                                &mut failures,
                            );
                        }
                    }
                    Ok(metadata) if metadata.is_file() => failures.push(format!(
                        "requirement `{}` evidence path `{}` is empty",
                        requirement.id,
                        evidence_path.display()
                    )),
                    Ok(_) => failures.push(format!(
                        "requirement `{}` evidence path `{}` exists but is not a file",
                        requirement.id,
                        evidence_path.display()
                    )),
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {
                        failures.push(format!(
                            "requirement `{}` evidence path `{}` does not exist",
                            requirement.id,
                            evidence_path.display()
                        ));
                    }
                    Err(error) => failures.push(format!(
                        "requirement `{}` evidence path `{}` is unreadable: {error}",
                        requirement.id,
                        evidence_path.display()
                    )),
                }
            }
            run_semantic_requirement_checks(requirement, &base_dir, options.mode, &mut failures);
        }
    }

    const REQUIRED_IDS: &[&str] = &[
        "version-story",
        "cuda-first-path",
        "wgpu-fallback",
        "megakernel-default",
        "optimization-corpus-4096",
        "optimization-benchmark-proof",
        "semantic-optimizer-registration",
        "proof-workloads-12",
        "cpu-only-100x-proof",
        "conformance-hard-gate",
        "docs-evidence-linked",
        "crate-metadata",
        "release-hygiene",
        "public-launch",
    ];

    for required in REQUIRED_IDS {
        if !ids.contains(required) {
            failures.push(format!("manifest is missing required id `{required}`"));
        }
    }

    if failures.is_empty() {
        let scope = match options.mode {
            GateMode::Final => "final launch",
            GateMode::Prepublish => "prepublication",
        };
        println!(
            "vyre-release-gate: {} requirement(s) closed for Vyre {} ({scope})",
            manifest.requirements.len(),
            manifest.release.vyre
        );
    } else {
        eprintln!("vyre-release-gate: {} release blocker(s):", failures.len());
        for failure in &failures {
            eprintln!("  - {failure}");
        }
        eprintln!("Fix: attach real evidence artifacts and close every manifest requirement.");
        std::process::exit(1);
    }
}
pub(super) fn is_manifest_command_evidence(evidence: &str) -> bool {
    evidence.starts_with("cargo_full ")
}
