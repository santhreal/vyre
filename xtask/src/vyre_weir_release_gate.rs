//! Evidence gate for the Vyre release objective.
//!
//! This gate intentionally checks artifacts, not intent. The release is
//! blocked until every requirement in `release/vyre-release-evidence.toml` is
//! closed and backed by concrete files.

use std::collections::BTreeSet;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct EvidenceManifest {
    schema_version: u32,
    plan_path: String,
    release: ReleaseNames,
    requirements: Vec<Requirement>,
}

#[derive(Debug, Deserialize)]
struct ReleaseNames {
    vyre: String,
    weir: String,
}

#[derive(Debug, Deserialize)]
struct Requirement {
    id: String,
    title: String,
    status: String,
    evidence: Vec<String>,
    minimum_evidence: usize,
}

const MAX_RELEASE_GATE_TEXT_BYTES: u64 = 16_777_216;

pub(crate) fn run(args: &[String]) {
    let manifest_path = match manifest_path_from_args(args) {
        Ok(path) => path,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };

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

    if manifest.schema_version != 1 {
        failures.push(format!(
            "schema_version must be 1, found {}",
            manifest.schema_version
        ));
    }
    if manifest.release.vyre.trim().is_empty() {
        failures.push("release.vyre is empty".to_string());
    }
    if manifest.release.weir.trim().is_empty() {
        failures.push("release.weir is empty".to_string());
    }

    let plan_path = resolve_manifest_path(&base_dir, &manifest.plan_path);
    if !plan_path.is_file() {
        failures.push(format!(
            "plan_path `{}` does not resolve to a file",
            plan_path.display()
        ));
    } else {
        match read_text_bounded(&plan_path) {
            Ok(plan) => {
                for phrase in [
                    "thousands of concrete",
                    "At least ten",
                    "CUDA-first",
                    "Linux subsystem corpus",
                    "Completion audit checklist",
                ] {
                    if !plan.contains(phrase) {
                        failures.push(format!(
                            "plan `{}` is missing required release phrase `{phrase}`",
                            plan_path.display()
                        ));
                    }
                }
            }
            Err(error) => failures.push(format!(
                "plan `{}` could not be read: {error}",
                plan_path.display()
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
            run_semantic_requirement_checks(requirement, &base_dir, &mut failures);
        }
    }

    const REQUIRED_IDS: &[&str] = &[
        "version-story",
        "cuda-first-path",
        "wgpu-fallback",
        "megakernel-default",
        "optimization-corpus-4096",
        "optimization-benchmark-proof",
        "weir-analysis-integration",
        "alias-aware-upgrades",
        "egraph-saturation",
        "proof-workloads-12",
        "cpu-only-100x-proof",
        "c-parser-linux-subsystem",
        "distributed-parser-coherence",
        "conformance-hard-gate",
        "modular-test-architecture",
        "exhaustive-verification",
        "docs-evidence-linked",
        "crate-metadata",
        "release-hygiene",
        "final-completion-audit",
    ];

    for required in REQUIRED_IDS {
        if !ids.contains(required) {
            failures.push(format!("manifest is missing required id `{required}`"));
        }
    }

    if failures.is_empty() {
        println!(
            "vyre-release-gate: {} requirement(s) closed for Vyre {} / Weir {}",
            manifest.requirements.len(),
            manifest.release.vyre,
            manifest.release.weir
        );
    } else {
        eprintln!(
            "vyre-release-gate: {} release blocker(s):",
            failures.len()
        );
        for failure in &failures {
            eprintln!("  - {failure}");
        }
        eprintln!("Fix: attach real evidence artifacts and close every manifest requirement.");
        std::process::exit(1);
    }
}

fn is_manifest_command_evidence(evidence: &str) -> bool {
    evidence.starts_with("cargo_full ")
}

fn run_semantic_requirement_checks(
    requirement: &Requirement,
    base_dir: &Path,
    failures: &mut Vec<String>,
) {
    match requirement.id.as_str() {
        "version-story" => {
            if !requirement.evidence.iter().any(|evidence| {
                evidence.contains("cargo_full")
                    && evidence.contains("version-matrix")
                    && evidence.contains("release/evidence/version/version-matrix.json")
            }) {
                failures.push(
                    "requirement `version-story` must include the cargo_full version-matrix evidence command"
                        .to_string(),
                );
            }
            let Some(matrix) =
                first_json_evidence(requirement, base_dir, "version-matrix.json", failures)
            else {
                return;
            };
            let blockers = matrix
                .get("blockers")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if blockers != 0 {
                failures.push(format!(
                    "requirement `version-story` matrix still reports {blockers} blocker(s)"
                ));
            }
            let vyre_release = matrix
                .get("requested_vyre_release")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            if vyre_release != "0.4.2" {
                failures.push(format!(
                    "requirement `version-story` requested_vyre_release is `{vyre_release}`, expected `0.4.2`"
                ));
            }
            let weir_release = matrix
                .get("requested_weir_release")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            if weir_release != "0.1.0" {
                failures.push(format!(
                    "requirement `version-story` requested_weir_release is `{weir_release}`, expected `0.1.0`"
                ));
            }
            if matrix
                .get("release_doc_tag_findings")
                .and_then(serde_json::Value::as_array)
                .is_none_or(|findings| !findings.is_empty())
            {
                failures.push(
                    "requirement `version-story` release docs must not contain bare v0.4.2 tag commands"
                        .to_string(),
                );
            }
            if matrix
                .get("release_note_token_findings")
                .and_then(serde_json::Value::as_array)
                .is_none_or(|findings| !findings.is_empty())
            {
                failures.push(
                    "requirement `version-story` release-note docs must include every required version and product-scoped tag token"
                        .to_string(),
                );
            }
            if matrix
                .get("missing_required_release_packages")
                .and_then(serde_json::Value::as_array)
                .is_none_or(|packages| !packages.is_empty())
            {
                failures.push(
                    "requirement `version-story` missing_required_release_packages must be empty"
                        .to_string(),
                );
            }
            let required_release_packages = matrix
                .get("required_release_packages")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default();
            for required_package in [
                "vyre@0.4.2",
                "vyre-driver-cuda@0.4.2",
                "vyre-driver-wgpu@0.4.2",
                "weir@0.1.0",
                "vyrec@0.1.0",
                "vyre-frontend-c@0.4.2",
            ] {
                if !required_release_packages
                    .iter()
                    .any(|package| package.as_str() == Some(required_package))
                {
                    failures.push(format!(
                        "requirement `version-story` required_release_packages must include `{required_package}`"
                    ));
                }
            }
            let Some(tag_story) = matrix
                .get("tag_story")
                .and_then(serde_json::Value::as_object)
            else {
                failures.push(
                    "requirement `version-story` version matrix is missing `tag_story`".to_string(),
                );
                return;
            };
            for (field, expected) in [
                ("vyre_rc_tag", "vyre-v0.4.2-rc.1"),
                ("weir_rc_tag", "weir-v0.1.0-rc.1"),
                (
                    "combined_release_train_rc_tag",
                    "vyre-0.4.2-weir-0.1.0-rc.1",
                ),
                ("vyre_tag", "vyre-v0.4.2"),
                ("weir_tag", "weir-v0.1.0"),
                ("combined_release_train_tag", "vyre-0.4.2-weir-0.1.0"),
            ] {
                let actual = tag_story
                    .get(field)
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");
                if actual != expected {
                    failures.push(format!(
                        "requirement `version-story` tag_story.{field} is `{actual}`, expected `{expected}`"
                    ));
                }
            }
            for required in [
                "vyre 0.4.2",
                "weir 0.1.0",
                "vyre-driver-cuda@0.4.2",
                "vyre-driver-wgpu@0.4.2",
                "vyre-v0.4.2-rc.1",
                "weir-v0.1.0-rc.1",
                "vyre-0.4.2-weir-0.1.0-rc.1",
                "vyre-v0.4.2",
                "weir-v0.1.0",
                "vyre-0.4.2-weir-0.1.0",
            ] {
                let present = tag_story
                    .get("required_in_release_notes")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|entries| {
                        entries.iter().any(|entry| entry.as_str() == Some(required))
                    });
                if !present {
                    failures.push(format!(
                        "requirement `version-story` tag_story.required_in_release_notes is missing `{required}`"
                    ));
                }
            }
            if let Some(tag_plan) =
                first_json_evidence(requirement, base_dir, "release-tag-plan.json", failures)
            {
                let tag_plan_blockers = tag_plan
                    .get("blockers")
                    .and_then(serde_json::Value::as_array)
                    .map_or(usize::MAX, Vec::len);
                if tag_plan_blockers != 0 {
                    failures.push(format!(
                        "requirement `version-story` release-tag-plan reports {tag_plan_blockers} blocker(s)"
                    ));
                }
                for (field, expected) in [
                    ("vyre_rc_tag", "vyre-v0.4.2-rc.1"),
                    ("weir_rc_tag", "weir-v0.1.0-rc.1"),
                    (
                        "combined_release_train_rc_tag",
                        "vyre-0.4.2-weir-0.1.0-rc.1",
                    ),
                    ("vyre_tag", "vyre-v0.4.2"),
                    ("weir_tag", "weir-v0.1.0"),
                    ("combined_release_train_tag", "vyre-0.4.2-weir-0.1.0"),
                ] {
                    if tag_plan.get(field).and_then(serde_json::Value::as_str) != Some(expected) {
                        failures.push(format!(
                            "requirement `version-story` release-tag-plan {field} must be `{expected}`"
                        ));
                    }
                }
                if !tag_plan
                    .get("required_gate_before_rc_tag")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|command| {
                        command.contains("version-matrix")
                            && command.contains("release-completion-audit")
                            && command.contains("vyre-release-gate")
                            && command.contains("scripts/apply-branch-protection.sh")
                            && command.contains("cargo_full")
                    })
                {
                    failures.push(
                    "requirement `version-story` release-tag-plan must require version matrix regeneration, completion audit, release gate, and branch-protection application before RC tag creation"
                        .to_string(),
                );
                }
                let order = tag_plan
                    .get("tag_creation_order")
                    .and_then(serde_json::Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let ordered_tags = order
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .collect::<Vec<_>>();
                for (rc, final_tag) in [
                    ("vyre-v0.4.2-rc.1", "vyre-v0.4.2"),
                    ("weir-v0.1.0-rc.1", "weir-v0.1.0"),
                    ("vyre-0.4.2-weir-0.1.0-rc.1", "vyre-0.4.2-weir-0.1.0"),
                ] {
                    let rc_index = ordered_tags.iter().position(|tag| *tag == rc);
                    let final_index = ordered_tags.iter().position(|tag| *tag == final_tag);
                    if !matches!((rc_index, final_index), (Some(left), Some(right)) if left < right)
                    {
                        failures.push(format!(
                            "requirement `version-story` release-tag-plan must list `{rc}` before `{final_tag}`"
                        ));
                    }
                }
                if !tag_plan
                    .get("required_gate_before_tag")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|command| {
                        command.contains("version-matrix")
                            && command.contains("release-completion-audit")
                            && command.contains("vyre-release-gate")
                            && command.contains("scripts/apply-branch-protection.sh")
                            && command.contains("cargo_full")
                    })
                {
                    failures.push(
                    "requirement `version-story` release-tag-plan must require version matrix regeneration, completion audit, release gate, and branch-protection application before tag creation"
                        .to_string(),
                );
                }
                let version_blockers = tag_plan
                    .get("version_matrix_blocker_count")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(u64::MAX);
                if version_blockers != 0 {
                    failures.push(format!(
                        "requirement `version-story` release-tag-plan carries {version_blockers} version matrix blocker(s)"
                    ));
                }
            }
        }
        "optimization-corpus-4096" => {
            let Some(corpus) =
                first_json_evidence(requirement, base_dir, "optimization-corpus.json", failures)
            else {
                return;
            };
            let generated = corpus
                .get("generated_cases")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let verified = corpus
                .get("verified_cases")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let optimized = corpus
                .get("optimized_cases")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let dataflow_analysis_cases = corpus
                .get("dataflow_analysis_cases")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let dataflow_analysis_optimized = corpus
                .get("dataflow_analysis_optimized_cases")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let non_converged = corpus
                .get("non_converged_cases")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(u64::MAX);
            let blockers = corpus
                .get("blockers")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            let required = corpus
                .get("required_min_cases")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(4_096);
            if required < 4_096 {
                failures.push(format!(
                    "requirement `optimization-corpus-4096` required_min_cases={required}; release floor is 4096"
                ));
            }
            if generated < required || generated < 4_096 {
                failures.push(format!(
                    "requirement `optimization-corpus-4096` generated {generated} cases; needs at least {required} and never below 4096"
                ));
            }
            if verified != generated {
                failures.push(format!(
                    "requirement `optimization-corpus-4096` verified {verified}/{generated} generated cases through verify_then_optimize"
                ));
            }
            if optimized == 0 {
                failures.push(
                    "requirement `optimization-corpus-4096` reports zero optimized cases; corpus is not proving rewrite coverage"
                        .to_string(),
                );
            }
            if dataflow_analysis_cases == 0 {
                failures.push(
                    "requirement `optimization-corpus-4096` reports zero dataflow-analysis-aware cases"
                        .to_string(),
                );
            }
            if dataflow_analysis_optimized < dataflow_analysis_cases {
                failures.push(format!(
                    "requirement `optimization-corpus-4096` optimized {dataflow_analysis_optimized}/{dataflow_analysis_cases} dataflow-analysis-aware cases"
                ));
            }
            if non_converged != 0 || blockers != 0 {
                failures.push(format!(
                    "requirement `optimization-corpus-4096` reports {non_converged} non-converged case(s) and {blockers} blocker(s)"
                ));
            }
            for suffix in [
                "optimization-corpus-contracts.json",
                "optimization-family-manifest.json",
                "optimization-analysis-fixtures.json",
                "optimization-case-manifest.json",
            ] {
                check_json_evidence_has_no_blockers(requirement, base_dir, suffix, failures);
            }
            if let Some(family_manifest) = first_json_evidence(
                requirement,
                base_dir,
                "optimization-family-manifest.json",
                failures,
            ) {
                let families = family_manifest
                    .get("families")
                    .and_then(serde_json::Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                if families.len() < 14 {
                    failures.push(format!(
                        "requirement `optimization-corpus-4096` family manifest lists {} optimization families; needs at least 14 required release families",
                        families.len()
                    ));
                }
                let declared_required = family_manifest
                    .get("required_family_count")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                if declared_required < 14 {
                    failures.push(format!(
                        "requirement `optimization-corpus-4096` family manifest declares {declared_required} required optimization families; needs all 14 release families"
                    ));
                }
                let missing_required = family_manifest
                    .get("missing_required_families")
                    .and_then(serde_json::Value::as_array)
                    .map_or(usize::MAX, Vec::len);
                if missing_required != 0 {
                    failures.push(format!(
                        "requirement `optimization-corpus-4096` family manifest reports {missing_required} missing required optimization family/families"
                    ));
                }
                for required in [
                    "algebraic",
                    "predicate",
                    "egraph",
                    "memory-layout",
                    "control-flow",
                    "vector-layout",
                    "A13-coalesce-fixture",
                    "A14-shared-mem-promote-fixture",
                    "A15-bank-conflict-fixture",
                    "A16-vec-pack-fixture",
                    "dataflow-analysis-dse",
                    "dataflow-analysis-loop-fusion",
                    "dataflow-analysis-loop-fission",
                    "dataflow-analysis-licm",
                ] {
                    let required_cases = families
                        .iter()
                        .find(|family| {
                            family.get("family").and_then(serde_json::Value::as_str)
                                == Some(required)
                        })
                        .and_then(|family| family.get("cases").and_then(serde_json::Value::as_u64))
                        .unwrap_or(0);
                    if required_cases < 128 {
                        failures.push(format!(
                            "requirement `optimization-corpus-4096` required family `{required}` has {required_cases} generated case(s), needs at least 128"
                        ));
                    }
                }
                for family in &families {
                    let name = family
                        .get("family")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("<unknown>");
                    if family
                        .get("family")
                        .and_then(serde_json::Value::as_str)
                        .is_none_or(str::is_empty)
                        || family
                            .get("cases")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(0)
                            == 0
                    {
                        failures.push(format!(
                            "requirement `optimization-corpus-4096` family manifest contains invalid family `{name}`"
                        ));
                    }
                }
            }
            if let Some(fixture_manifest) = first_json_evidence(
                requirement,
                base_dir,
                "optimization-analysis-fixtures.json",
                failures,
            ) {
                check_optimization_analysis_fixture_manifest(&fixture_manifest, failures);
            }
            if let Some(case_manifest) = first_json_evidence(
                requirement,
                base_dir,
                "optimization-case-manifest.json",
                failures,
            ) {
                let pass_instances = case_manifest
                    .get("pass_instance_count")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let unique_case_ids = case_manifest
                    .get("unique_case_ids")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let manifest_generated = case_manifest
                    .get("generated_cases")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let entries = case_manifest
                    .get("entries")
                    .and_then(serde_json::Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                if pass_instances != generated || manifest_generated != generated {
                    failures.push(format!(
                        "requirement `optimization-corpus-4096` case manifest pass_instance_count={pass_instances}, generated_cases={manifest_generated}, corpus generated_cases={generated}"
                    ));
                }
                if pass_instances < 4_096 || unique_case_ids != pass_instances {
                    failures.push(format!(
                        "requirement `optimization-corpus-4096` case manifest has {pass_instances} pass instance(s) and {unique_case_ids} unique id(s); needs >=4096 unique pass instances"
                    ));
                }
                if entries.len() as u64 != pass_instances {
                    failures.push(format!(
                        "requirement `optimization-corpus-4096` case manifest lists {} entrie(s), pass_instance_count is {pass_instances}",
                        entries.len()
                    ));
                }
                for field in [
                    "cases_with_child_bodies",
                    "cases_with_bindings",
                    "cases_with_literals",
                ] {
                    if case_manifest
                        .get(field)
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0)
                        == 0
                    {
                        failures.push(format!(
                            "requirement `optimization-corpus-4096` case manifest `{field}` must be nonzero"
                        ));
                    }
                }
                let malformed_entries = entries
                    .iter()
                    .filter(|entry| {
                        entry
                            .get("id")
                            .and_then(serde_json::Value::as_str)
                            .is_none_or(str::is_empty)
                            || entry
                                .get("family")
                                .and_then(serde_json::Value::as_str)
                                .is_none_or(str::is_empty)
                            || entry
                                .get("total_ops")
                                .and_then(serde_json::Value::as_u64)
                                .unwrap_or(0)
                                == 0
                    })
                    .count();
                if malformed_entries != 0 {
                    failures.push(format!(
                        "requirement `optimization-corpus-4096` case manifest contains {malformed_entries} malformed generated pass instance(s)"
                    ));
                }
            }
        }
        "optimization-benchmark-proof" | "alias-aware-upgrades" | "egraph-saturation" => {
            let Some(matrix) = first_json_evidence(
                requirement,
                base_dir,
                "optimization-integration-matrix.json",
                failures,
            ) else {
                return;
            };
            let blockers = matrix
                .get("blockers")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if blockers != 0 {
                failures.push(format!(
                    "requirement `{}` optimization matrix still reports {blockers} blocker(s)",
                    requirement.id
                ));
            }
            match requirement.id.as_str() {
                "optimization-benchmark-proof" => {
                    check_before_after_benchmark_report(
                        requirement,
                        base_dir,
                        "lower-rewrite-impact-before-after.json",
                        failures,
                    );
                    check_before_after_benchmark_report(
                        requirement,
                        base_dir,
                        "optimizer-impact-cuda.json",
                        failures,
                    );
                    check_before_after_benchmark_report(
                        requirement,
                        base_dir,
                        "pass-family-benchmarks.json",
                        failures,
                    );
                    check_json_evidence_has_no_blockers(
                        requirement,
                        base_dir,
                        "pass-family-benchmark-manifest.json",
                        failures,
                    );
                    if let Some(manifest) = first_json_evidence(
                        requirement,
                        base_dir,
                        "pass-family-benchmark-manifest.json",
                        failures,
                    ) {
                        if manifest.get("backend").and_then(serde_json::Value::as_str)
                            != Some("cuda")
                        {
                            failures.push(
                                "requirement `optimization-benchmark-proof` pass-family benchmark manifest must be cuda"
                                    .to_string(),
                            );
                        }
                        let cases = manifest
                            .get("cases")
                            .and_then(serde_json::Value::as_array)
                            .cloned()
                            .unwrap_or_default();
                        for required_family in REQUIRED_BENCHMARKED_OPTIMIZATION_FAMILIES {
                            let covered = manifest
                                .get("covered_pass_families")
                                .and_then(serde_json::Value::as_array)
                                .is_some_and(|families| {
                                    families
                                        .iter()
                                        .any(|family| family.as_str() == Some(required_family))
                                });
                            if !covered {
                                failures.push(format!(
                                    "requirement `optimization-benchmark-proof` pass-family manifest does not benchmark required family `{required_family}`"
                                ));
                            }
                        }
                        if manifest
                            .get("uncovered_pass_families")
                            .and_then(serde_json::Value::as_array)
                            .is_none_or(|families| !families.is_empty())
                        {
                            failures.push(
                                "requirement `optimization-benchmark-proof` pass-family manifest reports uncovered pass families"
                                    .to_string(),
                            );
                        }
                        for required_case in [
                            "lower.rewrites.impact.corpus",
                            "foundation.optimizer.impact",
                            "lower.egraph_saturation",
                            "lower.alias_aware_optimizations",
                        ] {
                            if !cases.iter().any(|case| {
                                case.get("case_id").and_then(serde_json::Value::as_str)
                                    == Some(required_case)
                                    && case.get("exists").and_then(serde_json::Value::as_bool)
                                        == Some(true)
                                    && case
                                        .get("read_error")
                                        .is_some_and(serde_json::Value::is_null)
                                    && case
                                        .get("required_custom_metrics")
                                        .and_then(serde_json::Value::as_array)
                                        .is_some_and(|metrics| !metrics.is_empty())
                                    && case
                                        .get("required_positive_metrics")
                                        .and_then(serde_json::Value::as_array)
                                        .is_some_and(|metrics| !metrics.is_empty())
                            }) {
                                failures.push(format!(
                                    "requirement `optimization-benchmark-proof` pass-family manifest is missing `{required_case}`"
                                ));
                            }
                        }
                        for case in &cases {
                            let Some(artifact) =
                                case.get("artifact").and_then(serde_json::Value::as_str)
                            else {
                                failures.push(
                                    "requirement `optimization-benchmark-proof` pass-family manifest case is missing artifact"
                                        .to_string(),
                                );
                                continue;
                            };
                            if case
                                .get("covered_pass_families")
                                .and_then(serde_json::Value::as_array)
                                .is_none_or(|families| families.is_empty())
                            {
                                failures.push(
                                    "requirement `optimization-benchmark-proof` pass-family manifest case lists no covered_pass_families"
                                        .to_string(),
                                );
                            }
                            for field in [
                                "missing_custom_metrics",
                                "non_positive_required_metrics",
                                "non_winning_cases",
                                "blockers",
                            ] {
                                if case
                                    .get(field)
                                    .and_then(serde_json::Value::as_array)
                                    .is_none_or(|items| !items.is_empty())
                                {
                                    failures.push(format!(
                                        "requirement `optimization-benchmark-proof` pass-family manifest case `{}` has non-empty `{field}`",
                                        case.get("case_id")
                                            .and_then(serde_json::Value::as_str)
                                            .unwrap_or("<unknown>")
                                    ));
                                }
                            }
                            let read_error = case.get("read_error");
                            if !read_error.is_some_and(serde_json::Value::is_null) {
                                failures.push(format!(
                                    "requirement `optimization-benchmark-proof` pass-family manifest case `{}` read_error={}",
                                    case.get("case_id")
                                        .and_then(serde_json::Value::as_str)
                                        .unwrap_or("<unknown>"),
                                    read_error
                                        .map(serde_json::Value::to_string)
                                        .unwrap_or_else(|| "<missing>".to_string())
                                ));
                            }
                            for field in ["min_wall_samples", "min_baseline_wall_samples"] {
                                if case
                                    .get(field)
                                    .and_then(serde_json::Value::as_u64)
                                    .unwrap_or(0)
                                    < 30
                                {
                                    failures.push(format!(
                                        "requirement `optimization-benchmark-proof` pass-family manifest case `{}` has `{field}` below 30",
                                        case.get("case_id")
                                            .and_then(serde_json::Value::as_str)
                                            .unwrap_or("<unknown>")
                                    ));
                                }
                            }
                            for field in [
                                "min_wall_p50",
                                "min_wall_p95",
                                "min_wall_p99",
                                "min_baseline_wall_p50",
                                "min_baseline_wall_p95",
                                "min_baseline_wall_p99",
                            ] {
                                if case
                                    .get(field)
                                    .and_then(serde_json::Value::as_u64)
                                    .unwrap_or(0)
                                    == 0
                                {
                                    failures.push(format!(
                                        "requirement `optimization-benchmark-proof` pass-family manifest case `{}` has non-positive `{field}`",
                                        case.get("case_id")
                                            .and_then(serde_json::Value::as_str)
                                            .unwrap_or("<unknown>")
                                    ));
                                }
                            }
                            let has_speed_win = case
                                .get("min_wall_speedup_x1000")
                                .and_then(serde_json::Value::as_u64)
                                .unwrap_or(0)
                                > 1_000;
                            let has_semantic_win = case
                                .get("non_winning_cases")
                                .and_then(serde_json::Value::as_array)
                                .is_some_and(|items| items.is_empty());
                            if !has_speed_win && !has_semantic_win {
                                failures.push(format!(
                                    "requirement `optimization-benchmark-proof` pass-family manifest case `{}` does not prove optimized wall_ns p50 beats baseline_wall_ns p50",
                                    case.get("case_id")
                                        .and_then(serde_json::Value::as_str)
                                        .unwrap_or("<unknown>")
                                ));
                            }
                            let Some(report) =
                                read_json_artifact_ref(requirement, base_dir, artifact, failures)
                            else {
                                continue;
                            };
                            let suffix = Path::new(artifact)
                                .file_name()
                                .and_then(|name| name.to_str())
                                .unwrap_or(artifact);
                            if let Some(metrics) = case
                                .get("required_custom_metrics")
                                .and_then(serde_json::Value::as_array)
                            {
                                for metric in metrics.iter().filter_map(serde_json::Value::as_str) {
                                    require_case_metric_present(
                                        requirement,
                                        suffix,
                                        &report,
                                        metric,
                                        failures,
                                    );
                                }
                            }
                            if let Some(metrics) = case
                                .get("required_positive_metrics")
                                .and_then(serde_json::Value::as_array)
                            {
                                for metric in metrics.iter().filter_map(serde_json::Value::as_str) {
                                    require_case_metric_positive(
                                        requirement,
                                        suffix,
                                        &report,
                                        metric,
                                        failures,
                                    );
                                }
                            }
                        }
                    }
                }
                "egraph-saturation" => {
                    check_json_evidence_has_no_blockers(
                        requirement,
                        base_dir,
                        "egraph-saturation-matrix.json",
                        failures,
                    );
                    check_marker_evidence_has_markers(
                        requirement,
                        base_dir,
                        "egraph-saturation-matrix.json",
                        failures,
                    );
                    check_json_evidence_has_no_blockers(
                        requirement,
                        base_dir,
                        "egraph-semantic-contracts.json",
                        failures,
                    );
                    check_marker_evidence_has_markers(
                        requirement,
                        base_dir,
                        "egraph-semantic-contracts.json",
                        failures,
                    );
                    check_before_after_benchmark_report(
                        requirement,
                        base_dir,
                        "egraph-before-after.json",
                        failures,
                    );
                    if let Some(report) = first_json_evidence(
                        requirement,
                        base_dir,
                        "egraph-before-after.json",
                        failures,
                    ) {
                        require_case_metric_positive(
                            requirement,
                            "egraph-before-after.json",
                            &report,
                            "egraph_equality_classes",
                            failures,
                        );
                        require_case_metric_positive(
                            requirement,
                            "egraph-before-after.json",
                            &report,
                            "egraph_bitwise_case_count",
                            failures,
                        );
                        require_case_metric_positive(
                            requirement,
                            "egraph-before-after.json",
                            &report,
                            "egraph_boolean_case_count",
                            failures,
                        );
                        require_case_metric_positive(
                            requirement,
                            "egraph-before-after.json",
                            &report,
                            "egraph_applied_rewrites",
                            failures,
                        );
                    }
                }
                "alias-aware-upgrades" => {
                    for suffix in [
                        "alias-aware-dse.json",
                        "alias-aware-stlf.json",
                        "alias-aware-licm.json",
                        "alias-aware-fusion-fission.json",
                    ] {
                        check_json_evidence_has_no_blockers(
                            requirement,
                            base_dir,
                            suffix,
                            failures,
                        );
                        check_marker_evidence_has_markers(requirement, base_dir, suffix, failures);
                    }
                    check_before_after_benchmark_report(
                        requirement,
                        base_dir,
                        "alias-aware-before-after.json",
                        failures,
                    );
                    if let Some(report) = first_json_evidence(
                        requirement,
                        base_dir,
                        "alias-aware-before-after.json",
                        failures,
                    ) {
                        for metric in [
                            "alias_pass_wins",
                            "alias_fact_count",
                            "alias_cross_binding_fact_count",
                            "reaching_def_fact_count",
                        ] {
                            require_case_metric_positive(
                                requirement,
                                "alias-aware-before-after.json",
                                &report,
                                metric,
                                failures,
                            );
                        }
                    }
                }
                _ => {}
            }
        }
        "cuda-first-path" => {
            let Some(matrix) =
                first_json_evidence(requirement, base_dir, "backend-matrix.json", failures)
            else {
                return;
            };
            let cuda_first = matrix
                .get("cuda_first")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            let blockers = matrix
                .get("blockers")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if !cuda_first {
                failures.push(
                    "requirement `cuda-first-path` backend matrix does not prove CUDA-first dispatch"
                        .to_string(),
                );
            }
            check_backend_matrix_schema("cuda-first-path", &matrix, failures);
            if blockers != 0 {
                failures.push(format!(
                    "requirement `cuda-first-path` backend matrix still reports {blockers} blocker(s)"
                ));
            }
            require_no_hidden_backend_fallback_findings("cuda-first-path", &matrix, failures);
            check_backend_gpu_probe("cuda-first-path", &matrix, failures);
            check_preferred_backend_gpu_only("cuda-first-path", &matrix, failures);
            check_backend_acquire_entry("cuda-first-path", &matrix, "cuda", failures);
            check_backend_feature_markers(
                "cuda-first-path",
                &matrix,
                "cuda_feature_markers",
                12,
                failures,
            );
            check_json_evidence_has_no_blockers(
                requirement,
                base_dir,
                "cuda-release-suite.json",
                failures,
            );
            check_backend_suite_report(requirement, base_dir, "cuda-release-suite.json", failures);
            check_benchmark_report_has_cases(
                requirement,
                base_dir,
                "cuda-ptx-patterns.json",
                failures,
            );
            check_json_evidence_has_no_blockers(
                requirement,
                base_dir,
                "bench-release-axes.json",
                failures,
            );
            if let Some(axes) =
                first_json_evidence(requirement, base_dir, "bench-release-axes.json", failures)
            {
                let source_artifacts = axes
                    .get("source_artifacts")
                    .and_then(serde_json::Value::as_array)
                    .map_or(0, Vec::len);
                if source_artifacts < 12 {
                    failures.push(format!(
                        "requirement `cuda-first-path` bench-release-axes has {source_artifacts} source artifact(s), needs at least 12"
                    ));
                }
            }
        }
        "wgpu-fallback" => {
            let Some(matrix) =
                first_json_evidence(requirement, base_dir, "backend-matrix.json", failures)
            else {
                return;
            };
            let present = matrix
                .get("wgpu_fallback_present")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            let blockers = matrix
                .get("blockers")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if !present {
                failures.push(
                    "requirement `wgpu-fallback` backend matrix does not prove acquireable WGPU fallback"
                        .to_string(),
                );
            }
            check_backend_matrix_schema("wgpu-fallback", &matrix, failures);
            if blockers != 0 {
                failures.push(format!(
                    "requirement `wgpu-fallback` backend matrix still reports {blockers} blocker(s)"
                ));
            }
            require_no_hidden_backend_fallback_findings("wgpu-fallback", &matrix, failures);
            check_backend_gpu_probe("wgpu-fallback", &matrix, failures);
            check_preferred_backend_gpu_only("wgpu-fallback", &matrix, failures);
            check_backend_acquire_entry("wgpu-fallback", &matrix, "wgpu", failures);
            check_backend_feature_markers(
                "wgpu-fallback",
                &matrix,
                "wgpu_feature_markers",
                7,
                failures,
            );
            check_json_evidence_has_no_blockers(
                requirement,
                base_dir,
                "wgpu-fallback-suite.json",
                failures,
            );
            check_backend_suite_report(requirement, base_dir, "wgpu-fallback-suite.json", failures);
        }
        "megakernel-default" => {
            let Some(backend_matrix) =
                first_json_evidence(requirement, base_dir, "backend-matrix.json", failures)
            else {
                return;
            };
            check_backend_feature_marker_id(
                "megakernel-default",
                &backend_matrix,
                "cuda_feature_markers",
                "megakernel-paired-speculation",
                failures,
            );
            check_backend_feature_marker_id(
                "megakernel-default",
                &backend_matrix,
                "wgpu_feature_markers",
                "megakernel-paired-speculation",
                failures,
            );
            let Some(matrix) = first_json_evidence(
                requirement,
                base_dir,
                "release-workload-matrix.json",
                failures,
            ) else {
                return;
            };
            let has_megakernel = matrix
                .get("families")
                .and_then(serde_json::Value::as_array)
                .and_then(|families| {
                    families.iter().find(|family| {
                        family.get("id").and_then(serde_json::Value::as_str)
                            == Some("megakernel-queued-batches")
                    })
                })
                .and_then(|family| family.get("matched_cases"))
                .and_then(serde_json::Value::as_array)
                .is_some_and(|cases| !cases.is_empty());
            let blockers = matrix
                .get("blockers")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if !has_megakernel {
                failures.push(
                    "requirement `megakernel-default` has no active megakernel-queued-batches workload in the release matrix"
                        .to_string(),
                );
            }
            if blockers != 0 {
                failures.push(format!(
                    "requirement `megakernel-default` workload matrix still reports {blockers} blocker(s)"
                ));
            }
            check_named_cuda_benchmark_report(
                requirement,
                base_dir,
                "megakernel-condition-cuda.json",
                failures,
            );
            check_named_cuda_benchmark_report(
                requirement,
                base_dir,
                "megakernel-latency-cuda.json",
                failures,
            );
        }
        "proof-workloads-12" => {
            let Some(matrix) = first_json_evidence(
                requirement,
                base_dir,
                "release-workload-matrix.json",
                failures,
            ) else {
                return;
            };
            let required = matrix
                .get("required_closed_families")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let matched = matrix
                .get("matched_required_families")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let release_cases = matrix
                .get("release_suite_case_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let blockers = matrix
                .get("blockers")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if required < 12 {
                failures.push(format!(
                    "requirement `proof-workloads-12` matrix requires only {required} workload families; needs at least 12"
                ));
            }
            if matched < 12 {
                failures.push(format!(
                    "requirement `proof-workloads-12` matrix covers {matched} workload families; needs at least 12"
                ));
            }
            if matched < required {
                failures.push(format!(
                    "requirement `proof-workloads-12` matrix covers {matched} of {required} required workload families"
                ));
            }
            if release_cases < matched {
                failures.push(format!(
                    "requirement `proof-workloads-12` matrix reports {release_cases} release cases for {matched} matched workload families"
                ));
            }
            if blockers != 0 {
                failures.push(format!(
                    "requirement `proof-workloads-12` matrix still reports {blockers} blocker(s)"
                ));
            }
            check_release_bench_targets(requirement, base_dir, failures);
            check_workload_matrix_artifact_coverage(requirement, base_dir, &matrix, failures);
            check_benchmark_evidence_reports(
                requirement,
                base_dir,
                "workload-",
                true,
                None,
                failures,
            );
        }
        "cpu-only-100x-proof" => {
            let Some(matrix) = first_json_evidence(
                requirement,
                base_dir,
                "release-workload-matrix.json",
                failures,
            ) else {
                return;
            };
            let contracts = matrix
                .get("cpu_sota_100x_contract_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            if contracts < 10 {
                failures.push(format!(
                    "requirement `cpu-only-100x-proof` has {contracts} CPU-SOTA 100x contract(s) in the workload matrix; needs at least 10"
                ));
            }
            let covered_families = matrix
                .get("cpu_sota_100x_family_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            if covered_families < 10 {
                failures.push(format!(
                    "requirement `cpu-only-100x-proof` has {covered_families} covered workload family/families with a CPU-SOTA 100x contract; needs at least 10"
                ));
            }
            let required_hundred_x = matrix
                .get("required_cpu_sota_100x_families")
                .and_then(serde_json::Value::as_array)
                .map_or(0, Vec::len);
            if required_hundred_x < 10 {
                failures.push(format!(
                    "requirement `cpu-only-100x-proof` matrix lists only {required_hundred_x} required 100x family/families; needs at least 10 release 100x families"
                ));
            }
            let missing_required_hundred_x = matrix
                .get("missing_required_cpu_sota_100x_families")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if missing_required_hundred_x != 0 {
                failures.push(format!(
                    "requirement `cpu-only-100x-proof` matrix reports {missing_required_hundred_x} missing required 100x family/families"
                ));
            }
            let contract_cases = matrix
                .get("cpu_sota_100x_contract_cases")
                .and_then(serde_json::Value::as_array)
                .map(|cases| {
                    cases
                        .iter()
                        .filter_map(serde_json::Value::as_str)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if contract_cases.is_empty() {
                failures.push(
                    "requirement `cpu-only-100x-proof` workload matrix does not list the active 100x contract case ids"
                        .to_string(),
                );
            }
            if let Some(proof) =
                first_json_evidence(requirement, base_dir, "cpu-only-100x-proof.json", failures)
            {
                let proof_blockers = proof
                    .get("blockers")
                    .and_then(serde_json::Value::as_array)
                    .map_or(0, Vec::len);
                if proof_blockers != 0 {
                    failures.push(format!(
                        "requirement `cpu-only-100x-proof` aggregate proof reports {proof_blockers} blocker(s)"
                    ));
                }
                if proof
                    .get("source_fingerprint")
                    .and_then(serde_json::Value::as_str)
                    .is_none_or(str::is_empty)
                {
                    failures.push(
                        "requirement `cpu-only-100x-proof` aggregate proof must preserve source_fingerprint"
                            .to_string(),
                    );
                }
                if proof.get("git").is_none_or(serde_json::Value::is_null) {
                    failures.push(
                        "requirement `cpu-only-100x-proof` aggregate proof must preserve git provenance object"
                            .to_string(),
                    );
                }
                let required_proof_cases = proof
                    .get("required_cpu_sota_100x_cases")
                    .and_then(serde_json::Value::as_array)
                    .map_or(0, Vec::len);
                if required_proof_cases < 10 {
                    failures.push(format!(
                        "requirement `cpu-only-100x-proof` aggregate proof lists {required_proof_cases} required 100x case(s); needs at least 10 release 100x cases"
                    ));
                }
                let missing_proof_cases = proof
                    .get("missing_required_cpu_sota_100x_cases")
                    .and_then(serde_json::Value::as_array)
                    .map_or(usize::MAX, Vec::len);
                if missing_proof_cases != 0 {
                    failures.push(format!(
                        "requirement `cpu-only-100x-proof` aggregate proof reports {missing_proof_cases} missing required 100x case(s)"
                    ));
                }
                let proof_contract_case_count = proof
                    .get("cases")
                    .and_then(serde_json::Value::as_array)
                    .map(|cases| {
                        for required_case in [
                            "release.condition_eval.1m",
                            "release.string_bitmap_scatter.1m",
                            "release.offset_count_aggregation.1m",
                            "release.entropy_window.1m",
                            "release.quantified_condition_loops.1m",
                            "release.alias_reaching_def.1m",
                            "release.ifds_witness.1m",
                            "release.c_ast_traversal.1m",
                            "release.megakernel_queue.1m",
                            "release.egraph_saturation.1m",
                            "sparse.compaction.count.1m",
                        ] {
                            if !cases.iter().any(|case| {
                                case.get("id").and_then(serde_json::Value::as_str)
                                    == Some(required_case)
                            }) {
                                failures.push(format!(
                                    "requirement `cpu-only-100x-proof` aggregate proof is missing required case `{required_case}`"
                                ));
                            }
                        }
                        cases.iter().filter(|case| {
                            case.get("id")
                                .and_then(serde_json::Value::as_str)
                                .is_some_and(|id| contract_cases.contains(&id))
                        }).count()
                    })
                    .unwrap_or(0);
                if proof_contract_case_count < 10 {
                    failures.push(format!(
                        "requirement `cpu-only-100x-proof` aggregate proof artifact contains {proof_contract_case_count} case(s) listed in cpu_sota_100x_contract_cases; needs at least 10"
                    ));
                }
                let aggregate_contract_cases = proof
                    .get("cpu_sota_100x_contract_case_count")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                if aggregate_contract_cases < 10 {
                    failures.push(format!(
                        "requirement `cpu-only-100x-proof` aggregate proof has {aggregate_contract_cases} CPU-SOTA 100x contract case(s); needs at least 10"
                    ));
                }
                let aggregate_passing_cases = proof
                    .get("cpu_sota_100x_passing_case_count")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                if aggregate_passing_cases < 10 {
                    failures.push(format!(
                        "requirement `cpu-only-100x-proof` aggregate proof has {aggregate_passing_cases} passing CPU-SOTA 100x case(s); needs at least 10"
                    ));
                }
                let min_wall_samples = proof
                    .get("min_wall_samples")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                if min_wall_samples < 30 {
                    failures.push(format!(
                        "requirement `cpu-only-100x-proof` aggregate proof min_wall_samples={min_wall_samples}; needs at least 30"
                    ));
                }
                let min_baseline_wall_samples = proof
                    .get("min_baseline_wall_samples")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                if min_baseline_wall_samples < 30 {
                    failures.push(format!(
                        "requirement `cpu-only-100x-proof` aggregate proof min_baseline_wall_samples={min_baseline_wall_samples}; needs at least 30"
                    ));
                }
                for field in [
                    "min_wall_p50",
                    "min_wall_p95",
                    "min_wall_p99",
                    "min_baseline_wall_p50",
                    "min_baseline_wall_p95",
                    "min_baseline_wall_p99",
                ] {
                    if proof
                        .get(field)
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0)
                        == 0
                    {
                        failures.push(format!(
                            "requirement `cpu-only-100x-proof` aggregate proof has non-positive `{field}`"
                        ));
                    }
                }
            }
            check_benchmark_evidence_reports(
                requirement,
                base_dir,
                "cpu-only-100x-proof.json",
                true,
                Some(100.0),
                failures,
            );
        }
        "c-parser-linux-subsystem" => {
            if !requirement.evidence.iter().any(|evidence| {
                evidence.contains("cargo_full")
                    && evidence.contains("c-parser-corpus")
                    && evidence.contains("--corpus")
                    && evidence.contains("-I")
                    && evidence.contains("-D")
            }) {
                failures.push(
                    "requirement `c-parser-linux-subsystem` must include a cargo_full c-parser-corpus command with --corpus, -I, and -D"
                        .to_string(),
                );
            }
            let Some(report) = first_json_evidence(
                requirement,
                base_dir,
                "c-parser-linux-subsystem.json",
                failures,
            ) else {
                return;
            };
            let total = report
                .get("total_files")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let parsed = report
                .get("parsed_files")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let failed = report
                .get("failed_files")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(u64::MAX);
            let source_bytes = report
                .get("total_source_bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let ast_bytes = report
                .get("total_ast_bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let vast_bytes = report
                .get("total_vast_bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let semantic_graph_bytes = report
                .get("total_semantic_graph_bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let blockers = report
                .get("blockers")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if total < 250 {
                failures.push(
                    format!(
                        "requirement `c-parser-linux-subsystem` corpus report contains {total} C file(s), needs at least 250"
                    ),
                );
            }
            if source_bytes < 4 * 1024 * 1024 {
                failures.push(format!(
                    "requirement `c-parser-linux-subsystem` corpus report contains {source_bytes} source byte(s), needs at least 4194304"
                ));
            }
            if report
                .get("linux_subsystem_candidate")
                .and_then(serde_json::Value::as_bool)
                != Some(true)
            {
                failures.push(
                    "requirement `c-parser-linux-subsystem` corpus report must prove linux_subsystem_candidate=true"
                        .to_string(),
                );
            }
            if report
                .get("corpus_root_canonical")
                .and_then(serde_json::Value::as_str)
                .is_none_or(str::is_empty)
            {
                failures.push(
                    "requirement `c-parser-linux-subsystem` corpus report must include corpus_root_canonical"
                        .to_string(),
                );
            }
            if report
                .get("corpus_fingerprint")
                .and_then(serde_json::Value::as_str)
                .is_none_or(|value| !value.starts_with("fnv64:"))
            {
                failures.push(
                    "requirement `c-parser-linux-subsystem` corpus report must include stable corpus_fingerprint"
                        .to_string(),
                );
            }
            if report
                .get("source_collection_mode")
                .and_then(serde_json::Value::as_str)
                != Some("recursive_all_c_files")
            {
                failures.push(
                    "requirement `c-parser-linux-subsystem` corpus report must prove recursive_all_c_files source collection"
                        .to_string(),
                );
            }
            if report
                .get("visited_dir_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                == 0
            {
                failures.push(
                    "requirement `c-parser-linux-subsystem` corpus report must prove nonzero recursive directory traversal"
                        .to_string(),
                );
            }
            for field in ["linux_root", "linux_subsystem", "linux_kbuild_file"] {
                if report
                    .get(field)
                    .and_then(serde_json::Value::as_str)
                    .is_none_or(str::is_empty)
                {
                    failures.push(format!(
                        "requirement `c-parser-linux-subsystem` corpus report must include `{field}` provenance"
                    ));
                }
            }
            if report
                .get("linux_kbuild_file_in_corpus")
                .and_then(serde_json::Value::as_bool)
                != Some(true)
            {
                failures.push(
                    "requirement `c-parser-linux-subsystem` corpus report must prove linux_kbuild_file_in_corpus=true"
                        .to_string(),
                );
            }
            let linux_subsystem = report
                .get("linux_subsystem")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            if !matches!(
                linux_subsystem,
                "kernel" | "fs" | "mm" | "net" | "drivers" | "lib"
            ) {
                failures.push(format!(
                    "requirement `c-parser-linux-subsystem` corpus report has unsupported linux_subsystem `{linux_subsystem}`"
                ));
            }
            let linux_depth = report
                .get("linux_subsystem_depth")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            if linux_depth == 0 {
                failures.push(
                    "requirement `c-parser-linux-subsystem` corpus report must prove linux_subsystem_depth > 0"
                        .to_string(),
                );
            }
            for field in ["include_dirs", "macros"] {
                if report
                    .get(field)
                    .and_then(serde_json::Value::as_array)
                    .is_none_or(Vec::is_empty)
                {
                    failures.push(format!(
                        "requirement `c-parser-linux-subsystem` corpus report `{field}` must be non-empty"
                    ));
                }
            }
            if ast_bytes == 0 || vast_bytes == 0 || semantic_graph_bytes == 0 {
                failures.push(format!(
                    "requirement `c-parser-linux-subsystem` AST/VAST/semantic evidence is incomplete: ast_bytes={ast_bytes}, vast_bytes={vast_bytes}, semantic_graph_bytes={semantic_graph_bytes}"
                ));
            }
            if blockers != 0 {
                failures.push(format!(
                    "requirement `c-parser-linux-subsystem` corpus report still has {blockers} blocker(s)"
                ));
            }
            if failed != 0 || parsed != total {
                failures.push(format!(
                    "requirement `c-parser-linux-subsystem` parsed {parsed}/{total} file(s), failed {failed}; release requires full corpus parse"
                ));
            }
            if let Some(manifest) = first_json_evidence(
                requirement,
                base_dir,
                "linux-subsystem-corpus-manifest.json",
                failures,
            ) {
                let manifest_files = manifest
                    .get("file_count")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                if manifest_files != total {
                    failures.push(format!(
                        "requirement `c-parser-linux-subsystem` corpus manifest lists {manifest_files} file(s), parse report lists {total}"
                    ));
                }
                let manifest_source_bytes = manifest
                    .get("total_source_bytes")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                if manifest_source_bytes != source_bytes {
                    failures.push(format!(
                        "requirement `c-parser-linux-subsystem` corpus manifest lists {manifest_source_bytes} source byte(s), parse report lists {source_bytes}"
                    ));
                }
                for field in [
                    "include_dirs",
                    "macros",
                    "corpus_root_canonical",
                    "linux_subsystem_candidate",
                    "linux_root",
                    "linux_subsystem",
                    "linux_subsystem_depth",
                    "linux_kbuild_file",
                    "linux_kbuild_file_in_corpus",
                    "corpus_fingerprint",
                    "source_collection_mode",
                    "visited_dir_count",
                ] {
                    if manifest.get(field) != report.get(field) {
                        failures.push(format!(
                            "requirement `c-parser-linux-subsystem` corpus manifest `{field}` does not match parse report"
                        ));
                    }
                }
                let manifest_entries = manifest
                    .get("files")
                    .and_then(serde_json::Value::as_array)
                    .map_or(0, Vec::len) as u64;
                if manifest_entries != total {
                    failures.push(format!(
                        "requirement `c-parser-linux-subsystem` corpus manifest has {manifest_entries} file entrie(s), parse report lists {total}"
                    ));
                }
                if let Some(files) = manifest.get("files").and_then(serde_json::Value::as_array) {
                    for file in files {
                        let path = file
                            .get("path")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("<unknown>");
                        if file.get("parsed").and_then(serde_json::Value::as_bool) != Some(true) {
                            failures.push(format!(
                                "requirement `c-parser-linux-subsystem` corpus manifest file `{path}` was not parsed successfully"
                            ));
                            continue;
                        }
                        for field in [
                            "source_bytes",
                            "object_bytes",
                            "ast_bytes",
                            "vast_bytes",
                            "semantic_graph_bytes",
                            "wall_ns",
                        ] {
                            if file
                                .get(field)
                                .and_then(serde_json::Value::as_u64)
                                .unwrap_or(0)
                                == 0
                            {
                                failures.push(format!(
                                    "requirement `c-parser-linux-subsystem` corpus manifest file `{path}` has zero `{field}`"
                                ));
                            }
                        }
                    }
                }
            }
            if let Some(diagnostics) = first_json_evidence(
                requirement,
                base_dir,
                "c-parser-diagnostics-summary.json",
                failures,
            ) {
                let diagnostic_failures = diagnostics
                    .get("failed_files")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(u64::MAX);
                if diagnostic_failures != failed {
                    failures.push(format!(
                        "requirement `c-parser-linux-subsystem` diagnostics report lists {diagnostic_failures} failure(s), parse report lists {failed}"
                    ));
                }
                let diagnostic_entries = diagnostics
                    .get("failures")
                    .and_then(serde_json::Value::as_array)
                    .map_or(usize::MAX, Vec::len);
                if failed == 0 && diagnostic_entries != 0 {
                    failures.push(format!(
                        "requirement `c-parser-linux-subsystem` diagnostics report has {diagnostic_entries} failure entrie(s) while parse report lists zero"
                    ));
                }
            }
            if let Some(throughput) =
                first_json_evidence(requirement, base_dir, "c-parser-throughput.json", failures)
            {
                let throughput_files = throughput
                    .get("parsed_files")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                if throughput_files != parsed {
                    failures.push(format!(
                        "requirement `c-parser-linux-subsystem` throughput report lists {throughput_files} parsed file(s), parse report lists {parsed}"
                    ));
                }
                let throughput_total = throughput
                    .get("total_files")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                if throughput_total != total {
                    failures.push(format!(
                        "requirement `c-parser-linux-subsystem` throughput report lists {throughput_total} total file(s), parse report lists {total}"
                    ));
                }
                let throughput_source_bytes = throughput
                    .get("total_source_bytes")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                if throughput_source_bytes != source_bytes {
                    failures.push(format!(
                        "requirement `c-parser-linux-subsystem` throughput report lists {throughput_source_bytes} source byte(s), parse report lists {source_bytes}"
                    ));
                }
                for field in [
                    "include_dirs",
                    "macros",
                    "corpus_root_canonical",
                    "linux_subsystem_candidate",
                    "linux_root",
                    "linux_subsystem",
                    "linux_subsystem_depth",
                    "linux_kbuild_file",
                    "linux_kbuild_file_in_corpus",
                    "corpus_fingerprint",
                    "source_collection_mode",
                    "visited_dir_count",
                ] {
                    if throughput.get(field) != report.get(field) {
                        failures.push(format!(
                            "requirement `c-parser-linux-subsystem` throughput report `{field}` does not match parse report"
                        ));
                    }
                }
                let wall_ns = throughput
                    .get("wall_ns")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let files_per_second = throughput
                    .get("files_per_second_x1000")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let mib_per_second = throughput
                    .get("mib_per_second_x1000")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                if wall_ns == 0 || files_per_second == 0 || mib_per_second == 0 {
                    failures.push(format!(
                        "requirement `c-parser-linux-subsystem` throughput is incomplete: wall_ns={wall_ns}, files_per_second_x1000={files_per_second}, mib_per_second_x1000={mib_per_second}"
                    ));
                }
            }
        }
        "distributed-parser-coherence" => {
            let Some(matrix) = first_json_evidence(
                requirement,
                base_dir,
                "distributed-parser-map.json",
                failures,
            ) else {
                return;
            };
            let components = matrix
                .get("components")
                .and_then(serde_json::Value::as_array)
                .map_or(0, Vec::len);
            let blockers = matrix
                .get("blockers")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if components == 0 {
                failures.push(
                    "requirement `distributed-parser-coherence` matrix contains zero components"
                        .to_string(),
                );
            }
            let component_ids = matrix
                .get("components")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default();
            for required in [
                "vyre-frontend-c",
                "vyrec",
                "weir",
                "security-analysis-consumer",
                "security-grammar-gen",
            ] {
                if !component_ids.iter().any(|component| {
                    component.get("id").and_then(serde_json::Value::as_str) == Some(required)
                        && component.get("exists").and_then(serde_json::Value::as_bool)
                            == Some(true)
                        && component
                            .get("missing_terms")
                            .and_then(serde_json::Value::as_array)
                            .is_some_and(Vec::is_empty)
                        && component
                            .get("missing_contract_topics")
                            .and_then(serde_json::Value::as_array)
                            .is_some_and(Vec::is_empty)
                        && component
                            .get("required_test_categories")
                            .and_then(serde_json::Value::as_array)
                            .is_some_and(|categories| !categories.is_empty())
                        && component
                            .get("missing_test_categories")
                            .and_then(serde_json::Value::as_array)
                            .is_some_and(Vec::is_empty)
                        && component
                            .get("unresolved_ownership_markers")
                            .and_then(serde_json::Value::as_array)
                            .is_some_and(Vec::is_empty)
                        && component
                            .get("required_files")
                            .and_then(serde_json::Value::as_array)
                            .is_some_and(|files| {
                                !files.is_empty()
                                    && files.iter().all(|file| {
                                        file.get("exists").and_then(serde_json::Value::as_bool)
                                            == Some(true)
                                            && file
                                                .get("source_bytes")
                                                .and_then(serde_json::Value::as_u64)
                                                .unwrap_or(0)
                                                > 0
                                            && file
                                                .get("read_error")
                                                .is_some_and(serde_json::Value::is_null)
                                    })
                            })
                }) {
                    failures.push(format!(
                        "requirement `distributed-parser-coherence` matrix is missing complete component `{required}`"
                    ));
                }
            }
            if blockers != 0 {
                failures.push(format!(
                    "requirement `distributed-parser-coherence` matrix still reports {blockers} blocker(s)"
                ));
            }
            for suffix in [
                "vyre-frontend-c-contracts.json",
                "vyrec-cli-contracts.json",
                "weir-contracts.json",
                "security-analysis-consumer-contracts.json",
                "security-grammar-gen-contracts.json",
            ] {
                check_json_evidence_has_no_blockers(requirement, base_dir, suffix, failures);
                check_parser_contract_evidence(requirement, base_dir, suffix, failures);
            }
        }
        "weir-analysis-integration" => {
            let Some(matrix) = first_json_evidence(
                requirement,
                base_dir,
                "weir-analysis-api-matrix.json",
                failures,
            ) else {
                return;
            };
            let analyses = matrix
                .get("analyses")
                .and_then(serde_json::Value::as_array)
                .map_or(0, Vec::len);
            let blockers = matrix
                .get("blockers")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            let schema_version = matrix
                .get("schema_version")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            if schema_version < 2 {
                failures.push(format!(
                    "requirement `weir-analysis-integration` matrix schema_version is {schema_version}, expected >= 2"
                ));
            }
            if analyses == 0 {
                failures.push(
                    "requirement `weir-analysis-integration` matrix contains zero analyses"
                        .to_string(),
                );
            }
            let inventory_registered = matrix
                .get("inventory_registered_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            if inventory_registered == 0 {
                failures.push(
                    "requirement `weir-analysis-integration` matrix contains zero inventory-registered analyses"
                        .to_string(),
                );
            }
            let required_api_item_count = matrix
                .get("required_api_item_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            if required_api_item_count < 100 {
                failures.push(format!(
                    "requirement `weir-analysis-integration` Weir matrix proves {required_api_item_count} required API item(s), needs at least 100"
                ));
            }
            let missing_api_item_count = matrix
                .get("missing_api_item_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(u64::MAX);
            if missing_api_item_count != 0 {
                failures.push(format!(
                    "requirement `weir-analysis-integration` Weir matrix reports {missing_api_item_count} missing required API item(s)"
                ));
            }
            for (field, label, minimum) in [
                ("property_test_count", "property", 15_u64),
                ("parity_test_count", "parity", 4_u64),
                ("adversarial_test_count", "adversarial", 1_u64),
                ("perf_test_count", "perf/scale", 2_u64),
                ("fuzz_test_count", "fuzz", 1_u64),
                ("gap_test_count", "gap", 1_u64),
            ] {
                let count = matrix
                    .get(field)
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                if count < minimum {
                    failures.push(format!(
                        "requirement `weir-analysis-integration` matrix contains {count} {label} test families; needs at least {minimum}"
                    ));
                }
            }
            let standalone_examples = matrix
                .get("standalone_example_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            if standalone_examples < 2 {
                failures.push(format!(
                    "requirement `weir-analysis-integration` matrix contains {standalone_examples} standalone example(s); needs at least 2 examples outside tests"
                ));
            }
            let standalone_serde_evidence = matrix
                .get("standalone_serde_evidence_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            if standalone_serde_evidence == 0 {
                failures.push(
                    "requirement `weir-analysis-integration` matrix must include at least one standalone serde evidence example for witness/soundness API records"
                        .to_string(),
                );
            }
            let standalone_serde_feature_guards = matrix
                .get("standalone_serde_feature_guard_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            if standalone_serde_feature_guards == 0 {
                failures.push(
                    "requirement `weir-analysis-integration` matrix must prove serde evidence examples declare required-features = [\"serde\"]"
                        .to_string(),
                );
            }
            if matrix
                .get("standalone_examples")
                .and_then(serde_json::Value::as_array)
                .is_none_or(|examples| examples.len() < 2)
            {
                failures.push(
                    "requirement `weir-analysis-integration` matrix must list at least 2 standalone example files"
                        .to_string(),
                );
            }
            let standalone_example_scan_errors = matrix
                .get("standalone_example_scan_errors")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if standalone_example_scan_errors != 0 {
                failures.push(format!(
                    "requirement `weir-analysis-integration` matrix reports {standalone_example_scan_errors} standalone example scan error(s)"
                ));
            }
            if let Some(examples) = matrix
                .get("standalone_examples")
                .and_then(serde_json::Value::as_array)
            {
                for example in examples {
                    let path = example
                        .get("path")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("<unknown>");
                    if example.get("exists").and_then(serde_json::Value::as_bool) != Some(true)
                        || example
                            .get("source_bytes")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(0)
                            == 0
                    {
                        failures.push(format!(
                            "requirement `weir-analysis-integration` standalone example `{path}` must exist and be non-empty"
                        ));
                    }
                    if !example
                        .get("read_error")
                        .is_some_and(serde_json::Value::is_null)
                    {
                        failures.push(format!(
                            "requirement `weir-analysis-integration` standalone example `{path}` read_error must be null"
                        ));
                    }
                    if example.get("has_main").and_then(serde_json::Value::as_bool) != Some(true) {
                        failures.push(format!(
                            "requirement `weir-analysis-integration` standalone example `{path}` must expose runnable fn main"
                        ));
                    }
                    if example
                        .get("uses_weir_crate")
                        .and_then(serde_json::Value::as_bool)
                        != Some(true)
                    {
                        failures.push(format!(
                            "requirement `weir-analysis-integration` standalone example `{path}` must import or reference the weir crate"
                        ));
                    }
                    let api_reference_count = example
                        .get("api_reference_count")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0);
                    if api_reference_count < 2 {
                        failures.push(format!(
                            "requirement `weir-analysis-integration` standalone example `{path}` references {api_reference_count} dataflow API token(s); needs at least 2"
                        ));
                    }
                    if path.ends_with("serde_evidence.rs")
                        && example
                            .get("has_serde_evidence")
                            .and_then(serde_json::Value::as_bool)
                            != Some(true)
                    {
                        failures.push(format!(
                            "requirement `weir-analysis-integration` standalone serde example `{path}` must report has_serde_evidence=true"
                        ));
                    }
                    let unresolved_markers = example
                        .get("unresolved_markers")
                        .and_then(serde_json::Value::as_array)
                        .map_or(usize::MAX, Vec::len);
                    if unresolved_markers != 0 {
                        failures.push(format!(
                            "requirement `weir-analysis-integration` standalone example `{path}` reports {unresolved_markers} unresolved marker(s)"
                        ));
                    }
                }
            }
            let untested_analyses = matrix
                .get("untested_analyses")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if untested_analyses != 0 {
                failures.push(format!(
                    "requirement `weir-analysis-integration` matrix reports {untested_analyses} Weir analysis module(s) without release test coverage"
                ));
            }
            if let Some(entries) = matrix.get("analyses").and_then(serde_json::Value::as_array) {
                for entry in entries {
                    let id = entry
                        .get("id")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("<unknown>");
                    let declares_op_id = entry
                        .get("declares_op_id")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false);
                    let registered = entry
                        .get("inventory_registered")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false);
                    let required_api_items = entry
                        .get("required_api_items")
                        .and_then(serde_json::Value::as_array)
                        .map_or(0, Vec::len);
                    let missing_api_items = entry
                        .get("missing_api_items")
                        .and_then(serde_json::Value::as_array)
                        .map_or(usize::MAX, Vec::len);
                    if required_api_items != 0 && missing_api_items != 0 {
                        failures.push(format!(
                            "requirement `weir-analysis-integration` analysis `{id}` reports {missing_api_items} missing required API item(s)"
                        ));
                    }
                    if id == "soundness" {
                        let required = entry
                            .get("required_policy_items")
                            .and_then(serde_json::Value::as_array)
                            .map_or(0, Vec::len);
                        let missing = entry
                            .get("missing_policy_items")
                            .and_then(serde_json::Value::as_array)
                            .map_or(usize::MAX, Vec::len);
                        if required < 6 || missing != 0 {
                            failures.push(
                                "requirement `weir-analysis-integration` soundness analysis must prove six policy API items and report zero missing items"
                                    .to_string(),
                            );
                        }
                    }
                    if declares_op_id && !registered {
                        failures.push(format!(
                            "requirement `weir-analysis-integration` analysis `{id}` declares OP_ID without inventory registration"
                        ));
                    }
                }
            }
            if blockers != 0 {
                failures.push(format!(
                    "requirement `weir-analysis-integration` matrix still reports {blockers} blocker(s)"
                ));
            }
            check_json_evidence_has_no_blockers(
                requirement,
                base_dir,
                "weir-vyre-integration-tests.json",
                failures,
            );
            if let Some(integration) = first_json_evidence(
                requirement,
                base_dir,
                "weir-vyre-integration-tests.json",
                failures,
            ) {
                let schema_version = integration
                    .get("schema_version")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                if schema_version < 2 {
                    failures.push(format!(
                        "requirement `weir-analysis-integration` Weir integration evidence schema_version is {schema_version}, expected >= 2"
                    ));
                }
            }
            if let Some(readme) = first_json_evidence(
                requirement,
                base_dir,
                "weir-readme-contracts.json",
                failures,
            ) {
                let schema_version = readme
                    .get("schema_version")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                if schema_version < 2 {
                    failures.push(format!(
                        "requirement `weir-analysis-integration` Weir README contract schema_version is {schema_version}, expected >= 2"
                    ));
                }
                if readme.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
                    failures.push(
                        "requirement `weir-analysis-integration` Weir README contract does not prove README.md exists"
                            .to_string(),
                    );
                }
                if readme
                    .get("source_bytes")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
                    == 0
                {
                    failures.push(
                        "requirement `weir-analysis-integration` Weir README contract reports empty README.md"
                            .to_string(),
                    );
                }
                if readme
                    .get("missing_tokens")
                    .and_then(serde_json::Value::as_array)
                    .is_none_or(|tokens| !tokens.is_empty())
                {
                    failures.push(
                        "requirement `weir-analysis-integration` Weir README is missing required API/version tokens"
                            .to_string(),
                    );
                }
                if readme
                    .get("example_count")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
                    == 0
                {
                    failures.push(
                        "requirement `weir-analysis-integration` Weir README has no Rust/TOML example block"
                            .to_string(),
                    );
                }
                let blockers = readme
                    .get("blockers")
                    .and_then(serde_json::Value::as_array)
                    .map_or(usize::MAX, Vec::len);
                if blockers != 0 {
                    failures.push(format!(
                        "requirement `weir-analysis-integration` Weir README contract reports {blockers} blocker(s)"
                    ));
                }
            }
            check_marker_evidence_has_markers(
                requirement,
                base_dir,
                "weir-facts-pass-firing.json",
                failures,
            );
            check_named_cuda_benchmark_report(
                requirement,
                base_dir,
                "dataflow-analysis-release.json",
                failures,
            );
        }
        "conformance-hard-gate" => {
            let Some(matrix) =
                first_json_evidence(requirement, base_dir, "conformance-matrix.json", failures)
            else {
                return;
            };
            let op_count = matrix
                .get("op_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let distinct_op_count = matrix
                .get("distinct_op_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let catalog_required_op_count = matrix
                .get("catalog_required_op_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let catalog_covered_op_count = matrix
                .get("catalog_covered_op_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let missing_catalog_ops = matrix
                .get("missing_catalog_ops")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            let op_matrix_blocked_release_count = matrix
                .get("op_matrix_blocked_release_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(u64::MAX);
            let release_backend_row_count = matrix
                .get("release_backend_row_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let missing_release_backend_rows = matrix
                .get("missing_release_backend_rows")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            let op_matrix_errors = matrix
                .get("op_matrix_errors")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if op_matrix_errors != 0 {
                failures.push(format!(
                    "requirement `conformance-hard-gate` matrix reports {op_matrix_errors} OP_MATRIX read/parse error(s)"
                ));
            }
            let fixture_input_count = matrix
                .get("fixture_input_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let expected_output_count = matrix
                .get("expected_output_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let blockers = matrix
                .get("blockers")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if op_count == 0 {
                failures.push(
                    "requirement `conformance-hard-gate` matrix contains zero op entries"
                        .to_string(),
                );
            }
            if op_count < 49 {
                failures.push(format!(
                    "requirement `conformance-hard-gate` matrix has {op_count} op entries, below release floor 49"
                ));
            }
            if distinct_op_count < 49 {
                failures.push(format!(
                    "requirement `conformance-hard-gate` matrix has {distinct_op_count} distinct op id(s), below release floor 49"
                ));
            }
            if catalog_required_op_count == 0
                || catalog_covered_op_count != catalog_required_op_count
                || missing_catalog_ops != 0
            {
                failures.push(format!(
                    "requirement `conformance-hard-gate` matrix covers {catalog_covered_op_count}/{catalog_required_op_count} OP_MATRIX-required op id(s), missing_catalog_ops={missing_catalog_ops}"
                ));
            }
            if op_matrix_blocked_release_count != 0 {
                failures.push(format!(
                    "requirement `conformance-hard-gate` matrix reports {op_matrix_blocked_release_count} OP_MATRIX release backend row(s) marked blocked_release"
                ));
            }
            let expected_release_backend_rows = catalog_required_op_count.saturating_mul(3);
            if release_backend_row_count < expected_release_backend_rows
                || missing_release_backend_rows != 0
            {
                failures.push(format!(
                    "requirement `conformance-hard-gate` matrix has release_backend_row_count={release_backend_row_count}, expected {expected_release_backend_rows}, missing_release_backend_rows={missing_release_backend_rows}"
                ));
            }
            if fixture_input_count != op_count {
                failures.push(format!(
                    "requirement `conformance-hard-gate` matrix fixture_input_count {fixture_input_count} must equal op_count {op_count}"
                ));
            }
            if expected_output_count != op_count {
                failures.push(format!(
                    "requirement `conformance-hard-gate` matrix expected_output_count {expected_output_count} must equal op_count {op_count}"
                ));
            }
            if matrix
                .get("duplicate_op_ids")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|duplicates| !duplicates.is_empty())
            {
                failures.push(
                    "requirement `conformance-hard-gate` matrix reports duplicate op id(s)"
                        .to_string(),
                );
            }
            let backends = matrix
                .get("dispatch_backends")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default();
            for required in ["cuda", "wgpu", "cpu-ref"] {
                if !backends
                    .iter()
                    .any(|backend| backend.as_str() == Some(required))
                {
                    failures.push(format!(
                        "requirement `conformance-hard-gate` matrix dispatch_backends is missing `{required}`"
                    ));
                }
            }
            let ci_gate_count = matrix
                .get("ci_blocking_gate_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let schema_version = matrix
                .get("schema_version")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            if schema_version < 2 {
                failures.push(format!(
                    "requirement `conformance-hard-gate` matrix schema_version is {schema_version}, expected >= 2"
                ));
            }
            if ci_gate_count < 3 {
                failures.push(format!(
                    "requirement `conformance-hard-gate` matrix reports only {ci_gate_count} blocking CI conformance gate(s), needs at least 3"
                ));
            }
            let ci_gates = matrix
                .get("ci_gates")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default();
            let required_ci_statuses = matrix
                .get("required_ci_statuses")
                .and_then(serde_json::Value::as_array)
                .map_or(0, Vec::len);
            if required_ci_statuses == 0 {
                failures.push(
                    "requirement `conformance-hard-gate` matrix parsed zero required CI status context(s)"
                        .to_string(),
                );
            }
            let missing_required_ci_statuses = matrix
                .get("missing_required_ci_statuses")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if missing_required_ci_statuses != 0 {
                failures.push(format!(
                    "requirement `conformance-hard-gate` matrix reports {missing_required_ci_statuses} required CI status context(s) missing from workflows"
                ));
            }
            let ci_status_scan_errors = matrix
                .get("ci_status_scan_errors")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if ci_status_scan_errors != 0 {
                failures.push(format!(
                    "requirement `conformance-hard-gate` matrix reports {ci_status_scan_errors} CI status scan error(s)"
                ));
            }
            let path_filtered_required_workflows = matrix
                .get("path_filtered_required_workflows")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if path_filtered_required_workflows != 0 {
                failures.push(format!(
                    "requirement `conformance-hard-gate` matrix reports {path_filtered_required_workflows} required workflow(s) still using path filters"
                ));
            }
            let missing_required_workflow_triggers = matrix
                .get("missing_required_workflow_triggers")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if missing_required_workflow_triggers != 0 {
                failures.push(format!(
                    "requirement `conformance-hard-gate` matrix reports {missing_required_workflow_triggers} required workflow(s) missing pull_request + push main trigger coverage"
                ));
            }
            let missing_fail_closed_fanins = matrix
                .get("missing_fail_closed_fanins")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if missing_fail_closed_fanins != 0 {
                failures.push(format!(
                    "requirement `conformance-hard-gate` matrix reports {missing_fail_closed_fanins} required fan-in job(s) missing fail-closed dependency checks"
                ));
            }
            for required_workflow in [
                "/Santh/.github/workflows/conform.yml",
                "/Santh/.github/workflows/gpu-parity.yml",
                "/Santh/.github/workflows/santh-ci.yml",
                "/Santh/.github/workflows/architectural-invariants.yml",
                "/Santh/.github/CI_REQUIRED.md",
                "/Santh/scripts/apply-branch-protection.sh",
                "/Santh/libs/performance/matching/vyre/.github/workflows/conform.yml",
                "/Santh/libs/performance/matching/vyre/.github/workflows/gpu-parity.yml",
            ] {
                if !ci_gates.iter().any(|gate| {
                    gate.get("workflow")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|workflow| workflow.ends_with(required_workflow))
                        && gate.get("present").and_then(serde_json::Value::as_bool) == Some(true)
                        && gate
                            .get("command_present")
                            .and_then(serde_json::Value::as_bool)
                            == Some(true)
                        && gate
                            .get("artifact_check_present")
                            .and_then(serde_json::Value::as_bool)
                            == Some(true)
                }) {
                    failures.push(format!(
                        "requirement `conformance-hard-gate` matrix is missing complete CI workflow `{required_workflow}`"
                    ));
                }
            }
            for required_gate in [
                "conformance matrix release blocker",
                "gpu-release-gate",
                "conform-release-gate",
                "Vyre structural release evidence",
                "Vyre/Weir final release gate",
                "Vyre/Weir final conformance artifact download",
                "Vyre/Weir final benchmark artifact download",
                "Vyre/Weir final conformance staging",
                "Vyre/Weir final benchmark staging",
                "Vyre/Weir final optimization staging",
                "Vyre/Weir final structural evidence",
                "Vyre/Weir final completion audit",
                "vyre-weir-final-release-evidence",
                "architectural-invariants",
                "required_status_checks",
            ] {
                if !ci_gates.iter().any(|gate| {
                    gate.get("gate").and_then(serde_json::Value::as_str) == Some(required_gate)
                        && gate.get("present").and_then(serde_json::Value::as_bool) == Some(true)
                        && gate
                            .get("command_present")
                            .and_then(serde_json::Value::as_bool)
                            == Some(true)
                        && gate
                            .get("artifact_check_present")
                            .and_then(serde_json::Value::as_bool)
                            == Some(true)
                }) {
                    failures.push(format!(
                        "requirement `conformance-hard-gate` matrix is missing complete CI gate `{required_gate}`"
                    ));
                }
            }
            if blockers != 0 {
                failures.push(format!(
                    "requirement `conformance-hard-gate` matrix still reports {blockers} blocker(s)"
                ));
            }
            for suffix in [
                "cuda-conformance.json",
                "wgpu-conformance.json",
                "reference-conformance.json",
                "release-gate-log.json",
            ] {
                check_json_evidence_has_no_blockers(requirement, base_dir, suffix, failures);
            }
            for suffix in [
                "cuda-conformance.json",
                "wgpu-conformance.json",
                "reference-conformance.json",
            ] {
                check_backend_conformance_report(requirement, base_dir, suffix, failures);
            }
            if let Some(log) =
                first_json_evidence(requirement, base_dir, "release-gate-log.json", failures)
            {
                let schema_version = log
                    .get("schema_version")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                if schema_version < 2 {
                    failures.push(format!(
                        "requirement `conformance-hard-gate` release log schema_version={schema_version}; expected schema>=2"
                    ));
                }
                let requested = log
                    .get("requested_backends")
                    .and_then(serde_json::Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                for backend in ["cuda", "wgpu", "cpu-ref"] {
                    if !requested
                        .iter()
                        .any(|entry| entry.as_str() == Some(backend))
                    {
                        failures.push(format!(
                            "requirement `conformance-hard-gate` release log requested_backends is missing `{backend}`"
                        ));
                    }
                }
                let statuses = log
                    .get("artifact_statuses")
                    .and_then(serde_json::Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                for artifact in [
                    "cuda-conformance.json",
                    "wgpu-conformance.json",
                    "reference-conformance.json",
                ] {
                    if !statuses.iter().any(|status| {
                        status.get("path").and_then(serde_json::Value::as_str) == Some(artifact)
                            && status.get("exists").and_then(serde_json::Value::as_bool)
                                == Some(true)
                            && status
                                .get("bytes")
                                .and_then(serde_json::Value::as_u64)
                                .unwrap_or(0)
                                > 0
                            && status
                                .get("read_error")
                                .is_some_and(serde_json::Value::is_null)
                    }) {
                        failures.push(format!(
                            "requirement `conformance-hard-gate` release log does not prove non-empty readable artifact `{artifact}`"
                        ));
                    }
                }
            }
        }
        "release-hygiene" => {
            let Some(matrix) =
                first_json_evidence(requirement, base_dir, "hygiene-matrix.json", failures)
            else {
                return;
            };
            let scanned = matrix
                .get("scanned_files")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let blockers = matrix
                .get("blockers")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if scanned == 0 {
                failures
                    .push("requirement `release-hygiene` scanned zero source files".to_string());
            }
            let finding_count = matrix
                .get("findings")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            let summary_count = matrix
                .get("finding_summary")
                .and_then(serde_json::Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.get("count").and_then(serde_json::Value::as_u64))
                        .sum::<u64>() as usize
                })
                .unwrap_or(usize::MAX);
            if finding_count != summary_count {
                failures.push(format!(
                    "requirement `release-hygiene` finding_summary count {summary_count} does not match findings count {finding_count}"
                ));
            }
            check_hygiene_release_surface_coverage("release-hygiene", &matrix, failures);
            for required_root in [
                "libs/performance/matching/vyre",
                "libs/dataflow/weir",
                "tools/vyrec",
                "libs/tools/security-analysis-consumer",
                "libs/shared/security-grammar-gen",
            ] {
                if !matrix
                    .get("scanned_roots")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|roots| {
                        roots.iter().any(|root| {
                            root.as_str()
                                .is_some_and(|root| root.contains(required_root))
                        })
                    })
                {
                    failures.push(format!(
                        "requirement `release-hygiene` scanned_roots is missing `{required_root}`"
                    ));
                }
            }
            if blockers != 0 {
                failures.push(format!(
                    "requirement `release-hygiene` matrix still reports {blockers} blocker(s)"
                ));
            }
            for suffix in [
                "no-stubs-scan.json",
                "no-hidden-fallback-scan.json",
                "resource-bound-scan.json",
                "error-surface-scan.json",
                "cargo-wrapper-scan.json",
                "audit-location-scan.json",
                "public-doc-scan.json",
                "test-hygiene-scan.json",
            ] {
                check_json_evidence_has_no_blockers(requirement, base_dir, suffix, failures);
            }
        }
        "modular-test-architecture" | "exhaustive-verification" => {
            let Some(matrix) =
                first_json_evidence(requirement, base_dir, "test-matrix.json", failures)
            else {
                return;
            };
            let test_files = matrix
                .get("test_files")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let blockers = matrix
                .get("blockers")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if test_files == 0 {
                failures.push(format!(
                    "requirement `{}` test matrix contains zero test files",
                    requirement.id
                ));
            }
            for (field, label) in [
                ("vyre_test_files", "Vyre"),
                ("weir_test_files", "Weir"),
                ("vyrec_test_files", "tools/vyrec"),
            ] {
                if matrix
                    .get(field)
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
                    == 0
                {
                    failures.push(format!(
                        "requirement `{}` test matrix contains zero {label} release-surface test files",
                        requirement.id
                    ));
                }
            }
            if blockers != 0 {
                failures.push(format!(
                    "requirement `{}` test matrix still reports {blockers} blocker(s)",
                    requirement.id
                ));
            }
            let layers = matrix
                .get("layers")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default();
            for required in [
                "unit",
                "integration",
                "property",
                "adversarial",
                "corpus",
                "benchmark",
                "conformance",
                "gap",
                "fuzz",
            ] {
                if !layers.iter().any(|layer| layer.as_str() == Some(required)) {
                    failures.push(format!(
                        "requirement `{}` test matrix is missing `{required}` layer evidence",
                        requirement.id
                    ));
                }
            }
            if !matrix
                .get("oversized_files")
                .and_then(serde_json::Value::as_array)
                .is_some_and(Vec::is_empty)
            {
                failures.push(format!(
                    "requirement `{}` test matrix still contains oversized test files",
                    requirement.id
                ));
            }
            if !matrix
                .get("god_test_candidates")
                .and_then(serde_json::Value::as_array)
                .is_some_and(Vec::is_empty)
            {
                failures.push(format!(
                    "requirement `{}` test matrix still contains monolithic tests.rs candidates",
                    requirement.id
                ));
            }
            check_release_surface_coverage(requirement, &matrix, failures);
            match requirement.id.as_str() {
                "modular-test-architecture" => {
                    for suffix in [
                        "modularization-map.json",
                        "oversized-test-closure.json",
                        "release-surface-suite-coverage.json",
                    ] {
                        check_json_evidence_has_no_blockers(
                            requirement,
                            base_dir,
                            suffix,
                            failures,
                        );
                    }
                    if let Some(modularization) = first_json_evidence(
                        requirement,
                        base_dir,
                        "modularization-map.json",
                        failures,
                    ) {
                        let directories = modularization
                            .get("directories")
                            .and_then(serde_json::Value::as_array)
                            .cloned()
                            .unwrap_or_default();
                        for required_surface in ["vyre", "weir", "vyrec"] {
                            if !directories.iter().any(|directory| {
                                directory.get("surface").and_then(serde_json::Value::as_str)
                                    == Some(required_surface)
                            }) {
                                failures.push(format!(
                                    "requirement `modular-test-architecture` modularization map is missing `{required_surface}` surface directories"
                                ));
                            }
                        }
                    }
                    if let Some(closure) = first_json_evidence(
                        requirement,
                        base_dir,
                        "oversized-test-closure.json",
                        failures,
                    ) {
                        if closure.get("closed").and_then(serde_json::Value::as_bool) != Some(true)
                        {
                            failures.push(
                                "requirement `modular-test-architecture` oversized-test closure is not closed"
                                    .to_string(),
                            );
                        }
                        if closure
                            .get("total_oversized_files")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(u64::MAX)
                            != 0
                        {
                            failures.push(
                                "requirement `modular-test-architecture` oversized-test closure still has oversized files"
                                    .to_string(),
                            );
                        }
                        if closure
                            .get("total_god_test_candidates")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(u64::MAX)
                            != 0
                        {
                            failures.push(
                                "requirement `modular-test-architecture` oversized-test closure still has monolithic tests.rs files"
                                    .to_string(),
                            );
                        }
                    }
                }
                "exhaustive-verification" => {
                    for suffix in [
                        "unit-suite.json",
                        "adversarial-suite.json",
                        "property-suite.json",
                        "conformance-suite.json",
                        "corpus-suite.json",
                        "benchmark-suite.json",
                        "gap-suite.json",
                        "fuzz-suite.json",
                    ] {
                        check_json_evidence_has_no_blockers(
                            requirement,
                            base_dir,
                            suffix,
                            failures,
                        );
                        if let Some(suite) =
                            first_json_evidence(requirement, base_dir, suffix, failures)
                        {
                            if suite
                                .get("file_count")
                                .and_then(serde_json::Value::as_u64)
                                .unwrap_or(0)
                                == 0
                            {
                                failures.push(format!(
                                    "requirement `exhaustive-verification` suite `{suffix}` has zero files"
                                ));
                            }
                            if suite
                                .get("vyre_file_count")
                                .and_then(serde_json::Value::as_u64)
                                .unwrap_or(0)
                                == 0
                            {
                                failures.push(format!(
                                    "requirement `exhaustive-verification` suite `{suffix}` has zero Vyre-side files"
                                ));
                            }
                            if suite
                                .get("dataflow_consumer_file_count")
                                .and_then(serde_json::Value::as_u64)
                                .unwrap_or(0)
                                == 0
                            {
                                failures.push(format!(
                                    "requirement `exhaustive-verification` suite `{suffix}` has zero Weir-side files"
                                ));
                            }
                            if suite
                                .get("vyrec_file_count")
                                .and_then(serde_json::Value::as_u64)
                                .unwrap_or(0)
                                == 0
                            {
                                failures.push(format!(
                                    "requirement `exhaustive-verification` suite `{suffix}` has zero tools/vyrec-side files"
                                ));
                            }
                        }
                    }
                    check_json_evidence_has_no_blockers(
                        requirement,
                        base_dir,
                        "release-surface-suite-coverage.json",
                        failures,
                    );
                }
                _ => {}
            }
        }
        "docs-evidence-linked" => {
            let Some(matrix) =
                first_json_evidence(requirement, base_dir, "docs-matrix.json", failures)
            else {
                return;
            };
            let docs = matrix
                .get("docs")
                .and_then(serde_json::Value::as_array)
                .map_or(0, Vec::len);
            let blockers = matrix
                .get("blockers")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if docs == 0 {
                failures.push(
                    "requirement `docs-evidence-linked` matrix contains zero docs".to_string(),
                );
            }
            if matrix
                .get("curated_proof_docs_preserved")
                .and_then(serde_json::Value::as_bool)
                != Some(true)
            {
                failures.push(
                    "requirement `docs-evidence-linked` docs matrix must prove curated proof Markdown is create-if-missing and not overwritten"
                        .to_string(),
                );
            }
            if matrix
                .get("limitation_findings")
                .and_then(serde_json::Value::as_array)
                .is_none_or(|findings| !findings.is_empty())
            {
                failures.push(
                    "requirement `docs-evidence-linked` matrix reports unapproved limitation or future-scope wording"
                        .to_string(),
                );
            }
            if let Some(entries) = matrix.get("docs").and_then(serde_json::Value::as_array) {
                for entry in entries {
                    let id = entry
                        .get("id")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("<unknown>");
                    if entry
                        .get("evidence_artifact_ref_count")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0)
                        == 0
                    {
                        failures.push(format!(
                            "requirement `docs-evidence-linked` doc `{id}` has zero concrete evidence artifact references"
                        ));
                    }
                    if entry
                        .get("missing_evidence_artifact_refs")
                        .and_then(serde_json::Value::as_array)
                        .is_none_or(|refs| !refs.is_empty())
                    {
                        failures.push(format!(
                            "requirement `docs-evidence-linked` doc `{id}` references missing evidence artifacts"
                        ));
                    }
                    if entry
                        .get("missing_topics")
                        .and_then(serde_json::Value::as_array)
                        .is_none_or(|topics| !topics.is_empty())
                    {
                        failures.push(format!(
                            "requirement `docs-evidence-linked` doc `{id}` has missing topics"
                        ));
                    }
                    if entry
                        .get("unresolved_markers")
                        .and_then(serde_json::Value::as_array)
                        .is_none_or(|markers| !markers.is_empty())
                    {
                        failures.push(format!(
                            "requirement `docs-evidence-linked` doc `{id}` has unresolved markers"
                        ));
                    }
                }
            }
            if blockers != 0 {
                failures.push(format!(
                    "requirement `docs-evidence-linked` matrix still reports {blockers} blocker(s)"
                ));
            }
            if let Some(readme) = first_json_evidence(
                requirement,
                base_dir,
                "vyre-readme-contracts.json",
                failures,
            ) {
                check_readme_contract("docs-evidence-linked", "Vyre", &readme, failures);
            }
            for suffix in [
                "vyre-readme-proof.md",
                "weir-readme-proof.md",
                "parser-doc-proof.md",
                "benchmark-doc-proof.md",
                "conformance-doc-proof.md",
                "release-notes.md",
            ] {
                check_markdown_evidence_ready(requirement, base_dir, suffix, failures);
            }
        }
        "crate-metadata" => {
            let Some(matrix) =
                first_json_evidence(requirement, base_dir, "metadata-matrix.json", failures)
            else {
                return;
            };
            let packages = matrix
                .get("packages")
                .and_then(serde_json::Value::as_array)
                .map_or(0, Vec::len);
            let blockers = matrix
                .get("blockers")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if packages == 0 {
                failures
                    .push("requirement `crate-metadata` matrix contains zero packages".to_string());
            }
            if matrix
                .get("root_patch_section_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(u64::MAX)
                != 0
            {
                failures.push(
                    "requirement `crate-metadata` matrix must report zero root [patch.crates-io] sections"
                        .to_string(),
                );
            }
            for (field, label) in [
                ("publishable_package_count", "publishable package"),
                ("vyre_package_count", "Vyre package"),
                ("weir_package_count", "Weir package"),
                (
                    "parser_release_surface_count",
                    "parser release-surface package",
                ),
                (
                    "non_publishable_release_surface_count",
                    "non-publishable release-surface package",
                ),
            ] {
                if matrix
                    .get(field)
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
                    == 0
                {
                    failures.push(format!(
                        "requirement `crate-metadata` matrix contains zero {label}(s)"
                    ));
                }
            }
            if matrix
                .get("parser_release_surface_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                < 2
            {
                failures.push(
                    "requirement `crate-metadata` matrix must include both parser release surfaces: vyrec and vyre-frontend-c"
                        .to_string(),
                );
            }
            let missing_required = matrix
                .get("missing_required_release_surfaces")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if missing_required != 0 {
                failures.push(format!(
                    "requirement `crate-metadata` matrix has {missing_required} missing required release surface(s)"
                ));
            }
            if let Some(entries) = matrix.get("packages").and_then(serde_json::Value::as_array) {
                if !entries.iter().any(|entry| {
                    entry.get("name").and_then(serde_json::Value::as_str) == Some("vyrec")
                        && entry.get("version").and_then(serde_json::Value::as_str) == Some("0.1.0")
                        && entry.get("readme").and_then(serde_json::Value::as_str)
                            == Some("README.md")
                        && entry
                            .get("release_surface")
                            .and_then(serde_json::Value::as_str)
                            == Some("parser-cli")
                }) {
                    failures.push(
                        "requirement `crate-metadata` matrix must include vyrec 0.1.0 parser-cli with README metadata"
                            .to_string(),
                    );
                }
                if !entries.iter().any(|entry| {
                    entry.get("name").and_then(serde_json::Value::as_str) == Some("vyre-frontend-c")
                        && entry.get("version").and_then(serde_json::Value::as_str) == Some("0.4.2")
                        && entry.get("readme").and_then(serde_json::Value::as_str)
                            == Some("README.md")
                        && entry
                            .get("release_kind")
                            .and_then(serde_json::Value::as_str)
                            == Some("non-publishable-release-surface")
                        && entry
                            .get("release_surface")
                            .and_then(serde_json::Value::as_str)
                            == Some("c-frontend")
                }) {
                    failures.push(
                        "requirement `crate-metadata` matrix must include vyre-frontend-c 0.4.2 as a c-frontend non-publishable release surface with README metadata"
                            .to_string(),
                    );
                }
                for (package_name, backend_surface) in [
                    ("vyre-driver-cuda", "cuda-backend"),
                    ("vyre-driver-wgpu", "wgpu-backend"),
                ] {
                    if !entries.iter().any(|entry| {
                        entry.get("name").and_then(serde_json::Value::as_str) == Some(package_name)
                            && entry.get("version").and_then(serde_json::Value::as_str)
                                == Some("0.4.2")
                            && entry.get("readme").and_then(serde_json::Value::as_str)
                                == Some("README.md")
                            && entry
                                .get("release_kind")
                                .and_then(serde_json::Value::as_str)
                                == Some("publishable-crate")
                            && entry
                                .get("release_surface")
                                .and_then(serde_json::Value::as_str)
                                == Some(backend_surface)
                    }) {
                        failures.push(format!(
                            "requirement `crate-metadata` matrix must include {package_name} 0.4.2 as a publishable {backend_surface} release surface with README metadata"
                        ));
                    }
                }
                for entry in entries {
                    let name = entry
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("<unknown>");
                    let release_kind = entry
                        .get("release_kind")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("");
                    if release_kind == "internal-tooling" {
                        continue;
                    }
                    let release_group = entry
                        .get("release_group")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("");
                    let expected = entry
                        .get("expected_version")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("");
                    let version = entry
                        .get("version")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("");
                    if expected.is_empty() || version != expected {
                        failures.push(format!(
                            "requirement `crate-metadata` package `{name}` release_group `{release_group}` has version `{version}`, expected `{expected}`"
                        ));
                    }
                    if entry
                        .get("example_count")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0)
                        == 0
                    {
                        failures.push(format!(
                            "requirement `crate-metadata` release package `{name}` has zero examples or README usage blocks"
                        ));
                    }
                    if release_kind == "publishable-crate"
                        && entry
                            .get("has_runnable_example")
                            .and_then(serde_json::Value::as_bool)
                            != Some(true)
                    {
                        failures.push(format!(
                            "requirement `crate-metadata` publishable release package `{name}` has no runnable examples/*.rs"
                        ));
                    }
                    if release_kind == "publishable-crate"
                        && entry
                            .get("has_api_referencing_example")
                            .and_then(serde_json::Value::as_bool)
                            != Some(true)
                    {
                        failures.push(format!(
                            "requirement `crate-metadata` publishable release package `{name}` has no API-referencing examples/*.rs"
                        ));
                    }
                }
            }
            if blockers != 0 {
                failures.push(format!(
                    "requirement `crate-metadata` matrix still reports {blockers} blocker(s)"
                ));
            }
            let Some(features) =
                first_json_evidence(requirement, base_dir, "feature-matrix.json", failures)
            else {
                return;
            };
            let feature_packages = features
                .get("packages")
                .and_then(serde_json::Value::as_array)
                .map_or(0, Vec::len);
            let feature_blockers = features
                .get("blockers")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if feature_packages == 0 {
                failures.push(
                    "requirement `crate-metadata` feature matrix contains zero packages"
                        .to_string(),
                );
            }
            if feature_blockers != 0 {
                failures.push(format!(
                    "requirement `crate-metadata` feature matrix still reports {feature_blockers} blocker(s)"
                ));
            }
            let missing_required_feature_packages = features
                .get("missing_required_release_packages")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if missing_required_feature_packages != 0 {
                failures.push(format!(
                    "requirement `crate-metadata` feature matrix has {missing_required_feature_packages} missing required release package(s)"
                ));
            }
            let feature_entries = features
                .get("packages")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default();
            for (package, required_features) in [
                ("vyre", &["cuda", "wgpu"][..]),
                ("vyre-driver-cuda", &["cuda"][..]),
                ("vyre-driver-wgpu", &["wgpu"][..]),
                ("weir", &["default", "serde"][..]),
            ] {
                let Some(entry) = feature_entries.iter().find(|entry| {
                    entry.get("name").and_then(serde_json::Value::as_str) == Some(package)
                }) else {
                    failures.push(format!(
                        "requirement `crate-metadata` feature matrix is missing `{package}`"
                    ));
                    continue;
                };
                let features = entry
                    .get("features")
                    .and_then(serde_json::Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                for required in required_features {
                    if !features
                        .iter()
                        .any(|feature| feature.as_str() == Some(required))
                    {
                        failures.push(format!(
                            "requirement `crate-metadata` package `{package}` is missing feature `{required}`"
                        ));
                    }
                }
            }
            if !feature_entries
                .iter()
                .any(|entry| entry.get("name").and_then(serde_json::Value::as_str) == Some("vyrec"))
            {
                failures.push(
                    "requirement `crate-metadata` feature matrix is missing `vyrec`".to_string(),
                );
            }
            for package in ["vyre", "vyre-driver-cuda", "vyre-driver-wgpu"] {
                let Some(entry) = feature_entries.iter().find(|entry| {
                    entry.get("name").and_then(serde_json::Value::as_str) == Some(package)
                }) else {
                    failures.push(format!(
                        "requirement `crate-metadata` feature matrix is missing `{package}`"
                    ));
                    continue;
                };
                if entry
                    .get("default_feature_members")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|members| !members.is_empty())
                {
                    failures.push(format!(
                        "requirement `crate-metadata` package `{package}` default feature set must be empty"
                    ));
                }
            }
            let Some(package_readiness) =
                first_json_evidence(requirement, base_dir, "publish-readiness.json", failures)
            else {
                return;
            };
            let package_blockers = package_readiness
                .get("blockers")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if package_blockers != 0 {
                failures.push(format!(
                    "requirement `crate-metadata` package readiness still reports {package_blockers} blocker(s)"
                ));
            }
            if package_readiness
                .get("release_train")
                .and_then(|train| train.get("cuda_release_path"))
                .and_then(serde_json::Value::as_bool)
                != Some(true)
            {
                failures.push(
                    "requirement `crate-metadata` package readiness must mark CUDA as the release path"
                        .to_string(),
                );
            }
            let publish_order = package_readiness
                .get("publish_order")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default();
            if publish_order.len() < 20 {
                failures.push(format!(
                    "requirement `crate-metadata` package readiness publish_order has only {} package(s)",
                    publish_order.len()
                ));
            }
            for required in [
                "vyre-macros",
                "vyre-spec",
                "vyre-foundation",
                "vyre-driver-cuda",
                "vyre-driver-wgpu",
                "vyre",
                "vyre-harness",
                "weir",
                "vyre-libs",
            ] {
                if !publish_order.iter().any(|entry| {
                    entry.get("package").and_then(serde_json::Value::as_str) == Some(required)
                }) {
                    failures.push(format!(
                        "requirement `crate-metadata` package readiness publish_order is missing `{required}`"
                    ));
                }
            }
            for field in ["missing_metadata_packages", "extra_metadata_packages"] {
                let count = package_readiness
                    .get(field)
                    .and_then(serde_json::Value::as_array)
                    .map_or(usize::MAX, Vec::len);
                if count != 0 {
                    failures.push(format!(
                        "requirement `crate-metadata` package readiness field `{field}` has {count} entrie(s)"
                    ));
                }
            }
            for field in ["dependency_order_edges", "versioned_local_dependencies"] {
                if package_readiness
                    .get(field)
                    .and_then(serde_json::Value::as_array)
                    .is_none_or(Vec::is_empty)
                {
                    failures.push(format!(
                        "requirement `crate-metadata` package readiness field `{field}` is empty"
                    ));
                }
            }
        }
        "final-completion-audit" => {
            if !requirement.evidence.iter().any(|evidence| {
                evidence.contains("cargo_full")
                    && evidence.contains("xtask")
                    && evidence.contains("release-evidence")
            }) {
                failures.push(
                    "requirement `final-completion-audit` must include the cargo_full release-evidence command as concrete evidence"
                        .to_string(),
                );
            }
            if !requirement.evidence.iter().any(|evidence| {
                evidence.contains("cargo_full")
                    && evidence.contains("xtask")
                    && evidence.contains("release-completion-audit")
            }) {
                failures.push(
                    "requirement `final-completion-audit` must include the cargo_full release-completion-audit command as concrete evidence"
                        .to_string(),
                );
            }
            if !requirement.evidence.iter().any(|evidence| {
                evidence.contains("cargo_full")
                    && evidence.contains("xtask")
                    && evidence.contains("vyre-release-gate")
            }) {
                failures.push(
                    "requirement `final-completion-audit` must include the cargo_full vyre-release-gate command as concrete evidence"
                        .to_string(),
                );
            }

            let Some(audit) =
                first_json_evidence(requirement, base_dir, "completion-audit.json", failures)
            else {
                return;
            };
            let blockers = audit
                .get("blockers")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            let open = audit
                .get("blocked_or_open_requirements")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(u64::MAX);
            if blockers != 0 || open != 0 {
                failures.push(format!(
                    "requirement `final-completion-audit` still reports {blockers} blocker(s) and {open} open requirement(s)"
                ));
            }

            let Some(run) =
                first_json_evidence(requirement, base_dir, "release-evidence-run.json", failures)
            else {
                return;
            };
            check_release_evidence_run(requirement, &run, failures);
        }
        _ => {}
    }
}

fn check_hygiene_release_surface_coverage(
    requirement_id: &str,
    matrix: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    let Some(coverage) = matrix.get("release_surface_coverage") else {
        failures.push(format!(
            "requirement `{requirement_id}` hygiene matrix is missing release_surface_coverage"
        ));
        return;
    };
    for field in [
        "vyre_workspace",
        "cuda_driver_crate",
        "wgpu_driver_crate",
        "weir_crate",
        "vyrec_tool",
        "security_analysis_tool",
        "security_grammar_gen",
        "release_scripts",
        "github_workflows",
        "branch_protection_controls",
    ] {
        if coverage.get(field).and_then(serde_json::Value::as_bool) != Some(true) {
            failures.push(format!(
                "requirement `{requirement_id}` hygiene release_surface_coverage.{field} must be true"
            ));
        }
    }
    for (field, required) in [
        (
            "resource_bound_patterns",
            &[
                "std_thread_sleep",
                "thread_sleep",
                "tokio_sleep",
                "unbounded_read",
            ][..],
        ),
        (
            "hidden_fallback_patterns",
            &[
                "silent_gpu_skip",
                "silent_gpu_skipped",
                "gpu_unavailable_skip",
                "cfg_not_gpu",
                "cpu_fallback",
                "software_fallback",
                "fallback_dispatch",
                "falling_back_to_cpu",
                "fallback_to_cpu",
                "synthetic_gpu_timing",
                "fake_gpu_timing_formula",
            ][..],
        ),
        (
            "release_tooling_patterns",
            &[
                "raw_workspace_cargo",
                "invalid_cargo_full_xtask",
                "heredoc",
                "missing_cargo_wrapper",
            ][..],
        ),
    ] {
        let values = coverage.get(field).and_then(serde_json::Value::as_array);
        for required_value in required {
            if !values.is_some_and(|values| {
                values
                    .iter()
                    .any(|value| value.as_str() == Some(*required_value))
            }) {
                failures.push(format!(
                    "requirement `{requirement_id}` hygiene release_surface_coverage.{field} is missing `{required_value}`"
                ));
            }
        }
    }
}

fn check_release_surface_coverage(
    requirement: &Requirement,
    matrix: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    let Some(surfaces) = matrix
        .get("surface_coverages")
        .and_then(serde_json::Value::as_array)
    else {
        failures.push(format!(
            "requirement `{}` test matrix is missing release surface coverage",
            requirement.id
        ));
        return;
    };
    if surfaces.len() != 3 {
        failures.push(format!(
            "requirement `{}` test matrix must report exactly Vyre, Weir, and tools/vyrec surface coverage",
            requirement.id
        ));
    }
    for surface_id in ["vyre", "weir", "vyrec"] {
        let Some(surface) = surfaces.iter().find(|surface| {
            surface.get("surface").and_then(serde_json::Value::as_str) == Some(surface_id)
        }) else {
            failures.push(format!(
                "requirement `{}` test matrix is missing `{surface_id}` surface coverage",
                requirement.id
            ));
            continue;
        };
        for (field, label) in [
            ("file_count", "test files"),
            ("assertion_count", "assertions"),
            ("entrypoint_count", "test entrypoints"),
        ] {
            if surface
                .get(field)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                == 0
            {
                failures.push(format!(
                    "requirement `{}` `{surface_id}` release surface has zero {label}",
                    requirement.id
                ));
            }
        }
        let missing_layers = surface
            .get("missing_layers")
            .and_then(serde_json::Value::as_array)
            .map_or(usize::MAX, Vec::len);
        if missing_layers != 0 {
            failures.push(format!(
                "requirement `{}` `{surface_id}` release surface reports {missing_layers} missing test layer(s)",
                requirement.id
            ));
        }
        let blockers = surface
            .get("blockers")
            .and_then(serde_json::Value::as_array)
            .map_or(usize::MAX, Vec::len);
        if blockers != 0 {
            failures.push(format!(
                "requirement `{}` `{surface_id}` release surface reports {blockers} blocker(s)",
                requirement.id
            ));
        }
    }
}

fn check_release_evidence_run(
    requirement: &Requirement,
    run: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    const REQUIRED_GENERATORS: &[(&str, &[&str])] = &[
        (
            "version-matrix",
            &["version-matrix.json", "release-tag-plan.json"],
        ),
        ("backend-matrix", &["backend-matrix.json"]),
        ("conformance-matrix", &["conformance-matrix.json"]),
        ("release-workload-matrix", &["release-workload-matrix.json"]),
        (
            "hygiene-matrix",
            &[
                "hygiene-matrix.json",
                "no-stubs-scan.json",
                "no-hidden-fallback-scan.json",
                "resource-bound-scan.json",
                "error-surface-scan.json",
                "cargo-wrapper-scan.json",
                "audit-location-scan.json",
                "public-doc-scan.json",
                "test-hygiene-scan.json",
            ],
        ),
        (
            "test-matrix",
            &[
                "test-matrix.json",
                "modularization-map.json",
                "oversized-test-closure.json",
                "release-surface-suite-coverage.json",
                "unit-suite.json",
                "adversarial-suite.json",
                "property-suite.json",
                "conformance-suite.json",
                "corpus-suite.json",
                "benchmark-suite.json",
                "gap-suite.json",
                "fuzz-suite.json",
            ],
        ),
        (
            "docs-matrix",
            &[
                "docs-matrix.json",
                "vyre-readme-contracts.json",
                "release-notes-version-story.md",
                "cuda-release-path.md",
                "wgpu-fallback-proof.md",
                "megakernel-default-proof.md",
                "optimization-proof.md",
                "egraph-saturation.md",
                "c-parser-linux-proof.md",
                "distributed-parser-coherence.md",
                "weir-integration.md",
                "test-architecture.md",
                "vyre-readme-proof.md",
                "weir-readme-proof.md",
                "parser-doc-proof.md",
                "benchmark-doc-proof.md",
                "conformance-doc-proof.md",
                "release-notes.md",
                "crate-metadata-proof.md",
                "release-hygiene-proof.md",
                "cpu-only-100x-proof.md",
            ],
        ),
        ("metadata-matrix", &["metadata-matrix.json"]),
        ("feature-matrix", &["feature-matrix.json"]),
        (
            "optimization-corpus",
            &[
                "optimization-corpus.json",
                "optimization-corpus-contracts.json",
                "optimization-family-manifest.json",
                "optimization-analysis-fixtures.json",
                "optimization-case-manifest.json",
            ],
        ),
        (
            "optimization-matrix",
            &[
                "optimization-integration-matrix.json",
                "alias-aware-dse.json",
                "alias-aware-stlf.json",
                "alias-aware-licm.json",
                "alias-aware-fusion-fission.json",
                "weir-facts-pass-firing.json",
                "egraph-saturation-matrix.json",
                "egraph-semantic-contracts.json",
            ],
        ),
        (
            "parser-coherence",
            &[
                "distributed-parser-map.json",
                "vyre-frontend-c-contracts.json",
                "vyrec-cli-contracts.json",
                "weir-contracts.json",
                "security-analysis-consumer-contracts.json",
                "security-grammar-gen-contracts.json",
            ],
        ),
        (
            "weir-matrix",
            &[
                "weir-analysis-api-matrix.json",
                "weir-vyre-integration-tests.json",
                "weir-readme-contracts.json",
            ],
        ),
    ];

    let total = run
        .get("total_commands")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let schema_version = run
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let successful = run
        .get("successful_commands")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let required = run
        .get("required_command_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let command_failures = run
        .get("command_failures")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    let artifact_failures = run
        .get("artifact_failures")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    let blockers = run
        .get("blockers")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if schema_version < 2
        || total < 13
        || required < 13
        || successful != total
        || command_failures != 0
        || artifact_failures != 0
        || blockers != 0
    {
        failures.push(format!(
            "requirement `{}` release-evidence-run must be schema>=2 and clean: schema_version={schema_version}, total={total}, required={required}, successful={successful}, command_failures={command_failures}, artifact_failures={artifact_failures}, blockers={blockers}",
            requirement.id
        ));
    }

    let commands = run
        .get("commands")
        .and_then(serde_json::Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    for (generator, expected_artifacts) in REQUIRED_GENERATORS {
        let Some(command) = commands.iter().find(|command| {
            command
                .get("args")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|args| {
                    args.iter()
                        .any(|arg| arg.as_str().is_some_and(|arg| arg == *generator))
                })
        }) else {
            failures.push(format!(
                "requirement `{}` release-evidence-run is missing generator `{generator}`",
                requirement.id
            ));
            continue;
        };

        let status = command
            .get("status")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if status != "success" {
            failures.push(format!(
                "requirement `{}` release-evidence-run generator `{generator}` status is `{status}`, expected `success`",
                requirement.id
            ));
        }

        let artifacts = command
            .get("expected_artifacts")
            .and_then(serde_json::Value::as_array)
            .map_or(&[][..], Vec::as_slice);
        let artifact_statuses = command
            .get("artifact_statuses")
            .and_then(serde_json::Value::as_array)
            .map_or(&[][..], Vec::as_slice);
        for expected in *expected_artifacts {
            if !artifacts.iter().any(|artifact| {
                artifact
                    .as_str()
                    .is_some_and(|artifact| artifact.ends_with(expected))
            }) {
                failures.push(format!(
                    "requirement `{}` release-evidence-run generator `{generator}` does not declare expected artifact `{expected}`",
                    requirement.id
                ));
            }
            let Some(status) = artifact_statuses.iter().find(|status| {
                status
                    .get("path")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|path| path.ends_with(expected))
            }) else {
                failures.push(format!(
                    "requirement `{}` release-evidence-run generator `{generator}` has no artifact status for `{expected}`",
                    requirement.id
                ));
                continue;
            };
            let exists = status
                .get("exists")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            let bytes = status
                .get("bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let read_error = status.get("read_error");
            let read_error_is_clean = read_error.is_some_and(serde_json::Value::is_null);
            if !exists || bytes == 0 || !read_error_is_clean {
                failures.push(format!(
                    "requirement `{}` release-evidence-run generator `{generator}` artifact `{expected}` exists={exists} bytes={bytes} read_error={}",
                    requirement.id,
                    read_error
                        .map(serde_json::Value::to_string)
                        .unwrap_or_else(|| "<missing>".to_string())
                ));
            }
        }
    }
}

fn check_optimization_analysis_fixture_manifest(
    value: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    let missing_required = value
        .get("missing_required_families")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_required != 0 {
        failures.push(format!(
            "requirement `optimization-corpus-4096` analysis fixture manifest has {missing_required} missing required family/families"
        ));
    }
    let total_fixture_cases = value
        .get("total_fixture_cases")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let total_triggered_cases = value
        .get("total_triggered_cases")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if total_fixture_cases < 512 || total_triggered_cases != total_fixture_cases {
        failures.push(format!(
            "requirement `optimization-corpus-4096` analysis fixture manifest has total_fixture_cases={total_fixture_cases}, total_triggered_cases={total_triggered_cases}; needs 512 fully-triggered A13-A16 cases"
        ));
    }
    let families = value
        .get("families")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    for required in [
        "A13-coalesce-fixture",
        "A14-shared-mem-promote-fixture",
        "A15-bank-conflict-fixture",
        "A16-vec-pack-fixture",
    ] {
        let Some(family) = families.iter().find(|family| {
            family.get("family").and_then(serde_json::Value::as_str) == Some(required)
        }) else {
            failures.push(format!(
                "requirement `optimization-corpus-4096` analysis fixture manifest is missing `{required}`"
            ));
            continue;
        };
        let cases = family
            .get("cases")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let triggered = family
            .get("triggered_cases")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let analysis_sites = family
            .get("analysis_sites")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if cases < 128 || triggered != cases || analysis_sites < cases {
            failures.push(format!(
                "requirement `optimization-corpus-4096` analysis fixture `{required}` has cases={cases}, triggered_cases={triggered}, analysis_sites={analysis_sites}; needs at least 128 cases, every case triggered, and at least one analysis site per case"
            ));
        }
        match required {
            "A13-coalesce-fixture" => {
                for field in [
                    "coalesced_unit_stride_sites",
                    "strided_sites",
                    "broadcast_sites",
                ] {
                    if family
                        .get(field)
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0)
                        == 0
                    {
                        failures.push(format!(
                            "requirement `optimization-corpus-4096` A13 analysis fixture has zero `{field}`"
                        ));
                    }
                }
            }
            "A14-shared-mem-promote-fixture" => {
                for field in ["shared_mem_candidates", "shared_mem_tile_bytes"] {
                    if family
                        .get(field)
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0)
                        == 0
                    {
                        failures.push(format!(
                            "requirement `optimization-corpus-4096` A14 analysis fixture has zero `{field}`"
                        ));
                    }
                }
            }
            "A15-bank-conflict-fixture" => {
                for field in ["bank_conflict_sites", "bank_conflict_critical_sites"] {
                    if family
                        .get(field)
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0)
                        == 0
                    {
                        failures.push(format!(
                            "requirement `optimization-corpus-4096` A15 analysis fixture has zero `{field}`"
                        ));
                    }
                }
            }
            "A16-vec-pack-fixture" => {
                for field in ["vec_pack_chains", "vec_pack_ops_eliminated"] {
                    if family
                        .get(field)
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0)
                        == 0
                    {
                        failures.push(format!(
                            "requirement `optimization-corpus-4096` A16 analysis fixture has zero `{field}`"
                        ));
                    }
                }
            }
            _ => {}
        }
    }
}

fn first_json_evidence(
    requirement: &Requirement,
    base_dir: &Path,
    suffix: &str,
    failures: &mut Vec<String>,
) -> Option<serde_json::Value> {
    let evidence = requirement
        .evidence
        .iter()
        .find(|path| path.ends_with(suffix) && !path.starts_with("cargo_full "));
    let Some(evidence) = evidence else {
        failures.push(format!(
            "requirement `{}` needs JSON evidence ending in `{suffix}`",
            requirement.id
        ));
        return None;
    };
    let path = resolve_manifest_path(base_dir, evidence);
    let text = match read_text_bounded(&path) {
        Ok(text) => text,
        Err(error) => {
            failures.push(format!(
                "requirement `{}` failed to read JSON evidence `{}`: {error}",
                requirement.id,
                path.display()
            ));
            return None;
        }
    };
    match serde_json::from_str(&text) {
        Ok(value) => Some(value),
        Err(error) => {
            failures.push(format!(
                "requirement `{}` evidence `{}` is invalid JSON: {error}",
                requirement.id,
                path.display()
            ));
            None
        }
    }
}

fn read_json_artifact_ref(
    requirement: &Requirement,
    base_dir: &Path,
    artifact: &str,
    failures: &mut Vec<String>,
) -> Option<serde_json::Value> {
    let path = resolve_artifact_path(base_dir, artifact);
    let text = match read_text_bounded(&path) {
        Ok(text) => text,
        Err(error) => {
            failures.push(format!(
                "requirement `{}` failed to read referenced JSON artifact `{}`: {error}",
                requirement.id,
                path.display()
            ));
            return None;
        }
    };
    match serde_json::from_str(&text) {
        Ok(value) => Some(value),
        Err(error) => {
            failures.push(format!(
                "requirement `{}` referenced artifact `{}` is invalid JSON: {error}",
                requirement.id,
                path.display()
            ));
            None
        }
    }
}

fn check_workload_matrix_artifact_coverage(
    requirement: &Requirement,
    base_dir: &Path,
    matrix: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    let manifest_evidence = requirement
        .evidence
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let Some(families) = matrix.get("families").and_then(serde_json::Value::as_array) else {
        failures.push(format!(
            "requirement `{}` workload matrix has no families array",
            requirement.id
        ));
        return;
    };

    let mut required_family_count = 0usize;
    let mut covered_family_count = 0usize;
    let mut artifact_paths = BTreeSet::new();
    let mut workload_numbers = BTreeSet::new();
    for family in families {
        let id = family
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        let required = family
            .get("required")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        if !required {
            continue;
        }
        required_family_count += 1;
        let matched_cases = family
            .get("matched_cases")
            .and_then(serde_json::Value::as_array)
            .map(|cases| {
                cases
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_default();
        if matched_cases.is_empty() {
            failures.push(format!(
                "requirement `{}` workload family `{id}` has no matched release benchmark cases",
                requirement.id
            ));
            continue;
        }
        let dispatch_policy = family
            .get("dispatch_policy")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if dispatch_policy.is_empty() {
            failures.push(format!(
                "requirement `{}` workload family `{id}` has no dispatch_policy",
                requirement.id
            ));
        }
        let bench_target_ids = family
            .get("bench_target_ids")
            .and_then(serde_json::Value::as_array)
            .map(|targets| {
                targets
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if bench_target_ids.is_empty()
            || !bench_target_ids
                .iter()
                .all(|target| target.starts_with("release.workload."))
        {
            failures.push(format!(
                "requirement `{}` workload family `{id}` must list release BENCH_TARGETS.toml target ids",
                requirement.id
            ));
        }
        if id == "megakernel-queued-batches" && dispatch_policy != "megakernel" {
            failures.push(format!(
                "requirement `{}` workload family `{id}` must use megakernel dispatch policy, found `{dispatch_policy}`",
                requirement.id
            ));
        }
        if dispatch_policy != "megakernel" {
            let justification = family
                .get("non_megakernel_justification")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            if justification.len() < 48 {
                failures.push(format!(
                    "requirement `{}` workload family `{id}` uses non-megakernel dispatch policy `{dispatch_policy}` without a concrete architectural or measured justification",
                    requirement.id
                ));
            }
        }
        covered_family_count += 1;
        if family
            .get("max_cpu_sota_min_speedup_x")
            .and_then(serde_json::Value::as_f64)
            .is_some_and(|speedup| speedup >= 100.0)
        {
            let hundred_x_cases = family
                .get("cpu_sota_100x_cases")
                .and_then(serde_json::Value::as_array)
                .map_or(0, Vec::len);
            if hundred_x_cases == 0 {
                failures.push(format!(
                    "requirement `{}` workload family `{id}` declares a 100x contract but lists no cpu_sota_100x_cases",
                    requirement.id
                ));
            }
        }
        let workload_number = family
            .get("release_plan_workload")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if workload_number == 0 || !workload_numbers.insert(workload_number) {
            failures.push(format!(
                "requirement `{}` workload family `{id}` has invalid or duplicate release_plan_workload `{workload_number}`",
                requirement.id
            ));
        }
        let Some(artifact) = family
            .get("evidence_artifact")
            .and_then(serde_json::Value::as_str)
        else {
            failures.push(format!(
                "requirement `{}` workload family `{id}` has no evidence_artifact",
                requirement.id
            ));
            continue;
        };
        if !artifact_paths.insert(artifact) {
            failures.push(format!(
                "requirement `{}` workload family `{id}` reuses evidence artifact `{artifact}`",
                requirement.id
            ));
        }
        if !artifact.starts_with("release/evidence/benchmarks/workload-") {
            failures.push(format!(
                "requirement `{}` workload family `{id}` artifact `{artifact}` is not a workload benchmark artifact",
                requirement.id
            ));
        }
        let manifest_artifact = artifact.strip_prefix("release/").unwrap_or(artifact);
        if !manifest_evidence.contains(manifest_artifact) {
            failures.push(format!(
                "requirement `{}` workload family `{id}` artifact `{manifest_artifact}` is not listed in release evidence manifest",
                requirement.id
            ));
        }
        if let Some(command) = family
            .get("benchmark_command")
            .and_then(serde_json::Value::as_str)
        {
            if !command.contains("cargo_full") || !command.contains(artifact) {
                failures.push(format!(
                    "requirement `{}` workload family `{id}` benchmark command does not use cargo_full and its evidence artifact",
                    requirement.id
                ));
            }
        } else {
            failures.push(format!(
                "requirement `{}` workload family `{id}` has no benchmark_command",
                requirement.id
            ));
        }
        if family
            .get("fair_cpu_sota_baseline_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            failures.push(format!(
                "requirement `{}` workload family `{id}` has no fair CPU-SOTA baseline crate bound to CUDA",
                requirement.id
            ));
        }
        if family
            .get("cpu_sota_baseline_names")
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len)
            == 0
        {
            failures.push(format!(
                "requirement `{}` workload family `{id}` has no named CPU-SOTA baseline provenance",
                requirement.id
            ));
        }
        if family
            .get("reproducible_cuda_command")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        {
            failures.push(format!(
                "requirement `{}` workload family `{id}` does not declare a reproducible CUDA benchmark command",
                requirement.id
            ));
        }

        let artifact_path = resolve_manifest_path(base_dir, manifest_artifact);
        let Ok(text) = read_text_bounded(&artifact_path) else {
            failures.push(format!(
                "requirement `{}` workload family `{id}` failed to read benchmark artifact `{}`",
                requirement.id,
                artifact_path.display()
            ));
            continue;
        };
        let Ok(report) = serde_json::from_str::<serde_json::Value>(&text) else {
            failures.push(format!(
                "requirement `{}` workload family `{id}` benchmark artifact `{}` is invalid JSON",
                requirement.id,
                artifact_path.display()
            ));
            continue;
        };
        let report_matches_family = report
            .get("cases")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|cases| {
                cases.iter().any(|case| {
                    case.get("id")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|case_id| matched_cases.contains(case_id))
                })
            });
        if !report_matches_family {
            failures.push(format!(
                "requirement `{}` workload family `{id}` artifact `{}` contains no case from its matched_cases",
                requirement.id,
                artifact_path.display()
            ));
        }
    }

    if required_family_count < 12 {
        failures.push(format!(
            "requirement `{}` matrix declares {required_family_count} required workload families; needs at least 12",
            requirement.id
        ));
    }
    if covered_family_count < 12 {
        failures.push(format!(
            "requirement `{}` has concrete artifacts for {covered_family_count} required workload families; needs at least 12",
            requirement.id
        ));
    }
}

fn check_release_bench_targets(
    requirement: &Requirement,
    base_dir: &Path,
    failures: &mut Vec<String>,
) {
    let path = base_dir.join("../docs/optimization/BENCH_TARGETS.toml");
    let text = match read_text_bounded(&path) {
        Ok(text) => text,
        Err(error) => {
            failures.push(format!(
                "requirement `{}` failed to read canonical benchmark targets `{}`: {error}",
                requirement.id,
                path.display()
            ));
            return;
        }
    };
    let target_count = text.matches("[[target]]").count();
    if target_count < 17 {
        failures.push(format!(
            "requirement `{}` benchmark target table contains {target_count} target(s); needs at least 17 including release workloads and optimization-proof targets",
            requirement.id
        ));
    }
    for required in [
        "release.workload.condition_eval",
        "release.workload.string_bitmap_scatter",
        "release.workload.offset_count_aggregation",
        "release.workload.pe_metadata",
        "release.workload.entropy_window",
        "release.workload.for_any_all_n",
        "release.workload.alias_reaching_def",
        "release.workload.ifds_witness",
        "release.workload.callgraph_reachability",
        "release.workload.c_ast_traversal",
        "release.workload.megakernel_stream",
        "release.workload.egraph_saturation",
        "release.workload.conformance_sparse_readback",
        "release.optimization.lower_rewrite_impact",
        "release.optimization.foundation_optimizer_impact",
    ] {
        if !text.contains(required) {
            failures.push(format!(
                "requirement `{}` benchmark target table is missing release target `{required}`",
                requirement.id
            ));
        }
    }
    if text.matches("baseline_class_values").count() != 1
        || !text.contains("\"cpu_sota\"")
        || !text.contains("min_speedup_over_cpu_sota")
    {
        failures.push(format!(
            "requirement `{}` benchmark target table must declare CPU-SOTA baseline classes and speedup thresholds",
            requirement.id
        ));
    }
}

fn check_benchmark_evidence_reports(
    requirement: &Requirement,
    base_dir: &Path,
    name_fragment: &str,
    require_cuda: bool,
    min_speedup_x: Option<f64>,
    failures: &mut Vec<String>,
) {
    let mut matched = 0usize;
    for evidence in &requirement.evidence {
        if !evidence.ends_with(".json") || !evidence.contains(name_fragment) {
            continue;
        }
        if evidence.ends_with("release-workload-matrix.json") {
            continue;
        }
        matched += 1;
        let path = resolve_manifest_path(base_dir, evidence);
        let text = match read_text_bounded(&path) {
            Ok(text) => text,
            Err(error) => {
                failures.push(format!(
                    "requirement `{}` failed to read benchmark evidence `{}`: {error}",
                    requirement.id,
                    path.display()
                ));
                continue;
            }
        };
        let report = match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(report) => report,
            Err(error) => {
                failures.push(format!(
                    "requirement `{}` benchmark evidence `{}` is invalid JSON: {error}",
                    requirement.id,
                    path.display()
                ));
                continue;
            }
        };
        check_single_benchmark_report(
            requirement,
            &path,
            &report,
            require_cuda,
            min_speedup_x,
            failures,
        );
    }
    if matched == 0 {
        failures.push(format!(
            "requirement `{}` has no benchmark evidence JSON matching `{name_fragment}`",
            requirement.id
        ));
    }
}

fn check_single_benchmark_report(
    requirement: &Requirement,
    path: &Path,
    report: &serde_json::Value,
    require_cuda: bool,
    min_speedup_x: Option<f64>,
    failures: &mut Vec<String>,
) {
    let failed = report
        .get("summary")
        .and_then(|summary| summary.get("failed"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    if failed != 0 {
        failures.push(format!(
            "requirement `{}` benchmark `{}` reports {failed} failed case(s)",
            requirement.id,
            path.display()
        ));
    }
    let selected_backend = report
        .get("selected_backend")
        .and_then(serde_json::Value::as_str);
    if require_cuda && selected_backend != Some("cuda") {
        failures.push(format!(
            "requirement `{}` benchmark `{}` selected backend `{:?}`, expected cuda",
            requirement.id,
            path.display(),
            selected_backend
        ));
    }
    if require_cuda {
        check_benchmark_cuda_environment_provenance(
            requirement,
            &path.display().to_string(),
            report,
            failures,
        );
    }
    check_benchmark_reproducibility_provenance(
        requirement,
        &path.display().to_string(),
        report,
        failures,
    );
    let Some(cases) = report.get("cases").and_then(serde_json::Value::as_array) else {
        failures.push(format!(
            "requirement `{}` benchmark `{}` has no cases array",
            requirement.id,
            path.display()
        ));
        return;
    };
    if cases.is_empty() {
        failures.push(format!(
            "requirement `{}` benchmark `{}` has zero cases",
            requirement.id,
            path.display()
        ));
    }
    for case in cases {
        let id = case
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if require_cuda
            && case.get("backend_id").and_then(serde_json::Value::as_str) != Some("cuda")
        {
            failures.push(format!(
                "requirement `{}` benchmark `{}` case `{id}` did not run on cuda",
                requirement.id,
                path.display()
            ));
        }
        if case.get("contract").is_none_or(serde_json::Value::is_null) {
            failures.push(format!(
                "requirement `{}` benchmark `{}` case `{id}` has no performance contract",
                requirement.id,
                path.display()
            ));
        }
        let metrics = case.get("metrics").and_then(serde_json::Value::as_object);
        let wall_samples = metrics
            .and_then(|metrics| metric_samples(metrics.get("wall_ns")))
            .unwrap_or(0);
        if wall_samples < 30 {
            failures.push(format!(
                "requirement `{}` benchmark `{}` case `{id}` has {wall_samples} wall_ns sample(s), needs at least 30",
                requirement.id,
                path.display()
            ));
        }
        require_benchmark_metric_percentiles(
            &requirement.id,
            &path.display().to_string(),
            id,
            metrics,
            "wall_ns",
            failures,
        );
        let contract_passed = case
            .get("performance")
            .and_then(|performance| performance.get("contract_passed"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        if !contract_passed {
            failures.push(format!(
                "requirement `{}` benchmark `{}` case `{id}` did not pass its performance contract",
                requirement.id,
                path.display()
            ));
        }
        if let Some(required_speedup) = min_speedup_x {
            if !case_has_cpu_sota_contract(case, required_speedup) {
                failures.push(format!(
                    "requirement `{}` benchmark `{}` case `{id}` must carry a CPU-SOTA performance contract with min_speedup_x >= {required_speedup:.2}",
                    requirement.id,
                    path.display()
                ));
            }
            let metrics = case.get("metrics").and_then(serde_json::Value::as_object);
            let wall_p50 = metrics.and_then(|metrics| metric_p50(metrics.get("wall_ns")));
            let baseline_p50 =
                metrics.and_then(|metrics| metric_p50(metrics.get("baseline_wall_ns")));
            require_benchmark_metric_percentiles(
                &requirement.id,
                &path.display().to_string(),
                id,
                metrics,
                "baseline_wall_ns",
                failures,
            );
            let wall_samples = metrics
                .and_then(|metrics| metric_samples(metrics.get("wall_ns")))
                .unwrap_or(0);
            if wall_samples < 30 {
                failures.push(format!(
                    "requirement `{}` benchmark `{}` case `{id}` has {wall_samples} wall_ns sample(s), needs at least 30",
                    requirement.id,
                    path.display()
                ));
            }
            match (wall_p50, baseline_p50) {
                (Some(wall), Some(baseline)) if wall > 0.0 => {
                    let measured_speedup = baseline / wall;
                    if measured_speedup < required_speedup {
                        failures.push(format!(
                            "requirement `{}` benchmark `{}` case `{id}` end-to-end p50 speedup was {measured_speedup:.2}x, needs at least {required_speedup:.2}x",
                            requirement.id,
                            path.display()
                        ));
                    }
                }
                _ => failures.push(format!(
                    "requirement `{}` benchmark `{}` case `{id}` must include p50 wall_ns and baseline_wall_ns metrics for the 100x proof",
                    requirement.id,
                    path.display()
                )),
            }
            let speedup = case
                .get("performance")
                .and_then(|performance| performance.get("speedup_x"))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            if speedup < required_speedup {
                failures.push(format!(
                    "requirement `{}` benchmark `{}` case `{id}` observed {speedup:.2}x, needs at least {required_speedup:.2}x",
                    requirement.id,
                    path.display()
                ));
            }
        }
    }
}

fn case_has_cpu_sota_contract(case: &serde_json::Value, required_speedup: f64) -> bool {
    case.get("contract")
        .and_then(|contract| contract.get("baselines"))
        .and_then(serde_json::Value::as_array)
        .is_some_and(|baselines| {
            baselines.iter().any(|baseline| {
                baseline.get("class").and_then(serde_json::Value::as_str) == Some("CpuSota")
                    && baseline
                        .get("min_speedup_x")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0)
                        >= required_speedup
            })
        })
}

fn require_no_hidden_backend_fallback_findings(
    requirement_id: &str,
    matrix: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    let Some(scan_errors) = matrix
        .get("hidden_fallback_scan_errors")
        .and_then(serde_json::Value::as_array)
    else {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix is missing hidden_fallback_scan_errors"
        ));
        return;
    };
    if !scan_errors.is_empty() {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix reports {} hidden fallback scan error(s)",
            scan_errors.len()
        ));
    }
    let Some(findings) = matrix
        .get("hidden_fallback_findings")
        .and_then(serde_json::Value::as_array)
    else {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix is missing hidden_fallback_findings"
        ));
        return;
    };
    if !findings.is_empty() {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix reports {} hidden fallback finding(s)",
            findings.len()
        ));
    }
}

fn check_backend_matrix_schema(
    requirement_id: &str,
    matrix: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    let schema_version = matrix
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if schema_version < 2 {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix schema_version is {schema_version}, expected >= 2"
        ));
    }
}

fn check_backend_gpu_probe(
    requirement_id: &str,
    matrix: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    if matrix
        .get("gpu_probe")
        .and_then(|probe| probe.get("nvidia_smi_ok"))
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix did not prove nvidia-smi GPU visibility"
        ));
    }
    let devices = matrix
        .get("gpu_probe")
        .and_then(|probe| probe.get("nvidia_smi_devices"))
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if devices == 0 {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix lists zero nvidia-smi devices"
        ));
    }
    let release_floor_device = matrix
        .get("gpu_probe")
        .and_then(|probe| probe.get("nvidia_smi_device_details"))
        .and_then(serde_json::Value::as_array)
        .is_some_and(|devices| {
            devices.iter().any(|device| {
                device
                    .get("memory_total_mib")
                    .and_then(serde_json::Value::as_u64)
                    .is_some_and(|mib| mib >= 16 * 1024)
                    && matches!(
                        (
                            device
                                .get("compute_capability_major")
                                .and_then(serde_json::Value::as_u64),
                            device
                                .get("compute_capability_minor")
                                .and_then(serde_json::Value::as_u64),
                        ),
                        (Some(major), Some(minor)) if (major, minor) >= (8, 0)
                    )
            })
        });
    if !release_floor_device {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix gpu_probe.nvidia_smi_device_details must include a CUDA GPU with >=16384 MiB VRAM and compute capability >=8.0"
        ));
    }
    for field in ["nvidia_driver_version", "nvidia_cuda_version"] {
        if matrix
            .get("gpu_probe")
            .and_then(|probe| probe.get(field))
            .and_then(serde_json::Value::as_str)
            .is_none_or(str::is_empty)
        {
            failures.push(format!(
                "requirement `{requirement_id}` backend matrix gpu_probe.{field} must be recorded"
            ));
        }
    }
}

fn check_backend_acquire_entry(
    requirement_id: &str,
    matrix: &serde_json::Value,
    backend_id: &str,
    failures: &mut Vec<String>,
) {
    let backends = matrix
        .get("backends")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    if !backends.iter().any(|backend| {
        backend.get("id").and_then(serde_json::Value::as_str) == Some(backend_id)
            && backend
                .get("dispatches")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
            && backend
                .get("acquire_ok")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
    }) {
        failures.push(format!(
            "requirement `{requirement_id}` backend `{backend_id}` must dispatch and acquire successfully"
        ));
    }
}

fn check_preferred_backend_gpu_only(
    requirement_id: &str,
    matrix: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    if matrix
        .get("preferred_backend_gpu_only")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix must prove preferred runtime acquisition is GPU-only"
        ));
    }
    let preferred = matrix
        .get("preferred_backend_id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if !matches!(preferred, "cuda" | "wgpu") {
        failures.push(format!(
            "requirement `{requirement_id}` preferred_backend_id `{preferred}` must be cuda or wgpu, never cpu-ref/reference"
        ));
    }
}

fn check_backend_feature_markers(
    requirement_id: &str,
    matrix: &serde_json::Value,
    field: &str,
    minimum: usize,
    failures: &mut Vec<String>,
) {
    let Some(markers) = matrix.get(field).and_then(serde_json::Value::as_array) else {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix is missing `{field}`"
        ));
        return;
    };
    if markers.len() < minimum {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix `{field}` has {} marker(s), needs at least {minimum}",
            markers.len()
        ));
    }
    let required_ids: &[&str] = match field {
        "cuda_feature_markers" => &[
            "tensor-core-fragment",
            "ldmatrix-cp-async",
            "predicated-execution",
            "instruction-scheduling",
            "ptx-vector-load-gap-scheduling",
            "ptx-compute-load-gap-scheduling",
            "ptx-vector-load-fusion",
            "ptx-vector-store-fusion",
            "async-copy-emitter",
            "mma-emitter",
            "cuda-resident-dispatch",
            "cuda-resident-io",
            "cuda-graph-launch",
            "cuda-module-cache",
            "cuda-ptx-source-cache",
            "cuda-ptx-target-probe",
            "megakernel-paired-speculation",
        ],
        "wgpu_feature_markers" => &[
            "wgpu-persistent-engine",
            "wgpu-megakernel-dispatcher",
            "wgpu-readback-ring",
            "wgpu-async-dispatch-prefetch",
            "wgpu-dispatch-scratch-reuse",
            "wgpu-disk-cache",
            "wgpu-no-cpu-fallback-test",
            "megakernel-paired-speculation",
        ],
        _ => &[],
    };
    for required_id in required_ids {
        if !markers.iter().any(|marker| {
            marker.get("id").and_then(serde_json::Value::as_str) == Some(*required_id)
        }) {
            failures.push(format!(
                "requirement `{requirement_id}` backend matrix `{field}` is missing required marker `{required_id}`"
            ));
        }
    }
    for marker in markers {
        let id = marker
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if marker.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
            failures.push(format!(
                "requirement `{requirement_id}` backend marker `{id}` in `{field}` does not exist"
            ));
        }
        if marker
            .get("source_bytes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            failures.push(format!(
                "requirement `{requirement_id}` backend marker `{id}` in `{field}` is empty"
            ));
        }
        if marker
            .get("missing_tokens")
            .and_then(serde_json::Value::as_array)
            .is_none_or(|tokens| !tokens.is_empty())
        {
            failures.push(format!(
                "requirement `{requirement_id}` backend marker `{id}` in `{field}` has missing implementation tokens"
            ));
        }
        if marker
            .get("unresolved_markers")
            .and_then(serde_json::Value::as_array)
            .is_none_or(|markers| !markers.is_empty())
        {
            failures.push(format!(
                "requirement `{requirement_id}` backend marker `{id}` in `{field}` has unresolved markers"
            ));
        }
    }
}

fn check_readme_contract(
    requirement_id: &str,
    product: &str,
    value: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    if value.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
        failures.push(format!(
            "requirement `{requirement_id}` {product} README contract does not prove README.md exists"
        ));
    }
    if value
        .get("source_bytes")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0)
        == 0
    {
        failures.push(format!(
            "requirement `{requirement_id}` {product} README contract reports empty README.md"
        ));
    }
    if value
        .get("missing_tokens")
        .and_then(serde_json::Value::as_array)
        .is_none_or(|tokens| !tokens.is_empty())
    {
        failures.push(format!(
            "requirement `{requirement_id}` {product} README is missing required API/version tokens"
        ));
    }
    if value
        .get("example_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0)
        == 0
    {
        failures.push(format!(
            "requirement `{requirement_id}` {product} README has no example block"
        ));
    }
    let blockers = value
        .get("blockers")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if blockers != 0 {
        failures.push(format!(
            "requirement `{requirement_id}` {product} README contract reports {blockers} blocker(s)"
        ));
    }
}

const REQUIRED_BENCHMARKED_OPTIMIZATION_FAMILIES: &[&str] = &[
    "algebraic",
    "predicate",
    "egraph",
    "memory-layout",
    "control-flow",
    "vector-layout",
    "A13-coalesce-fixture",
    "A14-shared-mem-promote-fixture",
    "A15-bank-conflict-fixture",
    "A16-vec-pack-fixture",
    "dataflow-analysis-dse",
    "dataflow-analysis-loop-fusion",
    "dataflow-analysis-loop-fission",
    "dataflow-analysis-licm",
];

fn check_before_after_benchmark_report(
    requirement: &Requirement,
    base_dir: &Path,
    suffix: &str,
    failures: &mut Vec<String>,
) {
    let Some(report) = first_json_evidence(requirement, base_dir, suffix, failures) else {
        return;
    };
    let failed = report
        .get("summary")
        .and_then(|summary| summary.get("failed"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    if failed != 0 {
        failures.push(format!(
            "requirement `{}` benchmark `{suffix}` reports {failed} failed case(s)",
            requirement.id
        ));
    }
    let selected_backend = report
        .get("selected_backend")
        .and_then(serde_json::Value::as_str);
    if selected_backend.is_some() && selected_backend != Some("cuda") {
        failures.push(format!(
            "requirement `{}` benchmark `{suffix}` selected backend `{:?}`, expected cuda",
            requirement.id, selected_backend
        ));
    }
    let Some(cases) = report.get("cases").and_then(serde_json::Value::as_array) else {
        failures.push(format!(
            "requirement `{}` benchmark `{suffix}` has no cases array",
            requirement.id
        ));
        return;
    };
    if cases.is_empty() {
        failures.push(format!(
            "requirement `{}` benchmark `{suffix}` has zero cases",
            requirement.id
        ));
    }
    for case in cases {
        let id = case
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        let metrics = case.get("metrics").and_then(serde_json::Value::as_object);
        let has_wall = metrics.is_some_and(|metrics| metrics.contains_key("wall_ns"));
        let has_baseline = metrics.is_some_and(|metrics| metrics.contains_key("baseline_wall_ns"));
        if !has_wall || !has_baseline {
            failures.push(format!(
                "requirement `{}` benchmark `{suffix}` case `{id}` must contain wall_ns and baseline_wall_ns metrics",
                requirement.id
            ));
        }
        let wall_samples = metrics
            .and_then(|metrics| metric_samples(metrics.get("wall_ns")))
            .unwrap_or(0);
        if wall_samples < 30 {
            failures.push(format!(
                "requirement `{}` benchmark `{suffix}` case `{id}` has {wall_samples} wall_ns sample(s), needs at least 30",
                requirement.id
            ));
        }
        let baseline_wall_samples = metrics
            .and_then(|metrics| metric_samples(metrics.get("baseline_wall_ns")))
            .unwrap_or(0);
        if baseline_wall_samples < 30 {
            failures.push(format!(
                "requirement `{}` benchmark `{suffix}` case `{id}` has {baseline_wall_samples} baseline_wall_ns sample(s), needs at least 30",
                requirement.id
            ));
        }
        require_benchmark_metric_percentiles(
            &requirement.id,
            suffix,
            id,
            metrics,
            "wall_ns",
            failures,
        );
        require_benchmark_metric_percentiles(
            &requirement.id,
            suffix,
            id,
            metrics,
            "baseline_wall_ns",
            failures,
        );
        if let Some(metrics) = metrics {
            let wall_p50 = active_gpu_metric_p50(metrics);
            let baseline_p50 = metric_p50(metrics.get("baseline_wall_ns"));
            let egraph_quality_win = suffix == "egraph-before-after.json"
                && metric_p50(metrics.get("egraph_output_ops"))
                    .zip(metric_p50(metrics.get("egraph_baseline_ops_after")))
                    .is_some_and(|(output, baseline)| output < baseline)
                && metric_p50(metrics.get("egraph_applied_rewrites"))
                    .is_some_and(|rewrites| rewrites > 0.0);
            match (wall_p50, baseline_p50) {
                (Some(wall), Some(baseline)) if wall < baseline => {}
                (Some(_), Some(_)) if egraph_quality_win => {}
                (Some(_), Some(_)) if before_after_semantic_win(id, metrics) => {}
                (Some(wall), Some(baseline)) => failures.push(format!(
                    "requirement `{}` benchmark `{suffix}` case `{id}` did not improve p50 wall time: wall={wall:.2}, baseline={baseline:.2}",
                    requirement.id
                )),
                _ => failures.push(format!(
                    "requirement `{}` benchmark `{suffix}` case `{id}` must contain p50 values for wall_ns and baseline_wall_ns",
                    requirement.id
                )),
            }
        }
    }
}

fn metric_p50(metric: Option<&serde_json::Value>) -> Option<f64> {
    let metric = metric?;
    metric_percentile(Some(metric), "p50")
        .or_else(|| metric.as_f64())
        .or_else(|| metric.as_u64().map(|value| value as f64))
}

fn active_gpu_metric_p50(metrics: &serde_json::Map<String, serde_json::Value>) -> Option<f64> {
    metric_p50(metrics.get("dispatch_ns"))
        .or_else(|| metric_p50(metrics.get("kernel_execute_ns")))
        .or_else(|| metric_p50(metrics.get("wall_ns")))
}

fn before_after_semantic_win(
    case_id: &str,
    metrics: &serde_json::Map<String, serde_json::Value>,
) -> bool {
    match case_id {
        "lower.rewrites.impact.corpus" => {
            metric_p50(metrics.get("lower_ops_eliminated")).is_some_and(|value| value > 0.0)
                || metric_p50(metrics.get("lower_optimized_issue_score"))
                    .zip(metric_p50(metrics.get("lower_baseline_issue_score")))
                    .is_some_and(|(optimized, baseline)| optimized < baseline)
        }
        "foundation.optimizer.impact" => {
            metric_p50(metrics.get("optimizer_nodes_eliminated")).is_some_and(|value| value > 0.0)
        }
        "lower.egraph_saturation" => {
            metric_p50(metrics.get("egraph_applied_rewrites")).is_some_and(|value| value > 0.0)
                && metric_p50(metrics.get("egraph_output_ops"))
                    .zip(metric_p50(metrics.get("egraph_baseline_ops_after")))
                    .is_some_and(|(output, baseline)| output < baseline)
        }
        "lower.alias_aware_optimizations" => {
            metric_p50(metrics.get("alias_pass_wins")).is_some_and(|value| value >= 5.0)
        }
        _ => false,
    }
}

fn metric_percentile(metric: Option<&serde_json::Value>, percentile: &str) -> Option<f64> {
    let metric = metric?;
    metric
        .get(percentile)
        .and_then(serde_json::Value::as_f64)
        .or_else(|| {
            metric
                .get(percentile)
                .and_then(serde_json::Value::as_u64)
                .map(|value| value as f64)
        })
}

fn metric_samples(metric: Option<&serde_json::Value>) -> Option<u64> {
    metric?.get("samples").and_then(serde_json::Value::as_u64)
}

fn require_benchmark_metric_percentiles(
    requirement_id: &str,
    benchmark: &str,
    case_id: &str,
    metrics: Option<&serde_json::Map<String, serde_json::Value>>,
    metric_name: &str,
    failures: &mut Vec<String>,
) {
    for percentile in ["p50", "p95", "p99"] {
        let value =
            metrics.and_then(|metrics| metric_percentile(metrics.get(metric_name), percentile));
        if !value.is_some_and(|value| value > 0.0) {
            failures.push(format!(
                "requirement `{requirement_id}` benchmark `{benchmark}` case `{case_id}` must include positive {percentile} {metric_name}"
            ));
        }
    }
}

fn check_named_cuda_benchmark_report(
    requirement: &Requirement,
    base_dir: &Path,
    suffix: &str,
    failures: &mut Vec<String>,
) {
    let Some(report) = first_json_evidence(requirement, base_dir, suffix, failures) else {
        return;
    };
    let path = requirement
        .evidence
        .iter()
        .find(|evidence| evidence.ends_with(suffix))
        .map(|evidence| resolve_manifest_path(base_dir, evidence))
        .unwrap_or_else(|| base_dir.join(suffix));
    check_single_benchmark_report(requirement, &path, &report, true, None, failures);
    if suffix == "dataflow-analysis-release.json" {
        require_case_metric_positive(requirement, suffix, &report, "weir_nodes", failures);
        require_case_metric_positive(requirement, suffix, &report, "weir_bitset_words", failures);
    }
    if suffix == "megakernel-condition-cuda.json" {
        for metric in [
            "megakernel_condition_slots",
            "megakernel_condition_fired",
            "megakernel_condition_slots_per_sec_x1000",
        ] {
            require_case_metric_positive(requirement, suffix, &report, metric, failures);
        }
    }
    if suffix == "megakernel-latency-cuda.json" {
        for metric in [
            "megakernel_slots",
            "megakernel_dispatch_latency_ns",
            "megakernel_slots_per_sec_x1000",
            "megakernel_roundtrip_buffers",
            "megakernel_speculation_samples",
            "megakernel_speculation_adopted",
            "megakernel_speculation_rejected",
            "megakernel_speculation_side_compile_cost_ns",
            "megakernel_speculation_autotune_records",
        ] {
            require_case_metric_positive(requirement, suffix, &report, metric, failures);
        }
    }
}

fn check_json_evidence_has_no_blockers(
    requirement: &Requirement,
    base_dir: &Path,
    suffix: &str,
    failures: &mut Vec<String>,
) {
    let Some(report) = first_json_evidence(requirement, base_dir, suffix, failures) else {
        return;
    };
    let blockers = report
        .get("blockers")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if blockers != 0 {
        failures.push(format!(
            "requirement `{}` evidence `{suffix}` reports {blockers} blocker(s)",
            requirement.id
        ));
    }
}

fn check_marker_evidence_has_markers(
    requirement: &Requirement,
    base_dir: &Path,
    suffix: &str,
    failures: &mut Vec<String>,
) {
    let Some(report) = first_json_evidence(requirement, base_dir, suffix, failures) else {
        return;
    };
    let markers = report
        .get("markers")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if markers == 0 {
        failures.push(format!(
            "requirement `{}` marker evidence `{suffix}` contains zero markers",
            requirement.id
        ));
    }
    for required in required_marker_ids_for_suffix(suffix) {
        if !report
            .get("markers")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|markers| {
                markers.iter().any(|marker| {
                    marker.get("id").and_then(serde_json::Value::as_str) == Some(required)
                })
            })
        {
            failures.push(format!(
                "requirement `{}` marker evidence `{suffix}` is missing required marker `{required}`",
                requirement.id
            ));
        }
    }
    let source_matrix = report
        .get("source_matrix")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if !source_matrix.ends_with("optimization-integration-matrix.json") {
        failures.push(format!(
            "requirement `{}` marker evidence `{suffix}` does not reference optimization-integration-matrix.json",
            requirement.id
        ));
    }
}

fn required_marker_ids_for_suffix(suffix: &str) -> &'static [&'static str] {
    if suffix == "alias-aware-dse.json" {
        &[
            "alias-aware-dse-entrypoint",
            "reaching-def-dse-entrypoint",
            "weir-alias-analysis-api",
            "weir-reaching-def-analysis-api",
            "dataflow-analysis-pipeline-entrypoint",
        ]
    } else if suffix == "alias-aware-stlf.json" {
        &[
            "alias-aware-stlf-entrypoint",
            "reaching-def-stlf-entrypoint",
            "weir-alias-analysis-api",
            "weir-reaching-def-analysis-api",
            "dataflow-analysis-pipeline-entrypoint",
            "dataflow-analysis-stlf-firing-test",
        ]
    } else if suffix == "alias-aware-licm.json" {
        &[
            "alias-aware-licm-entrypoint",
            "reaching-def-licm-entrypoint",
            "weir-alias-analysis-api",
            "weir-reaching-def-analysis-api",
        ]
    } else if suffix == "alias-aware-fusion-fission.json" {
        &[
            "alias-aware-loop-fusion-entrypoint",
            "reaching-def-loop-fusion-entrypoint",
            "alias-aware-loop-fission-entrypoint",
            "reaching-def-loop-fission-entrypoint",
            "weir-alias-analysis-api",
            "weir-reaching-def-analysis-api",
        ]
    } else if suffix == "weir-facts-pass-firing.json" {
        &[
            "alias-aware-dse-entrypoint",
            "reaching-def-dse-entrypoint",
            "alias-aware-stlf-entrypoint",
            "reaching-def-stlf-entrypoint",
            "alias-aware-licm-entrypoint",
            "reaching-def-licm-entrypoint",
            "alias-aware-loop-fusion-entrypoint",
            "reaching-def-loop-fusion-entrypoint",
            "alias-aware-loop-fission-entrypoint",
            "reaching-def-loop-fission-entrypoint",
            "weir-alias-analysis-api",
            "weir-reaching-def-analysis-api",
            "dataflow-analysis-pipeline-entrypoint",
        ]
    } else if suffix == "egraph-saturation-matrix.json"
        || suffix == "egraph-semantic-contracts.json"
    {
        &[
            "egraph-saturation",
            "egraph-canonical-pipeline-entrypoint",
            "egraph-algebraic-reassociation",
            "egraph-bitwise-reassociation",
        ]
    } else {
        &[]
    }
}

fn check_backend_feature_marker_id(
    requirement_id: &str,
    matrix: &serde_json::Value,
    field: &str,
    required_id: &str,
    failures: &mut Vec<String>,
) {
    let Some(markers) = matrix.get(field).and_then(serde_json::Value::as_array) else {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix is missing `{field}`"
        ));
        return;
    };
    let Some(marker) = markers
        .iter()
        .find(|marker| marker.get("id").and_then(serde_json::Value::as_str) == Some(required_id))
    else {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix `{field}` is missing required marker `{required_id}`"
        ));
        return;
    };
    if marker.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix `{field}` marker `{required_id}` does not exist"
        ));
    }
    let missing_tokens = marker
        .get("missing_tokens")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_tokens != 0 {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix `{field}` marker `{required_id}` reports {missing_tokens} missing token(s)"
        ));
    }
    let unresolved_markers = marker
        .get("unresolved_markers")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if unresolved_markers != 0 {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix `{field}` marker `{required_id}` reports {unresolved_markers} unresolved marker(s)"
        ));
    }
}

fn check_parser_contract_evidence(
    requirement: &Requirement,
    base_dir: &Path,
    suffix: &str,
    failures: &mut Vec<String>,
) {
    let Some(report) = first_json_evidence(requirement, base_dir, suffix, failures) else {
        return;
    };
    let expected_component = if suffix == "vyrec-cli-contracts.json" {
        "vyrec"
    } else {
        suffix.strip_suffix("-contracts.json").unwrap_or(suffix)
    };
    let component_id = report
        .get("component_id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if component_id != expected_component {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` has component_id `{component_id}`, expected `{expected_component}`",
            requirement.id
        ));
    }
    if report
        .get("role")
        .and_then(serde_json::Value::as_str)
        .is_none_or(|role| role.is_empty())
    {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` has an empty role",
            requirement.id
        ));
    }
    if report
        .get("root")
        .and_then(serde_json::Value::as_str)
        .is_none_or(|root| root.is_empty())
    {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` has an empty root",
            requirement.id
        ));
    }
    let required_terms = report
        .get("required_terms")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if required_terms == 0 {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` has no required terms",
            requirement.id
        ));
    }
    let missing_terms = report
        .get("missing_terms")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_terms != 0 {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` reports {missing_terms} missing term(s)",
            requirement.id
        ));
    }
    let required_contract_topics = report
        .get("required_contract_topics")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if required_contract_topics == 0 {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` has no required contract topics",
            requirement.id
        ));
    }
    let missing_contract_topics = report
        .get("missing_contract_topics")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_contract_topics != 0 {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` reports {missing_contract_topics} missing contract topic(s)",
            requirement.id
        ));
    }
    let required_test_categories = report
        .get("required_test_categories")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if required_test_categories == 0 {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` has no required test categories",
            requirement.id
        ));
    }
    let missing_test_categories = report
        .get("missing_test_categories")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_test_categories != 0 {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` reports {missing_test_categories} missing test categor(ies)",
            requirement.id
        ));
    }
    let required_evidence_trees = report
        .get("required_evidence_trees")
        .and_then(serde_json::Value::as_array);
    if required_evidence_trees.is_none_or(|trees| trees.len() < 3) {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` must list tests, benches, and fuzz evidence trees",
            requirement.id
        ));
    }
    if let Some(trees) = required_evidence_trees {
        for tree in trees {
            let tree_name = tree
                .get("tree")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("<unknown>");
            if tree.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
                failures.push(format!(
                    "requirement `{}` parser contract `{suffix}` evidence tree `{tree_name}` does not exist",
                    requirement.id
                ));
            }
            let source_bytes = tree
                .get("source_bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            if source_bytes == 0 {
                failures.push(format!(
                    "requirement `{}` parser contract `{suffix}` evidence tree `{tree_name}` has zero source bytes",
                    requirement.id
                ));
            }
            let unreadable = tree
                .get("unreadable_file_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(u64::MAX);
            if unreadable != 0 {
                failures.push(format!(
                    "requirement `{}` parser contract `{suffix}` evidence tree `{tree_name}` has {unreadable} unreadable source file(s)",
                    requirement.id
                ));
            }
        }
    }
    let unresolved_ownership_markers = report
        .get("unresolved_ownership_markers")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if unresolved_ownership_markers != 0 {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` reports {unresolved_ownership_markers} unresolved ownership marker(s)",
            requirement.id
        ));
    }
    let Some(required_files) = report
        .get("required_files")
        .and_then(serde_json::Value::as_array)
    else {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` has no required_files array",
            requirement.id
        ));
        return;
    };
    if required_files.is_empty() {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` has zero required file(s)",
            requirement.id
        ));
    }
    for file in required_files {
        let path = file
            .get("path")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if file.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
            failures.push(format!(
                "requirement `{}` parser contract `{suffix}` required file `{path}` does not exist",
                requirement.id
            ));
        }
        if file
            .get("source_bytes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            failures.push(format!(
                "requirement `{}` parser contract `{suffix}` required file `{path}` is empty",
                requirement.id
            ));
        }
        let read_error = file.get("read_error");
        if !read_error.is_some_and(serde_json::Value::is_null) {
            failures.push(format!(
                "requirement `{}` parser contract `{suffix}` required file `{path}` read_error={}",
                requirement.id,
                read_error
                    .map(serde_json::Value::to_string)
                    .unwrap_or_else(|| "<missing>".to_string())
            ));
        }
    }
}

fn check_backend_conformance_report(
    requirement: &Requirement,
    base_dir: &Path,
    suffix: &str,
    failures: &mut Vec<String>,
) {
    let Some(report) = first_json_evidence(requirement, base_dir, suffix, failures) else {
        return;
    };
    let schema_version = report
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if schema_version < 2 {
        failures.push(format!(
            "requirement `{}` backend conformance `{suffix}` schema_version={schema_version}; expected schema>=2",
            requirement.id
        ));
    }
    let expected_backend = match suffix {
        "cuda-conformance.json" => Some("cuda"),
        "wgpu-conformance.json" => Some("wgpu"),
        "reference-conformance.json" => Some("cpu-ref"),
        _ => None,
    };
    if let Some(expected) = expected_backend {
        let backend_id = report.get("backend_id").and_then(serde_json::Value::as_str);
        if backend_id != Some(expected) {
            failures.push(format!(
                "requirement `{}` backend conformance `{suffix}` reports backend `{:?}`, expected `{expected}`",
                requirement.id,
                backend_id
            ));
        }
    }
    let total_pairs = report
        .get("total_pairs")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let failed_pairs = report
        .get("failed_pairs")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    let distinct_op_count = report
        .get("distinct_op_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let catalog_required_op_count = report
        .get("catalog_required_op_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let catalog_covered_op_count = report
        .get("catalog_covered_op_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let missing_catalog_ops = report
        .get("missing_catalog_ops")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    let op_matrix_blocked_release_count = report
        .get("op_matrix_blocked_release_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    let release_backend_row_count = report
        .get("release_backend_row_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let missing_release_backend_rows = report
        .get("missing_release_backend_rows")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    let op_matrix_errors = report
        .get("op_matrix_errors")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if op_matrix_errors != 0 {
        failures.push(format!(
            "requirement `{}` backend conformance `{suffix}` reports {op_matrix_errors} OP_MATRIX read/parse error(s)",
            requirement.id
        ));
    }
    if total_pairs == 0 {
        failures.push(format!(
            "requirement `{}` backend conformance `{suffix}` reports zero op pairs",
            requirement.id
        ));
    }
    if total_pairs < 49 {
        failures.push(format!(
            "requirement `{}` backend conformance `{suffix}` reports {total_pairs} op pair(s), below release floor 49",
            requirement.id
        ));
    }
    if distinct_op_count < 49 {
        failures.push(format!(
            "requirement `{}` backend conformance `{suffix}` reports {distinct_op_count} distinct op id(s), below release floor 49",
            requirement.id
        ));
    }
    if catalog_required_op_count == 0
        || catalog_covered_op_count != catalog_required_op_count
        || missing_catalog_ops != 0
    {
        failures.push(format!(
            "requirement `{}` backend conformance `{suffix}` covers {catalog_covered_op_count}/{catalog_required_op_count} OP_MATRIX-required op id(s), missing_catalog_ops={missing_catalog_ops}",
            requirement.id
        ));
    }
    if op_matrix_blocked_release_count != 0 {
        failures.push(format!(
            "requirement `{}` backend conformance `{suffix}` reports {op_matrix_blocked_release_count} OP_MATRIX release backend row(s) marked blocked_release",
            requirement.id
        ));
    }
    let expected_release_backend_rows = catalog_required_op_count.saturating_mul(3);
    if release_backend_row_count < expected_release_backend_rows
        || missing_release_backend_rows != 0
    {
        failures.push(format!(
            "requirement `{}` backend conformance `{suffix}` has release_backend_row_count={release_backend_row_count}, expected {expected_release_backend_rows}, missing_release_backend_rows={missing_release_backend_rows}",
            requirement.id
        ));
    }
    if failed_pairs != 0 {
        failures.push(format!(
            "requirement `{}` backend conformance `{suffix}` reports {failed_pairs} failed pair(s)",
            requirement.id
        ));
    }
    if report
        .get("duplicate_op_ids")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|duplicates| !duplicates.is_empty())
    {
        failures.push(format!(
            "requirement `{}` backend conformance `{suffix}` reports duplicate op id(s)",
            requirement.id
        ));
    }
    if let (Some(expected), Some(pairs)) = (
        expected_backend,
        report.get("pairs").and_then(serde_json::Value::as_array),
    ) {
        for pair in pairs {
            let op_id = pair
                .get("op_id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("<unknown>");
            let backend_id = pair.get("backend_id").and_then(serde_json::Value::as_str);
            if backend_id != Some(expected) {
                failures.push(format!(
                    "requirement `{}` backend conformance `{suffix}` pair `{op_id}` reports backend `{:?}`, expected `{expected}`",
                    requirement.id,
                    backend_id
                ));
            }
        }
    }
}

fn check_benchmark_report_has_cases(
    requirement: &Requirement,
    base_dir: &Path,
    suffix: &str,
    failures: &mut Vec<String>,
) {
    let Some(report) = first_json_evidence(requirement, base_dir, suffix, failures) else {
        return;
    };
    let failed = report
        .get("summary")
        .and_then(|summary| summary.get("failed"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    if failed != 0 {
        failures.push(format!(
            "requirement `{}` benchmark `{suffix}` reports {failed} failed case(s)",
            requirement.id
        ));
    }
    let cases = report
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if cases == 0 {
        failures.push(format!(
            "requirement `{}` benchmark `{suffix}` reports zero cases",
            requirement.id
        ));
    }
    if suffix.contains("cuda")
        || report
            .get("selected_backend")
            .and_then(serde_json::Value::as_str)
            == Some("cuda")
    {
        check_benchmark_cuda_environment_provenance(requirement, suffix, &report, failures);
    }
    if suffix == "cuda-ptx-patterns.json" {
        require_case_metric_at_least(
            requirement,
            suffix,
            &report,
            "ptx_corpus_kernels",
            8.0,
            failures,
        );
        require_case_metric_equals(
            requirement,
            suffix,
            &report,
            "ptx_branch_labels",
            0.0,
            failures,
        );
        for metric in [
            "ptx_predication_candidates",
            "ptx_safe_predication_candidates",
            "ptx_vec_load_candidates",
            "ptx_vec_store_candidates",
            "ptx_async_copy_candidates",
            "ptx_tensor_core_candidates",
            "ptx_ldmatrix_capable_targets",
            "ptx_scheduled_fillers",
            "ptx_predicated_stores",
            "ptx_cp_async_emitted",
            "ptx_mma_sync_emitted",
            "ptx_vectorized_loads_emitted",
            "ptx_vectorized_stores_emitted",
            "ptx_bytes_emitted",
        ] {
            require_case_metric_positive(requirement, suffix, &report, metric, failures);
        }
        for metric in [
            "ptx_vector_kernel_scalar_loads",
            "ptx_vector_kernel_scalar_stores",
            "ptx_vector_kernel_scalar_index_adds",
        ] {
            require_case_metric_equals(requirement, suffix, &report, metric, 0.0, failures);
        }
    }
}

fn check_benchmark_cuda_environment_provenance(
    requirement: &Requirement,
    label: &str,
    report: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    let Some(environment) = report.get("environment") else {
        failures.push(format!(
            "requirement `{}` CUDA benchmark `{label}` has no environment provenance",
            requirement.id
        ));
        return;
    };
    let gpu_devices = environment
        .get("gpu_devices")
        .and_then(serde_json::Value::as_array);
    let first_gpu = gpu_devices.and_then(|devices| devices.first());
    if gpu_devices.is_none_or(|devices| devices.is_empty()) {
        failures.push(format!(
            "requirement `{}` CUDA benchmark `{label}` has no nvidia-smi gpu_devices provenance",
            requirement.id
        ));
    }
    if first_gpu
        .and_then(|device| device.get("name"))
        .and_then(serde_json::Value::as_str)
        .is_none_or(str::is_empty)
    {
        failures.push(format!(
            "requirement `{}` CUDA benchmark `{label}` has no GPU model from nvidia-smi",
            requirement.id
        ));
    }
    match first_gpu
        .and_then(|device| device.get("memory_total_mib"))
        .and_then(serde_json::Value::as_u64)
    {
        Some(mib) if mib >= 16 * 1024 => {}
        Some(mib) => failures.push(format!(
            "requirement `{}` CUDA benchmark `{label}` GPU memory is {mib} MiB, below release floor 16384 MiB",
            requirement.id
        )),
        None => failures.push(format!(
            "requirement `{}` CUDA benchmark `{label}` has no GPU memory_total_mib from nvidia-smi",
            requirement.id
        )),
    }
    match (
        first_gpu
            .and_then(|device| device.get("compute_capability_major"))
            .and_then(serde_json::Value::as_u64),
        first_gpu
            .and_then(|device| device.get("compute_capability_minor"))
            .and_then(serde_json::Value::as_u64),
    ) {
        (Some(major), Some(minor)) if (major, minor) >= (8, 0) => {}
        (Some(major), Some(minor)) => failures.push(format!(
            "requirement `{}` CUDA benchmark `{label}` compute capability is {major}.{minor}, below release floor 8.0",
            requirement.id
        )),
        _ => failures.push(format!(
            "requirement `{}` CUDA benchmark `{label}` has no compute capability from nvidia-smi",
            requirement.id
        )),
    }
    for field in ["nvidia_driver_version", "nvidia_cuda_version"] {
        if environment
            .get(field)
            .and_then(serde_json::Value::as_str)
            .is_none_or(str::is_empty)
        {
            failures.push(format!(
                "requirement `{}` CUDA benchmark `{label}` environment is missing `{field}` from nvidia-smi",
                requirement.id
            ));
        }
    }
}

fn check_benchmark_reproducibility_provenance(
    requirement: &Requirement,
    label: &str,
    report: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    if !json_has_nonempty_string_any(
        report,
        &[
            "source_fingerprint",
            "source_revision",
            "source_artifact_fingerprint",
            "commit_fingerprint",
        ],
    ) && !report
        .get("source_artifacts")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|items| !items.is_empty())
        && !report
            .get("git")
            .is_some_and(|git| json_has_nonempty_string_any(git, &["commit"]))
    {
        failures.push(format!(
            "requirement `{}` benchmark `{label}` must include source fingerprint or source artifact provenance",
            requirement.id
        ));
    }
    let environment = report.get("environment");
    if !environment.is_some_and(|environment| {
        json_has_nonempty_string_any(
            environment,
            &["host_cpu_model", "cpu_model", "host_cpu", "processor_model"],
        )
    }) {
        failures.push(format!(
            "requirement `{}` benchmark `{label}` must include host CPU model provenance",
            requirement.id
        ));
    }
    if !report
        .get("summary")
        .is_some_and(|summary| summary.get("cache_hit_rate").is_some())
    {
        failures.push(format!(
            "requirement `{}` benchmark `{label}` summary must include cache_hit_rate, even when null",
            requirement.id
        ));
    }
    let Some(cases) = report.get("cases").and_then(serde_json::Value::as_array) else {
        return;
    };
    for case in cases {
        let id = case
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if !json_has_nonempty_string_any(
            case,
            &[
                "dataset_fingerprint",
                "corpus_fingerprint",
                "input_fingerprint",
                "workload_fingerprint",
            ],
        ) && !case.get("contract").is_some_and(|contract| {
            json_has_nonempty_string_any(
                contract,
                &[
                    "dataset_fingerprint",
                    "corpus_fingerprint",
                    "input_fingerprint",
                    "workload_fingerprint",
                ],
            )
        }) {
            failures.push(format!(
                "requirement `{}` benchmark `{label}` case `{id}` must include dataset/corpus/input fingerprint provenance",
                requirement.id
            ));
        }
        if !case
            .get("correctness")
            .is_some_and(|correctness| !correctness.is_null())
            && !case.get("oracle").is_some_and(|oracle| !oracle.is_null())
        {
            failures.push(format!(
                "requirement `{}` benchmark `{label}` case `{id}` must include correctness oracle evidence",
                requirement.id
            ));
        }
        let metrics = case.get("metrics").and_then(serde_json::Value::as_object);
        for (metric_label, metric_names) in [
            (
                "cold compile or cold wall timing",
                &["cold_compile_ns", "cold_wall_ns", "compile_ns"][..],
            ),
            (
                "host-to-device transfer bytes",
                &[
                    "host_to_device_bytes",
                    "h2d_bytes",
                    "bytes_host_to_device",
                    "bytes_h2d",
                ][..],
            ),
            (
                "device-to-host transfer bytes",
                &[
                    "device_to_host_bytes",
                    "d2h_bytes",
                    "bytes_device_to_host",
                    "bytes_d2h",
                ][..],
            ),
            (
                "kernel launch count",
                &["kernel_launches", "launch_count", "launches"][..],
            ),
        ] {
            if !metrics_has_any(metrics, metric_names) {
                failures.push(format!(
                    "requirement `{}` benchmark `{label}` case `{id}` must include {metric_label} metric",
                    requirement.id
                ));
            }
        }
        if !case
            .get("optimization_passes")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|items| !items.is_empty())
            && !case
                .get("optimization_passes_applied")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|items| !items.is_empty())
        {
            failures.push(format!(
                "requirement `{}` benchmark `{label}` case `{id}` must list optimization passes applied",
                requirement.id
            ));
        }
    }
}

fn json_has_nonempty_string_any(value: &serde_json::Value, fields: &[&str]) -> bool {
    fields.iter().any(|field| {
        value
            .get(*field)
            .and_then(serde_json::Value::as_str)
            .is_some_and(|text| !text.trim().is_empty())
    })
}

fn metrics_has_any(
    metrics: Option<&serde_json::Map<String, serde_json::Value>>,
    fields: &[&str],
) -> bool {
    metrics.is_some_and(|metrics| {
        fields.iter().any(|field| {
            metrics.get(*field).is_some_and(|value| {
                metric_samples(Some(value)).is_some_and(|samples| samples > 0)
                    || metric_p50(Some(value)).is_some_and(|sample| sample > 0.0)
                    || value.as_u64().is_some()
                    || value.as_f64().is_some_and(|number| number >= 0.0)
            })
        })
    })
}

fn require_case_metric_at_least(
    requirement: &Requirement,
    suffix: &str,
    report: &serde_json::Value,
    metric: &str,
    minimum: f64,
    failures: &mut Vec<String>,
) {
    let Some(cases) = report.get("cases").and_then(serde_json::Value::as_array) else {
        return;
    };
    if !cases.iter().any(|case| {
        case.get("metrics")
            .and_then(serde_json::Value::as_object)
            .and_then(|metrics| metric_p50(metrics.get(metric)))
            .is_some_and(|value| value >= minimum)
    }) {
        failures.push(format!(
            "requirement `{}` benchmark `{suffix}` has no case with p50 `{metric}` >= {minimum}",
            requirement.id
        ));
    }
}

fn require_case_metric_equals(
    requirement: &Requirement,
    suffix: &str,
    report: &serde_json::Value,
    metric: &str,
    expected: f64,
    failures: &mut Vec<String>,
) {
    let Some(cases) = report.get("cases").and_then(serde_json::Value::as_array) else {
        return;
    };
    if !cases.iter().any(|case| {
        case.get("metrics")
            .and_then(serde_json::Value::as_object)
            .and_then(|metrics| metric_p50(metrics.get(metric)))
            .is_some_and(|value| (value - expected).abs() < f64::EPSILON)
    }) {
        failures.push(format!(
            "requirement `{}` benchmark `{suffix}` has no case with p50 `{metric}` == {expected}",
            requirement.id
        ));
    }
}

fn require_case_metric_positive(
    requirement: &Requirement,
    suffix: &str,
    report: &serde_json::Value,
    metric: &str,
    failures: &mut Vec<String>,
) {
    let Some(cases) = report.get("cases").and_then(serde_json::Value::as_array) else {
        return;
    };
    let observed = cases.iter().any(|case| {
        case.get("metrics")
            .and_then(serde_json::Value::as_object)
            .and_then(|metrics| metrics.get(metric))
            .and_then(|metric| {
                metric
                    .get("p50")
                    .and_then(serde_json::Value::as_f64)
                    .or_else(|| {
                        metric
                            .get("p50")
                            .and_then(serde_json::Value::as_u64)
                            .map(|v| v as f64)
                    })
            })
            .is_some_and(|value| value > 0.0)
    });
    if !observed {
        failures.push(format!(
            "requirement `{}` benchmark `{suffix}` has no positive `{metric}` p50 metric",
            requirement.id
        ));
    }
}

fn require_case_metric_present(
    requirement: &Requirement,
    suffix: &str,
    report: &serde_json::Value,
    metric: &str,
    failures: &mut Vec<String>,
) {
    let Some(cases) = report.get("cases").and_then(serde_json::Value::as_array) else {
        failures.push(format!(
            "requirement `{}` benchmark `{suffix}` has no cases array while checking `{metric}`",
            requirement.id
        ));
        return;
    };
    let observed = cases.iter().any(|case| {
        case.get("metrics")
            .and_then(serde_json::Value::as_object)
            .is_some_and(|metrics| metrics.contains_key(metric))
    });
    if !observed {
        failures.push(format!(
            "requirement `{}` benchmark `{suffix}` has no `{metric}` metric claimed by pass-family manifest",
            requirement.id
        ));
    }
}

fn check_backend_suite_report(
    requirement: &Requirement,
    base_dir: &Path,
    suffix: &str,
    failures: &mut Vec<String>,
) {
    let Some(report) = first_json_evidence(requirement, base_dir, suffix, failures) else {
        return;
    };
    let schema_version = report
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if schema_version < 2 {
        failures.push(format!(
            "requirement `{}` backend suite `{suffix}` schema_version={schema_version}; expected schema>=2",
            requirement.id
        ));
    }
    let family_count = report
        .get("family_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let artifact_count = report
        .get("artifacts")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if family_count == 0 || artifact_count == 0 {
        failures.push(format!(
            "requirement `{}` backend suite `{suffix}` reports family_count={family_count}, artifacts={artifact_count}",
            requirement.id
        ));
    }
    if family_count < 12 || artifact_count < 12 {
        failures.push(format!(
            "requirement `{}` backend suite `{suffix}` reports family_count={family_count}, artifacts={artifact_count}; release suites need at least 12 workload families",
            requirement.id
        ));
    }
    if family_count as usize != artifact_count {
        failures.push(format!(
            "requirement `{}` backend suite `{suffix}` family_count={family_count} does not match artifact count {artifact_count}",
            requirement.id
        ));
    }
    if let Some(blockers) = report.get("blockers").and_then(serde_json::Value::as_array) {
        for blocker in blockers {
            failures.push(format!(
                "requirement `{}` backend suite `{suffix}` reports blocker: {}",
                requirement.id,
                blocker.as_str().unwrap_or("<non-string blocker>")
            ));
        }
    }
    if let Some(statuses) = report
        .get("artifact_statuses")
        .and_then(serde_json::Value::as_array)
    {
        for status in statuses {
            let path = status
                .get("path")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("<unknown>");
            if status.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` is missing",
                    requirement.id
                ));
            }
            if status
                .get("bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                == 0
            {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` is empty",
                    requirement.id
                ));
            }
            let read_error = status.get("read_error");
            if !read_error.is_some_and(serde_json::Value::is_null) {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` read_error={}",
                    requirement.id,
                    read_error
                        .map(serde_json::Value::to_string)
                        .unwrap_or_else(|| "<missing>".to_string())
                ));
            }
            if status
                .get("family_id")
                .and_then(serde_json::Value::as_str)
                .is_none_or(str::is_empty)
            {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` has no family_id",
                    requirement.id
                ));
            }
            if status
                .get("requested_case_id")
                .and_then(serde_json::Value::as_str)
                .is_none_or(str::is_empty)
            {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` has no requested_case_id",
                    requirement.id
                ));
            }
            for field in ["source_fingerprint", "host_cpu_model"] {
                if status
                    .get(field)
                    .and_then(serde_json::Value::as_str)
                    .is_none_or(str::is_empty)
                {
                    failures.push(format!(
                        "requirement `{}` backend suite `{suffix}` artifact `{path}` has no `{field}` provenance",
                        requirement.id
                    ));
                }
            }
            if status
                .get("case_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                == 0
            {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` reports zero cases",
                    requirement.id
                ));
            }
            if status
                .get("failed_count")
                .and_then(serde_json::Value::as_u64)
                != Some(0)
            {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` reports nonzero or missing failed_count",
                    requirement.id
                ));
            }
            if status
                .get("nonmatching_case_backend_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(1)
                != 0
            {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` reports backend-mismatched cases",
                    requirement.id
                ));
            }
            if report.get("backend").and_then(serde_json::Value::as_str) == Some("cuda") {
                for field in ["gpu_model", "nvidia_driver_version", "nvidia_cuda_version"] {
                    if status
                        .get(field)
                        .and_then(serde_json::Value::as_str)
                        .is_none_or(str::is_empty)
                    {
                        failures.push(format!(
                            "requirement `{}` backend suite `{suffix}` CUDA artifact `{path}` has no `{field}` provenance",
                            requirement.id
                        ));
                    }
                }
                match status
                    .get("gpu_memory_total_mib")
                    .and_then(serde_json::Value::as_u64)
                {
                    Some(mib) if mib >= 16 * 1024 => {}
                    Some(mib) => failures.push(format!(
                        "requirement `{}` backend suite `{suffix}` CUDA artifact `{path}` reports {mib} MiB GPU memory, below release floor 16384 MiB",
                        requirement.id
                    )),
                    None => failures.push(format!(
                        "requirement `{}` backend suite `{suffix}` CUDA artifact `{path}` has no `gpu_memory_total_mib` provenance",
                        requirement.id
                    )),
                }
                match (
                    status
                        .get("gpu_compute_capability_major")
                        .and_then(serde_json::Value::as_u64),
                    status
                        .get("gpu_compute_capability_minor")
                        .and_then(serde_json::Value::as_u64),
                ) {
                    (Some(major), Some(minor)) if (major, minor) >= (8, 0) => {}
                    (Some(major), Some(minor)) => failures.push(format!(
                        "requirement `{}` backend suite `{suffix}` CUDA artifact `{path}` reports compute capability {major}.{minor}, below release floor 8.0",
                        requirement.id
                    )),
                    _ => failures.push(format!(
                        "requirement `{}` backend suite `{suffix}` CUDA artifact `{path}` has no compute capability provenance",
                        requirement.id
                    )),
                }
                if status
                    .get("min_cuda_ptx_source_cache_entries")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
                    == 0
                {
                    failures.push(format!(
                        "requirement `{}` backend suite `{suffix}` CUDA artifact `{path}` has non-positive `min_cuda_ptx_source_cache_entries`",
                        requirement.id
                    ));
                }
                for field in [
                    "min_cuda_ptx_source_cache_hits",
                    "min_cuda_ptx_source_cache_misses",
                ] {
                    if status
                        .get(field)
                        .and_then(serde_json::Value::as_u64)
                        .is_none()
                    {
                        failures.push(format!(
                            "requirement `{}` backend suite `{suffix}` CUDA artifact `{path}` is missing `{field}`",
                            requirement.id
                        ));
                    }
                }
            }
            if status
                .get("min_wall_samples")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                < 30
            {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` has fewer than 30 wall_ns samples",
                    requirement.id
                ));
            }
            if status
                .get("min_baseline_wall_samples")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                < 30
            {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` has fewer than 30 baseline_wall_ns samples",
                    requirement.id
                ));
            }
            for field in [
                "min_wall_p50",
                "min_wall_p95",
                "min_wall_p99",
                "min_baseline_wall_p50",
                "min_baseline_wall_p95",
                "min_baseline_wall_p99",
            ] {
                if status
                    .get(field)
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
                    == 0
                {
                    failures.push(format!(
                        "requirement `{}` backend suite `{suffix}` artifact `{path}` has non-positive `{field}`",
                        requirement.id
                    ));
                }
            }
            if status
                .get("blockers")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|blockers| !blockers.is_empty())
            {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` has semantic blockers",
                    requirement.id
                ));
            }
            if status
                .get("cpu_sota_100x_required")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
                && status
                    .get("cpu_sota_100x_passing_cases")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
                    == 0
            {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` requires CPU-SOTA 100x proof but has zero passing 100x case(s)",
                    requirement.id
                ));
            }
        }
    } else {
        failures.push(format!(
            "requirement `{}` backend suite `{suffix}` has no artifact_statuses",
            requirement.id
        ));
    }
    if let Some(artifacts) = report
        .get("artifacts")
        .and_then(serde_json::Value::as_array)
    {
        let expected_backend = report.get("backend").and_then(serde_json::Value::as_str);
        for artifact in artifacts {
            let Some(artifact) = artifact.as_str() else {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` contains a non-string artifact path",
                    requirement.id
                ));
                continue;
            };
            let path = resolve_artifact_path(base_dir, artifact);
            let text = match read_text_bounded(&path) {
                Ok(text) => text,
                Err(error) => {
                    failures.push(format!(
                        "requirement `{}` backend suite `{suffix}` failed to read listed artifact `{}`: {error}",
                        requirement.id,
                        path.display()
                    ));
                    continue;
                }
            };
            let report = match serde_json::from_str::<serde_json::Value>(&text) {
                Ok(report) => report,
                Err(error) => {
                    failures.push(format!(
                        "requirement `{}` backend suite `{suffix}` listed artifact `{}` is invalid JSON: {error}",
                        requirement.id,
                        path.display()
                    ));
                    continue;
                }
            };
            check_single_benchmark_report(requirement, &path, &report, false, None, failures);
            if let Some(expected_backend) = expected_backend {
                let selected_backend = report
                    .get("selected_backend")
                    .and_then(serde_json::Value::as_str);
                if selected_backend != Some(expected_backend) {
                    failures.push(format!(
                        "requirement `{}` backend suite `{suffix}` artifact `{}` selected backend `{:?}`, expected `{expected_backend}`",
                        requirement.id,
                        path.display(),
                        selected_backend
                    ));
                }
                if expected_backend == "cuda" {
                    let artifact_label = path.display().to_string();
                    for metric in [
                        "cuda_ptx_source_cache_entries",
                        "cuda_ptx_source_cache_hits",
                        "cuda_ptx_source_cache_misses",
                    ] {
                        require_case_metric_present(
                            requirement,
                            &artifact_label,
                            &report,
                            metric,
                            failures,
                        );
                    }
                    for metric in ["cuda_ptx_source_cache_entries"] {
                        require_case_metric_positive(
                            requirement,
                            &artifact_label,
                            &report,
                            metric,
                            failures,
                        );
                    }
                }
                if let Some(cases) = report.get("cases").and_then(serde_json::Value::as_array) {
                    for case in cases {
                        let id = case
                            .get("id")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("<unknown>");
                        let backend = case.get("backend_id").and_then(serde_json::Value::as_str);
                        if backend != Some(expected_backend) {
                            failures.push(format!(
                                "requirement `{}` backend suite `{suffix}` artifact `{}` case `{id}` backend `{:?}`, expected `{expected_backend}`",
                                requirement.id,
                                path.display(),
                                backend
                            ));
                        }
                    }
                }
            }
        }
    }
}

fn check_markdown_evidence_ready(
    requirement: &Requirement,
    base_dir: &Path,
    suffix: &str,
    failures: &mut Vec<String>,
) {
    let evidence = requirement
        .evidence
        .iter()
        .find(|path| path.ends_with(suffix));
    let Some(evidence) = evidence else {
        failures.push(format!(
            "requirement `{}` needs markdown evidence ending in `{suffix}`",
            requirement.id
        ));
        return;
    };
    let path = resolve_manifest_path(base_dir, evidence);
    let text = match read_text_bounded(&path) {
        Ok(text) => text,
        Err(error) => {
            failures.push(format!(
                "requirement `{}` failed to read markdown evidence `{}`: {error}",
                requirement.id,
                path.display()
            ));
            return;
        }
    };
    for marker in [
        "status: blocked",
        "status: open",
        "status: pending",
        "todo",
        "fixme",
        "placeholder",
        "stub",
        "tbd",
        "to be filled",
    ] {
        for line in text.lines() {
            let lowered = line.to_ascii_lowercase();
            if markdown_line_is_release_rule_text(&lowered) {
                continue;
            }
            if lowered.contains(marker) {
                failures.push(format!(
                    "requirement `{}` markdown evidence `{}` contains unresolved marker `{marker}`",
                    requirement.id,
                    path.display()
                ));
                break;
            }
        }
    }
    if text.trim().is_empty() {
        failures.push(format!(
            "requirement `{}` markdown evidence `{}` is empty",
            requirement.id,
            path.display()
        ));
    }
    if !text.contains("Evidence sources:") {
        failures.push(format!(
            "requirement `{}` markdown evidence `{}` does not list evidence sources",
            requirement.id,
            path.display()
        ));
    }
}

fn check_markdown_evidence_path_ready(
    requirement: &Requirement,
    path: &Path,
    manifest_path: &str,
    failures: &mut Vec<String>,
) {
    let text = match read_text_bounded(path) {
        Ok(text) => text,
        Err(error) => {
            failures.push(format!(
                "requirement `{}` failed to read markdown evidence `{}`: {error}",
                requirement.id,
                path.display()
            ));
            return;
        }
    };
    if text.trim().is_empty() {
        failures.push(format!(
            "requirement `{}` markdown evidence `{manifest_path}` is empty",
            requirement.id
        ));
    }
    for marker in [
        "status: blocked",
        "status: open",
        "status: pending",
        "todo",
        "fixme",
        "placeholder",
        "stub",
        "tbd",
        "to be filled",
    ] {
        for line in text.lines() {
            let lowered = line.to_ascii_lowercase();
            if markdown_line_is_release_rule_text(&lowered) {
                continue;
            }
            if lowered.contains(marker) {
                failures.push(format!(
                    "requirement `{}` markdown evidence `{manifest_path}` contains unresolved marker `{marker}`",
                    requirement.id
                ));
                break;
            }
        }
    }
    if manifest_path.starts_with("evidence/docs/") && !text.contains("Evidence sources:") {
        failures.push(format!(
            "requirement `{}` markdown evidence `{manifest_path}` does not list evidence sources",
            requirement.id
        ));
    }
}

fn markdown_line_is_release_rule_text(lowered: &str) -> bool {
    lowered.contains("no-stub")
        || lowered.contains("no shipped source")
        || lowered.contains("must not")
        || lowered.contains("not only")
        || lowered.contains("not optional")
        || lowered.contains("not a ")
        || lowered.contains("no todo")
        || lowered.contains("todo/fixme")
}

fn manifest_path_from_args(args: &[String]) -> Result<PathBuf, String> {
    let mut manifest_path = None;
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--manifest" => {
                let Some(path) = args.get(index + 1) else {
                    return Err("Fix: --manifest requires a path.".to_string());
                };
                manifest_path = Some(PathBuf::from(path));
                index += 2;
            }
            "--help" | "-h" => {
                println!(
                    "USAGE:\n  cargo_full run --bin xtask -- vyre-release-gate [--manifest PATH]\n\n\
                     Checks the Vyre release evidence manifest and fails until every \
                     requirement is closed with concrete evidence files."
                );
                std::process::exit(0);
            }
            other => {
                return Err(format!(
                    "Fix: unknown vyre-release-gate option `{other}`."
                ));
            }
        }
    }

    Ok(manifest_path.unwrap_or_else(default_manifest_path))
}

fn default_manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|path| path.join("release/vyre-release-evidence.toml"))
        .unwrap_or_else(|| PathBuf::from("release/vyre-release-evidence.toml"))
}

fn resolve_manifest_path(base_dir: &Path, path: &str) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        candidate
    } else {
        base_dir.join(candidate)
    }
}

fn resolve_artifact_path(base_dir: &Path, path: &str) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        return candidate;
    }
    if path.starts_with("release/") {
        return base_dir
            .parent()
            .map(|workspace| workspace.join(candidate))
            .unwrap_or_else(|| base_dir.join(path));
    }
    base_dir.join(candidate)
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    let mut reader = fs::File::open(path)?.take(MAX_RELEASE_GATE_TEXT_BYTES.saturating_add(1));
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    if text.len() as u64 > MAX_RELEASE_GATE_TEXT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} exceeds {MAX_RELEASE_GATE_TEXT_BYTES} byte release gate read cap",
                path.display()
            ),
        ));
    }
    Ok(text)
}
