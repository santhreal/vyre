//! Prompt-to-artifact completion audit for the Vyre/Weir release.

use std::collections::BTreeSet;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct EvidenceManifest {
    plan_path: String,
    requirements: Vec<Requirement>,
}

#[derive(Debug, Deserialize)]
struct Requirement {
    id: String,
    title: String,
    status: String,
    evidence: Vec<String>,
    minimum_evidence: usize,
}

#[derive(Debug, Serialize)]
struct CompletionAudit {
    schema_version: u32,
    objective: &'static str,
    success_criteria: Vec<&'static str>,
    prompt_to_artifact_checklist: Vec<ChecklistItem>,
    plan_path: String,
    total_requirements: usize,
    closed_requirements: usize,
    blocked_or_open_requirements: usize,
    requirements: Vec<RequirementAudit>,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ChecklistItem {
    requirement_id: &'static str,
    explicit_requirement: &'static str,
    required_artifacts_or_commands: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
struct RequirementAudit {
    id: String,
    title: String,
    status: String,
    minimum_evidence: usize,
    evidence_count: usize,
    existing_evidence_count: usize,
    missing_evidence: Vec<String>,
    semantic_blockers: Vec<String>,
    complete: bool,
}

const MAX_RELEASE_AUDIT_TEXT_BYTES: u64 = 16_777_216;

pub(crate) fn run(args: &[String]) {
    let config = match parse_args(args) {
        Ok(config) => config,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let manifest_text = match read_text_bounded(&config.manifest) {
        Ok(text) => text,
        Err(error) => {
            eprintln!(
                "Fix: failed to read release manifest `{}`: {error}",
                config.manifest.display()
            );
            std::process::exit(1);
        }
    };
    let manifest = match toml::from_str::<EvidenceManifest>(&manifest_text) {
        Ok(manifest) => manifest,
        Err(error) => {
            eprintln!(
                "Fix: release manifest `{}` is invalid TOML: {error}",
                config.manifest.display()
            );
            std::process::exit(1);
        }
    };
    let base_dir = config
        .manifest
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut blockers = Vec::new();
    let mut audits = Vec::new();
    for requirement in manifest.requirements {
        let mut missing = Vec::new();
        let mut semantic_blockers = Vec::new();
        let mut existing = 0usize;
        for evidence in &requirement.evidence {
            if is_manifest_command_evidence(evidence) {
                existing += 1;
                continue;
            }
            let path = resolve_manifest_path(&base_dir, evidence);
            let is_self_output = paths_equal(&path, &config.output);
            if is_self_output {
                existing += 1;
                continue;
            }
            match fs::metadata(&path) {
                Ok(metadata) if metadata.is_file() && metadata.len() > 0 => {
                    existing += 1;
                    inspect_evidence_semantics(evidence, &path, &mut semantic_blockers);
                }
                Ok(metadata) if metadata.is_file() => {
                    existing += 1;
                    semantic_blockers.push(format!("{} is empty", path.display()));
                }
                Ok(_) => {
                    semantic_blockers.push(format!("{} exists but is not a file", path.display()));
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    missing.push(evidence.clone());
                }
                Err(error) => {
                    semantic_blockers.push(format!("failed to stat {}: {error}", path.display()));
                }
            }
        }
        let complete = requirement.status == "closed"
            && requirement.evidence.len() >= requirement.minimum_evidence
            && missing.is_empty()
            && semantic_blockers.is_empty();
        if !complete {
            blockers.push(format!(
                "{} is status `{}` with {}/{} evidence files present and {} semantic blocker(s)",
                requirement.id,
                requirement.status,
                existing,
                requirement.evidence.len(),
                semantic_blockers.len()
            ));
        }
        audits.push(RequirementAudit {
            id: requirement.id,
            title: requirement.title,
            status: requirement.status,
            minimum_evidence: requirement.minimum_evidence,
            evidence_count: requirement.evidence.len(),
            existing_evidence_count: existing,
            missing_evidence: missing,
            semantic_blockers,
            complete,
        });
    }
    let checklist = release_checklist();
    let manifest_ids = audits
        .iter()
        .map(|audit| audit.id.as_str())
        .collect::<BTreeSet<_>>();
    let checklist_ids = checklist
        .iter()
        .map(|item| item.requirement_id)
        .collect::<BTreeSet<_>>();
    for missing in manifest_ids.difference(&checklist_ids) {
        blockers.push(format!(
            "final audit checklist is missing manifest requirement `{missing}`"
        ));
    }
    for extra in checklist_ids.difference(&manifest_ids) {
        blockers.push(format!(
            "final audit checklist references non-manifest requirement `{extra}`"
        ));
    }
    for item in &checklist {
        for artifact in item
            .required_artifacts_or_commands
            .iter()
            .copied()
            .filter(|entry| is_checklist_artifact(entry))
        {
            let path = resolve_checklist_artifact_path(&base_dir, artifact);
            let is_self_output = paths_equal(&path, &config.output);
            if is_self_output {
                continue;
            }
            match fs::metadata(&path) {
                Ok(metadata) if metadata.is_file() && metadata.len() > 0 => {
                    inspect_evidence_semantics(artifact, &path, &mut blockers);
                }
                Ok(metadata) if metadata.is_file() => {
                    blockers.push(format!(
                        "final audit checklist artifact `{artifact}` for `{}` is empty at {}",
                        item.requirement_id,
                        path.display()
                    ));
                }
                Ok(_) => {
                    blockers.push(format!(
                        "final audit checklist artifact `{artifact}` for `{}` exists but is not a file at {}",
                        item.requirement_id,
                        path.display()
                    ));
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    blockers.push(format!(
                        "final audit checklist artifact `{artifact}` for `{}` is missing at {}",
                        item.requirement_id,
                        path.display()
                    ));
                }
                Err(error) => {
                    blockers.push(format!(
                        "final audit checklist artifact `{artifact}` for `{}` is unreadable at {}: {error}",
                        item.requirement_id,
                        path.display()
                    ));
                }
            }
        }
    }
    let closed_requirements = audits.iter().filter(|audit| audit.complete).count();
    let audit = CompletionAudit {
        schema_version: 1,
        objective: "Make Vyre 0.4.2 and Weir 0.1.0 release-ready end to end",
        success_criteria: vec![
            "1. CUDA-first execution path is documented, benchmarked, and selected as the fast release substrate.",
            "2. WGPU fallback is functional, tested, benchmarked, and never hides a CUDA or CPU downgrade.",
            "3. Megakernel is the default high-throughput runtime path where legal, including paired speculation evidence.",
            "4. Non-megakernel dispatch is retained only with measured or architectural justification.",
            "5. Optimization infrastructure proves at least 4096 concrete rewrite/pass opportunities across families.",
            "6. Optimization passes have correctness tests and before/after benchmark evidence.",
            "7. Weir alias, reaching-def, points-to, callgraph, slicing, summary, loop, and fixpoint facts integrate into Vyre optimization.",
            "8. The distributed C parser parses a full selected Linux subsystem corpus with AST/semantic contract evidence.",
            "9. At least 12 proof workload families compare CUDA against serious CPU baselines with reproducible artifacts; the current matrix carries 13 required rows.",
            "10. At least ten formerly CPU-only workload families prove 100x+ wins or block release.",
            "11. Conformance gates block release for every claimed op/backend path with zero blocked_release OP_MATRIX rows.",
            "12. Test organization is modular, including distributed parser CLI evidence for tools/vyrec.",
            "13. Documentation is coherent and every release claim links to concrete evidence.",
            "14. Crate metadata, feature surfaces, readmes, examples, license, and version policy are consistent for Vyre 0.4.2 / Weir 0.1.0.",
            "15. Final hygiene review finds no unbounded caches, hidden fallbacks, placeholders, library panics, unactionable errors, or undocumented public API.",
        ],
        prompt_to_artifact_checklist: checklist,
        plan_path: manifest.plan_path,
        total_requirements: audits.len(),
        closed_requirements,
        blocked_or_open_requirements: audits.len().saturating_sub(closed_requirements),
        requirements: audits,
        blockers,
    };
    let json = match serde_json::to_string_pretty(&audit) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("Fix: failed to serialize completion audit: {error}");
            std::process::exit(1);
        }
    };
    if let Some(parent) = config.output.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            eprintln!("Fix: failed to create `{}`: {error}", parent.display());
            std::process::exit(1);
        }
    }
    if let Err(error) = fs::write(&config.output, format!("{json}\n")) {
        eprintln!(
            "Fix: failed to write `{}`: {error}",
            config.output.display()
        );
        std::process::exit(1);
    }
    println!(
        "release-completion-audit: wrote {}",
        config.output.display()
    );
    if !audit.blockers.is_empty() {
        std::process::exit(1);
    }
}

struct Config {
    manifest: PathBuf,
    output: PathBuf,
}

fn parse_args(args: &[String]) -> Result<Config, String> {
    let mut manifest = None;
    let mut output = None;
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--manifest" => {
                let Some(path) = args.get(index + 1) else {
                    return Err("Fix: --manifest requires a path.".to_string());
                };
                manifest = Some(PathBuf::from(path));
                index += 2;
            }
            "--output" => {
                let Some(path) = args.get(index + 1) else {
                    return Err("Fix: --output requires a path.".to_string());
                };
                output = Some(PathBuf::from(path));
                index += 2;
            }
            "--help" | "-h" => {
                println!(
                    "USAGE:\n  cargo_full run --bin xtask -- release-completion-audit [--manifest PATH] [--output PATH]\n\n\
                     Writes final prompt-to-artifact release audit evidence."
                );
                std::process::exit(0);
            }
            other => {
                return Err(format!(
                    "Fix: unknown release-completion-audit option `{other}`."
                ));
            }
        }
    }
    Ok(Config {
        manifest: manifest.unwrap_or_else(default_manifest),
        output: output.unwrap_or_else(default_output),
    })
}

fn default_manifest() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|path| path.join("release/vyre-release-evidence.toml"))
        .unwrap_or_else(|| PathBuf::from("release/vyre-release-evidence.toml"))
}

fn default_output() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|path| path.join("release/evidence/final/completion-audit.json"))
        .unwrap_or_else(|| PathBuf::from("release/evidence/final/completion-audit.json"))
}

fn release_checklist() -> Vec<ChecklistItem> {
    vec![
        ChecklistItem {
            requirement_id: "version-story",
            explicit_requirement: "Vyre manifests, dependency hints, lockfile path packages, docs, release notes, packaging, and product-scoped tags use the selected Vyre 0.4.2 / Weir 0.1.0 version story.",
            required_artifacts_or_commands: vec![
                "cargo_full run --bin xtask -- version-matrix --output release/evidence/version/version-matrix.json",
                "release/evidence/version/version-matrix.json",
                "release/evidence/version/release-tag-plan.json",
                "release/evidence/docs/release-notes-version-story.md",
            ],
        },
        ChecklistItem {
            requirement_id: "cuda-first-path",
            explicit_requirement: "CUDA is the documented, benchmarked, optimized primary release substrate.",
            required_artifacts_or_commands: vec![
                "cargo_full run --bin xtask -- backend-matrix --output release/evidence/backends/backend-matrix.json",
                "cargo_full run --bin xtask -- release-benchmarks --backend cuda",
                "release/evidence/benchmarks/cuda-release-suite.json",
                "release/evidence/benchmarks/cuda-ptx-patterns.json",
                "release/evidence/benchmarks/bench-release-axes.json",
            ],
        },
        ChecklistItem {
            requirement_id: "wgpu-fallback",
            explicit_requirement: "WGPU remains a real fallback path with conformance and benchmark evidence, not an untested branch.",
            required_artifacts_or_commands: vec![
                "cargo_full run --bin xtask -- release-benchmarks --backend wgpu",
                "release/evidence/backends/backend-matrix.json",
                "release/evidence/benchmarks/wgpu-fallback-suite.json",
                "release/evidence/conformance/wgpu-conformance.json",
                "release/evidence/docs/wgpu-fallback-proof.md",
            ],
        },
        ChecklistItem {
            requirement_id: "megakernel-default",
            explicit_requirement: "Megakernel is the default high-throughput path where legal and has latency/condition workload evidence.",
            required_artifacts_or_commands: vec![
                "release/evidence/backends/backend-matrix.json",
                "release/evidence/benchmarks/release-workload-matrix.json",
                "release/evidence/benchmarks/megakernel-condition-cuda.json",
                "release/evidence/benchmarks/megakernel-latency-cuda.json",
                "release/evidence/docs/megakernel-default-proof.md",
            ],
        },
        ChecklistItem {
            requirement_id: "optimization-corpus-4096",
            explicit_requirement: "Optimization scale is proved by at least 4096 verified semantic-preserving generated cases.",
            required_artifacts_or_commands: vec![
                "cargo_full run --bin xtask -- optimization-corpus --output release/evidence/optimization/optimization-corpus.json",
                "release/evidence/optimization/optimization-corpus.json",
                "release/evidence/optimization/optimization-corpus-contracts.json",
                "release/evidence/optimization/optimization-family-manifest.json",
                "release/evidence/optimization/optimization-analysis-fixtures.json",
                "release/evidence/optimization/optimization-case-manifest.json",
            ],
        },
        ChecklistItem {
            requirement_id: "optimization-benchmark-proof",
            explicit_requirement: "Optimization families have before/after benchmark evidence and pass-family attribution.",
            required_artifacts_or_commands: vec![
                "cargo_full run --bin xtask -- optimization-matrix --output release/evidence/optimization/optimization-integration-matrix.json",
                "release/evidence/optimization/lower-rewrite-impact-before-after.json",
                "release/evidence/optimization/optimizer-impact-cuda.json",
                "release/evidence/optimization/pass-family-benchmarks.json",
                "release/evidence/optimization/pass-family-benchmark-manifest.json",
            ],
        },
        ChecklistItem {
            requirement_id: "alias-aware-upgrades",
            explicit_requirement: "DSE, store-to-load forwarding, LICM, loop fusion, and loop fission consume Weir alias facts and prove before/after impact.",
            required_artifacts_or_commands: vec![
                "release/evidence/optimization/alias-aware-dse.json",
                "release/evidence/optimization/alias-aware-stlf.json",
                "release/evidence/optimization/alias-aware-licm.json",
                "release/evidence/optimization/alias-aware-fusion-fission.json",
                "release/evidence/benchmarks/alias-aware-before-after.json",
            ],
        },
        ChecklistItem {
            requirement_id: "egraph-saturation",
            explicit_requirement: "Bounded e-graph or egglog-family saturation is present where it beats hand rewrites and has semantic plus benchmark evidence.",
            required_artifacts_or_commands: vec![
                "release/evidence/optimization/egraph-saturation-matrix.json",
                "release/evidence/optimization/egraph-semantic-contracts.json",
                "release/evidence/benchmarks/egraph-before-after.json",
                "release/evidence/docs/egraph-saturation.md",
            ],
        },
        ChecklistItem {
            requirement_id: "weir-analysis-integration",
            explicit_requirement: "Weir exposes SSA, IFDS, reaching defs, points-to, alias, callgraph, slicing, summaries, loops, and fixpoint facts, and Vyre consumes those facts where they unlock safer optimization.",
            required_artifacts_or_commands: vec![
                "cargo_full run --bin xtask -- weir-matrix --output release/evidence/weir/weir-analysis-api-matrix.json",
                "release/evidence/weir/weir-analysis-api-matrix.json",
                "release/evidence/weir/weir-vyre-integration-tests.json",
                "release/evidence/weir/weir-readme-contracts.json",
                "../../../../dataflow/weir/Cargo.toml",
                "release/evidence/optimization/weir-facts-pass-firing.json",
                "release/evidence/benchmarks/dataflow-analysis-release.json",
                "release/evidence/docs/weir-integration.md",
            ],
        },
        ChecklistItem {
            requirement_id: "proof-workloads-12",
            explicit_requirement: "At least 12 proof workload families have reproducible CPU/GPU benchmark artifacts.",
            required_artifacts_or_commands: vec![
                "cargo_full run --bin xtask -- release-workload-matrix --output release/evidence/benchmarks/release-workload-matrix.json --enforce",
                "release/evidence/benchmarks/release-workload-matrix.json",
                "../docs/optimization/BENCH_TARGETS.toml",
                "release/evidence/benchmarks/workload-01-condition-eval.json",
                "release/evidence/benchmarks/workload-02-string-bitmap-scatter.json",
                "release/evidence/benchmarks/workload-03-offset-count-aggregation.json",
                "release/evidence/benchmarks/workload-04-metadata-conditions.json",
                "release/evidence/benchmarks/workload-05-entropy-window.json",
                "release/evidence/benchmarks/workload-06-quantified-condition-loops.json",
                "release/evidence/benchmarks/workload-07-alias-reaching-def.json",
                "release/evidence/benchmarks/workload-08-ifds-witness.json",
                "release/evidence/benchmarks/workload-09-c-ast-traversal.json",
                "release/evidence/benchmarks/workload-10-megakernel-queued-batches.json",
                "release/evidence/benchmarks/workload-11-egraph-saturation.json",
                "release/evidence/benchmarks/workload-12-sparse-output-compaction.json",
                "release/evidence/benchmarks/workload-13-callgraph-reachability.json",
            ],
        },
        ChecklistItem {
            requirement_id: "cpu-only-100x-proof",
            explicit_requirement: "At least ten release workloads prove 100x+ CUDA wins against CPU-SOTA baselines.",
            required_artifacts_or_commands: vec![
                "cargo_full run --bin xtask -- release-workload-matrix --output release/evidence/benchmarks/release-workload-matrix.json --enforce",
                "release/evidence/benchmarks/release-workload-matrix.json",
                "release/evidence/benchmarks/cpu-only-100x-proof.json",
                "release/evidence/benchmarks/megakernel-condition-100x-proof.json",
                "release/evidence/benchmarks/workload-01-condition-eval.json",
                "release/evidence/benchmarks/workload-02-string-bitmap-scatter.json",
                "release/evidence/benchmarks/workload-03-offset-count-aggregation.json",
                "release/evidence/benchmarks/workload-05-entropy-window.json",
                "release/evidence/benchmarks/workload-06-quantified-condition-loops.json",
                "release/evidence/benchmarks/workload-10-megakernel-queued-batches.json",
                "release/evidence/benchmarks/workload-12-sparse-output-compaction.json",
                "release/evidence/docs/cpu-only-100x-proof.md",
            ],
        },
        ChecklistItem {
            requirement_id: "c-parser-linux-subsystem",
            explicit_requirement: "The distributed C parser parses the selected full Linux subsystem corpus with AST, diagnostic, and throughput evidence.",
            required_artifacts_or_commands: vec![
                "cargo_full run --bin xtask -- c-parser-corpus --corpus DIR -I LINUX_INCLUDE_DIR -D CONFIG_SYMBOL=1 --output release/evidence/parser/c-parser-linux-subsystem.json",
                "release/evidence/parser/c-parser-linux-subsystem.json",
                "release/evidence/parser/linux-subsystem-corpus-manifest.json",
                "release/evidence/parser/c-parser-diagnostics-summary.json",
                "release/evidence/parser/c-parser-throughput.json",
            ],
        },
        ChecklistItem {
            requirement_id: "distributed-parser-coherence",
            explicit_requirement: "vyre-frontend-c, tools/vyrec, Weir, and Surge/SurgeC parser ownership boundaries are coherent and documented.",
            required_artifacts_or_commands: vec![
                "cargo_full run --bin xtask -- parser-coherence --output release/evidence/parser/distributed-parser-map.json",
                "release/evidence/parser/distributed-parser-map.json",
                "release/evidence/parser/vyre-frontend-c-contracts.json",
                "release/evidence/parser/vyrec-cli-contracts.json",
                "release/evidence/parser/weir-contracts.json",
                "release/evidence/parser/security-analysis-consumer-contracts.json",
                "release/evidence/parser/security-grammar-gen-contracts.json",
                "release/evidence/docs/distributed-parser-coherence.md",
            ],
        },
        ChecklistItem {
            requirement_id: "conformance-hard-gate",
            explicit_requirement: "Conformance blocks release for every claimed op/backend pair.",
            required_artifacts_or_commands: vec![
                "cargo_full run --bin xtask -- conformance-matrix --output release/evidence/conformance/conformance-matrix.json",
                "cargo_full run --bin xtask -- release-conformance --backend all",
                "release/evidence/conformance/cuda-conformance.json",
                "release/evidence/conformance/wgpu-conformance.json",
                "release/evidence/conformance/reference-conformance.json",
                "release/evidence/conformance/release-gate-log.json",
                "../../../../../.github/workflows/conform.yml",
                "../../../../../.github/workflows/gpu-parity.yml",
                "../../../../../.github/workflows/santh-ci.yml",
                "../../../../../.github/workflows/architectural-invariants.yml",
                "../../../../../.github/CI_REQUIRED.md",
                "../../../../../scripts/apply-branch-protection.sh",
            ],
        },
        ChecklistItem {
            requirement_id: "modular-test-architecture",
            explicit_requirement: "Massive tests are split into fixtures, contracts, properties, backend tests, corpus tests, benchmarks, and regressions.",
            required_artifacts_or_commands: vec![
                "cargo_full run --bin xtask -- test-matrix --output release/evidence/tests/test-matrix.json",
                "release/evidence/tests/test-matrix.json",
                "release/evidence/tests/modularization-map.json",
                "release/evidence/tests/oversized-test-closure.json",
                "release/evidence/tests/release-surface-suite-coverage.json",
                "release/evidence/docs/test-architecture.md",
            ],
        },
        ChecklistItem {
            requirement_id: "exhaustive-verification",
            explicit_requirement: "Normal, adversarial, property, conformance, corpus, benchmark, gap, and fuzz verification are present.",
            required_artifacts_or_commands: vec![
                "release/evidence/tests/unit-suite.json",
                "release/evidence/tests/adversarial-suite.json",
                "release/evidence/tests/property-suite.json",
                "release/evidence/tests/conformance-suite.json",
                "release/evidence/tests/corpus-suite.json",
                "release/evidence/tests/benchmark-suite.json",
                "release/evidence/tests/gap-suite.json",
                "release/evidence/tests/fuzz-suite.json",
                "release/evidence/tests/release-surface-suite-coverage.json",
            ],
        },
        ChecklistItem {
            requirement_id: "docs-evidence-linked",
            explicit_requirement: "User, contributor, benchmark, conformance, parser, optimization, and release docs link to concrete evidence.",
            required_artifacts_or_commands: vec![
                "cargo_full run --bin xtask -- docs-matrix --output release/evidence/docs/docs-matrix.json",
                "release/evidence/docs/docs-matrix.json",
                "release/evidence/docs/vyre-readme-contracts.json",
                "release/evidence/docs/vyre-readme-proof.md",
                "release/evidence/docs/weir-readme-proof.md",
                "release/evidence/docs/parser-doc-proof.md",
                "release/evidence/docs/benchmark-doc-proof.md",
                "release/evidence/docs/conformance-doc-proof.md",
                "release/evidence/docs/release-notes.md",
            ],
        },
        ChecklistItem {
            requirement_id: "crate-metadata",
            explicit_requirement: "Every release crate has coherent metadata, features, docs, readme, license, and version policy.",
            required_artifacts_or_commands: vec![
                "cargo_full run --bin xtask -- metadata-matrix --output release/evidence/metadata/metadata-matrix.json",
                "cargo_full run --bin xtask -- feature-matrix --output release/evidence/metadata/feature-matrix.json",
                "cargo_full run --bin xtask -- package-readiness --output release/evidence/package/publish-readiness.json",
                "release/evidence/metadata/metadata-matrix.json",
                "release/evidence/metadata/feature-matrix.json",
                "release/evidence/package/publish-readiness.json",
                "release/evidence/docs/crate-metadata-proof.md",
            ],
        },
        ChecklistItem {
            requirement_id: "release-hygiene",
            explicit_requirement: "No shipped stubs, hidden fallbacks, unbounded resources, hot-path sleeps, unactionable errors, or stale docs remain.",
            required_artifacts_or_commands: vec![
                "cargo_full run --bin xtask -- hygiene-matrix --output release/evidence/hygiene/hygiene-matrix.json",
                "release/evidence/hygiene/hygiene-matrix.json",
                "release/evidence/hygiene/no-stubs-scan.json",
                "release/evidence/hygiene/no-hidden-fallback-scan.json",
                "release/evidence/hygiene/resource-bound-scan.json",
                "release/evidence/hygiene/error-surface-scan.json",
                "release/evidence/hygiene/cargo-wrapper-scan.json",
                "release/evidence/hygiene/audit-location-scan.json",
                "release/evidence/hygiene/public-doc-scan.json",
                "release/evidence/hygiene/test-hygiene-scan.json",
                "release/evidence/docs/release-hygiene-proof.md",
                "../../../../../.github/workflows/architectural-invariants.yml",
                "../../../../../.github/CI_REQUIRED.md",
                "../../../../../scripts/apply-branch-protection.sh",
                "../../../../../scripts/architectural_invariants.sh",
            ],
        },
        ChecklistItem {
            requirement_id: "final-completion-audit",
            explicit_requirement: "The final audit maps every explicit requirement, named artifact, generator command, and gate to evidence and refuses completion while any requirement is open, blocked, missing, empty, or semantically weak.",
            required_artifacts_or_commands: vec![
                "cargo_full run --bin xtask -- release-evidence",
                "release/evidence/final/release-evidence-run.json",
                "cargo_full run --bin xtask -- release-completion-audit --output release/evidence/final/completion-audit.json",
                "cargo_full run --bin xtask -- vyre-release-gate",
                "cargo_full run --bin xtask -- launch-state --output release/evidence/final/public-launch-state.json",
                "release/evidence/final/completion-audit.json",
                "release/evidence/final/public-launch-state.json",
                "release/../scripts/final-launch.sh",
                "release/../scripts/publish-release.sh",
                "release/vyre-release-evidence.toml",
            ],
        },
    ]
}

fn inspect_evidence_semantics(evidence: &str, path: &Path, blockers: &mut Vec<String>) {
    if evidence.ends_with(".json") {
        inspect_json_evidence(evidence, path, blockers);
    } else if evidence.ends_with(".md") {
        inspect_markdown_evidence(evidence, path, blockers);
    } else if evidence.ends_with("BENCH_TARGETS.toml") {
        inspect_bench_targets_toml(evidence, path, blockers);
    }
}

fn inspect_bench_targets_toml(evidence: &str, path: &Path, blockers: &mut Vec<String>) {
    let text = match read_text_bounded(path) {
        Ok(text) => text,
        Err(error) => {
            blockers.push(format!(
                "{evidence}: failed to read benchmark target table: {error}"
            ));
            return;
        }
    };
    let target_count = text.matches("[[target]]").count();
    if target_count < 17 {
        blockers.push(format!(
            "{evidence}: benchmark target table contains {target_count} target(s); needs at least 17 including release workloads and optimization-proof targets"
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
            blockers.push(format!(
                "{evidence}: missing release benchmark target `{required}`"
            ));
        }
    }
    if !text.contains("\"cpu_sota\"") || !text.contains("min_speedup_over_cpu_sota") {
        blockers.push(format!(
            "{evidence}: benchmark target table must declare CPU-SOTA classes and speedup thresholds"
        ));
    }
}

fn inspect_json_evidence(evidence: &str, path: &Path, blockers: &mut Vec<String>) {
    let text = match read_text_bounded(path) {
        Ok(text) => text,
        Err(error) => {
            blockers.push(format!("{evidence}: failed to read JSON evidence: {error}"));
            return;
        }
    };
    let value = match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(value) => value,
        Err(error) => {
            blockers.push(format!("{evidence}: invalid JSON evidence: {error}"));
            return;
        }
    };
    let blocker_count = value
        .get("blockers")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if blocker_count != 0 {
        blockers.push(format!("{evidence}: reports {blocker_count} blocker(s)"));
    }
    let failed = value
        .get("summary")
        .and_then(|summary| summary.get("failed"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if failed != 0 {
        blockers.push(format!(
            "{evidence}: benchmark summary reports {failed} failed case(s)"
        ));
    }
    if let Some(cases) = value.get("cases").and_then(serde_json::Value::as_array) {
        if cases.is_empty() {
            blockers.push(format!("{evidence}: cases array is empty"));
        }
    }
    if !evidence.ends_with("cuda-release-suite.json")
        && !evidence.ends_with("wgpu-fallback-suite.json")
        && !evidence.contains("/conformance/")
        && (evidence.contains("cuda")
            || value
                .get("selected_backend")
                .and_then(serde_json::Value::as_str)
                == Some("cuda"))
    {
        inspect_benchmark_cuda_environment_semantics(evidence, &value, blockers);
    }
    if let Some(families) = value.get("families").and_then(serde_json::Value::as_array) {
        if families.is_empty() {
            blockers.push(format!("{evidence}: workload families array is empty"));
        }
    }
    if let Some(packages) = value.get("packages").and_then(serde_json::Value::as_array) {
        if packages.is_empty() {
            blockers.push(format!("{evidence}: packages array is empty"));
        }
    }
    if let Some(entries) = value.get("entries").and_then(serde_json::Value::as_array) {
        if entries.is_empty() {
            blockers.push(format!("{evidence}: entries array is empty"));
        }
    }
    if value
        .get("op_count")
        .and_then(serde_json::Value::as_u64)
        .is_some_and(|count| count == 0)
    {
        blockers.push(format!("{evidence}: op_count is zero"));
    }
    if value
        .get("total_files")
        .and_then(serde_json::Value::as_u64)
        .is_some_and(|count| count == 0)
    {
        blockers.push(format!("{evidence}: total_files is zero"));
    }
    if value
        .get("scanned_files")
        .and_then(serde_json::Value::as_u64)
        .is_some_and(|count| count == 0)
    {
        blockers.push(format!("{evidence}: scanned_files is zero"));
    }
    if evidence.ends_with("hygiene-matrix.json") {
        inspect_hygiene_matrix_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("version-matrix.json") {
        inspect_version_matrix_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("backend-matrix.json") {
        inspect_backend_matrix_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("feature-matrix.json") {
        inspect_feature_matrix_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("metadata-matrix.json") {
        inspect_metadata_matrix_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("publish-readiness.json") {
        inspect_package_readiness_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("public-launch-state.json") {
        inspect_public_launch_state_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("docs-matrix.json") {
        inspect_docs_matrix_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("bench-release-axes.json") {
        inspect_release_axes_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("cuda-release-suite.json")
        || evidence.ends_with("wgpu-fallback-suite.json")
    {
        inspect_backend_suite_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("cuda-ptx-patterns.json") {
        inspect_cuda_ptx_pattern_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("megakernel-condition-cuda.json") {
        inspect_megakernel_condition_cuda_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("megakernel-latency-cuda.json") {
        inspect_megakernel_latency_cuda_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("release-tag-plan.json") {
        inspect_release_tag_plan_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("release-workload-matrix.json") {
        inspect_release_workload_matrix_semantics(evidence, &value, blockers);
    }
    if evidence.contains("release/evidence/benchmarks/workload-")
        && evidence.ends_with(".json")
        && !evidence.ends_with("release-workload-matrix.json")
    {
        inspect_workload_benchmark_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("cpu-only-100x-proof.json") {
        inspect_cpu_100x_benchmark_semantics(evidence, &value, blockers);
    }
    if is_before_after_benchmark_evidence(evidence) {
        inspect_before_after_benchmark_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("optimization-corpus.json")
        || evidence.ends_with("optimization-corpus-contracts.json")
    {
        inspect_optimization_corpus_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("optimization-family-manifest.json") {
        inspect_optimization_family_manifest_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("optimization-analysis-fixtures.json") {
        inspect_optimization_analysis_fixture_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("optimization-case-manifest.json") {
        inspect_optimization_case_manifest_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("pass-family-benchmark-manifest.json") {
        inspect_pass_family_benchmark_manifest_semantics(evidence, path, &value, blockers);
    }
    if is_marker_evidence(evidence) {
        inspect_marker_evidence_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("weir-analysis-api-matrix.json") {
        inspect_weir_matrix_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("weir-vyre-integration-tests.json") {
        inspect_schema_version_at_least(evidence, &value, 2, blockers);
    }
    if evidence.ends_with("weir-readme-contracts.json") {
        inspect_weir_readme_contract_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("vyre-readme-contracts.json") {
        inspect_weir_readme_contract_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("cuda-conformance.json")
        || evidence.ends_with("wgpu-conformance.json")
        || evidence.ends_with("reference-conformance.json")
    {
        inspect_backend_conformance_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("conformance-matrix.json") {
        inspect_conformance_matrix_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("release-gate-log.json") {
        inspect_release_conformance_log_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("release-evidence-run.json") {
        inspect_release_evidence_run_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("c-parser-linux-subsystem.json") {
        inspect_c_parser_corpus_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("linux-subsystem-corpus-manifest.json") {
        inspect_c_parser_manifest_semantics(evidence, path, &value, blockers);
    }
    if evidence.ends_with("c-parser-diagnostics-summary.json") {
        inspect_c_parser_diagnostics_semantics(evidence, path, &value, blockers);
    }
    if evidence.ends_with("c-parser-throughput.json") {
        inspect_c_parser_throughput_semantics(evidence, path, &value, blockers);
    }
    if evidence.ends_with("distributed-parser-map.json") {
        inspect_distributed_parser_map_semantics(evidence, &value, blockers);
    }
    if is_parser_contract_evidence(evidence) {
        inspect_parser_contract_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("oversized-test-closure.json") {
        inspect_oversized_test_closure_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("test-matrix.json") {
        inspect_test_matrix_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("modularization-map.json") {
        inspect_modularization_map_semantics(evidence, &value, blockers);
    }
    if evidence.ends_with("release-surface-suite-coverage.json") {
        inspect_surface_coverage_semantics(evidence, &value, blockers);
    }
    if is_test_suite_evidence(evidence) {
        inspect_suite_evidence_semantics(evidence, &value, blockers);
    }
    let open = value
        .get("blocked_or_open_requirements")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if open != 0 {
        blockers.push(format!(
            "{evidence}: completion audit reports {open} blocked/open requirement(s)"
        ));
    }
}

fn is_marker_evidence(evidence: &str) -> bool {
    evidence.ends_with("alias-aware-dse.json")
        || evidence.ends_with("alias-aware-stlf.json")
        || evidence.ends_with("alias-aware-licm.json")
        || evidence.ends_with("alias-aware-fusion-fission.json")
        || evidence.ends_with("weir-facts-pass-firing.json")
        || evidence.ends_with("egraph-saturation-matrix.json")
        || evidence.ends_with("egraph-semantic-contracts.json")
}

fn inspect_marker_evidence_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let markers = value
        .get("markers")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if markers == 0 {
        blockers.push(format!("{evidence}: marker evidence contains zero markers"));
    }
    for required in required_marker_ids_for_evidence(evidence) {
        if !value
            .get("markers")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|markers| {
                markers.iter().any(|marker| {
                    marker.get("id").and_then(serde_json::Value::as_str) == Some(required)
                })
            })
        {
            blockers.push(format!(
                "{evidence}: missing required optimization marker `{required}`"
            ));
        }
    }
    if !value
        .get("source_matrix")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|source| source.ends_with("optimization-integration-matrix.json"))
    {
        blockers.push(format!(
            "{evidence}: source_matrix must reference optimization-integration-matrix.json"
        ));
    }
}

fn required_marker_ids_for_evidence(evidence: &str) -> &'static [&'static str] {
    if evidence.ends_with("alias-aware-dse.json") {
        &[
            "alias-aware-dse-entrypoint",
            "reaching-def-dse-entrypoint",
            "weir-alias-analysis-api",
            "weir-reaching-def-analysis-api",
            "dataflow-analysis-pipeline-entrypoint",
        ]
    } else if evidence.ends_with("alias-aware-stlf.json") {
        &[
            "alias-aware-stlf-entrypoint",
            "reaching-def-stlf-entrypoint",
            "weir-alias-analysis-api",
            "weir-reaching-def-analysis-api",
            "dataflow-analysis-pipeline-entrypoint",
            "dataflow-analysis-stlf-firing-test",
        ]
    } else if evidence.ends_with("alias-aware-licm.json") {
        &[
            "alias-aware-licm-entrypoint",
            "reaching-def-licm-entrypoint",
            "weir-alias-analysis-api",
            "weir-reaching-def-analysis-api",
        ]
    } else if evidence.ends_with("alias-aware-fusion-fission.json") {
        &[
            "alias-aware-loop-fusion-entrypoint",
            "reaching-def-loop-fusion-entrypoint",
            "alias-aware-loop-fission-entrypoint",
            "reaching-def-loop-fission-entrypoint",
            "weir-alias-analysis-api",
            "weir-reaching-def-analysis-api",
        ]
    } else if evidence.ends_with("weir-facts-pass-firing.json") {
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
    } else if evidence.ends_with("egraph-saturation-matrix.json")
        || evidence.ends_with("egraph-semantic-contracts.json")
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

fn inspect_weir_matrix_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if schema_version < 2 {
        blockers.push(format!(
            "{evidence}: schema_version is {schema_version}, expected >= 2"
        ));
    }
    let inventory_registered = value
        .get("inventory_registered_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if inventory_registered == 0 {
        blockers.push(format!(
            "{evidence}: inventory_registered_count must be nonzero"
        ));
    }
    let required_api_item_count = value
        .get("required_api_item_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if required_api_item_count < 100 {
        blockers.push(format!(
            "{evidence}: required_api_item_count is {required_api_item_count}; release matrix must prove at least 100 named Weir public API items"
        ));
    }
    let missing_api_item_count = value
        .get("missing_api_item_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    if missing_api_item_count != 0 {
        blockers.push(format!(
            "{evidence}: missing_api_item_count is {missing_api_item_count}; release requires zero missing Weir public API items"
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
        let count = value
            .get(field)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if count < minimum {
            blockers.push(format!(
                "{evidence}: {label} test family count is {count}; needs at least {minimum}"
            ));
        }
    }
    let standalone_examples = value
        .get("standalone_example_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if standalone_examples < 2 {
        blockers.push(format!(
            "{evidence}: standalone_example_count is {standalone_examples}; needs at least 2 examples outside tests"
        ));
    }
    let standalone_serde_evidence = value
        .get("standalone_serde_evidence_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if standalone_serde_evidence == 0 {
        blockers.push(format!(
            "{evidence}: standalone_serde_evidence_count must be nonzero"
        ));
    }
    let standalone_serde_feature_guards = value
        .get("standalone_serde_feature_guard_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if standalone_serde_feature_guards == 0 {
        blockers.push(format!(
            "{evidence}: standalone_serde_feature_guard_count must prove required-features = [\"serde\"] for serde evidence examples"
        ));
    }
    let example_files = value
        .get("standalone_examples")
        .and_then(serde_json::Value::as_array);
    if example_files.is_none_or(|examples| examples.len() < 2) {
        blockers.push(format!(
            "{evidence}: standalone_examples must list at least 2 example files"
        ));
    }
    let standalone_example_scan_errors = value
        .get("standalone_example_scan_errors")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if standalone_example_scan_errors != 0 {
        blockers.push(format!(
            "{evidence}: reports {standalone_example_scan_errors} standalone example scan error(s)"
        ));
    }
    if let Some(examples) = example_files {
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
                blockers.push(format!(
                    "{evidence}: standalone example `{path}` must exist and be non-empty"
                ));
            }
            if !example
                .get("read_error")
                .is_some_and(serde_json::Value::is_null)
            {
                blockers.push(format!(
                    "{evidence}: standalone example `{path}` read_error must be null"
                ));
            }
            if example.get("has_main").and_then(serde_json::Value::as_bool) != Some(true) {
                blockers.push(format!(
                    "{evidence}: standalone example `{path}` must expose runnable fn main"
                ));
            }
            if example
                .get("uses_weir_crate")
                .and_then(serde_json::Value::as_bool)
                != Some(true)
            {
                blockers.push(format!(
                    "{evidence}: standalone example `{path}` must import or reference the weir crate"
                ));
            }
            let api_reference_count = example
                .get("api_reference_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            if api_reference_count < 2 {
                blockers.push(format!(
                    "{evidence}: standalone example `{path}` references {api_reference_count} dataflow API token(s); needs at least 2"
                ));
            }
            if path.ends_with("serde_evidence.rs")
                && example
                    .get("has_serde_evidence")
                    .and_then(serde_json::Value::as_bool)
                    != Some(true)
            {
                blockers.push(format!(
                    "{evidence}: standalone serde example `{path}` must report has_serde_evidence=true"
                ));
            }
            let unresolved_markers = example
                .get("unresolved_markers")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            if unresolved_markers != 0 {
                blockers.push(format!(
                    "{evidence}: standalone example `{path}` reports {unresolved_markers} unresolved marker(s)"
                ));
            }
        }
    }
    let untested_analyses = value
        .get("untested_analyses")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if untested_analyses != 0 {
        blockers.push(format!(
            "{evidence}: {untested_analyses} Weir analysis module(s) lack release test coverage"
        ));
    }
    let Some(analyses) = value.get("analyses").and_then(serde_json::Value::as_array) else {
        blockers.push(format!("{evidence}: missing analyses array"));
        return;
    };
    if analyses.is_empty() {
        blockers.push(format!("{evidence}: analyses array is empty"));
    }
    for entry in analyses {
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
            blockers.push(format!(
                "{evidence}: analysis `{id}` reports {missing_api_items} missing required API item(s)"
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
                blockers.push(format!(
                    "{evidence}: soundness analysis must prove six policy API items and report zero missing items"
                ));
            }
        }
        if declares_op_id && !registered {
            blockers.push(format!(
                "{evidence}: analysis `{id}` declares OP_ID without inventory registration"
            ));
        }
    }
}

fn inspect_backend_conformance_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if schema_version < 2 {
        blockers.push(format!(
            "{evidence}: schema_version={schema_version}; backend conformance evidence must be schema>=2"
        ));
    }
    let expected_backend = if evidence.ends_with("cuda-conformance.json") {
        "cuda"
    } else if evidence.ends_with("wgpu-conformance.json") {
        "wgpu"
    } else {
        "cpu-ref"
    };
    if value.get("backend_id").and_then(serde_json::Value::as_str) != Some(expected_backend) {
        blockers.push(format!(
            "{evidence}: backend_id must be `{expected_backend}`"
        ));
    }
    let total_pairs = value
        .get("total_pairs")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let distinct_op_count = value
        .get("distinct_op_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let catalog_required_op_count = value
        .get("catalog_required_op_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let catalog_covered_op_count = value
        .get("catalog_covered_op_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let missing_catalog_ops = value
        .get("missing_catalog_ops")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    let op_matrix_blocked_release_count = value
        .get("op_matrix_blocked_release_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    let release_backend_row_count = value
        .get("release_backend_row_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let missing_release_backend_rows = value
        .get("missing_release_backend_rows")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    let op_matrix_errors = value
        .get("op_matrix_errors")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if op_matrix_errors != 0 {
        blockers.push(format!(
            "{evidence}: reports {op_matrix_errors} OP_MATRIX read/parse error(s)"
        ));
    }
    if total_pairs < 49 {
        blockers.push(format!(
            "{evidence}: total_pairs is {total_pairs}, below release floor 49"
        ));
    }
    if distinct_op_count < 49 {
        blockers.push(format!(
            "{evidence}: distinct_op_count is {distinct_op_count}, below release floor 49"
        ));
    }
    if catalog_required_op_count == 0
        || catalog_covered_op_count != catalog_required_op_count
        || missing_catalog_ops != 0
    {
        blockers.push(format!(
            "{evidence}: covers {catalog_covered_op_count}/{catalog_required_op_count} OP_MATRIX-required op id(s), missing_catalog_ops={missing_catalog_ops}"
        ));
    }
    if op_matrix_blocked_release_count != 0 {
        blockers.push(format!(
            "{evidence}: op_matrix_blocked_release_count must be zero, got {op_matrix_blocked_release_count}"
        ));
    }
    let expected_release_backend_rows = catalog_required_op_count.saturating_mul(3);
    if release_backend_row_count < expected_release_backend_rows
        || missing_release_backend_rows != 0
    {
        blockers.push(format!(
            "{evidence}: release_backend_row_count={release_backend_row_count}, expected {expected_release_backend_rows}, missing_release_backend_rows={missing_release_backend_rows}"
        ));
    }
    if value
        .get("failed_pairs")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX)
        != 0
    {
        blockers.push(format!("{evidence}: failed_pairs must be zero"));
    }
    if value
        .get("duplicate_op_ids")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|duplicates| !duplicates.is_empty())
    {
        blockers.push(format!("{evidence}: duplicate_op_ids must be empty"));
    }
}

fn inspect_conformance_matrix_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let op_count = value
        .get("op_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let distinct_op_count = value
        .get("distinct_op_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let catalog_required_op_count = value
        .get("catalog_required_op_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let catalog_covered_op_count = value
        .get("catalog_covered_op_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let missing_catalog_ops = value
        .get("missing_catalog_ops")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    let op_matrix_blocked_release_count = value
        .get("op_matrix_blocked_release_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    let op_matrix_errors = value
        .get("op_matrix_errors")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if op_matrix_errors != 0 {
        blockers.push(format!(
            "{evidence}: reports {op_matrix_errors} OP_MATRIX read/parse error(s)"
        ));
    }
    let fixture_input_count = value
        .get("fixture_input_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let expected_output_count = value
        .get("expected_output_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if op_count < 49 {
        blockers.push(format!(
            "{evidence}: op_count is {op_count}, below release floor 49"
        ));
    }
    if distinct_op_count < 49 {
        blockers.push(format!(
            "{evidence}: distinct_op_count is {distinct_op_count}, below release floor 49"
        ));
    }
    if catalog_required_op_count == 0
        || catalog_covered_op_count != catalog_required_op_count
        || missing_catalog_ops != 0
    {
        blockers.push(format!(
            "{evidence}: covers {catalog_covered_op_count}/{catalog_required_op_count} OP_MATRIX-required op id(s), missing_catalog_ops={missing_catalog_ops}"
        ));
    }
    if op_matrix_blocked_release_count != 0 {
        blockers.push(format!(
            "{evidence}: op_matrix_blocked_release_count must be zero, got {op_matrix_blocked_release_count}"
        ));
    }
    if fixture_input_count != op_count {
        blockers.push(format!(
            "{evidence}: fixture_input_count {fixture_input_count} must equal op_count {op_count}"
        ));
    }
    if expected_output_count != op_count {
        blockers.push(format!(
            "{evidence}: expected_output_count {expected_output_count} must equal op_count {op_count}"
        ));
    }
    if value
        .get("duplicate_op_ids")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|duplicates| !duplicates.is_empty())
    {
        blockers.push(format!("{evidence}: duplicate_op_ids must be empty"));
    }
    let backends = value
        .get("dispatch_backends")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    for required in ["cuda", "wgpu", "cpu-ref"] {
        if !backends
            .iter()
            .any(|backend| backend.as_str() == Some(required))
        {
            blockers.push(format!(
                "{evidence}: dispatch_backends must include `{required}`"
            ));
        }
    }
    let ci_gate_count = value
        .get("ci_blocking_gate_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if schema_version < 2 {
        blockers.push(format!(
            "{evidence}: schema_version is {schema_version}, expected >= 2"
        ));
    }
    if ci_gate_count < 3 {
        blockers.push(format!(
            "{evidence}: ci_blocking_gate_count is {ci_gate_count}, needs at least 3"
        ));
    }
    let ci_gates = value
        .get("ci_gates")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let required_ci_statuses = value
        .get("required_ci_statuses")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if required_ci_statuses == 0 {
        blockers.push(format!(
            "{evidence}: parsed zero required CI status context(s)"
        ));
    }
    let missing_required_ci_statuses = value
        .get("missing_required_ci_statuses")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_required_ci_statuses != 0 {
        blockers.push(format!(
            "{evidence}: {missing_required_ci_statuses} required CI status context(s) are missing from workflows"
        ));
    }
    let ci_status_scan_errors = value
        .get("ci_status_scan_errors")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if ci_status_scan_errors != 0 {
        blockers.push(format!(
            "{evidence}: {ci_status_scan_errors} CI status scan error(s) make workflow status evidence incomplete"
        ));
    }
    let path_filtered_required_workflows = value
        .get("path_filtered_required_workflows")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if path_filtered_required_workflows != 0 {
        blockers.push(format!(
            "{evidence}: {path_filtered_required_workflows} required workflow(s) still use path filters"
        ));
    }
    let missing_required_workflow_triggers = value
        .get("missing_required_workflow_triggers")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_required_workflow_triggers != 0 {
        blockers.push(format!(
            "{evidence}: {missing_required_workflow_triggers} required workflow(s) are missing pull_request + push main trigger coverage"
        ));
    }
    let missing_fail_closed_fanins = value
        .get("missing_fail_closed_fanins")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_fail_closed_fanins != 0 {
        blockers.push(format!(
            "{evidence}: {missing_fail_closed_fanins} required fan-in job(s) are missing fail-closed dependency checks"
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
            blockers.push(format!(
                "{evidence}: missing complete CI conformance workflow `{required_workflow}`"
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
            blockers.push(format!(
                "{evidence}: missing complete CI conformance gate `{required_gate}`"
            ));
        }
    }
}

fn inspect_release_conformance_log_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if schema_version < 2 {
        blockers.push(format!(
            "{evidence}: schema_version={schema_version}; release conformance log must be schema>=2"
        ));
    }
    if !value
        .get("command")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|command| {
            command.contains("cargo_full") && command.contains("release-conformance")
        })
    {
        blockers.push(format!(
            "{evidence}: command must run release-conformance through cargo_full"
        ));
    }
    let requested = value
        .get("requested_backends")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    for backend in ["cuda", "wgpu", "cpu-ref"] {
        if !requested
            .iter()
            .any(|entry| entry.as_str() == Some(backend))
        {
            blockers.push(format!(
                "{evidence}: requested_backends is missing `{backend}`"
            ));
        }
    }
    let statuses = value
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
                && status.get("exists").and_then(serde_json::Value::as_bool) == Some(true)
                && status
                    .get("bytes")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
                    > 0
                && status
                    .get("read_error")
                    .is_some_and(serde_json::Value::is_null)
        }) {
            blockers.push(format!(
                "{evidence}: does not prove non-empty readable conformance artifact `{artifact}`"
            ));
        }
    }
}

fn inspect_release_evidence_run_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
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

    let command_count = value
        .get("command_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let total_commands = value
        .get("total_commands")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let successful_commands = value
        .get("successful_commands")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let command_failures = value
        .get("command_failures")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    let artifact_failures = value
        .get("artifact_failures")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    let blocker_count = value
        .get("blockers")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    let required_count = value
        .get("required_command_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if schema_version < 2 || command_count < 13 || total_commands < 13 || required_count < 13 {
        blockers.push(format!(
            "{evidence}: schema_version={schema_version}, command_count={command_count}, total_commands={total_commands}, required_command_count={required_count}; structural release evidence must cover every generator with schema>=2"
        ));
    }
    if successful_commands != total_commands || command_failures != 0 || artifact_failures != 0 {
        blockers.push(format!(
            "{evidence}: successful_commands={successful_commands}, total_commands={total_commands}, command_failures={command_failures}, artifact_failures={artifact_failures}; release evidence run must be clean"
        ));
    }
    if blocker_count != 0 {
        blockers.push(format!(
            "{evidence}: release-evidence-run recorded {blocker_count} blocker(s)"
        ));
    }
    let Some(commands) = value.get("commands").and_then(serde_json::Value::as_array) else {
        blockers.push(format!("{evidence}: missing commands array"));
        return;
    };
    for (required, expected_artifacts) in REQUIRED_GENERATORS {
        let Some(command) = commands.iter().find(|command| {
            command
                .get("args")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|args| {
                    args.first().and_then(serde_json::Value::as_str) == Some(*required)
                })
                && command.get("required").and_then(serde_json::Value::as_bool) == Some(true)
        }) else {
            blockers.push(format!(
                "{evidence}: release-evidence run is missing required generator `{required}` with expected artifacts"
            ));
            continue;
        };
        if command.get("status").and_then(serde_json::Value::as_str) != Some("success") {
            blockers.push(format!(
                "{evidence}: release-evidence generator `{required}` did not report success"
            ));
        }
        let artifacts = command
            .get("expected_artifacts")
            .and_then(serde_json::Value::as_array)
            .map_or(&[][..], Vec::as_slice);
        let statuses = command
            .get("artifact_statuses")
            .and_then(serde_json::Value::as_array)
            .map_or(&[][..], Vec::as_slice);
        for expected in *expected_artifacts {
            if !artifacts.iter().any(|artifact| {
                artifact
                    .as_str()
                    .is_some_and(|artifact| artifact.ends_with(expected))
            }) {
                blockers.push(format!(
                    "{evidence}: release-evidence generator `{required}` does not declare expected artifact `{expected}`"
                ));
            }
            let Some(status) = statuses.iter().find(|status| {
                status
                    .get("path")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|path| path.ends_with(expected))
            }) else {
                blockers.push(format!(
                    "{evidence}: release-evidence generator `{required}` has no artifact status for `{expected}`"
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
                blockers.push(format!(
                    "{evidence}: release-evidence generator `{required}` artifact `{expected}` exists={exists} bytes={bytes} read_error={}",
                    read_error
                        .map(serde_json::Value::to_string)
                        .unwrap_or_else(|| "<missing>".to_string())
                ));
            }
        }
    }
}

fn inspect_pass_family_benchmark_manifest_semantics(
    evidence: &str,
    path: &Path,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    if value.get("backend").and_then(serde_json::Value::as_str) != Some("cuda") {
        blockers.push(format!(
            "{evidence}: backend must be cuda for the release path"
        ));
    }
    let required = value
        .get("required_case_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let Some(cases) = value.get("cases").and_then(serde_json::Value::as_array) else {
        blockers.push(format!("{evidence}: missing cases array"));
        return;
    };
    if cases.len() < required as usize || cases.len() < 4 {
        blockers.push(format!(
            "{evidence}: lists {} optimization benchmark case(s), needs at least {required} and never below 4",
            cases.len()
        ));
    }
    for required_family in REQUIRED_BENCHMARKED_OPTIMIZATION_FAMILIES {
        let covered = value
            .get("covered_pass_families")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|families| {
                families
                    .iter()
                    .any(|family| family.as_str() == Some(required_family))
            });
        if !covered {
            blockers.push(format!(
                "{evidence}: pass-family benchmark manifest does not cover `{required_family}`"
            ));
        }
    }
    if value
        .get("uncovered_pass_families")
        .and_then(serde_json::Value::as_array)
        .is_none_or(|families| !families.is_empty())
    {
        blockers.push(format!(
            "{evidence}: uncovered_pass_families must exist and be empty"
        ));
    }
    for required_case in [
        "lower.rewrites.impact.corpus",
        "foundation.optimizer.impact",
        "lower.egraph_saturation",
        "lower.alias_aware_optimizations",
    ] {
        if !cases.iter().any(|case| {
            case.get("case_id").and_then(serde_json::Value::as_str) == Some(required_case)
                && case.get("exists").and_then(serde_json::Value::as_bool) == Some(true)
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
            blockers.push(format!(
                "{evidence}: missing complete benchmark manifest entry for `{required_case}`"
            ));
        }
    }
    for case in cases {
        let Some(artifact) = case.get("artifact").and_then(serde_json::Value::as_str) else {
            blockers.push(format!("{evidence}: manifest case is missing artifact"));
            continue;
        };
        if case
            .get("covered_pass_families")
            .and_then(serde_json::Value::as_array)
            .is_none_or(|families| families.is_empty())
        {
            blockers.push(format!(
                "{evidence}: manifest case is missing covered_pass_families"
            ));
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
                blockers.push(format!(
                    "{evidence}: manifest case `{}` has non-empty `{field}`",
                    case.get("case_id")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("<unknown>")
                ));
            }
        }
        let read_error = case.get("read_error");
        if !read_error.is_some_and(serde_json::Value::is_null) {
            blockers.push(format!(
                "{evidence}: manifest case `{}` read_error={}",
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
                blockers.push(format!(
                    "{evidence}: manifest case `{}` has `{field}` below 30",
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
                blockers.push(format!(
                    "{evidence}: manifest case `{}` has non-positive `{field}`",
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
            blockers.push(format!(
                "{evidence}: manifest case `{}` does not prove optimized wall_ns p50 beats baseline_wall_ns p50",
                case.get("case_id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("<unknown>")
            ));
        }
        let Some(report) = read_referenced_release_json(path, artifact, blockers) else {
            continue;
        };
        let suffix = Path::new(artifact)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(artifact);
        let metrics = case
            .get("required_custom_metrics")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        if metrics.is_empty() {
            blockers.push(format!(
                "{evidence}: manifest case `{}` lists no required_custom_metrics",
                case.get("case_id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("<unknown>")
            ));
        }
        for metric in metrics.iter().filter_map(serde_json::Value::as_str) {
            if !benchmark_report_has_metric(&report, metric) {
                blockers.push(format!(
                    "{evidence}: referenced benchmark `{suffix}` is missing metric `{metric}`"
                ));
            }
        }
        let positive_metrics = case
            .get("required_positive_metrics")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        if positive_metrics.is_empty() {
            blockers.push(format!(
                "{evidence}: manifest case `{}` lists no required_positive_metrics",
                case.get("case_id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("<unknown>")
            ));
        }
        for metric in positive_metrics
            .iter()
            .filter_map(serde_json::Value::as_str)
        {
            if !benchmark_report_has_positive_metric(&report, metric) {
                blockers.push(format!(
                    "{evidence}: referenced benchmark `{suffix}` has no positive p50 metric `{metric}`"
                ));
            }
        }
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

fn inspect_optimization_corpus_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let required = value
        .get("required_min_cases")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(4_096);
    let generated = value
        .get("generated_cases")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let verified = value
        .get("verified_cases")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let optimized = value
        .get("optimized_cases")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let dataflow_analysis_cases = value
        .get("dataflow_analysis_cases")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let dataflow_analysis_optimized = value
        .get("dataflow_analysis_optimized_cases")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let non_converged = value
        .get("non_converged_cases")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    let ops_before = value
        .get("total_ops_before")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let ops_after = value
        .get("total_ops_after")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if required < 4_096 {
        blockers.push(format!(
            "{evidence}: required_min_cases is {required}; release floor is 4096"
        ));
    }
    if generated < required || generated < 4_096 {
        blockers.push(format!(
            "{evidence}: generated_cases is {generated}; needs at least {required} and never below 4096"
        ));
    }
    if verified != generated {
        blockers.push(format!(
            "{evidence}: verified_cases {verified} does not equal generated_cases {generated}"
        ));
    }
    if optimized == 0 {
        blockers.push(format!(
            "{evidence}: optimized_cases is zero; corpus does not prove optimizer firing"
        ));
    }
    if dataflow_analysis_cases == 0 {
        blockers.push(format!(
            "{evidence}: dataflow_analysis_cases is zero; corpus does not prove Weir-aware optimizer firing"
        ));
    }
    if dataflow_analysis_optimized < dataflow_analysis_cases {
        blockers.push(format!(
            "{evidence}: dataflow_analysis_optimized_cases {dataflow_analysis_optimized} is below dataflow_analysis_cases {dataflow_analysis_cases}"
        ));
    }
    if non_converged != 0 {
        blockers.push(format!(
            "{evidence}: non_converged_cases is {non_converged}; release requires zero"
        ));
    }
    if ops_before == 0 || ops_after == 0 {
        blockers.push(format!(
            "{evidence}: total_ops_before={ops_before}, total_ops_after={ops_after}; corpus must include real IR size evidence"
        ));
    }
}

fn read_referenced_release_json(
    manifest_path: &Path,
    artifact: &str,
    blockers: &mut Vec<String>,
) -> Option<serde_json::Value> {
    let artifact_path = if Path::new(artifact).is_absolute() {
        PathBuf::from(artifact)
    } else if artifact.starts_with("release/") {
        manifest_path
            .ancestors()
            .nth(4)
            .map(|workspace| workspace.join(artifact))
            .unwrap_or_else(|| PathBuf::from(artifact))
    } else {
        manifest_path
            .parent()
            .map(|parent| parent.join(artifact))
            .unwrap_or_else(|| PathBuf::from(artifact))
    };
    let text = match read_text_bounded(&artifact_path) {
        Ok(text) => text,
        Err(error) => {
            blockers.push(format!(
                "{}: failed to read referenced benchmark artifact `{}`: {error}",
                manifest_path.display(),
                artifact_path.display()
            ));
            return None;
        }
    };
    match serde_json::from_str(&text) {
        Ok(value) => Some(value),
        Err(error) => {
            blockers.push(format!(
                "{}: referenced benchmark artifact `{}` is invalid JSON: {error}",
                manifest_path.display(),
                artifact_path.display()
            ));
            None
        }
    }
}

fn benchmark_report_has_metric(report: &serde_json::Value, metric: &str) -> bool {
    report
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|cases| {
            cases.iter().any(|case| {
                case.get("metrics")
                    .and_then(serde_json::Value::as_object)
                    .is_some_and(|metrics| metrics.contains_key(metric))
            })
        })
}

fn benchmark_report_has_positive_metric(report: &serde_json::Value, metric: &str) -> bool {
    report
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|cases| {
            cases.iter().any(|case| {
                case.get("metrics")
                    .and_then(serde_json::Value::as_object)
                    .and_then(|metrics| metrics.get(metric))
                    .and_then(|value| metric_p50(Some(value)))
                    .is_some_and(|value| value > 0.0)
            })
        })
}

fn inspect_benchmark_cuda_environment_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let Some(environment) = value.get("environment") else {
        blockers.push(format!(
            "{evidence}: CUDA benchmark missing environment provenance"
        ));
        return;
    };
    let gpu_devices = environment
        .get("gpu_devices")
        .and_then(serde_json::Value::as_array);
    let first_gpu = gpu_devices.and_then(|devices| devices.first());
    if gpu_devices.is_none_or(|devices| devices.is_empty()) {
        blockers.push(format!(
            "{evidence}: CUDA benchmark has no nvidia-smi gpu_devices provenance"
        ));
    }
    if first_gpu
        .and_then(|device| device.get("name"))
        .and_then(serde_json::Value::as_str)
        .is_none_or(str::is_empty)
    {
        blockers.push(format!(
            "{evidence}: CUDA benchmark has no GPU model from nvidia-smi"
        ));
    }
    match first_gpu
        .and_then(|device| device.get("memory_total_mib"))
        .and_then(serde_json::Value::as_u64)
    {
        Some(mib) if mib >= 16 * 1024 => {}
        Some(mib) => blockers.push(format!(
            "{evidence}: CUDA benchmark GPU memory is {mib} MiB, below release floor 16384 MiB"
        )),
        None => blockers.push(format!(
            "{evidence}: CUDA benchmark has no GPU memory_total_mib from nvidia-smi"
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
        (Some(major), Some(minor)) => blockers.push(format!(
            "{evidence}: CUDA benchmark compute capability is {major}.{minor}, below release floor 8.0"
        )),
        _ => blockers.push(format!(
            "{evidence}: CUDA benchmark has no compute capability from nvidia-smi"
        )),
    }
    for field in ["nvidia_driver_version", "nvidia_cuda_version"] {
        if environment
            .get(field)
            .and_then(serde_json::Value::as_str)
            .is_none_or(str::is_empty)
        {
            blockers.push(format!(
                "{evidence}: CUDA benchmark environment is missing `{field}` from nvidia-smi"
            ));
        }
    }
}

fn inspect_hygiene_matrix_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    if value
        .get("finding_summary")
        .and_then(serde_json::Value::as_array)
        .is_none()
    {
        blockers.push(format!("{evidence}: missing finding_summary"));
    }
    let finding_count = value
        .get("findings")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    let summary_count = value
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
        blockers.push(format!(
            "{evidence}: finding_summary count {summary_count} does not match findings count {finding_count}"
        ));
    }
    inspect_hygiene_release_surface_coverage(evidence, value, blockers);
    let Some(roots) = value
        .get("scanned_roots")
        .and_then(serde_json::Value::as_array)
    else {
        blockers.push(format!("{evidence}: missing scanned_roots"));
        return;
    };
    for required_root in [
        "libs/performance/matching/vyre",
        "libs/dataflow/weir",
        "tools/vyrec",
        "libs/tools/security-analysis-consumer",
        "libs/shared/security-grammar-gen",
    ] {
        if !roots.iter().any(|root| {
            root.as_str()
                .is_some_and(|root| root.contains(required_root))
        }) {
            blockers.push(format!(
                "{evidence}: scanned_roots is missing `{required_root}`"
            ));
        }
    }
}

fn inspect_hygiene_release_surface_coverage(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let Some(coverage) = value.get("release_surface_coverage") else {
        blockers.push(format!("{evidence}: missing release_surface_coverage"));
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
            blockers.push(format!(
                "{evidence}: release_surface_coverage.{field} must be true"
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
                blockers.push(format!(
                    "{evidence}: release_surface_coverage.{field} is missing `{required_value}`"
                ));
            }
        }
    }
}

fn inspect_optimization_analysis_fixture_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let missing_required = value
        .get("missing_required_families")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_required != 0 {
        blockers.push(format!(
            "{evidence}: missing_required_families has {missing_required} entrie(s), expected zero"
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
        blockers.push(format!(
            "{evidence}: total_fixture_cases={total_fixture_cases}, total_triggered_cases={total_triggered_cases}; needs 512 fully-triggered A13-A16 cases"
        ));
    }
    let Some(families) = value.get("families").and_then(serde_json::Value::as_array) else {
        blockers.push(format!("{evidence}: missing families array"));
        return;
    };
    for required in [
        "A13-coalesce-fixture",
        "A14-shared-mem-promote-fixture",
        "A15-bank-conflict-fixture",
        "A16-vec-pack-fixture",
    ] {
        let Some(family) = families.iter().find(|family| {
            family.get("family").and_then(serde_json::Value::as_str) == Some(required)
        }) else {
            blockers.push(format!(
                "{evidence}: missing analysis fixture family `{required}`"
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
            blockers.push(format!(
                "{evidence}: analysis fixture `{required}` has cases={cases}, triggered_cases={triggered}, analysis_sites={analysis_sites}; needs at least 128 cases, every case triggered, and at least one analysis site per case"
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
                        blockers.push(format!("{evidence}: A13 fixture has zero `{field}`"));
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
                        blockers.push(format!("{evidence}: A14 fixture has zero `{field}`"));
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
                        blockers.push(format!("{evidence}: A15 fixture has zero `{field}`"));
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
                        blockers.push(format!("{evidence}: A16 fixture has zero `{field}`"));
                    }
                }
            }
            _ => {}
        }
    }
}

fn inspect_optimization_family_manifest_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let Some(families) = value.get("families").and_then(serde_json::Value::as_array) else {
        blockers.push(format!("{evidence}: missing families array"));
        return;
    };
    if families.len() < 14 {
        blockers.push(format!(
            "{evidence}: lists {} optimization families; needs at least 14 required release families",
            families.len()
        ));
    }
    let declared_required = value
        .get("required_family_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if declared_required < 14 {
        blockers.push(format!(
            "{evidence}: declares {declared_required} required optimization families; needs all 14 release families"
        ));
    }
    let missing_required = value
        .get("missing_required_families")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_required != 0 {
        blockers.push(format!(
            "{evidence}: missing_required_families reports {missing_required} missing required optimization family/families"
        ));
    }
    for family in families {
        let name = family
            .get("family")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if family
            .get("cases")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            blockers.push(format!(
                "{evidence}: optimization family `{name}` has zero generated cases"
            ));
        }
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
                family.get("family").and_then(serde_json::Value::as_str) == Some(required)
            })
            .and_then(|family| family.get("cases").and_then(serde_json::Value::as_u64))
            .unwrap_or(0);
        if required_cases < 128 {
            blockers.push(format!(
                "{evidence}: required optimization family `{required}` has {required_cases} generated case(s), needs at least 128"
            ));
        }
    }
}

fn inspect_optimization_case_manifest_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let pass_instances = value
        .get("pass_instance_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let generated_cases = value
        .get("generated_cases")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let unique_case_ids = value
        .get("unique_case_ids")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if pass_instances < 4_096 {
        blockers.push(format!(
            "{evidence}: pass_instance_count {pass_instances} is below release floor 4096"
        ));
    }
    if generated_cases != pass_instances {
        blockers.push(format!(
            "{evidence}: generated_cases {generated_cases} does not match pass_instance_count {pass_instances}"
        ));
    }
    if unique_case_ids != pass_instances {
        blockers.push(format!(
            "{evidence}: unique_case_ids {unique_case_ids} does not match pass_instance_count {pass_instances}"
        ));
    }
    let duplicate_case_ids = value
        .get("duplicate_case_ids")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if duplicate_case_ids != 0 {
        blockers.push(format!(
            "{evidence}: duplicate_case_ids contains {duplicate_case_ids} duplicate id(s)"
        ));
    }
    let family_count = value
        .get("family_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let required_family_count = value
        .get("required_family_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(12);
    if family_count < required_family_count || family_count < 12 {
        blockers.push(format!(
            "{evidence}: family_count {family_count} is below required family count {required_family_count}"
        ));
    }
    let Some(entries) = value.get("entries").and_then(serde_json::Value::as_array) else {
        blockers.push(format!("{evidence}: missing entries array"));
        return;
    };
    if entries.len() as u64 != pass_instances {
        blockers.push(format!(
            "{evidence}: entries array has {} entries, pass_instance_count is {pass_instances}",
            entries.len()
        ));
    }
    for field in [
        "cases_with_child_bodies",
        "cases_with_bindings",
        "cases_with_literals",
    ] {
        if value
            .get(field)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            blockers.push(format!("{evidence}: `{field}` must be nonzero"));
        }
    }
    for entry in entries {
        let id = entry
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if entry
            .get("id")
            .and_then(serde_json::Value::as_str)
            .is_none_or(str::is_empty)
            || entry
                .get("family")
                .and_then(serde_json::Value::as_str)
                .is_none_or(str::is_empty)
        {
            blockers.push(format!(
                "{evidence}: case manifest entry `{id}` is missing id or family"
            ));
        }
        if entry
            .get("total_ops")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            blockers.push(format!(
                "{evidence}: case manifest entry `{id}` has zero total_ops"
            ));
        }
    }
}

fn is_test_suite_evidence(evidence: &str) -> bool {
    [
        "unit-suite.json",
        "adversarial-suite.json",
        "property-suite.json",
        "conformance-suite.json",
        "corpus-suite.json",
        "benchmark-suite.json",
        "gap-suite.json",
        "fuzz-suite.json",
    ]
    .iter()
    .any(|suffix| evidence.ends_with(suffix))
}

fn inspect_oversized_test_closure_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    if value.get("closed").and_then(serde_json::Value::as_bool) != Some(true) {
        blockers.push(format!("{evidence}: oversized test closure must be closed"));
    }
    if value
        .get("total_oversized_files")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX)
        != 0
    {
        blockers.push(format!("{evidence}: total_oversized_files must be zero"));
    }
    if value
        .get("total_god_test_candidates")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX)
        != 0
    {
        blockers.push(format!(
            "{evidence}: total_god_test_candidates must be zero"
        ));
    }
    if value
        .get("required_split_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX)
        != 0
    {
        blockers.push(format!("{evidence}: required_split_count must be zero"));
    }
    if !value
        .get("oversized_files")
        .and_then(serde_json::Value::as_array)
        .is_some_and(Vec::is_empty)
    {
        blockers.push(format!(
            "{evidence}: oversized_files must be an empty array"
        ));
    }
    if !value
        .get("god_test_candidates")
        .and_then(serde_json::Value::as_array)
        .is_some_and(Vec::is_empty)
    {
        blockers.push(format!(
            "{evidence}: god_test_candidates must be an empty array"
        ));
    }
}

fn inspect_test_matrix_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let test_files = value
        .get("test_files")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if test_files == 0 {
        blockers.push(format!("{evidence}: test_files is zero"));
    }
    for (field, label) in [
        ("vyre_test_files", "Vyre"),
        ("weir_test_files", "Weir"),
        ("vyrec_test_files", "tools/vyrec"),
    ] {
        if value
            .get(field)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            blockers.push(format!(
                "{evidence}: {label} release-surface test file count is zero"
            ));
        }
    }
    let layers = value
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
            blockers.push(format!("{evidence}: missing `{required}` test layer"));
        }
    }
    if value
        .get("oversized_files")
        .and_then(serde_json::Value::as_array)
        .is_none_or(|files| !files.is_empty())
    {
        blockers.push(format!(
            "{evidence}: oversized_files must exist and be empty"
        ));
    }
    if value
        .get("god_test_candidates")
        .and_then(serde_json::Value::as_array)
        .is_none_or(|files| !files.is_empty())
    {
        blockers.push(format!(
            "{evidence}: god_test_candidates must exist and be empty"
        ));
    }
    inspect_surface_entries(evidence, value.get("surface_coverages"), blockers);
}

fn inspect_surface_coverage_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    inspect_top_level_blockers(evidence, value, blockers);
    inspect_surface_entries(evidence, value.get("surfaces"), blockers);
}

fn inspect_top_level_blockers(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let blocker_count = value
        .get("blockers")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if blocker_count != 0 {
        blockers.push(format!("{evidence}: reports {blocker_count} blocker(s)"));
    }
}

fn inspect_surface_entries(
    evidence: &str,
    maybe_surfaces: Option<&serde_json::Value>,
    blockers: &mut Vec<String>,
) {
    let Some(surfaces) = maybe_surfaces.and_then(serde_json::Value::as_array) else {
        blockers.push(format!(
            "{evidence}: missing release surface coverage array"
        ));
        return;
    };
    if surfaces.len() != 3 {
        blockers.push(format!(
            "{evidence}: release surface coverage must contain exactly Vyre, Weir, and tools/vyrec"
        ));
    }
    for required_surface in ["vyre", "weir", "vyrec"] {
        let Some(surface) = surfaces.iter().find(|surface| {
            surface.get("surface").and_then(serde_json::Value::as_str) == Some(required_surface)
        }) else {
            blockers.push(format!(
                "{evidence}: missing `{required_surface}` release surface coverage"
            ));
            continue;
        };
        if surface
            .get("file_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            blockers.push(format!(
                "{evidence}: `{required_surface}` release surface has zero test files"
            ));
        }
        if surface
            .get("assertion_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            blockers.push(format!(
                "{evidence}: `{required_surface}` release surface has zero assertions"
            ));
        }
        if surface
            .get("entrypoint_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            blockers.push(format!(
                "{evidence}: `{required_surface}` release surface has zero executable test entrypoints"
            ));
        }
        let missing_layers = surface
            .get("missing_layers")
            .and_then(serde_json::Value::as_array)
            .map_or(usize::MAX, Vec::len);
        if missing_layers != 0 {
            blockers.push(format!(
                "{evidence}: `{required_surface}` release surface reports {missing_layers} missing test layer(s)"
            ));
        }
        let blockers_count = surface
            .get("blockers")
            .and_then(serde_json::Value::as_array)
            .map_or(usize::MAX, Vec::len);
        if blockers_count != 0 {
            blockers.push(format!(
                "{evidence}: `{required_surface}` release surface reports {blockers_count} blocker(s)"
            ));
        }
    }
}

fn inspect_modularization_map_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let Some(directories) = value
        .get("directories")
        .and_then(serde_json::Value::as_array)
    else {
        blockers.push(format!("{evidence}: missing directories array"));
        return;
    };
    if directories.is_empty() {
        blockers.push(format!("{evidence}: directories array is empty"));
    }
    for required_surface in ["vyre", "weir", "vyrec"] {
        if !directories.iter().any(|directory| {
            directory.get("surface").and_then(serde_json::Value::as_str) == Some(required_surface)
        }) {
            blockers.push(format!(
                "{evidence}: modularization map is missing `{required_surface}` surface directories"
            ));
        }
    }
    for directory in directories {
        if directory.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
            let path = directory
                .get("path")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("<unknown>");
            blockers.push(format!("{evidence}: modular directory `{path}` is missing"));
        }
    }
}

fn inspect_suite_evidence_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let file_count = value
        .get("file_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if file_count == 0 {
        blockers.push(format!("{evidence}: file_count is zero"));
    }
    let vyre_file_count = value
        .get("vyre_file_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let dataflow_consumer_file_count = value
        .get("dataflow_consumer_file_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let vyrec_file_count = value
        .get("vyrec_file_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if vyre_file_count == 0 {
        blockers.push(format!("{evidence}: vyre_file_count is zero"));
    }
    if dataflow_consumer_file_count == 0 {
        blockers.push(format!("{evidence}: dataflow_consumer_file_count is zero"));
    }
    if vyrec_file_count == 0 {
        blockers.push(format!("{evidence}: vyrec_file_count is zero"));
    }
    let Some(files) = value.get("files").and_then(serde_json::Value::as_array) else {
        blockers.push(format!("{evidence}: missing files array"));
        return;
    };
    if files.is_empty() {
        blockers.push(format!("{evidence}: files array is empty"));
        return;
    }
    let active_files = files
        .iter()
        .filter(|file| {
            file.get("has_test_entrypoint")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
                || file
                    .get("assertion_count")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
                    > 0
                || file
                    .get("layers")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|layers| {
                        layers
                            .iter()
                            .any(|layer| layer.as_str() == Some("benchmark"))
                    })
        })
        .count();
    if active_files == 0 {
        blockers.push(format!(
            "{evidence}: suite has no assertion-bearing, entrypoint-bearing, or benchmark file"
        ));
    }
}

fn is_before_after_benchmark_evidence(evidence: &str) -> bool {
    [
        "lower-rewrite-impact-before-after.json",
        "optimizer-impact-cuda.json",
        "pass-family-benchmarks.json",
        "egraph-before-after.json",
        "alias-aware-before-after.json",
    ]
    .iter()
    .any(|suffix| evidence.ends_with(suffix))
}

fn inspect_before_after_benchmark_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    if value
        .get("selected_backend")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|backend| backend != "cuda")
    {
        blockers.push(format!("{evidence}: selected_backend must be cuda"));
    }
    if evidence.ends_with("cpu-only-100x-proof.json") {
        if value
            .get("source_fingerprint")
            .and_then(serde_json::Value::as_str)
            .is_none_or(str::is_empty)
        {
            blockers.push(format!(
                "{evidence}: aggregate proof must preserve source_fingerprint"
            ));
        }
        if value.get("git").is_none_or(serde_json::Value::is_null) {
            blockers.push(format!(
                "{evidence}: aggregate proof must preserve git provenance object"
            ));
        }
        let contract_case_count = value
            .get("cpu_sota_100x_contract_case_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if contract_case_count < 10 {
            blockers.push(format!(
                "{evidence}: cpu_sota_100x_contract_case_count is {contract_case_count}; needs at least 10"
            ));
        }
        let passing_case_count = value
            .get("cpu_sota_100x_passing_case_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if passing_case_count < 10 {
            blockers.push(format!(
                "{evidence}: cpu_sota_100x_passing_case_count is {passing_case_count}; needs at least 10"
            ));
        }
        let min_wall_samples = value
            .get("min_wall_samples")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if min_wall_samples < 30 {
            blockers.push(format!(
                "{evidence}: min_wall_samples is {min_wall_samples}; needs at least 30"
            ));
        }
        let min_baseline_wall_samples = value
            .get("min_baseline_wall_samples")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if min_baseline_wall_samples < 30 {
            blockers.push(format!(
                "{evidence}: min_baseline_wall_samples is {min_baseline_wall_samples}; needs at least 30"
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
            if value
                .get(field)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                == 0
            {
                blockers.push(format!(
                    "{evidence}: aggregate proof has non-positive `{field}`"
                ));
            }
        }
    }
    let Some(cases) = value.get("cases").and_then(serde_json::Value::as_array) else {
        blockers.push(format!("{evidence}: missing cases array"));
        return;
    };
    if cases.is_empty() {
        blockers.push(format!("{evidence}: cases array is empty"));
        return;
    }
    for case in cases {
        let id = case
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        let metrics = case.get("metrics").and_then(serde_json::Value::as_object);
        let wall = metrics.and_then(active_gpu_metric_p50);
        let baseline = metrics.and_then(|metrics| metric_p50(metrics.get("baseline_wall_ns")));
        let wall_samples = metrics
            .and_then(|metrics| metric_samples(metrics.get("wall_ns")))
            .unwrap_or(0);
        if wall_samples < 30 {
            blockers.push(format!(
                "{evidence}: case `{id}` has {wall_samples} wall_ns sample(s), needs at least 30"
            ));
        }
        let baseline_wall_samples = metrics
            .and_then(|metrics| metric_samples(metrics.get("baseline_wall_ns")))
            .unwrap_or(0);
        if baseline_wall_samples < 30 {
            blockers.push(format!(
                "{evidence}: case `{id}` has {baseline_wall_samples} baseline_wall_ns sample(s), needs at least 30"
            ));
        }
        require_benchmark_metric_percentiles(evidence, id, metrics, "wall_ns", blockers);
        require_benchmark_metric_percentiles(evidence, id, metrics, "baseline_wall_ns", blockers);
        let egraph_quality_win = evidence.ends_with("egraph-before-after.json")
            && metrics
                .and_then(|metrics| {
                    metric_p50(metrics.get("egraph_output_ops"))
                        .zip(metric_p50(metrics.get("egraph_baseline_ops_after")))
                })
                .is_some_and(|(output, baseline)| output < baseline)
            && metrics
                .and_then(|metrics| metric_p50(metrics.get("egraph_applied_rewrites")))
                .is_some_and(|rewrites| rewrites > 0.0)
            && metrics
                .and_then(|metrics| metric_p50(metrics.get("egraph_bitwise_case_count")))
                .is_some_and(|cases| cases >= 192.0)
            && metrics
                .and_then(|metrics| metric_p50(metrics.get("egraph_boolean_case_count")))
                .is_some_and(|cases| cases >= 128.0);
        if evidence.ends_with("alias-aware-before-after.json") {
            for metric in [
                "alias_pass_wins",
                "alias_fact_count",
                "alias_cross_binding_fact_count",
                "reaching_def_fact_count",
            ] {
                if !metrics
                    .and_then(|metrics| metric_p50(metrics.get(metric)))
                    .is_some_and(|value| value > 0.0)
                {
                    blockers.push(format!(
                        "{evidence}: case `{id}` must include positive p50 `{metric}`"
                    ));
                }
            }
        }
        match (wall, baseline) {
            (Some(wall), Some(baseline)) if wall < baseline => {}
            (Some(_), Some(_)) if egraph_quality_win => {}
            (Some(_), Some(_)) if before_after_semantic_win(id, metrics) => {}
            (Some(wall), Some(baseline)) => blockers.push(format!(
                "{evidence}: case `{id}` did not improve p50 wall time: wall={wall:.2}, baseline={baseline:.2}"
            )),
            _ => blockers.push(format!(
                "{evidence}: case `{id}` must include p50 wall_ns and baseline_wall_ns"
            )),
        }
    }
}

fn before_after_semantic_win(
    case_id: &str,
    metrics: Option<&serde_json::Map<String, serde_json::Value>>,
) -> bool {
    let Some(metrics) = metrics else {
        return false;
    };
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

fn inspect_release_tag_plan_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
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
        if value.get(field).and_then(serde_json::Value::as_str) != Some(expected) {
            blockers.push(format!("{evidence}: {field} must be `{expected}`"));
        }
    }
    let order = value
        .get("tag_creation_order")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    for required in [
        "vyre-v0.4.2-rc.1",
        "weir-v0.1.0-rc.1",
        "vyre-0.4.2-weir-0.1.0-rc.1",
        "vyre-v0.4.2",
        "weir-v0.1.0",
        "vyre-0.4.2-weir-0.1.0",
    ] {
        if !order.iter().any(|entry| entry.as_str() == Some(required)) {
            blockers.push(format!(
                "{evidence}: tag_creation_order is missing `{required}`"
            ));
        }
    }
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
        if !matches!((rc_index, final_index), (Some(left), Some(right)) if left < right) {
            blockers.push(format!(
                "{evidence}: tag_creation_order must list `{rc}` before `{final_tag}`"
            ));
        }
    }
    if !value
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
        blockers.push(format!(
            "{evidence}: required_gate_before_rc_tag must include version matrix, completion audit, release gate, branch-protection application, and cargo_full"
        ));
    }
    if !value
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
        blockers.push(format!(
            "{evidence}: required_gate_before_tag must include version matrix, completion audit, release gate, branch-protection application, and cargo_full"
        ));
    }
    let version_blockers = value
        .get("version_matrix_blocker_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    if version_blockers != 0 {
        blockers.push(format!(
            "{evidence}: version_matrix_blocker_count is {version_blockers}, expected zero"
        ));
    }
}

fn inspect_feature_matrix_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let missing_required = value
        .get("missing_required_release_packages")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_required != 0 {
        blockers.push(format!(
            "{evidence}: missing_required_release_packages has {missing_required} entrie(s), expected zero"
        ));
    }
    let Some(packages) = value.get("packages").and_then(serde_json::Value::as_array) else {
        blockers.push(format!("{evidence}: missing packages array"));
        return;
    };
    if !packages
        .iter()
        .any(|entry| entry.get("name").and_then(serde_json::Value::as_str) == Some("vyrec"))
    {
        blockers.push(format!("{evidence}: missing package `vyrec`"));
    }
    for package in ["vyre", "vyre-driver-cuda", "vyre-driver-wgpu"] {
        let Some(entry) = packages
            .iter()
            .find(|entry| entry.get("name").and_then(serde_json::Value::as_str) == Some(package))
        else {
            blockers.push(format!("{evidence}: missing package `{package}`"));
            continue;
        };
        if entry
            .get("default_feature_members")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|members| !members.is_empty())
        {
            blockers.push(format!(
                "{evidence}: package `{package}` default feature set must be empty"
            ));
        }
    }
    for (package, required_features) in [
        ("vyre-driver-cuda", &["cuda"][..]),
        ("vyre-driver-wgpu", &["wgpu"][..]),
        ("weir", &["default", "serde"][..]),
    ] {
        let Some(entry) = packages
            .iter()
            .find(|entry| entry.get("name").and_then(serde_json::Value::as_str) == Some(package))
        else {
            blockers.push(format!("{evidence}: missing package `{package}`"));
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
                .any(|feature| feature.as_str() == Some(*required))
            {
                blockers.push(format!(
                    "{evidence}: package `{package}` missing feature `{required}`"
                ));
            }
        }
    }
    let Some(vyre) = packages
        .iter()
        .find(|entry| entry.get("name").and_then(serde_json::Value::as_str) == Some("vyre"))
    else {
        return;
    };
    let features = vyre
        .get("features")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    for required in ["cuda", "wgpu"] {
        if !features
            .iter()
            .any(|feature| feature.as_str() == Some(required))
        {
            blockers.push(format!(
                "{evidence}: top-level vyre crate missing feature `{required}`"
            ));
        }
    }
}

fn inspect_metadata_matrix_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    for (field, label) in [
        ("publishable_package_count", "publishable package"),
        ("vyre_package_count", "Vyre package"),
        ("weir_package_count", "Weir package"),
        (
            "non_publishable_release_surface_count",
            "non-publishable release-surface package",
        ),
    ] {
        if value
            .get(field)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            blockers.push(format!("{evidence}: contains zero {label}(s)"));
        }
    }
    if value
        .get("parser_release_surface_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0)
        < 2
    {
        blockers.push(format!(
            "{evidence}: parser_release_surface_count must cover vyrec and vyre-frontend-c"
        ));
    }
    let missing_required = value
        .get("missing_required_release_surfaces")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_required != 0 {
        blockers.push(format!(
            "{evidence}: missing_required_release_surfaces has {missing_required} entrie(s), expected zero"
        ));
    }
    if value
        .get("root_patch_section_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX)
        != 0
    {
        blockers.push(format!(
            "{evidence}: root_patch_section_count must be present and zero"
        ));
    }
    let Some(packages) = value.get("packages").and_then(serde_json::Value::as_array) else {
        blockers.push(format!("{evidence}: missing packages array"));
        return;
    };
    if !packages.iter().any(|package| {
        package.get("name").and_then(serde_json::Value::as_str) == Some("vyrec")
            && package.get("version").and_then(serde_json::Value::as_str) == Some("0.1.0")
            && package.get("readme").and_then(serde_json::Value::as_str) == Some("README.md")
            && package
                .get("release_surface")
                .and_then(serde_json::Value::as_str)
                == Some("parser-cli")
    }) {
        blockers.push(format!(
            "{evidence}: missing vyrec 0.1.0 parser-cli package metadata with README.md"
        ));
    }
    if !packages.iter().any(|package| {
        package.get("name").and_then(serde_json::Value::as_str) == Some("vyre-frontend-c")
            && package.get("version").and_then(serde_json::Value::as_str) == Some("0.4.2")
            && package.get("readme").and_then(serde_json::Value::as_str) == Some("README.md")
            && package
                .get("release_kind")
                .and_then(serde_json::Value::as_str)
                == Some("non-publishable-release-surface")
            && package
                .get("release_surface")
                .and_then(serde_json::Value::as_str)
                == Some("c-frontend")
    }) {
        blockers.push(format!(
            "{evidence}: missing vyre-frontend-c 0.4.2 c-frontend non-publishable release-surface metadata with README.md"
        ));
    }
    for (package_name, backend_surface) in [
        ("vyre-driver-cuda", "cuda-backend"),
        ("vyre-driver-wgpu", "wgpu-backend"),
    ] {
        if !packages.iter().any(|package| {
            package.get("name").and_then(serde_json::Value::as_str) == Some(package_name)
                && package.get("version").and_then(serde_json::Value::as_str) == Some("0.4.2")
                && package.get("readme").and_then(serde_json::Value::as_str) == Some("README.md")
                && package
                    .get("release_kind")
                    .and_then(serde_json::Value::as_str)
                    == Some("publishable-crate")
                && package
                    .get("release_surface")
                    .and_then(serde_json::Value::as_str)
                    == Some(backend_surface)
        }) {
            blockers.push(format!(
                "{evidence}: missing {package_name} 0.4.2 publishable {backend_surface} release-surface metadata with README.md"
            ));
        }
    }
    for package in packages {
        let name = package
            .get("name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        let release_kind = package
            .get("release_kind")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if release_kind == "internal-tooling" {
            continue;
        }
        let release_group = package
            .get("release_group")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let expected = package
            .get("expected_version")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let version = package
            .get("version")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if expected.is_empty() || version != expected {
            blockers.push(format!(
                "{evidence}: package `{name}` release_group `{release_group}` has version `{version}`, expected `{expected}`"
            ));
        }
        if package
            .get("example_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            blockers.push(format!(
                "{evidence}: release package `{name}` has zero examples or README usage blocks"
            ));
        }
        if release_kind == "publishable-crate"
            && package
                .get("has_runnable_example")
                .and_then(serde_json::Value::as_bool)
                != Some(true)
        {
            blockers.push(format!(
                "{evidence}: publishable release package `{name}` has no runnable examples/*.rs"
            ));
        }
        if release_kind == "publishable-crate"
            && package
                .get("has_api_referencing_example")
                .and_then(serde_json::Value::as_bool)
                != Some(true)
        {
            blockers.push(format!(
                "{evidence}: publishable release package `{name}` has no API-referencing examples/*.rs"
            ));
        }
    }
}

fn inspect_package_readiness_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let blocker_count = value
        .get("blockers")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if blocker_count != 0 {
        blockers.push(format!(
            "{evidence}: package readiness still reports {blocker_count} blocker(s)"
        ));
    }
    if value
        .get("release_train")
        .and_then(|train| train.get("cuda_release_path"))
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        blockers.push(format!("{evidence}: cuda_release_path must be true"));
    }
    let publish_order = value
        .get("publish_order")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    if publish_order.len() < 20 {
        blockers.push(format!(
            "{evidence}: publish_order contains {} package(s), expected the full release train",
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
        if !publish_order
            .iter()
            .any(|entry| entry.get("package").and_then(serde_json::Value::as_str) == Some(required))
        {
            blockers.push(format!("{evidence}: publish_order is missing `{required}`"));
        }
    }
    let missing_metadata = value
        .get("missing_metadata_packages")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    let extra_metadata = value
        .get("extra_metadata_packages")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_metadata != 0 || extra_metadata != 0 {
        blockers.push(format!(
            "{evidence}: publish_order and metadata disagree: {missing_metadata} missing, {extra_metadata} extra"
        ));
    }
    if value
        .get("dependency_order_edges")
        .and_then(serde_json::Value::as_array)
        .is_none_or(Vec::is_empty)
    {
        blockers.push(format!("{evidence}: dependency_order_edges is empty"));
    }
    if value
        .get("versioned_local_dependencies")
        .and_then(serde_json::Value::as_array)
        .is_none_or(Vec::is_empty)
    {
        blockers.push(format!("{evidence}: versioned_local_dependencies is empty"));
    }
    let verify_passed = value
        .get("package_verify_passed")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    for required in ["vyre-macros@0.4.2", "vyre-spec@0.4.2", "vyre-lints@0.4.2"] {
        if !verify_passed
            .iter()
            .any(|entry| entry.as_str() == Some(required))
        {
            blockers.push(format!(
                "{evidence}: package_verify_passed is missing `{required}`"
            ));
        }
    }
    let non_publish_surfaces = value
        .get("non_publish_release_surfaces")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    for required in ["vyre-frontend-c", "vyrec"] {
        if !non_publish_surfaces
            .iter()
            .any(|entry| entry.get("package").and_then(serde_json::Value::as_str) == Some(required))
        {
            blockers.push(format!(
                "{evidence}: non_publish_release_surfaces is missing `{required}`"
            ));
        }
    }
}

fn inspect_public_launch_state_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let blocker_count = value
        .get("blockers")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if blocker_count != 0 {
        blockers.push(format!(
            "{evidence}: public launch is incomplete with {blocker_count} blocker(s)"
        ));
    }
    if value
        .get("completion_status")
        .and_then(serde_json::Value::as_str)
        != Some("complete")
    {
        blockers.push(format!("{evidence}: completion_status is not `complete`"));
    }
    let external_actions = value
        .get("external_actions")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    for required in [
        "cargo publish approved crates in dependency order",
        "make repositories public",
        "git push release branch and tags",
    ] {
        let Some(action) = external_actions.iter().find(|action| {
            action.get("action").and_then(serde_json::Value::as_str) == Some(required)
        }) else {
            blockers.push(format!(
                "{evidence}: external action `{required}` is missing"
            ));
            continue;
        };
        if action.get("status").and_then(serde_json::Value::as_str) != Some("complete") {
            blockers.push(format!(
                "{evidence}: external action `{required}` is not complete"
            ));
        }
    }
}

fn inspect_docs_matrix_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    if value
        .get("curated_proof_docs_preserved")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        blockers.push(format!(
            "{evidence}: curated_proof_docs_preserved must be true"
        ));
    }
    let Some(docs) = value.get("docs").and_then(serde_json::Value::as_array) else {
        blockers.push(format!("{evidence}: missing docs array"));
        return;
    };
    if docs.is_empty() {
        blockers.push(format!("{evidence}: docs array is empty"));
        return;
    }
    if value
        .get("limitation_findings")
        .and_then(serde_json::Value::as_array)
        .is_none_or(|findings| !findings.is_empty())
    {
        blockers.push(format!(
            "{evidence}: limitation_findings must exist and be empty"
        ));
    }
    for doc in docs {
        let id = doc
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if doc.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
            blockers.push(format!("{evidence}: required doc `{id}` does not exist"));
        }
        if doc
            .get("contains_release_evidence_rule")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        {
            blockers.push(format!(
                "{evidence}: required doc `{id}` does not reference release evidence"
            ));
        }
        if doc
            .get("evidence_artifact_ref_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            blockers.push(format!(
                "{evidence}: required doc `{id}` has zero concrete evidence artifact references"
            ));
        }
        if doc
            .get("missing_evidence_artifact_refs")
            .and_then(serde_json::Value::as_array)
            .is_none_or(|refs| !refs.is_empty())
        {
            blockers.push(format!(
                "{evidence}: required doc `{id}` references missing evidence artifacts"
            ));
        }
        if doc
            .get("missing_topics")
            .and_then(serde_json::Value::as_array)
            .is_none_or(|topics| !topics.is_empty())
        {
            blockers.push(format!(
                "{evidence}: required doc `{id}` has missing topics"
            ));
        }
        if doc
            .get("unresolved_markers")
            .and_then(serde_json::Value::as_array)
            .is_none_or(|markers| !markers.is_empty())
        {
            blockers.push(format!(
                "{evidence}: required doc `{id}` has unresolved markers"
            ));
        }
    }
}

fn inspect_release_axes_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let source_artifacts = value
        .get("source_artifacts")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if source_artifacts < 12 {
        blockers.push(format!(
            "{evidence}: source_artifacts has {source_artifacts} entrie(s), needs at least 12"
        ));
    }
    for field in [
        "warm_us_per_file",
        "cold_pipeline_build_ms",
        "gbs_scan_throughput",
        "ulp_drift_max",
        "max_vram_mib",
    ] {
        if value.get(field).is_none_or(serde_json::Value::is_null) {
            blockers.push(format!("{evidence}: missing benchmark axis `{field}`"));
        }
    }
}

fn inspect_backend_suite_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if schema_version < 2 {
        blockers.push(format!(
            "{evidence}: schema_version={schema_version}; backend suite evidence must be schema>=2"
        ));
    }
    let family_count = value
        .get("family_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let artifact_count = value
        .get("artifacts")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len) as u64;
    if family_count == 0 || artifact_count == 0 || family_count != artifact_count {
        blockers.push(format!(
            "{evidence}: family_count={family_count}, artifact_count={artifact_count}"
        ));
    }
    if family_count < 12 || artifact_count < 12 {
        blockers.push(format!(
            "{evidence}: family_count={family_count}, artifact_count={artifact_count}; release backend suites need at least 12 workload families"
        ));
    }
    if let Some(suite_blockers) = value.get("blockers").and_then(serde_json::Value::as_array) {
        for blocker in suite_blockers {
            blockers.push(format!(
                "{evidence}: suite blocker: {}",
                blocker.as_str().unwrap_or("<non-string blocker>")
            ));
        }
    }
    let Some(statuses) = value
        .get("artifact_statuses")
        .and_then(serde_json::Value::as_array)
    else {
        blockers.push(format!("{evidence}: missing artifact_statuses"));
        return;
    };
    if statuses.len() as u64 != artifact_count {
        blockers.push(format!(
            "{evidence}: artifact_statuses has {} entrie(s), artifacts has {artifact_count}",
            statuses.len()
        ));
    }
    for status in statuses {
        let path = status
            .get("path")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if status.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
            blockers.push(format!("{evidence}: suite artifact `{path}` is missing"));
        }
        if status
            .get("bytes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            blockers.push(format!("{evidence}: suite artifact `{path}` is empty"));
        }
        let read_error = status.get("read_error");
        if !read_error.is_some_and(serde_json::Value::is_null) {
            blockers.push(format!(
                "{evidence}: suite artifact `{path}` read_error={}",
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
            blockers.push(format!(
                "{evidence}: suite artifact `{path}` has no family_id"
            ));
        }
        if status
            .get("requested_case_id")
            .and_then(serde_json::Value::as_str)
            .is_none_or(str::is_empty)
        {
            blockers.push(format!(
                "{evidence}: suite artifact `{path}` has no requested_case_id"
            ));
        }
        for field in ["source_fingerprint", "host_cpu_model"] {
            if status
                .get(field)
                .and_then(serde_json::Value::as_str)
                .is_none_or(str::is_empty)
            {
                blockers.push(format!(
                    "{evidence}: suite artifact `{path}` has no `{field}` provenance"
                ));
            }
        }
        if status
            .get("case_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            blockers.push(format!("{evidence}: suite artifact `{path}` has no cases"));
        }
        if status
            .get("failed_count")
            .and_then(serde_json::Value::as_u64)
            != Some(0)
        {
            blockers.push(format!(
                "{evidence}: suite artifact `{path}` has nonzero or missing failed_count"
            ));
        }
        if status
            .get("nonmatching_case_backend_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1)
            != 0
        {
            blockers.push(format!(
                "{evidence}: suite artifact `{path}` has backend-mismatched case(s)"
            ));
        }
        if value.get("backend").and_then(serde_json::Value::as_str) == Some("cuda") {
            for field in ["gpu_model", "nvidia_driver_version", "nvidia_cuda_version"] {
                if status
                    .get(field)
                    .and_then(serde_json::Value::as_str)
                    .is_none_or(str::is_empty)
                {
                    blockers.push(format!(
                        "{evidence}: CUDA suite artifact `{path}` has no `{field}` provenance"
                    ));
                }
            }
            match status
                .get("gpu_memory_total_mib")
                .and_then(serde_json::Value::as_u64)
            {
                Some(mib) if mib >= 16 * 1024 => {}
                Some(mib) => blockers.push(format!(
                    "{evidence}: CUDA suite artifact `{path}` reports {mib} MiB GPU memory, below release floor 16384 MiB"
                )),
                None => blockers.push(format!(
                    "{evidence}: CUDA suite artifact `{path}` has no `gpu_memory_total_mib` provenance"
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
                (Some(major), Some(minor)) => blockers.push(format!(
                    "{evidence}: CUDA suite artifact `{path}` reports compute capability {major}.{minor}, below release floor 8.0"
                )),
                _ => blockers.push(format!(
                    "{evidence}: CUDA suite artifact `{path}` has no compute capability provenance"
                )),
            }
            if status
                .get("min_cuda_ptx_source_cache_entries")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                == 0
            {
                blockers.push(format!(
                    "{evidence}: CUDA suite artifact `{path}` has non-positive `min_cuda_ptx_source_cache_entries`"
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
                    blockers.push(format!(
                        "{evidence}: CUDA suite artifact `{path}` is missing `{field}`"
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
            blockers.push(format!(
                "{evidence}: suite artifact `{path}` has fewer than 30 wall_ns samples"
            ));
        }
        if status
            .get("min_baseline_wall_samples")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            < 30
        {
            blockers.push(format!(
                "{evidence}: suite artifact `{path}` has fewer than 30 baseline_wall_ns samples"
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
                blockers.push(format!(
                    "{evidence}: suite artifact `{path}` has non-positive `{field}`"
                ));
            }
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
            blockers.push(format!(
                "{evidence}: suite artifact `{path}` requires CPU-SOTA 100x proof but has zero passing 100x case(s)"
            ));
        }
        if let Some(status_blockers) = status.get("blockers").and_then(serde_json::Value::as_array)
        {
            for blocker in status_blockers {
                blockers.push(format!(
                    "{evidence}: suite artifact `{path}` blocker: {}",
                    blocker.as_str().unwrap_or("<non-string blocker>")
                ));
            }
        }
    }
}

fn inspect_cuda_ptx_pattern_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    if !benchmark_report_metric_p50_at_least(value, "ptx_corpus_kernels", 8.0) {
        blockers.push(format!(
            "{evidence}: CUDA PTX pattern benchmark must cover all 8 release corpus kernels"
        ));
    }
    if !benchmark_report_metric_p50_equals(value, "ptx_branch_labels", 0.0) {
        blockers.push(format!(
            "{evidence}: CUDA PTX pattern benchmark must emit zero ptx_branch_labels for predicated fast paths"
        ));
    }
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
        if !benchmark_report_has_positive_metric(value, metric) {
            blockers.push(format!(
                "{evidence}: CUDA PTX pattern benchmark has no positive p50 `{metric}`"
            ));
        }
    }
    for metric in [
        "ptx_vector_kernel_scalar_loads",
        "ptx_vector_kernel_scalar_stores",
        "ptx_vector_kernel_scalar_index_adds",
    ] {
        if !benchmark_report_metric_p50_equals(value, metric, 0.0) {
            blockers.push(format!(
                "{evidence}: CUDA PTX vector fusion benchmark must report p50 `{metric}` == 0"
            ));
        }
    }
}

fn inspect_megakernel_condition_cuda_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    inspect_named_cuda_benchmark_semantics(evidence, value, blockers);
    for metric in [
        "megakernel_condition_slots",
        "megakernel_condition_fired",
        "megakernel_condition_slots_per_sec_x1000",
    ] {
        if !benchmark_report_has_positive_metric(value, metric) {
            blockers.push(format!(
                "{evidence}: megakernel condition CUDA benchmark has no positive p50 `{metric}`"
            ));
        }
    }
}

fn inspect_megakernel_latency_cuda_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    inspect_named_cuda_benchmark_semantics(evidence, value, blockers);
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
        if !benchmark_report_has_positive_metric(value, metric) {
            blockers.push(format!(
                "{evidence}: megakernel latency CUDA benchmark has no positive p50 `{metric}`"
            ));
        }
    }
}

fn inspect_named_cuda_benchmark_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    if value
        .get("selected_backend")
        .and_then(serde_json::Value::as_str)
        != Some("cuda")
    {
        blockers.push(format!("{evidence}: selected_backend must be cuda"));
    }
    let Some(cases) = value.get("cases").and_then(serde_json::Value::as_array) else {
        blockers.push(format!("{evidence}: missing cases array"));
        return;
    };
    if cases.is_empty() {
        blockers.push(format!("{evidence}: cases array is empty"));
        return;
    }
    for case in cases {
        let id = case
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if case.get("backend_id").and_then(serde_json::Value::as_str) != Some("cuda") {
            blockers.push(format!("{evidence}: case `{id}` backend_id must be cuda"));
        }
        let metrics = case.get("metrics").and_then(serde_json::Value::as_object);
        let wall_samples = metrics
            .and_then(|metrics| metric_samples(metrics.get("wall_ns")))
            .unwrap_or(0);
        if wall_samples < 30 {
            blockers.push(format!(
                "{evidence}: case `{id}` has {wall_samples} wall_ns sample(s), needs at least 30"
            ));
        }
        require_benchmark_metric_percentiles(evidence, id, metrics, "wall_ns", blockers);
    }
}

fn benchmark_report_metric_p50_at_least(
    value: &serde_json::Value,
    metric: &str,
    minimum: f64,
) -> bool {
    value
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|cases| {
            cases.iter().any(|case| {
                case.get("metrics")
                    .and_then(serde_json::Value::as_object)
                    .and_then(|metrics| metric_p50(metrics.get(metric)))
                    .is_some_and(|value| value >= minimum)
            })
        })
}

fn benchmark_report_metric_p50_equals(
    value: &serde_json::Value,
    metric: &str,
    expected: f64,
) -> bool {
    value
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|cases| {
            cases.iter().any(|case| {
                case.get("metrics")
                    .and_then(serde_json::Value::as_object)
                    .and_then(|metrics| metric_p50(metrics.get(metric)))
                    .is_some_and(|value| (value - expected).abs() < f64::EPSILON)
            })
        })
}

fn inspect_backend_matrix_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if schema_version < 2 {
        blockers.push(format!(
            "{evidence}: schema_version is {schema_version}, expected >= 2"
        ));
    }
    if value.get("cuda_first").and_then(serde_json::Value::as_bool) != Some(true) {
        blockers.push(format!("{evidence}: cuda_first must be true"));
    }
    if value
        .get("wgpu_fallback_present")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        blockers.push(format!("{evidence}: wgpu_fallback_present must be true"));
    }
    if value
        .get("preferred_backend_gpu_only")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        blockers.push(format!(
            "{evidence}: preferred_backend_gpu_only must be true"
        ));
    }
    let preferred = value
        .get("preferred_backend_id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if !matches!(preferred, "cuda" | "wgpu") {
        blockers.push(format!(
            "{evidence}: preferred_backend_id `{preferred}` must be cuda or wgpu"
        ));
    }
    if value
        .get("gpu_probe")
        .and_then(|probe| probe.get("nvidia_smi_ok"))
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        blockers.push(format!("{evidence}: gpu_probe.nvidia_smi_ok must be true"));
    }
    let gpu_devices = value
        .get("gpu_probe")
        .and_then(|probe| probe.get("nvidia_smi_devices"))
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if gpu_devices == 0 {
        blockers.push(format!(
            "{evidence}: gpu_probe.nvidia_smi_devices must list at least one GPU"
        ));
    }
    let release_floor_device = value
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
        blockers.push(format!(
            "{evidence}: gpu_probe.nvidia_smi_device_details must include a CUDA GPU with >=16384 MiB VRAM and compute capability >=8.0"
        ));
    }
    for field in ["nvidia_driver_version", "nvidia_cuda_version"] {
        if value
            .get("gpu_probe")
            .and_then(|probe| probe.get(field))
            .and_then(serde_json::Value::as_str)
            .is_none_or(str::is_empty)
        {
            blockers.push(format!("{evidence}: gpu_probe.{field} must be recorded"));
        }
    }
    let backends = value
        .get("backends")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    for required in ["cuda", "wgpu"] {
        if !backends.iter().any(|backend| {
            backend.get("id").and_then(serde_json::Value::as_str) == Some(required)
                && backend
                    .get("dispatches")
                    .and_then(serde_json::Value::as_bool)
                    == Some(true)
                && backend
                    .get("acquire_ok")
                    .and_then(serde_json::Value::as_bool)
                    == Some(true)
        }) {
            blockers.push(format!(
                "{evidence}: backend `{required}` must dispatch and acquire successfully"
            ));
        }
    }
    for (field, minimum) in [
        ("cuda_feature_markers", 12usize),
        ("wgpu_feature_markers", 7usize),
    ] {
        let Some(markers) = value.get(field).and_then(serde_json::Value::as_array) else {
            blockers.push(format!("{evidence}: missing {field}"));
            continue;
        };
        if markers.len() < minimum {
            blockers.push(format!(
                "{evidence}: {field} has {} marker(s), needs at least {minimum}",
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
                blockers.push(format!(
                    "{evidence}: {field} is missing required marker `{required_id}`"
                ));
            }
        }
        for marker in markers {
            let id = marker
                .get("id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("<unknown>");
            if marker.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
                blockers.push(format!("{evidence}: {field} marker `{id}` does not exist"));
            }
            if marker
                .get("source_bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                == 0
            {
                blockers.push(format!("{evidence}: {field} marker `{id}` is empty"));
            }
            if marker
                .get("missing_tokens")
                .and_then(serde_json::Value::as_array)
                .is_none_or(|tokens| !tokens.is_empty())
            {
                blockers.push(format!(
                    "{evidence}: {field} marker `{id}` has missing tokens"
                ));
            }
            if marker
                .get("unresolved_markers")
                .and_then(serde_json::Value::as_array)
                .is_none_or(|markers| !markers.is_empty())
            {
                blockers.push(format!(
                    "{evidence}: {field} marker `{id}` has unresolved markers"
                ));
            }
        }
    }
    let Some(scan_errors) = value
        .get("hidden_fallback_scan_errors")
        .and_then(serde_json::Value::as_array)
    else {
        blockers.push(format!("{evidence}: missing hidden_fallback_scan_errors"));
        return;
    };
    if !scan_errors.is_empty() {
        blockers.push(format!(
            "{evidence}: reports {} hidden fallback scan error(s)",
            scan_errors.len()
        ));
    }
    let Some(findings) = value
        .get("hidden_fallback_findings")
        .and_then(serde_json::Value::as_array)
    else {
        blockers.push(format!("{evidence}: missing hidden_fallback_findings"));
        return;
    };
    if !findings.is_empty() {
        blockers.push(format!(
            "{evidence}: reports {} hidden fallback finding(s)",
            findings.len()
        ));
    }
}

fn inspect_release_workload_matrix_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let required = value
        .get("required_closed_families")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if required < 12 {
        blockers.push(format!(
            "{evidence}: required_closed_families is {required}; needs at least 12"
        ));
    }
    let matched = value
        .get("matched_required_families")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if matched < 12 {
        blockers.push(format!(
            "{evidence}: matched_required_families is {matched}; needs at least 12"
        ));
    }
    if matched < required {
        blockers.push(format!(
            "{evidence}: matched_required_families {matched} is below required_closed_families {required}"
        ));
    }
    let release_cases = value
        .get("release_suite_case_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if release_cases < matched {
        blockers.push(format!(
            "{evidence}: release_suite_case_count {release_cases} is below matched_required_families {matched}"
        ));
    }
    let family_count = value
        .get("cpu_sota_100x_family_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if family_count < 10 {
        blockers.push(format!(
            "{evidence}: cpu_sota_100x_family_count is {family_count}; needs at least 10"
        ));
    }
    let required_hundred_x = value
        .get("required_cpu_sota_100x_families")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if required_hundred_x < 10 {
        blockers.push(format!(
            "{evidence}: required_cpu_sota_100x_families lists {required_hundred_x} family/families; needs at least 10 release 100x families"
        ));
    }
    let missing_required_hundred_x = value
        .get("missing_required_cpu_sota_100x_families")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_required_hundred_x != 0 {
        blockers.push(format!(
            "{evidence}: missing_required_cpu_sota_100x_families reports {missing_required_hundred_x} missing required family/families"
        ));
    }
    let case_count = value
        .get("cpu_sota_100x_contract_cases")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if case_count < 10 {
        blockers.push(format!(
            "{evidence}: cpu_sota_100x_contract_cases lists {case_count} active case id(s); needs at least 10"
        ));
    }
    let Some(families) = value.get("families").and_then(serde_json::Value::as_array) else {
        blockers.push(format!("{evidence}: missing workload families array"));
        return;
    };
    let mut required_family_count = 0usize;
    let mut covered_family_count = 0usize;
    let mut artifacts = BTreeSet::new();
    let mut workload_numbers = BTreeSet::new();
    for family in families {
        let id = family
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if family.get("required").and_then(serde_json::Value::as_bool) != Some(true) {
            continue;
        }
        required_family_count += 1;
        let matched_cases = family
            .get("matched_cases")
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len);
        if matched_cases == 0 {
            blockers.push(format!(
                "{evidence}: required workload family `{id}` has no matched_cases"
            ));
        } else {
            covered_family_count += 1;
        }
        let dispatch_policy = family
            .get("dispatch_policy")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if dispatch_policy.is_empty() {
            blockers.push(format!(
                "{evidence}: required workload family `{id}` has no dispatch_policy"
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
            blockers.push(format!(
                "{evidence}: required workload family `{id}` must list release BENCH_TARGETS.toml target ids"
            ));
        }
        if id == "megakernel-queued-batches" && dispatch_policy != "megakernel" {
            blockers.push(format!(
                "{evidence}: required workload family `{id}` must use megakernel dispatch policy, found `{dispatch_policy}`"
            ));
        }
        if dispatch_policy != "megakernel" {
            let justification = family
                .get("non_megakernel_justification")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            if justification.len() < 48 {
                blockers.push(format!(
                    "{evidence}: required workload family `{id}` uses non-megakernel dispatch policy `{dispatch_policy}` without a concrete architectural or measured justification"
                ));
            }
        }
        let cpu_sota_contracts = family
            .get("cpu_sota_contracts")
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len);
        if cpu_sota_contracts == 0 {
            blockers.push(format!(
                "{evidence}: required workload family `{id}` has no CPU-SOTA baseline contract"
            ));
        }
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
                blockers.push(format!(
                    "{evidence}: required workload family `{id}` declares a 100x contract but lists no cpu_sota_100x_cases"
                ));
            }
        }
        let workload_number = family
            .get("release_plan_workload")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if workload_number == 0 || !workload_numbers.insert(workload_number) {
            blockers.push(format!(
                "{evidence}: required workload family `{id}` has invalid or duplicate release_plan_workload `{workload_number}`"
            ));
        }
        let artifact = family
            .get("evidence_artifact")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if artifact.is_empty() {
            blockers.push(format!(
                "{evidence}: required workload family `{id}` has no evidence_artifact"
            ));
        } else {
            if !artifacts.insert(artifact) {
                blockers.push(format!(
                    "{evidence}: required workload family `{id}` reuses evidence artifact `{artifact}`"
                ));
            }
            if !artifact.starts_with("release/evidence/benchmarks/workload-") {
                blockers.push(format!(
                    "{evidence}: required workload family `{id}` artifact `{artifact}` is not a workload benchmark artifact"
                ));
            }
        }
        let command = family
            .get("benchmark_command")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if !command.contains("cargo_full") || !command.contains(artifact) {
            blockers.push(format!(
                "{evidence}: required workload family `{id}` benchmark_command must use cargo_full and its evidence_artifact"
            ));
        }
        if family
            .get("fair_cpu_sota_baseline_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            blockers.push(format!(
                "{evidence}: required workload family `{id}` has no fair CPU-SOTA baseline crate bound to CUDA"
            ));
        }
        if family
            .get("cpu_sota_baseline_names")
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len)
            == 0
        {
            blockers.push(format!(
                "{evidence}: required workload family `{id}` has no named CPU-SOTA baseline provenance"
            ));
        }
        if family
            .get("reproducible_cuda_command")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        {
            blockers.push(format!(
                "{evidence}: required workload family `{id}` does not declare a reproducible CUDA benchmark command"
            ));
        }
    }
    if required_family_count < 12 {
        blockers.push(format!(
            "{evidence}: declares {required_family_count} required workload families; needs at least 12"
        ));
    }
    if covered_family_count < 12 {
        blockers.push(format!(
            "{evidence}: covers {covered_family_count} required workload families; needs at least 12"
        ));
    }
}

fn inspect_workload_benchmark_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    if value
        .get("selected_backend")
        .and_then(serde_json::Value::as_str)
        != Some("cuda")
    {
        blockers.push(format!("{evidence}: selected_backend must be cuda"));
    }
    inspect_workload_benchmark_provenance(evidence, value, blockers);
    let Some(cases) = value.get("cases").and_then(serde_json::Value::as_array) else {
        blockers.push(format!("{evidence}: missing cases array"));
        return;
    };
    if cases.is_empty() {
        blockers.push(format!("{evidence}: cases array is empty"));
        return;
    }
    for case in cases {
        let id = case
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if case.get("backend_id").and_then(serde_json::Value::as_str) != Some("cuda") {
            blockers.push(format!("{evidence}: case `{id}` backend_id must be cuda"));
        }
        if case.get("contract").is_none_or(serde_json::Value::is_null) {
            blockers.push(format!("{evidence}: case `{id}` is missing a contract"));
        }
        if !case
            .get("performance")
            .and_then(|performance| performance.get("contract_passed"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            blockers.push(format!(
                "{evidence}: case `{id}` must pass its performance contract"
            ));
        }
        let metrics = case.get("metrics").and_then(serde_json::Value::as_object);
        let wall = metrics.and_then(active_gpu_metric_p50);
        let baseline = metrics.and_then(|metrics| metric_p50(metrics.get("baseline_wall_ns")));
        let wall_samples = metrics
            .and_then(|metrics| metric_samples(metrics.get("wall_ns")))
            .unwrap_or(0);
        if wall_samples < 30 {
            blockers.push(format!(
                "{evidence}: case `{id}` has {wall_samples} wall_ns sample(s), needs at least 30"
            ));
        }
        let baseline_wall_samples = metrics
            .and_then(|metrics| metric_samples(metrics.get("baseline_wall_ns")))
            .unwrap_or(0);
        if baseline_wall_samples < 30 {
            blockers.push(format!(
                "{evidence}: case `{id}` has {baseline_wall_samples} baseline_wall_ns sample(s), needs at least 30"
            ));
        }
        require_benchmark_metric_percentiles(evidence, id, metrics, "wall_ns", blockers);
        require_benchmark_metric_percentiles(evidence, id, metrics, "baseline_wall_ns", blockers);
        match (wall, baseline) {
            (Some(wall), Some(baseline)) if wall > 0.0 && baseline > wall => {}
            (Some(wall), Some(baseline)) => blockers.push(format!(
                "{evidence}: case `{id}` did not beat p50 CPU/SOTA baseline: wall={wall:.2}, baseline={baseline:.2}"
            )),
            _ => blockers.push(format!(
                "{evidence}: case `{id}` must include p50 wall_ns and baseline_wall_ns"
            )),
        }
        let speedup = case
            .get("performance")
            .and_then(|performance| performance.get("speedup_x"))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        if speedup <= 1.0 {
            blockers.push(format!(
                "{evidence}: case `{id}` speedup_x must be greater than 1.0"
            ));
        }
    }
}

fn inspect_workload_benchmark_provenance(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    if !has_nonempty_string_any(
        value,
        &[
            "source_fingerprint",
            "source_revision",
            "source_artifact_fingerprint",
            "commit_fingerprint",
        ],
    ) && !value
        .get("source_artifacts")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|items| !items.is_empty())
        && !value
            .get("git")
            .is_some_and(|git| has_nonempty_string_any(git, &["commit"]))
    {
        blockers.push(format!(
            "{evidence}: benchmark report must include source fingerprint or source artifact provenance"
        ));
    }
    let environment = value.get("environment");
    if !environment.is_some_and(|environment| {
        has_nonempty_string_any(
            environment,
            &["host_cpu_model", "cpu_model", "host_cpu", "processor_model"],
        )
    }) {
        blockers.push(format!(
            "{evidence}: benchmark environment must include host CPU model provenance"
        ));
    }
    let summary = value.get("summary");
    if !summary.is_some_and(|summary| summary.get("cache_hit_rate").is_some()) {
        blockers.push(format!(
            "{evidence}: benchmark summary must include cache_hit_rate, even when null"
        ));
    }
    let Some(cases) = value.get("cases").and_then(serde_json::Value::as_array) else {
        return;
    };
    for case in cases {
        let id = case
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if !has_nonempty_string_any(
            case,
            &[
                "dataset_fingerprint",
                "corpus_fingerprint",
                "input_fingerprint",
                "workload_fingerprint",
            ],
        ) && !case.get("contract").is_some_and(|contract| {
            has_nonempty_string_any(
                contract,
                &[
                    "dataset_fingerprint",
                    "corpus_fingerprint",
                    "input_fingerprint",
                    "workload_fingerprint",
                ],
            )
        }) {
            blockers.push(format!(
                "{evidence}: case `{id}` must include dataset/corpus/input fingerprint provenance"
            ));
        }
        if !case
            .get("correctness")
            .is_some_and(|correctness| !correctness.is_null())
            && !case.get("oracle").is_some_and(|oracle| !oracle.is_null())
        {
            blockers.push(format!(
                "{evidence}: case `{id}` must include correctness oracle evidence"
            ));
        }
        let metrics = case.get("metrics").and_then(serde_json::Value::as_object);
        for (label, metric_names) in [
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
                blockers.push(format!(
                    "{evidence}: case `{id}` must include {label} metric"
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
            blockers.push(format!(
                "{evidence}: case `{id}` must list optimization passes applied"
            ));
        }
    }
}

fn has_nonempty_string_any(value: &serde_json::Value, fields: &[&str]) -> bool {
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

fn inspect_weir_readme_contract_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    inspect_schema_version_at_least(evidence, value, 2, blockers);
    if value.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
        blockers.push(format!("{evidence}: README.md must exist"));
    }
    if value
        .get("source_bytes")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0)
        == 0
    {
        blockers.push(format!("{evidence}: README.md is empty"));
    }
    if value
        .get("missing_tokens")
        .and_then(serde_json::Value::as_array)
        .is_none_or(|tokens| !tokens.is_empty())
    {
        blockers.push(format!(
            "{evidence}: missing_tokens must exist and be empty"
        ));
    }
    if value
        .get("example_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0)
        == 0
    {
        blockers.push(format!(
            "{evidence}: README.md must contain at least one Rust or TOML example"
        ));
    }
    if value
        .get("blockers")
        .and_then(serde_json::Value::as_array)
        .is_none_or(|items| !items.is_empty())
    {
        blockers.push(format!("{evidence}: blockers must exist and be empty"));
    }
}

fn inspect_schema_version_at_least(
    evidence: &str,
    value: &serde_json::Value,
    minimum: u64,
    blockers: &mut Vec<String>,
) {
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if schema_version < minimum {
        blockers.push(format!(
            "{evidence}: schema_version is {schema_version}, expected >= {minimum}"
        ));
    }
}

fn inspect_c_parser_corpus_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let total = value
        .get("total_files")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let parsed = value
        .get("parsed_files")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let failed = value
        .get("failed_files")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    let source_bytes = value
        .get("total_source_bytes")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let ast_bytes = value
        .get("total_ast_bytes")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let vast_bytes = value
        .get("total_vast_bytes")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let semantic_graph_bytes = value
        .get("total_semantic_graph_bytes")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if total < 250 {
        blockers.push(format!(
            "{evidence}: total_files {total} is below Linux subsystem floor 250"
        ));
    }
    if parsed != total || failed != 0 {
        blockers.push(format!(
            "{evidence}: parsed_files={parsed}, total_files={total}, failed_files={failed}; full corpus parse required"
        ));
    }
    if source_bytes < 4 * 1024 * 1024 {
        blockers.push(format!(
            "{evidence}: total_source_bytes {source_bytes} is below Linux subsystem floor 4194304"
        ));
    }
    if value
        .get("linux_subsystem_candidate")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        blockers.push(format!(
            "{evidence}: linux_subsystem_candidate must be true"
        ));
    }
    if value
        .get("corpus_root_canonical")
        .and_then(serde_json::Value::as_str)
        .is_none_or(str::is_empty)
    {
        blockers.push(format!("{evidence}: missing corpus_root_canonical"));
    }
    inspect_corpus_fingerprint(evidence, value, blockers);
    inspect_linux_subsystem_provenance(evidence, value, blockers);
    inspect_c_parser_collection_provenance(evidence, value, blockers);
    for field in ["include_dirs", "macros"] {
        if value
            .get(field)
            .and_then(serde_json::Value::as_array)
            .is_none_or(Vec::is_empty)
        {
            blockers.push(format!(
                "{evidence}: reproducibility field `{field}` must be non-empty"
            ));
        }
    }
    if ast_bytes == 0 || vast_bytes == 0 || semantic_graph_bytes == 0 {
        blockers.push(format!(
            "{evidence}: AST/VAST/semantic section bytes are incomplete: ast={ast_bytes}, vast={vast_bytes}, semantic={semantic_graph_bytes}"
        ));
    }
    let file_entries = value
        .get("files")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len) as u64;
    if file_entries != parsed {
        blockers.push(format!(
            "{evidence}: files array has {file_entries} entries, parsed_files is {parsed}"
        ));
    }
    let failure_entries = value
        .get("failures")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len) as u64;
    if failure_entries != failed {
        blockers.push(format!(
            "{evidence}: failures array has {failure_entries} entries, failed_files is {failed}"
        ));
    }
}

fn inspect_c_parser_manifest_semantics(
    evidence: &str,
    path: &Path,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let file_count = value
        .get("file_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let source_bytes = value
        .get("total_source_bytes")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let entries = value
        .get("files")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len) as u64;
    if file_count < 250 {
        blockers.push(format!(
            "{evidence}: file_count {file_count} is below Linux subsystem floor 250"
        ));
    }
    if source_bytes < 4 * 1024 * 1024 {
        blockers.push(format!(
            "{evidence}: total_source_bytes {source_bytes} is below Linux subsystem floor 4194304"
        ));
    }
    if value
        .get("linux_subsystem_candidate")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        blockers.push(format!(
            "{evidence}: linux_subsystem_candidate must be true"
        ));
    }
    if value
        .get("corpus_root_canonical")
        .and_then(serde_json::Value::as_str)
        .is_none_or(str::is_empty)
    {
        blockers.push(format!("{evidence}: missing corpus_root_canonical"));
    }
    inspect_corpus_fingerprint(evidence, value, blockers);
    inspect_linux_subsystem_provenance(evidence, value, blockers);
    inspect_c_parser_collection_provenance(evidence, value, blockers);
    for field in ["include_dirs", "macros"] {
        if value
            .get(field)
            .and_then(serde_json::Value::as_array)
            .is_none_or(Vec::is_empty)
        {
            blockers.push(format!(
                "{evidence}: reproducibility field `{field}` must be non-empty"
            ));
        }
    }
    if entries != file_count {
        blockers.push(format!(
            "{evidence}: files array has {entries} entries, file_count is {file_count}"
        ));
    }
    if let Some(parse_report) = read_sibling_json(path, "c-parser-linux-subsystem.json", blockers) {
        for (manifest_field, parse_field) in [
            ("file_count", "total_files"),
            ("total_source_bytes", "total_source_bytes"),
            ("linux_subsystem_candidate", "linux_subsystem_candidate"),
            ("corpus_root_canonical", "corpus_root_canonical"),
            ("linux_root", "linux_root"),
            ("linux_subsystem", "linux_subsystem"),
            ("linux_subsystem_depth", "linux_subsystem_depth"),
            ("linux_kbuild_file", "linux_kbuild_file"),
            ("linux_kbuild_file_in_corpus", "linux_kbuild_file_in_corpus"),
            ("corpus_fingerprint", "corpus_fingerprint"),
            ("source_collection_mode", "source_collection_mode"),
            ("visited_dir_count", "visited_dir_count"),
            ("include_dirs", "include_dirs"),
            ("macros", "macros"),
        ] {
            let manifest_value = value.get(manifest_field);
            let parse_value = parse_report.get(parse_field);
            if manifest_value != parse_value {
                blockers.push(format!(
                    "{evidence}: `{manifest_field}` does not match c-parser-linux-subsystem.json `{parse_field}`"
                ));
            }
        }
        let parsed_files = parse_report
            .get("parsed_files")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if file_count != parsed_files {
            blockers.push(format!(
                "{evidence}: file_count {file_count} does not match c-parser-linux-subsystem.json parsed_files {parsed_files}"
            ));
        }
        let parse_entries = parse_report
            .get("files")
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len) as u64;
        if entries != parse_entries {
            blockers.push(format!(
                "{evidence}: files array has {entries} entries but c-parser-linux-subsystem.json has {parse_entries}"
            ));
        }
    }
    if let Some(files) = value.get("files").and_then(serde_json::Value::as_array) {
        for file in files {
            let path = file
                .get("path")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("<unknown>");
            if file
                .get("source_bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                == 0
            {
                blockers.push(format!(
                    "{evidence}: manifest file `{path}` has zero source_bytes"
                ));
            }
            if file.get("parsed").and_then(serde_json::Value::as_bool) != Some(true) {
                blockers.push(format!(
                    "{evidence}: manifest file `{path}` was not parsed successfully"
                ));
                continue;
            }
            for field in [
                "object_bytes",
                "ast_bytes",
                "vast_bytes",
                "semantic_graph_bytes",
            ] {
                if file
                    .get(field)
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
                    == 0
                {
                    blockers.push(format!(
                        "{evidence}: manifest file `{path}` has zero `{field}`"
                    ));
                }
            }
            if file
                .get("wall_ns")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                == 0
            {
                blockers.push(format!(
                    "{evidence}: manifest file `{path}` has zero wall_ns"
                ));
            }
        }
    }
}

fn inspect_linux_subsystem_provenance(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    for field in ["linux_root", "linux_subsystem", "linux_kbuild_file"] {
        if value
            .get(field)
            .and_then(serde_json::Value::as_str)
            .is_none_or(str::is_empty)
        {
            blockers.push(format!(
                "{evidence}: missing Linux provenance field `{field}`"
            ));
        }
    }
    if value
        .get("linux_kbuild_file_in_corpus")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        blockers.push(format!(
            "{evidence}: linux_kbuild_file_in_corpus must be true"
        ));
    }
    let linux_subsystem = value
        .get("linux_subsystem")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if !matches!(
        linux_subsystem,
        "kernel" | "fs" | "mm" | "net" | "drivers" | "lib"
    ) {
        blockers.push(format!(
            "{evidence}: unsupported linux_subsystem `{linux_subsystem}`"
        ));
    }
    let linux_depth = value
        .get("linux_subsystem_depth")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if linux_depth == 0 {
        blockers.push(format!(
            "{evidence}: linux_subsystem_depth must be greater than zero"
        ));
    }
}

fn inspect_c_parser_collection_provenance(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    if value
        .get("source_collection_mode")
        .and_then(serde_json::Value::as_str)
        != Some("recursive_all_c_files")
    {
        blockers.push(format!(
            "{evidence}: source_collection_mode must be recursive_all_c_files"
        ));
    }
    let visited_dir_count = value
        .get("visited_dir_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if visited_dir_count == 0 {
        blockers.push(format!(
            "{evidence}: visited_dir_count must prove recursive corpus traversal"
        ));
    }
}

fn inspect_corpus_fingerprint(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    if value
        .get("corpus_fingerprint")
        .and_then(serde_json::Value::as_str)
        .is_none_or(|fingerprint| !fingerprint.starts_with("fnv64:"))
    {
        blockers.push(format!("{evidence}: missing stable corpus_fingerprint"));
    }
}

fn inspect_c_parser_diagnostics_semantics(
    evidence: &str,
    path: &Path,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let failed = value
        .get("failed_files")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    let failures = value
        .get("failures")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len) as u64;
    if failed != 0 || failures != 0 {
        blockers.push(format!(
            "{evidence}: parser diagnostics still report failed_files={failed}, failure entries={failures}"
        ));
    }
    if let Some(parse_report) = read_sibling_json(path, "c-parser-linux-subsystem.json", blockers) {
        let parse_failed = parse_report
            .get("failed_files")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(u64::MAX);
        if failed != parse_failed {
            blockers.push(format!(
                "{evidence}: failed_files {failed} does not match c-parser-linux-subsystem.json failed_files {parse_failed}"
            ));
        }
        let parse_failures = parse_report
            .get("failures")
            .and_then(serde_json::Value::as_array)
            .map_or(usize::MAX, Vec::len) as u64;
        if failures != parse_failures {
            blockers.push(format!(
                "{evidence}: failure entries {failures} do not match c-parser-linux-subsystem.json failure entries {parse_failures}"
            ));
        }
    }
}

fn inspect_c_parser_throughput_semantics(
    evidence: &str,
    path: &Path,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let total = value
        .get("total_files")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let parsed = value
        .get("parsed_files")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if value
        .get("linux_subsystem_candidate")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        blockers.push(format!(
            "{evidence}: linux_subsystem_candidate must be true"
        ));
    }
    if value
        .get("corpus_root_canonical")
        .and_then(serde_json::Value::as_str)
        .is_none_or(str::is_empty)
    {
        blockers.push(format!("{evidence}: missing corpus_root_canonical"));
    }
    inspect_corpus_fingerprint(evidence, value, blockers);
    inspect_linux_subsystem_provenance(evidence, value, blockers);
    inspect_c_parser_collection_provenance(evidence, value, blockers);
    for field in ["include_dirs", "macros"] {
        if value
            .get(field)
            .and_then(serde_json::Value::as_array)
            .is_none_or(|items| items.is_empty())
        {
            blockers.push(format!("{evidence}: `{field}` must be non-empty"));
        }
    }
    let source_bytes = value
        .get("total_source_bytes")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let wall_ns = value
        .get("wall_ns")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let files_per_second = value
        .get("files_per_second_x1000")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let mib_per_second = value
        .get("mib_per_second_x1000")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if total < 250 || parsed != total {
        blockers.push(format!(
            "{evidence}: throughput covers parsed_files={parsed}, total_files={total}; full Linux subsystem throughput requires at least 250 parsed files"
        ));
    }
    if source_bytes < 4 * 1024 * 1024 {
        blockers.push(format!(
            "{evidence}: total_source_bytes {source_bytes} is below Linux subsystem floor 4194304"
        ));
    }
    if wall_ns == 0 || files_per_second == 0 || mib_per_second == 0 {
        blockers.push(format!(
            "{evidence}: throughput rates are incomplete: wall_ns={wall_ns}, files_per_second_x1000={files_per_second}, mib_per_second_x1000={mib_per_second}"
        ));
    }
    if let Some(parse_report) = read_sibling_json(path, "c-parser-linux-subsystem.json", blockers) {
        for field in [
            "total_files",
            "parsed_files",
            "total_source_bytes",
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
            if value.get(field) != parse_report.get(field) {
                blockers.push(format!(
                    "{evidence}: throughput field `{field}` does not match c-parser-linux-subsystem.json"
                ));
            }
        }
    }
}

fn read_sibling_json(
    path: &Path,
    sibling: &str,
    blockers: &mut Vec<String>,
) -> Option<serde_json::Value> {
    let sibling_path = path.parent()?.join(sibling);
    let text = match read_text_bounded(&sibling_path) {
        Ok(text) => text,
        Err(error) => {
            blockers.push(format!(
                "{}: failed to read sibling artifact `{}`: {error}",
                path.display(),
                sibling_path.display()
            ));
            return None;
        }
    };
    match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(value) => Some(value),
        Err(error) => {
            blockers.push(format!(
                "{}: sibling artifact `{}` is invalid JSON: {error}",
                path.display(),
                sibling_path.display()
            ));
            None
        }
    }
}

fn is_parser_contract_evidence(evidence: &str) -> bool {
    [
        "vyre-frontend-c-contracts.json",
        "vyrec-cli-contracts.json",
        "weir-contracts.json",
        "security-analysis-consumer-contracts.json",
        "security-grammar-gen-contracts.json",
    ]
    .iter()
    .any(|suffix| evidence.ends_with(suffix))
}

fn inspect_distributed_parser_map_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let Some(components) = value
        .get("components")
        .and_then(serde_json::Value::as_array)
    else {
        blockers.push(format!("{evidence}: missing components array"));
        return;
    };
    for required in [
        "vyre-frontend-c",
        "vyrec",
        "weir",
        "security-analysis-consumer",
        "security-grammar-gen",
    ] {
        if !components.iter().any(|component| {
            component.get("id").and_then(serde_json::Value::as_str) == Some(required)
                && component.get("exists").and_then(serde_json::Value::as_bool) == Some(true)
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
            blockers.push(format!(
                "{evidence}: missing complete parser ownership component `{required}`"
            ));
        }
    }
}

fn inspect_parser_contract_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let component_id = value
        .get("component_id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let expected = if evidence.ends_with("vyrec-cli-contracts.json") {
        "vyrec"
    } else {
        evidence
            .rsplit('/')
            .next()
            .and_then(|file| file.strip_suffix("-contracts.json"))
            .unwrap_or("")
    };
    if component_id != expected {
        blockers.push(format!(
            "{evidence}: component_id `{component_id}` does not match expected `{expected}`"
        ));
    }
    if value
        .get("role")
        .and_then(serde_json::Value::as_str)
        .is_none_or(|role| role.is_empty())
    {
        blockers.push(format!("{evidence}: parser contract role is empty"));
    }
    if value
        .get("root")
        .and_then(serde_json::Value::as_str)
        .is_none_or(|root| root.is_empty())
    {
        blockers.push(format!("{evidence}: parser contract root is empty"));
    }
    let required_terms = value
        .get("required_terms")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if required_terms == 0 {
        blockers.push(format!("{evidence}: parser contract has no required_terms"));
    }
    let missing_terms = value
        .get("missing_terms")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_terms != 0 {
        blockers.push(format!(
            "{evidence}: parser contract reports {missing_terms} missing term(s)"
        ));
    }
    let required_contract_topics = value
        .get("required_contract_topics")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if required_contract_topics == 0 {
        blockers.push(format!(
            "{evidence}: parser contract has no required_contract_topics"
        ));
    }
    let missing_contract_topics = value
        .get("missing_contract_topics")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_contract_topics != 0 {
        blockers.push(format!(
            "{evidence}: parser contract reports {missing_contract_topics} missing contract topic(s)"
        ));
    }
    let required_test_categories = value
        .get("required_test_categories")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if required_test_categories == 0 {
        blockers.push(format!(
            "{evidence}: parser contract has no required_test_categories"
        ));
    }
    let missing_test_categories = value
        .get("missing_test_categories")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_test_categories != 0 {
        blockers.push(format!(
            "{evidence}: parser contract reports {missing_test_categories} missing test categor(ies)"
        ));
    }
    let required_evidence_trees = value
        .get("required_evidence_trees")
        .and_then(serde_json::Value::as_array);
    if required_evidence_trees.is_none_or(|trees| trees.len() < 3) {
        blockers.push(format!(
            "{evidence}: parser contract must list tests, benches, and fuzz evidence trees"
        ));
    }
    if let Some(trees) = required_evidence_trees {
        for tree in trees {
            let tree_name = tree
                .get("tree")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("<unknown>");
            if tree.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
                blockers.push(format!(
                    "{evidence}: parser contract evidence tree `{tree_name}` does not exist"
                ));
            }
            let source_bytes = tree
                .get("source_bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            if source_bytes == 0 {
                blockers.push(format!(
                    "{evidence}: parser contract evidence tree `{tree_name}` has zero source bytes"
                ));
            }
            let unreadable = tree
                .get("unreadable_file_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(u64::MAX);
            if unreadable != 0 {
                blockers.push(format!(
                    "{evidence}: parser contract evidence tree `{tree_name}` has {unreadable} unreadable source file(s)"
                ));
            }
        }
    }
    let unresolved_ownership_markers = value
        .get("unresolved_ownership_markers")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if unresolved_ownership_markers != 0 {
        blockers.push(format!(
            "{evidence}: parser contract reports {unresolved_ownership_markers} unresolved ownership marker(s)"
        ));
    }
    let Some(files) = value
        .get("required_files")
        .and_then(serde_json::Value::as_array)
    else {
        blockers.push(format!(
            "{evidence}: parser contract missing required_files"
        ));
        return;
    };
    if files.is_empty() {
        blockers.push(format!(
            "{evidence}: parser contract required_files is empty"
        ));
    }
    for file in files {
        let path = file
            .get("path")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if file.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
            blockers.push(format!("{evidence}: required file `{path}` does not exist"));
        }
        if file
            .get("source_bytes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            blockers.push(format!("{evidence}: required file `{path}` is empty"));
        }
        let read_error = file.get("read_error");
        if !read_error.is_some_and(serde_json::Value::is_null) {
            blockers.push(format!(
                "{evidence}: required file `{path}` read_error={}",
                read_error
                    .map(serde_json::Value::to_string)
                    .unwrap_or_else(|| "<missing>".to_string())
            ));
        }
    }
}

fn inspect_cpu_100x_benchmark_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    if value
        .get("selected_backend")
        .and_then(serde_json::Value::as_str)
        != Some("cuda")
    {
        blockers.push(format!("{evidence}: selected_backend must be cuda"));
    }
    let Some(cases) = value.get("cases").and_then(serde_json::Value::as_array) else {
        blockers.push(format!("{evidence}: missing cases array"));
        return;
    };
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
        if !cases
            .iter()
            .any(|case| case.get("id").and_then(serde_json::Value::as_str) == Some(required_case))
        {
            blockers.push(format!(
                "{evidence}: missing required CPU-SOTA 100x proof case `{required_case}`"
            ));
        }
    }
    for case in cases {
        let id = case
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if case.get("backend_id").and_then(serde_json::Value::as_str) != Some("cuda") {
            blockers.push(format!("{evidence}: case `{id}` backend_id must be cuda"));
        }
        if !case
            .get("performance")
            .and_then(|performance| performance.get("contract_passed"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            blockers.push(format!(
                "{evidence}: case `{id}` must pass its performance contract"
            ));
        }
        let metrics = case.get("metrics").and_then(serde_json::Value::as_object);
        let wall = metrics.and_then(active_gpu_metric_p50);
        let baseline = metrics.and_then(|metrics| metric_p50(metrics.get("baseline_wall_ns")));
        let wall_samples = metrics
            .and_then(|metrics| metric_samples(metrics.get("wall_ns")))
            .unwrap_or(0);
        if wall_samples < 30 {
            blockers.push(format!(
                "{evidence}: case `{id}` has {wall_samples} wall_ns sample(s), needs at least 30"
            ));
        }
        let baseline_wall_samples = metrics
            .and_then(|metrics| metric_samples(metrics.get("baseline_wall_ns")))
            .unwrap_or(0);
        if baseline_wall_samples < 30 {
            blockers.push(format!(
                "{evidence}: case `{id}` has {baseline_wall_samples} baseline_wall_ns sample(s), needs at least 30"
            ));
        }
        require_benchmark_metric_percentiles(evidence, id, metrics, "wall_ns", blockers);
        require_benchmark_metric_percentiles(evidence, id, metrics, "baseline_wall_ns", blockers);
        match (wall, baseline) {
            (Some(wall), Some(baseline)) if wall > 0.0 && baseline / wall >= 100.0 => {}
            (Some(wall), Some(baseline)) if wall > 0.0 => blockers.push(format!(
                "{evidence}: case `{id}` end-to-end p50 speedup is {:.2}x, needs 100.00x",
                baseline / wall
            )),
            _ => blockers.push(format!(
                "{evidence}: case `{id}` must include p50 wall_ns and baseline_wall_ns"
            )),
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

fn metric_p50(value: Option<&serde_json::Value>) -> Option<f64> {
    metric_percentile(value, "p50")
}

fn active_gpu_metric_p50(metrics: &serde_json::Map<String, serde_json::Value>) -> Option<f64> {
    metric_p50(metrics.get("dispatch_ns"))
        .or_else(|| metric_p50(metrics.get("kernel_execute_ns")))
        .or_else(|| metric_p50(metrics.get("wall_ns")))
}

fn metric_percentile(value: Option<&serde_json::Value>, percentile: &str) -> Option<f64> {
    value
        .and_then(|value| value.get(percentile))
        .and_then(serde_json::Value::as_f64)
        .or_else(|| {
            value
                .and_then(|value| value.get(percentile))
                .and_then(serde_json::Value::as_u64)
                .map(|value| value as f64)
        })
}

fn metric_samples(value: Option<&serde_json::Value>) -> Option<u64> {
    value?.get("samples").and_then(serde_json::Value::as_u64)
}

fn require_benchmark_metric_percentiles(
    evidence: &str,
    case_id: &str,
    metrics: Option<&serde_json::Map<String, serde_json::Value>>,
    metric_name: &str,
    blockers: &mut Vec<String>,
) {
    for percentile in ["p50", "p95", "p99"] {
        let value =
            metrics.and_then(|metrics| metric_percentile(metrics.get(metric_name), percentile));
        if !value.is_some_and(|value| value > 0.0) {
            blockers.push(format!(
                "{evidence}: case `{case_id}` must include positive {percentile} {metric_name}"
            ));
        }
    }
}

fn inspect_version_matrix_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    if value
        .get("requested_vyre_release")
        .and_then(serde_json::Value::as_str)
        != Some("0.4.2")
    {
        blockers.push(format!(
            "{evidence}: requested_vyre_release must be `0.4.2`"
        ));
    }
    if value
        .get("requested_weir_release")
        .and_then(serde_json::Value::as_str)
        != Some("0.1.0")
    {
        blockers.push(format!(
            "{evidence}: requested_weir_release must be `0.1.0`"
        ));
    }
    if value
        .get("release_doc_tag_findings")
        .and_then(serde_json::Value::as_array)
        .is_none_or(|findings| !findings.is_empty())
    {
        blockers.push(format!(
            "{evidence}: release_doc_tag_findings must exist and be empty"
        ));
    }
    if value
        .get("release_note_token_findings")
        .and_then(serde_json::Value::as_array)
        .is_none_or(|findings| !findings.is_empty())
    {
        blockers.push(format!(
            "{evidence}: release_note_token_findings must exist and be empty"
        ));
    }
    if value
        .get("missing_required_release_packages")
        .and_then(serde_json::Value::as_array)
        .is_none_or(|packages| !packages.is_empty())
    {
        blockers.push(format!(
            "{evidence}: missing_required_release_packages must exist and be empty"
        ));
    }
    let required_release_packages = value
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
            blockers.push(format!(
                "{evidence}: required_release_packages must include `{required_package}`"
            ));
        }
    }
    let Some(tag_story) = value
        .get("tag_story")
        .and_then(serde_json::Value::as_object)
    else {
        blockers.push(format!("{evidence}: missing tag_story"));
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
        if tag_story.get(field).and_then(serde_json::Value::as_str) != Some(expected) {
            blockers.push(format!(
                "{evidence}: tag_story.{field} must be `{expected}`"
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
            .is_some_and(|entries| entries.iter().any(|entry| entry.as_str() == Some(required)));
        if !present {
            blockers.push(format!(
                "{evidence}: tag_story.required_in_release_notes is missing `{required}`"
            ));
        }
    }
}

fn inspect_markdown_evidence(evidence: &str, path: &Path, blockers: &mut Vec<String>) {
    let text = match read_text_bounded(path) {
        Ok(text) => text,
        Err(error) => {
            blockers.push(format!(
                "{evidence}: failed to read markdown evidence: {error}"
            ));
            return;
        }
    };
    if text.trim().is_empty() {
        blockers.push(format!("{evidence}: markdown evidence is empty"));
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
                blockers.push(format!(
                    "{evidence}: markdown evidence contains unresolved marker `{marker}`"
                ));
                break;
            }
        }
    }
    if evidence.starts_with("evidence/docs/") && !text.contains("Evidence sources:") {
        blockers.push(format!(
            "{evidence}: generated docs evidence does not list evidence sources"
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

fn paths_equal(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }
    match (left.canonicalize(), right.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn resolve_manifest_path(base_dir: &Path, path: &str) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        candidate
    } else {
        base_dir.join(candidate)
    }
}

fn is_checklist_artifact(entry: &str) -> bool {
    !entry.starts_with("cargo_full ")
        && (entry.ends_with(".json")
            || entry.ends_with(".md")
            || entry.ends_with(".yml")
            || entry.ends_with(".yaml")
            || entry.ends_with(".toml")
            || entry.starts_with("release/"))
}

fn is_manifest_command_evidence(evidence: &str) -> bool {
    evidence.starts_with("cargo_full ")
}

fn resolve_checklist_artifact_path(base_dir: &Path, path: &str) -> PathBuf {
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
    let mut reader = fs::File::open(path)?.take(MAX_RELEASE_AUDIT_TEXT_BYTES.saturating_add(1));
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    if text.len() as u64 > MAX_RELEASE_AUDIT_TEXT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} exceeds {MAX_RELEASE_AUDIT_TEXT_BYTES} byte release audit read cap",
                path.display()
            ),
        ));
    }
    Ok(text)
}
