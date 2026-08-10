//! JSON/TOML/Markdown semantic inspection for release completion audit evidence.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use super::paths::{markdown_line_is_release_rule_text, read_text_bounded};

fn inspect_contract_baselines(
    evidence: &str,
    artifact: Option<&str>,
    report: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let artifact_context = artifact
        .map(|artifact| format!("suite artifact `{artifact}` "))
        .unwrap_or_default();
    for issue in crate::benchmark_evidence_semantics::contract_backend_issues(report) {
        match issue {
            crate::benchmark_evidence_semantics::ContractBackendIssue::MissingBaselines {
                case_id,
                backend_id,
            } => blockers.push(format!(
                "{evidence}: {artifact_context}case `{case_id}` backend `{backend_id}` has a performance contract with no baselines"
            )),
            crate::benchmark_evidence_semantics::ContractBackendIssue::NoApplicableBaseline {
                case_id,
                backend_id,
            } => blockers.push(format!(
                "{evidence}: {artifact_context}case `{case_id}` backend `{backend_id}` has no applicable performance contract baseline"
            )),
        }
    }
}

include!("evidence_dispatch.rs");
include!("weir_backend_conformance.rs");
include!("conformance_evidence.rs");
include!("release_run_optimization_benchmarks.rs");
include!("optimization_hygiene.rs");
include!("optimization_test_matrices.rs");
include!("suite_benchmark_release.rs");
include!("metadata_launch.rs");
include!("backend_suites.rs");
include!("backend_workload_matrices.rs");
include!("benchmark_parser_semantics.rs");
include!("c_parser_evidence.rs");
include!("parser_cpu_version_evidence.rs");
