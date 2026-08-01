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

include!("part1.rs");
include!("part2.rs");
include!("part3.rs");
include!("part4.rs");
include!("part5.rs");
include!("part6.rs");
include!("part7.rs");
include!("part8.rs");
include!("part9.rs");
include!("part10.rs");
include!("part11.rs");
include!("part12.rs");
include!("part13.rs");
