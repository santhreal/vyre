//! Proof plan summary, its catalog and execution hashes, and the `plan` subcommand.

use crate::artifact_json::write_json_artifact;
use crate::backend_selection::{select_backends, semantic_execution_backends};
use crate::operation_selection::{
    prepare_entry, select_entries, unified_entries, PreparedEntry, UnifiedEntry,
};
use crate::proof_options::{parse_proof_options, ProofOptions};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub(crate) struct ProofPlanSummary {
    pub(crate) backend_count: usize,
    pub(crate) op_count: usize,
    pub(crate) pair_count: usize,
    pub(crate) witness_case_count: usize,
    pub(crate) catalog_hash: String,
    pub(crate) execution_hash: String,
    pub(crate) selection: ProofSelectionSummary,
}

#[derive(Debug, Serialize)]
pub(crate) struct ProofSelectionSummary {
    pub(crate) backend_filter: String,
    pub(crate) ops_filter: String,
    pub(crate) shard_index: Option<usize>,
    pub(crate) shard_count: Option<usize>,
    pub(crate) universe_backend_count: usize,
    pub(crate) universe_op_count: usize,
    pub(crate) selected_backend_count: usize,
    pub(crate) selected_op_count: usize,
}

#[derive(Debug, Serialize)]
struct ProofPlanArtifact {
    pub(crate) wire_format_version: u32,
    pub(crate) plan: ProofPlanSummary,
    pub(crate) backends: Vec<String>,
    pub(crate) ops: Vec<String>,
}

pub(crate) fn emit_plan(args: impl IntoIterator<Item = String>) -> Result<(), String> {
    let options = parse_proof_options("plan", args)?;
    let all_backends = semantic_execution_backends()?;
    if all_backends.is_empty() {
        return Err(
            "plan refused to emit: no dispatch-capable backend is linked into this binary. \
             Fix: build with `--features gpu` or link another real dispatch backend."
                .to_string(),
        );
    }
    let backends = select_backends(&all_backends, &options.backend_filter)?;
    let all_entries = unified_entries();
    let entries = select_entries(&all_entries, &options.ops_filter, options.shard)?;
    let mut prepared_entries = Vec::with_capacity(entries.len());
    let mut failures = Vec::new();
    for entry in entries {
        match prepare_entry(entry) {
            Ok(prepared) => prepared_entries.push(prepared),
            Err(error) => failures.push(format!("{}: {}", entry.id, error)),
        }
    }
    if !failures.is_empty() {
        return Err(format!(
            "plan refused to emit because {} selected op(s) cannot produce executable witnesses:\n{}\nFix: repair every witness before planning conformance shards.",
            failures.len(),
            failures.join("\n")
        ));
    }
    let pair_count = backends.len().saturating_mul(prepared_entries.len());
    let plan = proof_plan_summary(
        &all_backends,
        &all_entries,
        &backends,
        &prepared_entries,
        pair_count,
        &options,
    );
    let artifact = ProofPlanArtifact {
        wire_format_version: 1,
        backends: backends
            .iter()
            .map(|backend| backend.id.to_string())
            .collect(),
        ops: prepared_entries
            .iter()
            .map(|entry| entry.id.to_string())
            .collect(),
        plan,
    };
    let json = serde_json::to_string_pretty(&artifact).map_err(|error| {
        format!("failed to serialize proof plan: {error}. Fix: keep plan fields JSON-serializable.")
    })?;
    if let Some(out) = options.out.as_deref() {
        write_json_artifact(out, json, "proof plan")
    } else {
        println!("{json}");
        Ok(())
    }
}

pub(crate) fn proof_plan_summary(
    universe_backends: &[&'static vyre_driver::BackendRegistration],
    universe_entries: &[UnifiedEntry],
    backends: &[&'static vyre_driver::BackendRegistration],
    entries: &[PreparedEntry],
    pair_count: usize,
    options: &ProofOptions,
) -> ProofPlanSummary {
    let mut catalog_hasher = blake3::Hasher::new();
    catalog_hasher.update(b"vyre-conform/proof-catalog/v2");
    for backend in universe_backends {
        catalog_hasher.update(backend.id.as_bytes());
    }
    for entry in universe_entries {
        catalog_hasher.update(entry.id.as_bytes());
    }

    let mut execution_hasher = blake3::Hasher::new();
    execution_hasher.update(b"vyre-conform/proof-execution/v2");
    for backend in backends {
        execution_hasher.update(backend.id.as_bytes());
    }
    let mut witness_case_count = 0usize;
    for entry in entries {
        execution_hasher.update(entry.id.as_bytes());
        execution_hasher.update(&entry.cases.len().to_le_bytes());
        execution_hasher.update(&entry.program.buffers().len().to_le_bytes());
        execution_hasher.update(&entry.input_plan.source_count().to_le_bytes());
        execution_hasher.update(&entry.input_plan.zeroed_input_count().to_le_bytes());
        execution_hasher.update(&entry.reference_cases.len().to_le_bytes());
        witness_case_count += entry.cases.len().saturating_mul(backends.len());
    }
    let selection = ProofSelectionSummary {
        backend_filter: options.backend_filter.clone(),
        ops_filter: options.ops_filter.clone(),
        shard_index: options.shard.map(|shard| shard.index),
        shard_count: options.shard.map(|shard| shard.count),
        universe_backend_count: universe_backends.len(),
        universe_op_count: universe_entries.len(),
        selected_backend_count: backends.len(),
        selected_op_count: entries.len(),
    };
    ProofPlanSummary {
        backend_count: backends.len(),
        op_count: entries.len(),
        pair_count,
        witness_case_count,
        catalog_hash: catalog_hasher.finalize().to_hex().to_string(),
        execution_hash: execution_hasher.finalize().to_hex().to_string(),
        selection,
    }
}

pub(crate) fn hash_proof_plan(hasher: &mut blake3::Hasher, plan: &ProofPlanSummary) {
    hasher.update(plan.catalog_hash.as_bytes());
    hasher.update(plan.execution_hash.as_bytes());
    hasher.update(&plan.backend_count.to_le_bytes());
    hasher.update(&plan.op_count.to_le_bytes());
    hasher.update(&plan.pair_count.to_le_bytes());
    hasher.update(&plan.witness_case_count.to_le_bytes());
    hasher.update(plan.selection.backend_filter.as_bytes());
    hasher.update(plan.selection.ops_filter.as_bytes());
    hash_optional_usize(hasher, plan.selection.shard_index);
    hash_optional_usize(hasher, plan.selection.shard_count);
    hasher.update(&plan.selection.universe_backend_count.to_le_bytes());
    hasher.update(&plan.selection.universe_op_count.to_le_bytes());
    hasher.update(&plan.selection.selected_backend_count.to_le_bytes());
    hasher.update(&plan.selection.selected_op_count.to_le_bytes());
}

fn hash_optional_usize(hasher: &mut blake3::Hasher, value: Option<usize>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(&value.to_le_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
}
