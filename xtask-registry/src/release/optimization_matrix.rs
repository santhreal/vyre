//! Hold the optimizer integration matrix to the pass catalog the source declares.

use serde::Serialize;
use xtask::artifact_gate::Inspection;

use crate::release::optimizer_pass_rows::{self, OptimizerPassRow};

/// The artifact this gate owns, relative to the workspace root.
const ARTIFACT: &str = "release/evidence/optimization/optimization-integration-matrix.json";

#[derive(Debug, Serialize)]
struct OptimizationMatrix {
    schema_version: u32,
    architecture: &'static str,
    semantic_optimizer_owner: &'static str,
    verified_lowering_owner: &'static str,
    target_strategy_owner: &'static str,
    executable_passes: usize,
    catalog_entries: usize,
    entries: Vec<OptimizationMatrixEntry>,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct OptimizationMatrixEntry {
    #[serde(flatten)]
    pass: OptimizerPassRow,
    input: &'static str,
    output: &'static str,
}

xtask::artifact_gate! {
    /// Holds the optimizer integration matrix to the live pass catalog.
    OptimizationMatrixGate,
    name: "optimization-matrix",
    help: "Regenerate release/evidence/optimization/optimization-integration-matrix.json from the \
       live optimizer pass catalog and report every line the committed artifact disagrees on. \
       Proves the artifact lists exactly the catalog entries the source registers, that no pass \
       id repeats, and that every entry names an owner, invariant, proof and benchmark. Proves \
       nothing about whether a pass is correct, ever fires, or improves anything: the named \
       proof and benchmark are strings the catalog carries, not results this gate reads.",
    inspect: |ctx| inspect(),
}

/// What the optimizer catalog says, and the artifact that records it.
fn inspect() -> Inspection {
    let mut inspection = Inspection::new();
    let (executable_passes, rows) = optimizer_pass_rows::collect();
    let mut blockers = optimizer_pass_rows::duplicate_id_blockers(&rows);
    for blocker in &blockers {
        inspection.blocked(
            ARTIFACT,
            blocker.clone(),
            "Give each optimizer catalog entry a unique id. An id is the only handle the published \
             evidence offers, so a repeated one makes every row under it ambiguous.",
        );
    }
    let entries = rows
        .into_iter()
        .map(|pass| OptimizationMatrixEntry {
            pass,
            input: "vyre-foundation Program",
            output: "semantically equivalent vyre-foundation Program",
        })
        .collect::<Vec<_>>();
    if entries.iter().any(|entry| {
        entry.pass.owner.is_empty()
            || entry.pass.invariant.is_empty()
            || entry.pass.proof.is_empty()
            || entry.pass.benchmark.is_empty()
    }) {
        let blocker =
            "optimizer catalog contains an entry without owner, invariant, proof, or benchmark"
                .to_string();
        inspection.blocked(
            ARTIFACT,
            blocker.clone(),
            "Fill in the missing field in vyre-foundation's optimizer pass catalog. An entry \
             without an owner or an invariant publishes a pass nobody answers for.",
        );
        blockers.push(blocker);
    }
    let matrix = OptimizationMatrix {
        schema_version: 2,
        architecture: "semantic Program optimizer -> verified representation lowering -> concrete target strategy",
        semantic_optimizer_owner: "vyre-foundation",
        verified_lowering_owner: "vyre-lower",
        target_strategy_owner: "concrete emitters and drivers",
        executable_passes,
        catalog_entries: entries.len(),
        entries,
        blockers,
    };
    inspection.generates(ARTIFACT, &matrix);
    inspection
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimization_matrix_detects_incomplete_catalog_entries() {
        let mut inspection = Inspection::new();
        let entry_with_empty_fields = OptimizationMatrixEntry {
            pass: OptimizerPassRow {
                id: "pass_a".to_string(),
                kind: "pass",
                owner: "",
                phase: "phase".to_string(),
                boundary: "boundary".to_string(),
                requires: Vec::new(),
                invalidates: Vec::new(),
                capabilities: Vec::new(),
                preserves_abi: true,
                invariant: "invariant",
                termination: "termination",
                proof: "proof",
                benchmark: "benchmark",
            },
            input: "input",
            output: "output",
        };

        let entries = vec![entry_with_empty_fields];
        if entries.iter().any(|entry| {
            entry.pass.owner.is_empty()
                || entry.pass.invariant.is_empty()
                || entry.pass.proof.is_empty()
                || entry.pass.benchmark.is_empty()
        }) {
            inspection.blocked(
                ARTIFACT,
                "optimizer catalog contains an entry without owner, invariant, proof, or benchmark"
                    .to_string(),
                "Fill in the missing field.",
            );
        }

        let blocked = &inspection.findings;
        assert_eq!(blocked.len(), 1);
        assert!(blocked[0]
            .message
            .contains("without owner, invariant, proof, or benchmark"));
    }
}
