//! Generate source-owned optimization integration evidence.

use std::path::PathBuf;

use serde::Serialize;

use crate::release::optimizer_pass_rows::{self, OptimizerPassRow};

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

pub(crate) fn run(args: &[String]) {
    let output = match parse_output(args) {
        Ok(output) => output,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let (executable_passes, rows) = optimizer_pass_rows::collect();
    let mut blockers = optimizer_pass_rows::duplicate_id_blockers(&rows);
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
        blockers.push(
            "optimizer catalog contains an entry without owner, invariant, proof, or benchmark"
                .to_string(),
        );
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
    xtask::output_arg::write_json(&output, &matrix);
    println!("optimization-matrix: wrote {}", output.display());
    if !matrix.blockers.is_empty() {
        std::process::exit(1);
    }
}

fn parse_output(args: &[String]) -> Result<PathBuf, String> {
    xtask::output_arg::parse_output_arg(
        args,
        "optimization-matrix",
        "Writes source-owned semantic optimizer integration evidence.",
        default_output,
    )
}

fn default_output() -> PathBuf {
    xtask::checkout::checkout_root()
        .join("release/evidence/optimization/optimization-integration-matrix.json")
}
