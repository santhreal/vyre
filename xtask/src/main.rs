//! xtask dispatcher for the vyre workspace.

use std::env;
use std::process;

mod abstraction_gate;
mod artifact_paths;
mod backend_matrix;
mod bench_crossback;
mod bench_release;
mod benchmark_evidence_semantics;
mod binary;
mod catalog;
mod check_cat_a;
mod check_tier_deps;
mod compile;
mod conformance_evidence_semantics;
mod conformance_matrix;
mod dedup_report;
mod dep_drift;
mod docs_check;
mod dup_scan;
mod feature_matrix;
mod gate1;
mod gates;
mod hash;
mod heuristic_audit;
mod hot_path_scan;
mod hygiene_matrix;
mod implementation_family;
mod json_output;
mod launch_contract;
mod launch_state;
mod lego_audit;
mod lego_quick;
mod list_ops;
mod manifest_walk;
mod metadata_matrix;
mod op_matrix;
mod operation_schema;
mod optimization_corpus;
mod optimization_docs;
mod optimization_matrix;
mod output_arg;
mod ownership;
mod package_readiness;
mod platform_boundary;
mod print_composition;
mod release_backend_rows;
mod release_benchmarks;
mod release_conformance;
mod release_evidence;
mod release_gate;
mod release_train;
mod release_workload_matrix;
mod repo_boundary;
mod research_key;
mod research_source_ledger;
mod shrink;
mod subcommands;
mod text_markers;
mod toml_config;
mod trace_f32;
mod use_paths;
mod verify_rewrite_proofs;
mod version_matrix;
mod vyre_release_gate;
mod whats_similar;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Fix: missing subcommand. See --help.");
        process::exit(1);
    }

    let name = args[1].as_str();
    if name == "--help" || name == "-h" {
        print!("{}", subcommands::help_text());
        process::exit(0);
    }
    match subcommands::find(name) {
        Some(entry) => (entry.run)(&args),
        None => {
            eprintln!("Fix: unknown subcommand '{name}'. See --help.");
            process::exit(1);
        }
    }
}
