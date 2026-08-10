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
mod command_matrix;
mod compile;
mod conformance_evidence_semantics;
mod conformance_matrix;
mod dedup_report;
mod dep_drift;
mod docs_check;
mod feature_matrix;
mod gate1;
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
mod lint_shape_tests;
mod list_ops;
mod manifest_walk;
mod metadata_matrix;
mod op_matrix;
mod operation_schema;
mod optimization_corpus;
mod optimization_matrix;
mod output_arg;
mod ownership;
mod package_readiness;
mod parser_coherence;
mod paths;
mod platform_boundary;
mod print_composition;
mod quick;
mod quick_cache;
mod recursion_gate;
mod release_backend_rows;
mod release_benchmarks;
mod release_completion_audit;
mod release_conformance;
mod release_evidence;
mod release_gate;
mod release_train;
mod release_workload_matrix;
mod repo_boundary;
mod research_key;
mod research_source_ledger;
mod shrink;
mod source_similar;
mod test_matrix;
mod text_markers;
mod toml_config;
mod trace_f32;
mod use_paths;
mod verify_rewrite_proofs;
mod version_matrix;
mod vyre_release_gate;
mod weir_matrix;
mod whats_similar;

fn print_help() {
    println!(
        "vyre xtask runner\n\
         \n\
         USAGE:\n\
           cargo_full run --bin xtask -- <subcommand> [options]\n\
         \n\
         SUBCOMMANDS:\n\
           quick-check --op NAME               Run minimal <5s verification path for a single op\n\
           abstraction-gate                     Enforce registered building-block boundaries\n\
           bench-crossback [program]           Cross-backend perf table\n\
           backend-matrix [--output PATH]      Probe linked CUDA/WGPU backend release policy\n\
           bench-release [--backend all]        Run the legacy cross-backend release benchmark coordinator\n\
           shrink <file.vir> <oracle.sh>       Delta-debug a crashing vyre wire formulation down to a minimal reproducer\n\
           check-cat-a                         Run every Cat-A pre-merge gate\n\
           check-tier-deps                     Reject upward tier path dependencies (T4→T1 only)\n\
           command-matrix [--output PATH] [--check] Generate/check xtask command owner/proof matrix\n\
           compile <program.vir> --to TARGET   Emit target artifact(s) (wgsl/spirv/secondary_text/native_module/hlsl)\n\
           conformance-matrix [--check] [--output PATH] Enumerate/check release op/backend conformance coverage\n\
           dep-drift                           Fail if any repo manifest pins a workspace-managed dependency to a different version\n\
           docs-check                           Validate manifest-backed documentation lifecycle and generated navigation\n\
           feature-matrix [--output PATH]      Generate Vyre/Weir crate feature evidence matrix\n\
           print-composition <op_id>           Walk an op's Region tree and print its decomposition chain\n\
           trace-f32 <op_id>                   Run an op's test_inputs through vyre-reference and dump expected_output literal\n\
           gate1                               Enforce Gate 1 complexity budget (CI floor)\n\
           launch-state [--output PATH]       Generate public launch completion state evidence\n\
           list-ops [--write PATH|--check]     Render or check the schema-derived operation inventory\n\
           metadata-matrix [--output PATH]     Generate Vyre/Weir crate metadata evidence\n\
           operation-schema [--output PATH] [--check] [--validate PATH]  Generate or verify the canonical live operation contract schema\n\
           op-matrix [--output PATH]           Generate operation/backend coverage evidence\n\
           optimization-matrix [--output PATH] Generate release optimization integration evidence\n\
           package-readiness [--output PATH]  Generate pre-publish package order evidence\n\
           optimization-corpus [--output PATH]  Generate release optimization corpus manifest\n\
           parser-coherence [--output PATH]   Generate distributed C parser ownership evidence\n\
           platform-boundary                  Fail on consumer names in platform crate docs/comments\n\
           version-matrix [--output PATH]      Generate Vyre/Weir manifest version matrix\n\
           weir-matrix [--output PATH]         Generate Weir analysis API evidence matrix\n\
           catalog [--out DIR] [--check]       Emit one markdown table per subsystem under docs/catalog; --check gates drift\n\
           release-gate                        Pre-publish sanity checks (catalog + gate1 + Cargo.lock clean)\n\
           release-workload-matrix [--output PATH]  Generate cheap release workload family evidence\n\
           release-benchmarks [--backend cuda] Generate long-running release benchmark artifacts\n\
           release-conformance [--backend all] Generate real backend conformance artifacts\n\
           release-completion-audit [--output PATH]  Generate final prompt-to-artifact audit evidence\n\
           release-evidence                    Generate cheap structural release evidence artifacts\n\
           vyre-release-gate [--prepublish] [--manifest PATH]  Enforce final or prepublication evidence closure\n\
           recursion-gate [--strict]           Enforce recursion thesis (every Tier-2.5 primitive has a vyre-self consumer)\n\
           heuristic-audit [--strict]          Surface hand-rolled heuristics that should be self-consumer calls\n\
           verify-rewrite-proofs               Verify optimizer rewrite proof fixtures\n\
           hygiene-matrix [--output PATH]      Scan Vyre/Weir source hygiene release blockers\n\
           lego-audit [--report-only|--with-repo|--write-baseline] [--duplicate-report-json PATH] Deeper LEGO-block enforcement and composition baseline management\n\
           lego-quick [--all] [--source-similar] Fast pre-commit gate plus optional source-dedup scan\n\
           whats-similar (--op-id <id>|--all) [--duplicate-report-json PATH] Pre-write/all-pairs duplicate query by IR shape\n\
           source-similar [--root PATH] [--check] [--include-untracked] [--duplicate-report-json PATH] Repo-wide Rust source duplicate scanner\n\
           hot-path-scan [--strict]            Scan files in HOT_PATHS.toml for clone/alloc/lock patterns\n\
           test-matrix [--output PATH]         Generate Vyre/Weir test architecture evidence\n\
           lint-shape-tests [--strict]         Scan test modules for shape-only assertions\n\
         \n\
           --help                              Print this message\n"
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Fix: missing subcommand. See --help.");
        process::exit(1);
    }

    match args[1].as_str() {
        "quick-check" => quick::cmd_quick_check(&args),
        "abstraction-gate" => abstraction_gate::run(&args),
        "bench-crossback" => bench_crossback::run(&args),
        "backend-matrix" => backend_matrix::run(&args),
        "bench-release" => bench_release::run(&args),
        "shrink" => shrink::run(&args),
        "check-cat-a" => check_cat_a::run(&args),
        "check-tier-deps" => check_tier_deps::run(&args),
        "command-matrix" => command_matrix::run(&args),
        "compile" => compile::run(&args),
        "conformance-matrix" => conformance_matrix::run(&args),
        "dep-drift" => dep_drift::run(&args),
        "docs-check" => docs_check::run(&args),
        "feature-matrix" => feature_matrix::run(&args),
        "print-composition" => print_composition::run(&args),
        "list-ops" => list_ops::run(&args),
        "metadata-matrix" => metadata_matrix::run(&args),
        "operation-schema" => operation_schema::run(&args),
        "op-matrix" => op_matrix::run(&args),
        "optimization-matrix" => optimization_matrix::run(&args),
        "package-readiness" => package_readiness::run(&args),
        "optimization-corpus" => optimization_corpus::run(&args),
        "parser-coherence" => parser_coherence::run(&args),
        "platform-boundary" => platform_boundary::run(&args),
        "catalog" => catalog::run(&args),
        "release-gate" => release_gate::run(&args),
        "release-workload-matrix" => release_workload_matrix::run(&args),
        "release-benchmarks" => release_benchmarks::run(&args),
        "release-conformance" => release_conformance::run(&args),
        "release-completion-audit" => release_completion_audit::run(&args),
        "release-evidence" => release_evidence::run(&args),
        "vyre-release-gate" => vyre_release_gate::run(&args),
        "recursion-gate" => recursion_gate::run(&args),
        "heuristic-audit" => heuristic_audit::run(&args),
        "hygiene-matrix" => hygiene_matrix::run(&args),
        "trace-f32" => trace_f32::run_cmd(&args),
        "verify-rewrite-proofs" => verify_rewrite_proofs::run(&args),
        "version-matrix" => version_matrix::run(&args),
        "weir-matrix" => weir_matrix::run(&args),
        "gate1" => gate1::run(&args),
        "lego-audit" => lego_audit::run(&args),
        "lego-quick" => lego_quick::run(&args),
        "whats-similar" => whats_similar::run(&args),
        "source-similar" => source_similar::run(&args),
        "hot-path-scan" => hot_path_scan::run(&args),
        "test-matrix" => test_matrix::run(&args),
        "lint-shape-tests" => lint_shape_tests::run(&args),
        "launch-state" => launch_state::run(&args),
        "--help" | "-h" => {
            print_help();
            process::exit(0);
        }
        _ => {
            eprintln!("Fix: unknown subcommand '{}'. See --help.", args[1]);
            process::exit(1);
        }
    }
}
