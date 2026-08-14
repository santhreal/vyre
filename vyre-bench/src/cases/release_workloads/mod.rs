//! Compiler-grade release macro benchmark workloads and the evidence records that gate them.

mod callgraph_reachability;
mod families;
mod macro_registry;
mod metadata_condition;
mod registration;
mod release_evidence;
mod run_assembly;
mod sparse_compaction;
mod synthetic_count;
mod synthetic_oracle;
mod synthetic_programs;

pub use callgraph_reachability::CallgraphReachabilityStep;
pub use macro_registry::{
    build_release_count_macro_case_for_records, build_release_macro_case_for_records,
    build_release_macro_cases_for_family_and_records, build_release_macro_program,
    build_release_macro_program_for_records, release_count_macro_program_specs_for_records,
    release_macro_program_specs, release_macro_program_specs_for_family_and_records,
    release_macro_program_specs_for_records, ReleaseMacroFamily, ReleaseMacroGeneratedCase,
    ReleaseMacroProgramSpec,
};
pub use metadata_condition::MetadataConditionBatch;
pub use release_evidence::{
    validate_release_math_nn_kernel_evidence, validate_release_scan_competitor_corpus_metadata,
    ReleaseMathNnKernelEvidence, ReleaseScanCompetitorCorpusMetadata,
};
pub use sparse_compaction::SparseOutputCompactionCount;
