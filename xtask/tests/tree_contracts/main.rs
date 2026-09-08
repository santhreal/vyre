//! Contracts that judge the checked-in tree from files, in one link unit.
//!
//! Each module below owns one feature area and is read on its own. They share a
//! harness rather than a subject: every one resolves the checkout root, builds a
//! fixture workspace, runs a repository generator or parses source text, and
//! asserts on the result. Ten separate integration targets linked the crate ten
//! times and carried ten copies of the same three helpers.
//!
//! `docs_references` and `release_docs` are modules of the crate's `all_tests`
//! harness rather than of this one: they are the two suites a documentation or
//! release change re-runs on its own, and a filter over that harness selects
//! them without linking every tree contract here. They reach the same helpers
//! through `tests/workspace_sources`.

#![forbid(unsafe_code)]

#[path = "../workspace_sources/mod.rs"]
mod workspace_sources;

mod architecture_docs;
mod canonical_first_workgroup_guard;
mod cargo_invocation_resolution;
mod ci_required_contexts;
mod ci_workflow_references;
mod cli_surface;
mod codeowners_paths;
mod config_space_contracts;
mod crate_ownership_registry;
mod crate_readmes;
mod docs_manifest_completeness;
mod exit_states_a_cause;
mod feature_isolation;
mod manifest_dependency_tables;
mod msrv_toolchain;
mod nested_byte_rows;
#[cfg(feature = "public-api-tool")]
mod public_api_snapshot_inventory;
mod relation_import_certificates;
/// Unix only: the subject is a shell script, and the release hosts run it.
#[cfg(unix)]
mod release_shell_toml_reader;
mod release_provenance_contracts;
mod subcommand_dispatch;
mod test_mutation_hygiene;
mod testing_guides;
mod tree_walk_order;

