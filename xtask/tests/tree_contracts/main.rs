//! Contracts that judge the checked-in tree from files, in one link unit.
//!
//! Each module below owns one feature area and is read on its own. They share a
//! harness rather than a subject: every one resolves the checkout root, builds a
//! fixture workspace, runs a repository generator or parses source text, and
//! asserts on the result. Ten separate integration targets linked the crate ten
//! times and carried ten copies of the same three helpers.
//!
//! `docs_references` and `release_docs` stay separate targets because the
//! workspace contract and `docs/DOCUMENTATION_COVERAGE.md` name their focused
//! commands; they reach the same harness through `tests/common`.

#![forbid(unsafe_code)]

#[path = "../common/mod.rs"]
mod common;

mod architecture_docs;
mod canonical_first_workgroup_guard;
mod ci_workflow_references;
mod cli_docs;
mod crate_ownership_registry;
mod crate_readmes;
mod feature_isolation;
mod public_api_snapshot_inventory;
mod relation_import_certificates;
mod testing_guides;
