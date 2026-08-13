//! The subcommands that judge the working tree, and the sweep that runs them.
//!
//! A gate here reads the repository, reports findings, and fails. It never
//! writes release evidence and never regenerates documentation; those are
//! `release` and `docs`. `gates::gates` is the sweep itself and the wiring
//! meta-check that keeps every registered gate connected to a pinned
//! baseline and a workflow.

pub(crate) mod abstraction_gate;
pub(crate) mod check_cat_a;
pub(crate) mod check_tier_deps;
pub(crate) mod dedup_report;
pub(crate) mod dep_drift;
pub(crate) mod dup_scan;
pub(crate) mod gate1;
pub(crate) mod gates;
pub(crate) mod heuristic_audit;
pub(crate) mod hot_path_scan;
pub(crate) mod hygiene_matrix;
pub(crate) mod implementation_family;
pub(crate) mod lego_audit;
pub(crate) mod lego_quick;
pub(crate) mod ownership;
pub(crate) mod platform_boundary;
pub(crate) mod use_paths;
pub(crate) mod verify_rewrite_proofs;
pub(crate) mod whats_similar;
