//! The subcommands that judge the working tree, and the sweep that runs them.
//!
//! A gate here reads the repository, reports findings, and fails. It never
//! writes release evidence and never regenerates documentation; those are
//! `release` and `docs`. `gates::sweep` is the sweep itself and the wiring
//! meta-check that keeps every registered gate connected to a pinned
//! baseline and a workflow.

pub mod check_cat_a;
pub mod check_tier_deps;
pub mod dedup_report;
pub mod dep_drift;
pub mod dup_scan;
pub mod feature_isolation;
pub mod hot_path_scan;
pub mod hygiene_matrix;
pub mod implementation_family;
pub mod ownership;
pub mod platform_boundary;
pub mod sweep;
pub mod use_paths;
