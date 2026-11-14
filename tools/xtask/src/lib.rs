//! Workspace structural gates, duplication scanner, coverage pruner, and optional IR debugger.

pub mod cfg_test;
pub mod coverage_objects;
pub mod gates;
pub mod source_scan;
pub mod workspace_manifest;

#[cfg(feature = "debug")]
pub mod debug;

pub use coverage_objects::prune as coverage_objects_prune;
pub use workspace_manifest::{
    crate_ident, find_member_directory, manifest_directories, member_directories, member_directory,
    read_lanes, read_source_bounded, source_file_roster, workspace_excludes, workspace_members,
    workspace_root, workspace_root_from, MAX_SOURCE_BYTES,
};
