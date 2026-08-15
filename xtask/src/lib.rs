//! Repository, release and documentation tooling for the vyre workspace.
//!
//! This crate links no vyre crate, and that is its whole point. Every gate used
//! to live in one `xtask` binary that linked the compiler, the drivers and the
//! benchmark harness, so editing any source file in the workspace rebuilt about
//! forty crates before the first gate could read a single line of text. The
//! subcommands that only read source, manifests, workflows and evidence files
//! live here and rebuild only themselves.
//!
//! The rule that decides where a subcommand lives: a subcommand links a vyre
//! crate only if it must observe something that does not exist in source text,
//! which means the live operation registry, a linked backend driver, or a
//! measured benchmark probe. Those live in `xtask-registry` and
//! `xtask-evidence`, and this crate builds and runs them on demand.
//!
//! Items are `pub` rather than `pub(crate)` when the subcommands in those two
//! crates use them: the crate boundary is no longer the tool boundary.

pub mod artifact_paths;
pub mod artifact_gate;
pub mod binary;
pub mod checkout;
pub mod delegate;
pub mod docs;
pub mod gates;
pub mod hash;
pub mod json_output;
pub mod manifest_walk;
pub mod output_arg;
pub mod release;
pub mod shrink;
pub mod subcommands;
pub mod text_markers;
pub mod toml_config;
