//! Resident work-queue protocol, host mirrors, scheduling policy, and IO.
//!
//! Artifact compilation and target selection live in `vyre-megakernel`.
//! Authenticated execution and recovery live in
//! [`crate::artifact_admission::ArtifactSession`].
//! This module owns only mutable queue policy and wire state.

#[cfg(feature = "megakernel-batch")]
pub mod advanced;
pub mod atomic_relaxed;
pub mod automata_worklist;
#[cfg(test)]
mod body_preorder;
pub mod builder;
pub mod descriptor;
pub mod handlers;
pub mod io;
mod lru_tick_cache;
pub mod mixed_work;
pub mod planner;
pub mod policy;
pub mod protocol;
mod protocol_api;
pub mod readback;
pub mod resident;
pub mod ring;
#[cfg(feature = "megakernel-batch")]
pub mod rule_catalog;
pub mod scaling;
pub mod scheduler;
pub mod speculation;
mod staging_reserve;
pub mod task;
pub mod telemetry;
pub mod workspace_adapter;
pub mod workspace_layout;

/// Stateless owner of resident work-queue encoding and decoding operations.
#[derive(Clone, Copy, Debug, Default)]
pub struct ResidentWorkQueue;

/// Test-only builder variant, re-exported for the integration suites.
#[cfg(test)]
pub use builder::build_program_with_self_loading_miss_handler;
/// Ring-slot state transition. `protocol_api` is private, so this is the one
/// public path to it.
pub use protocol_api::RingSlotTransition;
