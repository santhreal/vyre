//! Artifact execution, resident work queues, and zero-copy IO.
//!
//! Runtime construction starts from an authenticated [`artifact_admission::ArtifactSession`].
//! Immutable compiler artifacts are materialized through registered target
//! devices; the runtime holds bindings, retained state, queueing, recovery,
//! and IO.

// `pipeline_error_closure` is an in-crate test module that names this crate by
// its own name, which resolves only through this alias.
#[cfg(test)]
extern crate self as vyre_runtime;

// The `PipelineError` variant-space closure. An exhaustive match over a
// `#[non_exhaustive]` enum is legal only inside the crate that defines it.
#[cfg(test)]
mod pipeline_error_closure;

mod error;
pub(crate) use error::closed_enum;
pub use error::*;

/// Canonical artifact-envelope authentication and exact-format admission.
pub mod artifact_admission;
mod semantic_execution;
pub use semantic_execution::RegisteredSemanticExecutor;

/// Resident work-queue protocols, scheduling policy, and runtime IO.
pub mod resident_work_queue;

/// Authenticated persistent execution over retained artifact bindings.
pub mod persistent_executor;
/// Content-addressed authenticated artifact cache.
pub mod pipeline_cache;

/// Structured artifact-session recovery without message parsing or recompilation.
pub mod recovery;

/// Linux io_uring integration. Compiled out on macOS / Windows.
#[cfg(target_os = "linux")]
pub mod uring;

/// Failure-domain and recovery classification shared with drivers.
pub use vyre_foundation::failure_domain::{
    FailureDomain, RecoveryClass, RecoveryDisposition, StateOwnerRecovery, TypedRecoveryError,
};
