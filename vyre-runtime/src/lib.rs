//! Artifact execution, resident work queues, resource residency, and zero-copy IO.
//!
//! Runtime construction starts from an authenticated [`artifact_admission::ArtifactSession`].
//! Immutable compiler artifacts are materialized through registered target
//! devices; runtime policy owns bindings, retained state, queueing, recovery,
//! resource residency, IO, and telemetry.

// Every `unsafe` construct in this crate is refused unless the file or the
// single item carrying it also carries `allow(unsafe_code)`, beside the comment
// discharging the caller obligation. A block added anywhere else is a compile
// error rather than an inherited permission. The four files below are the
// permitted places; each `unsafe-permitted` line is parsed by the runtime
// unsafe-permission test, which rejects a grant in any file this list does not
// name and any name here that carries no grant. `uring/raw_platform.rs` is the
// only file-level allowance; the rest are one item each.
//
// unsafe-permitted: uring/raw_platform.rs
// unsafe-permitted: uring/buffer.rs
// unsafe-permitted: uring/ring.rs
// unsafe-permitted: uring_completion_pump.rs

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

/// External resource admission, layout/usage transition execution, and timeline synchronization.
mod external_resource_admission;
/// Paged resource residency contracts, geometry validation, and candidate planning.
pub mod paged_resource;
/// Backend-neutral immutable-resource and mutable-state residency.
pub mod resource_residency;
/// Authenticated typed resource transfers, residency composition, and integrity.
pub mod resource_transfer;
/// Radix retained-page-cache lifecycle, immutable identity, and copy-on-write allocation.
pub mod retained_page_cache;
/// Bounded routed-work queues, route-based scheduling, and inter-device exchange.
pub mod routed_work_queue;
/// Speculative state transactions, provisional retained state, and transactional verification/rollback.
pub mod speculative_transaction;
pub use external_resource_admission::{
    AdmittedExternalResourceLease, ExternalAdmissionError, ExternalResourceAdmissionManager,
};

/// Resident work-queue protocols, scheduling policy, and runtime IO.
pub mod resident_work_queue;

/// Authenticated persistent execution over retained artifact bindings.
pub mod persistent_executor;
/// Content-addressed authenticated artifact cache.
pub mod pipeline_cache;

/// Atomic guarded state machines, prepare/commit journals, and restart budgets.
pub mod atomic_recovery;
/// Generation-scoped cache namespaces and rolling upgrade coordination.
mod generation_namespace;
/// Structured artifact-session recovery without message parsing or recompilation.
pub mod recovery;
/// Differential megakernel replay log  -  captures every published
/// ring slot so a later cert run can diff epoch-by-epoch execution
/// against a live backend.
pub mod replay;

/// Backend routing policy for execution plans.
pub mod routing;

/// Multi-GPU work partitioning across runtime backends.
pub mod scheduler;

/// Multi-tenant megakernel multiplexing  -  one persistent kernel per
/// GPU, shared across producer tools via the `tenant_id` field already
/// in the ring protocol.
pub mod tenant;

/// Linux io_uring integration. Compiled out on macOS / Windows.
#[cfg(target_os = "linux")]
pub mod uring;

/// Completion pump for an optional Linux io_uring stream.
pub mod uring_completion_pump;

/// Mandatory finite session quotas and typed session identity.
pub use vyre_foundation::failure_domain::{
    FailureDomain, RecoveryClass, RecoveryDisposition, StateOwnerRecovery, TypedRecoveryError,
};

mod session_quota;
pub use session_quota::*;

/// Structured concurrency, cancellation propagation, and worker/device quarantine.
pub mod structured_concurrency;

pub use generation_namespace::{
    GenerationScopedNamespace, RollingUpgradeCoordinator, UpgradePhase,
};
