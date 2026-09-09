//! Artifact execution, resident work queues, resource residency, and zero-copy IO.
//!
//! Runtime construction starts from an authenticated [`artifact_admission::ArtifactSession`].
//! Immutable compiler artifacts are materialized through registered target
//! devices; runtime policy owns bindings, retained state, queueing, recovery,
//! resource residency, IO, and telemetry.

// A fixture module shared with the integration suites names this crate by its
// own name, so the same file compiles inside the library and inside a test
// binary.
#[cfg(test)]
extern crate self as vyre_runtime;

// The retained-cache key fixture the integration proofs own.
#[cfg(test)]
#[path = "../tests/retained_cache_fixtures/mod.rs"]
mod retained_cache_fixtures;

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

/// External resource admission, layout/usage transition execution, and timeline synchronization (Row 111).
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

/// Atomic guarded state machines, prepare/commit journals, and restart budgets (Row 122).
pub mod atomic_recovery;
/// Generation-scoped cache namespaces and rolling upgrade coordination (Row 120).
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
pub use uring_completion_pump::{UringCompletionPump, UringPollState};

/// Mandatory finite session quotas and typed session identity.
pub use vyre_foundation::{FailureDomain, RecoveryClass, RecoveryDisposition, TypedRecoveryError};

mod session_quota;
pub use session_quota::*;

/// Structured concurrency, cancellation propagation, and worker/device quarantine.
pub mod structured_concurrency;
pub use structured_concurrency::*;

pub use atomic_recovery::{
    authoritative_runtime_state_owner_registry, AtomicGuardedState, GuardedState,
    PrepareCommitJournal, PrepareTicket, StateOwnerRecovery, SupervisedRestartBudget,
};
pub use generation_namespace::{
    GenerationScopedNamespace, RollingUpgradeCoordinator, UpgradePhase,
};
