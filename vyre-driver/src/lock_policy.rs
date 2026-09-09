//! Canonical failure domain and lock poisoning authority for driver and runtime state.
//!
//! A `Mutex` or `RwLock` is poisoned when a thread panicked while holding it,
//! leaving the guarded memory in an arbitrary intermediate state.
//!
//! Subsystems must not make ad-hoc local decisions (such as unconditionally recovering
//! with `into_inner`, panicking in place, or ignoring poison). Instead, every lock
//! belongs to an explicitly declared [`RecoveryClass`](vyre_foundation::RecoveryClass):
//!
//! 1. [`RecoveryClass::TransactionallyRecoverable`](vyre_foundation::RecoveryClass::TransactionallyRecoverable): In-flight mutation was aborted. The guarded
//!    state is discarded and a typed [`BackendError`] is reported to the caller.
//! 2. [`RecoveryClass::RestartableFromCanonicalInput`](vyre_foundation::RecoveryClass::RestartableFromCanonicalInput): Caches, staging pools, or memoized
//!    entries that can be cleanly discarded/reset to an empty valid state and restarted.
//! 3. [`RecoveryClass::DeviceContextFatal`](vyre_foundation::RecoveryClass::DeviceContextFatal): Device-bound queues, command encoders,
//!    or device handles where poison indicates corrupted GPU submission state. The
//!    device is marked lost and [`BackendError::DeviceLost`] is reported.
//! 4. [`RecoveryClass::ProcessFatal`](vyre_foundation::RecoveryClass::ProcessFatal): Foreign ICD dynamic loader dispatch tables,
//!    global driver runtime init, or external C-ABI boundaries where corrupt state
//!    causes silent memory corruption or SIGSEGV in foreign frames. Process is aborted.
//! 5. [`RecoveryClass::InvariantViolation`](vyre_foundation::RecoveryClass::InvariantViolation): Critical internal data structure
//!    corruption violating compiler invariants. Process is aborted with diagnostic details.

use std::sync::{Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

pub use vyre_foundation::{
    govern_mutex_restartable, govern_rwlock_write_restartable, invariant_violation_poison,
    reclaim_poisoned_for_teardown, reclaim_poisoned_irreplaceable_state, FailureDomain,
    RecoveryClass, RecoveryDisposition, TypedRecoveryError,
};

use crate::BackendError;

/// End the process, naming the owner and the state its poisoned lock guards.
///
/// `owner` names the subsystem holding the lock and `state` names what the
/// lock excludes concurrent access to: `"the device factory"` and `"the graphics loader dispatch
/// table"`, not `"LOADER_STARTUP"`.
pub fn process_fatal_poison(owner: &str, state: &str) -> ! {
    vyre_foundation::failure_domain::process_fatal_poison(owner, state)
}

/// Report a governed poison decision in the vocabulary a backend caller selects on.
///
/// The decision itself is `vyre-foundation`'s; this only renames it. A device
/// domain becomes [`BackendError::DeviceLost`] so a caller reacquires a device,
/// and every other domain becomes `BackendError::PoisonedLock` so a caller
/// rebuilds host state.
fn as_backend_error(error: &TypedRecoveryError, owner: &str, state: &str) -> BackendError {
    match error.domain {
        FailureDomain::DeviceContext => BackendError::DeviceLost {
            backend: owner.to_string(),
            device: state.to_string(),
            generation: 0,
            message: error.reason.clone(),
        },
        FailureDomain::MemoryState
        | FailureDomain::DiskJournal
        | FailureDomain::WorkerProcess
        | FailureDomain::SessionLifecycle
        | FailureDomain::NetworkTransport => BackendError::PoisonedLock {
            lock_error: format!("{}. {}", error.reason, error.fix),
        },
    }
}

/// Take a guard over process-owned state, or report the poison as recoverable.
///
/// The guarded value is discarded with the guard: a caller that receives the
/// error rebuilds it rather than reading what the panic left.
pub fn recoverable_poison<T>(result: Result<T, PoisonError<T>>) -> Result<T, BackendError> {
    result.map_err(BackendError::poisoned_lock)
}

/// Take a mutex guard governed by an explicit failure domain contract.
pub fn govern_mutex<'a, T>(
    mutex: &'a Mutex<T>,
    owner: &'static str,
    state: &'static str,
    class: RecoveryClass,
) -> Result<MutexGuard<'a, T>, BackendError> {
    govern_mutex_with_reset(mutex, owner, state, class, |_| {})
}

/// Take a mutex guard with an explicit reset action executed if restartable state is recovered.
pub fn govern_mutex_with_reset<'a, T, F>(
    mutex: &'a Mutex<T>,
    owner: &'static str,
    state: &'static str,
    class: RecoveryClass,
    reset_on_restart: F,
) -> Result<MutexGuard<'a, T>, BackendError>
where
    F: FnOnce(&mut T),
{
    vyre_foundation::failure_domain::govern_mutex_with_reset(
        mutex,
        owner,
        state,
        class,
        reset_on_restart,
    )
    .map_err(|error| as_backend_error(&error, owner, state))
}

/// Take a read lock governed by an explicit failure domain contract.
pub fn govern_rwlock_read<'a, T>(
    rwlock: &'a RwLock<T>,
    owner: &'static str,
    state: &'static str,
    class: RecoveryClass,
) -> Result<RwLockReadGuard<'a, T>, BackendError> {
    vyre_foundation::failure_domain::govern_rwlock_read(rwlock, owner, state, class)
        .map_err(|error| as_backend_error(&error, owner, state))
}

/// Take a write lock governed by an explicit failure domain contract.
pub fn govern_rwlock_write<'a, T>(
    rwlock: &'a RwLock<T>,
    owner: &'static str,
    state: &'static str,
    class: RecoveryClass,
) -> Result<RwLockWriteGuard<'a, T>, BackendError> {
    govern_rwlock_write_with_reset(rwlock, owner, state, class, |_| {})
}

/// Take a write lock with an explicit reset action executed if restartable state is recovered.
pub fn govern_rwlock_write_with_reset<'a, T, F>(
    rwlock: &'a RwLock<T>,
    owner: &'static str,
    state: &'static str,
    class: RecoveryClass,
    reset_on_restart: F,
) -> Result<RwLockWriteGuard<'a, T>, BackendError>
where
    F: FnOnce(&mut T),
{
    vyre_foundation::failure_domain::govern_rwlock_write_with_reset(
        rwlock,
        owner,
        state,
        class,
        reset_on_restart,
    )
    .map_err(|error| as_backend_error(&error, owner, state))
}
