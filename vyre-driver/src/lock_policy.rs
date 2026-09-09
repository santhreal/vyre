//! Canonical failure domain and lock poisoning authority for driver and runtime state.
//!
//! A `Mutex` or `RwLock` is poisoned when a thread panicked while holding it,
//! leaving the guarded memory in an arbitrary intermediate state.
//!
//! Subsystems must not make ad-hoc local decisions (such as unconditionally recovering
//! with `into_inner`, panicking in place, or ignoring poison). Instead, every lock
//! belongs to an explicitly declared [`FailureDomain`](crate::lock_policy::FailureDomain):
//!
//! 1. [`FailureDomain::Transactional`](crate::lock_policy::FailureDomain::Transactional): In-flight mutation was aborted. The guarded
//!    state is discarded and a typed [`BackendError`] is reported to the caller.
//! 2. [`FailureDomain::RestartableFromCanonical`](crate::lock_policy::FailureDomain::RestartableFromCanonical): Caches, staging pools, or memoized
//!    entries that can be cleanly discarded/reset to an empty valid state and restarted.
//! 3. [`FailureDomain::DeviceContextFatal`](crate::lock_policy::FailureDomain::DeviceContextFatal): Device-bound queues, command encoders,
//!    or device handles where poison indicates corrupted GPU submission state. The
//!    device is marked lost and [`BackendError::DeviceLost`] is reported.
//! 4. [`FailureDomain::ProcessFatal`](crate::lock_policy::FailureDomain::ProcessFatal): Foreign ICD dynamic loader dispatch tables,
//!    global driver runtime init, or external C-ABI boundaries where corrupt state
//!    causes silent memory corruption or SIGSEGV in foreign frames. Process is aborted.
//! 5. [`FailureDomain::InvariantViolation`](crate::lock_policy::FailureDomain::InvariantViolation): Critical internal data structure
//!    corruption violating compiler invariants. Process is aborted with diagnostic details.

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

pub use vyre_foundation::{
    FailureDomain as SystemFailureDomain, RecoveryClass, RecoveryDisposition, TypedRecoveryError,
};

use crate::BackendError;

/// Declared failure domain and recovery contract of a lock owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FailureDomain {
    /// In-flight transaction was aborted; state discarded and error reported.
    Transactional,
    /// Cache/pool restartable from canonical input or empty state.
    RestartableFromCanonical,
    /// GPU device context corrupted; device marked lost.
    DeviceContextFatal,
    /// Unrecoverable native ICD / C-ABI memory state; process terminated immediately.
    ProcessFatal,
    /// Subsystem internal invariant violated; unrecoverable bug.
    InvariantViolation,
}

impl From<RecoveryClass> for FailureDomain {
    fn from(class: RecoveryClass) -> Self {
        match class {
            RecoveryClass::TransactionallyRecoverable => Self::Transactional,
            RecoveryClass::RestartableFromCanonicalInput => Self::RestartableFromCanonical,
            RecoveryClass::DeviceContextFatal => Self::DeviceContextFatal,
            RecoveryClass::ProcessFatal => Self::ProcessFatal,
            RecoveryClass::InvariantViolation | _ => Self::InvariantViolation,
        }
    }
}

impl From<FailureDomain> for RecoveryClass {
    fn from(domain: FailureDomain) -> Self {
        match domain {
            FailureDomain::Transactional => Self::TransactionallyRecoverable,
            FailureDomain::RestartableFromCanonical => Self::RestartableFromCanonicalInput,
            FailureDomain::DeviceContextFatal => Self::DeviceContextFatal,
            FailureDomain::ProcessFatal => Self::ProcessFatal,
            FailureDomain::InvariantViolation => Self::InvariantViolation,
        }
    }
}

/// Authoritative declaration registry mapping each known driver lock to its recovery class and domain.
#[must_use]
pub fn authoritative_driver_lock_registry() -> BTreeMap<&'static str, FailureDomain> {
    let mut map = BTreeMap::new();
    // vyre-driver
    map.insert("vyre-driver/src/launch_facts.rs:LAUNCH_MEASUREMENTS", FailureDomain::Transactional);
    map.insert("vyre-driver/src/observability.rs:EVENTS", FailureDomain::Transactional);
    map.insert("vyre-driver/src/observability.rs:LOCK", FailureDomain::Transactional);
    map.insert("vyre-driver/src/backend/resident_sequence.rs:submitted", FailureDomain::Transactional);
    map.insert("vyre-driver/src/grid_sync/resident_dispatch.rs:buffers", FailureDomain::Transactional);
    map.insert("vyre-driver/src/grid_sync/resident_dispatch.rs:freed", FailureDomain::Transactional);
    map.insert("vyre-driver/src/pipeline/cache.rs:pending_flushes", FailureDomain::Transactional);
    // vyre-driver-wgpu
    map.insert("vyre-driver-wgpu/src/lib.rs:shape_history", FailureDomain::Transactional);
    map.insert("vyre-driver-wgpu/src/strict_float.rs:VERDICTS", FailureDomain::RestartableFromCanonical);
    map.insert("vyre-driver-wgpu/src/buffer/bind_group_cache/mod.rs:cache", FailureDomain::RestartableFromCanonical);
    map.insert("vyre-driver-wgpu/src/buffer/staging/mod.rs:inner", FailureDomain::RestartableFromCanonical);
    map.insert("vyre-driver-wgpu/src/pipeline/disk_cache_entries.rs:TEST_DISK_PIPELINE_CACHE_ROOT", FailureDomain::Transactional);
    map.insert("vyre-driver-wgpu/src/pipeline/disk_cache/io.rs:PENDING_DURABLE_CACHE_FILES", FailureDomain::Transactional);
    map.insert("vyre-driver-wgpu/src/runtime/prerecorded.rs:cb", FailureDomain::DeviceContextFatal);
    map.insert("vyre-driver-wgpu/src/runtime/device/acquire.rs:LOADER_STARTUP", FailureDomain::ProcessFatal);
    map
}

/// End the process, naming the owner and the state its poisoned lock guards.
///
/// `owner` names the subsystem holding the lock and `state` names what the
/// lock excludes concurrent access to: `"the device factory"` and `"the graphics loader dispatch
/// table"`, not `"LOADER_STARTUP"`.
pub fn process_fatal_poison(owner: &str, state: &str) -> ! {
    eprintln!(
        "vyre: {owner} holds a poisoned lock over {state}. A thread panicked while that lock \
         was held, so the state behind it is half written and no owner in this process can \
         rebuild it. Fix: report the earlier panic. The process ends here rather than \
         faulting inside the code that state belongs to."
    );
    std::process::abort()
}

/// Take a guard over process-owned state, or report the poison as recoverable.
///
/// The guarded value is discarded with the guard: a caller that receives the
/// error rebuilds it rather than reading what the panic left.
pub fn recoverable_poison<T>(
    result: Result<T, PoisonError<T>>,
) -> Result<T, BackendError> {
    result.map_err(BackendError::poisoned_lock)
}

/// Take a mutex guard governed by an explicit failure domain contract.
pub fn govern_mutex<'a, T>(
    mutex: &'a Mutex<T>,
    owner: &'static str,
    state: &'static str,
    domain: FailureDomain,
) -> Result<MutexGuard<'a, T>, BackendError> {
    govern_mutex_with_reset(mutex, owner, state, domain, |_| {})
}

/// Take a mutex guard with an explicit reset action executed if restartable state is recovered.
pub fn govern_mutex_with_reset<'a, T, F>(
    mutex: &'a Mutex<T>,
    owner: &'static str,
    state: &'static str,
    domain: FailureDomain,
    reset_on_restart: F,
) -> Result<MutexGuard<'a, T>, BackendError>
where
    F: FnOnce(&mut T),
{
    match mutex.lock() {
        Ok(guard) => Ok(guard),
        Err(poison) => match domain {
            FailureDomain::ProcessFatal | FailureDomain::InvariantViolation => {
                process_fatal_poison(owner, state);
            }
            FailureDomain::Transactional => Err(BackendError::poisoned_lock(poison)),
            FailureDomain::DeviceContextFatal => {
                Err(BackendError::DeviceLost {
                    backend: owner.to_string(),
                    device: state.to_string(),
                    generation: 0,
                    message: format!("lock over `{state}` in `{owner}` was poisoned by a previous panic"),
                })
            }
            FailureDomain::RestartableFromCanonical => {
                let mut guard = poison.into_inner();
                reset_on_restart(&mut guard);
                Ok(guard)
            }
        },
    }
}

/// Take a read lock governed by an explicit failure domain contract.
pub fn govern_rwlock_read<'a, T>(
    rwlock: &'a RwLock<T>,
    owner: &'static str,
    state: &'static str,
    domain: FailureDomain,
) -> Result<RwLockReadGuard<'a, T>, BackendError> {
    match rwlock.read() {
        Ok(guard) => Ok(guard),
        Err(poison) => match domain {
            FailureDomain::ProcessFatal | FailureDomain::InvariantViolation => {
                process_fatal_poison(owner, state);
            }
            FailureDomain::Transactional => Err(BackendError::poisoned_lock(poison)),
            FailureDomain::DeviceContextFatal => {
                Err(BackendError::DeviceLost {
                    backend: owner.to_string(),
                    device: state.to_string(),
                    generation: 0,
                    message: format!("lock over `{state}` in `{owner}` was poisoned by a previous panic"),
                })
            }
            FailureDomain::RestartableFromCanonical => {
                // Read lock cannot mutate to reset; return poisoned error to force write-side recovery
                Err(BackendError::poisoned_lock(poison))
            }
        },
    }
}

/// Take a write lock governed by an explicit failure domain contract.
pub fn govern_rwlock_write<'a, T>(
    rwlock: &'a RwLock<T>,
    owner: &'static str,
    state: &'static str,
    domain: FailureDomain,
) -> Result<RwLockWriteGuard<'a, T>, BackendError> {
    govern_rwlock_write_with_reset(rwlock, owner, state, domain, |_| {})
}

/// Take a write lock with an explicit reset action executed if restartable state is recovered.
pub fn govern_rwlock_write_with_reset<'a, T, F>(
    rwlock: &'a RwLock<T>,
    owner: &'static str,
    state: &'static str,
    domain: FailureDomain,
    reset_on_restart: F,
) -> Result<RwLockWriteGuard<'a, T>, BackendError>
where
    F: FnOnce(&mut T),
{
    match rwlock.write() {
        Ok(guard) => Ok(guard),
        Err(poison) => match domain {
            FailureDomain::ProcessFatal | FailureDomain::InvariantViolation => {
                process_fatal_poison(owner, state);
            }
            FailureDomain::Transactional => Err(BackendError::poisoned_lock(poison)),
            FailureDomain::DeviceContextFatal => {
                Err(BackendError::DeviceLost {
                    backend: owner.to_string(),
                    device: state.to_string(),
                    generation: 0,
                    message: format!("lock over `{state}` in `{owner}` was poisoned by a previous panic"),
                })
            }
            FailureDomain::RestartableFromCanonical => {
                let mut guard = poison.into_inner();
                reset_on_restart(&mut guard);
                Ok(guard)
            }
        },
    }
}
