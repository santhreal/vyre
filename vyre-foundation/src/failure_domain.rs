//! Explicit failure domains and recovery classes for mutable state owners.
//!
//! WHY: closes the class "uncoordinated failure policies where poisoned locks panic,
//! recover partially mutated state, or repeat non-idempotent side effects upon retry".
//! Defines explicit failure domains, recovery classes, and typed state transitions.

use core::fmt;
use std::string::String;
use std::sync::{Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};
/// Explicit failure domain identifying which subsystem boundary failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FailureDomain {
    /// In-memory state (caches, registries, heaps, transient buffers).
    MemoryState,
    /// Hardware device context (GPU queue, device loss, VRAM allocator).
    DeviceContext,
    /// Persistent disk journal or cache storage (disk full, I/O failure).
    DiskJournal,
    /// Supervised worker process (crash, OOM, timeout).
    WorkerProcess,
    /// Session lifecycle (connection reset, client disconnect).
    SessionLifecycle,
    /// Network and transport boundary (socket reset, framing error).
    NetworkTransport,
}

impl FailureDomain {
    /// All failure domains.
    pub const ALL: &'static [Self] = {
        const ALL: &[FailureDomain] = &[
            FailureDomain::MemoryState,
            FailureDomain::DeviceContext,
            FailureDomain::DiskJournal,
            FailureDomain::WorkerProcess,
            FailureDomain::SessionLifecycle,
            FailureDomain::NetworkTransport,
        ];
        let mut i = 0;
        while i < ALL.len() {
            match ALL[i] {
                FailureDomain::MemoryState
                | FailureDomain::DeviceContext
                | FailureDomain::DiskJournal
                | FailureDomain::WorkerProcess
                | FailureDomain::SessionLifecycle
                | FailureDomain::NetworkTransport => {}
            }
            i += 1;
        }
        ALL
    };

    /// Stable string identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MemoryState => "memory_state",
            Self::DeviceContext => "device_context",
            Self::DiskJournal => "disk_journal",
            Self::WorkerProcess => "worker_process",
            Self::SessionLifecycle => "session_lifecycle",
            Self::NetworkTransport => "network_transport",
        }
    }
}

impl fmt::Display for FailureDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Recovery class defining how a failure can be safely remediated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RecoveryClass {
    /// Side effect is rolled back via prepare/commit journal and idempotency key.
    TransactionallyRecoverable,
    /// State can be rebuilt completely from immutable canonical persisted input.
    RestartableFromCanonicalInput,
    /// Hardware device context poisoned; requires releasing handles and acquiring fresh device.
    DeviceContextFatal,
    /// Worker process died or leaked; requires terminating and respawning under supervised budget.
    ProcessFatal,
    /// Compiler or engine internal invariant violated; unrecoverable defect.
    InvariantViolation,
}

impl RecoveryClass {
    /// All recovery classes.
    pub const ALL: &'static [Self] = {
        const ALL: &[RecoveryClass] = &[
            RecoveryClass::TransactionallyRecoverable,
            RecoveryClass::RestartableFromCanonicalInput,
            RecoveryClass::DeviceContextFatal,
            RecoveryClass::ProcessFatal,
            RecoveryClass::InvariantViolation,
        ];
        let mut i = 0;
        while i < ALL.len() {
            match ALL[i] {
                RecoveryClass::TransactionallyRecoverable
                | RecoveryClass::RestartableFromCanonicalInput
                | RecoveryClass::DeviceContextFatal
                | RecoveryClass::ProcessFatal
                | RecoveryClass::InvariantViolation => {}
            }
            i += 1;
        }
        ALL
    };

    /// Stable string identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TransactionallyRecoverable => "transactionally_recoverable",
            Self::RestartableFromCanonicalInput => "restartable_from_canonical_input",
            Self::DeviceContextFatal => "device_context_fatal",
            Self::ProcessFatal => "process_fatal",
            Self::InvariantViolation => "invariant_violation",
        }
    }

    /// Whether this recovery class allows retry without process/device restart.
    #[must_use]
    pub const fn allows_inline_recovery(self) -> bool {
        match self {
            Self::TransactionallyRecoverable | Self::RestartableFromCanonicalInput => true,
            Self::DeviceContextFatal | Self::ProcessFatal | Self::InvariantViolation => false,
        }
    }
}

impl fmt::Display for RecoveryClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Actionable remediation disposition for a failure event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryDisposition {
    /// Safe to retry immediately using the same idempotency key.
    CanRetryImmediate,
    /// Rebuilding state from immutable input is required before subsequent operations.
    RequiresRebuild,
    /// Must discard existing device context and acquire a new device.
    RequiresDeviceReacquisition,
    /// Worker process must be restarted under supervised restart budget.
    RequiresProcessRestart,
    /// Fatal error; cannot be recovered.
    Fatal,
}

/// Structured error carrying failure domain, recovery class, and documented corrective action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypedRecoveryError {
    /// Owning failure domain.
    pub domain: FailureDomain,
    /// Recovery class.
    pub recovery_class: RecoveryClass,
    /// Actionable disposition.
    pub disposition: RecoveryDisposition,
    /// Detailed diagnostic reason.
    pub reason: String,
    /// Documented corrective action for caller.
    pub fix: String,
}

impl fmt::Display for TypedRecoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] ({}) {}. {}",
            self.domain, self.recovery_class, self.reason, self.fix
        )
    }
}

impl core::error::Error for TypedRecoveryError {}

impl TypedRecoveryError {
    /// Create a new typed recovery error with explicit failure domain and recovery class.
    #[must_use]
    pub fn new(
        domain: FailureDomain,
        recovery_class: RecoveryClass,
        disposition: RecoveryDisposition,
        reason: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self {
            domain,
            recovery_class,
            disposition,
            reason: reason.into(),
            fix: fix.into(),
        }
    }
}

/// End the process, reporting the owner and the state its poisoned lock guards.
///
/// `owner` names the subsystem holding the lock and `state` names what the
/// lock excludes concurrent access to.
pub fn process_fatal_poison(owner: &str, state: &str) -> ! {
    eprintln!(
        "FATAL: invariant violation: lock over `{state}` in `{owner}` was poisoned by a prior thread panic. \
         Fix: treat the earlier panic as the root defect; compiler substrate invariants were violated."
    );
    std::process::abort();
}

/// Take a mutex guard over state that is rebuildable from canonical input.
///
/// A poisoned lock here means a panic left the guarded state partly written.
/// `reset_on_restart` returns it to an empty valid state, the poison flag is
/// cleared so the next acquisition is an ordinary one, and the guard is handed
/// back. Every entry discarded this way is derivable again from the input the
/// owner already holds, which is what makes discarding it correct rather than
/// lossy.
pub fn govern_mutex_restartable<'a, T, F>(
    mutex: &'a Mutex<T>,
    owner: &'static str,
    state: &'static str,
    reset_on_restart: F,
) -> MutexGuard<'a, T>
where
    F: FnOnce(&mut T),
{
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poison) => {
            eprintln!(
                "vyre: {owner} recovered a poisoned lock over {state} by discarding it. A thread \
                 panicked while that lock was held, so the state behind it is half written and is \
                 rebuilt from canonical input on demand. Fix: report the earlier panic."
            );
            mutex.clear_poison();
            let mut guard = poison.into_inner();
            reset_on_restart(&mut guard);
            guard
        }
    }
}

/// Take poisoned state during teardown so the handles it holds are released.
///
/// A destructor is the last owner of whatever a panic left behind. Refusing that
/// state leaks every handle inside it, and a leaked device resource outlives the
/// process that could have freed it. Teardown reads the state to release those
/// handles and never publishes it to a caller, so a half-written value cannot be
/// observed as if it were whole.
pub fn reclaim_poisoned_for_teardown<T>(
    state: Result<T, PoisonError<T>>,
    owner: &str,
    guarded: &str,
) -> T {
    match state {
        Ok(value) => value,
        Err(poison) => {
            eprintln!(
                "vyre: {owner} is tearing down a poisoned lock over {guarded}. A thread panicked \
                 while that lock was held. Teardown releases the handles it still holds rather \
                 than leaking them. Fix: report the earlier panic."
            );
            poison.into_inner()
        }
    }
}

/// Take a mutex guard governed by an explicit failure domain contract.
pub fn govern_mutex<'a, T>(
    mutex: &'a Mutex<T>,
    owner: &'static str,
    state: &'static str,
    class: RecoveryClass,
) -> Result<MutexGuard<'a, T>, TypedRecoveryError> {
    govern_mutex_with_reset(mutex, owner, state, class, |_| {})
}

/// Take a mutex guard with an explicit reset action executed if restartable state is recovered.
pub fn govern_mutex_with_reset<'a, T, F>(
    mutex: &'a Mutex<T>,
    owner: &'static str,
    state: &'static str,
    class: RecoveryClass,
    reset_on_restart: F,
) -> Result<MutexGuard<'a, T>, TypedRecoveryError>
where
    F: FnOnce(&mut T),
{
    match mutex.lock() {
        Ok(guard) => Ok(guard),
        Err(poison) => match class {
            RecoveryClass::ProcessFatal | RecoveryClass::InvariantViolation => {
                process_fatal_poison(owner, state);
            }
            RecoveryClass::TransactionallyRecoverable => Err(TypedRecoveryError::new(
                FailureDomain::MemoryState,
                class,
                RecoveryDisposition::RequiresRebuild,
                format!("lock over `{state}` in `{owner}` was poisoned"),
                format!("Fix: rebuild `{owner}` state machine from canonical input."),
            )),
            RecoveryClass::DeviceContextFatal => Err(TypedRecoveryError::new(
                FailureDomain::DeviceContext,
                class,
                RecoveryDisposition::RequiresDeviceReacquisition,
                format!("device-bound lock over `{state}` in `{owner}` was poisoned"),
                format!("Fix: drop `{owner}` and reacquire a fresh device context."),
            )),
            RecoveryClass::RestartableFromCanonicalInput => {
                mutex.clear_poison();
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
    class: RecoveryClass,
) -> Result<RwLockReadGuard<'a, T>, TypedRecoveryError> {
    match rwlock.read() {
        Ok(guard) => Ok(guard),
        Err(poison) => match class {
            RecoveryClass::ProcessFatal | RecoveryClass::InvariantViolation => {
                process_fatal_poison(owner, state);
            }
            RecoveryClass::TransactionallyRecoverable
            | RecoveryClass::RestartableFromCanonicalInput => Err(TypedRecoveryError::new(
                FailureDomain::MemoryState,
                class,
                RecoveryDisposition::RequiresRebuild,
                format!("read lock over `{state}` in `{owner}` was poisoned: {poison}"),
                format!("Fix: rebuild `{owner}` state machine from canonical input."),
            )),
            RecoveryClass::DeviceContextFatal => Err(TypedRecoveryError::new(
                FailureDomain::DeviceContext,
                class,
                RecoveryDisposition::RequiresDeviceReacquisition,
                format!(
                    "device-bound read lock over `{state}` in `{owner}` was poisoned: {poison}"
                ),
                format!("Fix: drop `{owner}` and reacquire a fresh device context."),
            )),
        },
    }
}

/// Take a write lock governed by an explicit failure domain contract.
pub fn govern_rwlock_write<'a, T>(
    rwlock: &'a RwLock<T>,
    owner: &'static str,
    state: &'static str,
    class: RecoveryClass,
) -> Result<RwLockWriteGuard<'a, T>, TypedRecoveryError> {
    govern_rwlock_write_with_reset(rwlock, owner, state, class, |_| {})
}

/// Take a write lock with an explicit reset action executed if restartable state is recovered.
pub fn govern_rwlock_write_with_reset<'a, T, F>(
    rwlock: &'a RwLock<T>,
    owner: &'static str,
    state: &'static str,
    class: RecoveryClass,
    reset_on_restart: F,
) -> Result<RwLockWriteGuard<'a, T>, TypedRecoveryError>
where
    F: FnOnce(&mut T),
{
    match rwlock.write() {
        Ok(guard) => Ok(guard),
        Err(poison) => match class {
            RecoveryClass::ProcessFatal | RecoveryClass::InvariantViolation => {
                process_fatal_poison(owner, state);
            }
            RecoveryClass::TransactionallyRecoverable => Err(TypedRecoveryError::new(
                FailureDomain::MemoryState,
                class,
                RecoveryDisposition::RequiresRebuild,
                format!("write lock over `{state}` in `{owner}` was poisoned: {poison}"),
                format!("Fix: rebuild `{owner}` state machine from canonical input."),
            )),
            RecoveryClass::DeviceContextFatal => Err(TypedRecoveryError::new(
                FailureDomain::DeviceContext,
                class,
                RecoveryDisposition::RequiresDeviceReacquisition,
                format!(
                    "device-bound write lock over `{state}` in `{owner}` was poisoned: {poison}"
                ),
                format!("Fix: drop `{owner}` and reacquire a fresh device context."),
            )),
            RecoveryClass::RestartableFromCanonicalInput => {
                rwlock.clear_poison();
                let mut guard = poison.into_inner();
                reset_on_restart(&mut guard);
                Ok(guard)
            }
        },
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_domain_all_contains_every_variant() {
        for domain in FailureDomain::ALL {
            match domain {
                FailureDomain::MemoryState
                | FailureDomain::DeviceContext
                | FailureDomain::DiskJournal
                | FailureDomain::WorkerProcess
                | FailureDomain::SessionLifecycle
                | FailureDomain::NetworkTransport => {}
            }
        }
        assert_eq!(FailureDomain::ALL.len(), 6);
    }

    #[test]
    fn recovery_class_all_contains_every_variant() {
        for class in RecoveryClass::ALL {
            match class {
                RecoveryClass::TransactionallyRecoverable
                | RecoveryClass::RestartableFromCanonicalInput
                | RecoveryClass::DeviceContextFatal
                | RecoveryClass::ProcessFatal
                | RecoveryClass::InvariantViolation => {}
            }
        }
        assert_eq!(RecoveryClass::ALL.len(), 5);
    }
}
