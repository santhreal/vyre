//! Explicit failure domains and recovery classes for mutable state owners.
//!
//! WHY: closes the class "uncoordinated failure policies where poisoned locks panic,
//! recover partially mutated state, or repeat non-idempotent side effects upon retry".
//! Defines explicit failure domains, recovery classes, and typed state transitions.

use core::fmt;
use std::string::String;
use std::sync::{
    Condvar, Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard,
};
use std::time::Duration;
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

/// Implemented by every owner of mutable shared state, stating the blast
/// radius of a fault under its locks and how that fault is remediated.
///
/// A poisoned lock means a thread panicked while holding it, so the guarded
/// value may be half written. Answering that at whichever call site observed
/// it first gives one owner as many recoveries as it has entry points. The
/// impl is the single record instead: a `Mutex`, `RwLock`, `DashMap` or
/// `AtomicGuardedState` field commits its struct to one domain and one class,
/// and the `lock-poison-policy` gate rejects an owning struct that has no
/// impl.
///
/// The trait is defined here rather than in the runtime because the driver
/// owns device-bound state of its own and sits below the runtime, so a home
/// above either of them is the only one both can reach.
pub trait StateOwnerRecovery {
    /// Failure domain this state owner belongs to.
    fn failure_domain(&self) -> FailureDomain;
    /// Recovery class defining how failures are remediated.
    fn recovery_class(&self) -> RecoveryClass;
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
        "vyre: {owner} holds a poisoned lock over {state}. A thread panicked while that lock was \
         held, and this state has no recovery that leaves the process sound, so the process ends \
         here rather than continuing on a value nothing can vouch for. \
         Fix: report the earlier panic."
    );
    std::process::abort();
}

/// Panic on a poisoned lock over state whose partial mutation invalidates every
/// answer derived from it.
///
/// WHY: an invariant violation is scoped to the unit of work that reads the
/// state, not to the whole process. Unwinding lets a supervised caller report
/// which unit failed and reclaim what that unit held, while
/// [`process_fatal_poison`] takes down every unrelated unit in the same
/// process. The two are separate recovery classes because they have separate
/// blast radii, and a policy that aborts for both certifies neither.
///
/// `owner` names the subsystem holding the lock and `state` names what the
/// lock excludes concurrent access to.
///
/// # Panics
///
/// Always. The message names `state` and `owner`.
pub fn invariant_violation_poison(owner: &str, state: &str) -> ! {
    panic!(
        "vyre: {state} was poisoned in {owner}. A thread panicked while that lock was held, so \
         the state behind it is half written and reading it would publish a corrupt value as \
         truth. Fix: report the earlier panic."
    );
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

/// Take a write guard over state that is rebuildable from canonical input.
///
/// WHY: the `RwLock` counterpart of [`govern_mutex_restartable`], so a
/// restartable owner picks its lock kind without picking a different recovery.
/// A poisoned lock here means a panic left the guarded state partly written.
/// `reset_on_restart` returns it to an empty valid state, the poison flag is
/// cleared so the next acquisition is an ordinary one, and the guard is handed
/// back. Every entry discarded this way is derivable again from the input the
/// owner already holds, which is what makes discarding it correct rather than
/// lossy.
pub fn govern_rwlock_write_restartable<'a, T, F>(
    rwlock: &'a RwLock<T>,
    owner: &'static str,
    state: &'static str,
    reset_on_restart: F,
) -> RwLockWriteGuard<'a, T>
where
    F: FnOnce(&mut T),
{
    match rwlock.write() {
        Ok(guard) => guard,
        Err(poison) => {
            eprintln!(
                "vyre: {owner} recovered a poisoned lock over {state} by discarding it. A thread \
                 panicked while that lock was held, so the state behind it is half written and is \
                 rebuilt from canonical input on demand. Fix: report the earlier panic."
            );
            rwlock.clear_poison();
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

/// Report that irreplaceable state was read past a panic.
///
/// WHY: discarding this state loses the only record of something the process
/// already did: an external handle the device still holds, a committed
/// idempotency key, a supervised worker. Discarding it leaks the handle or lets
/// a retry repeat a side effect, and both are worse than reading past a panic.
/// One insertion writes one entry, so a panic leaves entries whole and leaves
/// only the sequence across them incomplete, which the caller's own state
/// machine records.
fn report_reclaimed(owner: &str, state: &str) {
    eprintln!(
        "vyre: {owner} recovered a poisoned lock over {state} and kept it. A thread panicked \
         while that lock was held. That state names resources this process still owns, so \
         discarding it would leak them. Fix: report the earlier panic."
    );
}

/// Whether an acquisition was the one that read past a panic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reclaimed {
    /// The lock was not poisoned.
    No,
    /// This acquisition found the poison flag set and cleared it. Because the
    /// flag is cleared under the same acquisition, exactly one acquisition ever
    /// observes this for a given panic, which is what lets an owner run a
    /// one-time transition here without repeating it.
    Once,
}

/// Take a mutex guard over state no owner can rebuild, reporting whether this
/// acquisition is the one that recovered it.
///
/// The poison flag is cleared here rather than by the caller, so one panic
/// costs one recovery instead of one per acquisition for the life of the
/// process, and clearing the flag stays with the record of who accepted the
/// state and why. The state is kept because discarding it loses the only
/// record of something the process already did, such as an external handle the
/// device still holds or a committed idempotency key, which leaks the handle
/// or lets a retry repeat a side effect.
pub fn reclaim_poisoned_mutex_observed<'a, T>(
    mutex: &'a Mutex<T>,
    owner: &str,
    state: &str,
) -> (MutexGuard<'a, T>, Reclaimed) {
    match mutex.lock() {
        Ok(guard) => (guard, Reclaimed::No),
        Err(poison) => {
            report_reclaimed(owner, state);
            mutex.clear_poison();
            (poison.into_inner(), Reclaimed::Once)
        }
    }
}

/// Take a mutex guard over state no owner can rebuild.
///
/// [`reclaim_poisoned_mutex_observed`] for an owner that runs a one-time
/// transition when the recovery happens.
pub fn reclaim_poisoned_mutex<'a, T>(
    mutex: &'a Mutex<T>,
    owner: &str,
    state: &str,
) -> MutexGuard<'a, T> {
    reclaim_poisoned_mutex_observed(mutex, owner, state).0
}

/// Take a read guard over state no owner can rebuild.
///
/// The `RwLock` read counterpart of [`reclaim_poisoned_mutex`], so an owner
/// picks its lock kind without picking a different recovery.
pub fn reclaim_poisoned_read<'a, T>(
    rwlock: &'a RwLock<T>,
    owner: &str,
    state: &str,
) -> RwLockReadGuard<'a, T> {
    match rwlock.read() {
        Ok(guard) => guard,
        Err(poison) => {
            report_reclaimed(owner, state);
            rwlock.clear_poison();
            poison.into_inner()
        }
    }
}

/// Take a write guard over state no owner can rebuild.
///
/// The `RwLock` write counterpart of [`reclaim_poisoned_mutex`], so an owner
/// picks its lock kind without picking a different recovery.
pub fn reclaim_poisoned_write<'a, T>(
    rwlock: &'a RwLock<T>,
    owner: &str,
    state: &str,
) -> RwLockWriteGuard<'a, T> {
    match rwlock.write() {
        Ok(guard) => guard,
        Err(poison) => {
            report_reclaimed(owner, state);
            rwlock.clear_poison();
            poison.into_inner()
        }
    }
}

/// Wait on a condition variable over state no owner can rebuild.
///
/// A wait releases the mutex and takes it again, so it observes a panic in
/// another waiter exactly as an acquisition does. `mutex` is the lock `guard`
/// came from, which is what the flag is cleared on.
pub fn reclaim_poisoned_condvar_wait<'a, T>(
    condvar: &Condvar,
    mutex: &'a Mutex<T>,
    guard: MutexGuard<'a, T>,
    owner: &str,
    state: &str,
) -> MutexGuard<'a, T> {
    match condvar.wait(guard) {
        Ok(reacquired) => reacquired,
        Err(poison) => {
            report_reclaimed(owner, state);
            mutex.clear_poison();
            poison.into_inner()
        }
    }
}

/// Wait on a condition variable with a deadline over state no owner can rebuild.
///
/// WHY: an unbounded wait turns one stalled holder into a stalled caller with
/// no report. The returned flag states whether the wait ended on `timeout`
/// rather than on a notification, so a stand-down terminates and the caller
/// names the bound it waited out instead of blocking forever.
///
/// A wait releases the mutex and takes it again, so it observes a panic in
/// another waiter exactly as an acquisition does. `mutex` is the lock `guard`
/// came from, which is what the flag is cleared on.
pub fn reclaim_poisoned_condvar_wait_timeout<'a, T>(
    condvar: &Condvar,
    mutex: &'a Mutex<T>,
    guard: MutexGuard<'a, T>,
    timeout: Duration,
    owner: &str,
    state: &str,
) -> (MutexGuard<'a, T>, bool) {
    match condvar.wait_timeout(guard, timeout) {
        Ok((reacquired, result)) => (reacquired, result.timed_out()),
        Err(poison) => {
            report_reclaimed(owner, state);
            mutex.clear_poison();
            let (reacquired, result) = poison.into_inner();
            (reacquired, result.timed_out())
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
            RecoveryClass::ProcessFatal => process_fatal_poison(owner, state),
            RecoveryClass::InvariantViolation => invariant_violation_poison(owner, state),
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
            RecoveryClass::ProcessFatal => process_fatal_poison(owner, state),
            RecoveryClass::InvariantViolation => invariant_violation_poison(owner, state),
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
            RecoveryClass::ProcessFatal => process_fatal_poison(owner, state),
            RecoveryClass::InvariantViolation => invariant_violation_poison(owner, state),
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
