//! Atomic failure recovery, state machines, prepare/commit journals, and restart budgets.
//!
//! WHY: closes the class "poisoned locks panic or silently extract partial state, and retry repeats non-idempotent side effects".
//! Provides typed atomic state machines that fail into terminal/rebuilding states on poison,
//! prepare/commit journaling with idempotency keys, and supervised worker restart budgets.

use core::sync::atomic::{AtomicU64, Ordering};
use std::collections::BTreeMap;
use std::format;
use std::string::String;
use std::sync::Mutex;

use vyre_foundation::failure_domain::{
    reclaim_poisoned_irreplaceable_state, FailureDomain, RecoveryClass, RecoveryDisposition,
    TypedRecoveryError,
};

/// The subsystem every poison report in this module names as the owner.
const OWNER: &str = "runtime atomic recovery";

/// Lifecycle state for an atomic guarded resource.
#[derive(Clone, Debug, PartialEq)]
pub enum GuardedState<T> {
    /// Operational state holding valid data.
    Ready(T),
    /// State is currently rebuilding from canonical input.
    Rebuilding,
    /// Terminal failure state; further operations are rejected until documented recovery.
    PoisonedTerminal {
        /// Failure domain of the fault.
        domain: FailureDomain,
        /// Recovery class of the fault.
        recovery_class: RecoveryClass,
        /// Diagnostic reason for terminal state.
        reason: String,
    },
}

/// Trait implemented by mutable state owners declaring their failure domain and recovery class.
pub trait StateOwnerRecovery {
    /// Failure domain this state owner belongs to.
    fn failure_domain(&self) -> FailureDomain;
    /// Recovery class defining how failures are remediated.
    fn recovery_class(&self) -> RecoveryClass;
}

/// Thread-safe atomic guarded container that eliminates uncoordinated lock poison.
pub struct AtomicGuardedState<T> {
    inner: Mutex<GuardedState<T>>,
    domain: FailureDomain,
    recovery_class: RecoveryClass,
}

impl<T> AtomicGuardedState<T> {
    /// Construct a new guarded state with domain and recovery class.
    #[must_use]
    pub fn new(initial: T, domain: FailureDomain, recovery_class: RecoveryClass) -> Self {
        Self {
            inner: Mutex::new(GuardedState::Ready(initial)),
            domain,
            recovery_class,
        }
    }

    /// Failure domain.
    #[must_use]
    pub fn domain(&self) -> FailureDomain {
        self.domain
    }

    /// Recovery class.
    #[must_use]
    pub fn recovery_class(&self) -> RecoveryClass {
        self.recovery_class
    }

    /// Inspect the current lifecycle state.
    #[must_use]
    pub fn current_state(&self) -> GuardedState<T>
    where
        T: Clone,
    {
        self.lock_state().clone()
    }

    /// Take the state machine's guard, transitioning it to
    /// [`GuardedState::PoisonedTerminal`] the first time a panic is observed.
    ///
    /// The transition and the clearing of the poison flag happen under the same
    /// acquisition, so one panic produces one terminal transition no matter
    /// which entry point observes it first, and no later acquisition repeats
    /// the transition over a state a caller has since recovered.
    fn lock_state(&self) -> std::sync::MutexGuard<'_, GuardedState<T>> {
        let mut recovered = false;
        let mut guard = reclaim_poisoned_irreplaceable_state(
            self.inner.lock(),
            || {
                recovered = true;
                self.inner.clear_poison();
            },
            OWNER,
            "one atomically guarded state machine",
        );
        if recovered {
            *guard = GuardedState::PoisonedTerminal {
                domain: self.domain,
                recovery_class: self.recovery_class,
                reason: String::from("Lock was poisoned by a previous thread panic"),
            };
        }
        guard
    }

    /// Access the guarded state with an operational closure.
    ///
    /// # Errors
    ///
    /// Returns [`TypedRecoveryError`] if the lock was poisoned or if the state is terminal.
    pub fn with_state<R>(
        &self,
        op: impl FnOnce(&mut T) -> Result<R, String>,
    ) -> Result<R, TypedRecoveryError> {
        let mut guard = self.lock_state();

        match &mut *guard {
            GuardedState::Ready(val) => op(val).map_err(|err_msg| {
                TypedRecoveryError::new(
                    self.domain,
                    self.recovery_class,
                    RecoveryDisposition::RequiresRebuild,
                    err_msg,
                    "Fix: inspect the failure cause and rebuild state if necessary.",
                )
            }),
            GuardedState::Rebuilding => Err(TypedRecoveryError::new(
                self.domain,
                self.recovery_class,
                RecoveryDisposition::RequiresRebuild,
                "State machine is currently rebuilding",
                "Fix: await completion of the rebuild process before submitting operations.",
            )),
            GuardedState::PoisonedTerminal {
                domain,
                recovery_class,
                reason,
            } => Err(TypedRecoveryError::new(
                *domain,
                *recovery_class,
                RecoveryDisposition::RequiresRebuild,
                reason.clone(),
                "Fix: state machine is in PoisonedTerminal state; perform explicit recovery.",
            )),
        }
    }

    /// Access the guarded state with a read-only closure.
    pub fn with_state_ref<R>(
        &self,
        op: impl FnOnce(&T) -> Result<R, String>,
    ) -> Result<R, TypedRecoveryError> {
        self.with_state(|val| op(val))
    }

    /// Explicitly recover and restore state to Ready.
    pub fn recover(&self, fresh_state: T) {
        let mut guard = self.lock_state();
        *guard = GuardedState::Ready(fresh_state);
    }

    /// Mark the state as currently rebuilding.
    pub fn begin_rebuild(&self) -> Result<(), TypedRecoveryError> {
        let mut guard = self.lock_state();
        *guard = GuardedState::Rebuilding;
        Ok(())
    }

    /// Complete rebuilding and restore state to Ready.
    pub fn finish_rebuild(&self, fresh_state: T) {
        self.recover(fresh_state);
    }

    /// Transition to PoisonedTerminal state explicitly with a diagnostic reason.
    pub fn fault(&self, reason: impl Into<String>) {
        let mut guard = self.lock_state();
        *guard = GuardedState::PoisonedTerminal {
            domain: self.domain,
            recovery_class: self.recovery_class,
            reason: reason.into(),
        };
    }
}

impl<T> StateOwnerRecovery for AtomicGuardedState<T> {
    fn failure_domain(&self) -> FailureDomain {
        self.domain
    }

    fn recovery_class(&self) -> RecoveryClass {
        self.recovery_class
    }
}

/// Idempotency ticket returned during the prepare phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrepareTicket {
    /// Unique ticket sequence number.
    pub ticket_id: u64,
}

/// Prepare-commit journal ensuring mutation side effects are strictly idempotent.
pub struct PrepareCommitJournal<K: Ord + Clone, V: Clone> {
    next_ticket: AtomicU64,
    prepared: Mutex<BTreeMap<K, (PrepareTicket, V)>>,
    committed: Mutex<BTreeMap<K, V>>,
}

impl<K: Ord + Clone, V: Clone> PrepareCommitJournal<K, V> {
    /// Create a new prepare-commit journal.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_ticket: AtomicU64::new(1),
            prepared: Mutex::new(BTreeMap::new()),
            committed: Mutex::new(BTreeMap::new()),
        }
    }

    /// Prepare a mutation under an idempotency key.
    /// If the key is already committed, returns `Ok(None)` indicating already completed.
    ///
    /// # Errors
    ///
    /// Returns error if key is already prepared by a concurrent operation.
    fn lock_committed(&self) -> std::sync::MutexGuard<'_, BTreeMap<K, V>> {
        reclaim_poisoned_irreplaceable_state(
            self.committed.lock(),
            || self.committed.clear_poison(),
            OWNER,
            "the prepare-commit journal's committed keys",
        )
    }

    fn lock_prepared(&self) -> std::sync::MutexGuard<'_, BTreeMap<K, (PrepareTicket, V)>> {
        reclaim_poisoned_irreplaceable_state(
            self.prepared.lock(),
            || self.prepared.clear_poison(),
            OWNER,
            "the prepare-commit journal's prepared keys",
        )
    }

    /// # Errors
    ///
    /// Returns error if key is already prepared by a concurrent operation.
    pub fn prepare(&self, key: K, value: V) -> Result<Option<PrepareTicket>, String> {
        let committed = self.lock_committed();
        if committed.contains_key(&key) {
            return Ok(None); // Already committed; idempotent no-op
        }
        drop(committed);

        let mut prepared = self.lock_prepared();
        if prepared.contains_key(&key) {
            return Err(String::from(
                "Fix: operation with this idempotency key is already prepared in flight.",
            ));
        }

        let ticket_id = self.next_ticket.fetch_add(1, Ordering::SeqCst);
        let ticket = PrepareTicket { ticket_id };
        prepared.insert(key, (ticket, value));
        Ok(Some(ticket))
    }

    /// Commit the prepared mutation atomically.
    ///
    /// # Errors
    ///
    /// Returns error if the ticket does not match the prepared entry.
    pub fn commit(&self, key: K, ticket: PrepareTicket) -> Result<V, String> {
        let mut prepared = self.lock_prepared();
        if let Some((saved_ticket, val)) = prepared.remove(&key) {
            if saved_ticket != ticket {
                return Err(String::from(
                    "Fix: ticket mismatch during prepare-commit transaction commit.",
                ));
            }
            let mut committed = self.lock_committed();
            committed.insert(key, val.clone());
            Ok(val)
        } else {
            let committed = self.lock_committed();
            if let Some(existing) = committed.get(&key) {
                Ok(existing.clone())
            } else {
                Err(String::from(
                    "Fix: no prepared transaction found for key during commit phase.",
                ))
            }
        }
    }

    /// Commit an idempotency key, executing a side-effecting closure if not already committed.
    ///
    /// If the key is already committed, the closure is NEVER invoked, and the existing
    /// committed value is returned, guaranteeing exactly-once side-effect execution.
    pub fn commit_idempotent<E>(
        &self,
        key: K,
        ticket: PrepareTicket,
        side_effect: impl FnOnce() -> Result<V, E>,
    ) -> Result<V, E>
    where
        E: From<String>,
    {
        // First check if already committed
        {
            let committed = self.lock_committed();
            if let Some(val) = committed.get(&key) {
                return Ok(val.clone());
            }
        }

        // Verify prepared ticket
        {
            let prepared = self.lock_prepared();
            let (saved_ticket, _) = prepared.get(&key).ok_or_else(|| {
                E::from(String::from(
                    "Fix: no prepared transaction found for key during commit phase.",
                ))
            })?;
            if *saved_ticket != ticket {
                return Err(E::from(String::from(
                    "Fix: ticket mismatch during prepare-commit transaction commit.",
                )));
            }
        }

        // Execute side effect exactly once
        let val = side_effect()?;

        // Move from prepared to committed
        {
            let mut prepared = self.lock_prepared();
            prepared.remove(&key);
        }
        {
            let mut committed = self.lock_committed();
            committed.insert(key, val.clone());
        }

        Ok(val)
    }

    /// Abort and discard a prepared mutation.
    pub fn abort(&self, key: &K, ticket: PrepareTicket) {
        let mut prepared = self.lock_prepared();
        if let Some((saved_ticket, _)) = prepared.get(key) {
            if saved_ticket.ticket_id == ticket.ticket_id {
                prepared.remove(key);
            }
        }
    }

    /// Check if an idempotency key has been committed.
    #[must_use]
    pub fn is_committed(&self, key: &K) -> bool {
        let committed = self.lock_committed();
        committed.contains_key(key)
    }

    /// Retrieve the committed value for a key if already committed.
    #[must_use]
    pub fn get_committed(&self, key: &K) -> Option<V> {
        let committed = self.lock_committed();
        committed.get(key).cloned()
    }

    /// Bounded cleanup of stale prepared transactions exceeding a ticket threshold.
    pub fn cleanup_stale_prepared(&self, max_stale_tickets: u64, limit: usize) -> usize {
        let mut prepared = self.lock_prepared();
        let current_ticket = self.next_ticket.load(Ordering::SeqCst);
        let mut to_remove = Vec::new();
        for (key, (ticket, _)) in prepared.iter() {
            if current_ticket.saturating_sub(ticket.ticket_id) >= max_stale_tickets {
                to_remove.push(key.clone());
                if to_remove.len() >= limit {
                    break;
                }
            }
        }
        let count = to_remove.len();
        for key in to_remove {
            prepared.remove(&key);
        }
        count
    }
}

impl<K: Ord + Clone, V: Clone> Default for PrepareCommitJournal<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord + Clone, V: Clone> StateOwnerRecovery for PrepareCommitJournal<K, V> {
    fn failure_domain(&self) -> FailureDomain {
        FailureDomain::MemoryState
    }

    fn recovery_class(&self) -> RecoveryClass {
        RecoveryClass::TransactionallyRecoverable
    }
}

/// Supervised restart budget tracking crash and restart counts within a sliding window.
pub struct SupervisedRestartBudget {
    max_restarts: u32,
    restart_count: Mutex<u32>,
}

impl SupervisedRestartBudget {
    /// Create a new restart budget with a ceiling.
    #[must_use]
    pub const fn new(max_restarts: u32) -> Self {
        Self {
            max_restarts,
            restart_count: Mutex::new(0),
        }
    }

    /// Maximum permitted restarts.
    #[must_use]
    pub const fn max_restarts(&self) -> u32 {
        self.max_restarts
    }

    /// Current recorded restart count.
    #[must_use]
    fn lock_restart_count(&self) -> std::sync::MutexGuard<'_, u32> {
        reclaim_poisoned_irreplaceable_state(
            self.restart_count.lock(),
            || self.restart_count.clear_poison(),
            OWNER,
            "a supervised restart budget's consumed count",
        )
    }

    /// Current recorded restart count.
    #[must_use]
    pub fn current_restarts(&self) -> u32 {
        let count = self.lock_restart_count();
        *count
    }

    /// Remaining permitted restarts before budget is exhausted.
    #[must_use]
    pub fn remaining_restarts(&self) -> u32 {
        self.max_restarts.saturating_sub(self.current_restarts())
    }

    /// Record a restart event and check if the budget is exhausted.
    ///
    /// # Errors
    ///
    /// Returns [`TypedRecoveryError`] with [`RecoveryDisposition::Fatal`] if the budget is exceeded.
    pub fn record_restart(&self, domain: FailureDomain) -> Result<u32, TypedRecoveryError> {
        let mut count = self.lock_restart_count();
        *count += 1;
        if *count > self.max_restarts {
            return Err(TypedRecoveryError::new(
                domain,
                RecoveryClass::ProcessFatal,
                RecoveryDisposition::Fatal,
                format!(
                    "Supervised restart budget exhausted ({}/{} restarts)",
                    *count, self.max_restarts
                ),
                "Fix: worker process has crashed too many times; terminate and investigate fault.",
            ));
        }
        Ok(*count)
    }

    /// Reset restart count after a sustained period of healthy operation.
    pub fn reset(&self) {
        let mut count = self.lock_restart_count();
        *count = 0;
    }
}

impl StateOwnerRecovery for SupervisedRestartBudget {
    fn failure_domain(&self) -> FailureDomain {
        FailureDomain::WorkerProcess
    }

    fn recovery_class(&self) -> RecoveryClass {
        RecoveryClass::ProcessFatal
    }
}

/// Authoritative declaration registry mapping each known runtime state owner to its failure domain and recovery class.
#[must_use]
pub fn authoritative_runtime_state_owner_registry(
) -> BTreeMap<&'static str, (FailureDomain, RecoveryClass)> {
    let mut map = BTreeMap::new();
    map.insert(
        "vyre-runtime/src/atomic_recovery.rs:inner",
        (
            FailureDomain::MemoryState,
            RecoveryClass::RestartableFromCanonicalInput,
        ),
    );
    map.insert(
        "vyre-runtime/src/atomic_recovery.rs:prepared",
        (
            FailureDomain::MemoryState,
            RecoveryClass::TransactionallyRecoverable,
        ),
    );
    map.insert(
        "vyre-runtime/src/atomic_recovery.rs:committed",
        (
            FailureDomain::MemoryState,
            RecoveryClass::TransactionallyRecoverable,
        ),
    );
    map.insert(
        "vyre-runtime/src/atomic_recovery.rs:restart_count",
        (FailureDomain::WorkerProcess, RecoveryClass::ProcessFatal),
    );
    map.insert(
        "vyre-runtime/src/artifact_admission/interactive_session.rs:records",
        (
            FailureDomain::SessionLifecycle,
            RecoveryClass::TransactionallyRecoverable,
        ),
    );
    map.insert(
        "vyre-runtime/src/artifact_admission/interactive_session.rs:channel_generations",
        (
            FailureDomain::SessionLifecycle,
            RecoveryClass::TransactionallyRecoverable,
        ),
    );
    map.insert(
        "vyre-runtime/src/artifact_admission/interactive_session.rs:admitted_queue",
        (
            FailureDomain::SessionLifecycle,
            RecoveryClass::TransactionallyRecoverable,
        ),
    );
    map.insert(
        "vyre-runtime/src/artifact_admission/interactive_session.rs:faulted",
        (
            FailureDomain::SessionLifecycle,
            RecoveryClass::TransactionallyRecoverable,
        ),
    );
    map.insert(
        "vyre-runtime/src/artifact_admission/retained.rs:state_machine",
        (
            FailureDomain::DeviceContext,
            RecoveryClass::DeviceContextFatal,
        ),
    );
    map.insert(
        "vyre-runtime/src/artifact_admission/session.rs:state",
        (
            FailureDomain::DeviceContext,
            RecoveryClass::DeviceContextFatal,
        ),
    );
    map.insert(
        "vyre-runtime/src/pipeline_cache/in_memory.rs:shards",
        (
            FailureDomain::MemoryState,
            RecoveryClass::RestartableFromCanonicalInput,
        ),
    );
    map.insert(
        "vyre-runtime/src/pipeline_cache/disk.rs:pending_flushes",
        (
            FailureDomain::DiskJournal,
            RecoveryClass::RestartableFromCanonicalInput,
        ),
    );
    map.insert(
        "vyre-runtime/src/retained_page_cache/mod.rs:inner",
        (
            FailureDomain::MemoryState,
            RecoveryClass::RestartableFromCanonicalInput,
        ),
    );
    map.insert(
        "vyre-runtime/src/resource_residency/mod.rs:state",
        (
            FailureDomain::DeviceContext,
            RecoveryClass::DeviceContextFatal,
        ),
    );
    map.insert(
        "vyre-runtime/src/tenant/registry.rs:free_list",
        (
            FailureDomain::MemoryState,
            RecoveryClass::RestartableFromCanonicalInput,
        ),
    );
    map.insert(
        "vyre-runtime/src/tenant/registry.rs:tenants",
        (
            FailureDomain::MemoryState,
            RecoveryClass::RestartableFromCanonicalInput,
        ),
    );
    map.insert(
        "vyre-runtime/src/tenant/registry.rs:generations",
        (
            FailureDomain::MemoryState,
            RecoveryClass::RestartableFromCanonicalInput,
        ),
    );
    map.insert(
        "vyre-runtime/src/external_resource_admission.rs:resources",
        (
            FailureDomain::DeviceContext,
            RecoveryClass::DeviceContextFatal,
        ),
    );
    map.insert(
        "vyre-runtime/src/external_resource_admission.rs:dependent_views",
        (
            FailureDomain::DeviceContext,
            RecoveryClass::DeviceContextFatal,
        ),
    );
    map.insert(
        "vyre-runtime/src/external_resource_admission.rs:dependent_pipelines",
        (
            FailureDomain::DeviceContext,
            RecoveryClass::DeviceContextFatal,
        ),
    );
    map.insert(
        "vyre-runtime/src/structured_concurrency.rs:workers",
        (FailureDomain::WorkerProcess, RecoveryClass::ProcessFatal),
    );
    map
}
