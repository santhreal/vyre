//! Atomic failure recovery, state machines, prepare/commit journals, and restart budgets.
//!
//! WHY: closes the class "poisoned locks panic or silently extract partial state, and retry repeats non-idempotent side effects".
//! Provides typed atomic state machines that fail into terminal/rebuilding states on poison,
//! prepare/commit journaling with idempotency keys, and supervised worker restart budgets.

use std::collections::BTreeMap;
use std::format;
use std::string::String;
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use vyre_foundation::failure_domain::{
    FailureDomain, RecoveryClass, RecoveryDisposition, TypedRecoveryError,
};

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

    /// Access the guarded state with an operational closure.
    ///
    /// # Errors
    ///
    /// Returns [`TypedRecoveryError`] if the lock was poisoned or if the state is terminal.
    pub fn with_state<R>(
        &self,
        op: impl FnOnce(&mut T) -> Result<R, String>,
    ) -> Result<R, TypedRecoveryError> {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(poison) => {
                let mut inner_guard = poison.into_inner();
                *inner_guard = GuardedState::PoisonedTerminal {
                    domain: self.domain,
                    recovery_class: self.recovery_class,
                    reason: String::from("Lock was poisoned by a previous thread panic"),
                };
                return Err(TypedRecoveryError::new(
                    self.domain,
                    self.recovery_class,
                    RecoveryDisposition::RequiresRebuild,
                    "Lock was poisoned by a previous thread panic",
                    "Fix: invoke recover() to rebuild the state machine from canonical input.",
                ));
            }
        };

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

    /// Explicitly recover and restore state to Ready.
    pub fn recover(&self, fresh_state: T) {
        if let Ok(mut guard) = self.inner.lock() {
            *guard = GuardedState::Ready(fresh_state);
        } else if let Err(poison) = self.inner.lock() {
            let mut guard = poison.into_inner();
            *guard = GuardedState::Ready(fresh_state);
        }
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
    pub fn prepare(&self, key: K, value: V) -> Result<Option<PrepareTicket>, String> {
        let committed = self.committed.lock().unwrap();
        if committed.contains_key(&key) {
            return Ok(None); // Already committed; idempotent no-op
        }
        drop(committed);

        let mut prepared = self.prepared.lock().unwrap();
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
        let (saved_ticket, val) = prepared.remove(&key).ok_or_else(|| {
            format!("Fix: no prepared transaction found for key during commit phase.")
        })?;

        if saved_ticket != ticket {
            return Err(String::from(
                "Fix: ticket mismatch during prepare-commit transaction commit.",
            ));
        }

        let mut committed = self.committed.lock().unwrap();
        committed.insert(key, val.clone());
        Ok(val)
    }

    /// Abort and discard a prepared mutation.
    pub fn abort(&self, key: &K, ticket: PrepareTicket) {
        let mut prepared = self.prepared.lock().unwrap();
        if let Some((saved_ticket, _)) = prepared.get(key) {
            if saved_ticket.ticket_id == ticket.ticket_id {
                prepared.remove(key);
            }
        }
    }

    /// Check if an idempotency key has been committed.
    #[must_use]
    pub fn is_committed(&self, key: &K) -> bool {
        self.committed.lock().unwrap().contains_key(key)
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

    /// Record a restart event and check if the budget is exhausted.
    ///
    /// # Errors
    ///
    pub fn record_restart(&self, domain: FailureDomain) -> Result<u32, TypedRecoveryError> {
        let mut count = self.restart_count.lock().unwrap();
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
        *self.restart_count.lock().unwrap() = 0;
    }
}
