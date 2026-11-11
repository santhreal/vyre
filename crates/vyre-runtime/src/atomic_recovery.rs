//! Atomic failure recovery, state machines, prepare/commit journals, and restart budgets.
//!
//! WHY: closes the class "poisoned locks panic or silently extract partial state, and retry repeats non-idempotent side effects".
//! Provides typed atomic state machines that fail into terminal/rebuilding states on poison,
//! prepare/commit journaling with idempotency keys, and supervised worker restart budgets.

use core::sync::atomic::{AtomicU64, Ordering};
use std::collections::{BTreeMap, BTreeSet};
use std::format;
use std::string::String;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use vyre_foundation::failure_domain::{
    reclaim_poisoned_condvar_wait_timeout, reclaim_poisoned_mutex, reclaim_poisoned_mutex_observed,
    FailureDomain, Reclaimed, RecoveryClass, RecoveryDisposition, StateOwnerRecovery,
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

/// Thread-safe atomic guarded container that eliminates uncoordinated lock poison.
#[derive(Debug)]
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
        let (mut guard, reclaimed) = reclaim_poisoned_mutex_observed(
            &self.inner,
            OWNER,
            "one atomically guarded state machine",
        );
        if reclaimed == Reclaimed::Once {
            *guard = GuardedState::PoisonedTerminal {
                domain: self.domain,
                recovery_class: self.recovery_class,
                reason: String::from("Lock was poisoned by a previous thread panic"),
            };
        }
        guard
    }

    /// Access the guarded state with an operational closure whose error type
    /// is the caller's.
    ///
    /// WHY: an owner keeps its own error vocabulary while the reason for a
    /// rejection stays the governed one. A rejection arrives as
    /// `E::from(TypedRecoveryError)`, which states the failure domain, the
    /// recovery class and the corrective action, so a caller cannot mistake an
    /// operation this owner refused to start for one that ran and failed.
    ///
    /// # Errors
    ///
    /// Returns the closure's error, or the rejection this owner reports while
    /// it is rebuilding or terminal.
    pub fn try_with_state<R, E>(&self, op: impl FnOnce(&mut T) -> Result<R, E>) -> Result<R, E>
    where
        E: From<TypedRecoveryError>,
    {
        let mut guard = self.lock_state();

        match &mut *guard {
            GuardedState::Ready(value) => op(value),
            GuardedState::Rebuilding => Err(E::from(TypedRecoveryError::new(
                self.domain,
                self.recovery_class,
                RecoveryDisposition::RequiresRebuild,
                "State machine is currently rebuilding",
                "Fix: await completion of the rebuild process before submitting operations.",
            ))),
            GuardedState::PoisonedTerminal {
                domain,
                recovery_class,
                reason,
            } => Err(E::from(TypedRecoveryError::new(
                *domain,
                *recovery_class,
                RecoveryDisposition::RequiresRebuild,
                reason.clone(),
                "Fix: state machine is in PoisonedTerminal state; perform explicit recovery.",
            ))),
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
        let domain = self.domain;
        let recovery_class = self.recovery_class;
        self.try_with_state(|value| {
            op(value).map_err(|reason| {
                TypedRecoveryError::new(
                    domain,
                    recovery_class,
                    RecoveryDisposition::RequiresRebuild,
                    reason,
                    "Fix: inspect the failure cause and rebuild state if necessary.",
                )
            })
        })
    }

    /// Access the guarded state with a read-only closure.
    ///
    /// # Errors
    ///
    /// Returns [`TypedRecoveryError`] if the lock was poisoned or if the state is terminal.
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

    /// Move the owner into [`GuardedState::Rebuilding`], where every operation
    /// is rejected until [`finish_rebuild`](Self::finish_rebuild) completes it.
    ///
    /// # Errors
    ///
    /// Returns [`TypedRecoveryError`] when a rebuild is already in progress.
    /// The transition and the check share one acquisition, so two callers
    /// cannot both hold the rebuild and the second is told the first owns it
    /// rather than silently restarting it.
    pub fn begin_rebuild(&self) -> Result<(), TypedRecoveryError> {
        let mut guard = self.lock_state();
        if matches!(*guard, GuardedState::Rebuilding) {
            return Err(TypedRecoveryError::new(
                self.domain,
                self.recovery_class,
                RecoveryDisposition::RequiresRebuild,
                "A rebuild of this state machine is already in progress",
                "Fix: await the in-progress rebuild instead of starting a second one.",
            ));
        }
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

/// How long a caller stands down for an in-flight commit of the same
/// idempotency key before reporting the wait instead of extending it.
///
/// A stand-down that never ends turns one stalled committer into a stalled
/// caller with no report, so the wait carries a bound and the error a caller
/// receives when it elapses states that bound.
pub const DEFAULT_COMMIT_STANDDOWN: Duration = Duration::from_secs(5);

/// The prepared, committed and in-flight keys of one journal.
struct JournalState<K: Ord, V> {
    prepared: BTreeMap<K, (PrepareTicket, V)>,
    committed: BTreeMap<K, V>,
    executing: BTreeSet<K>,
}

/// Prepare-commit journal ensuring mutation side effects are strictly idempotent.
///
/// WHY: the three sets are one owner under one lock. Split across separate
/// locks, two callers holding the same idempotency key each observed an
/// uncommitted key, each ran the side effect, and each published it, which is
/// the duplicate submission the journal exists to prevent.
pub struct PrepareCommitJournal<K: Ord + Clone, V: Clone> {
    next_ticket: AtomicU64,
    state: Mutex<JournalState<K, V>>,
    commit_settled: Condvar,
    standdown: Duration,
}

/// The name every poison report over one journal's state gives that state.
const JOURNAL_STATE: &str = "one prepare-commit journal's prepared, committed and in-flight keys";

/// Clears one key's in-flight mark even when the side effect unwinds.
///
/// A side effect runs with no lock held, so a panic inside it would otherwise
/// leave the key marked in flight for the life of the process and stand every
/// later retry down to its bound.
struct InFlightCommit<'journal, K: Ord + Clone, V: Clone> {
    journal: &'journal PrepareCommitJournal<K, V>,
    key: K,
}

impl<K: Ord + Clone, V: Clone> Drop for InFlightCommit<'_, K, V> {
    fn drop(&mut self) {
        let mut state = self.journal.lock_state();
        state.executing.remove(&self.key);
        drop(state);
        self.journal.commit_settled.notify_all();
    }
}

impl<K: Ord + Clone, V: Clone> PrepareCommitJournal<K, V> {
    /// Create a new prepare-commit journal standing down for
    /// [`DEFAULT_COMMIT_STANDDOWN`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_standdown(DEFAULT_COMMIT_STANDDOWN)
    }

    /// Create a journal whose in-flight stand-down uses an explicit bound.
    #[must_use]
    pub fn with_standdown(standdown: Duration) -> Self {
        Self {
            next_ticket: AtomicU64::new(1),
            state: Mutex::new(JournalState {
                prepared: BTreeMap::new(),
                committed: BTreeMap::new(),
                executing: BTreeSet::new(),
            }),
            commit_settled: Condvar::new(),
            standdown,
        }
    }

    /// Longest a caller waits for an in-flight commit of the same key.
    #[must_use]
    pub const fn standdown(&self) -> Duration {
        self.standdown
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, JournalState<K, V>> {
        reclaim_poisoned_mutex(&self.state, OWNER, JOURNAL_STATE)
    }

    /// Prepare a mutation under an idempotency key.
    ///
    /// Returns `Ok(None)` when the key is already committed, which is the
    /// idempotent no-op a retry of a completed effect resolves to.
    ///
    /// # Errors
    ///
    /// Returns an error when the key is already prepared or already in flight,
    /// because a second ticket over one key is a second side effect.
    pub fn prepare(&self, key: K, value: V) -> Result<Option<PrepareTicket>, String> {
        let mut state = self.lock_state();
        if state.committed.contains_key(&key) {
            return Ok(None);
        }
        if state.prepared.contains_key(&key) || state.executing.contains(&key) {
            return Err(String::from(
                "Fix: operation with this idempotency key is already prepared in flight.",
            ));
        }

        let ticket_id = self.next_ticket.fetch_add(1, Ordering::SeqCst);
        let ticket = PrepareTicket { ticket_id };
        state.prepared.insert(key, (ticket, value));
        Ok(Some(ticket))
    }

    /// Commit the prepared mutation atomically.
    ///
    /// # Errors
    ///
    /// Returns an error if the ticket does not match the prepared entry.
    pub fn commit(&self, key: K, ticket: PrepareTicket) -> Result<V, String> {
        let mut state = self.lock_state();
        if let Some(existing) = state.committed.get(&key) {
            return Ok(existing.clone());
        }
        let Some((saved_ticket, value)) = state.prepared.remove(&key) else {
            return Err(String::from(
                "Fix: no prepared transaction found for key during commit phase.",
            ));
        };
        if saved_ticket != ticket {
            state.prepared.insert(key, (saved_ticket, value));
            return Err(String::from(
                "Fix: ticket mismatch during prepare-commit transaction commit.",
            ));
        }
        state.committed.insert(key, value.clone());
        Ok(value)
    }

    /// Commit an idempotency key, executing a side-effecting closure if not already committed.
    ///
    /// The closure runs at most once per key for the life of the journal. A
    /// caller arriving while another holds the key stands down until that
    /// commit settles and then reads the committed value, so a retry of a
    /// submission, publication, allocation, signature issuance or cache
    /// insertion produces one effect and one published value.
    ///
    /// # Errors
    ///
    /// Returns the closure's error, or a stand-down report naming the bound
    /// when a commit already in flight for this key does not settle within
    /// [`standdown`](Self::standdown). The key stays prepared in both cases, so
    /// the caller retries rather than losing the record.
    pub fn commit_idempotent<E>(
        &self,
        key: K,
        ticket: PrepareTicket,
        side_effect: impl FnOnce() -> Result<V, E>,
    ) -> Result<V, E>
    where
        E: From<String>,
    {
        let mut state = self.lock_state();
        let deadline = Instant::now() + self.standdown;
        while state.executing.contains(&key) {
            if let Some(value) = state.committed.get(&key) {
                return Ok(value.clone());
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(E::from(format!(
                    "Fix: a commit for this idempotency key has been in flight longer than {} ms; \
                     await or abort it before retrying.",
                    self.standdown.as_millis()
                )));
            }
            let (reacquired, _) = reclaim_poisoned_condvar_wait_timeout(
                &self.commit_settled,
                &self.state,
                state,
                remaining,
                OWNER,
                JOURNAL_STATE,
            );
            state = reacquired;
        }

        if let Some(value) = state.committed.get(&key) {
            return Ok(value.clone());
        }
        match state.prepared.get(&key) {
            None => {
                return Err(E::from(String::from(
                    "Fix: no prepared transaction found for key during commit phase.",
                )))
            }
            Some((saved_ticket, _)) if *saved_ticket != ticket => {
                return Err(E::from(String::from(
                    "Fix: ticket mismatch during prepare-commit transaction commit.",
                )))
            }
            Some(_) => {}
        }
        state.executing.insert(key.clone());
        drop(state);

        // The mark is held across the side effect, which is what makes the
        // effect exactly once. Dropping this clears it whether the effect
        // returns a value, reports an error, or unwinds.
        let in_flight = InFlightCommit {
            journal: self,
            key: key.clone(),
        };
        let value = side_effect()?;

        let mut state = self.lock_state();
        state.prepared.remove(&key);
        state.committed.insert(key, value.clone());
        drop(state);
        drop(in_flight);

        Ok(value)
    }

    /// Abort and discard a prepared mutation.
    ///
    /// Returns whether the entry was discarded. A key whose side effect is in
    /// flight is never discarded, because releasing it would let a second
    /// prepare issue a second ticket over an effect already running.
    pub fn abort(&self, key: &K, ticket: PrepareTicket) -> bool {
        let mut state = self.lock_state();
        if state.executing.contains(key) {
            return false;
        }
        if state
            .prepared
            .get(key)
            .is_some_and(|(saved_ticket, _)| saved_ticket.ticket_id == ticket.ticket_id)
        {
            state.prepared.remove(key);
            return true;
        }
        false
    }

    /// Check if an idempotency key has been committed.
    #[must_use]
    pub fn is_committed(&self, key: &K) -> bool {
        self.lock_state().committed.contains_key(key)
    }

    /// Retrieve the committed value for a key if already committed.
    #[must_use]
    pub fn get_committed(&self, key: &K) -> Option<V> {
        self.lock_state().committed.get(key).cloned()
    }

    /// Count the keys whose side effect is running right now.
    #[must_use]
    pub fn in_flight_len(&self) -> usize {
        self.lock_state().executing.len()
    }

    /// Discard at most `limit` prepared entries older than `max_stale_tickets`.
    ///
    /// Cleanup is bounded by `limit` and skips a key whose side effect is in
    /// flight, so repeated invocation converges and never discards the record
    /// a running effect is about to commit against.
    pub fn cleanup_stale_prepared(&self, max_stale_tickets: u64, limit: usize) -> usize {
        let mut state = self.lock_state();
        let current_ticket = self.next_ticket.load(Ordering::SeqCst);
        let mut to_remove = Vec::new();
        for (key, (ticket, _)) in &state.prepared {
            if state.executing.contains(key) {
                continue;
            }
            if current_ticket.saturating_sub(ticket.ticket_id) >= max_stale_tickets {
                to_remove.push(key.clone());
                if to_remove.len() >= limit {
                    break;
                }
            }
        }
        let count = to_remove.len();
        for key in to_remove {
            state.prepared.remove(&key);
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

/// Supervised restart budget bounding how often one disposable worker or
/// device context is respawned before the fault is reported as fatal.
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
        reclaim_poisoned_mutex(
            &self.restart_count,
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
