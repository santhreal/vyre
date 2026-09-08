//! Tests for explicit failure domains, atomic guarded states, prepare/commit journals, and restart budgets.
//!
//! WHY: proves Row 122:
//! - Poisoned locks transition atomically to typed PoisonedTerminal state and reject further operations.
//! - State can be rebuilt cleanly from canonical input via explicit recover().
//! - Prepare/commit journals guarantee side effect idempotency under repeated calls.
//! - Supervised restart budgets enforce worker crash bounds and fail closed.

use std::panic;
use std::sync::Arc;
use std::thread;

use vyre_foundation::failure_domain::{
    FailureDomain, RecoveryClass, RecoveryDisposition,
};
use vyre_runtime::atomic_recovery::{
    AtomicGuardedState, GuardedState, PrepareCommitJournal, SupervisedRestartBudget,
};

#[test]
fn atomic_guarded_state_transitions_to_poisoned_terminal_on_panic() {
    let state = Arc::new(AtomicGuardedState::new(
        vec![1, 2, 3],
        FailureDomain::MemoryState,
        RecoveryClass::RestartableFromCanonicalInput,
    ));

    let state_clone = Arc::clone(&state);
    let handle = thread::spawn(move || {
        let _ = state_clone.with_state(|_vec| -> Result<(), String> {
            panic!("Intentional worker fault to trigger lock poison");
        });
    });

    let _ = handle.join();

    // Subsequent operation from another thread must see typed TypedRecoveryError, not a raw panic
    let err = state
        .with_state(|vec| {
            vec.push(4);
            Ok(())
        })
        .expect_err("Fix: poisoned state must reject operations with TypedRecoveryError.");

    assert_eq!(err.domain, FailureDomain::MemoryState);
    assert_eq!(
        err.recovery_class,
        RecoveryClass::RestartableFromCanonicalInput
    );
    assert_eq!(err.disposition, RecoveryDisposition::RequiresRebuild);
    assert!(err.fix.contains("Fix:"));

    // Explicit recovery restores state to Ready
    state.recover(vec![10, 20]);
    let len = state
        .with_state(|vec| Ok(vec.len()))
        .expect("Fix: recovered state must accept operations normally.");
    assert_eq!(len, 2);
}

#[test]
fn prepare_commit_journal_guarantees_idempotent_side_effects() {
    let journal = PrepareCommitJournal::<String, u64>::new();
    let key = String::from("submission_key_1001");

    // Phase 1: Prepare
    let ticket = journal
        .prepare(key.clone(), 42)
        .expect("Fix: initial prepare must succeed.")
        .expect("Fix: ticket must be returned.");

    // Phase 2: Commit
    let committed_val = journal
        .commit(key.clone(), ticket)
        .expect("Fix: commit must succeed with valid ticket.");
    assert_eq!(committed_val, 42);
    assert!(journal.is_committed(&key));

    // Duplicate prepare with the same idempotency key returns Ok(None) - idempotent no-op
    let duplicate_prepare = journal
        .prepare(key.clone(), 999)
        .expect("Fix: prepare on already committed key must return Ok(None).");
    assert!(
        duplicate_prepare.is_none(),
        "Fix: already committed idempotency key must not execute new side effects."
    );
}

#[test]
fn prepare_commit_journal_aborted_ticket_cleans_state() {
    let journal = PrepareCommitJournal::<String, u64>::new();
    let key = String::from("submission_key_abort");

    let ticket = journal
        .prepare(key.clone(), 100)
        .unwrap()
        .unwrap();

    journal.abort(&key, ticket);
    assert!(!journal.is_committed(&key));

    // Key can now be prepared again cleanly
    let new_ticket = journal
        .prepare(key.clone(), 200)
        .expect("Fix: after abort, key must be preparable again.");
    assert!(new_ticket.is_some());
}

#[test]
fn supervised_restart_budget_exhausts_and_fails_closed() {
    let budget = SupervisedRestartBudget::new(3);

    assert_eq!(budget.record_restart(FailureDomain::WorkerProcess).unwrap(), 1);
    assert_eq!(budget.record_restart(FailureDomain::WorkerProcess).unwrap(), 2);
    assert_eq!(budget.record_restart(FailureDomain::WorkerProcess).unwrap(), 3);

    // 4th restart exceeds ceiling of 3
    let err = budget
        .record_restart(FailureDomain::WorkerProcess)
        .expect_err("Fix: exceeding restart budget must fail closed with ProcessFatal error.");

    assert_eq!(err.domain, FailureDomain::WorkerProcess);
    assert_eq!(err.recovery_class, RecoveryClass::ProcessFatal);
    assert_eq!(err.disposition, RecoveryDisposition::Fatal);
    assert!(err.fix.contains("Fix:"));
}
