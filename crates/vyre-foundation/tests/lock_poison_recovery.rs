//! Poison recovery in the governed lock seam: what a caller observes after a
//! panic left guarded state half written.
//!
//! WHY: closes the class "recovering from a poisoned lock leaves the lock
//! poisoned, so every later acquisition re-runs the recovery". Recovery for
//! restartable state discards the guarded value, and a lock that stays poisoned
//! discards it again on every acquisition for the life of the process: one
//! panic silently empties a live cache forever. The reset action must fire once.
//!
//! Also covered: teardown, where refusing poisoned state leaks the handles it
//! holds, so the destructor must observe it rather than fail closed.
//!
//! Not covered: which recovery class a given lock belongs in. That is a
//! judgement about what the lock guards, and review is its gate.

use std::cell::Cell;
use std::sync::{Mutex, RwLock};

use vyre_foundation::failure_domain::{
    govern_mutex_restartable, govern_mutex_with_reset, govern_rwlock_write_with_reset,
    reclaim_poisoned_for_teardown, RecoveryClass,
};

/// Poison `mutex` by panicking with its guard held.
///
/// A scoped thread borrows the lock instead of sharing it through an `Arc`, so
/// the caller keeps the unique reference `get_mut` needs afterwards.
fn poison_mutex(mutex: &Mutex<Vec<u32>>) {
    let panicked = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let mut guard = mutex.lock().expect("Fix: the lock is not poisoned yet");
                guard.push(7);
                panic!("the panic that poisons the lock");
            })
            .join()
    });
    assert!(panicked.is_err(), "the poisoning thread must have panicked");
    assert!(mutex.is_poisoned(), "the lock must be poisoned");
}

/// Poison `rwlock` by panicking with its write guard held.
fn poison_rwlock(rwlock: &RwLock<Vec<u32>>) {
    let panicked = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let mut guard = rwlock.write().expect("Fix: the lock is not poisoned yet");
                guard.push(7);
                panic!("the panic that poisons the lock");
            })
            .join()
    });
    assert!(panicked.is_err(), "the poisoning thread must have panicked");
    assert!(rwlock.is_poisoned(), "the lock must be poisoned");
}

/// `govern_mutex_restartable` resets once and leaves an unpoisoned lock.
///
/// The second acquisition is an ordinary one: it observes what the first caller
/// wrote after the reset. A lock left poisoned would reset again here and
/// report an empty vector, which is the defect this covers.
#[test]
fn restartable_recovery_resets_once_and_clears_the_poison() {
    let mutex = Mutex::new(Vec::new());
    poison_mutex(&mutex);

    let resets = Cell::new(0u32);
    let mut first = govern_mutex_restartable(&mutex, "the test owner", "the test state", |state| {
        state.clear();
        resets.set(resets.get() + 1);
    });
    assert_eq!(*first, Vec::<u32>::new(), "the reset must empty the state");
    first.push(11);
    drop(first);

    assert!(
        !mutex.is_poisoned(),
        "recovery must clear the poison, or every later acquisition resets again"
    );

    let second = govern_mutex_restartable(&mutex, "the test owner", "the test state", |state| {
        state.clear();
        resets.set(resets.get() + 1);
    });
    assert_eq!(
        *second,
        vec![11],
        "the second acquisition must observe what the first caller wrote"
    );
    drop(second);

    assert_eq!(
        resets.get(),
        1,
        "the reset must fire once for one panic, not once per acquisition"
    );
}

/// `govern_mutex_with_reset` clears the poison for restartable state.
///
/// The classifying variant of the case above: the same recovery reached through
/// the class parameter rather than the dedicated entry point.
#[test]
fn governed_mutex_reset_clears_the_poison_for_restartable_state() {
    let mutex = Mutex::new(Vec::new());
    poison_mutex(&mutex);

    let guard = govern_mutex_with_reset(
        &mutex,
        "the test owner",
        "the test state",
        RecoveryClass::RestartableFromCanonicalInput,
        |state| state.clear(),
    )
    .expect("Fix: restartable state recovers rather than reporting an error");
    assert_eq!(*guard, Vec::<u32>::new());
    drop(guard);

    assert!(!mutex.is_poisoned(), "recovery must clear the poison");
}

/// `govern_rwlock_write_with_reset` clears the poison for restartable state.
///
/// The write lock is the other half of the union: read recovery cannot mutate,
/// so the write side is where the reset and the clear belong.
#[test]
fn governed_rwlock_write_reset_clears_the_poison_for_restartable_state() {
    let rwlock = RwLock::new(Vec::new());
    poison_rwlock(&rwlock);

    let guard = govern_rwlock_write_with_reset(
        &rwlock,
        "the test owner",
        "the test state",
        RecoveryClass::RestartableFromCanonicalInput,
        |state| state.clear(),
    )
    .expect("Fix: restartable state recovers rather than reporting an error");
    assert_eq!(*guard, Vec::<u32>::new());
    drop(guard);

    assert!(!rwlock.is_poisoned(), "recovery must clear the poison");
}

/// A recovery class that is not restartable reports rather than resetting.
///
/// The boundary of the clear: only the restartable arm recovers a guard, so
/// only it may clear the flag. A transactional owner rebuilds from canonical
/// input instead, and the lock stays poisoned for whoever asks next.
#[test]
fn transactional_recovery_reports_and_leaves_the_poison_in_place() {
    let mutex = Mutex::new(Vec::new());
    poison_mutex(&mutex);

    let error = govern_mutex_with_reset(
        &mutex,
        "the test owner",
        "the test state",
        RecoveryClass::TransactionallyRecoverable,
        |state| state.clear(),
    )
    .expect_err("Fix: transactional state reports the poison rather than recovering it");
    assert_eq!(
        error.recovery_class,
        RecoveryClass::TransactionallyRecoverable
    );
    assert!(
        mutex.is_poisoned(),
        "a reported poison stays poisoned: nothing rebuilt the state"
    );
}

/// An unpoisoned lock never runs the reset.
///
/// The negative case that keeps the recovery path from becoming the ordinary
/// path: an owner that clears state on every acquisition is a cache that never
/// holds anything.
#[test]
fn an_unpoisoned_lock_does_not_run_the_reset() {
    let mutex = Mutex::new(vec![1u32, 2, 3]);
    let mut ran = false;
    let guard = govern_mutex_restartable(&mutex, "the test owner", "the test state", |state| {
        state.clear();
        ran = true;
    });
    assert_eq!(*guard, vec![1, 2, 3]);
    drop(guard);
    assert!(!ran, "the reset must not run when nothing was poisoned");
}

/// Teardown observes poisoned state so the handles inside it are released.
///
/// A destructor is the last owner of what a panic left. Failing closed here
/// leaks every handle in the guarded value, so teardown takes the state and
/// reports the poison instead.
#[test]
fn teardown_reclaims_poisoned_state_so_handles_are_released() {
    let mut mutex = Mutex::new(Vec::new());
    poison_mutex(&mutex);

    let state = reclaim_poisoned_for_teardown(mutex.get_mut(), "the test owner", "the test state");
    assert_eq!(
        *state,
        vec![7],
        "teardown must observe what the panic left so it can release it"
    );
}

/// Teardown of unpoisoned state hands back the same value.
///
/// The reclamation is not a recovery: it neither resets nor discards, because a
/// destructor has nothing to hand a later caller.
#[test]
fn teardown_of_unpoisoned_state_returns_it_unchanged() {
    let mut mutex = Mutex::new(vec![1u32, 2, 3]);
    let state = reclaim_poisoned_for_teardown(mutex.get_mut(), "the test owner", "the test state");
    assert_eq!(*state, vec![1, 2, 3]);
}
