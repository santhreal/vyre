//! The two poison domains behave differently, and each is proven by the thing
//! that distinguishes it.
//!
//! WHY: a poisoned lock had three answers in this tree and none of them was
//! written down, so "poisoned" meant a stale value in one crate, a fine value
//! in the next, and an unsound process in a third. The recoverable domain is
//! proven by a caller receiving an error it can act on; the process-fatal
//! domain is proven by there being no caller at all, which only a child process
//! can observe.
//!
//! What this does not catch: whether a given lock was placed in the right
//! domain. That is a judgement about what the lock guards, and the gate for it
//! is review, not this file.

use vyre_driver::lock_policy::{process_fatal_poison, recoverable_poison};

/// Set in the child so the process-fatal test can call the thing it measures.
const CHILD: &str = "VYRE_LOCK_POISON_FATAL_CHILD";

#[test]
fn a_process_fatal_poison_ends_the_process_and_names_the_owner_and_the_state() {
    if std::env::var_os(CHILD).is_some() {
        process_fatal_poison(
            "the wgpu device factory",
            "the graphics loader dispatch table",
        );
    }

    let executable = std::env::current_exe()
        .expect("Fix: the test binary must name its own path to re-enter as a child");
    let output = std::process::Command::new(executable)
        .args([
            "--exact",
            "--nocapture",
            "lock_poison_policy::a_process_fatal_poison_ends_the_process_and_names_the_owner_and_the_state",
        ])
        .env(CHILD, "1")
        .output()
        .expect("Fix: the child test process must start");

    assert!(
        !output.status.success(),
        "a process-fatal poison returned to its caller; child exited {:?} with stdout {}",
        output.status,
        String::from_utf8_lossy(&output.stdout)
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(
            "the wgpu device factory holds a poisoned lock over the graphics loader dispatch table"
        ),
        "the process ended without naming the owner and the state: {stderr}"
    );
    assert!(
        stderr.contains("Fix: report the earlier panic"),
        "the process ended without naming the corrective action: {stderr}"
    );
}

#[test]
fn a_recoverable_poison_reaches_the_caller_as_an_error_carrying_no_guarded_value() {
    let lock = std::sync::Mutex::new(7i32);
    let guard = lock.lock().expect("Fix: a fresh mutex is not poisoned");
    let poisoned = Err(std::sync::PoisonError::new(guard));

    let error = recoverable_poison(poisoned)
        .expect_err("Fix: a poisoned result must reach the caller as an error");

    let rendered = error.to_string();
    assert!(
        !rendered.contains('7'),
        "the guarded value survived into the error the caller reads: {rendered}"
    );
}

#[test]
fn a_healthy_lock_passes_its_guard_through_the_recoverable_domain_untouched() {
    let lock = std::sync::Mutex::new(7i32);
    let guard = recoverable_poison(lock.lock()).expect("Fix: an unpoisoned lock yields its guard");

    assert_eq!(*guard, 7);
}
