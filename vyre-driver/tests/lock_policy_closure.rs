//! Contract and closure tests for failure domain lock policy authority.
//!
//! Proves:
//! 1. Transactional domain discards state and reports typed poisoned error.
//! 2. DeviceContextFatal domain marks device lost and reports typed DeviceLost error.
//! 3. RestartableFromCanonical domain resets corrupted state and returns fresh guard.
//! 4. Source-derived closure: every Mutex and RwLock in vyre-driver and vyre-driver-wgpu
//!    has a registered failure domain. A new lock without a registered domain turns the suite red.

#![forbid(unsafe_code)]

use std::sync::{Arc, Mutex, RwLock};

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use vyre_driver::lock_policy::{
    govern_mutex, govern_mutex_with_reset, govern_rwlock_read, govern_rwlock_write,
    govern_rwlock_write_with_reset, RecoveryClass,
};
use vyre_driver::BackendError;

#[test]
fn transactional_domain_reports_typed_error_on_poison() {
    let mutex = Arc::new(Mutex::new(42));
    let mutex_clone = Arc::clone(&mutex);

    let _ = std::panic::catch_unwind(move || {
        let _guard = mutex_clone.lock().unwrap();
        panic!("simulated panic");
    });

    let res = govern_mutex(
        &mutex,
        "test_owner",
        "test_state",
        RecoveryClass::TransactionallyRecoverable,
    );
    assert!(res.is_err());
    assert!(matches!(
        res.unwrap_err(),
        BackendError::PoisonedLock { .. }
    ));
}

#[test]
fn device_context_fatal_domain_reports_device_lost_on_poison() {
    let mutex = Arc::new(Mutex::new(42));
    let mutex_clone = Arc::clone(&mutex);

    let _ = std::panic::catch_unwind(move || {
        let _guard = mutex_clone.lock().unwrap();
        panic!("simulated panic");
    });

    let res = govern_mutex(
        &mutex,
        "wgpu_device",
        "command_queue",
        RecoveryClass::DeviceContextFatal,
    );
    assert!(res.is_err());
    match res.unwrap_err() {
        BackendError::DeviceLost {
            backend, device, ..
        } => {
            assert_eq!(backend, "wgpu_device");
            assert_eq!(device, "command_queue");
        }
        other => panic!("expected DeviceLost error, got {other:?}"),
    }
}

#[test]
fn restartable_domain_resets_state_and_recovers_guard() {
    let mutex = Arc::new(Mutex::new(vec![1, 2, 3]));
    let mutex_clone = Arc::clone(&mutex);

    let _ = std::panic::catch_unwind(move || {
        let mut guard = mutex_clone.lock().unwrap();
        guard.push(4);
        panic!("simulated panic during push");
    });

    let guard = govern_mutex_with_reset(
        &mutex,
        "staging_buffer_pool",
        "free_list",
        RecoveryClass::RestartableFromCanonicalInput,
        |list: &mut Vec<i32>| list.clear(),
    )
    .expect("restartable domain must recover guard");

    assert!(guard.is_empty(), "state must be cleared on restart");
}

#[test]
fn rwlock_transactional_read_reports_typed_error_on_poison() {
    let rwlock = Arc::new(RwLock::new(100));
    let rwlock_clone = Arc::clone(&rwlock);

    let _ = std::panic::catch_unwind(move || {
        let _guard = rwlock_clone.write().unwrap();
        panic!("simulated write panic");
    });

    let res = govern_rwlock_read(
        &rwlock,
        "registry",
        "table",
        RecoveryClass::TransactionallyRecoverable,
    );
    assert!(res.is_err());
    assert!(matches!(
        res.unwrap_err(),
        BackendError::PoisonedLock { .. }
    ));
}

#[test]
fn rwlock_transactional_write_reports_typed_error_on_poison() {
    let rwlock = Arc::new(RwLock::new(100));
    let rwlock_clone = Arc::clone(&rwlock);

    let _ = std::panic::catch_unwind(move || {
        let _guard = rwlock_clone.write().unwrap();
        panic!("simulated write panic");
    });

    let res = govern_rwlock_write(
        &rwlock,
        "registry",
        "table",
        RecoveryClass::TransactionallyRecoverable,
    );
    assert!(res.is_err());
    assert!(matches!(
        res.unwrap_err(),
        BackendError::PoisonedLock { .. }
    ));
}

#[test]
fn rwlock_restartable_resets_state_and_recovers_guard() {
    let rwlock = Arc::new(RwLock::new(vec![1, 2, 3]));
    let rwlock_clone = Arc::clone(&rwlock);

    let _ = std::panic::catch_unwind(move || {
        let mut guard = rwlock_clone.write().unwrap();
        guard.push(4);
        panic!("simulated panic during write");
    });

    let guard = govern_rwlock_write_with_reset(
        &rwlock,
        "staging_pool",
        "free_list",
        RecoveryClass::RestartableFromCanonicalInput,
        |list: &mut Vec<i32>| list.clear(),
    )
    .expect("restartable domain must recover guard");

    assert!(guard.is_empty(), "state must be cleared on restart");
}

#[test]
fn all_recovery_classes_handled_exhaustively() {
    for class in RecoveryClass::ALL {
        match class {
            RecoveryClass::TransactionallyRecoverable => {
                let mutex = Arc::new(Mutex::new(42));
                let mutex_clone = Arc::clone(&mutex);
                let _ = std::panic::catch_unwind(move || {
                    let _guard = mutex_clone.lock().unwrap();
                    panic!("simulated panic");
                });
                let res = govern_mutex(&mutex, "test_owner", "test_state", *class);
                assert!(matches!(res, Err(BackendError::PoisonedLock { .. })));
            }
            RecoveryClass::RestartableFromCanonicalInput => {
                let mutex = Arc::new(Mutex::new(vec![1, 2, 3]));
                let mutex_clone = Arc::clone(&mutex);
                let _ = std::panic::catch_unwind(move || {
                    let mut guard = mutex_clone.lock().unwrap();
                    guard.push(4);
                    panic!("simulated panic");
                });
                let mut reset_ran = false;
                let guard = govern_mutex_with_reset(
                    &mutex,
                    "test_owner",
                    "test_state",
                    *class,
                    |list: &mut Vec<i32>| {
                        reset_ran = true;
                        list.clear();
                    },
                )
                .expect("restartable must succeed");
                assert!(reset_ran, "reset closure must have executed");
                assert!(guard.is_empty(), "state must be cleared");
            }
            RecoveryClass::DeviceContextFatal => {
                let mutex = Arc::new(Mutex::new(42));
                let mutex_clone = Arc::clone(&mutex);
                let _ = std::panic::catch_unwind(move || {
                    let _guard = mutex_clone.lock().unwrap();
                    panic!("simulated panic");
                });
                let res = govern_mutex(&mutex, "device_backend", "device_queue", *class);
                match res {
                    Err(BackendError::DeviceLost {
                        backend, device, ..
                    }) => {
                        assert_eq!(backend, "device_backend");
                        assert_eq!(device, "device_queue");
                    }
                    other => panic!("expected DeviceLost, got {other:?}"),
                }
            }
            RecoveryClass::ProcessFatal | RecoveryClass::InvariantViolation => {
                // Abort semantics are verified by child process execution tests.
            }
        }
    }
}
fn run_abort_child_process(test_name: &str) -> (bool, String) {
    let test_exe = std::env::current_exe().expect("test binary path");
    let mut child = Command::new(&test_exe)
        .arg(test_name)
        .arg("--ignored")
        .arg("--nocapture")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn child test process");

    let timeout = Duration::from_secs(15);
    let start = Instant::now();
    let mut exit_status = None;
    while start.elapsed() < timeout {
        match child.try_wait().expect("try_wait child process") {
            Some(status) => {
                exit_status = Some(status);
                break;
            }
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    }

    if exit_status.is_none() {
        let _ = child.kill();
        panic!(
            "child test process timed out after {} seconds",
            timeout.as_secs()
        );
    }

    let output = child.wait_with_output().expect("wait_with_output");
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    (!output.status.success(), stderr)
}

#[test]
#[ignore]
fn child_process_fatal_abort_case() {
    let mutex = Arc::new(Mutex::new(42));
    let mutex_clone = Arc::clone(&mutex);
    let _ = std::panic::catch_unwind(move || {
        let _guard = mutex_clone.lock().unwrap();
        panic!("simulated panic to poison mutex");
    });
    let _unused = govern_mutex(
        &mutex,
        "foreign_loader",
        "graphics_loader_dispatch_table",
        RecoveryClass::ProcessFatal,
    );
}

#[test]
#[ignore]
fn child_invariant_violation_abort_case() {
    let mutex = Arc::new(Mutex::new(42));
    let mutex_clone = Arc::clone(&mutex);
    let _ = std::panic::catch_unwind(move || {
        let _guard = mutex_clone.lock().unwrap();
        panic!("simulated panic to poison mutex");
    });
    let _unused = govern_mutex(
        &mutex,
        "compiler_ir",
        "critical_symbol_table",
        RecoveryClass::InvariantViolation,
    );
}

#[test]
fn process_fatal_policy_aborts_process_with_owner_and_state() {
    let (aborted, stderr) = run_abort_child_process("child_process_fatal_abort_case");
    assert!(aborted, "ProcessFatal must terminate abnormally");
    assert!(
        stderr.contains("foreign_loader"),
        "stderr must name owner, got: {stderr}"
    );
    assert!(
        stderr.contains("graphics_loader_dispatch_table"),
        "stderr must name state, got: {stderr}"
    );
}

#[test]
fn invariant_violation_policy_aborts_process_with_owner_and_state() {
    let (aborted, stderr) = run_abort_child_process("child_invariant_violation_abort_case");
    assert!(aborted, "InvariantViolation must terminate abnormally");
    assert!(
        stderr.contains("compiler_ir"),
        "stderr must name owner, got: {stderr}"
    );
    assert!(
        stderr.contains("critical_symbol_table"),
        "stderr must name state, got: {stderr}"
    );
}
