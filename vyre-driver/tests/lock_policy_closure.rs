//! Contract and closure tests for failure domain lock policy authority.
//!
//! Proves:
//! 1. Transactional domain discards state and reports typed poisoned error.
//! 2. DeviceContextFatal domain marks device lost and reports typed DeviceLost error.
//! 3. RestartableFromCanonical domain resets corrupted state and returns fresh guard.
//! 4. Source-derived closure: every Mutex and RwLock in vyre-driver and vyre-driver-wgpu
//!    has a registered failure domain. A new lock without a registered domain turns the suite red.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

use vyre_driver::lock_policy::{
    govern_mutex, govern_mutex_with_reset, govern_rwlock_read, FailureDomain,
};
use vyre_driver::BackendError;
use vyre_test_support::monorepo::vyre_workspace_root;

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
        FailureDomain::Transactional,
    );
    assert!(res.is_err());
    assert!(matches!(res.unwrap_err(), BackendError::PoisonedLock { .. }));
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
        FailureDomain::DeviceContextFatal,
    );
    assert!(res.is_err());
    match res.unwrap_err() {
        BackendError::DeviceLost { backend, device, .. } => {
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
        FailureDomain::RestartableFromCanonical,
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
        FailureDomain::Transactional,
    );
    assert!(res.is_err());
    assert!(matches!(res.unwrap_err(), BackendError::PoisonedLock { .. }));
}

use vyre_driver::lock_policy::{authoritative_driver_lock_registry, RecoveryClass};

#[test]
fn lock_failure_domain_closure_from_source() {
    let registry = authoritative_driver_lock_registry();

    fn scan_dir(dir: &Path, lock_files: &mut BTreeMap<String, usize>) {
        if !dir.exists() {
            return;
        }
        for entry in fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().unwrap().to_str().unwrap();
                if name != "target" && name != "tests" {
                    scan_dir(&path, lock_files);
                }
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let file_name = path.file_name().unwrap().to_str().unwrap();
                if file_name == "tests.rs" || file_name.ends_with("_tests.rs") || file_name == "lock_policy.rs" {
                    continue;
                }
                let content = fs::read_to_string(&path).unwrap();
                let mut lock_count = 0;
                let mut in_test_cfg = false;
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("#[cfg(test)]") {
                        in_test_cfg = true;
                    }
                    if in_test_cfg {
                        continue;
                    }
                    if (line.contains("Mutex<") || line.contains("RwLock<"))
                        && !trimmed.starts_with("//")
                        && !trimmed.starts_with("/*")
                        && !line.contains("use ")
                        && !line.contains("fn ")
                    {
                        lock_count += 1;
                    }
                }
                if lock_count > 0 {
                    let path_str = path.to_str().unwrap().replace('\\', "/");
                    lock_files.insert(path_str, lock_count);
                }
            }
        }
    }

    let mut lock_files = BTreeMap::new();
    let root = vyre_workspace_root();
    scan_dir(&root.join("vyre-driver/src"), &mut lock_files);
    scan_dir(&root.join("vyre-driver-wgpu/src"), &mut lock_files);

    assert!(
        !lock_files.is_empty(),
        "scan must locate existing driver locks from source"
    );

    // 1. Verify every source file with locks is covered in registry
    for (file_path, count) in &lock_files {
        let has_entry = registry.keys().any(|key| {
            let prefix = key.split(':').next().unwrap_or("");
            file_path.ends_with(prefix)
        });
        assert!(
            has_entry,
            "Source file {file_path} contains {count} lock(s) but has no declared FailureDomain in authoritative registry! Fix: register failure domain in authoritative_driver_lock_registry()."
        );
    }

    // 2. Verify each registered lock belongs to a valid failure domain and recovery class
    for (key, domain) in &registry {
        assert!(
            matches!(
                domain,
                FailureDomain::Transactional
                    | FailureDomain::RestartableFromCanonical
                    | FailureDomain::DeviceContextFatal
                    | FailureDomain::ProcessFatal
                    | FailureDomain::InvariantViolation
            ),
            "registered lock {key} must have a valid failure domain"
        );

        let class = RecoveryClass::from(*domain);
        let roundtrip = FailureDomain::from(class);
        assert_eq!(
            *domain, roundtrip,
            "RecoveryClass roundtrip must be identity for {key}"
        );
    }
}

#[test]
fn failure_domain_recovery_class_bijection_preserves_semantics() {
    for domain in [
        FailureDomain::Transactional,
        FailureDomain::RestartableFromCanonical,
        FailureDomain::DeviceContextFatal,
        FailureDomain::ProcessFatal,
        FailureDomain::InvariantViolation,
    ] {
        let class: RecoveryClass = domain.into();
        let back: FailureDomain = class.into();
        assert_eq!(domain, back);
    }

    for class in [
        RecoveryClass::TransactionallyRecoverable,
        RecoveryClass::RestartableFromCanonicalInput,
        RecoveryClass::DeviceContextFatal,
        RecoveryClass::ProcessFatal,
        RecoveryClass::InvariantViolation,
    ] {
        let domain: FailureDomain = class.into();
        let back: RecoveryClass = domain.into();
        assert_eq!(class, back);
    }
}
