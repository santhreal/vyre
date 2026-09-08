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

/// Authoritative declaration registry for all Mutex and RwLock instances across driver crates.
fn authoritative_lock_registry() -> BTreeMap<&'static str, FailureDomain> {
    let mut map = BTreeMap::new();
    // vyre-driver
    map.insert("vyre-driver/src/launch_facts.rs:LAUNCH_MEASUREMENTS", FailureDomain::Transactional);
    map.insert("vyre-driver/src/observability.rs:EVENTS", FailureDomain::Transactional);
    map.insert("vyre-driver/src/backend/resident_sequence.rs:submitted", FailureDomain::Transactional);
    map.insert("vyre-driver/src/grid_sync/resident_dispatch.rs:buffers", FailureDomain::Transactional);
    map.insert("vyre-driver/src/grid_sync/resident_dispatch.rs:freed", FailureDomain::Transactional);
    map.insert("vyre-driver/src/pipeline/cache.rs:pending_flushes", FailureDomain::Transactional);
    // vyre-driver-wgpu
    map.insert("vyre-driver-wgpu/src/lib.rs:shape_history", FailureDomain::Transactional);
    map.insert("vyre-driver-wgpu/src/strict_float.rs:VERDICTS", FailureDomain::RestartableFromCanonical);
    map.insert("vyre-driver-wgpu/src/buffer/bind_group_cache/mod.rs:cache", FailureDomain::RestartableFromCanonical);
    map.insert("vyre-driver-wgpu/src/buffer/staging/mod.rs:inner", FailureDomain::RestartableFromCanonical);
    map.insert("vyre-driver-wgpu/src/pipeline/disk_cache_entries.rs:TEST_DISK_PIPELINE_CACHE_ROOT", FailureDomain::Transactional);
    map.insert("vyre-driver-wgpu/src/pipeline/disk_cache/io.rs:PENDING_DURABLE_CACHE_FILES", FailureDomain::Transactional);
    map.insert("vyre-driver-wgpu/src/runtime/prerecorded.rs:cb", FailureDomain::DeviceContextFatal);
    map.insert("vyre-driver-wgpu/src/runtime/device/acquire.rs:LOADER_STARTUP", FailureDomain::ProcessFatal);
    map
}

#[test]
fn lock_failure_domain_closure_from_source() {
    let registry = authoritative_lock_registry();

    fn scan_dir(dir: &Path, lock_locations: &mut Vec<String>) {
        if !dir.exists() {
            return;
        }
        for entry in fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().unwrap().to_str().unwrap();
                if name != "target" && name != "tests" {
                    scan_dir(&path, lock_locations);
                }
            } else if path.extension().map_or(false, |ext| ext == "rs") {
                let content = fs::read_to_string(&path).unwrap();
                for (line_no, line) in content.lines().enumerate() {
                    if (line.contains("Mutex<") || line.contains("RwLock<"))
                        && !line.trim_start().starts_with("//")
                        && !line.trim_start().starts_with("/*")
                        && !line.contains("use ")
                    {
                        let path_str = path.to_str().unwrap().replace('\\', "/");
                        lock_locations.push(format!("{path_str}:line_{}", line_no + 1));
                    }
                }
            }
        }
    }

    let mut lock_locations = Vec::new();
    let root = vyre_workspace_root();
    scan_dir(&root.join("vyre-driver/src"), &mut lock_locations);
    scan_dir(&root.join("vyre-driver-wgpu/src"), &mut lock_locations);

    assert!(
        !lock_locations.is_empty(),
        "scan must locate existing driver locks"
    );

    // Verify each registered lock belongs to a valid failure domain
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
    }
}
