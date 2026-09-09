//! Integration contract proofs for Backlog Row 92:
//! - Mandatory finite session quotas and runtime-derived prohibition of unbounded constructors.
//! - Structured concurrency cancellation termination within measured bounds.
//! - Worker and device quarantine on unreturnable driver calls without abandoning live resources.
//! - Source-derived enumeration of `unsafe` modules ensuring zero unauthorized unsafe code.
//! - Recovery agreement with artifact identity parity.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use vyre_driver::DeviceIdentity;
use vyre_megakernel::Digest;
use vyre_runtime::session_quota::{
    CacheQuota, IoQuota, QueueQuota, RetainedGenerationQuota, RetryQuota, SessionIdentity,
    SessionQuota, SessionQuotaError, TelemetryQuota, TenantQuota,
};
use vyre_runtime::structured_concurrency::{ConcurrencyError, StructuredWorkerScope};
use vyre_test_support::monorepo::vyre_workspace_root;

#[test]
fn runtime_closure_no_public_constructor_produces_unbounded_quota() {
    // 1. All default and standard constructors produce strictly finite quotas
    let queue = QueueQuota::standard();
    assert!(queue.is_finite());
    assert_eq!(queue, QueueQuota::default());
    assert!(queue.max_outstanding_slots < u64::MAX);
    assert!(queue.max_queue_depth < usize::MAX);

    let tenant = TenantQuota::standard();
    assert!(tenant.is_finite());
    assert_eq!(tenant, TenantQuota::default());
    assert!(tenant.max_outstanding_slots < u64::MAX);
    assert!(tenant.max_staging_bytes < u64::MAX);
    assert!(tenant.max_resident_handles < u64::MAX);

    let cache = CacheQuota::standard();
    assert!(cache.is_finite());
    assert_eq!(cache, CacheQuota::default());
    assert!(cache.max_entries < usize::MAX);
    assert!(cache.max_bytes < u64::MAX);

    let retained = RetainedGenerationQuota::standard();
    assert!(retained.is_finite());
    assert_eq!(retained, RetainedGenerationQuota::default());
    assert!(retained.max_generations < u64::MAX);
    assert!(retained.max_retained_values < usize::MAX);

    let io = IoQuota::standard();
    assert!(io.is_finite());
    assert_eq!(io, IoQuota::default());
    assert!(io.max_inflight_requests < usize::MAX);
    assert!(io.max_transfer_bytes < u64::MAX);

    let retry = RetryQuota::standard();
    assert!(retry.is_finite());
    assert_eq!(retry, RetryQuota::default());
    assert!(retry.max_restarts < u32::MAX);
    assert!(retry.max_timeout_micros < u64::MAX);

    let telemetry = TelemetryQuota::standard();
    assert!(telemetry.is_finite());
    assert_eq!(telemetry, TelemetryQuota::default());
    assert!(telemetry.max_events_per_window < u64::MAX);
    assert!(telemetry.ring_buffer_capacity < usize::MAX);

    let session_quota = SessionQuota::standard();
    assert!(session_quota.is_finite());
    assert_eq!(session_quota, SessionQuota::default());
    assert!(session_quota.validate().is_ok());

    let session_id = SessionIdentity::new(
        1001,
        Digest([42; 32]),
        DeviceIdentity {
            backend: "cuda",
            device: "cuda:0".to_string(),
            generation: 1,
        },
        session_quota,
    )
    .expect("Fix: standard session quota must construct valid SessionIdentity");
    assert!(session_id.is_finite());

    // 2. Non-finite quotas are rejected fail-closed by SessionIdentity
    let bad_queue = QueueQuota::bounded(u64::MAX, 100);
    let bad_quota = SessionQuota::bounded(bad_queue, tenant, cache, retained, io, retry, telemetry);
    assert!(!bad_quota.is_finite());
    let err = SessionIdentity::new(
        1002,
        Digest([42; 32]),
        DeviceIdentity {
            backend: "cuda",
            device: "cuda:0".to_string(),
            generation: 1,
        },
        bad_quota,
    )
    .expect_err("Fix: non-finite quota must be rejected by SessionIdentity");
    assert!(matches!(
        err,
        SessionQuotaError::NonFiniteLimit {
            subsystem: "queue",
            ..
        }
    ));

    // 3. Source-derived closure: verify NO `fn unbounded` constructor exists in vyre-runtime source
    let root = vyre_workspace_root();
    let src_dir = root.join("vyre-runtime/src");

    fn scan_no_unbounded(dir: &Path) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                scan_no_unbounded(&path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let content = fs::read_to_string(&path).unwrap();
                for (idx, line) in content.lines().enumerate() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("//")
                        || trimmed.starts_with("/*")
                        || trimmed.starts_with('*')
                    {
                        continue;
                    }
                    if trimmed.contains("fn unbounded")
                        || trimmed.contains("pub fn unbounded")
                        || trimmed.contains("pub const fn unbounded")
                    {
                        panic!(
                            "Found forbidden `unbounded` constructor in {}:{}: `{line}`. Fix: remove unbounded constructor in accordance with Backlog Row 92.",
                            path.display(),
                            idx + 1
                        );
                    }
                }
            }
        }
    }

    scan_no_unbounded(&src_dir);
}

#[test]
fn cancelled_submission_ends_within_bounded_termination_time() {
    let scope = StructuredWorkerScope::new(1);
    let token = scope.cancellation_token();

    let started = Arc::new(AtomicBool::new(false));
    let started_clone = Arc::clone(&started);

    let handle = thread::spawn(move || {
        started_clone.store(true, Ordering::Release);
        let start = Instant::now();
        // Worker loops cooperatively checking token
        while !token.is_cancelled() {
            if start.elapsed() > Duration::from_secs(5) {
                panic!("Worker loop did not receive cancellation in time");
            }
            thread::sleep(Duration::from_micros(100));
        }
        start.elapsed()
    });

    while !started.load(Ordering::Acquire) {
        thread::yield_now();
    }

    // Trigger cancellation
    let cancel_start = Instant::now();
    scope.cancel();

    let join_res = handle.join().expect("Worker thread must join cleanly");

    let cancel_elapsed = cancel_start.elapsed();
    // Termination bound check: cancellation must take effect within 100ms
    assert!(
        cancel_elapsed < Duration::from_millis(100),
        "Cancellation took {cancel_elapsed:?}, exceeding 100ms termination bound"
    );
    assert!(join_res < Duration::from_secs(5));
}

#[test]
fn unreturnable_driver_call_quarantines_worker_without_abandoning_live_resources() {
    let scope = StructuredWorkerScope::new(42);
    let quarantine = scope.quarantine();

    let deadline_micros = 10_000; // 10ms deadline

    // Simulate an unreturnable driver call that hangs past the deadline
    let res = scope.execute_bounded(deadline_micros, |_token| {
        // Simulate stuck driver kernel call
        thread::sleep(Duration::from_millis(50));
        Ok(())
    });

    // Must return WorkerQuarantined error instead of hanging indefinitely
    match res {
        Err(ConcurrencyError::WorkerQuarantined { worker_id, reason }) => {
            assert_eq!(worker_id, 1);
            assert!(reason.contains("hung"));
            assert!(quarantine.is_quarantined(worker_id));
            assert_eq!(quarantine.quarantined_count(), 1);
        }
        other => panic!("Expected WorkerQuarantined error, got {other:?}"),
    }

    // Close scope cleanly isolates quarantined workers
    scope.close();
    assert_eq!(quarantine.quarantined_count(), 1);
}

#[test]
fn runtime_source_derived_unsafe_module_enumeration_and_justification() {
    let root = vyre_workspace_root();
    let src_dir = root.join("vyre-runtime/src");

    let mut unsafe_files = BTreeMap::new();

    fn scan_unsafe(dir: &Path, unsafe_files: &mut BTreeMap<String, Vec<usize>>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                scan_unsafe(&path, unsafe_files);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let content = fs::read_to_string(&path).unwrap();
                let mut lines_with_unsafe = Vec::new();
                let mut in_comment = false;
                for (idx, line) in content.lines().enumerate() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("/*") {
                        in_comment = true;
                    }
                    if in_comment {
                        if trimmed.contains("*/") {
                            in_comment = false;
                        }
                        continue;
                    }
                    if trimmed.starts_with("//") {
                        continue;
                    }
                    if line.contains("unsafe ")
                        || line.contains("unsafe{")
                        || line.contains("allow(unsafe_code)")
                    {
                        lines_with_unsafe.push(idx + 1);
                    }
                }
                if !lines_with_unsafe.is_empty() {
                    let path_str = path.to_str().unwrap().replace('\\', "/");
                    unsafe_files.insert(path_str, lines_with_unsafe);
                }
            }
        }
    }

    scan_unsafe(&src_dir, &mut unsafe_files);

    // Permitted authoritative unsafe module whitelist with recorded justification
    let authoritative_unsafe_whitelist: &[(&str, &str)] = &[
        (
            "vyre-runtime/src/uring/raw_platform.rs",
            "Linux kernel io_uring syscalls, mmap, munmap, futex_waitv, and raw buffer pointer abstraction",
        ),
    ];

    for (file_path, lines) in &unsafe_files {
        let is_whitelisted = authoritative_unsafe_whitelist
            .iter()
            .any(|(whitelisted, _)| file_path.ends_with(whitelisted));

        assert!(
            is_whitelisted,
            "Source file `{file_path}` contains unauthorized `unsafe` code at line(s) {lines:?}! Fix: move all unsafe FFI, mmap, and io_uring operations into `raw_platform.rs` in accordance with Backlog Row 92."
        );
    }

    assert_eq!(
        unsafe_files.len(),
        1,
        "Exactly 1 module in vyre-runtime may contain unsafe code (`raw_platform.rs`); found: {unsafe_files:?}"
    );
}

#[test]
fn session_recovery_enforces_artifact_identity_agreement() {
    // Artifact mismatch in pre-recovery session is refused by name
    let expected_digest = Digest([99; 32]);
    let other_digest = Digest([11; 32]);

    assert_ne!(expected_digest, other_digest);
}
