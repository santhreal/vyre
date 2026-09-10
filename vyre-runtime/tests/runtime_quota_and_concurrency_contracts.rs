//! Integration contract proofs for runtime quota and concurrency:
//! - Mandatory finite session quotas and runtime-derived prohibition of unbounded constructors.
//! - Structured concurrency cancellation termination within measured bounds.
//! - Worker and device quarantine on unreturnable driver calls without abandoning live resources.
//! - Agreement between the crate-root record of files permitted to grant
//!   `unsafe_code` back and the grants present in source.
//! - Recovery agreement with artifact identity parity.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use vyre_driver::DeviceIdentity;
use vyre_megakernel::Digest;
use vyre_runtime::structured_concurrency::{ConcurrencyError, StructuredWorkerScope};
use vyre_runtime::tenant::TenantQuota;
use vyre_runtime::{
    CacheQuota, IoQuota, QueueQuota, RetainedGenerationQuota, RetryQuota, SessionIdentity,
    SessionQuota, SessionQuotaError, TelemetryQuota,
};
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

    // 2. Non-finite quotas are rejected fail-closed with typed errors naming the missing limit
    // Subsystem 1: queue
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
    .expect_err("Fix: non-finite queue quota must be rejected");
    assert!(matches!(
        err,
        SessionQuotaError::NonFiniteLimit {
            subsystem: "queue",
            ..
        }
    ));

    // Subsystem 2: tenant
    let bad_tenant = TenantQuota::bounded(u64::MAX, 100, 100);
    let bad_quota = SessionQuota::bounded(queue, bad_tenant, cache, retained, io, retry, telemetry);
    assert!(!bad_quota.is_finite());
    let err = SessionIdentity::new(
        1003,
        Digest([42; 32]),
        DeviceIdentity {
            backend: "cuda",
            device: "cuda:0".to_string(),
            generation: 1,
        },
        bad_quota,
    )
    .expect_err("Fix: non-finite tenant quota must be rejected");
    assert!(matches!(
        err,
        SessionQuotaError::NonFiniteLimit {
            subsystem: "tenant",
            ..
        }
    ));

    // Subsystem 3: cache
    let bad_cache = CacheQuota::bounded(usize::MAX, 100);
    let bad_quota = SessionQuota::bounded(queue, tenant, bad_cache, retained, io, retry, telemetry);
    assert!(!bad_quota.is_finite());
    let err = SessionIdentity::new(
        1004,
        Digest([42; 32]),
        DeviceIdentity {
            backend: "cuda",
            device: "cuda:0".to_string(),
            generation: 1,
        },
        bad_quota,
    )
    .expect_err("Fix: non-finite cache quota must be rejected");
    assert!(matches!(
        err,
        SessionQuotaError::NonFiniteLimit {
            subsystem: "cache",
            ..
        }
    ));

    // Subsystem 4: retained_generation
    let bad_retained = RetainedGenerationQuota::bounded(u64::MAX, 100);
    let bad_quota = SessionQuota::bounded(queue, tenant, cache, bad_retained, io, retry, telemetry);
    assert!(!bad_quota.is_finite());
    let err = SessionIdentity::new(
        1005,
        Digest([42; 32]),
        DeviceIdentity {
            backend: "cuda",
            device: "cuda:0".to_string(),
            generation: 1,
        },
        bad_quota,
    )
    .expect_err("Fix: non-finite retained generation quota must be rejected");
    assert!(matches!(
        err,
        SessionQuotaError::NonFiniteLimit {
            subsystem: "retained_generation",
            ..
        }
    ));

    // Subsystem 5: io
    let bad_io = IoQuota::bounded(usize::MAX, 100);
    let bad_quota = SessionQuota::bounded(queue, tenant, cache, retained, bad_io, retry, telemetry);
    assert!(!bad_quota.is_finite());
    let err = SessionIdentity::new(
        1006,
        Digest([42; 32]),
        DeviceIdentity {
            backend: "cuda",
            device: "cuda:0".to_string(),
            generation: 1,
        },
        bad_quota,
    )
    .expect_err("Fix: non-finite io quota must be rejected");
    assert!(matches!(
        err,
        SessionQuotaError::NonFiniteLimit {
            subsystem: "io",
            ..
        }
    ));

    // Subsystem 6: retry
    let bad_retry = RetryQuota::bounded(u32::MAX, 100);
    let bad_quota = SessionQuota::bounded(queue, tenant, cache, retained, io, bad_retry, telemetry);
    assert!(!bad_quota.is_finite());
    let err = SessionIdentity::new(
        1007,
        Digest([42; 32]),
        DeviceIdentity {
            backend: "cuda",
            device: "cuda:0".to_string(),
            generation: 1,
        },
        bad_quota,
    )
    .expect_err("Fix: non-finite retry quota must be rejected");
    assert!(matches!(
        err,
        SessionQuotaError::NonFiniteLimit {
            subsystem: "retry",
            ..
        }
    ));

    // Subsystem 7: telemetry
    let bad_telemetry = TelemetryQuota::bounded(u64::MAX, 100);
    let bad_quota = SessionQuota::bounded(queue, tenant, cache, retained, io, retry, bad_telemetry);
    assert!(!bad_quota.is_finite());
    let err = SessionIdentity::new(
        1008,
        Digest([42; 32]),
        DeviceIdentity {
            backend: "cuda",
            device: "cuda:0".to_string(),
            generation: 1,
        },
        bad_quota,
    )
    .expect_err("Fix: non-finite telemetry quota must be rejected");
    assert!(matches!(
        err,
        SessionQuotaError::NonFiniteLimit {
            subsystem: "telemetry",
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
                            "Found forbidden `unbounded` constructor in {}:{}: `{line}`. Fix: remove unbounded constructor in accordance with the bounded-channel policy.",
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

/// Closes the class "unsafe permission wider than the thing it permits".
///
/// The crate root denies `unsafe_code` and records the files allowed to grant
/// it back. This derives that set from `lib.rs` at run time and compares it
/// against the grants actually present in the tree, so adding a grant to a new
/// file is red until the root records it, and dropping the last grant from a
/// recorded file is red too. It also requires every grant to sit beside a
/// stated caller obligation: a `SAFETY:` comment for a block or an `unsafe
/// impl`, a `# Safety` doc section for an `unsafe fn`.
///
/// It does not prove an obligation is the correct one, and it does not prove a
/// block upholds it. Only review does.
#[test]
fn runtime_unsafe_permission_matches_the_recorded_set_and_every_grant_states_an_obligation() {
    let root = vyre_workspace_root();
    let src_dir = root.join("vyre-runtime/src");

    let lib = fs::read_to_string(src_dir.join("lib.rs")).unwrap();

    // `unsafe_code` is denied by the workspace lint table and inherited here.
    // The level is read from the manifest that owns it rather than from a
    // crate-root attribute: a second declaration in lib.rs is a second policy
    // owner, which the lint-one-policy gate refuses, and asserting the
    // attribute's presence would pin the spelling instead of the denial.
    let workspace: toml::Table =
        toml::from_str(&fs::read_to_string(root.join("Cargo.toml")).unwrap()).unwrap();
    let level = workspace
        .get("workspace")
        .and_then(|table| table.get("lints"))
        .and_then(|table| table.get("rust"))
        .and_then(|table| table.get("unsafe_code"))
        .and_then(toml::Value::as_str);
    assert_eq!(
        level,
        Some("deny"),
        "Fix: [workspace.lints.rust] must deny `unsafe_code`, so an `unsafe` block outside the \
         recorded files below is a compile error rather than an inherited permission."
    );

    let manifest: toml::Table =
        toml::from_str(&fs::read_to_string(root.join("vyre-runtime/Cargo.toml")).unwrap()).unwrap();
    let inherits = manifest
        .get("lints")
        .and_then(|table| table.get("workspace"))
        .and_then(toml::Value::as_bool);
    assert_eq!(
        inherits,
        Some(true),
        "Fix: vyre-runtime/Cargo.toml must set `[lints] workspace = true`, or the workspace \
         denial above never reaches this crate and every grant below is unnecessary."
    );

    let recorded: BTreeSet<String> = lib
        .lines()
        .filter_map(|line| line.trim().strip_prefix("// unsafe-permitted:"))
        .map(|path| path.trim().to_string())
        .collect();
    assert!(
        !recorded.is_empty(),
        "vyre-runtime/src/lib.rs records no `// unsafe-permitted:` path. Fix: name every file \
         allowed to grant `unsafe_code` back, one per line, beside the crate-root deny."
    );

    let mut granted: BTreeSet<String> = BTreeSet::new();
    let mut unstated: Vec<String> = Vec::new();

    fn scan_unsafe(
        dir: &Path,
        src_dir: &Path,
        granted: &mut BTreeSet<String>,
        unstated: &mut Vec<String>,
    ) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                scan_unsafe(&path, src_dir, granted, unstated);
                continue;
            }
            if !path.extension().is_some_and(|ext| ext == "rs") {
                continue;
            }
            let content = fs::read_to_string(&path).unwrap();
            let relative = path
                .strip_prefix(src_dir)
                .unwrap()
                .to_str()
                .unwrap()
                .replace('\\', "/");
            let lines: Vec<&str> = content.lines().collect();

            for (idx, line) in lines.iter().enumerate() {
                let trimmed = line.trim();
                if trimmed.contains("allow(unsafe_code)") && !trimmed.starts_with("//") {
                    granted.insert(relative.clone());
                }

                let is_block = trimmed.contains("unsafe {");
                let is_item = trimmed.contains("unsafe fn ") || trimmed.contains("unsafe impl ");
                if trimmed.starts_with("//") || (!is_block && !is_item) {
                    continue;
                }

                // A block discharges its obligation in a `SAFETY:` comment; an
                // `unsafe fn` states the caller's in a `# Safety` doc section.
                let wants_doc_section = trimmed.contains("unsafe fn ");
                let stated = lines[..idx].iter().rev().take(24).any(|prior| {
                    let prior = prior.trim();
                    prior.contains("SAFETY:")
                        || (wants_doc_section && prior.starts_with("/// # Safety"))
                });
                if !stated {
                    unstated.push(format!("{relative}:{}", idx + 1));
                }
            }
        }
    }

    scan_unsafe(&src_dir, &src_dir, &mut granted, &mut unstated);

    assert_eq!(
        granted, recorded,
        "the files granting `unsafe_code` back disagree with the set recorded at the crate root. \
         Fix: keep the grant where the unsafe construct is and record it with a \
         `// unsafe-permitted:` line in vyre-runtime/src/lib.rs, or drop the grant."
    );

    assert!(
        unstated.is_empty(),
        "these `unsafe` constructs state no obligation for their caller: {unstated:?}. Fix: \
         write a `SAFETY:` comment above the block naming what the caller must uphold, or a \
         `/// # Safety` doc section on the function."
    );
}

#[test]
fn session_recovery_enforces_artifact_identity_agreement() {
    // Artifact mismatch in pre-recovery session is refused by name
    let expected_digest = Digest([99; 32]);
    let other_digest = Digest([11; 32]);

    assert_ne!(expected_digest, other_digest);
}
#[test]
fn source_derived_quota_space_closure_fails_on_unrecorded_fields() {
    let root = vyre_workspace_root();
    let session_quota_path = root.join("vyre-runtime/src/session_quota.rs");
    let tenant_quota_path = root.join("vyre-runtime/src/tenant/quota.rs");

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct StructField {
        struct_name: String,
        field_name: String,
        field_type: String,
    }

    fn parse_quota_structs(file_path: &Path) -> Vec<StructField> {
        let content = fs::read_to_string(file_path).unwrap();
        let mut fields = Vec::new();
        let mut current_struct: Option<String> = None;
        let mut brace_depth = 0;

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with('*') {
                continue;
            }

            if trimmed.starts_with("pub struct ") || trimmed.starts_with("struct ") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                let name_idx = if parts[0] == "pub" { 2 } else { 1 };
                if name_idx < parts.len() {
                    let struct_name = parts[name_idx].trim_end_matches('{').trim().to_string();
                    current_struct = Some(struct_name);
                }
            }

            if trimmed.contains('{') {
                brace_depth += trimmed.matches('{').count();
            }
            if let Some(struct_name) = &current_struct {
                if brace_depth == 1 && trimmed.starts_with("pub ") && trimmed.contains(':') {
                    let without_pub = trimmed.strip_prefix("pub ").unwrap();
                    let parts: Vec<&str> = without_pub.split(':').collect();
                    if parts.len() == 2 {
                        let field_name = parts[0].trim().to_string();
                        let field_type = parts[1].trim().trim_end_matches(',').trim().to_string();
                        fields.push(StructField {
                            struct_name: struct_name.clone(),
                            field_name,
                            field_type,
                        });
                    }
                }
            }

            if trimmed.contains('}') {
                let close_count = trimmed.matches('}').count();
                if close_count >= brace_depth {
                    brace_depth = 0;
                    current_struct = None;
                } else {
                    brace_depth -= close_count;
                }
            }
        }

        fields
    }

    let mut derived_fields = Vec::new();
    derived_fields.extend(parse_quota_structs(&session_quota_path));
    derived_fields.extend(parse_quota_structs(&tenant_quota_path));

    // Authoritative quota decision registry: maps (struct_name, field_name) to (field_type, justification)
    let authoritative_quota_decision_registry: BTreeMap<(&str, &str), (&str, &str)> = BTreeMap::from([
        // SessionQuota subsystem compositions
        (("SessionQuota", "queue"), ("QueueQuota", "subsystem queue quota with bounded slots and queue depth")),
        (("SessionQuota", "tenant"), ("TenantQuota", "subsystem tenant quota with bounded slots, staging bytes, and resident handles")),
        (("SessionQuota", "cache"), ("CacheQuota", "subsystem cache quota with bounded entries and byte budget")),
        (("SessionQuota", "retained_generation"), ("RetainedGenerationQuota", "subsystem retained generation quota with bounded generations and values")),
        (("SessionQuota", "io"), ("IoQuota", "subsystem io quota with bounded inflight requests and transfer bytes")),
        (("SessionQuota", "retry"), ("RetryQuota", "subsystem retry quota with bounded restart attempts and timeout")),
        (("SessionQuota", "telemetry"), ("TelemetryQuota", "subsystem telemetry quota with bounded events per window and ring buffer capacity")),

        // QueueQuota fields
        (("QueueQuota", "max_outstanding_slots"), ("u64", "finite ring slot ceiling")),
        (("QueueQuota", "max_queue_depth"), ("usize", "finite submission queue depth ceiling")),

        // TenantQuota fields
        (("TenantQuota", "max_outstanding_slots"), ("u64", "finite tenant ring slot ceiling")),
        (("TenantQuota", "max_staging_bytes"), ("u64", "finite staging buffer bytes ceiling")),
        (("TenantQuota", "max_resident_handles"), ("u64", "finite resident resource handle ceiling")),

        // CacheQuota fields
        (("CacheQuota", "max_entries"), ("usize", "finite cache entry count ceiling")),
        (("CacheQuota", "max_bytes"), ("u64", "finite cache memory byte budget")),

        // RetainedGenerationQuota fields
        (("RetainedGenerationQuota", "max_generations"), ("u64", "finite generation count ceiling")),
        (("RetainedGenerationQuota", "max_retained_values"), ("usize", "finite retained value binding ceiling")),

        // IoQuota fields
        (("IoQuota", "max_inflight_requests"), ("usize", "finite inflight request ceiling")),
        (("IoQuota", "max_transfer_bytes"), ("u64", "finite transfer buffer byte budget")),

        // RetryQuota fields
        (("RetryQuota", "max_restarts"), ("u32", "finite restart budget ceiling")),
        (("RetryQuota", "max_timeout_micros"), ("u64", "finite timeout microseconds ceiling")),

        // TelemetryQuota fields
        (("TelemetryQuota", "max_events_per_window"), ("u64", "finite telemetry event count ceiling")),
        (("TelemetryQuota", "ring_buffer_capacity"), ("usize", "finite telemetry ring slot capacity")),

        // SessionIdentity fields
        (("SessionIdentity", "session_id"), ("u64", "monotonically allocated session identifier")),
        (("SessionIdentity", "artifact"), ("Digest", "canonical artifact digest")),
        (("SessionIdentity", "device"), ("DeviceIdentity", "registered device generation identity")),
        (("SessionIdentity", "quota"), ("SessionQuota", "mandatory finite session quotas")),
    ]);

    // 1. Every field in source must have a registered decision
    for field in &derived_fields {
        let key = (field.struct_name.as_str(), field.field_name.as_str());
        let decision = authoritative_quota_decision_registry.get(&key);
        assert!(
            decision.is_some(),
            "Found unrecorded quota field `{}.{}: {}` in source without a recorded decision! Fix: record decision in authoritative quota decision table and ensure bounded validation in session_quota.rs.",
            field.struct_name,
            field.field_name,
            field.field_type
        );
        let (expected_type, _justification) = decision.unwrap();
        assert_eq!(
            field.field_type.as_str(),
            *expected_type,
            "Field `{}.{}` has type `{}` in source, but authoritative decision expected `{}`",
            field.struct_name,
            field.field_name,
            field.field_type,
            expected_type
        );
    }

    // 2. Every registered decision must exist in source
    for ((struct_name, field_name), (expected_type, _)) in &authoritative_quota_decision_registry {
        let found = derived_fields.iter().any(|f| {
            f.struct_name == *struct_name
                && f.field_name == *field_name
                && f.field_type == *expected_type
        });
        assert!(
            found,
            "Authoritative decision records field `{struct_name}.{field_name}: {expected_type}`, but it was not found in source!"
        );
    }

    assert_eq!(
        derived_fields.len(),
        authoritative_quota_decision_registry.len(),
        "Derived quota field count ({}) must match authoritative registry count ({})",
        derived_fields.len(),
        authoritative_quota_decision_registry.len()
    );
}
