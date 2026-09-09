//! Tests for explicit failure domains, atomic guarded states, prepare/commit journals, and restart budgets.
//!
//! WHY: proves Row 122:
//! - Poisoned locks transition atomically to typed PoisonedTerminal state and reject further operations.
//! - State can be rebuilt cleanly from canonical input via explicit recover().
//! - Prepare/commit journals guarantee side effect idempotency under repeated calls.
//! - Supervised restart budgets enforce worker crash bounds and fail closed.
//! - Source-derived runtime closure: every mutable state owner in vyre-runtime has a registered failure domain and recovery class.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs;
use std::panic;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use vyre_foundation::{FailureDomain, RecoveryClass, RecoveryDisposition};
use vyre_megakernel::{Digest, RealTimeDeadline};
use vyre_runtime::artifact_admission::{
    InteractiveAdmissionError, InteractiveChannelId, InteractiveCompletion,
    InteractiveSessionStateMachine, InteractiveSubmissionRequest, PriorityClass,
};
use vyre_runtime::{
    authoritative_runtime_state_owner_registry, AtomicGuardedState, GuardedState,
    PrepareCommitJournal, SupervisedRestartBudget,
};
use vyre_test_support::monorepo::vyre_workspace_root;

#[test]
fn atomic_guarded_state_transitions_to_poisoned_terminal_on_panic() {
    let state = Arc::new(AtomicGuardedState::new(
        vec![1, 2, 3],
        FailureDomain::MemoryState,
        RecoveryClass::RestartableFromCanonicalInput,
    ));

    // Initial state is observable as Ready
    assert_eq!(state.current_state(), GuardedState::Ready(vec![1, 2, 3]));

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

    // Lifecycle state is observable as PoisonedTerminal
    match state.current_state() {
        GuardedState::PoisonedTerminal {
            domain,
            recovery_class,
            reason,
        } => {
            assert_eq!(domain, FailureDomain::MemoryState);
            assert_eq!(recovery_class, RecoveryClass::RestartableFromCanonicalInput);
            assert!(reason.contains("poisoned"));
        }
        other => panic!("Expected PoisonedTerminal variant, got {other:?}"),
    }

    // Explicit recovery restores state to Ready
    state.recover(vec![10, 20]);
    assert_eq!(state.current_state(), GuardedState::Ready(vec![10, 20]));
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

    let ticket = journal.prepare(key.clone(), 100).unwrap().unwrap();

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

    assert_eq!(budget.max_restarts(), 3);
    assert_eq!(budget.remaining_restarts(), 3);

    assert_eq!(
        budget.record_restart(FailureDomain::WorkerProcess).unwrap(),
        1
    );
    assert_eq!(budget.remaining_restarts(), 2);

    assert_eq!(
        budget.record_restart(FailureDomain::WorkerProcess).unwrap(),
        2
    );
    assert_eq!(budget.remaining_restarts(), 1);

    assert_eq!(
        budget.record_restart(FailureDomain::WorkerProcess).unwrap(),
        3
    );
    assert_eq!(budget.remaining_restarts(), 0);

    // 4th restart exceeds ceiling of 3
    let err = budget
        .record_restart(FailureDomain::WorkerProcess)
        .expect_err("Fix: exceeding restart budget must fail closed with ProcessFatal error.");

    assert_eq!(err.domain, FailureDomain::WorkerProcess);
    assert_eq!(err.recovery_class, RecoveryClass::ProcessFatal);
    assert_eq!(err.disposition, RecoveryDisposition::Fatal);
    assert!(err.fix.contains("Fix:"));

    // Reset restores budget
    budget.reset();
    assert_eq!(budget.current_restarts(), 0);
    assert_eq!(budget.remaining_restarts(), 3);
}

#[test]
fn source_derived_mutable_state_owner_closure_test() {
    let registry = authoritative_runtime_state_owner_registry();

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
                if file_name == "tests.rs" || file_name.ends_with("_tests.rs") {
                    continue;
                }
                let content = fs::read_to_string(&path).unwrap();
                let mut lock_count = 0;
                let mut in_test_mod = false;
                let mut test_mod_depth = 0;
                let mut pending_test_cfg = false;
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("//")
                        || trimmed.starts_with("/*")
                        || trimmed.starts_with('*')
                    {
                        continue;
                    }
                    if trimmed.starts_with("#[cfg(test)]") {
                        pending_test_cfg = true;
                        continue;
                    }
                    if pending_test_cfg {
                        if trimmed.starts_with("mod ") {
                            in_test_mod = true;
                            test_mod_depth = 0;
                        }
                        pending_test_cfg = false;
                    }
                    if in_test_mod {
                        test_mod_depth += trimmed.matches('{').count();
                        let close_count = trimmed.matches('}').count();
                        if close_count >= test_mod_depth {
                            in_test_mod = false;
                            test_mod_depth = 0;
                        } else {
                            test_mod_depth -= close_count;
                        }
                        continue;
                    }
                    if (line.contains("Mutex<")
                        || line.contains("RwLock<")
                        || line.contains("DashMap<"))
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
    scan_dir(&root.join("vyre-runtime/src"), &mut lock_files);

    assert!(
        !lock_files.is_empty(),
        "scan must locate existing runtime state owners from source"
    );

    // 1. Verify every source file with locks/state owners is covered in the authoritative registry
    for (file_path, count) in &lock_files {
        let has_entry = registry.keys().any(|key| {
            let prefix = key.split(':').next().unwrap_or("");
            file_path.ends_with(prefix)
        });
        assert!(
            has_entry,
            "Source file {file_path} contains {count} mutable state owner(s) but has no declared FailureDomain / RecoveryClass in authoritative registry! Fix: register failure domain in authoritative_runtime_state_owner_registry()."
        );
    }

    // 2. Verify each registered owner has a valid failure domain and recovery class
    for (key, (domain, class)) in &registry {
        assert!(
            FailureDomain::ALL.contains(domain),
            "registered state owner {key} must have a valid FailureDomain"
        );
        assert!(
            RecoveryClass::ALL.contains(class),
            "registered state owner {key} must have a valid RecoveryClass"
        );
    }

    // 3. Verify vyre-megakernel contains no uncatalogued mutable state
    let mut megakernel_locks = BTreeMap::new();
    scan_dir(&root.join("vyre-megakernel/src"), &mut megakernel_locks);
    assert!(
        megakernel_locks.is_empty(),
        "vyre-megakernel must remain pure and immutable; found unexpected mutable state in: {megakernel_locks:?}"
    );
}

#[test]
fn idempotent_commit_with_simulated_crash_produces_exact_single_side_effect() {
    let journal = PrepareCommitJournal::<String, u64>::new();
    let key = String::from("idempotent_publish_4040");

    let side_effect_executions = Arc::new(AtomicUsize::new(0));

    // Phase 1: Prepare
    let ticket = journal
        .prepare(key.clone(), 0)
        .expect("Fix: prepare must succeed")
        .expect("Fix: ticket returned");

    // Phase 2: Commit with simulated crash right after first attempt
    let execs_clone = Arc::clone(&side_effect_executions);
    let committed_val_1 = journal
        .commit_idempotent(key.clone(), ticket, || -> Result<u64, String> {
            execs_clone.fetch_add(1, Ordering::SeqCst);
            Ok(9999)
        })
        .expect("Fix: first commit must succeed");

    assert_eq!(committed_val_1, 9999);
    assert_eq!(
        side_effect_executions.load(Ordering::SeqCst),
        1,
        "side effect must execute exactly once on first commit"
    );

    // Simulated retry after crash: caller retries the exact same commit with the same idempotency key
    let execs_clone_2 = Arc::clone(&side_effect_executions);
    let committed_val_2 = journal
        .commit_idempotent(key.clone(), ticket, || -> Result<u64, String> {
            execs_clone_2.fetch_add(1, Ordering::SeqCst);
            Ok(9999)
        })
        .expect("Fix: retry commit must succeed idempotently");

    assert_eq!(committed_val_2, 9999);
    assert_eq!(
        side_effect_executions.load(Ordering::SeqCst),
        1,
        "Fix: repeated commit must NEVER duplicate side effect; expected 1 execution, got {}",
        side_effect_executions.load(Ordering::SeqCst)
    );

    // If a different/wrong key is supplied with an un-prepared ticket, it fails and does NOT execute
    let execs_clone_3 = Arc::clone(&side_effect_executions);
    let invalid_res = journal.commit_idempotent(
        String::from("unknown_key"),
        ticket,
        || -> Result<u64, String> {
            execs_clone_3.fetch_add(1, Ordering::SeqCst);
            Ok(1111)
        },
    );
    assert!(invalid_res.is_err());
    assert_eq!(
        side_effect_executions.load(Ordering::SeqCst),
        1,
        "unprepared key must reject and not execute side effect"
    );
}

#[test]
fn interactive_session_state_machine_fault_injection_and_idempotency() {
    let sm = InteractiveSessionStateMachine::new();
    let channel = InteractiveChannelId(1);

    let req1 = InteractiveSubmissionRequest {
        channel_id: channel,
        frame_generation: 1,
        deadline: RealTimeDeadline::InteractiveFrame {
            frame_target_ns: 16_666_666,
            target_fps: 60,
        },
        priority: PriorityClass::Normal,
        estimated_duration_ns: 50_000,
        artifact: Digest([1; 32]),
    };

    let id1 = sm
        .admit(req1, 1_000_000)
        .expect("Fix: admission must succeed");
    sm.prepare(id1).expect("Fix: prepare must succeed");
    sm.submit(id1).expect("Fix: submit must succeed");

    let completion = sm
        .complete(id1, 2_000_000)
        .expect("Fix: complete must succeed");
    assert!(
        matches!(completion, InteractiveCompletion::Success { request_id, .. } if request_id == id1)
    );
    // Fault injection: simulate device loss / state machine fault
    sm.fault_all("Simulated GPU device reset")
        .expect("Fix: fault_all must succeed");

    let req2 = InteractiveSubmissionRequest {
        channel_id: channel,
        frame_generation: 2,
        deadline: RealTimeDeadline::InteractiveFrame {
            frame_target_ns: 16_666_666,
            target_fps: 60,
        },
        priority: PriorityClass::Normal,
        estimated_duration_ns: 50_000,
        artifact: Digest([2; 32]),
    };

    let admit_err = sm
        .admit(req2, 3_000_000)
        .expect_err("Fix: faulted session must reject admission");
    assert!(matches!(admit_err, InteractiveAdmissionError::DeviceLoss));
}

#[test]
fn bounded_cleanup_and_supervision_ceilings() {
    let journal = PrepareCommitJournal::<String, u64>::new();

    // Prepare 20 items
    for i in 0..20 {
        let _ = journal.prepare(format!("item_{i}"), i).unwrap();
    }

    // Cleanup with bound of 5 removes at most 5 items
    let cleaned = journal.cleanup_stale_prepared(0, 5);
    assert_eq!(
        cleaned, 5,
        "cleanup must be strictly bounded by limit parameter"
    );

    // Second cleanup cleans the next 5 items
    let cleaned_2 = journal.cleanup_stale_prepared(0, 5);
    assert_eq!(cleaned_2, 5);
}
