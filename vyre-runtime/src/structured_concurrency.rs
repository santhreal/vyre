//! Structured concurrency, cancellation propagation, and worker/device quarantine.
//!
//! Structured concurrency owns every worker and device operation, propagates cancellation,
//! and recycles or quarantines a worker or device after an unreturnable driver call
//! rather than abandoning threads with live resources.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use thiserror::Error;
use vyre_driver::BackendError;
use vyre_foundation::reclaim_poisoned_irreplaceable_state;

/// The subsystem every poison report in this module names as the owner.
const OWNER: &str = "runtime structured concurrency";

/// Structured concurrency error variants.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ConcurrencyError {
    /// Operation was cancelled cooperatively before or during execution.
    #[error(
        "operation was cancelled cooperatively. Fix: discard partial state and handle cancellation"
    )]
    Cancelled,
    /// Operation execution deadline exceeded.
    #[error("operation exceeded deadline ({elapsed_micros}µs / {deadline_micros}µs). Fix: increase execution quota or optimize operation")]
    DeadlineExceeded {
        /// Elapsed time in microseconds.
        elapsed_micros: u64,
        /// Configured deadline in microseconds.
        deadline_micros: u64,
    },
    /// Worker or device call hung and was quarantined to prevent resource leak.
    #[error("worker {worker_id} hung in driver call and was quarantined: {reason}. Fix: restart worker process and release device bindings")]
    WorkerQuarantined {
        /// Quarantined worker id.
        worker_id: u64,
        /// Reason for quarantine.
        reason: String,
    },
    /// Worker thread panicked before returning.
    #[error("worker thread panicked: {0}. Fix: investigate panicking code")]
    WorkerPanicked(String),
}

/// Cooperative cancellation token supporting hierarchy and propagation.
#[derive(Debug, Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Create a new active cancellation token.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Trigger cancellation.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Check if cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

/// Status of a managed worker in the structured concurrency scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerStatus {
    /// Worker is currently executing.
    Active,
    /// Worker finished successfully.
    Completed,
    /// Worker terminated cleanly after cancellation.
    Cancelled,
    /// Worker encountered an unreturnable driver call and is quarantined.
    Quarantined {
        /// Diagnostic reason for quarantine.
        reason: String,
        /// Monotonic timestamp in nanoseconds when quarantined.
        timestamp_ns: u64,
    },
}

/// Thread-safe registry tracking quarantined workers and devices.
#[derive(Debug, Default)]
pub struct WorkerQuarantine {
    workers: Mutex<BTreeMap<u64, WorkerStatus>>,
}

impl WorkerQuarantine {
    /// Create a new worker quarantine registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            workers: Mutex::new(BTreeMap::new()),
        }
    }

    fn lock_workers(&self) -> std::sync::MutexGuard<'_, BTreeMap<u64, WorkerStatus>> {
        reclaim_poisoned_irreplaceable_state(
            self.workers.lock(),
            || self.workers.clear_poison(),
            OWNER,
            "the worker quarantine registry",
        )
    }

    /// Register a worker as active.
    pub fn register_worker(&self, worker_id: u64) {
        self.lock_workers().insert(worker_id, WorkerStatus::Active);
    }

    /// Quarantine a worker following an unreturnable driver call or hang.
    pub fn quarantine_worker(&self, worker_id: u64, reason: String, timestamp_ns: u64) {
        self.lock_workers().insert(
            worker_id,
            WorkerStatus::Quarantined {
                reason,
                timestamp_ns,
            },
        );
    }

    /// Mark a worker completed.
    pub fn mark_completed(&self, worker_id: u64) {
        self.lock_workers()
            .insert(worker_id, WorkerStatus::Completed);
    }

    /// Check if a worker is quarantined.
    #[must_use]
    pub fn is_quarantined(&self, worker_id: u64) -> bool {
        matches!(
            self.lock_workers().get(&worker_id),
            Some(WorkerStatus::Quarantined { .. })
        )
    }

    /// Count currently quarantined workers.
    #[must_use]
    pub fn quarantined_count(&self) -> usize {
        self.lock_workers()
            .values()
            .filter(|status| matches!(status, WorkerStatus::Quarantined { .. }))
            .count()
    }

    /// Count currently active workers.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.lock_workers()
            .values()
            .filter(|status| matches!(status, WorkerStatus::Active))
            .count()
    }
}

/// Supervisor scope guaranteeing that all spawned workers are tracked,
/// cancellation is propagated, and hung/unreturnable driver calls are quarantined.
pub struct StructuredWorkerScope {
    scope_id: u64,
    next_worker_id: AtomicU64,
    token: CancellationToken,
    quarantine: Arc<WorkerQuarantine>,
    workers: Mutex<Vec<(u64, JoinHandle<()>)>>,
}

impl StructuredWorkerScope {
    /// Create a new structured worker scope.
    #[must_use]
    pub fn new(scope_id: u64) -> Self {
        Self {
            scope_id,
            next_worker_id: AtomicU64::new(1),
            token: CancellationToken::new(),
            quarantine: Arc::new(WorkerQuarantine::new()),
            workers: Mutex::new(Vec::new()),
        }
    }

    /// Create a structured worker scope sharing an existing quarantine registry.
    #[must_use]
    pub fn with_quarantine(scope_id: u64, quarantine: Arc<WorkerQuarantine>) -> Self {
        Self {
            scope_id,
            next_worker_id: AtomicU64::new(1),
            token: CancellationToken::new(),
            quarantine,
            workers: Mutex::new(Vec::new()),
        }
    }

    /// Scope identifier.
    #[must_use]
    pub fn scope_id(&self) -> u64 {
        self.scope_id
    }

    /// Cancellation token for this scope.
    #[must_use]
    pub fn cancellation_token(&self) -> CancellationToken {
        self.token.clone()
    }

    /// Quarantine registry reference.
    #[must_use]
    pub fn quarantine(&self) -> Arc<WorkerQuarantine> {
        Arc::clone(&self.quarantine)
    }

    /// Execute a supervised bounded device or driver operation with cancellation and deadline.
    ///
    /// If the operation completes within the deadline, its result is returned.
    /// If cancelled, [`ConcurrencyError::Cancelled`] is returned.
    /// If an unreturnable driver call hangs past the deadline, the worker is quarantined
    /// and [`ConcurrencyError::WorkerQuarantined`] is returned, ensuring no thread
    /// outlives the session holding live resources without quarantine.
    pub fn execute_bounded<T: Send + 'static>(
        &self,
        deadline_micros: u64,
        operation: impl FnOnce(&CancellationToken) -> Result<T, BackendError> + Send + 'static,
    ) -> Result<T, ConcurrencyError> {
        if self.token.is_cancelled() {
            return Err(ConcurrencyError::Cancelled);
        }

        let worker_id = self.next_worker_id.fetch_add(1, Ordering::SeqCst);
        self.quarantine.register_worker(worker_id);

        let token = self.token.clone();
        let quarantine = Arc::clone(&self.quarantine);

        let result_slot = Arc::new(Mutex::new(None));
        let result_slot_clone = Arc::clone(&result_slot);

        let handle = thread::spawn(move || {
            let res = operation(&token);
            let mut slot = reclaim_poisoned_irreplaceable_state(
                result_slot_clone.lock(),
                || result_slot_clone.clear_poison(),
                OWNER,
                "one bounded worker's result slot",
            );
            *slot = Some(res);
            quarantine.mark_completed(worker_id);
        });

        let start = Instant::now();
        let timeout = Duration::from_micros(deadline_micros);
        let poll_step = Duration::from_micros(100);

        while start.elapsed() < timeout {
            if self.token.is_cancelled() {
                return Err(ConcurrencyError::Cancelled);
            }

            {
                let mut slot = reclaim_poisoned_irreplaceable_state(
                    result_slot.lock(),
                    || result_slot.clear_poison(),
                    OWNER,
                    "one bounded worker's result slot",
                );
                if let Some(res) = slot.take() {
                    let _ = handle.join();
                    return res.map_err(|err| ConcurrencyError::WorkerPanicked(err.to_string()));
                }
            }

            thread::sleep(poll_step);
        }

        // Operation hung or exceeded deadline  -  quarantine worker
        let elapsed = start.elapsed().as_micros() as u64;
        self.quarantine.quarantine_worker(
            worker_id,
            format!("Driver call hung after {elapsed}µs (deadline: {deadline_micros}µs)"),
            elapsed,
        );

        let mut workers = reclaim_poisoned_irreplaceable_state(
            self.workers.lock(),
            || self.workers.clear_poison(),
            OWNER,
            "the scope's live worker join handles",
        );
        workers.push((worker_id, handle));

        Err(ConcurrencyError::WorkerQuarantined {
            worker_id,
            reason: format!("Driver call hung after {elapsed}µs"),
        })
    }

    /// Cancel all workers in this scope.
    pub fn cancel(&self) {
        self.token.cancel();
    }

    /// Ensure no thread outlives the session holding live resources without recorded quarantine.
    pub fn close(&self) {
        self.cancel();
        let mut workers = reclaim_poisoned_irreplaceable_state(
            self.workers.lock(),
            || self.workers.clear_poison(),
            OWNER,
            "the scope's live worker join handles",
        );
        for (id, handle) in workers.drain(..) {
            if !self.quarantine.is_quarantined(id) {
                let _ = handle.join();
            }
        }
    }
}

impl Drop for StructuredWorkerScope {
    fn drop(&mut self) {
        self.close();
    }
}
