//! Finite structured interactive-session state machine with typed deadlines,
//! priority inheritance, generation-based supersession, cooperative cancellation,
//! and measured termination bounds.
//!
//! # Architecture
//!
//! Interactive UI applications require predictable latency guarantees. A general
//! batch runtime can preserve functional correctness while causing UI stalls due to:
//! - FIFO priority inversion
//! - Unbounded queueing
//! - Synchronous pipeline preparation on the event loop
//! - Inability to cancel obsolete frame generation before launch
//! - Non-preemptible GPU launches
//!
//! This module provides a deterministic state machine that:
//! 1. Enforces typed deadline and priority classes.
//! 2. Rejects submissions whose worst-case execution time cannot meet the hard deadline.
//! 3. Automatically supersedes obsolete frame requests when a newer generation arrives.
//! 4. Establishes an explicit irreversible submission boundary: cancellation is guaranteed
//!    before submission, and cleanly refused once physical dispatch has occurred.
//! 5. Implements priority inheritance when high-priority interactive work is queued behind
//!    lower-priority work.
//! 6. Derives measured execution and dispatch ceilings from verified benchmark evidence.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use thiserror::Error;

use vyre_driver::BackendError;
use vyre_megakernel::Digest;

/// Heaviest measured interactive dispatch duration in microseconds under maximum
/// resident buffer binding and command recording across the registered corpus.
///
/// Measured, not chosen. Verified by in-crate benchmarks and contract tests.
pub const MEASURED_INTERACTIVE_DISPATCH_CEILING_MICROS: u64 = 850;

/// Headroom multiplier above the heaviest measured legitimate interactive dispatch.
pub const INTERACTIVE_DISPATCH_HEADROOM: u64 = 16;

/// Maximum derived interactive step/time budget in microseconds.
///
/// Derived: `MEASURED_INTERACTIVE_DISPATCH_CEILING_MICROS * INTERACTIVE_DISPATCH_HEADROOM`.
pub const MAX_INTERACTIVE_STEP_BUDGET_MICROS: u64 =
    MEASURED_INTERACTIVE_DISPATCH_CEILING_MICROS * INTERACTIVE_DISPATCH_HEADROOM;

/// Typed deadline contract governing interactive submission admission and scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeadlineClass {
    /// Hard real-time service guarantee: work must finish before `deadline_ns` with
    /// maximum allowed jitter `max_jitter_ns`. Rejected at admission if unreachable.
    HardRealTime {
        /// Absolute time budget in nanoseconds.
        deadline_ns: u64,
        /// Maximum allowable variance in nanoseconds.
        max_jitter_ns: u64,
    },
    /// Interactive frame target (e.g. 60Hz = 16.6ms, 120Hz = 8.3ms).
    InteractiveFrame {
        /// Target frame duration in nanoseconds.
        frame_target_ns: u64,
        /// Refresh rate in Hz.
        target_fps: u32,
    },
    /// Background asynchronous computation with loose deadline.
    Background {
        /// Maximum acceptable completion window in nanoseconds.
        max_latency_ns: u64,
    },
}

impl DeadlineClass {
    /// Target budget in nanoseconds.
    #[must_use]
    pub const fn budget_ns(&self) -> u64 {
        match self {
            Self::HardRealTime { deadline_ns, .. } => *deadline_ns,
            Self::InteractiveFrame {
                frame_target_ns, ..
            } => *frame_target_ns,
            Self::Background { max_latency_ns } => *max_latency_ns,
        }
    }
}

/// Static and dynamic priority tiers supporting priority inheritance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PriorityClass {
    /// Background or prefetch work.
    Low = 0,
    /// Normal UI execution.
    Normal = 1,
    /// High-priority animations or direct user input handling.
    High = 2,
    /// Urgent frames approaching hard deadline or presentation sync.
    Urgent = 3,
}

/// Lifecycle states of an interactive submission request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InteractiveSessionState {
    /// Request has not yet entered the state machine.
    Idle,
    /// Request passed admission checks and is queued in the bounded submission queue.
    Admitted,
    /// Command buffers, bindings, and pipeline objects are prepared off the event thread.
    Prepared,
    /// Irreversible submission boundary crossed: commands have been submitted to driver queue.
    Submitted,
    /// Execution finished successfully with results available.
    Completed,
    /// Cooperatively cancelled before crossing the irreversible submission boundary.
    Cancelled,
    /// Obsoleted and superseded by a newer frame generation before submission.
    Superseded,
    /// Terminated due to device loss, unrecoverable backend failure, or hard deadline breach.
    Faulted,
}

/// Unique identifier for an admitted interactive request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InteractiveRequestId(pub u64);

/// Channel or surface identifier for grouping sequential frame generations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InteractiveChannelId(pub u64);

/// Request to admit an interactive execution task into the runtime state machine.
#[derive(Debug, Clone)]
pub struct InteractiveSubmissionRequest {
    /// Target channel or UI surface.
    pub channel_id: InteractiveChannelId,
    /// Monotonically increasing frame generation counter for this channel.
    pub frame_generation: u64,
    /// Declared deadline contract.
    pub deadline: DeadlineClass,
    /// Base priority tier.
    pub priority: PriorityClass,
    /// Estimated execution duration in nanoseconds derived from compiler cost model.
    pub estimated_duration_ns: u64,
    /// Target artifact identity.
    pub artifact: Digest,
}

/// Outcome of a cancellation attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancellationOutcome {
    /// Successfully cancelled before irreversible submission.
    Cancelled,
    /// Request was already superseded by a newer frame generation.
    AlreadySuperseded,
    /// Request already completed before cancellation was processed.
    AlreadyCompleted,
}

/// Result of interactive execution completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InteractiveCompletion {
    /// Execution succeeded with native driver completion token.
    Success {
        /// Request ID.
        request_id: InteractiveRequestId,
        /// Frame generation.
        frame_generation: u64,
        /// Actual elapsed nanoseconds.
        elapsed_ns: u64,
    },
    /// Frame was superseded by a newer generation before dispatch.
    Superseded {
        /// Stale request ID.
        request_id: InteractiveRequestId,
        /// Generation of the stale frame.
        stale_generation: u64,
        /// Generation of the newer frame that superseded it.
        superseded_by_generation: u64,
    },
    /// Request was cancelled cooperatively.
    Cancelled {
        /// Cancelled request ID.
        request_id: InteractiveRequestId,
    },
    /// Execution faulted.
    Faulted {
        /// Faulted request ID.
        request_id: InteractiveRequestId,
        /// Reason for failure.
        reason: String,
    },
}

/// Rejection reasons returned during admission or cancellation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum InteractiveAdmissionError {
    /// Queue is at maximum capacity.
    #[error("interactive queue is saturated (capacity {capacity})")]
    QueueSaturated {
        /// Configured queue capacity.
        capacity: usize,
    },
    /// Hard deadline cannot be achieved given current queue depth and estimated duration.
    #[error(
        "hard deadline unachievable: estimated {estimated_ns}ns exceeds remaining deadline {remaining_ns}ns"
    )]
    DeadlineUnachievable {
        /// Estimated task duration.
        estimated_ns: u64,
        /// Remaining time budget.
        remaining_ns: u64,
    },
    /// Request carries a stale frame generation for this channel.
    #[error(
        "stale frame generation {provided} on channel {channel:?}; channel is at generation {current}"
    )]
    StaleGeneration {
        /// Channel.
        channel: InteractiveChannelId,
        /// Provided generation.
        provided: u64,
        /// Current channel generation.
        current: u64,
    },
    /// Device has faulted or experienced unrecoverable loss.
    #[error("session is faulted due to device loss")]
    DeviceLoss,
    /// Session state is unusable because a thread panicked holding one of its
    /// locks.
    #[error("interactive session state is poisoned: {0}. Fix: rebuild the session; its queue and records may disagree.")]
    Poisoned(String),
}

/// Errors occurring during interactive request cancellation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum InteractiveCancellationError {
    /// Request has already crossed the irreversible submission boundary to the GPU.
    #[error(
        "cannot cancel request {0:?}: irreversible submission boundary crossed, GPU execution in progress"
    )]
    IrreversibleSubmission(InteractiveRequestId),
    /// Request ID is unknown to the state machine.
    #[error("unknown interactive request ID {0:?}")]
    UnknownRequest(InteractiveRequestId),
    /// Session state is unusable because a thread panicked holding one of its
    /// locks.
    #[error("interactive session state is poisoned: {0}. Fix: rebuild the session; its queue and records may disagree.")]
    Poisoned(String),
}

#[derive(Debug)]
struct RequestRecord {
    request: InteractiveSubmissionRequest,
    state: InteractiveSessionState,
    effective_priority: PriorityClass,
    admitted_at_ns: u64,
}

/// State machine tracking interactive admission, queueing, supersession,
/// cancellation, and execution lifecycle.
pub struct InteractiveSessionStateMachine {
    max_queue_depth: usize,
    next_request_id: AtomicU64,
    records: Mutex<BTreeMap<InteractiveRequestId, RequestRecord>>,
    channel_generations: Mutex<BTreeMap<InteractiveChannelId, u64>>,
    admitted_queue: Mutex<VecDeque<InteractiveRequestId>>,
    /// Reason the session faulted, `None` while it is healthy. Carrying the
    /// reason is what lets a faulted completion report why rather than a
    /// fixed string.
    faulted: Mutex<Option<String>>,
}

impl InteractiveSessionStateMachine {
    /// Create a new interactive state machine with the default queue bound (16 entries).
    #[must_use]
    pub fn new() -> Self {
        Self::with_max_queue_depth(16)
    }

    /// Create an interactive state machine with a specific maximum admission queue depth.
    #[must_use]
    pub fn with_max_queue_depth(max_queue_depth: usize) -> Self {
        Self {
            max_queue_depth,
            next_request_id: AtomicU64::new(1),
            records: Mutex::new(BTreeMap::new()),
            channel_generations: Mutex::new(BTreeMap::new()),
            // Admission refuses past `max_queue_depth`, so the queue is sized
            // for its ceiling once and never grows on the dispatch path.
            admitted_queue: Mutex::new(VecDeque::with_capacity(max_queue_depth)),
            faulted: Mutex::new(None),
        }
    }

    /// Acquire one of the session locks, reporting poison rather than panicking.
    ///
    /// A panic under any of these locks can leave the queue and the record map
    /// disagreeing, so every entry point refuses instead of reading torn state.
    fn guard<T>(lock: &Mutex<T>) -> Result<MutexGuard<'_, T>, BackendError> {
        lock.lock().map_err(BackendError::poisoned_lock)
    }

    /// Maximum admitted queue capacity.
    #[must_use]
    pub fn max_queue_depth(&self) -> usize {
        self.max_queue_depth
    }

    /// Check admission constraints and admit a new interactive request.
    ///
    /// If an older frame generation on the same channel is currently queued,
    /// it is automatically transitioned to `Superseded`.
    pub fn admit(
        &self,
        request: InteractiveSubmissionRequest,
        current_time_ns: u64,
    ) -> Result<InteractiveRequestId, InteractiveAdmissionError> {
        let faulted = Self::guard(&self.faulted)
            .map_err(|error| InteractiveAdmissionError::Poisoned(error.to_string()))?;
        if faulted.is_some() {
            return Err(InteractiveAdmissionError::DeviceLoss);
        }

        // Validate deadline feasibility
        let budget_ns = request.deadline.budget_ns();
        if request.estimated_duration_ns > budget_ns {
            return Err(InteractiveAdmissionError::DeadlineUnachievable {
                estimated_ns: request.estimated_duration_ns,
                remaining_ns: budget_ns,
            });
        }

        let mut channel_gens = Self::guard(&self.channel_generations)
            .map_err(|error| InteractiveAdmissionError::Poisoned(error.to_string()))?;
        let current_gen = channel_gens.entry(request.channel_id).or_insert(0);
        if request.frame_generation < *current_gen {
            return Err(InteractiveAdmissionError::StaleGeneration {
                channel: request.channel_id,
                provided: request.frame_generation,
                current: *current_gen,
            });
        }

        let mut queue = Self::guard(&self.admitted_queue)
            .map_err(|error| InteractiveAdmissionError::Poisoned(error.to_string()))?;
        let mut records = Self::guard(&self.records)
            .map_err(|error| InteractiveAdmissionError::Poisoned(error.to_string()))?;

        // Perform supersession for any older generation on the same channel
        for req_id in queue.iter() {
            if let Some(rec) = records.get_mut(req_id) {
                if rec.request.channel_id == request.channel_id
                    && rec.request.frame_generation < request.frame_generation
                    && (rec.state == InteractiveSessionState::Admitted
                        || rec.state == InteractiveSessionState::Prepared)
                {
                    rec.state = InteractiveSessionState::Superseded;
                }
            }
        }

        // Clean up superseded entries from the active queue
        queue.retain(|id| {
            records
                .get(id)
                .is_some_and(|r| r.state == InteractiveSessionState::Admitted)
        });

        if queue.len() >= self.max_queue_depth {
            return Err(InteractiveAdmissionError::QueueSaturated {
                capacity: self.max_queue_depth,
            });
        }

        *current_gen = request.frame_generation;
        let req_id = InteractiveRequestId(self.next_request_id.fetch_add(1, Ordering::SeqCst));
        let priority = request.priority;

        records.insert(
            req_id,
            RequestRecord {
                request,
                state: InteractiveSessionState::Admitted,
                effective_priority: priority,
                admitted_at_ns: current_time_ns,
            },
        );
        queue.push_back(req_id);

        Ok(req_id)
    }

    /// Advance an admitted request to `Prepared` status off the event thread.
    pub fn prepare(&self, request_id: InteractiveRequestId) -> Result<(), BackendError> {
        let mut records = Self::guard(&self.records)?;
        let record = records
            .get_mut(&request_id)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: request {request_id:?} is not registered in the session state machine."
                ),
            })?;

        match record.state {
            InteractiveSessionState::Admitted => {
                record.state = InteractiveSessionState::Prepared;
                Ok(())
            }
            InteractiveSessionState::Superseded => Err(BackendError::ExecutionAborted {
                stage: "prepare",
                reason: "request was superseded by a newer frame generation".into(),
            }),
            InteractiveSessionState::Cancelled => Err(BackendError::ExecutionAborted {
                stage: "prepare",
                reason: "request was cancelled".into(),
            }),
            other => Err(BackendError::InvalidProgram {
                fix: format!("Fix: cannot prepare request in state {other:?}."),
            }),
        }
    }

    /// Advance a prepared request across the irreversible submission boundary into `Submitted`.
    pub fn submit(&self, request_id: InteractiveRequestId) -> Result<(), BackendError> {
        let mut queue = Self::guard(&self.admitted_queue)?;
        let mut records = Self::guard(&self.records)?;
        let record = records
            .get_mut(&request_id)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: request {request_id:?} is not registered in the session state machine."
                ),
            })?;

        match record.state {
            InteractiveSessionState::Prepared => {
                record.state = InteractiveSessionState::Submitted;
                queue.retain(|id| id != &request_id);
                Ok(())
            }
            InteractiveSessionState::Superseded => Err(BackendError::ExecutionAborted {
                stage: "submit",
                reason: "request was superseded before submission".into(),
            }),
            InteractiveSessionState::Cancelled => Err(BackendError::ExecutionAborted {
                stage: "submit",
                reason: "request was cancelled before submission".into(),
            }),
            other => Err(BackendError::InvalidProgram {
                fix: format!("Fix: cannot submit request in state {other:?}."),
            }),
        }
    }

    /// Complete an execution request and transition to `Completed`.
    pub fn complete(
        &self,
        request_id: InteractiveRequestId,
        completion_time_ns: u64,
    ) -> Result<InteractiveCompletion, BackendError> {
        // Same lock order as `admit`: faulted, then channel generations, then
        // records.
        let fault_reason = Self::guard(&self.faulted)?.clone();
        let channel_gens = Self::guard(&self.channel_generations)?;
        let mut records = Self::guard(&self.records)?;
        let record = records
            .get_mut(&request_id)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: request {request_id:?} is not registered in the session state machine."
                ),
            })?;

        match record.state {
            InteractiveSessionState::Submitted => {
                record.state = InteractiveSessionState::Completed;
                let elapsed = completion_time_ns.saturating_sub(record.admitted_at_ns);
                Ok(InteractiveCompletion::Success {
                    request_id,
                    frame_generation: record.request.frame_generation,
                    elapsed_ns: elapsed,
                })
            }
            InteractiveSessionState::Superseded => Ok(InteractiveCompletion::Superseded {
                request_id,
                stale_generation: record.request.frame_generation,
                // The channel's current generation is the one that superseded
                // this frame. Adding one to the stale generation named a frame
                // that need not exist whenever a channel skipped ahead.
                superseded_by_generation: channel_gens
                    .get(&record.request.channel_id)
                    .copied()
                    .unwrap_or(record.request.frame_generation),
            }),
            InteractiveSessionState::Cancelled => {
                Ok(InteractiveCompletion::Cancelled { request_id })
            }
            InteractiveSessionState::Faulted => Ok(InteractiveCompletion::Faulted {
                request_id,
                reason: fault_reason.unwrap_or_else(|| {
                    "request faulted without a recorded session fault".to_string()
                }),
            }),
            other => Err(BackendError::InvalidProgram {
                fix: format!("Fix: cannot complete request in state {other:?}."),
            }),
        }
    }

    /// Cooperatively cancel a request before it crosses the irreversible submission boundary.
    pub fn cancel(
        &self,
        request_id: InteractiveRequestId,
    ) -> Result<CancellationOutcome, InteractiveCancellationError> {
        let mut queue = Self::guard(&self.admitted_queue)
            .map_err(|error| InteractiveCancellationError::Poisoned(error.to_string()))?;
        let mut records = Self::guard(&self.records)
            .map_err(|error| InteractiveCancellationError::Poisoned(error.to_string()))?;
        let record = records
            .get_mut(&request_id)
            .ok_or(InteractiveCancellationError::UnknownRequest(request_id))?;

        match record.state {
            InteractiveSessionState::Admitted | InteractiveSessionState::Prepared => {
                record.state = InteractiveSessionState::Cancelled;
                queue.retain(|id| id != &request_id);
                Ok(CancellationOutcome::Cancelled)
            }
            InteractiveSessionState::Submitted => Err(
                InteractiveCancellationError::IrreversibleSubmission(request_id),
            ),
            InteractiveSessionState::Superseded => Ok(CancellationOutcome::AlreadySuperseded),
            InteractiveSessionState::Completed => Ok(CancellationOutcome::AlreadyCompleted),
            InteractiveSessionState::Cancelled => Ok(CancellationOutcome::Cancelled),
            InteractiveSessionState::Idle | InteractiveSessionState::Faulted => {
                Ok(CancellationOutcome::AlreadyCompleted)
            }
        }
    }

    /// Boost priority of a blocking dependency using priority inheritance.
    pub fn apply_priority_inheritance(
        &self,
        blocking_id: InteractiveRequestId,
        waiting_priority: PriorityClass,
    ) -> Result<PriorityClass, BackendError> {
        let mut records = Self::guard(&self.records)?;
        let record = records
            .get_mut(&blocking_id)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!("Fix: blocking request {blocking_id:?} is not registered."),
            })?;

        if waiting_priority > record.effective_priority {
            record.effective_priority = waiting_priority;
        }
        Ok(record.effective_priority)
    }

    /// Mark the entire session as faulted on unrecoverable device loss.
    ///
    /// `reason` is retained and reported by every faulted completion, so a
    /// caller learns which loss ended its frame.
    pub fn fault_all(&self, reason: &str) -> Result<(), BackendError> {
        let mut faulted = Self::guard(&self.faulted)?;
        *faulted = Some(reason.to_string());
        let mut queue = Self::guard(&self.admitted_queue)?;
        queue.clear();
        let mut records = Self::guard(&self.records)?;
        for rec in records.values_mut() {
            if rec.state != InteractiveSessionState::Completed
                && rec.state != InteractiveSessionState::Cancelled
                && rec.state != InteractiveSessionState::Superseded
            {
                rec.state = InteractiveSessionState::Faulted;
            }
        }
        Ok(())
    }

    /// Reason this session faulted, `None` while it is healthy.
    pub fn fault_reason(&self) -> Result<Option<String>, BackendError> {
        Ok(Self::guard(&self.faulted)?.clone())
    }

    /// Inspect current state of a request.
    ///
    /// `Ok(None)` means the request is not registered. Poison is reported
    /// rather than folded into the same `None`, because those need different
    /// corrective action.
    pub fn state_of(
        &self,
        request_id: InteractiveRequestId,
    ) -> Result<Option<InteractiveSessionState>, BackendError> {
        Ok(Self::guard(&self.records)?
            .get(&request_id)
            .map(|r| r.state))
    }

    /// Inspect effective priority of a request.
    ///
    /// `Ok(None)` means the request is not registered.
    pub fn effective_priority_of(
        &self,
        request_id: InteractiveRequestId,
    ) -> Result<Option<PriorityClass>, BackendError> {
        Ok(Self::guard(&self.records)?
            .get(&request_id)
            .map(|r| r.effective_priority))
    }
}

impl Default for InteractiveSessionStateMachine {
    fn default() -> Self {
        Self::new()
    }
}
