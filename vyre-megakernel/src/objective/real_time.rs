//! Real-time interactive objective family, measurement boundaries, and hard-constraint verification (Row 110).
//!
//! Throughput-oriented GPU compilation can fail an interactive application when mean kernel duration
//! omits queueing, submission, command encoding, pipeline creation, synchronization, wake-up,
//! presentation handoff, and tail behavior.
//!
//! This module defines versioned real-time objectives specifying deadlines, jitter, queueing limits,
//! arrival traces, memory ceilings, energy policies, and complete input-to-visible measurement records.

use serde::{Deserialize, Serialize};

use crate::cost::CostBreakdown;
use crate::DeviceFacts;

/// Current real-time objective schema version.
pub const REAL_TIME_OBJECTIVE_SCHEMA_VERSION: u16 = 1;

/// Declared deadline contract for real-time interactive compilation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealTimeDeadline {
    /// Hard real-time service guarantee: work must complete within `deadline_ns`
    /// with maximum permissible jitter `max_jitter_ns`.
    HardRealTime {
        /// Absolute time budget in nanoseconds.
        deadline_ns: u64,
        /// Maximum allowed variance across successive dispatches in nanoseconds.
        max_jitter_ns: u64,
    },
    /// Interactive frame target (e.g. 60Hz = 16.6ms, 120Hz = 8.3ms, 240Hz = 4.16ms).
    InteractiveFrame {
        /// Target frame duration in nanoseconds.
        frame_target_ns: u64,
        /// Target refresh rate in frames per second.
        target_fps: u32,
    },
    /// Soft real-time or background task with loose deadline bounds.
    Background {
        /// Maximum acceptable completion window in nanoseconds.
        max_latency_ns: u64,
    },
}

impl RealTimeDeadline {
    /// Return the deadline budget in nanoseconds.
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

    /// Return the maximum allowed jitter in nanoseconds, if bounded.
    #[must_use]
    pub const fn max_jitter_ns(&self) -> Option<u64> {
        match self {
            Self::HardRealTime { max_jitter_ns, .. } => Some(*max_jitter_ns),
            _ => None,
        }
    }
}

/// Latency percentile targeted by the real-time objective.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatencyPercentile {
    /// 50th percentile (median).
    P50,
    /// 90th percentile.
    P90,
    /// 95th percentile.
    P95,
    /// 99th percentile.
    P99,
    /// 99.9th percentile.
    P999,
    /// Strict worst-case observed latency.
    WorstCase,
}

/// State of device caches, allocations, and pipelines when measuring latency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarmColdState {
    /// Steady-state warm execution (resident allocations and warm pipelines).
    WarmOnly,
    /// Cold-start dispatches included in latency assessment.
    ColdIncluded,
    /// Transition between distinct pipeline layouts or memory topologies.
    Transitional,
}

/// Device energy and power envelope policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnergyPolicy {
    /// Unconstrained performance mode.
    Unconstrained,
    /// Peak sustained throughput within thermal envelope.
    PeakPerformance,
    /// Balanced power-efficiency mode.
    Balanced,
    /// Low-power mode for battery/embedded constraints.
    PowerSave,
    /// Thermally throttled operation.
    ThermalThrottled,
}

/// Workload interference assumptions on the target device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterferenceAssumptions {
    /// Exclusively dedicated GPU queues and hardware engines.
    Isolated,
    /// Shared GPU queues with competing compute kernels.
    SharedQueue,
    /// Concurrent presentation engine and compositor activity.
    ConcurrentPresentation,
    /// Heavy background asynchronous compute work.
    HeavyBackgroundWork,
}

/// Workload arrival distribution model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadArrivalTrace {
    /// Periodic arrival with fixed period and maximum arrival jitter.
    Periodic {
        /// Base period between arrivals in nanoseconds.
        period_ns: u64,
        /// Maximum arrival jitter in nanoseconds.
        jitter_ns: u64,
    },
    /// Bursty arrivals with batch size and inter-burst interval.
    Burst {
        /// Number of concurrent events in each burst.
        burst_size: u32,
        /// Interval between bursts in nanoseconds.
        interval_ns: u64,
    },
    /// Poisson distributed arrivals with declared mean interval.
    Poisson {
        /// Mean time between arrivals in nanoseconds.
        mean_interval_ns: u64,
    },
    /// Uniformly distributed random arrivals in range `[min_interval_ns, max_interval_ns]`.
    UniformRandom {
        /// Minimum interval in nanoseconds.
        min_interval_ns: u64,
        /// Maximum interval in nanoseconds.
        max_interval_ns: u64,
    },
}

/// Versioned real-time objective specification (Row 110).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RealTimeObjective {
    version: u16,
    deadline: RealTimeDeadline,
    percentile: LatencyPercentile,
    jitter_limit_ns: Option<u64>,
    queueing_limit_ns: Option<u64>,
    admission_limit_ns: Option<u64>,
    warm_cold_state: WarmColdState,
    compile_budget_us: Option<u64>,
    pipeline_creation_budget_us: Option<u64>,
    memory_ceiling_bytes: Option<u64>,
    energy_policy: EnergyPolicy,
    interference: InterferenceAssumptions,
    arrival_trace: WorkloadArrivalTrace,
}

impl RealTimeObjective {
    /// Create a real-time objective targeting a 60 FPS interactive frame deadline (16.6ms).
    #[must_use]
    pub const fn interactive_60fps() -> Self {
        Self {
            version: REAL_TIME_OBJECTIVE_SCHEMA_VERSION,
            deadline: RealTimeDeadline::InteractiveFrame {
                frame_target_ns: 16_666_667,
                target_fps: 60,
            },
            percentile: LatencyPercentile::P99,
            jitter_limit_ns: Some(2_000_000), // 2ms jitter cap
            queueing_limit_ns: Some(4_000_000),
            admission_limit_ns: Some(1_000_000),
            warm_cold_state: WarmColdState::WarmOnly,
            compile_budget_us: Some(50_000),
            pipeline_creation_budget_us: Some(5_000),
            memory_ceiling_bytes: None,
            energy_policy: EnergyPolicy::PeakPerformance,
            interference: InterferenceAssumptions::ConcurrentPresentation,
            arrival_trace: WorkloadArrivalTrace::Periodic {
                period_ns: 16_666_667,
                jitter_ns: 1_000_000,
            },
        }
    }

    /// Create a real-time objective targeting a 120 FPS high-refresh frame deadline (8.33ms).
    #[must_use]
    pub const fn interactive_120fps() -> Self {
        Self {
            version: REAL_TIME_OBJECTIVE_SCHEMA_VERSION,
            deadline: RealTimeDeadline::InteractiveFrame {
                frame_target_ns: 8_333_333,
                target_fps: 120,
            },
            percentile: LatencyPercentile::P999,
            jitter_limit_ns: Some(1_000_000), // 1ms jitter cap
            queueing_limit_ns: Some(2_000_000),
            admission_limit_ns: Some(500_000),
            warm_cold_state: WarmColdState::WarmOnly,
            compile_budget_us: Some(25_000),
            pipeline_creation_budget_us: Some(2_500),
            memory_ceiling_bytes: None,
            energy_policy: EnergyPolicy::PeakPerformance,
            interference: InterferenceAssumptions::ConcurrentPresentation,
            arrival_trace: WorkloadArrivalTrace::Periodic {
                period_ns: 8_333_333,
                jitter_ns: 500_000,
            },
        }
    }

    /// Create a hard real-time objective with strict absolute deadline and jitter limit.
    #[must_use]
    pub const fn hard_real_time(deadline_ns: u64, max_jitter_ns: u64) -> Self {
        Self {
            version: REAL_TIME_OBJECTIVE_SCHEMA_VERSION,
            deadline: RealTimeDeadline::HardRealTime {
                deadline_ns,
                max_jitter_ns,
            },
            percentile: LatencyPercentile::WorstCase,
            jitter_limit_ns: Some(max_jitter_ns),
            queueing_limit_ns: Some(deadline_ns / 4),
            admission_limit_ns: Some(deadline_ns / 8),
            warm_cold_state: WarmColdState::WarmOnly,
            compile_budget_us: Some(10_000),
            pipeline_creation_budget_us: Some(1_000),
            memory_ceiling_bytes: None,
            energy_policy: EnergyPolicy::Unconstrained,
            interference: InterferenceAssumptions::Isolated,
            arrival_trace: WorkloadArrivalTrace::Periodic {
                period_ns: deadline_ns,
                jitter_ns: max_jitter_ns,
            },
        }
    }

    /// Return the objective schema version.
    #[must_use]
    pub const fn version(&self) -> u16 {
        self.version
    }

    /// Return the declared deadline.
    #[must_use]
    pub const fn deadline(&self) -> RealTimeDeadline {
        self.deadline
    }

    /// Return the targeted latency percentile.
    #[must_use]
    pub const fn percentile(&self) -> LatencyPercentile {
        self.percentile
    }

    /// Return the jitter limit in nanoseconds.
    #[must_use]
    pub const fn jitter_limit_ns(&self) -> Option<u64> {
        self.jitter_limit_ns
    }

    /// Return the queueing limit in nanoseconds.
    #[must_use]
    pub const fn queueing_limit_ns(&self) -> Option<u64> {
        self.queueing_limit_ns
    }

    /// Return the memory ceiling in bytes.
    #[must_use]
    pub const fn memory_ceiling_bytes(&self) -> Option<u64> {
        self.memory_ceiling_bytes
    }

    /// Set a memory ceiling in bytes.
    #[must_use]
    pub const fn with_memory_ceiling(mut self, bytes: u64) -> Self {
        self.memory_ceiling_bytes = Some(bytes);
        self
    }

    /// Set a compilation budget in microseconds.
    #[must_use]
    pub const fn with_compile_budget_us(mut self, budget_us: u64) -> Self {
        self.compile_budget_us = Some(budget_us);
        self
    }

    /// Set an arrival trace.
    #[must_use]
    pub const fn with_arrival_trace(mut self, trace: WorkloadArrivalTrace) -> Self {
        self.arrival_trace = trace;
        self
    }

    /// Check whether a candidate schedule's cost breakdown satisfies all hard real-time constraints.
    ///
    /// Hard constraints include:
    /// 1. Estimated launch duration must not exceed deadline budget.
    /// 2. Launch duration must not violate the jitter ceiling.
    /// 3. Memory residency must not exceed memory ceiling.
    ///
    /// # Errors
    pub fn satisfies_constraints(
        &self,
        cost: &CostBreakdown,
        _device: DeviceFacts,
    ) -> Result<(), RealTimeViolation> {
        let launch_duration_ns = cost.total;
        let budget_ns = self.deadline.budget_ns();

        if launch_duration_ns > budget_ns {
            return Err(RealTimeViolation::DeadlineExceeded {
                limit_ns: budget_ns,
                achieved_ns: launch_duration_ns,
            });
        }

        if let Some(jitter_cap) = self.jitter_limit_ns {
            // Worst-case synchronization/rendezvous overhead as variance floor
            let estimated_variance_ns = cost
                .barriers
                .saturating_mul(1_000)
                .saturating_add(cost.grid_syncs.saturating_mul(10_000));
            if estimated_variance_ns > jitter_cap {
                return Err(RealTimeViolation::JitterExceeded {
                    limit_ns: jitter_cap,
                    achieved_ns: estimated_variance_ns,
                });
            }
        }

        if let Some(mem_cap) = self.memory_ceiling_bytes {
            let peak_bytes = cost.planned_peak_bytes;
            if peak_bytes > mem_cap {
                return Err(RealTimeViolation::MemoryCeilingExceeded {
                    limit_bytes: mem_cap,
                    achieved_bytes: peak_bytes,
                });
            }
        }

        Ok(())
    }
}

/// Hard constraint violation during real-time candidate filtering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RealTimeViolation {
    /// Schedule execution time exceeded hard deadline.
    DeadlineExceeded {
        /// Stated limit in nanoseconds.
        limit_ns: u64,
        /// Estimated/achieved figure in nanoseconds.
        achieved_ns: u64,
    },
    /// Schedule synchronization or launch variance exceeded jitter limit.
    JitterExceeded {
        /// Stated limit in nanoseconds.
        limit_ns: u64,
        /// Estimated/achieved figure in nanoseconds.
        achieved_ns: u64,
    },
    /// Schedule queueing delay exceeded admission limit.
    QueueingExceeded {
        /// Stated limit in nanoseconds.
        limit_ns: u64,
        /// Estimated/achieved figure in nanoseconds.
        achieved_ns: u64,
    },
    /// Schedule peak memory exceeded memory ceiling.
    MemoryCeilingExceeded {
        /// Stated ceiling in bytes.
        limit_bytes: u64,
        /// Achieved bytes.
        achieved_bytes: u64,
    },
}

impl core::fmt::Display for RealTimeViolation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DeadlineExceeded {
                limit_ns,
                achieved_ns,
            } => {
                write!(f, "hard deadline exceeded: {achieved_ns}ns > {limit_ns}ns")
            }
            Self::JitterExceeded {
                limit_ns,
                achieved_ns,
            } => {
                write!(f, "jitter limit exceeded: {achieved_ns}ns > {limit_ns}ns")
            }
            Self::QueueingExceeded {
                limit_ns,
                achieved_ns,
            } => {
                write!(
                    f,
                    "queueing delay limit exceeded: {achieved_ns}ns > {limit_ns}ns"
                )
            }
            Self::MemoryCeilingExceeded {
                limit_bytes,
                achieved_bytes,
            } => {
                write!(
                    f,
                    "memory ceiling exceeded: {achieved_bytes}B > {limit_bytes}B"
                )
            }
        }
    }
}

/// Complete measurement record covering the full boundary from admitted input event
/// through externally observed presentation completion.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputToVisibleMeasurement {
    /// Timestamp when input event was admitted into the session (nanoseconds).
    pub admitted_timestamp_ns: u64,
    /// Duration spent encoding commands into device-native format (nanoseconds).
    pub command_encoding_ns: u64,
    /// Duration spent waiting in driver submission queue (nanoseconds).
    pub queue_wait_ns: u64,
    /// Duration spent creating or binding pipeline objects (nanoseconds).
    pub pipeline_creation_ns: u64,
    /// Duration spent in physical GPU kernel execution (nanoseconds).
    pub device_kernel_ns: u64,
    /// Duration spent in device-to-device and fence synchronization (nanoseconds).
    pub synchronization_ns: u64,
    /// Duration from kernel completion to visible presentation handoff (nanoseconds).
    pub presentation_handoff_ns: u64,
    /// Total end-to-end input-to-visible latency (nanoseconds).
    pub total_latency_ns: u64,
    /// Raw latency samples recorded across dispatches.
    pub raw_samples_ns: Vec<u64>,
    /// Queue depth observed at time of admission.
    pub queue_depth_at_admission: usize,
    /// Measurement uncertainty in nanoseconds (clock resolution & jitter).
    pub uncertainty_ns: u64,
    /// Whether any hard deadline was missed during the measurement run.
    pub missed_deadlines_count: u32,
}

impl InputToVisibleMeasurement {
    /// Compute a complete measurement record from individual stage durations.
    #[must_use]
    pub fn new(
        admitted_timestamp_ns: u64,
        command_encoding_ns: u64,
        queue_wait_ns: u64,
        pipeline_creation_ns: u64,
        device_kernel_ns: u64,
        synchronization_ns: u64,
        presentation_handoff_ns: u64,
        raw_samples_ns: Vec<u64>,
        queue_depth_at_admission: usize,
        deadline_ns: Option<u64>,
    ) -> Self {
        let total_latency_ns = command_encoding_ns
            + queue_wait_ns
            + pipeline_creation_ns
            + device_kernel_ns
            + synchronization_ns
            + presentation_handoff_ns;

        let missed_deadlines_count = match deadline_ns {
            Some(limit) => raw_samples_ns
                .iter()
                .filter(|&&sample| sample > limit)
                .count() as u32,
            None => 0,
        };

        Self {
            admitted_timestamp_ns,
            command_encoding_ns,
            queue_wait_ns,
            pipeline_creation_ns,
            device_kernel_ns,
            synchronization_ns,
            presentation_handoff_ns,
            total_latency_ns,
            raw_samples_ns,
            queue_depth_at_admission,
            uncertainty_ns: 50, // 50ns clock resolution
            missed_deadlines_count,
        }
    }

    /// Calculate the p99 latency in nanoseconds across raw samples.
    #[must_use]
    pub fn p99_latency_ns(&self) -> u64 {
        if self.raw_samples_ns.is_empty() {
            return self.total_latency_ns;
        }
        let mut sorted = self.raw_samples_ns.clone();
        sorted.sort_unstable();
        let idx = (sorted.len() * 99) / 100;
        sorted[idx.min(sorted.len() - 1)]
    }

    /// Calculate worst-case latency in nanoseconds.
    #[must_use]
    pub fn worst_case_latency_ns(&self) -> u64 {
        self.raw_samples_ns
            .iter()
            .copied()
            .max()
            .unwrap_or(self.total_latency_ns)
    }

    /// Calculate latency jitter (max - min) in nanoseconds.
    #[must_use]
    pub fn jitter_ns(&self) -> u64 {
        if self.raw_samples_ns.is_empty() {
            return 0;
        }
        let min = *self.raw_samples_ns.iter().min().unwrap_or(&0);
        let max = *self.raw_samples_ns.iter().max().unwrap_or(&0);
        max.saturating_sub(min)
    }
}
