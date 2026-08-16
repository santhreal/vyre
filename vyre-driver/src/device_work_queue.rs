#![allow(unused_imports)]
//! Backend-neutral device-side work queue planning for dependent dataflow execution.

use crate::numeric::BackendNumericPolicy;

const DEVICE_WORK_QUEUE_NUMERIC: BackendNumericPolicy =
    BackendNumericPolicy::new("device work queue");

/// Host synchronization policy for a device device-side work queue.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkQueueHostSync {
    /// Host reads only final completion state after device-side draining.
    FinalOnly,
    /// Host participates during queue draining.
    HostParticipates,
}

/// Work queue workload profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceWorkQueueProfile {
    /// Initial active work items enqueued before launch.
    pub initial_items: u64,
    /// Maximum resident queue capacity in work items.
    pub queue_capacity: u64,
    /// ABI bytes per queue entry.
    pub entry_bytes: u64,
    /// Bytes required for queue head/tail counters and changed flags.
    pub control_bytes: u64,
    /// Caller-approved device-memory budget.
    pub budget_bytes: u64,
    /// Host synchronization policy.
    pub host_sync: WorkQueueHostSync,
}

/// Work queue profile where a resident queue should reserve device-side
/// expansion headroom in addition to the initial frontier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceWorkQueueExpansionProfile {
    /// Initial active work items enqueued before launch.
    pub initial_items: u64,
    /// Additional device-produced work items the queue should absorb when the
    /// explicit queue budget leaves enough room.
    pub expansion_items: u64,
    /// ABI bytes per queue entry.
    pub entry_bytes: u64,
    /// Bytes required for queue head/tail counters and changed flags.
    pub control_bytes: u64,
    /// Caller-approved device-memory budget for the resident queue.
    pub budget_bytes: u64,
    /// Host synchronization policy.
    pub host_sync: WorkQueueHostSync,
}

/// Device-side work queue execution plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceWorkQueuePlan {
    /// Resident queue bytes.
    pub queue_bytes: u64,
    /// Resident control bytes.
    pub control_bytes: u64,
    /// Total resident bytes.
    pub resident_bytes: u64,
    /// Queue occupancy in basis points before device-side expansion.
    pub initial_occupancy_bps: u32,
    /// Whether the plan guarantees final-state-only host synchronization.
    pub final_only_host_sync: bool,
}

/// Device-side work queue drain strategy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceWorkQueueDrainStrategy {
    /// One resident drain window covers the whole queue.
    SingleResidentDrain,
    /// Queue capacity is split into multiple resident drain windows to bound
    /// per-launch queue pressure without host participation.
    ChunkedResidentDrain,
}

/// Device-side work queue plan with bounded resident drain windows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceWorkQueueBackpressurePlan {
    /// Base resident queue byte plan.
    pub queue: DeviceWorkQueuePlan,
    /// Selected resident drain strategy.
    pub strategy: DeviceWorkQueueDrainStrategy,
    /// Maximum queue entries drained by one device-side window.
    pub items_per_chunk: u64,
    /// Number of resident drain windows required to cover queue capacity.
    pub chunks: u64,
    /// Whether the backpressure plan preserves final-state-only host sync.
    pub final_only_host_sync: bool,
}

/// Device work queue planning errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeviceWorkQueueError {
    /// Queue capacity must be non-zero.
    ZeroCapacity,
    /// Entry ABI width must be explicit and non-zero.
    ZeroEntryBytes,
    /// Device-side drain chunk size must be non-zero.
    ZeroDrainChunk,
    /// Initial queue contents exceed capacity.
    InitialItemsExceedCapacity {
        /// Initial active items.
        initial_items: u64,
        /// Queue capacity.
        queue_capacity: u64,
    },
    /// Host participation would reintroduce CPU orchestration.
    HostParticipationRejected,
    /// Byte arithmetic overflowed.
    ByteCountOverflow {
        /// Field being computed.
        field: &'static str,
    },
    /// Queue does not fit the explicit device budget.
    OverBudget {
        /// Required bytes.
        required_bytes: u64,
        /// Budget bytes.
        budget_bytes: u64,
    },
}

impl std::fmt::Display for DeviceWorkQueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroCapacity => write!(
                f,
                "device work queue capacity is zero. Fix: size the resident queue before launch."
            ),
            Self::ZeroEntryBytes => write!(
                f,
                "device work queue entry_bytes is zero. Fix: pass the concrete queue-entry ABI width."
            ),
            Self::ZeroDrainChunk => write!(
                f,
                "device work queue drain chunk is zero. Fix: pass a non-zero device-side drain window."
            ),
            Self::InitialItemsExceedCapacity {
                initial_items,
                queue_capacity,
            } => write!(
                f,
                "device work queue initial_items={initial_items} exceeds queue_capacity={queue_capacity}. Fix: shard initial frontier items or increase explicit queue capacity."
            ),
            Self::HostParticipationRejected => write!(
                f,
                "device work queue rejected host participation. Fix: use final-only completion readback so dependent dataflow stays device-side."
            ),
            Self::ByteCountOverflow { field } => write!(
                f,
                "device work queue overflowed while computing {field}. Fix: shard the dependent dataflow workload before queue planning."
            ),
            Self::OverBudget {
                required_bytes,
                budget_bytes,
            } => write!(
                f,
                "device work queue requires {required_bytes} bytes but budget allows {budget_bytes}. Fix: reduce queue capacity, shard the graph, or raise the explicit device budget."
            ),
        }
    }
}

impl std::error::Error for DeviceWorkQueueError {}

fn checked_add(lhs: u64, rhs: u64, field: &'static str) -> Result<u64, DeviceWorkQueueError> {
    lhs.checked_add(rhs)
        .ok_or(DeviceWorkQueueError::ByteCountOverflow { field })
}

fn checked_mul(lhs: u64, rhs: u64, field: &'static str) -> Result<u64, DeviceWorkQueueError> {
    lhs.checked_mul(rhs)
        .ok_or(DeviceWorkQueueError::ByteCountOverflow { field })
}

/// Plan a device-resident work queue for dependent dataflow execution.
pub fn plan_device_work_queue(
    profile: DeviceWorkQueueProfile,
) -> Result<DeviceWorkQueuePlan, DeviceWorkQueueError> {
    if profile.queue_capacity == 0 {
        return Err(DeviceWorkQueueError::ZeroCapacity);
    }
    if profile.entry_bytes == 0 {
        return Err(DeviceWorkQueueError::ZeroEntryBytes);
    }
    if profile.initial_items > profile.queue_capacity {
        return Err(DeviceWorkQueueError::InitialItemsExceedCapacity {
            initial_items: profile.initial_items,
            queue_capacity: profile.queue_capacity,
        });
    }
    if profile.host_sync != WorkQueueHostSync::FinalOnly {
        return Err(DeviceWorkQueueError::HostParticipationRejected);
    }

    let queue_bytes = checked_mul(profile.queue_capacity, profile.entry_bytes, "queue bytes")?;
    let resident_bytes = checked_add(queue_bytes, profile.control_bytes, "resident bytes")?;
    if resident_bytes > profile.budget_bytes {
        return Err(DeviceWorkQueueError::OverBudget {
            required_bytes: resident_bytes,
            budget_bytes: profile.budget_bytes,
        });
    }
    let initial_occupancy_bps = DEVICE_WORK_QUEUE_NUMERIC.ratio_basis_points_u64(
        profile.initial_items,
        profile.queue_capacity,
        0,
        "device work queue initial occupancy",
    );

    Ok(DeviceWorkQueuePlan {
        queue_bytes,
        control_bytes: profile.control_bytes,
        resident_bytes,
        initial_occupancy_bps,
        final_only_host_sync: true,
    })
}

/// Plan a device-resident work queue that preserves initial-frontier capacity
/// and uses remaining queue budget for device-side expansion headroom.
pub fn plan_device_work_queue_with_expansion(
    profile: DeviceWorkQueueExpansionProfile,
) -> Result<DeviceWorkQueuePlan, DeviceWorkQueueError> {
    let desired_capacity = checked_add(
        profile.initial_items,
        profile.expansion_items,
        "queue expansion capacity",
    )?;
    if profile.entry_bytes == 0 {
        return plan_device_work_queue(DeviceWorkQueueProfile {
            initial_items: profile.initial_items,
            queue_capacity: desired_capacity,
            entry_bytes: profile.entry_bytes,
            control_bytes: profile.control_bytes,
            budget_bytes: profile.budget_bytes,
            host_sync: profile.host_sync,
        });
    }
    let budget_capacity =
        profile.budget_bytes.saturating_sub(profile.control_bytes) / profile.entry_bytes;
    let queue_capacity = desired_capacity
        .min(budget_capacity)
        .max(profile.initial_items);
    plan_device_work_queue(DeviceWorkQueueProfile {
        initial_items: profile.initial_items,
        queue_capacity,
        entry_bytes: profile.entry_bytes,
        control_bytes: profile.control_bytes,
        budget_bytes: profile.budget_bytes,
        host_sync: profile.host_sync,
    })
}

/// Plan a device-resident work queue plus bounded device-side drain windows.
pub fn plan_device_work_queue_backpressure(
    profile: DeviceWorkQueueProfile,
    max_items_per_drain_launch: u64,
) -> Result<DeviceWorkQueueBackpressurePlan, DeviceWorkQueueError> {
    if max_items_per_drain_launch == 0 {
        return Err(DeviceWorkQueueError::ZeroDrainChunk);
    }
    let queue = plan_device_work_queue(profile)?;
    let chunks = div_ceil_u64(
        profile.queue_capacity,
        max_items_per_drain_launch,
        "drain chunks",
    )?;
    let strategy = if chunks == 1 {
        DeviceWorkQueueDrainStrategy::SingleResidentDrain
    } else {
        DeviceWorkQueueDrainStrategy::ChunkedResidentDrain
    };
    Ok(DeviceWorkQueueBackpressurePlan {
        queue,
        strategy,
        items_per_chunk: max_items_per_drain_launch.min(profile.queue_capacity),
        chunks,
        final_only_host_sync: true,
    })
}

fn div_ceil_u64(lhs: u64, rhs: u64, field: &'static str) -> Result<u64, DeviceWorkQueueError> {
    DEVICE_WORK_QUEUE_NUMERIC
        .checked_ceil_div_u64(lhs, rhs)
        .ok_or(DeviceWorkQueueError::ByteCountOverflow { field })
}
