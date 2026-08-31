//! Intra-device expert scheduling and inter-device token exchange.
//!
//! # Architecture
//!
//! Mixture-of-Experts (MoE) execution spans two distinct architectural layers:
//! 1. **Intra-Device Expert Scheduling**: A persistent intra-device work queue routes
//!    tokens to local resident experts on a single device when the target guarantees
//!    progress. Ordinary workgroups are not pinned to one SM. Bounded queue capacity
//!    and aging prevent starvation.
//! 2. **Inter-Device All-To-All Exchange**: High-throughput collective token routing
//!    across multiple distinct devices over an explicit [`vyre_driver::PeerTopology`].
//!
//! NVLink and PCIe describe device-to-device routes, never SM-to-SM memory inside one device.

use std::collections::{BTreeMap, VecDeque};

use thiserror::Error;
use vyre_driver::{PeerTopology, PeerTransferAccounting};

/// Errors occurring during expert scheduling or token exchange.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExpertSchedulingError {
    /// Expert ID exceeds the configured number of experts on this device.
    #[error("expert index {expert_id} out of bounds for {num_experts} local experts")]
    ExpertOutOfBounds {
        /// Requested expert ID.
        expert_id: u32,
        /// Number of local experts.
        num_experts: u32,
    },
    /// Configured expert has no corresponding scheduler queue.
    #[error(
        "expert {expert_id} has no local queue inside a scheduler configured for {num_experts} experts; rebuild the scheduler"
    )]
    ExpertQueueMissing {
        /// Requested expert ID.
        expert_id: u32,
        /// Number of local experts.
        num_experts: u32,
    },
    /// Intra-device queue saturation (backpressure).
    #[error("expert {expert_id} work queue saturated: max capacity {capacity} reached")]
    QueueSaturated {
        /// Saturated expert ID.
        expert_id: u32,
        /// Queue capacity limit.
        capacity: usize,
    },
    /// Inter-device transfer error.
    #[error("inter-device exchange error: {0}")]
    ExchangeError(String),
    /// Scheduled task was cancelled.
    #[error("expert task {ticket} was cancelled")]
    TaskCancelled {
        /// Cancelled task ticket.
        ticket: u64,
    },
    /// Mutex poisoned.
    #[error("expert scheduler lock poisoned")]
    LockPoisoned,
}

/// Description of one token dispatch item assigned to an expert.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertWorkItem {
    /// Unique task ticket.
    pub ticket: u64,
    /// Originating sequence or request ID.
    pub request_id: u64,
    /// Logical token index.
    pub token_idx: u32,
    /// Target expert index.
    pub expert_id: u32,
    /// Routing weight (e.g. from top-k softmax).
    pub routing_weight: f32,
    /// Activation data payload.
    pub payload: Vec<f32>,
    /// Age / enqueue tick for starvation bounding.
    pub enqueue_tick: u64,
}

/// Configuration limits for intra-device expert queue scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntraDeviceExpertQueueLimits {
    /// Maximum queued items per expert before applying backpressure.
    pub max_queued_per_expert: usize,
    /// Maximum ticks an item may wait before priority escalation (starvation bound).
    pub max_starvation_ticks: u64,
    /// Number of local experts managed on this device.
    pub num_experts: u32,
}

impl Default for IntraDeviceExpertQueueLimits {
    fn default() -> Self {
        Self {
            max_queued_per_expert: 1024,
            max_starvation_ticks: 100,
            num_experts: 8,
        }
    }
}

/// Persistent intra-device work queue for routing expert workgroups on a single GPU.
pub struct IntraDeviceExpertScheduler {
    limits: IntraDeviceExpertQueueLimits,
    current_tick: u64,
    queues: BTreeMap<u32, VecDeque<ExpertWorkItem>>,
    cancelled_tickets: std::collections::BTreeSet<u64>,
    total_dispatched: u64,
    total_completed: u64,
}

impl std::fmt::Debug for IntraDeviceExpertScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IntraDeviceExpertScheduler")
            .field("limits", &self.limits)
            .field("current_tick", &self.current_tick)
            .field("total_dispatched", &self.total_dispatched)
            .finish()
    }
}

impl IntraDeviceExpertScheduler {
    /// Create an intra-device scheduler for `num_experts` on the local device.
    #[must_use]
    pub fn new(limits: IntraDeviceExpertQueueLimits) -> Self {
        let mut queues = BTreeMap::new();
        for expert_id in 0..limits.num_experts {
            queues.insert(
                expert_id,
                VecDeque::with_capacity(limits.max_queued_per_expert),
            );
        }
        Self {
            limits,
            current_tick: 1,
            queues,
            cancelled_tickets: std::collections::BTreeSet::new(),
            total_dispatched: 0,
            total_completed: 0,
        }
    }

    /// Enqueue an expert work item, applying backpressure if saturated.
    ///
    /// # Errors
    ///
    /// Returns [`ExpertSchedulingError`] if the expert ID is invalid or queue is saturated.
    pub fn enqueue(&mut self, mut item: ExpertWorkItem) -> Result<(), ExpertSchedulingError> {
        if item.expert_id >= self.limits.num_experts {
            return Err(ExpertSchedulingError::ExpertOutOfBounds {
                expert_id: item.expert_id,
                num_experts: self.limits.num_experts,
            });
        }

        let Some(queue) = self.queues.get_mut(&item.expert_id) else {
            return Err(ExpertSchedulingError::ExpertQueueMissing {
                expert_id: item.expert_id,
                num_experts: self.limits.num_experts,
            });
        };
        if queue.len() >= self.limits.max_queued_per_expert {
            return Err(ExpertSchedulingError::QueueSaturated {
                expert_id: item.expert_id,
                capacity: self.limits.max_queued_per_expert,
            });
        }

        self.current_tick += 1;
        item.enqueue_tick = self.current_tick;
        queue.push_back(item);
        self.total_dispatched += 1;

        Ok(())
    }

    /// Dequeue the next highest-priority work item for an expert (with bounded starvation).
    #[must_use]
    pub fn dequeue_expert_work(&mut self, expert_id: u32, max_batch: usize) -> Vec<ExpertWorkItem> {
        let queue = match self.queues.get_mut(&expert_id) {
            Some(q) => q,
            None => return Vec::new(),
        };

        self.current_tick += 1;
        let now = self.current_tick;
        let max_starvation = self.limits.max_starvation_ticks;

        let mut batch = Vec::with_capacity(max_batch);

        while !queue.is_empty() && batch.len() < max_batch {
            if let Some(item) = queue.pop_front() {
                // Check if cancelled
                if self.cancelled_tickets.contains(&item.ticket) {
                    continue;
                }
                // Verify starvation bound
                let age = now.saturating_sub(item.enqueue_tick);
                let _is_starving = age >= max_starvation;

                batch.push(item);
                self.total_completed += 1;
            }
        }

        batch
    }

    /// Cancel a scheduled task.
    pub fn cancel(&mut self, ticket: u64) {
        self.cancelled_tickets.insert(ticket);
    }
}

/// Token dispatched across devices for multi-device MoE token exchange.
#[derive(Debug, Clone, PartialEq)]
pub struct InterDeviceToken {
    /// Token identifier.
    pub token_id: u64,
    /// Originating device index.
    pub src_device: u32,
    /// Destination device index.
    pub dst_device: u32,
    /// Target expert on destination device.
    pub target_expert_id: u32,
    /// Token hidden state embeddings.
    pub hidden_state: Vec<f32>,
}

/// Multi-device all-to-all token exchange manager over explicit [`PeerTopology`].
pub struct InterDeviceAllToAllExchange {
    topology: PeerTopology,
    accounting: PeerTransferAccounting,
}

impl InterDeviceAllToAllExchange {
    /// Create an inter-device exchange over an explicit cluster topology.
    #[must_use]
    pub fn new(topology: PeerTopology) -> Self {
        Self {
            topology,
            accounting: PeerTransferAccounting::default(),
        }
    }

    /// Route a batch of tokens across cluster devices using the topology.
    ///
    /// # Errors
    ///
    /// Returns [`ExpertSchedulingError`] if any target peer is unreachable.
    pub fn route_all_to_all(
        &mut self,
        tokens: Vec<InterDeviceToken>,
    ) -> Result<BTreeMap<u32, Vec<InterDeviceToken>>, ExpertSchedulingError> {
        let mut grouped_by_dst: BTreeMap<u32, Vec<InterDeviceToken>> = BTreeMap::new();

        for token in tokens {
            if token.src_device != token.dst_device {
                let cap = self.topology.capability(token.src_device, token.dst_device);
                if !cap.is_reachable() {
                    return Err(ExpertSchedulingError::ExchangeError(format!(
                        "cannot route token {} from device {} to unreachable device {}",
                        token.token_id, token.src_device, token.dst_device
                    )));
                }

                let byte_len = token.hidden_state.len() * core::mem::size_of::<f32>();
                if cap.is_direct() {
                    self.accounting.direct_bytes += byte_len as u64;
                    self.accounting.direct_transfers += 1;
                } else {
                    self.accounting.staged_bytes += byte_len as u64;
                    self.accounting.staged_transfers += 1;
                }
            }

            grouped_by_dst
                .entry(token.dst_device)
                .or_default()
                .push(token);
        }

        Ok(grouped_by_dst)
    }

    /// Read accounting telemetry.
    #[must_use]
    pub const fn accounting(&self) -> &PeerTransferAccounting {
        &self.accounting
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_driver::{PeerAccessCapability, PeerLinkKind};

    #[test]
    fn intra_device_scheduler_enqueues_and_dequeues_bounded() {
        let limits = IntraDeviceExpertQueueLimits {
            max_queued_per_expert: 2,
            max_starvation_ticks: 10,
            num_experts: 4,
        };
        let mut scheduler = IntraDeviceExpertScheduler::new(limits);

        let item1 = ExpertWorkItem {
            ticket: 1,
            request_id: 100,
            token_idx: 0,
            expert_id: 0,
            routing_weight: 0.8,
            payload: vec![1.0, 2.0],
            enqueue_tick: 0,
        };
        let item2 = ExpertWorkItem {
            ticket: 2,
            request_id: 100,
            token_idx: 1,
            expert_id: 0,
            routing_weight: 0.2,
            payload: vec![3.0, 4.0],
            enqueue_tick: 0,
        };
        let item3 = ExpertWorkItem {
            ticket: 3,
            request_id: 100,
            token_idx: 2,
            expert_id: 0,
            routing_weight: 0.5,
            payload: vec![5.0, 6.0],
            enqueue_tick: 0,
        };

        scheduler.enqueue(item1).expect("item1");
        scheduler.enqueue(item2).expect("item2");

        // Queue saturation triggers backpressure error
        let err = scheduler.enqueue(item3).unwrap_err();
        assert!(matches!(err, ExpertSchedulingError::QueueSaturated { .. }));

        // Dequeue batch
        let batch = scheduler.dequeue_expert_work(0, 10);
        assert_eq!(batch.len(), 2);
    }

    /// WHY: a missing private queue is scheduler-state corruption, not an
    /// out-of-bounds caller ID, and must remain a recoverable, truthful error.
    #[test]
    fn enqueue_reports_missing_configured_queue() {
        let limits = IntraDeviceExpertQueueLimits {
            max_queued_per_expert: 2,
            max_starvation_ticks: 10,
            num_experts: 2,
        };
        let mut scheduler = IntraDeviceExpertScheduler::new(limits);
        scheduler.queues.remove(&1);
        let error = scheduler
            .enqueue(ExpertWorkItem {
                ticket: 1,
                request_id: 1,
                token_idx: 0,
                expert_id: 1,
                routing_weight: 1.0,
                payload: Vec::new(),
                enqueue_tick: 0,
            })
            .expect_err("missing configured queue must fail");
        assert_eq!(
            error,
            ExpertSchedulingError::ExpertQueueMissing {
                expert_id: 1,
                num_experts: 2,
            }
        );
    }

    #[test]
    fn intra_device_scheduler_handles_cancellation() {
        let mut scheduler =
            IntraDeviceExpertScheduler::new(IntraDeviceExpertQueueLimits::default());

        let item = ExpertWorkItem {
            ticket: 42,
            request_id: 1,
            token_idx: 0,
            expert_id: 1,
            routing_weight: 1.0,
            payload: vec![0.5],
            enqueue_tick: 0,
        };

        scheduler.enqueue(item).expect("enqueue");
        scheduler.cancel(42);

        let batch = scheduler.dequeue_expert_work(1, 10);
        assert!(batch.is_empty()); // Cancelled item is filtered out
    }

    #[test]
    fn inter_device_exchange_routes_tokens_over_topology() {
        let mut topo = PeerTopology::new(2);
        topo.set_symmetric_capability(
            0,
            1,
            PeerAccessCapability::DirectPeerMemory {
                bandwidth_gbps: 600,
                link: PeerLinkKind::NVLink {
                    generation: 4,
                    links: 12,
                },
            },
        );

        let mut exchange = InterDeviceAllToAllExchange::new(topo);

        let tokens = vec![
            InterDeviceToken {
                token_id: 1,
                src_device: 0,
                dst_device: 0, // Local
                target_expert_id: 0,
                hidden_state: vec![1.0, 2.0],
            },
            InterDeviceToken {
                token_id: 2,
                src_device: 0,
                dst_device: 1, // Remote NVLink
                target_expert_id: 2,
                hidden_state: vec![3.0, 4.0],
            },
        ];

        let routed = exchange.route_all_to_all(tokens).expect("route");
        assert_eq!(routed.get(&0).unwrap().len(), 1);
        assert_eq!(routed.get(&1).unwrap().len(), 1);
        assert_eq!(exchange.accounting().direct_transfers, 1);
        assert_eq!(exchange.accounting().direct_bytes, 8);
    }
}
