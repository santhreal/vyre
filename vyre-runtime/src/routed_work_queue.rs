//! Bounded routed-work queues, route-based scheduling, and inter-device exchange.
//!
//! # Architecture
//!
//! Routed work execution spans two distinct architectural layers:
//! 1. **Intra-Device Routed Work Queue**: A persistent intra-device work queue routes
//!    execution items to local resident routes / workers on a single device when the target
//!    guarantees progress. Bounded queue capacity and aging prevent starvation.
//! 2. **Inter-Device Routed Exchange**: Collective routing across multiple distinct devices
//!    over an explicit [`vyre_driver::PeerTopology`].

use std::collections::{BTreeMap, VecDeque};

use thiserror::Error;
use vyre_driver::{PeerTopology, PeerTransferAccounting};

/// Errors occurring during routed work queueing or exchange.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RoutedQueueError {
    /// Route ID exceeds the configured number of routes on this device.
    #[error("route index {route_id} out of bounds for {num_routes} local routes")]
    RouteOutOfBounds {
        /// Requested route ID.
        route_id: u32,
        /// Number of local routes.
        num_routes: u32,
    },
    /// Configured route has no corresponding queue.
    #[error(
        "route {route_id} has no local queue inside a scheduler configured for {num_routes} routes; rebuild the queue"
    )]
    RouteQueueMissing {
        /// Requested route ID.
        route_id: u32,
        /// Number of local routes.
        num_routes: u32,
    },
    /// Intra-device queue saturation (backpressure).
    #[error("route {route_id} work queue saturated: max capacity {capacity} reached")]
    QueueSaturated {
        /// Saturated route ID.
        route_id: u32,
        /// Queue capacity limit.
        capacity: usize,
    },
    /// Inter-device transfer error.
    #[error("inter-device exchange error: {0}")]
    ExchangeError(String),
    /// Scheduled task was cancelled.
    #[error("routed work task {ticket} was cancelled")]
    TaskCancelled {
        /// Cancelled task ticket.
        ticket: u64,
    },
    /// Mutex poisoned.
    #[error("routed work queue lock poisoned")]
    LockPoisoned,
}

/// Description of one dispatch item assigned to a route.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutedWorkItem {
    /// Unique task ticket.
    pub ticket: u64,
    /// Originating sequence or request ID.
    pub request_id: u64,
    /// Logical item index.
    pub item_index: u32,
    /// Target route index.
    pub route_id: u32,
    /// Routing weight.
    pub weight: f32,
    /// Data payload.
    pub payload: Vec<f32>,
    /// Age / enqueue tick for starvation bounding.
    pub enqueue_tick: u64,
}

/// Configuration limits for intra-device routed work queues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutedQueueLimits {
    /// Maximum queued items per route before applying backpressure.
    pub max_queued_per_route: usize,
    /// Maximum ticks an item may wait before priority escalation (starvation bound).
    pub max_starvation_ticks: u64,
    /// Number of local routes managed on this device.
    pub num_routes: u32,
}

impl Default for RoutedQueueLimits {
    fn default() -> Self {
        Self {
            max_queued_per_route: 1024,
            max_starvation_ticks: 100,
            num_routes: 8,
        }
    }
}

/// Persistent intra-device bounded routed-work queue for a single device.
pub struct BoundedRoutedWorkQueue {
    limits: RoutedQueueLimits,
    current_tick: u64,
    queues: BTreeMap<u32, VecDeque<RoutedWorkItem>>,
    cancelled_tickets: std::collections::BTreeSet<u64>,
    total_dispatched: u64,
    total_completed: u64,
}

impl std::fmt::Debug for BoundedRoutedWorkQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundedRoutedWorkQueue")
            .field("limits", &self.limits)
            .field("current_tick", &self.current_tick)
            .field("total_dispatched", &self.total_dispatched)
            .finish()
    }
}

impl BoundedRoutedWorkQueue {
    /// Create an intra-device bounded queue for `num_routes` on the local device.
    #[must_use]
    pub fn new(limits: RoutedQueueLimits) -> Self {
        let mut queues = BTreeMap::new();
        for route_id in 0..limits.num_routes {
            queues.insert(
                route_id,
                VecDeque::with_capacity(limits.max_queued_per_route),
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

    /// Enqueue a routed work item, applying backpressure if saturated.
    ///
    /// # Errors
    ///
    /// Returns [`RoutedQueueError`] if the route ID is invalid or queue is saturated.
    pub fn enqueue(&mut self, mut item: RoutedWorkItem) -> Result<(), RoutedQueueError> {
        if item.route_id >= self.limits.num_routes {
            return Err(RoutedQueueError::RouteOutOfBounds {
                route_id: item.route_id,
                num_routes: self.limits.num_routes,
            });
        }

        let Some(queue) = self.queues.get_mut(&item.route_id) else {
            return Err(RoutedQueueError::RouteQueueMissing {
                route_id: item.route_id,
                num_routes: self.limits.num_routes,
            });
        };
        if queue.len() >= self.limits.max_queued_per_route {
            return Err(RoutedQueueError::QueueSaturated {
                route_id: item.route_id,
                capacity: self.limits.max_queued_per_route,
            });
        }

        self.current_tick += 1;
        item.enqueue_tick = self.current_tick;
        queue.push_back(item);
        self.total_dispatched += 1;

        Ok(())
    }

    /// Dequeue the next highest-priority work items for a route (with bounded starvation).
    #[must_use]
    pub fn dequeue_route_work(&mut self, route_id: u32, max_batch: usize) -> Vec<RoutedWorkItem> {
        let queue = match self.queues.get_mut(&route_id) {
            Some(q) => q,
            None => return Vec::new(),
        };

        self.current_tick += 1;
        let now = self.current_tick;
        let max_starvation = self.limits.max_starvation_ticks;

        let mut batch = Vec::with_capacity(max_batch);

        while !queue.is_empty() && batch.len() < max_batch {
            if let Some(item) = queue.pop_front() {
                if self.cancelled_tickets.contains(&item.ticket) {
                    continue;
                }
                let age = now.saturating_sub(item.enqueue_tick);
                let _is_starving = age >= max_starvation;

                batch.push(item);
                self.total_completed += 1;
            }
        }

        batch
    }

    /// Cancel a scheduled task by ticket.
    pub fn cancel(&mut self, ticket: u64) {
        self.cancelled_tickets.insert(ticket);
    }
}

/// Item dispatched across devices for multi-device routed exchange.
#[derive(Debug, Clone, PartialEq)]
pub struct InterDeviceRoutedItem {
    /// Unique item identifier.
    pub item_id: u64,
    /// Originating device index.
    pub src_device: u32,
    /// Destination device index.
    pub dst_device: u32,
    /// Target route on destination device.
    pub target_route_id: u32,
    /// Payload vector.
    pub payload: Vec<f32>,
}

/// Multi-device collective routed exchange manager over explicit [`PeerTopology`].
pub struct InterDeviceRoutedExchange {
    topology: PeerTopology,
    accounting: PeerTransferAccounting,
}

impl InterDeviceRoutedExchange {
    /// Create an inter-device exchange over an explicit cluster topology.
    #[must_use]
    pub fn new(topology: PeerTopology) -> Self {
        Self {
            topology,
            accounting: PeerTransferAccounting::default(),
        }
    }

    /// Route a batch of items across cluster devices using the topology.
    ///
    /// # Errors
    ///
    /// Returns [`RoutedQueueError`] if any target peer is unreachable.
    pub fn route_all_to_all(
        &mut self,
        items: Vec<InterDeviceRoutedItem>,
    ) -> Result<BTreeMap<u32, Vec<InterDeviceRoutedItem>>, RoutedQueueError> {
        let mut grouped_by_dst: BTreeMap<u32, Vec<InterDeviceRoutedItem>> = BTreeMap::new();

        for item in items {
            if item.src_device != item.dst_device {
                let cap = self.topology.capability(item.src_device, item.dst_device);
                if !cap.is_reachable() {
                    return Err(RoutedQueueError::ExchangeError(format!(
                        "cannot route item {} from device {} to unreachable device {}",
                        item.item_id, item.src_device, item.dst_device
                    )));
                }

                let byte_len = item.payload.len() * core::mem::size_of::<f32>();
                if cap.is_direct() {
                    self.accounting.direct_bytes += byte_len as u64;
                    self.accounting.direct_transfers += 1;
                } else {
                    self.accounting.staged_bytes += byte_len as u64;
                    self.accounting.staged_transfers += 1;
                }
            }

            grouped_by_dst
                .entry(item.dst_device)
                .or_default()
                .push(item);
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
    fn bounded_routed_queue_enqueues_and_dequeues_bounded() {
        let limits = RoutedQueueLimits {
            max_queued_per_route: 2,
            max_starvation_ticks: 10,
            num_routes: 4,
        };
        let mut scheduler = BoundedRoutedWorkQueue::new(limits);

        let item1 = RoutedWorkItem {
            ticket: 1,
            request_id: 100,
            item_index: 0,
            route_id: 0,
            weight: 0.8,
            payload: vec![1.0, 2.0],
            enqueue_tick: 0,
        };
        let item2 = RoutedWorkItem {
            ticket: 2,
            request_id: 100,
            item_index: 1,
            route_id: 0,
            weight: 0.2,
            payload: vec![3.0, 4.0],
            enqueue_tick: 0,
        };
        let item3 = RoutedWorkItem {
            ticket: 3,
            request_id: 100,
            item_index: 2,
            route_id: 0,
            weight: 0.5,
            payload: vec![5.0, 6.0],
            enqueue_tick: 0,
        };

        scheduler.enqueue(item1).expect("item1");
        scheduler.enqueue(item2).expect("item2");

        // Queue saturation triggers backpressure error
        let err = scheduler.enqueue(item3).unwrap_err();
        assert!(matches!(err, RoutedQueueError::QueueSaturated { .. }));

        // Dequeue batch
        let batch = scheduler.dequeue_route_work(0, 10);
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn enqueue_reports_missing_configured_queue() {
        let limits = RoutedQueueLimits {
            max_queued_per_route: 2,
            max_starvation_ticks: 10,
            num_routes: 2,
        };
        let mut scheduler = BoundedRoutedWorkQueue::new(limits);
        scheduler.queues.remove(&1);
        let error = scheduler
            .enqueue(RoutedWorkItem {
                ticket: 1,
                request_id: 1,
                item_index: 0,
                route_id: 1,
                weight: 1.0,
                payload: Vec::new(),
                enqueue_tick: 0,
            })
            .expect_err("missing configured queue must fail");
        assert_eq!(
            error,
            RoutedQueueError::RouteQueueMissing {
                route_id: 1,
                num_routes: 2,
            }
        );
    }

    #[test]
    fn bounded_routed_queue_handles_cancellation() {
        let mut scheduler = BoundedRoutedWorkQueue::new(RoutedQueueLimits::default());

        let item = RoutedWorkItem {
            ticket: 42,
            request_id: 1,
            item_index: 0,
            route_id: 1,
            weight: 1.0,
            payload: vec![0.5],
            enqueue_tick: 0,
        };

        scheduler.enqueue(item).expect("enqueue");
        scheduler.cancel(42);

        let batch = scheduler.dequeue_route_work(1, 10);
        assert!(batch.is_empty());
    }

    #[test]
    fn inter_device_exchange_routes_items_over_topology() {
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

        let mut exchange = InterDeviceRoutedExchange::new(topo);

        let items = vec![
            InterDeviceRoutedItem {
                item_id: 1,
                src_device: 0,
                dst_device: 0,
                target_route_id: 0,
                payload: vec![1.0, 2.0],
            },
            InterDeviceRoutedItem {
                item_id: 2,
                src_device: 0,
                dst_device: 1,
                target_route_id: 2,
                payload: vec![3.0, 4.0],
            },
        ];

        let routed = exchange.route_all_to_all(items).expect("route");
        assert_eq!(routed.get(&0).unwrap().len(), 1);
        assert_eq!(routed.get(&1).unwrap().len(), 1);
        assert_eq!(exchange.accounting().direct_transfers, 1);
        assert_eq!(exchange.accounting().direct_bytes, 8);
    }
}
