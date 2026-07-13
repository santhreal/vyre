//! CUDA backend runtime telemetry.

use std::sync::atomic::{AtomicU64, Ordering};

use vyre_driver::accounting::{atomic_max_u64, pinning_atomic_increment_u64};
use vyre_driver::LaunchPlan;

use crate::backend::accounting::checked_add_u64;

/// Point-in-time CUDA backend telemetry.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaTelemetrySnapshot {
    /// Bytes copied from host memory into CUDA device-visible memory.
    pub host_to_device_bytes: u64,
    /// Bytes copied from CUDA device memory into host memory.
    pub device_to_host_bytes: u64,
    /// Device-to-host bytes that are final user-visible output readbacks.
    pub readback_bytes: u64,
    /// Bytes requested from the transient device allocation pool.
    pub transient_allocation_bytes_requested: u64,
    /// Bytes requested as long-lived resident CUDA allocations.
    pub resident_allocation_bytes_requested: u64,
    /// Bytes uploaded for CUDA kernel parameter blocks.
    pub param_upload_bytes: u64,
    /// CUDA kernel launches issued through `cuLaunchKernel` or cooperative launch.
    pub kernel_launches: u64,
    /// CUDA graph replays issued through `cuGraphLaunch`.
    pub cuda_graph_launches: u64,
    /// CUDA graph requests satisfied from a materialized host-output cache.
    pub cuda_graph_materialized_cache_hits: u64,
    /// Batched cudaGraph replay chunks issued by compiled pipelines.
    pub cuda_graph_batched_replay_chunks: u64,
    /// Individual cudaGraph lanes launched inside batched replay chunks.
    pub cuda_graph_batched_replay_lanes: u64,
    /// Host synchronization points against CUDA events or streams.
    pub sync_points: u64,
    /// Non-empty host-to-device copy operations.
    pub host_upload_operations: u64,
    /// Non-empty device-to-host readback operations.
    pub device_readback_operations: u64,
    /// Successful timed dispatches reported by CUDA timed entrypoints.
    pub timed_dispatches: u64,
    /// Timed dispatches that included CUDA event-backed device time.
    pub timed_device_measurements: u64,
    /// Timed dispatches that completed without CUDA event-backed device time.
    pub timed_dispatches_missing_device_time: u64,
    /// Aggregate host-observed timed dispatch duration.
    pub timed_wall_ns_total: u64,
    /// Aggregate CUDA event-observed device duration.
    pub timed_device_ns_total: u64,
    /// Maximum CUDA event-observed device duration.
    pub timed_device_ns_max: u64,
    /// Aggregate host enqueue duration for timed dispatches.
    pub timed_enqueue_ns_total: u64,
    /// Aggregate host wait/readback duration for timed dispatches.
    pub timed_wait_ns_total: u64,
    /// Aggregate scheduled CUDA thread slots across kernel launches.
    pub scheduled_thread_slots: u64,
    /// Kernel launches whose exact scheduled thread-slot product exceeded u64 telemetry width.
    pub scheduled_thread_slot_overflows: u64,
    /// Runtime telemetry counter additions that exceeded u64 counter width.
    pub telemetry_counter_overflows: u64,
    /// Resident dispatches that took the host-buffer borrowed fallback path.
    pub resident_borrowed_fallback_dispatches: u64,
    /// Aggregate logical element count submitted across kernel launches.
    pub launched_elements: u64,
    /// Aggregate scheduled CUDA thread slots that carried no logical element.
    pub wasted_thread_slots: u64,
    /// Logical element utilization over scheduled CUDA thread slots in basis points.
    pub logical_thread_utilization_bps: u32,
    /// Empty scheduled CUDA thread slots in basis points.
    pub logical_thread_waste_bps: u32,
    /// Unclamped logical element density per scheduled CUDA thread slot in basis points.
    pub logical_elements_per_thread_slot_bps: u64,
    /// Transient device-allocation-pool acquisitions served from the free-list
    /// (no `cuMemAlloc`) this telemetry epoch (the numerator of the pool hit rate).
    pub device_pool_hits: u64,
    /// Transient device-allocation-pool acquisitions that fell through to a real
    /// `cuMemAlloc_v2` this telemetry epoch (an empty bucket).
    pub device_pool_misses: u64,
    /// Sum of per-launch achieved-occupancy basis points over launches whose
    /// occupancy was measured this epoch (the numerator of the mean occupancy).
    pub launch_occupancy_bps_sum: u64,
    /// Kernel launches whose achieved occupancy was measured (driver occupancy
    /// query succeeded) this epoch (the denominator of the mean occupancy).
    pub occupancy_measured_launches: u64,
    /// Kernel launches whose occupancy could NOT be measured this epoch (bad
    /// geometry or a driver occupancy-query failure). Surfaced loudly so a zero
    /// mean is never confused with unmeasured launches (Law 10 (no silent gap)).
    pub occupancy_unmeasured_launches: u64,
}

impl CudaTelemetrySnapshot {
    /// Mean achieved kernel occupancy in basis points over the launches that were
    /// measured this epoch (`launch_occupancy_bps_sum / occupancy_measured_launches`),
    /// or `0` when no launch was measured. Read alongside
    /// `occupancy_unmeasured_launches`: a high unmeasured count means the mean
    /// covers only part of the workload. This is the W3-6 per-kernel occupancy
    /// evidence, aggregated for the snapshot.
    #[must_use]
    pub fn mean_occupancy_bps(&self) -> u32 {
        if self.occupancy_measured_launches == 0 {
            return 0;
        }
        (self.launch_occupancy_bps_sum / self.occupancy_measured_launches) as u32
    }
    /// Transient device-allocation-pool hit rate in basis points
    /// (`hits * 10000 / (hits + misses)`), or `0` when there were no acquisitions.
    /// This is the pool-hit-rate evidence W3-4 asks for, computed from the two raw
    /// counters so a consumer can also see the acquisition volume behind the ratio.
    #[must_use]
    pub fn device_pool_hit_rate_bps(&self) -> u32 {
        let total = self.device_pool_hits + self.device_pool_misses;
        if total == 0 {
            return 0;
        }
        // hits <= total <= u64::MAX, and hits*10000 fits in u128; the ratio is
        // <= 10000 so the u32 cast is exact.
        ((u128::from(self.device_pool_hits) * 10_000) / u128::from(total)) as u32
    }

    /// Render CUDA runtime counters as Prometheus exposition text.
    #[must_use]
    pub fn to_prometheus_text(self) -> String {
        format!(
            concat!(
                "vyre_cuda_host_to_device_bytes_total {}\n",
                "vyre_cuda_device_to_host_bytes_total {}\n",
                "vyre_cuda_readback_bytes_total {}\n",
                "vyre_cuda_transient_allocation_bytes_requested_total {}\n",
                "vyre_cuda_resident_allocation_bytes_requested_total {}\n",
                "vyre_cuda_param_upload_bytes_total {}\n",
                "vyre_cuda_kernel_launches_total {}\n",
                "vyre_cuda_graph_launches_total {}\n",
                "vyre_cuda_graph_materialized_cache_hits_total {}\n",
                "vyre_cuda_graph_batched_replay_chunks_total {}\n",
                "vyre_cuda_graph_batched_replay_lanes_total {}\n",
                "vyre_cuda_sync_points_total {}\n",
                "vyre_cuda_host_upload_operations_total {}\n",
                "vyre_cuda_device_readback_operations_total {}\n",
                "vyre_cuda_timed_dispatches_total {}\n",
                "vyre_cuda_timed_device_measurements_total {}\n",
                "vyre_cuda_timed_dispatches_missing_device_time_total {}\n",
                "vyre_cuda_timed_wall_ns_total {}\n",
                "vyre_cuda_timed_device_ns_total {}\n",
                "vyre_cuda_timed_device_ns_max {}\n",
                "vyre_cuda_timed_enqueue_ns_total {}\n",
                "vyre_cuda_timed_wait_ns_total {}\n",
                "vyre_cuda_scheduled_thread_slots_total {}\n",
                "vyre_cuda_scheduled_thread_slot_overflows_total {}\n",
                "vyre_cuda_telemetry_counter_overflows_total {}\n",
                "vyre_cuda_resident_borrowed_fallback_dispatches_total {}\n",
                "vyre_cuda_launched_elements_total {}\n",
                "vyre_cuda_wasted_thread_slots_total {}\n",
                "vyre_cuda_logical_thread_utilization_bps {}\n",
                "vyre_cuda_logical_thread_waste_bps {}\n",
                "vyre_cuda_logical_elements_per_thread_slot_bps {}\n",
                "vyre_cuda_device_pool_hits_total {}\n",
                "vyre_cuda_device_pool_misses_total {}\n",
                "vyre_cuda_device_pool_hit_rate_bps {}\n",
                "vyre_cuda_launch_occupancy_bps_sum {}\n",
                "vyre_cuda_occupancy_measured_launches_total {}\n",
                "vyre_cuda_occupancy_unmeasured_launches_total {}\n",
                "vyre_cuda_mean_occupancy_bps {}\n"
            ),
            self.host_to_device_bytes,
            self.device_to_host_bytes,
            self.readback_bytes,
            self.transient_allocation_bytes_requested,
            self.resident_allocation_bytes_requested,
            self.param_upload_bytes,
            self.kernel_launches,
            self.cuda_graph_launches,
            self.cuda_graph_materialized_cache_hits,
            self.cuda_graph_batched_replay_chunks,
            self.cuda_graph_batched_replay_lanes,
            self.sync_points,
            self.host_upload_operations,
            self.device_readback_operations,
            self.timed_dispatches,
            self.timed_device_measurements,
            self.timed_dispatches_missing_device_time,
            self.timed_wall_ns_total,
            self.timed_device_ns_total,
            self.timed_device_ns_max,
            self.timed_enqueue_ns_total,
            self.timed_wait_ns_total,
            self.scheduled_thread_slots,
            self.scheduled_thread_slot_overflows,
            self.telemetry_counter_overflows,
            self.resident_borrowed_fallback_dispatches,
            self.launched_elements,
            self.wasted_thread_slots,
            self.logical_thread_utilization_bps,
            self.logical_thread_waste_bps,
            self.logical_elements_per_thread_slot_bps,
            self.device_pool_hits,
            self.device_pool_misses,
            self.device_pool_hit_rate_bps(),
            self.launch_occupancy_bps_sum,
            self.occupancy_measured_launches,
            self.occupancy_unmeasured_launches,
            self.mean_occupancy_bps()
        )
    }
}

/// Atomic CUDA backend telemetry counters.
#[derive(Debug, Default)]
pub(crate) struct CudaTelemetry {
    host_to_device_bytes: AtomicU64,
    device_to_host_bytes: AtomicU64,
    readback_bytes: AtomicU64,
    transient_allocation_bytes_requested: AtomicU64,
    resident_allocation_bytes_requested: AtomicU64,
    param_upload_bytes: AtomicU64,
    kernel_launches: AtomicU64,
    cuda_graph_launches: AtomicU64,
    cuda_graph_materialized_cache_hits: AtomicU64,
    cuda_graph_batched_replay_chunks: AtomicU64,
    cuda_graph_batched_replay_lanes: AtomicU64,
    sync_points: AtomicU64,
    host_upload_operations: AtomicU64,
    device_readback_operations: AtomicU64,
    timed_dispatches: AtomicU64,
    timed_device_measurements: AtomicU64,
    timed_dispatches_missing_device_time: AtomicU64,
    timed_wall_ns_total: AtomicU64,
    timed_device_ns_total: AtomicU64,
    timed_device_ns_max: AtomicU64,
    timed_enqueue_ns_total: AtomicU64,
    timed_wait_ns_total: AtomicU64,
    scheduled_thread_slots: AtomicU64,
    scheduled_thread_slot_overflows: AtomicU64,
    telemetry_counter_overflows: AtomicU64,
    resident_borrowed_fallback_dispatches: AtomicU64,
    launched_elements: AtomicU64,
    launch_occupancy_bps_sum: AtomicU64,
    occupancy_measured_launches: AtomicU64,
    occupancy_unmeasured_launches: AtomicU64,
}

impl CudaTelemetry {
    #[must_use]
    pub(crate) fn snapshot(&self) -> CudaTelemetrySnapshot {
        let scheduled_thread_slots = self.scheduled_thread_slots.load(Ordering::Relaxed);
        let launched_elements = self.launched_elements.load(Ordering::Relaxed);
        let used_slots = launched_elements.min(scheduled_thread_slots);
        let wasted_thread_slots = scheduled_thread_slots - used_slots;
        CudaTelemetrySnapshot {
            host_to_device_bytes: self.host_to_device_bytes.load(Ordering::Relaxed),
            device_to_host_bytes: self.device_to_host_bytes.load(Ordering::Relaxed),
            readback_bytes: self.readback_bytes.load(Ordering::Relaxed),
            transient_allocation_bytes_requested: self
                .transient_allocation_bytes_requested
                .load(Ordering::Relaxed),
            resident_allocation_bytes_requested: self
                .resident_allocation_bytes_requested
                .load(Ordering::Relaxed),
            param_upload_bytes: self.param_upload_bytes.load(Ordering::Relaxed),
            kernel_launches: self.kernel_launches.load(Ordering::Relaxed),
            cuda_graph_launches: self.cuda_graph_launches.load(Ordering::Relaxed),
            cuda_graph_materialized_cache_hits: self
                .cuda_graph_materialized_cache_hits
                .load(Ordering::Relaxed),
            cuda_graph_batched_replay_chunks: self
                .cuda_graph_batched_replay_chunks
                .load(Ordering::Relaxed),
            cuda_graph_batched_replay_lanes: self
                .cuda_graph_batched_replay_lanes
                .load(Ordering::Relaxed),
            sync_points: self.sync_points.load(Ordering::Relaxed),
            host_upload_operations: self.host_upload_operations.load(Ordering::Relaxed),
            device_readback_operations: self.device_readback_operations.load(Ordering::Relaxed),
            timed_dispatches: self.timed_dispatches.load(Ordering::Relaxed),
            timed_device_measurements: self.timed_device_measurements.load(Ordering::Relaxed),
            timed_dispatches_missing_device_time: self
                .timed_dispatches_missing_device_time
                .load(Ordering::Relaxed),
            timed_wall_ns_total: self.timed_wall_ns_total.load(Ordering::Relaxed),
            timed_device_ns_total: self.timed_device_ns_total.load(Ordering::Relaxed),
            timed_device_ns_max: self.timed_device_ns_max.load(Ordering::Relaxed),
            timed_enqueue_ns_total: self.timed_enqueue_ns_total.load(Ordering::Relaxed),
            timed_wait_ns_total: self.timed_wait_ns_total.load(Ordering::Relaxed),
            scheduled_thread_slots,
            scheduled_thread_slot_overflows: self
                .scheduled_thread_slot_overflows
                .load(Ordering::Relaxed),
            telemetry_counter_overflows: self.telemetry_counter_overflows.load(Ordering::Relaxed),
            resident_borrowed_fallback_dispatches: self
                .resident_borrowed_fallback_dispatches
                .load(Ordering::Relaxed),
            launched_elements,
            wasted_thread_slots,
            logical_thread_utilization_bps: utilization_bps(
                launched_elements,
                scheduled_thread_slots,
            ),
            logical_thread_waste_bps: utilization_bps(wasted_thread_slots, scheduled_thread_slots),
            logical_elements_per_thread_slot_bps: elements_per_slot_bps(
                launched_elements,
                scheduled_thread_slots,
            ),
            // The device-pool hit/miss counters live on the pool (their only
            // source of truth); the backend's `telemetry_snapshot` overlays them.
            // A bare `CudaTelemetry::snapshot` (pool-agnostic) reports zero.
            device_pool_hits: 0,
            device_pool_misses: 0,
            launch_occupancy_bps_sum: self.launch_occupancy_bps_sum.load(Ordering::Relaxed),
            occupancy_measured_launches: self.occupancy_measured_launches.load(Ordering::Relaxed),
            occupancy_unmeasured_launches: self
                .occupancy_unmeasured_launches
                .load(Ordering::Relaxed),
        }
    }

    pub(crate) fn reset(&self) {
        self.host_to_device_bytes.store(0, Ordering::Relaxed);
        self.device_to_host_bytes.store(0, Ordering::Relaxed);
        self.readback_bytes.store(0, Ordering::Relaxed);
        self.transient_allocation_bytes_requested
            .store(0, Ordering::Relaxed);
        self.resident_allocation_bytes_requested
            .store(0, Ordering::Relaxed);
        self.param_upload_bytes.store(0, Ordering::Relaxed);
        self.kernel_launches.store(0, Ordering::Relaxed);
        self.cuda_graph_launches.store(0, Ordering::Relaxed);
        self.cuda_graph_materialized_cache_hits
            .store(0, Ordering::Relaxed);
        self.cuda_graph_batched_replay_chunks
            .store(0, Ordering::Relaxed);
        self.cuda_graph_batched_replay_lanes
            .store(0, Ordering::Relaxed);
        self.sync_points.store(0, Ordering::Relaxed);
        self.host_upload_operations.store(0, Ordering::Relaxed);
        self.device_readback_operations.store(0, Ordering::Relaxed);
        self.timed_dispatches.store(0, Ordering::Relaxed);
        self.timed_device_measurements.store(0, Ordering::Relaxed);
        self.timed_dispatches_missing_device_time
            .store(0, Ordering::Relaxed);
        self.timed_wall_ns_total.store(0, Ordering::Relaxed);
        self.timed_device_ns_total.store(0, Ordering::Relaxed);
        self.timed_device_ns_max.store(0, Ordering::Relaxed);
        self.timed_enqueue_ns_total.store(0, Ordering::Relaxed);
        self.timed_wait_ns_total.store(0, Ordering::Relaxed);
        self.scheduled_thread_slots.store(0, Ordering::Relaxed);
        self.scheduled_thread_slot_overflows
            .store(0, Ordering::Relaxed);
        self.telemetry_counter_overflows.store(0, Ordering::Relaxed);
        self.resident_borrowed_fallback_dispatches
            .store(0, Ordering::Relaxed);
        self.launched_elements.store(0, Ordering::Relaxed);
        self.launch_occupancy_bps_sum.store(0, Ordering::Relaxed);
        self.occupancy_measured_launches.store(0, Ordering::Relaxed);
        self.occupancy_unmeasured_launches
            .store(0, Ordering::Relaxed);
    }

    /// Record one launch whose achieved occupancy was measured (`occupancy_bps`
    /// in basis points, 0..=10000): adds to the running sum and increments the
    /// measured-launch count so the snapshot can report the mean.
    pub(crate) fn record_launch_occupancy_bps(&self, occupancy_bps: u64) {
        self.add(
            "launch_occupancy_bps_sum",
            &self.launch_occupancy_bps_sum,
            occupancy_bps,
        );
        self.add(
            "occupancy_measured_launches",
            &self.occupancy_measured_launches,
            1,
        );
    }

    /// Record one launch whose occupancy could NOT be measured (bad geometry or a
    /// driver occupancy-query failure), surfaced so a partial mean is never
    /// mistaken for full coverage (Law 10 (no silent measurement gap)).
    pub(crate) fn record_launch_occupancy_unmeasured(&self) {
        self.add(
            "occupancy_unmeasured_launches",
            &self.occupancy_unmeasured_launches,
            1,
        );
    }

    pub(crate) fn record_resident_borrowed_fallback_dispatch(&self) {
        self.add(
            "resident_borrowed_fallback_dispatches",
            &self.resident_borrowed_fallback_dispatches,
            1,
        );
    }

    pub(crate) fn record_host_to_device_bytes(&self, bytes: u64) {
        self.add("host_to_device_bytes", &self.host_to_device_bytes, bytes);
    }

    pub(crate) fn record_device_to_host_readback(&self, bytes: u64) {
        self.add("device_to_host_bytes", &self.device_to_host_bytes, bytes);
        self.add("readback_bytes", &self.readback_bytes, bytes);
    }

    pub(crate) fn record_transient_allocation_bytes(&self, bytes: u64) {
        self.add(
            "transient_allocation_bytes_requested",
            &self.transient_allocation_bytes_requested,
            bytes,
        );
    }

    pub(crate) fn record_resident_allocation_bytes(&self, bytes: u64) {
        self.add(
            "resident_allocation_bytes_requested",
            &self.resident_allocation_bytes_requested,
            bytes,
        );
    }

    pub(crate) fn record_param_upload_bytes(&self, bytes: u64) {
        self.add("param_upload_bytes", &self.param_upload_bytes, bytes);
    }

    pub(crate) fn record_kernel_launch(&self, launch: &LaunchPlan) {
        self.add("kernel_launches", &self.kernel_launches, 1);
        if let Some(slots) = scheduled_thread_slots(launch) {
            self.add(
                "scheduled_thread_slots",
                &self.scheduled_thread_slots,
                slots,
            );
        } else {
            self.add(
                "scheduled_thread_slot_overflows",
                &self.scheduled_thread_slot_overflows,
                1,
            );
        }
        self.add(
            "launched_elements",
            &self.launched_elements,
            u64::from(launch.element_count),
        );
    }

    pub(crate) fn record_cuda_graph_launch(&self) {
        self.add("cuda_graph_launches", &self.cuda_graph_launches, 1);
    }

    pub(crate) fn record_cuda_graph_materialized_cache_hit(&self) {
        self.add(
            "cuda_graph_materialized_cache_hits",
            &self.cuda_graph_materialized_cache_hits,
            1,
        );
    }

    pub(crate) fn record_cuda_graph_batched_replay(&self, lanes: u64) {
        self.add(
            "cuda_graph_batched_replay_chunks",
            &self.cuda_graph_batched_replay_chunks,
            1,
        );
        self.add(
            "cuda_graph_batched_replay_lanes",
            &self.cuda_graph_batched_replay_lanes,
            lanes,
        );
    }

    pub(crate) fn record_sync_point(&self) {
        self.add("sync_points", &self.sync_points, 1);
    }

    pub(crate) fn record_host_upload_operations(&self, operations: u64) {
        self.add(
            "host_upload_operations",
            &self.host_upload_operations,
            operations,
        );
    }

    pub(crate) fn record_device_readback_operations(&self, operations: u64) {
        self.add(
            "device_readback_operations",
            &self.device_readback_operations,
            operations,
        );
    }

    pub(crate) fn record_timed_dispatch(
        &self,
        wall_ns: u64,
        device_ns: Option<u64>,
        enqueue_ns: Option<u64>,
        wait_ns: Option<u64>,
    ) {
        self.add("timed_dispatches", &self.timed_dispatches, 1);
        self.add("timed_wall_ns_total", &self.timed_wall_ns_total, wall_ns);
        match device_ns {
            Some(device_ns) => {
                self.add(
                    "timed_device_measurements",
                    &self.timed_device_measurements,
                    1,
                );
                self.add(
                    "timed_device_ns_total",
                    &self.timed_device_ns_total,
                    device_ns,
                );
                self.record_max("timed_device_ns_max", &self.timed_device_ns_max, device_ns);
            }
            None => {
                self.add(
                    "timed_dispatches_missing_device_time",
                    &self.timed_dispatches_missing_device_time,
                    1,
                );
            }
        }
        if let Some(enqueue_ns) = enqueue_ns {
            self.add(
                "timed_enqueue_ns_total",
                &self.timed_enqueue_ns_total,
                enqueue_ns,
            );
        }
        if let Some(wait_ns) = wait_ns {
            self.add("timed_wait_ns_total", &self.timed_wait_ns_total, wait_ns);
        }
    }

    fn add(&self, name: &'static str, counter: &AtomicU64, value: u64) -> bool {
        if value == 0 {
            return true;
        }
        let result = checked_add_u64(counter, value, |current, attempted| {
            vyre_driver::BackendError::new(format!(
                "CUDA telemetry counter `{name}` overflowed u64: current={current}, add={attempted}. Fix: rotate telemetry snapshots or shard the dispatch accounting window before counters overflow."
            ))
        });
        if let Err(error) = result {
            tracing::error!("{error}");
            self.record_counter_overflow(name);
            return false;
        }
        true
    }

    fn record_max(&self, name: &'static str, counter: &AtomicU64, value: u64) {
        let _ = name;
        atomic_max_u64(counter, value, Ordering::Relaxed);
    }

    fn record_counter_overflow(&self, source_counter: &'static str) {
        pinning_atomic_increment_u64(
            &self.telemetry_counter_overflows,
            Ordering::Relaxed,
            Ordering::Relaxed,
            || {
                tracing::error!(
                "CUDA telemetry overflow counter overflowed while recording `{source_counter}`. Fix: rotate telemetry snapshots before overflow diagnostics exceed u64."
            );
            },
        );
    }
}

fn scheduled_thread_slots(launch: &LaunchPlan) -> Option<u64> {
    let exact = launch
        .grid
        .iter()
        .chain(launch.workgroup.iter())
        .try_fold(1_u128, |acc, dim| acc.checked_mul(u128::from(*dim)));
    let exact = exact?;
    u64::try_from(exact).ok()
}

fn utilization_bps(used: u64, scheduled: u64) -> u32 {
    crate::numeric::CUDA_NUMERIC
        .ratio_basis_points_u64(used, scheduled, 0, "telemetry utilization")
        .min(10_000)
}

fn elements_per_slot_bps(elements: u64, scheduled: u64) -> u64 {
    crate::numeric::CUDA_NUMERIC.ratio_basis_points_u64_wide(
        elements,
        scheduled,
        0,
        "telemetry logical-elements-per-thread-slot",
    )
}

#[cfg(test)]
mod tests {
    use super::{CudaTelemetry, CudaTelemetrySnapshot};

    #[test]
    fn device_pool_hit_rate_bps_is_exact_and_zero_safe() {
        // No acquisitions -> 0, never a divide-by-zero.
        let empty = CudaTelemetrySnapshot {
            device_pool_hits: 0,
            device_pool_misses: 0,
            ..CudaTelemetrySnapshot::default()
        };
        assert_eq!(empty.device_pool_hit_rate_bps(), 0);

        // 3 hits / 1 miss over 4 acquisitions -> 7500 bps.
        let mixed = CudaTelemetrySnapshot {
            device_pool_hits: 3,
            device_pool_misses: 1,
            ..CudaTelemetrySnapshot::default()
        };
        assert_eq!(mixed.device_pool_hit_rate_bps(), 7_500);

        // All hits -> full 10000 bps; the Prometheus text carries every field.
        let all_hits = CudaTelemetrySnapshot {
            device_pool_hits: 9,
            device_pool_misses: 0,
            ..CudaTelemetrySnapshot::default()
        };
        assert_eq!(all_hits.device_pool_hit_rate_bps(), 10_000);
        let text = all_hits.to_prometheus_text();
        assert!(text.contains("vyre_cuda_device_pool_hits_total 9"));
        assert!(text.contains("vyre_cuda_device_pool_misses_total 0"));
        assert!(text.contains("vyre_cuda_device_pool_hit_rate_bps 10000"));

        // Large counts stay exact through the u128 intermediate (no overflow, no
        // rounding past the true ratio).
        let large = CudaTelemetrySnapshot {
            device_pool_hits: u64::MAX / 2,
            device_pool_misses: u64::MAX / 2,
            ..CudaTelemetrySnapshot::default()
        };
        assert_eq!(large.device_pool_hit_rate_bps(), 5_000);
    }

    #[test]
    fn mean_occupancy_bps_is_exact_and_zero_safe_and_counts_unmeasured() {
        // No measured launch -> mean 0, never a divide-by-zero.
        let none = CudaTelemetrySnapshot {
            launch_occupancy_bps_sum: 0,
            occupancy_measured_launches: 0,
            occupancy_unmeasured_launches: 5,
            ..CudaTelemetrySnapshot::default()
        };
        assert_eq!(none.mean_occupancy_bps(), 0);
        // The unmeasured count is preserved so a zero mean is not mistaken for
        // "measured 0% occupancy".
        assert_eq!(none.occupancy_unmeasured_launches, 5);

        // 6000 + 8000 over 2 measured launches -> mean 7000.
        let measured = CudaTelemetrySnapshot {
            launch_occupancy_bps_sum: 14_000,
            occupancy_measured_launches: 2,
            occupancy_unmeasured_launches: 0,
            ..CudaTelemetrySnapshot::default()
        };
        assert_eq!(measured.mean_occupancy_bps(), 7_000);
        let text = measured.to_prometheus_text();
        assert!(text.contains("vyre_cuda_mean_occupancy_bps 7000"));
        assert!(text.contains("vyre_cuda_occupancy_measured_launches_total 2"));
        assert!(text.contains("vyre_cuda_occupancy_unmeasured_launches_total 0"));
    }

    #[test]
    fn record_launch_occupancy_accumulates_mean_and_resets() {
        let telemetry = CudaTelemetry::default();
        telemetry.record_launch_occupancy_bps(4_000);
        telemetry.record_launch_occupancy_bps(6_000);
        telemetry.record_launch_occupancy_unmeasured();
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.occupancy_measured_launches, 2);
        assert_eq!(snapshot.occupancy_unmeasured_launches, 1);
        assert_eq!(snapshot.launch_occupancy_bps_sum, 10_000);
        assert_eq!(snapshot.mean_occupancy_bps(), 5_000);

        telemetry.reset();
        let cleared = telemetry.snapshot();
        assert_eq!(cleared.occupancy_measured_launches, 0);
        assert_eq!(cleared.occupancy_unmeasured_launches, 0);
        assert_eq!(cleared.launch_occupancy_bps_sum, 0);
        assert_eq!(cleared.mean_occupancy_bps(), 0);
    }

    #[test]
    fn snapshot_accumulates_and_resets_counters() {
        let telemetry = CudaTelemetry::default();
        telemetry.record_host_to_device_bytes(16);
        telemetry.record_device_to_host_readback(8);
        telemetry.record_transient_allocation_bytes(32);
        telemetry.record_resident_allocation_bytes(64);
        telemetry.record_param_upload_bytes(4);
        telemetry.record_cuda_graph_launch();
        telemetry.record_cuda_graph_materialized_cache_hit();
        telemetry.record_cuda_graph_batched_replay(4);
        telemetry.record_sync_point();
        telemetry.record_host_upload_operations(2);
        telemetry.record_device_readback_operations(1);
        telemetry.record_timed_dispatch(100, Some(40), Some(25), Some(35));

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.host_to_device_bytes, 16);
        assert_eq!(snapshot.device_to_host_bytes, 8);
        assert_eq!(snapshot.readback_bytes, 8);
        assert_eq!(snapshot.transient_allocation_bytes_requested, 32);
        assert_eq!(snapshot.resident_allocation_bytes_requested, 64);
        assert_eq!(snapshot.param_upload_bytes, 4);
        assert_eq!(snapshot.cuda_graph_launches, 1);
        assert_eq!(snapshot.cuda_graph_materialized_cache_hits, 1);
        assert_eq!(snapshot.cuda_graph_batched_replay_chunks, 1);
        assert_eq!(snapshot.cuda_graph_batched_replay_lanes, 4);
        assert_eq!(snapshot.sync_points, 1);
        assert_eq!(snapshot.host_upload_operations, 2);
        assert_eq!(snapshot.device_readback_operations, 1);
        assert_eq!(snapshot.timed_dispatches, 1);
        assert_eq!(snapshot.timed_device_measurements, 1);
        assert_eq!(snapshot.timed_dispatches_missing_device_time, 0);
        assert_eq!(snapshot.timed_wall_ns_total, 100);
        assert_eq!(snapshot.timed_device_ns_total, 40);
        assert_eq!(snapshot.timed_device_ns_max, 40);
        assert_eq!(snapshot.timed_enqueue_ns_total, 25);
        assert_eq!(snapshot.timed_wait_ns_total, 35);
        assert_eq!(snapshot.wasted_thread_slots, 0);
        assert_eq!(snapshot.scheduled_thread_slot_overflows, 0);
        assert_eq!(snapshot.telemetry_counter_overflows, 0);
        assert_eq!(snapshot.logical_thread_utilization_bps, 0);
        assert_eq!(snapshot.logical_thread_waste_bps, 0);
        assert_eq!(snapshot.logical_elements_per_thread_slot_bps, 0);
        let prometheus = snapshot.to_prometheus_text();
        assert!(prometheus.contains("vyre_cuda_graph_materialized_cache_hits_total 1\n"));
        assert!(prometheus.contains("vyre_cuda_graph_batched_replay_chunks_total 1\n"));
        assert!(prometheus.contains("vyre_cuda_graph_batched_replay_lanes_total 4\n"));
        assert!(prometheus.contains("vyre_cuda_sync_points_total 1\n"));
        assert!(prometheus.contains("vyre_cuda_timed_dispatches_total 1\n"));
        assert!(prometheus.contains("vyre_cuda_timed_device_ns_total 40\n"));
        assert!(prometheus.contains("vyre_cuda_timed_device_ns_max 40\n"));
        assert!(prometheus.contains("vyre_cuda_telemetry_counter_overflows_total 0\n"));

        telemetry.reset();
        assert_eq!(telemetry.snapshot(), Default::default());
    }

    #[test]
    fn launch_snapshot_reports_logical_thread_utilization_proxy() {
        let telemetry = CudaTelemetry::default();
        let launch = vyre_driver::LaunchPlan {
            grid: [1, 1, 1],
            workgroup: [128, 1, 1],
            element_count: 64,
            param_words: Vec::new(),
            max_binding_alignment: 4,
        };
        telemetry.record_kernel_launch(&launch);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.kernel_launches, 1);
        assert_eq!(snapshot.scheduled_thread_slots, 128);
        assert_eq!(snapshot.scheduled_thread_slot_overflows, 0);
        assert_eq!(snapshot.telemetry_counter_overflows, 0);
        assert_eq!(snapshot.launched_elements, 64);
        assert_eq!(snapshot.wasted_thread_slots, 64);
        assert_eq!(snapshot.logical_thread_utilization_bps, 5_000);
        assert_eq!(snapshot.logical_thread_waste_bps, 5_000);
        assert_eq!(snapshot.logical_elements_per_thread_slot_bps, 5_000);
    }

    #[test]
    fn launch_snapshot_reports_unclamped_logical_element_density() {
        let telemetry = CudaTelemetry::default();
        let launch = vyre_driver::LaunchPlan {
            grid: [1, 1, 1],
            workgroup: [32, 1, 1],
            element_count: 96,
            param_words: Vec::new(),
            max_binding_alignment: 4,
        };
        telemetry.record_kernel_launch(&launch);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.scheduled_thread_slots, 32);
        assert_eq!(snapshot.scheduled_thread_slot_overflows, 0);
        assert_eq!(snapshot.telemetry_counter_overflows, 0);
        assert_eq!(snapshot.launched_elements, 96);
        assert_eq!(snapshot.wasted_thread_slots, 0);
        assert_eq!(snapshot.logical_thread_utilization_bps, 10_000);
        assert_eq!(snapshot.logical_thread_waste_bps, 0);
        assert_eq!(snapshot.logical_elements_per_thread_slot_bps, 30_000);
    }

    #[test]
    fn launch_snapshot_records_thread_slot_overflow_instead_of_panicking() {
        let telemetry = CudaTelemetry::default();
        let launch = vyre_driver::LaunchPlan {
            grid: [u32::MAX, u32::MAX, u32::MAX],
            workgroup: [1024, 1024, 64],
            element_count: 1,
            param_words: Vec::new(),
            max_binding_alignment: 4,
        };
        telemetry.record_kernel_launch(&launch);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.kernel_launches, 1);
        assert_eq!(snapshot.scheduled_thread_slots, 0);
        assert_eq!(snapshot.scheduled_thread_slot_overflows, 1);
        assert_eq!(snapshot.launched_elements, 1);
    }

    #[test]
    fn telemetry_counter_overflow_is_counted_instead_of_panicking_or_saturating() {
        use std::sync::atomic::Ordering;

        let telemetry = CudaTelemetry::default();
        telemetry
            .host_to_device_bytes
            .store(u64::MAX - 3, Ordering::Relaxed);

        telemetry.record_host_to_device_bytes(8);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.host_to_device_bytes, u64::MAX - 3);
        assert_eq!(snapshot.telemetry_counter_overflows, 1);
    }

    #[test]
    fn timed_dispatch_records_missing_device_time_without_losing_wall_time() {
        let telemetry = CudaTelemetry::default();
        telemetry.record_timed_dispatch(77, None, Some(11), Some(22));
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.timed_dispatches, 1);
        assert_eq!(snapshot.timed_device_measurements, 0);
        assert_eq!(snapshot.timed_dispatches_missing_device_time, 1);
        assert_eq!(snapshot.timed_wall_ns_total, 77);
        assert_eq!(snapshot.timed_device_ns_total, 0);
        assert_eq!(snapshot.timed_device_ns_max, 0);
        assert_eq!(snapshot.timed_enqueue_ns_total, 11);
        assert_eq!(snapshot.timed_wait_ns_total, 22);
    }

    #[test]
    fn timed_dispatch_tracks_max_device_latency() {
        let telemetry = CudaTelemetry::default();
        telemetry.record_timed_dispatch(10, Some(3), None, None);
        telemetry.record_timed_dispatch(20, Some(30), None, None);
        telemetry.record_timed_dispatch(30, Some(7), None, None);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.timed_dispatches, 3);
        assert_eq!(snapshot.timed_device_measurements, 3);
        assert_eq!(snapshot.timed_wall_ns_total, 60);
        assert_eq!(snapshot.timed_device_ns_total, 40);
        assert_eq!(snapshot.timed_device_ns_max, 30);
    }

    #[test]
    fn telemetry_production_paths_do_not_panic_on_counter_or_ratio_overflow() {
        let source = include_str!("telemetry.rs");
        let production = source
            .split("#[cfg(test)]")
            .next()
            .expect("Fix: telemetry source must contain production section");
        assert!(
            !production.contains(concat!("panic", "!("))
                && !production.contains(".unwrap_or_else(")
                && !production.contains(".expect("),
            "Fix: CUDA telemetry production paths must record overflow diagnostics instead of panicking."
        );
        assert!(
            production.contains("record_counter_overflow")
                && production.contains("scheduled_thread_slot_overflows")
                && production.contains("record_timed_dispatch")
                && production.contains("tracing::error!"),
            "Fix: CUDA telemetry overflow paths must stay observable after removing release-path panics."
        );
        assert!(
            production.contains("crate::numeric::CUDA_NUMERIC.ratio_basis_points_u64"),
            "Fix: CUDA telemetry basis-point math must use the shared backend numeric policy."
        );
    }
}
