use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use vyre_driver::BackendError;

use super::buffer_plan::PlannedBuffer;
use super::resident::{metal_physical_buffer_len, MetalResidentBufferTable};

pub(crate) type MetalMetricCounters = Arc<MetalMetrics>;

#[derive(Default)]
pub(crate) struct MetalMetrics {
    pub(super) pipeline_cache_hits: AtomicU64,
    pub(super) pipeline_cache_misses: AtomicU64,
    pub(super) pipeline_cache_miss_empty_cache: AtomicU64,
    pub(super) pipeline_cache_miss_program_changed: AtomicU64,
    pub(super) pipeline_cache_miss_dispatch_policy_changed: AtomicU64,
    pub(super) pipeline_cache_miss_device_or_runtime_changed: AtomicU64,
    pub(super) pipeline_cache_miss_key_absent: AtomicU64,
    pub(super) buffer_allocation_count: AtomicU64,
    pub(super) buffer_allocation_bytes: AtomicU64,
    pub(super) host_to_device_copy_count: AtomicU64,
    pub(super) host_to_device_bytes: AtomicU64,
    pub(super) device_to_host_copy_count: AtomicU64,
    pub(super) device_to_host_bytes: AtomicU64,
    pub(super) output_readback_bytes: AtomicU64,
}

/// Every scalar counter `backend_metric_snapshot` reports, paired with the
/// atomic that holds it.
pub(super) const METAL_COUNTERS: [(&str, fn(&MetalMetrics) -> &AtomicU64); 14] = [
    ("metal_pipeline_cache_hits", |m| &m.pipeline_cache_hits),
    ("metal_pipeline_cache_misses", |m| &m.pipeline_cache_misses),
    ("metal_pipeline_cache_miss_empty_cache", |m| {
        &m.pipeline_cache_miss_empty_cache
    }),
    ("metal_pipeline_cache_miss_program_changed", |m| {
        &m.pipeline_cache_miss_program_changed
    }),
    ("metal_pipeline_cache_miss_dispatch_policy_changed", |m| {
        &m.pipeline_cache_miss_dispatch_policy_changed
    }),
    ("metal_pipeline_cache_miss_device_or_runtime_changed", |m| {
        &m.pipeline_cache_miss_device_or_runtime_changed
    }),
    ("metal_pipeline_cache_miss_key_absent", |m| {
        &m.pipeline_cache_miss_key_absent
    }),
    ("metal_buffer_allocation_count", |m| {
        &m.buffer_allocation_count
    }),
    ("metal_buffer_allocation_bytes", |m| {
        &m.buffer_allocation_bytes
    }),
    ("metal_host_to_device_copy_count", |m| {
        &m.host_to_device_copy_count
    }),
    ("metal_host_to_device_bytes", |m| &m.host_to_device_bytes),
    ("metal_device_to_host_copy_count", |m| {
        &m.device_to_host_copy_count
    }),
    ("metal_device_to_host_bytes", |m| &m.device_to_host_bytes),
    ("metal_output_readback_bytes", |m| &m.output_readback_bytes),
];

pub(super) fn record_planned_buffer_metrics(metrics: &MetalMetrics, buffers: &[PlannedBuffer]) {
    let mut allocation_count = 0_u64;
    let mut allocation_bytes = 0_u64;
    let mut host_to_device_copy_count = 0_u64;
    let mut host_to_device_bytes = 0_u64;
    for buffer in buffers {
        if buffer.allocated_bytes > 0 {
            allocation_count = allocation_count.saturating_add(1);
            allocation_bytes =
                allocation_bytes.saturating_add(usize_to_u64_saturating(buffer.allocated_bytes));
        }
        if buffer.host_to_device_bytes > 0 {
            host_to_device_copy_count = host_to_device_copy_count.saturating_add(1);
            host_to_device_bytes = host_to_device_bytes
                .saturating_add(usize_to_u64_saturating(buffer.host_to_device_bytes));
        }
    }
    add_atomic_metric(&metrics.buffer_allocation_count, allocation_count);
    add_atomic_metric(&metrics.buffer_allocation_bytes, allocation_bytes);
    add_atomic_metric(
        &metrics.host_to_device_copy_count,
        host_to_device_copy_count,
    );
    add_atomic_metric(&metrics.host_to_device_bytes, host_to_device_bytes);
}

pub(super) fn record_output_readback_metrics(metrics: &MetalMetrics, outputs: &[Vec<u8>]) {
    let mut readback_count = 0_u64;
    let mut readback_bytes = 0_u64;
    for output in outputs {
        if !output.is_empty() {
            readback_count = readback_count.saturating_add(1);
            readback_bytes = readback_bytes.saturating_add(usize_to_u64_saturating(output.len()));
        }
    }
    add_atomic_metric(&metrics.device_to_host_copy_count, readback_count);
    add_atomic_metric(&metrics.device_to_host_bytes, readback_bytes);
    add_atomic_metric(&metrics.output_readback_bytes, readback_bytes);
}

pub(super) fn record_host_to_device_copy(metrics: &MetalMetrics, byte_len: usize) {
    if byte_len == 0 {
        return;
    }
    add_atomic_metric(&metrics.host_to_device_copy_count, 1);
    add_atomic_metric(
        &metrics.host_to_device_bytes,
        usize_to_u64_saturating(byte_len),
    );
}

pub(super) fn record_device_to_host_copy(metrics: &MetalMetrics, byte_len: usize) {
    if byte_len == 0 {
        return;
    }
    add_atomic_metric(&metrics.device_to_host_copy_count, 1);
    add_atomic_metric(
        &metrics.device_to_host_bytes,
        usize_to_u64_saturating(byte_len),
    );
}

pub(super) fn record_buffer_allocation(metrics: &MetalMetrics, byte_len: usize) {
    add_atomic_metric(&metrics.buffer_allocation_count, 1);
    add_atomic_metric(
        &metrics.buffer_allocation_bytes,
        usize_to_u64_saturating(metal_physical_buffer_len(byte_len)),
    );
}

fn add_atomic_metric(counter: &AtomicU64, value: u64) {
    if value == 0 {
        return;
    }
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_add(value))
    });
}

pub(crate) fn push_resident_table_metrics(
    resident_buffers: &MetalResidentBufferTable,
    metrics: &mut Vec<(&'static str, u64)>,
) {
    match resident_buffers.lock() {
        Ok(table) => {
            metrics.push(("metal_resident_buffer_count", table.len() as u64));
            let resident_bytes = table
                .values()
                .try_fold(0_u64, |total, resident| {
                    u64::try_from(resident.byte_len)
                        .ok()
                        .and_then(|byte_len| total.checked_add(byte_len))
                })
                .unwrap_or(u64::MAX);
            metrics.push(("metal_resident_bytes", resident_bytes));
        }
        Err(_poison) => {
            metrics.push(("metal_resident_buffer_count", u64::MAX));
            metrics.push(("metal_resident_bytes", u64::MAX));
            metrics.push(("metal_resident_buffer_error", 1_u64));
            tracing::error!(
                "metal resident_buffers Mutex is poisoned; \
                 resident buffer metrics are sentinel values (u64::MAX). \
                 Fix: a background dispatch thread panicked while holding \
                 the resident buffer table lock."
            );
        }
    }
}

pub(super) fn usize_to_u64_saturating(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

pub(super) fn bytes_per_second_to_gbps(value: u64) -> u32 {
    let gbps = value / 1_000_000_000;
    u32::try_from(gbps).unwrap_or(u32::MAX)
}

pub(super) fn elapsed_ns(started: Instant, field: &'static str) -> Result<u64, BackendError> {
    u64::try_from(started.elapsed().as_nanos()).map_err(|error| BackendError::InvalidProgram {
        fix: format!(
            "Fix: {field} timing cannot fit u64 nanoseconds: {error}. Split telemetry windows."
        ),
    })
}
