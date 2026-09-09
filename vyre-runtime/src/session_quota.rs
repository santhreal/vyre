//! Mandatory finite runtime session quotas and typed session identity.
//!
//! Every queue, tenant, cache, retained generation, IO request, retry, and telemetry stream
//! carries a mandatory finite quota that is part of session identity. No unbounded constructor
//! exists on any quota type.

use thiserror::Error;
use vyre_driver::DeviceIdentity;
use vyre_megakernel::Digest;

pub use crate::tenant::TenantQuota;

/// Error returned when an unbounded or invalid quota is rejected.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SessionQuotaError {
    /// Quota limit was non-finite (equal to u64::MAX or usize::MAX).
    #[error("non-finite quota rejected for subsystem `{subsystem}`: limit `{limit}` must be strictly bounded. Fix: specify explicit finite resource limits with bounded() or standard()")]
    NonFiniteLimit {
        /// Subsystem name.
        subsystem: &'static str,
        /// Limit name.
        limit: &'static str,
    },
}

/// Bounded quota for runtime work queues and submission depths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueQuota {
    /// Maximum host-visible ring slots outstanding.
    pub max_outstanding_slots: u64,
    /// Maximum submission queue depth.
    pub max_queue_depth: usize,
}

impl QueueQuota {
    /// Standard finite queue quota.
    pub const DEFAULT_MAX_OUTSTANDING_SLOTS: u64 = 1024;
    /// Standard finite submission queue depth.
    pub const DEFAULT_MAX_QUEUE_DEPTH: usize = 256;

    /// Build a bounded queue quota with explicit finite limits.
    #[must_use]
    pub const fn bounded(max_outstanding_slots: u64, max_queue_depth: usize) -> Self {
        Self {
            max_outstanding_slots,
            max_queue_depth,
        }
    }

    /// Standard finite queue quota derived from runtime policy.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            max_outstanding_slots: Self::DEFAULT_MAX_OUTSTANDING_SLOTS,
            max_queue_depth: Self::DEFAULT_MAX_QUEUE_DEPTH,
        }
    }

    /// Check if all quota limits are strictly finite.
    #[must_use]
    pub const fn is_finite(&self) -> bool {
        self.max_outstanding_slots < u64::MAX && self.max_queue_depth < usize::MAX
    }
}

impl Default for QueueQuota {
    fn default() -> Self {
        Self::standard()
    }
}

/// Bounded quota for in-memory and disk pipeline caches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheQuota {
    /// Maximum cache entries.
    pub max_entries: usize,
    /// Maximum cache memory in bytes.
    pub max_bytes: u64,
}

impl CacheQuota {
    /// Standard finite cache entry limit (4096 entries).
    pub const DEFAULT_MAX_ENTRIES: usize = 4096;
    /// Standard finite cache byte budget (256 MiB).
    pub const DEFAULT_MAX_BYTES: u64 = 256 * 1024 * 1024;

    /// Build a bounded cache quota with explicit finite limits.
    #[must_use]
    pub const fn bounded(max_entries: usize, max_bytes: u64) -> Self {
        Self {
            max_entries,
            max_bytes,
        }
    }

    /// Standard finite cache quota derived from runtime policy.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            max_entries: Self::DEFAULT_MAX_ENTRIES,
            max_bytes: Self::DEFAULT_MAX_BYTES,
        }
    }

    /// Check if all quota limits are strictly finite.
    #[must_use]
    pub const fn is_finite(&self) -> bool {
        self.max_entries < usize::MAX && self.max_bytes < u64::MAX
    }
}

impl Default for CacheQuota {
    fn default() -> Self {
        Self::standard()
    }
}

/// Bounded quota for retained artifact generations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetainedGenerationQuota {
    /// Maximum retained generations.
    pub max_generations: u64,
    /// Maximum retained value bindings.
    pub max_retained_values: usize,
}

impl RetainedGenerationQuota {
    /// Standard finite generation limit (64 generations).
    pub const DEFAULT_MAX_GENERATIONS: u64 = 64;
    /// Standard finite retained value count (256 values).
    pub const DEFAULT_MAX_RETAINED_VALUES: usize = 256;

    /// Build a bounded retained generation quota with explicit finite limits.
    #[must_use]
    pub const fn bounded(max_generations: u64, max_retained_values: usize) -> Self {
        Self {
            max_generations,
            max_retained_values,
        }
    }

    /// Standard finite retained generation quota.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            max_generations: Self::DEFAULT_MAX_GENERATIONS,
            max_retained_values: Self::DEFAULT_MAX_RETAINED_VALUES,
        }
    }

    /// Check if all quota limits are strictly finite.
    #[must_use]
    pub const fn is_finite(&self) -> bool {
        self.max_generations < u64::MAX && self.max_retained_values < usize::MAX
    }
}

impl Default for RetainedGenerationQuota {
    fn default() -> Self {
        Self::standard()
    }
}

/// Bounded quota for IO requests and data streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IoQuota {
    /// Maximum concurrent in-flight IO requests.
    pub max_inflight_requests: usize,
    /// Maximum IO staging bytes.
    pub max_transfer_bytes: u64,
}

impl IoQuota {
    /// Standard finite in-flight request limit (128 requests).
    pub const DEFAULT_MAX_INFLIGHT_REQUESTS: usize = 128;
    /// Standard finite transfer byte budget (64 MiB).
    pub const DEFAULT_MAX_TRANSFER_BYTES: u64 = 64 * 1024 * 1024;

    /// Build a bounded IO quota with explicit finite limits.
    #[must_use]
    pub const fn bounded(max_inflight_requests: usize, max_transfer_bytes: u64) -> Self {
        Self {
            max_inflight_requests,
            max_transfer_bytes,
        }
    }

    /// Standard finite IO quota derived from runtime policy.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            max_inflight_requests: Self::DEFAULT_MAX_INFLIGHT_REQUESTS,
            max_transfer_bytes: Self::DEFAULT_MAX_TRANSFER_BYTES,
        }
    }

    /// Check if all quota limits are strictly finite.
    #[must_use]
    pub const fn is_finite(&self) -> bool {
        self.max_inflight_requests < usize::MAX && self.max_transfer_bytes < u64::MAX
    }
}

impl Default for IoQuota {
    fn default() -> Self {
        Self::standard()
    }
}

/// Bounded quota for supervisor restart budgets and retry attempts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryQuota {
    /// Maximum allowed crash restarts.
    pub max_restarts: u32,
    /// Maximum operation timeout in microseconds.
    pub max_timeout_micros: u64,
}

impl RetryQuota {
    /// Standard finite restart ceiling (5 restarts).
    pub const DEFAULT_MAX_RESTARTS: u32 = 5;
    /// Standard finite timeout (30 seconds).
    pub const DEFAULT_MAX_TIMEOUT_MICROS: u64 = 30_000_000;

    /// Build a bounded retry quota with explicit finite limits.
    #[must_use]
    pub const fn bounded(max_restarts: u32, max_timeout_micros: u64) -> Self {
        Self {
            max_restarts,
            max_timeout_micros,
        }
    }

    /// Standard finite retry quota.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            max_restarts: Self::DEFAULT_MAX_RESTARTS,
            max_timeout_micros: Self::DEFAULT_MAX_TIMEOUT_MICROS,
        }
    }

    /// Check if all quota limits are strictly finite.
    #[must_use]
    pub const fn is_finite(&self) -> bool {
        self.max_restarts < u32::MAX && self.max_timeout_micros < u64::MAX
    }
}

impl Default for RetryQuota {
    fn default() -> Self {
        Self::standard()
    }
}

/// Bounded quota for telemetry events and telemetry ring buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TelemetryQuota {
    /// Maximum telemetry events per collection window.
    pub max_events_per_window: u64,
    /// Telemetry ring buffer capacity in slots.
    pub ring_buffer_capacity: usize,
}

impl TelemetryQuota {
    /// Standard finite telemetry event ceiling (10,000 events).
    pub const DEFAULT_MAX_EVENTS_PER_WINDOW: u64 = 10_000;
    /// Standard finite telemetry ring capacity (1024 slots).
    pub const DEFAULT_RING_BUFFER_CAPACITY: usize = 1024;

    /// Build a bounded telemetry quota with explicit finite limits.
    #[must_use]
    pub const fn bounded(max_events_per_window: u64, ring_buffer_capacity: usize) -> Self {
        Self {
            max_events_per_window,
            ring_buffer_capacity,
        }
    }

    /// Standard finite telemetry quota.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            max_events_per_window: Self::DEFAULT_MAX_EVENTS_PER_WINDOW,
            ring_buffer_capacity: Self::DEFAULT_RING_BUFFER_CAPACITY,
        }
    }

    /// Check if all quota limits are strictly finite.
    #[must_use]
    pub const fn is_finite(&self) -> bool {
        self.max_events_per_window < u64::MAX && self.ring_buffer_capacity < usize::MAX
    }
}

impl Default for TelemetryQuota {
    fn default() -> Self {
        Self::standard()
    }
}

/// Comprehensive finite session quota enforcing resource bounds across all subsystems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionQuota {
    /// Queue subsystem quota.
    pub queue: QueueQuota,
    /// Tenant subsystem quota.
    pub tenant: TenantQuota,
    /// Cache subsystem quota.
    pub cache: CacheQuota,
    /// Retained generation quota.
    pub retained_generation: RetainedGenerationQuota,
    /// IO subsystem quota.
    pub io: IoQuota,
    /// Retry/restart subsystem quota.
    pub retry: RetryQuota,
    /// Telemetry subsystem quota.
    pub telemetry: TelemetryQuota,
}

impl SessionQuota {
    /// Build an explicit finite session quota.
    #[must_use]
    pub const fn bounded(
        queue: QueueQuota,
        tenant: TenantQuota,
        cache: CacheQuota,
        retained_generation: RetainedGenerationQuota,
        io: IoQuota,
        retry: RetryQuota,
        telemetry: TelemetryQuota,
    ) -> Self {
        Self {
            queue,
            tenant,
            cache,
            retained_generation,
            io,
            retry,
            telemetry,
        }
    }

    /// Standard finite session quota.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            queue: QueueQuota::standard(),
            tenant: TenantQuota::standard(),
            cache: CacheQuota::standard(),
            retained_generation: RetainedGenerationQuota::standard(),
            io: IoQuota::standard(),
            retry: RetryQuota::standard(),
            telemetry: TelemetryQuota::standard(),
        }
    }

    /// Check if all subsystem quotas are strictly finite.
    #[must_use]
    pub const fn is_finite(&self) -> bool {
        self.queue.is_finite()
            && self.tenant.is_finite()
            && self.cache.is_finite()
            && self.retained_generation.is_finite()
            && self.io.is_finite()
            && self.retry.is_finite()
            && self.telemetry.is_finite()
    }

    /// Validate that all quota limits are finite.
    ///
    /// # Errors
    ///
    /// Returns [`SessionQuotaError::NonFiniteLimit`] if any quota limit is unbounded.
    pub fn validate(&self) -> Result<(), SessionQuotaError> {
        if !self.queue.is_finite() {
            return Err(SessionQuotaError::NonFiniteLimit {
                subsystem: "queue",
                limit: "max_outstanding_slots or max_queue_depth",
            });
        }
        if !self.tenant.is_finite() {
            return Err(SessionQuotaError::NonFiniteLimit {
                subsystem: "tenant",
                limit: "tenant resource quotas",
            });
        }
        if !self.cache.is_finite() {
            return Err(SessionQuotaError::NonFiniteLimit {
                subsystem: "cache",
                limit: "max_entries or max_bytes",
            });
        }
        if !self.retained_generation.is_finite() {
            return Err(SessionQuotaError::NonFiniteLimit {
                subsystem: "retained_generation",
                limit: "max_generations or max_retained_values",
            });
        }
        if !self.io.is_finite() {
            return Err(SessionQuotaError::NonFiniteLimit {
                subsystem: "io",
                limit: "max_inflight_requests or max_transfer_bytes",
            });
        }
        if !self.retry.is_finite() {
            return Err(SessionQuotaError::NonFiniteLimit {
                subsystem: "retry",
                limit: "max_restarts or max_timeout_micros",
            });
        }
        if !self.telemetry.is_finite() {
            return Err(SessionQuotaError::NonFiniteLimit {
                subsystem: "telemetry",
                limit: "max_events_per_window or ring_buffer_capacity",
            });
        }
        Ok(())
    }
}

impl Default for SessionQuota {
    fn default() -> Self {
        Self::standard()
    }
}

/// Strongly typed session identity carrying mandatory finite quotas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionIdentity {
    /// Monotonically allocated session identifier.
    pub session_id: u64,
    /// Canonical artifact digest.
    pub artifact: Digest,
    /// Registered device generation identity.
    pub device: DeviceIdentity,
    /// Mandatory finite session quotas.
    pub quota: SessionQuota,
}

impl SessionIdentity {
    /// Construct a new session identity, validating that all quotas are finite.
    ///
    /// # Errors
    ///
    /// Returns [`SessionQuotaError::NonFiniteLimit`] if any quota in `quota` is unbounded.
    pub fn new(
        session_id: u64,
        artifact: Digest,
        device: DeviceIdentity,
        quota: SessionQuota,
    ) -> Result<Self, SessionQuotaError> {
        quota.validate()?;
        Ok(Self {
            session_id,
            artifact,
            device,
            quota,
        })
    }

    /// Check if the session quotas are finite.
    #[must_use]
    pub const fn is_finite(&self) -> bool {
        self.quota.is_finite()
    }
}
