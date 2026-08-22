/// Host-visible tenant runtime counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TenantRuntimeCounters {
    /// Tenant id.
    pub tenant_id: u32,
    /// Number of slots ever published by this tenant.
    pub published_count: u64,
    /// Number of slots observed drained for this tenant.
    pub drained_count: u64,
    /// Current host-visible backlog (`published_count - drained_count`).
    pub outstanding_slots: u64,
    /// Configured outstanding-slot cap for this tenant.
    pub max_outstanding_slots: u64,
    /// Number of quiesce calls recorded for this tenant.
    pub quiesce_calls: u64,
    /// Number of quiesce calls that timed out.
    pub quiesce_timeouts: u64,
    /// Cumulative nanoseconds spent waiting for this tenant to drain.
    pub quiesce_wait_ns: u64,
}

/// Host-visible tenant quota counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TenantQuotaCounters {
    /// Tenant id.
    pub tenant_id: u32,
    /// Current reserved staging bytes.
    pub staging_bytes: u64,
    /// Configured staging byte cap.
    pub max_staging_bytes: u64,
    /// Current reserved resident handle count.
    pub resident_handles: u64,
    /// Configured resident handle cap.
    pub max_resident_handles: u64,
}
