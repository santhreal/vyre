//! Per-tenant resource ceilings and the saturating counters behind them.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::PipelineError;

use super::error::TenantError;

/// Per-tenant resource quota.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TenantQuota {
    /// Maximum host-visible ring slots the tenant may keep outstanding.
    pub max_outstanding_slots: u64,
    /// Maximum staging bytes the tenant may reserve for pending work.
    pub max_staging_bytes: u64,
    /// Maximum resident handles the tenant may hold at once.
    pub max_resident_handles: u64,
}

impl TenantQuota {
    /// Unbounded tenant quota for compatibility with the legacy registration
    /// API. Individual fields are still normalized to at least one resource
    /// slot during registration.
    #[must_use]
    pub const fn unbounded() -> Self {
        Self {
            max_outstanding_slots: u64::MAX,
            max_staging_bytes: u64::MAX,
            max_resident_handles: u64::MAX,
        }
    }

    /// Build a bounded tenant quota.
    #[must_use]
    pub const fn bounded(
        max_outstanding_slots: u64,
        max_staging_bytes: u64,
        max_resident_handles: u64,
    ) -> Self {
        Self {
            max_outstanding_slots,
            max_staging_bytes,
            max_resident_handles,
        }
    }
}

pub(super) fn saturating_atomic_add_u64(counter: &AtomicU64, value: u64, _label: &'static str) {
    let mut current = counter.load(Ordering::Acquire);
    loop {
        let next = current.saturating_add(value);
        match counter.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return,
            Err(observed) => current = observed,
        }
    }
}

pub(super) fn saturating_atomic_sub_u64(counter: &AtomicU64, value: u64, _label: &'static str) {
    let mut current = counter.load(Ordering::Acquire);
    loop {
        let next = current.saturating_sub(value);
        match counter.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return,
            Err(observed) => current = observed,
        }
    }
}

pub(super) fn reserve_resource_quota(
    counter: &AtomicU64,
    value: u64,
    cap: u64,
    backpressure: impl Fn() -> TenantError,
    overflow_fix: &'static str,
) -> Result<(), TenantError> {
    vyre_driver::accounting::checked_atomic_update_u64_with_order(
        counter,
        Ordering::Acquire,
        Ordering::AcqRel,
        Ordering::Acquire,
        |used| {
            let next = vyre_driver::accounting::checked_add_u64_lazy(used, value, || {
                TenantError::Pipeline(PipelineError::QueueFull {
                    queue: "tenant resource quota",
                    fix: overflow_fix,
                })
            })?;
            if next > cap {
                return Err(backpressure());
            }
            Ok(next)
        },
        |_, _| Ok(()),
    )?;
    Ok(())
}

pub(super) fn release_resource_quota(
    counter: &AtomicU64,
    value: u64,
    tenant_id: u32,
    resource: &'static str,
) -> Result<(), TenantError> {
    vyre_driver::accounting::checked_atomic_update_u64_with_order(
        counter,
        Ordering::Acquire,
        Ordering::AcqRel,
        Ordering::Acquire,
        |used| {
            used.checked_sub(value)
                .ok_or(TenantError::ResourceUnderflow {
                    tenant_id,
                    resource,
                    requested: value,
                    used,
                })
        },
        |_, _| Ok(()),
    )?;
    Ok(())
}
