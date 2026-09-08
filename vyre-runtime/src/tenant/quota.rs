//! Per-tenant resource ceilings, the saturating counters behind them, and the
//! handle surface that reserves and releases against them.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::{CounterArithmetic, CounterScope, PipelineError};

use super::counters::TenantQuotaCounters;
use super::error::TenantError;
use super::handle::TenantHandle;

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

impl Default for TenantQuota {
    fn default() -> Self {
        Self {
            max_outstanding_slots: 65_536,
            max_staging_bytes: 1024 * 1024 * 1024,
            max_resident_handles: 16_384,
        }
    }
}

impl TenantQuota {
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
    tenant_id: u32,
    resource: &'static str,
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
                TenantError::Pipeline(PipelineError::CounterOverflow {
                    scope: CounterScope::Tenant(tenant_id),
                    counter: resource,
                    arithmetic: CounterArithmetic::Sum,
                    lhs: used,
                    rhs: value,
                    bits: 64,
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

impl TenantHandle {
    /// Reserve staging bytes against this tenant's quota.
    pub fn reserve_staging_bytes(&self, byte_count: u64) -> Result<(), TenantError> {
        self.ensure_not_revoked()?;
        reserve_resource_quota(
            &self.state.staging_bytes,
            byte_count,
            self.state.max_staging_bytes,
            self.state.id,
            "staging bytes",
            || TenantError::StagingBackpressure {
                tenant_id: self.state.id,
                requested: byte_count,
                used: self.state.staging_bytes.load(Ordering::Acquire),
                cap: self.state.max_staging_bytes,
            },
            "release staging reservations or recreate the tenant before reserving more bytes",
        )
    }

    /// Release staging bytes previously reserved by this tenant.
    pub fn release_staging_bytes(&self, byte_count: u64) -> Result<(), TenantError> {
        release_resource_quota(
            &self.state.staging_bytes,
            byte_count,
            self.state.id,
            "staging bytes",
        )
    }

    /// Reserve resident handles against this tenant's quota.
    pub fn reserve_resident_handles(&self, handle_count: u64) -> Result<(), TenantError> {
        self.ensure_not_revoked()?;
        reserve_resource_quota(
            &self.state.resident_handles,
            handle_count,
            self.state.max_resident_handles,
            self.state.id,
            "resident handles",
            || TenantError::ResidentHandleBackpressure {
                tenant_id: self.state.id,
                requested: handle_count,
                used: self.state.resident_handles.load(Ordering::Acquire),
                cap: self.state.max_resident_handles,
            },
            "release resident handles or recreate the tenant before reserving more handles",
        )
    }

    /// Release resident handles previously reserved by this tenant.
    pub fn release_resident_handles(&self, handle_count: u64) -> Result<(), TenantError> {
        release_resource_quota(
            &self.state.resident_handles,
            handle_count,
            self.state.id,
            "resident handles",
        )
    }

    /// Snapshot quota counters for this tenant.
    #[must_use]
    pub fn quota_counters(&self) -> TenantQuotaCounters {
        TenantQuotaCounters {
            tenant_id: self.state.id,
            staging_bytes: self.state.staging_bytes.load(Ordering::Acquire),
            max_staging_bytes: self.state.max_staging_bytes,
            resident_handles: self.state.resident_handles.load(Ordering::Acquire),
            max_resident_handles: self.state.max_resident_handles,
        }
    }

    pub(super) fn release_all_resource_reservations(&self) {
        self.state.staging_bytes.store(0, Ordering::Release);
        self.state.resident_handles.store(0, Ordering::Release);
    }
}
