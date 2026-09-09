//! The registry that issues tenant ids and opcode windows.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::DashMap;

use crate::resident_work_queue::protocol::opcode::SHUTDOWN;
use crate::{CounterArithmetic, CounterScope, PipelineError};

use super::counters::TenantRuntimeCounters;
use super::error::TenantError;
use super::handle::{TenantHandle, TenantState};
use super::quiesce::tenant_registry_retry_idle;
use super::quota::TenantQuota;
use super::{OPCODE_RANGE_PER_TENANT, TENANT_ID_MAX, TENANT_OPCODE_BASE};

/// Maximum number of distinct opcode windows below SHUTDOWN.
pub(super) const MAX_TENANT_OPCODE_WINDOWS: u32 =
    (SHUTDOWN - TENANT_OPCODE_BASE) / OPCODE_RANGE_PER_TENANT;

/// Thread-safe tenant registry. One per megakernel instance.
pub struct TenantRegistry {
    pub(super) tenants: DashMap<u32, TenantHandle>,
    pub(super) generations: DashMap<u32, u32>,
    pub(super) free_list: std::sync::Mutex<Vec<u32>>,
    pub(super) next_id: AtomicU32,
}

impl Default for TenantRegistry {
    fn default() -> Self {
        Self {
            tenants: DashMap::new(),
            generations: DashMap::new(),
            free_list: std::sync::Mutex::new(Vec::new()),
            next_id: AtomicU32::new(0),
        }
    }
}

impl TenantRegistry {
    /// Take the free list, rebuilding it from the live tenant map after a panic.
    ///
    /// The list is a derived index: the canonical input is `next_id` and the
    /// tenant map, so a half-written list is discarded and re-derived rather
    /// than read.
    fn lock_free_list(&self) -> std::sync::MutexGuard<'_, Vec<u32>> {
        vyre_foundation::failure_domain::govern_mutex_restartable(
            &self.free_list,
            "runtime tenant registry",
            "the retired tenant id free list",
            |free_list| {
                free_list.clear();
                let current_next = self.next_id.load(Ordering::Relaxed);
                for id in (1..current_next).rev() {
                    if !self.tenants.contains_key(&id) {
                        free_list.push(id);
                    }
                }
            },
        )
    }

    /// Fresh registry with no tenants.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new tenant with the standard finite quota policy.
    /// Returns a handle whose opcode range is reserved until
    /// [`unregister`](Self::unregister) is called.
    ///
    /// # Errors
    ///
    /// Returns [`TenantError::RegistryFull`] when the tenant id or
    /// opcode space is exhausted.
    pub fn register(&self, label: impl Into<String>) -> Result<TenantHandle, TenantError> {
        self.register_with_quotas(label, TenantQuota::standard())
    }

    /// Register a new tenant with a bounded outstanding-slot budget.
    ///
    /// # Errors
    ///
    /// Returns [`TenantError::NonFiniteQuota`] if `max_outstanding_slots` is not
    /// finite (`u64::MAX`). Returns [`TenantError::RegistryFull`] when the tenant
    /// id or opcode space is exhausted.
    pub fn register_with_backpressure(
        &self,
        label: impl Into<String>,
        max_outstanding_slots: u64,
    ) -> Result<TenantHandle, TenantError> {
        if max_outstanding_slots == u64::MAX {
            return Err(TenantError::NonFiniteQuota {
                field: "max_outstanding_slots",
                value: max_outstanding_slots,
                fix: "register tenants with a finite outstanding-slot limit",
            });
        }
        self.register_with_quotas(
            label,
            TenantQuota {
                max_outstanding_slots,
                ..TenantQuota::standard()
            },
        )
    }

    /// Register a tenant with explicit ring-slot, staging-byte, and
    /// resident-handle quotas.
    ///
    /// # Errors
    ///
    /// Returns [`TenantError::NonFiniteQuota`] if any quota limit is not finite (`u64::MAX`).
    /// Returns [`TenantError::RegistryFull`] when the tenant id or opcode space
    /// is exhausted.
    pub fn register_with_quotas(
        &self,
        label: impl Into<String>,
        quota: TenantQuota,
    ) -> Result<TenantHandle, TenantError> {
        if quota.max_outstanding_slots == u64::MAX {
            return Err(TenantError::NonFiniteQuota {
                field: "max_outstanding_slots",
                value: quota.max_outstanding_slots,
                fix: "register tenants with a finite outstanding-slot limit",
            });
        }
        if quota.max_staging_bytes == u64::MAX {
            return Err(TenantError::NonFiniteQuota {
                field: "max_staging_bytes",
                value: quota.max_staging_bytes,
                fix: "register tenants with a finite staging-byte limit",
            });
        }
        if quota.max_resident_handles == u64::MAX {
            return Err(TenantError::NonFiniteQuota {
                field: "max_resident_handles",
                value: quota.max_resident_handles,
                fix: "register tenants with a finite resident-handle limit",
            });
        }
        let (id, generation) = {
            let mut free = self.lock_free_list();
            if let Some(recycled_id) = free.pop() {
                let mut entry = self.generations.entry(recycled_id).or_insert(1);
                *entry = entry.wrapping_add(1).max(1);
                (recycled_id, *entry)
            } else {
                let mut registration_retries = 0u64;
                let issued = vyre_driver::accounting::checked_atomic_update_u32_with_order(
                    &self.next_id,
                    Ordering::Relaxed,
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                    |current| {
                        if current >= MAX_TENANT_OPCODE_WINDOWS || current >= TENANT_ID_MAX {
                            return Err(TenantError::RegistryFull { issued: current });
                        }
                        let id = current.max(1);
                        id.checked_add(1)
                            .ok_or(TenantError::RegistryFull { issued: current })
                    },
                    |_, _| {
                        tenant_registry_retry_idle(registration_retries);
                        let retries = registration_retries;
                        registration_retries = vyre_driver::accounting::checked_add_u64_lazy(
                            retries,
                            1,
                            || {
                                TenantError::Pipeline(PipelineError::CounterOverflow {
                                    scope: CounterScope::TenantRegistry,
                                    counter: "registration retry count",
                                    arithmetic: CounterArithmetic::Sum,
                                    lhs: retries,
                                    rhs: 1,
                                    bits: 64,
                                    fix:
                                        "retry registration later; the id allocator has not settled",
                                })
                            },
                        )?;
                        Ok(())
                    },
                )?;
                let id = issued.max(1);
                self.generations.insert(id, 1);
                (id, 1)
            }
        };

        let tenant_offset = vyre_driver::accounting::checked_mul_u32_value(
            id,
            OPCODE_RANGE_PER_TENANT,
            TenantError::RegistryFull { issued: id },
        )?;
        let base_opcode = vyre_driver::accounting::checked_add_u32_value(
            TENANT_OPCODE_BASE,
            tenant_offset,
            TenantError::RegistryFull { issued: id },
        )?;
        let top_opcode = vyre_driver::accounting::checked_add_u32_value(
            base_opcode,
            OPCODE_RANGE_PER_TENANT,
            TenantError::RegistryFull { issued: id },
        )?;
        if top_opcode == SHUTDOWN {
            return Err(TenantError::RegistryFull { issued: id });
        }
        let handle = TenantHandle {
            state: Arc::new(TenantState {
                id,
                generation,
                base_opcode,
                opcode_cap: OPCODE_RANGE_PER_TENANT,
                published_count: AtomicU64::new(0),
                max_outstanding_slots: quota.max_outstanding_slots.max(1),
                staging_bytes: AtomicU64::new(0),
                max_staging_bytes: quota.max_staging_bytes.max(1),
                resident_handles: AtomicU64::new(0),
                max_resident_handles: quota.max_resident_handles.max(1),
                drained_count: AtomicU64::new(0),
                quiesce_calls: AtomicU64::new(0),
                quiesce_timeouts: AtomicU64::new(0),
                quiesce_wait_ns: AtomicU64::new(0),
                revoked: AtomicU32::new(0),
                label: label.into(),
            }),
        };
        self.tenants.insert(id, handle.clone());
        Ok(handle)
    }

    /// Unregister a tenant. Future publishes on the handle fail
    /// with [`TenantError::Revoked`]. In-flight slots already on
    /// the GPU still execute  -  the host is responsible for
    /// quiescing before unregister if it needs that guarantee.
    pub fn unregister(&self, tenant_id: u32) -> Option<TenantHandle> {
        let (_, handle) = self.tenants.remove(&tenant_id)?;
        handle.state.revoked.store(1, Ordering::Release);
        handle.release_all_resource_reservations();
        let mut free = self.lock_free_list();
        free.push(tenant_id);
        Some(handle)
    }

    /// Snapshot of active tenants for observability / diagnostics.
    #[must_use]
    pub fn active_tenants(&self) -> Vec<TenantHandle> {
        let mut out = Vec::with_capacity(self.tenants.len());
        out.extend(self.tenants.iter().map(|entry| entry.value().clone()));
        out.sort_by_key(TenantHandle::id);
        out
    }

    /// Snapshot active tenants into caller-owned storage.
    pub fn active_tenants_into(&self, out: &mut Vec<TenantHandle>) {
        out.clear();
        out.reserve(self.tenants.len());
        self.tenants
            .iter()
            .for_each(|entry| out.push(entry.value().clone()));
        out.sort_by_key(TenantHandle::id);
    }

    /// Look up a tenant by id. Returns `None` if the id was
    /// unregistered.
    #[must_use]
    pub fn lookup(&self, tenant_id: u32) -> Option<TenantHandle> {
        self.tenants
            .get(&tenant_id)
            .map(|entry| entry.value().clone())
    }

    /// Snapshot runtime counters for every active tenant.
    #[must_use]
    pub fn runtime_counters(&self) -> Vec<TenantRuntimeCounters> {
        let mut out = Vec::with_capacity(self.tenants.len());
        self.tenants
            .iter()
            .map(|entry| entry.value().runtime_counters())
            .for_each(|counters| out.push(counters));
        out.sort_by_key(|counters| counters.tenant_id);
        out
    }

    /// Snapshot runtime counters into caller-owned storage.
    pub fn runtime_counters_into(&self, out: &mut Vec<TenantRuntimeCounters>) {
        out.clear();
        out.reserve(self.tenants.len());
        self.tenants
            .iter()
            .map(|entry| entry.value().runtime_counters())
            .for_each(|counters| out.push(counters));
        out.sort_by_key(|counters| counters.tenant_id);
    }
}

impl crate::atomic_recovery::StateOwnerRecovery for TenantRegistry {
    fn failure_domain(&self) -> crate::FailureDomain {
        crate::FailureDomain::MemoryState
    }

    fn recovery_class(&self) -> crate::RecoveryClass {
        crate::RecoveryClass::RestartableFromCanonicalInput
    }
}
