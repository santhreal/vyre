//! The registry that issues tenant ids and opcode windows.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::DashMap;

use crate::resident_work_queue::protocol::opcode::SHUTDOWN;
use crate::PipelineError;

use super::error::TenantError;
use super::handle::{tenant_registry_retry_idle, TenantHandle, TenantRuntimeCounters, TenantState};
use super::quota::TenantQuota;
use super::{OPCODE_RANGE_PER_TENANT, TENANT_ID_MAX, TENANT_OPCODE_BASE};

/// Maximum number of distinct opcode windows below SHUTDOWN.
pub(super) const MAX_TENANT_OPCODE_WINDOWS: u32 =
    (SHUTDOWN - TENANT_OPCODE_BASE) / OPCODE_RANGE_PER_TENANT;

/// Thread-safe tenant registry. One per megakernel instance.
pub struct TenantRegistry {
    tenants: DashMap<u32, TenantHandle>,
    generations: DashMap<u32, u32>,
    free_list: std::sync::Mutex<Vec<u32>>,
    next_id: AtomicU32,
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

/// Caller-owned scratch for repeated concurrent-tenant selection.
#[derive(Debug, Default)]
pub struct TenantSelectionScratch {
    pub(super) active_ids: Vec<u32>,
    pub(super) selected_indices: Vec<usize>,
}

impl TenantSelectionScratch {
    /// Construct empty tenant-selection scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            active_ids: Vec::new(),
            selected_indices: Vec::new(),
        }
    }
}

impl TenantRegistry {
    /// Fresh registry with no tenants.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new tenant with the given diagnostic label.
    /// Returns a handle whose opcode range is reserved until
    /// [`unregister`](Self::unregister) is called.
    ///
    /// # Errors
    ///
    /// Returns [`TenantError::RegistryFull`] when the tenant id or
    /// opcode space is exhausted.
    pub fn register(&self, label: impl Into<String>) -> Result<TenantHandle, TenantError> {
        self.register_with_backpressure(label, u64::MAX)
    }

    /// Register a new tenant with a bounded outstanding-slot budget.
    ///
    /// # Errors
    ///
    /// Returns [`TenantError::RegistryFull`] when the tenant id or opcode space
    /// is exhausted.
    pub fn register_with_backpressure(
        &self,
        label: impl Into<String>,
        max_outstanding_slots: u64,
    ) -> Result<TenantHandle, TenantError> {
        self.register_with_quotas(
            label,
            TenantQuota {
                max_outstanding_slots,
                ..TenantQuota::unbounded()
            },
        )
    }

    /// Register a tenant with explicit ring-slot, staging-byte, and
    /// resident-handle quotas.
    ///
    /// # Errors
    ///
    /// Returns [`TenantError::RegistryFull`] when the tenant id or opcode space
    /// is exhausted.
    pub fn register_with_quotas(
        &self,
        label: impl Into<String>,
        quota: TenantQuota,
    ) -> Result<TenantHandle, TenantError> {
        let (id, generation) = {
            let mut free = self.free_list.lock().unwrap_or_else(|e| e.into_inner());
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
                        registration_retries = vyre_driver::accounting::checked_add_u64_lazy(
                            registration_retries,
                            1,
                            || {
                                TenantError::Pipeline(PipelineError::QueueFull {
                                    queue: "tenant",
                                    fix: "tenant registration retry counter overflowed u64; retry registration later",
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
        if top_opcode >= SHUTDOWN {
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
        let mut free = self.free_list.lock().unwrap_or_else(|e| e.into_inner());
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

    /// Select a maximal independent subset of tenants for a fair
    /// schedule slot.
    ///
    /// `conflict_adj[i*n+j] != 0` means tenants `i` and `j` cannot
    /// share the same dispatch slot (e.g., both pinned to the same
    /// queue, or both holding mutually-exclusive opcode locks). The
    /// Returns a Vec of tenant ids in selection order. Empty if no
    /// tenants are active.
    #[must_use]
    pub fn select_concurrent_tenants(&self, conflict_adj: &[u32]) -> Vec<u32> {
        let mut out = Vec::new();
        let mut scratch = TenantSelectionScratch::new();
        self.select_concurrent_tenants_into(conflict_adj, &mut out, &mut scratch);
        out
    }

    /// Select a maximal independent tenant subset into caller-owned storage.
    pub fn select_concurrent_tenants_into(
        &self,
        conflict_adj: &[u32],
        out: &mut Vec<u32>,
        scratch: &mut TenantSelectionScratch,
    ) {
        out.clear();
        scratch.active_ids.clear();
        scratch.active_ids.reserve(self.tenants.len());
        self.tenants
            .iter()
            .map(|entry| entry.value().id())
            .for_each(|id| scratch.active_ids.push(id));
        scratch.active_ids.sort_unstable();
        let n = scratch.active_ids.len();
        if n == 0 {
            return;
        }
        if vyre_driver::accounting::checked_mul_usize_lazy(n, n, || ()).ok()
            != Some(conflict_adj.len())
        {
            // Degenerate: caller didn't supply a matching adjacency.
            // Default to all-tenants-can-run (no conflicts).
            out.reserve(n);
            out.extend(scratch.active_ids.iter().copied());
            return;
        }
        if conflict_adj.iter().all(|conflict| *conflict == 0) {
            out.reserve(n);
            out.extend(scratch.active_ids.iter().copied());
            return;
        }
        scratch.selected_indices.clear();
        scratch.selected_indices.reserve(n);
        'candidate: for candidate_idx in 0..n {
            for &selected_idx in &scratch.selected_indices {
                if conflict_adj[candidate_idx * n + selected_idx] != 0
                    || conflict_adj[selected_idx * n + candidate_idx] != 0
                {
                    continue 'candidate;
                }
            }
            scratch.selected_indices.push(candidate_idx);
        }
        out.reserve(scratch.selected_indices.len());
        for &index in &scratch.selected_indices {
            if let Some(&id) = scratch.active_ids.get(index) {
                out.push(id);
            }
        }
    }
}
