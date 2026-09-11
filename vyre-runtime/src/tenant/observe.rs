//! Read-only views of the tenant registry.
//!
//! Issuing and retiring an id mutates the registry; observing it does not.
//! Every method here sorts by tenant id, so two views taken from the same
//! registry agree on order even though the underlying map has none.

use super::counters::TenantRuntimeCounters;
use super::handle::TenantHandle;
use super::registry::TenantRegistry;

impl TenantRegistry {
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
