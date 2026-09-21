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
        let mut out = Vec::new();
        self.active_tenants_into(&mut out);
        out
    }

    /// Snapshot active tenants into caller-owned storage.
    pub fn active_tenants_into(&self, out: &mut Vec<TenantHandle>) {
        out.clear();
        let tenants = self.read_tenants();
        out.reserve(tenants.len());
        out.extend(tenants.values().cloned());
        drop(tenants);
        out.sort_by_key(TenantHandle::id);
    }

    /// Look up a tenant by id. Returns `None` if the id was
    /// unregistered.
    #[must_use]
    pub fn lookup(&self, tenant_id: u32) -> Option<TenantHandle> {
        self.read_tenants().get(&tenant_id).cloned()
    }

    /// Snapshot runtime counters for every active tenant.
    #[must_use]
    pub fn runtime_counters(&self) -> Vec<TenantRuntimeCounters> {
        let mut out = Vec::new();
        self.runtime_counters_into(&mut out);
        out
    }

    /// Snapshot runtime counters into caller-owned storage.
    pub fn runtime_counters_into(&self, out: &mut Vec<TenantRuntimeCounters>) {
        out.clear();
        let tenants = self.read_tenants();
        out.reserve(tenants.len());
        out.extend(tenants.values().map(TenantHandle::runtime_counters));
        drop(tenants);
        out.sort_by_key(|counters| counters.tenant_id);
    }
}
