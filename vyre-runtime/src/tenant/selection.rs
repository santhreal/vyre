use super::registry::TenantRegistry;

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
