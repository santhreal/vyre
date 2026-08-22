//! Report types for shared-store race legality analysis.

use serde::{Deserialize, Serialize};

/// Legality classification for a shared memory store site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SharedStoreLegality {
    /// Safe because the kernel launches exactly 1 invocation per workgroup or is
    /// guarded by a single-invocation predicate (e.g. `local_id == 0`).
    RaceFreeSingleInvocation,
    /// Safe because the write is performed via an atomic operation.
    RaceFreeAtomic,
    /// Safe because the index is non-uniform and varies per local invocation.
    RaceFreeDistinctIndices,
    /// Illegal: multiple invocations write non-atomically to the same constant or
    /// uniform shared memory index without synchronization or single-invocation guard.
    IllegalMultiInvocationConstantStore,
}

/// One inspected `StoreShared` site in a kernel descriptor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedStoreRaceSite {
    /// Body op index where the store appears.
    pub op_index: usize,
    /// Shared memory binding slot being written.
    pub binding_slot: u32,
    /// Legality classification.
    pub legality: SharedStoreLegality,
}

/// Aggregated report of shared store race legality across a kernel descriptor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedStoreRaceReport {
    /// Identifier of the analyzed kernel descriptor.
    pub kernel_id: String,
    /// Evaluated shared store sites.
    pub sites: Vec<SharedStoreRaceSite>,
}

impl SharedStoreRaceReport {
    /// True when every shared store site in the kernel is proven race-free.
    #[must_use]
    pub fn is_race_free(&self) -> bool {
        self.sites.iter().all(|site| {
            matches!(
                site.legality,
                SharedStoreLegality::RaceFreeSingleInvocation
                    | SharedStoreLegality::RaceFreeAtomic
                    | SharedStoreLegality::RaceFreeDistinctIndices
            )
        })
    }

    /// Number of illegal racing store sites.
    #[must_use]
    pub fn race_count(&self) -> usize {
        self.sites
            .iter()
            .filter(|site| {
                matches!(
                    site.legality,
                    SharedStoreLegality::IllegalMultiInvocationConstantStore
                )
            })
            .count()
    }
}
