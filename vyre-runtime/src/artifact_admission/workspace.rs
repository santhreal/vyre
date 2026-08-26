//! Resident storage for the values an artifact produces for itself.
//!
//! The artifact records one region per such value: its byte count, its offset in
//! the workspace allocation, and the span of entry points that bind it. The
//! runtime allocates that plan and binds it. It never packs, resizes, merges, or
//! reorders a region, because the selected schedule already decided where every
//! cross-entry value lives and two packers disagreeing binds the wrong buffer
//! rather than a slower one.
//!
//! A [`Resource`] names a whole backend buffer and has no sub-range view, so one
//! buffer is allocated per recorded region instead of slicing a single
//! allocation. The recorded byte count sizes each buffer exactly and the
//! recorded offset order fixes the allocation order, so the allocation is a
//! faithful projection of the plan on every backend.

use std::collections::BTreeMap;

use vyre_driver::{ArtifactMaterializer, BackendError, Resource};
use vyre_megakernel::{ArtifactValueId, WorkspacePlan};

/// Resident buffers backing every workspace region the artifact recorded.
#[derive(Debug)]
pub struct ArtifactWorkspace {
    total_bytes: u64,
    regions: BTreeMap<ArtifactValueId, Resource>,
}

impl ArtifactWorkspace {
    /// Bytes the recorded plan reserves across every region.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Allocated buffer for each workspace-owned canonical value.
    #[must_use]
    pub fn regions(&self) -> &BTreeMap<ArtifactValueId, Resource> {
        &self.regions
    }

    /// Whether the artifact allocates `value` for itself.
    #[must_use]
    pub fn owns(&self, value: ArtifactValueId) -> bool {
        self.regions.contains_key(&value)
    }

    /// Allocate one buffer per recorded region, in recorded offset order.
    ///
    /// # Errors
    ///
    /// Returns the materializer rejection when a region cannot be allocated, and
    /// an invalid-program rejection when a recorded byte count exceeds the host
    /// address space. Buffers already allocated for the same plan are released
    /// before the rejection is returned.
    pub(super) fn allocate(
        plan: &WorkspacePlan,
        materializer: &dyn ArtifactMaterializer,
    ) -> Result<Self, BackendError> {
        let mut regions = BTreeMap::new();
        for region in &plan.regions {
            let allocation = usize::try_from(region.bytes)
                .map_err(|_| BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: workspace region for canonical value {} reserves {} bytes, past the host address space.",
                        region.value.0, region.bytes
                    ),
                })
                .and_then(|byte_len| materializer.allocate_resident(byte_len));
            match allocation {
                Ok(resource) => {
                    regions.insert(region.value, resource);
                }
                Err(error) => {
                    release(regions, materializer);
                    return Err(error);
                }
            }
        }
        Ok(Self {
            total_bytes: plan.total_bytes,
            regions,
        })
    }

    /// Release every allocated region through the owning materializer.
    ///
    /// # Errors
    ///
    /// Returns the first materializer rejection. Every remaining region is
    /// released first, so one failing buffer never leaks the others.
    pub(super) fn free(self, materializer: &dyn ArtifactMaterializer) -> Result<(), BackendError> {
        let mut first = None;
        for resource in self.regions.into_values() {
            if let Err(error) = materializer.free_resident(resource) {
                first = first.or(Some(error));
            }
        }
        match first {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// Release a partial allocation and keep the rejection that stopped it.
fn release(regions: BTreeMap<ArtifactValueId, Resource>, materializer: &dyn ArtifactMaterializer) {
    for resource in regions.into_values() {
        drop(materializer.free_resident(resource));
    }
}
