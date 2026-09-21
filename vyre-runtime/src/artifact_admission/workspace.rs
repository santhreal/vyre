//! Resident storage for the values an artifact produces for itself.
//!
//! The artifact records one region per allocation the runtime makes, and every
//! value placed in that region names its bytes inside it. The runtime allocates
//! that plan and binds it. It never packs, resizes, merges, reorders, or reuses a
//! region, because the selected schedule already decided where every value lives
//! and two packers disagreeing binds the wrong buffer rather than a slower one.
//!
//! A [`Resource`] names a whole backend buffer and has no sub-range view, so one
//! buffer is allocated per artifact-owned region. A region that holds several
//! placements is one buffer serving values whose live ranges are disjoint, so
//! every such value binds the same buffer and the buffer is released once.

use std::collections::BTreeMap;

use vyre_driver::{ArtifactMaterializer, BackendError, Resource};
use vyre_megakernel::allocation::AllocationPlan;
use vyre_megakernel::ArtifactValueId;

/// Resident buffers backing every artifact-owned region the plan records.
#[derive(Debug)]
pub struct ArtifactWorkspace {
    total_bytes: u64,
    buffers: Vec<Resource>,
    bindings: BTreeMap<ArtifactValueId, Resource>,
}

impl ArtifactWorkspace {
    /// Bytes the recorded plan reserves across every artifact-owned region.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Buffer each artifact-owned canonical value binds.
    ///
    /// Two values placed in one reused region name the same buffer.
    #[must_use]
    pub fn bindings(&self) -> &BTreeMap<ArtifactValueId, Resource> {
        &self.bindings
    }

    /// Buffers allocated for this plan, in recorded region order.
    #[must_use]
    pub fn buffers(&self) -> &[Resource] {
        &self.buffers
    }

    /// Whether the artifact allocates `value` for itself.
    #[must_use]
    pub fn owns(&self, value: ArtifactValueId) -> bool {
        self.bindings.contains_key(&value)
    }

    /// Allocate one buffer per artifact-owned region, in recorded region order.
    ///
    /// # Errors
    ///
    /// Returns the materializer rejection when a region cannot be allocated, and
    /// an invalid-program rejection when a recorded byte count exceeds the host
    /// address space. Buffers already allocated for the same plan are released
    /// before the rejection is returned.
    pub(super) fn allocate(
        plan: &AllocationPlan,
        materializer: &dyn ArtifactMaterializer,
    ) -> Result<Self, BackendError> {
        let mut buffers = Vec::new();
        let mut bindings = BTreeMap::new();
        for region in plan.owned() {
            let allocation = usize::try_from(region.bytes)
                .map_err(|_| BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: an artifact-owned region at offset {} reserves {} bytes, past the host address space.",
                        region.offset, region.bytes
                    ),
                })
                .and_then(|byte_len| materializer.allocate_resident(byte_len));
            match allocation {
                Ok(resource) => {
                    for placement in &region.placements {
                        bindings.insert(placement.value, resource.clone());
                    }
                    buffers.push(resource);
                }
                Err(error) => {
                    release(buffers, materializer);
                    return Err(error);
                }
            }
        }
        Ok(Self {
            total_bytes: plan.owned_bytes(),
            buffers,
            bindings,
        })
    }

    /// Release every allocated buffer through the owning materializer.
    ///
    /// # Errors
    ///
    /// Returns the first materializer rejection. Every remaining buffer is
    /// released first, so one failing buffer never leaks the others.
    pub(super) fn free(self, materializer: &dyn ArtifactMaterializer) -> Result<(), BackendError> {
        let mut first = None;
        for resource in self.buffers {
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
fn release(buffers: Vec<Resource>, materializer: &dyn ArtifactMaterializer) {
    for resource in buffers {
        drop(materializer.free_resident(resource));
    }
}
