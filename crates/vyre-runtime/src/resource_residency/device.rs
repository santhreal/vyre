use std::sync::Arc;

use vyre_driver::{ArtifactMaterializer, BackendError, Resource};

/// Resident-resource operations required by the ownership manager.
///
/// Production sessions use [`MaterializerResourceDevice`]. The trait also
/// permits deterministic fault injection in contract tests.
pub trait ResidentResourceDevice: Send + Sync {
    /// Allocate one resident resource.
    fn allocate(&self, byte_len: usize) -> Result<Resource, BackendError>;
    /// Upload immutable resources as one logical transfer.
    fn upload_many(&self, uploads: &[(&Resource, &[u8])]) -> Result<(), BackendError>;
    /// Upload one mutable-state byte range.
    fn upload_at(
        &self,
        resource: &Resource,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), BackendError>;
    /// Release one resident resource.
    fn free(&self, resource: Resource) -> Result<(), BackendError>;
}

/// Residency adapter bound to one artifact materializer device generation.
#[derive(Clone)]
pub struct MaterializerResourceDevice {
    materializer: Arc<dyn ArtifactMaterializer>,
}

impl MaterializerResourceDevice {
    /// Bind residency operations to one authenticated artifact materializer.
    #[must_use]
    pub fn new(materializer: Arc<dyn ArtifactMaterializer>) -> Self {
        Self { materializer }
    }
}

impl ResidentResourceDevice for MaterializerResourceDevice {
    fn allocate(&self, byte_len: usize) -> Result<Resource, BackendError> {
        self.materializer.allocate_resident(byte_len)
    }

    fn upload_many(&self, uploads: &[(&Resource, &[u8])]) -> Result<(), BackendError> {
        for (resource, bytes) in uploads {
            self.materializer.upload_resident(resource, bytes)?;
        }
        Ok(())
    }

    fn upload_at(
        &self,
        resource: &Resource,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), BackendError> {
        self.materializer
            .upload_resident_at(resource, offset, bytes)
    }

    fn free(&self, resource: Resource) -> Result<(), BackendError> {
        self.materializer.free_resident(resource)
    }
}
